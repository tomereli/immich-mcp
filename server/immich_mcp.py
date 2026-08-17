#!/usr/bin/env python3
"""MCP server exposing a self-hosted Immich photo library.

Built for one job: letting a model actually look at the photographs instead of being
handed a URL it cannot fetch. Images come back as real MCP image content blocks,
resized and JPEG-encoded, never as links.

Capability is set by MCP_MODE on this server rather than by the Immich API key: the
key is local and never leaves the machine, while this endpoint is the exposed one.
MCP_MODE=read registers only read tools; MCP_MODE=full adds upload and album tools.
Neither mode registers anything that deletes.

Transport is streamable HTTP with stateless JSON responses, because the intended
client is a hosted assistant that only connects to remote URL-based MCP servers.
"""

import base64
import io
import json
import logging
import os
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

import httpx
from PIL import Image as PILImage
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ImageContent, TextContent
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route

# --------------------------------------------------------------------------- config

IMMICH_URL = os.environ.get("IMMICH_URL", "http://immich-server:2283").rstrip("/")
IMMICH_API_KEY = os.environ.get("IMMICH_API_KEY", "")
MCP_BEARER_TOKEN = os.environ.get("MCP_BEARER_TOKEN", "")
ALLOW_QUERY_TOKEN = os.environ.get("MCP_ALLOW_QUERY_TOKEN", "").lower() in ("1", "true", "yes")
HTTP_TIMEOUT = 30.0

# Capability lives HERE, on the remote-facing server, not in the Immich API key.
# The key is local and never leaves the machine; this endpoint is the exposed one,
# so this is where "what may it do" belongs.
#   read  - list/inspect/fetch only. Write tools are never registered.
#   full  - adds upload, album creation, and adding assets to albums.
# Nothing in either mode can delete. That is deliberate and not configurable.
MCP_MODE = os.environ.get("MCP_MODE", "read").strip().lower()
if MCP_MODE not in ("read", "full"):
    raise SystemExit(f"MCP_MODE must be 'read' or 'full', got {MCP_MODE!r}")

# Names the host in the "Immich is unreachable" message. A self-hosted library is
# usually down because the machine under it is off, and an agent that can name that
# machine gives its user something to act on instead of a stack trace.
IMMICH_HOST_HINT = os.environ.get("IMMICH_HOST_HINT", "").strip()

# Immich's own "preview" rendition is already JPEG at roughly 1440px on the long edge.
# Serving from it means this server never has to decode HEIC at all — no libheif, no
# pillow-heif, no surprises when Immich changes its storage format.
PREVIEW_LONG_EDGE = 1440

if not IMMICH_API_KEY:
    raise SystemExit("IMMICH_API_KEY is not set. Refusing to start.")
if not MCP_BEARER_TOKEN:
    raise SystemExit("MCP_BEARER_TOKEN is not set. Refusing to start an unauthenticated tunnel.")

# Endpoint at /mcp to match the convention every other MCP server uses, so the same
# probe scripts and connector config work against either implementation.
#
# DNS-rebinding protection is disabled deliberately. It rejects any Host header not on
# an allowlist, which breaks every legitimate caller here: other containers reach this
# as `immich-mcp-ro:3001`, the Tailscale Funnel presents the public `.ts.net` name, and
# local probes use `localhost`. Nothing is lost by turning it off, because the bearer
# token below is what actually gates access - and unlike a Host header, an attacker
# cannot set it.
_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
mcp = FastMCP(
    "immich_mcp",
    stateless_http=True,
    json_response=True,
    streamable_http_path="/mcp",
    transport_security=_security,
)

# --------------------------------------------------------------------------- helpers


class AssetOrder(str, Enum):
    NEWEST = "newest"
    OLDEST = "oldest"


class ImmichDown(Exception):
    """Immich did not answer at all — almost always a powered-off desktop."""


def _headers() -> Dict[str, str]:
    return {"x-api-key": IMMICH_API_KEY, "Accept": "application/json"}


async def _request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    url = f"{IMMICH_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.request(method, url, headers=_headers(), **kwargs)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
        raise ImmichDown(str(exc)) from exc
    resp.raise_for_status()
    return resp


async def _get_json(path: str, **kwargs: Any) -> Any:
    return (await _request("GET", path, **kwargs)).json()


async def _post_json(path: str, payload: Dict[str, Any]) -> Any:
    return (await _request("POST", path, json=payload)).json()


def _error(exc: Exception) -> str:
    """One place that turns exceptions into something an agent can act on."""
    if isinstance(exc, ImmichDown):
        where = (
            f"Check that {IMMICH_HOST_HINT} is powered on and that Docker is running."
            if IMMICH_HOST_HINT
            else "Check that the machine hosting Immich is powered on and that Immich is running."
        )
        return (
            "Immich is not responding — the machine hosting it may be off. "
            "Nothing is wrong with this connector; the photo library itself is "
            f"unreachable. {where}"
        )
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 401:
            return "Immich rejected the API key. It may have been revoked or regenerated."
        if code == 403:
            return "The API key lacks permission for this operation."
        if code == 404:
            return "No such album or asset in Immich. The id may be stale — re-list to get current ids."
        return f"Immich returned HTTP {code}."
    return f"Unexpected {type(exc).__name__}: {exc}"


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _days_ago(value: Optional[str]) -> Optional[int]:
    ts = _parse_ts(value)
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - ts).days)


async def _album_assets(album_id: str) -> List[Dict[str, Any]]:
    """Immich v3 dropped the assets array from GET /api/albums/<id>; search is the way."""
    data = await _post_json("/api/search/metadata", {"albumIds": [album_id], "size": 1000})
    return (data.get("assets") or {}).get("items") or []


def _summarise(asset: Dict[str, Any]) -> Dict[str, Any]:
    taken = asset.get("fileCreatedAt")
    return {
        "asset_id": asset.get("id"),
        "filename": asset.get("originalFileName"),
        "taken_at": taken,
        "days_ago": _days_ago(taken),
        "type": "video" if asset.get("type") == "VIDEO" else "image",
    }


# --------------------------------------------------------------------------- inputs


class ListAlbumsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListAssetsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    album_id: str = Field(..., description="Album id from immich_list_albums.", min_length=1)
    limit: int = Field(default=20, description="Maximum assets to return.", ge=1, le=200)
    offset: int = Field(default=0, description="Assets to skip, for pagination.", ge=0)
    order: AssetOrder = Field(
        default=AssetOrder.NEWEST,
        description="'newest' or 'oldest' by capture date. Use 'oldest' to walk a subject's history forwards.",
    )


class GetImageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    asset_id: str = Field(..., description="Asset id from immich_list_assets.", min_length=1)
    max_dimension: int = Field(
        default=1024,
        description=(
            "Longest edge in pixels. 1024 is ample for most visual judgements; larger "
            f"values are capped at {PREVIEW_LONG_EDGE}, the size of Immich's own preview."
        ),
        ge=128,
        le=PREVIEW_LONG_EDGE,
    )


class AssetMetadataInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    asset_id: str = Field(..., description="Asset id from immich_list_assets.", min_length=1)


# --------------------------------------------------------------------------- tools


@mcp.tool(
    name="immich_list_albums",
    annotations={
        "title": "List Immich albums",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def immich_list_albums() -> str:
    """List every album in the Immich library.

    Start here. Album ids from this call feed immich_list_assets.

    Returns:
        str: JSON with schema
        {
          "count": int,
          "albums": [
            {"album_id": str, "name": str, "asset_count": int,
             "most_recent_asset_at": str|null, "most_recent_days_ago": int|null}
          ]
        }
        On failure, a plain-language sentence beginning "Immich is not responding"
        or "Immich returned HTTP ...".
    """
    try:
        albums = await _get_json("/api/albums")
        rows = [
            {
                "album_id": a.get("id"),
                "name": a.get("albumName"),
                "asset_count": a.get("assetCount", 0),
                "most_recent_asset_at": a.get("endDate"),
                "most_recent_days_ago": _days_ago(a.get("endDate")),
            }
            for a in albums
        ]
        rows.sort(key=lambda r: r["name"] or "")
        return json.dumps({"count": len(rows), "albums": rows}, ensure_ascii=False, indent=1)
    except Exception as exc:  # noqa: BLE001 - surfaced as an agent-readable string
        return _error(exc)


@mcp.tool(
    name="immich_list_assets",
    annotations={
        "title": "List assets in an Immich album",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def immich_list_assets(
    album_id: Annotated[str, Field(description="Album id from immich_list_albums.")],
    limit: Annotated[int, Field(description="Max assets to return (1-200).", ge=1, le=200)] = 20,
    offset: Annotated[int, Field(description="Assets to skip, for pagination.", ge=0)] = 0,
    order: Annotated[AssetOrder, Field(description="newest or oldest by capture date.")] = AssetOrder.NEWEST,
) -> str:
    """List the photographs in one album, newest or oldest first.

    Each entry carries days_ago so comparisons across dates need no arithmetic.
    Videos appear here with type "video"; immich_get_image cannot render them.

    Returns:
        str: JSON with schema
        {
          "album_id": str, "total": int, "count": int, "offset": int,
          "has_more": bool, "next_offset": int|null,
          "assets": [
            {"asset_id": str, "filename": str, "taken_at": str,
             "days_ago": int, "type": "image"|"video"}
          ]
        }
    """
    try:
        items = await _album_assets(album_id)
        items.sort(
            key=lambda a: _parse_ts(a.get("fileCreatedAt")) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=(order == AssetOrder.NEWEST),
        )
        total = len(items)
        page = items[offset : offset + limit]
        consumed = offset + len(page)
        return json.dumps(
            {
                "album_id": album_id,
                "total": total,
                "count": len(page),
                "offset": offset,
                "has_more": consumed < total,
                "next_offset": consumed if consumed < total else None,
                "assets": [_summarise(a) for a in page],
            },
            ensure_ascii=False,
            indent=1,
        )
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@mcp.tool(
    name="immich_get_image",
    annotations={
        "title": "Fetch an Immich photo as an image",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def immich_get_image(
    asset_id: Annotated[str, Field(description="Asset id from immich_list_assets.")],
    max_dimension: Annotated[int, Field(description="Longest edge in px (128-1440).", ge=128, le=1440)] = 1024,
) -> list:
    """Fetch one photograph as an actual image, resized and JPEG-encoded.

    This is the tool that lets Claude see rather than be told. The capture date
    travels in the accompanying text so the picture is never undated.

    Sourced from Immich's preview rendition, which is already JPEG — the original
    HEIC is never touched.

    Returns:
        list: [TextContent describing the asset and its capture date, ImageContent]
        On failure, a single TextContent explaining what went wrong.
    """
    try:
        meta = await _get_json(f"/api/assets/{asset_id}")
        if meta.get("type") == "VIDEO":
            return [
                TextContent(
                    type="text",
                    text=(
                        f"{meta.get('originalFileName')} is a video, and this tool returns "
                        "still images only — extracting a frame is not supported. Use "
                        "immich_list_assets to find a photograph in the same album instead."
                    ),
                )
            ]

        resp = await _request(
            "GET", f"/api/assets/{asset_id}/thumbnail", params={"size": "preview"}
        )
        img = PILImage.open(io.BytesIO(resp.content))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.thumbnail((max_dimension, max_dimension), PILImage.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        payload = buf.getvalue()

        taken = meta.get("fileCreatedAt")
        ago = _days_ago(taken)

        return [
            TextContent(
                type="text",
                text=(
                    f"{meta.get('originalFileName')} — taken {taken}"
                    + (f" ({ago} days ago)" if ago is not None else "")
                    + f", shown at {img.width}x{img.height}."
                ),
            ),
            ImageContent(
                type="image",
                data=base64.b64encode(payload).decode("ascii"),
                mimeType="image/jpeg",
            ),
        ]
    except Exception as exc:  # noqa: BLE001
        return [TextContent(type="text", text=_error(exc))]


@mcp.tool(
    name="immich_get_asset_metadata",
    annotations={
        "title": "Get Immich asset metadata",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def immich_get_asset_metadata(
    asset_id: Annotated[str, Field(description="Asset id from immich_list_assets.")],
) -> str:
    """Get one asset's date, dimensions, size and album membership without fetching pixels.

    Use when the date or size is all that is needed — it costs a fraction of
    immich_get_image.

    Returns:
        str: JSON with schema
        {
          "asset_id": str, "filename": str, "type": "image"|"video",
          "taken_at": str, "days_ago": int, "mime_type": str,
          "width": int|null, "height": int|null, "file_size_bytes": int|null,
          "camera": str|null, "albums": [{"album_id": str, "name": str}]
        }
    """
    try:
        meta = await _get_json(f"/api/assets/{asset_id}")
        exif = meta.get("exifInfo") or {}
        try:
            albums = await _get_json("/api/albums", params={"assetId": asset_id})
        except Exception:  # noqa: BLE001 - album membership is a nicety, not the point
            albums = []
        taken = meta.get("fileCreatedAt")
        return json.dumps(
            {
                "asset_id": meta.get("id"),
                "filename": meta.get("originalFileName"),
                "type": "video" if meta.get("type") == "VIDEO" else "image",
                "taken_at": taken,
                "days_ago": _days_ago(taken),
                "mime_type": meta.get("originalMimeType"),
                "width": exif.get("exifImageWidth"),
                "height": exif.get("exifImageHeight"),
                "file_size_bytes": exif.get("fileSizeInByte"),
                "camera": exif.get("model"),
                "albums": [{"album_id": a.get("id"), "name": a.get("albumName")} for a in albums],
            },
            ensure_ascii=False,
            indent=1,
        )
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# --------------------------------------------------------------------------- writes
# Registered only when MCP_MODE=full. In read mode these tools do not exist at all -
# they are absent from tools/list, so there is nothing for a client to talk itself into.
#
# One rule holds across everything below: NOTHING HERE DESTROYS A PHOTOGRAPH.
#
#   immich_delete_album      removes the album, never its contents (Immich's own
#                            behaviour - assets outlive the albums that held them)
#   immich_remove_from_album takes assets out of an album, leaving them in the library
#   immich_trash_assets      moves to Immich's trash, which keeps them for the retention
#                            period configured on the server (30 days by default)
#   immich_restore_assets    brings them back
#
# The Immich API takes a `force` flag on asset deletion that bypasses the trash and
# erases immediately. It is never sent. Deliberately there is no tool, and no argument,
# that can reach it: an agent acting on a misheard "clean that up" should cost a click
# to undo, not a photograph. Emptying the trash is done in Immich's own UI, by a human.

if MCP_MODE == "full":

    async def _find_album_by_name(name: str) -> Optional[Dict[str, Any]]:
        """Case-insensitive exact match, so 'hoya' finds the album called 'Hoya'."""
        albums = await _get_json("/api/albums")
        target = name.strip().casefold()
        for a in albums:
            if (a.get("albumName") or "").strip().casefold() == target:
                return a
        return None

    @mcp.tool(
        name="immich_upload_image",
        annotations={"title": "Upload an image to Immich", "readOnlyHint": False,
                     "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def immich_upload_image(
        image_base64: Annotated[str, Field(description="Base64-encoded image bytes (JPEG or PNG).")],
        filename: Annotated[str, Field(description="Filename to store, e.g. hoya-2026-08-17.jpg")],
        album_id: Annotated[Optional[str], Field(description="Album id to file it under.")] = None,
        album_name: Annotated[Optional[str], Field(
            description="Album name to file it under. Matched case-insensitively; created if absent. "
                        "Use this instead of album_id when working from a name the user said out loud."
        )] = None,
        taken_at: Annotated[Optional[str], Field(
            description="ISO 8601 capture time, e.g. 2026-08-17T09:30:00Z. Defaults to now. "
                        "Set it when the photograph was taken earlier than the upload."
        )] = None,
    ) -> str:
        """Upload a new image into Immich and file it under an album in one step.

        This is the tool that removes the manual round trip: photograph in, correct
        album, done. Give it album_name and the album is found or created as needed -
        no need to list albums first.

        Additive only. It never replaces or removes an existing asset. If Immich
        recognises the bytes as one it already holds it returns status "duplicate"
        and files the existing asset instead of storing a second copy.

        Returns:
            str: JSON {"asset_id": str, "status": "created"|"duplicate",
                       "album_id": str|null, "album_name": str|null,
                       "album_created": bool, "added_to_album": bool}
                 or a plain-language error string.
        """
        try:
            raw = base64.b64decode(image_base64, validate=False)
            now = datetime.now(timezone.utc).isoformat()
            when = (taken_at or "").strip() or now

            # Resolve the album before uploading, so a bad album name fails before
            # anything is stored rather than leaving an orphaned asset behind.
            album_created = False
            resolved_name = None
            if album_name and not album_id:
                existing = await _find_album_by_name(album_name)
                if existing:
                    album_id = existing.get("id")
                    resolved_name = existing.get("albumName")
                else:
                    made = (await _request("POST", "/api/albums",
                                           json={"albumName": album_name.strip()})).json()
                    album_id = made.get("id")
                    resolved_name = made.get("albumName")
                    album_created = True

            files = {"assetData": (filename, raw, "application/octet-stream")}
            data = {
                "deviceAssetId": f"claude-mcp-{filename}-{now}",
                "deviceId": "claude-mcp",
                "fileCreatedAt": when,
                "fileModifiedAt": when,
            }
            created = (await _request("POST", "/api/assets", files=files, data=data)).json()
            asset_id = created.get("id")

            added = False
            if album_id and asset_id:
                await _request("PUT", f"/api/albums/{album_id}/assets", json={"ids": [asset_id]})
                added = True

            return json.dumps(
                {
                    "asset_id": asset_id,
                    "status": created.get("status"),
                    "album_id": album_id,
                    "album_name": resolved_name,
                    "album_created": album_created,
                    "added_to_album": added,
                },
                ensure_ascii=False, indent=1,
            )
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @mcp.tool(
        name="immich_create_album",
        annotations={"title": "Create an Immich album", "readOnlyHint": False,
                     "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def immich_create_album(
        name: Annotated[str, Field(description="Album name.")],
        description: Annotated[Optional[str], Field(description="Optional description.")] = None,
    ) -> str:
        """Create a new, empty album.

        Returns:
            str: JSON {"album_id": str, "name": str} or an error string.
        """
        try:
            payload: Dict[str, Any] = {"albumName": name}
            if description:
                payload["description"] = description
            a = (await _request("POST", "/api/albums", json=payload)).json()
            return json.dumps({"album_id": a.get("id"), "name": a.get("albumName")},
                              ensure_ascii=False, indent=1)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @mcp.tool(
        name="immich_add_to_album",
        annotations={"title": "Add existing assets to an album", "readOnlyHint": False,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def immich_add_to_album(
        album_id: Annotated[str, Field(description="Target album id.")],
        asset_ids: Annotated[List[str], Field(description="Asset ids to add.")],
    ) -> str:
        """Add assets that already exist in Immich to an album. Never removes any.

        Returns:
            str: JSON list of {"id": str, "success": bool, "error": str|null}
                 or an error string.
        """
        try:
            r = (await _request("PUT", f"/api/albums/{album_id}/assets",
                                json={"ids": asset_ids})).json()
            return json.dumps(r, ensure_ascii=False, indent=1)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @mcp.tool(
        name="immich_update_album",
        annotations={"title": "Rename or describe an Immich album", "readOnlyHint": False,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def immich_update_album(
        album_id: Annotated[str, Field(description="Album id from immich_list_albums.")],
        name: Annotated[Optional[str], Field(description="New album name. Omit to leave unchanged.")] = None,
        description: Annotated[Optional[str], Field(description="New description. Omit to leave unchanged.")] = None,
    ) -> str:
        """Rename an album or change its description. Contents are untouched.

        Returns:
            str: JSON {"album_id": str, "name": str, "description": str}
                 or a plain-language error string.
        """
        try:
            payload: Dict[str, Any] = {}
            if name is not None:
                payload["albumName"] = name
            if description is not None:
                payload["description"] = description
            if not payload:
                return "Nothing to change — pass name, description, or both."
            a = (await _request("PATCH", f"/api/albums/{album_id}", json=payload)).json()
            return json.dumps(
                {"album_id": a.get("id"), "name": a.get("albumName"),
                 "description": a.get("description")},
                ensure_ascii=False, indent=1,
            )
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @mcp.tool(
        name="immich_remove_from_album",
        annotations={"title": "Remove assets from an album", "readOnlyHint": False,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def immich_remove_from_album(
        album_id: Annotated[str, Field(description="Album id to remove them from.")],
        asset_ids: Annotated[List[str], Field(description="Asset ids to remove from this album.")],
    ) -> str:
        """Take assets out of an album. The photographs stay in the library.

        This is filing, not deletion — use it to move a photo to the right plant.
        To put a photo in the trash instead, use immich_trash_assets.

        Returns:
            str: JSON list of {"id": str, "success": bool, "error": str|null}
                 or a plain-language error string.
        """
        try:
            r = (await _request("DELETE", f"/api/albums/{album_id}/assets",
                                json={"ids": asset_ids})).json()
            return json.dumps(r, ensure_ascii=False, indent=1)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @mcp.tool(
        name="immich_delete_album",
        annotations={"title": "Delete an Immich album", "readOnlyHint": False,
                     "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def immich_delete_album(
        album_id: Annotated[str, Field(description="Album id from immich_list_albums.")],
    ) -> str:
        """Delete an album. The photographs it held stay in the library.

        Immich albums are labels, not folders — removing one never removes its
        contents. The assets remain and can be gathered into a new album.

        Returns:
            str: JSON {"deleted_album_id": str, "assets_kept": int} or an error string.
        """
        try:
            # Report how many assets survive, so the caller can see nothing was lost.
            kept = len(await _album_assets(album_id))
            await _request("DELETE", f"/api/albums/{album_id}")
            return json.dumps({"deleted_album_id": album_id, "assets_kept": kept},
                              ensure_ascii=False, indent=1)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @mcp.tool(
        name="immich_trash_assets",
        annotations={"title": "Move photos to the Immich trash", "readOnlyHint": False,
                     "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def immich_trash_assets(
        asset_ids: Annotated[List[str], Field(description="Asset ids to move to the trash.")],
    ) -> str:
        """Move photographs to Immich's trash, where they are recoverable.

        This is the strongest delete available here, and it is reversible:
        immich_restore_assets brings them back, and Immich keeps trashed items for
        the retention period set on the server (30 days by default). Permanently
        erasing them is only possible from Immich's own interface.

        Returns:
            str: JSON {"trashed": int, "asset_ids": [str], "recoverable": true,
                       "restore_with": "immich_restore_assets"}
                 or a plain-language error string.
        """
        try:
            # force is never sent. Immich defaults to the trash; that is the point.
            await _request("DELETE", "/api/assets", json={"ids": asset_ids})
            return json.dumps(
                {"trashed": len(asset_ids), "asset_ids": asset_ids, "recoverable": True,
                 "restore_with": "immich_restore_assets"},
                ensure_ascii=False, indent=1,
            )
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @mcp.tool(
        name="immich_restore_assets",
        annotations={"title": "Restore photos from the Immich trash", "readOnlyHint": False,
                     "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def immich_restore_assets(
        asset_ids: Annotated[List[str], Field(description="Asset ids to bring back out of the trash.")],
    ) -> str:
        """Bring photographs back out of the trash.

        Returns:
            str: JSON {"restored": int, "asset_ids": [str]} or an error string.
        """
        try:
            await _request("POST", "/api/trash/restore/assets", json={"ids": asset_ids})
            return json.dumps({"restored": len(asset_ids), "asset_ids": asset_ids},
                              ensure_ascii=False, indent=1)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)


# --------------------------------------------------------------------------- transport


# Paths a client probes to discover an OAuth authorization server. They must be
# answered with a plain 404 and nothing else.
#
# This is subtle and it costs an evening if you get it wrong. Under the MCP auth
# spec a 401 means "authenticate, here is where to start", so answering these
# probes with 401 sends the client hunting for an authorization server that does
# not exist — it then fails to register a client and reports that the *server's*
# sign-in service is broken. A 404 says plainly: there is no OAuth here, use the
# credential you were already given.
#
# The same trap exists outside this process. Immich's SPA returns its index page
# with 200 for any unrecognised path, so if Immich is mounted at the root of the
# same hostname it will answer these probes with HTML and the client will try to
# parse it as OAuth metadata. Give this server its own origin — a dedicated port
# or hostname — rather than a sub-path beside a web app, so that the probes land
# here and get the 404.
OAUTH_DISCOVERY_PREFIX = "/.well-known/"


class BearerAuth(BaseHTTPMiddleware):
    """The tunnel URL is guessable in principle; this makes knowing it insufficient."""

    async def dispatch(self, request, call_next):
        if request.url.path == "/healthz":
            return await call_next(request)

        if request.url.path.startswith(OAUTH_DISCOVERY_PREFIX):
            return JSONResponse({"error": "not_found"}, status_code=404)

        header = request.headers.get("authorization", "")
        if header.startswith("Bearer ") and header[7:] == MCP_BEARER_TOKEN:
            return await call_next(request)

        # Fallback for clients that cannot attach a custom header. Off by default:
        # tokens in query strings leak into proxy and server logs, so only enable
        # this if the connector UI leaves no other option.
        if ALLOW_QUERY_TOKEN and request.query_params.get("token") == MCP_BEARER_TOKEN:
            return await call_next(request)

        return JSONResponse({"error": "unauthorized"}, status_code=401)


async def _healthz(_request):
    return JSONResponse({"ok": True, "immich_url": IMMICH_URL})


app = mcp.streamable_http_app()
app.router.routes.append(Route("/healthz", _healthz))
app.add_middleware(BearerAuth)


class RedactToken(logging.Filter):
    """Keep the bearer token out of the access log.

    Uvicorn logs the full request line, so ?token=... would otherwise sit in
    `docker logs` in plaintext indefinitely. TLS protects the token in flight;
    nothing protects it once it has been written to disk. This is the gap that
    makes query-string auth genuinely worse than a header, so close it.
    """

    _pat = re.compile(r"(token=)[^&\s\"']+")

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                self._pat.sub(r"\1[REDACTED]", a) if isinstance(a, str) else a
                for a in record.args
            )
        if isinstance(record.msg, str):
            record.msg = self._pat.sub(r"\1[REDACTED]", record.msg)
        return True


if __name__ == "__main__":
    import uvicorn

    for _name in ("uvicorn.access", "uvicorn.error"):
        logging.getLogger(_name).addFilter(RedactToken())

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "3000")))
