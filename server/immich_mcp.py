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
# Note there is no delete tool in either mode, by design.

if MCP_MODE == "full":

    class UploadInput(BaseModel):
        model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

        image_base64: str = Field(..., description="Base64-encoded image bytes (JPEG or PNG).", min_length=16)
        filename: str = Field(..., description="Filename to store, e.g. 'hoya-2026-08-17.jpg'.", min_length=1, max_length=255)
        album_id: Optional[str] = Field(default=None, description="Optional album to add it to immediately.")

    class CreateAlbumInput(BaseModel):
        model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

        name: str = Field(..., description="Album name.", min_length=1, max_length=200)
        description: Optional[str] = Field(default=None, description="Optional description.", max_length=1000)

    class AddToAlbumInput(BaseModel):
        model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

        album_id: str = Field(..., description="Target album id.", min_length=1)
        asset_ids: List[str] = Field(..., description="Asset ids to add.", min_length=1, max_length=100)

    @mcp.tool(
        name="immich_upload_image",
        annotations={"title": "Upload an image to Immich", "readOnlyHint": False,
                     "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def immich_upload_image(
        image_base64: Annotated[str, Field(description="Base64-encoded image bytes (JPEG or PNG).")],
        filename: Annotated[str, Field(description="Filename to store, e.g. hoya-2026-08-17.jpg")],
        album_id: Annotated[Optional[str], Field(description="Optional album to add it to.")] = None,
    ) -> str:
        """Upload a new image into Immich, optionally straight into an album.

        Additive only - this never replaces or removes an existing asset.

        Returns:
            str: JSON {"asset_id": str, "status": str, "added_to_album": bool}
                 or a plain-language error string.
        """
        try:
            raw = base64.b64decode(image_base64, validate=False)
            now = datetime.now(timezone.utc).isoformat()
            files = {"assetData": (filename, raw, "application/octet-stream")}
            data = {
                "deviceAssetId": f"claude-mcp-{filename}-{now}",
                "deviceId": "claude-mcp",
                "fileCreatedAt": now,
                "fileModifiedAt": now,
            }
            resp = await _request("POST", "/api/assets", files=files, data=data)
            created = resp.json()
            asset_id = created.get("id")

            added = False
            if album_id and asset_id:
                await _request("PUT", f"/api/albums/{album_id}/assets",
                               json={"ids": [asset_id]})
                added = True

            return json.dumps(
                {"asset_id": asset_id, "status": created.get("status"), "added_to_album": added},
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


# --------------------------------------------------------------------------- transport


class BearerAuth(BaseHTTPMiddleware):
    """The tunnel URL is guessable in principle; this makes knowing it insufficient."""

    async def dispatch(self, request, call_next):
        if request.url.path == "/healthz":
            return await call_next(request)

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
