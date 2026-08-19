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

import asyncio
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

# The one path that serves MCP. Everything else on this server is a 404 - see the
# note above BearerAuth for why that is load-bearing and not merely tidy.
MCP_PATH = "/mcp"

mcp = FastMCP(
    "immich_mcp",
    stateless_http=True,
    json_response=True,
    streamable_http_path=MCP_PATH,
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


# ----------------------------------------------------------------------- video

# Eight frames at 768px is already a substantial slice of a context window, so the
# cap is deliberate and there is no "every frame" mode. The subjects this exists
# for - an espresso extraction, a plant moving in wind - change over seconds, not
# milliseconds, so a handful of chosen moments carries nearly all the information
# in the clip.
MAX_VIDEO_FRAMES = 8

# Past this much drift between the requested and the delivered moment, the frame
# is labelled rather than silently mislabelled. Reasoning about *when* something
# happened from a frame stamped with a time it does not show is worse than having
# no frame at all.
SEEK_DRIFT_TOLERANCE = 0.2

# showinfo prints one of these per decoded frame, on stderr.
_PTS_RE = re.compile(r"pts_time:\s*([0-9]+\.?[0-9]*)")


class FFmpegMissing(Exception):
    """ffmpeg/ffprobe are absent - the image was built without them."""


def _video_url(asset_id: str, source: str) -> str:
    tail = "original" if source == "original" else "video/playback"
    return f"{IMMICH_URL}/api/assets/{asset_id}/{tail}"


def _ff_headers() -> str:
    """ffmpeg wants CRLF-terminated header lines."""
    return f"x-api-key: {IMMICH_API_KEY}\r\n"


async def _run(cmd: List[str], timeout: float = 180.0):
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise FFmpegMissing(str(exc)) from exc
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode, out, err


def _fraction(value: Optional[str]) -> Optional[float]:
    """ffprobe reports frame rates as '30000/1001'."""
    try:
        num, _, den = (value or "").partition("/")
        d = float(den or 1)
        return round(float(num) / d, 3) if d else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


async def _probe_video(asset_id: str) -> Dict[str, Any]:
    """Read duration, fps, rotation and audio presence straight from the container.

    Immich reports only a duration for videos and leaves exifInfo empty, so the
    fields needed to choose sensible timestamps have to come from the file.
    """
    code, out, err = await _run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams",
            "-headers", _ff_headers(),
            _video_url(asset_id, "original"),
        ],
        timeout=90.0,
    )
    if code != 0:
        raise RuntimeError((err or b"").decode(errors="replace")[-300:] or "ffprobe failed")

    data = json.loads(out or b"{}")
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})

    # Rotation lives in side data on modern files and in a tag on older ones.
    rotation = None
    for sd in video.get("side_data_list") or []:
        if sd.get("rotation") is not None:
            rotation = sd["rotation"]
    if rotation is None:
        try:
            rotation = int((video.get("tags") or {}).get("rotate"))
        except (TypeError, ValueError):
            rotation = None

    duration = (data.get("format") or {}).get("duration") or video.get("duration")
    return {
        "duration_seconds": round(float(duration), 3) if duration else None,
        "fps": _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate")),
        "rotation": int(rotation) if rotation is not None else 0,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "width": video.get("width"),
        "height": video.get("height"),
        "codec": video.get("codec_name"),
    }


async def _extract_frame(asset_id: str, at: float, max_dimension: int, source: str):
    """Decode one frame and return (jpeg_bytes, actual_pts, width, height).

    -ss before -i seeks by keyframe index, which is fast but can land seconds
    early on phone video; -accurate_seek makes ffmpeg decode forward from that
    keyframe to the requested moment, keeping the speed and losing the lie.
    -copyts then leaves timestamps on the source timeline so showinfo reports
    where the frame really came from, which is what gets reported back.
    Rotation is left to ffmpeg's default autorotate, which is on: phone video
    stores a rotation flag rather than rotated pixels, and an unrotated frame
    wastes the budget on letterboxing. It is not passed explicitly because
    -autorotate is a boolean flag, so "-autorotate 1" makes ffmpeg read the 1 as
    an output filename. The returned width and height are the proof it applied.
    """
    scale = (
        f"scale='min(iw,{max_dimension})':'min(ih,{max_dimension})'"
        ":force_original_aspect_ratio=decrease"
    )
    code, out, err = await _run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info",
            "-accurate_seek", "-ss", f"{at:.3f}", "-copyts",
            "-headers", _ff_headers(),
            "-i", _video_url(asset_id, source),
            "-frames:v", "1",
            "-vf", f"showinfo,{scale}",
            "-f", "image2", "-c:v", "png", "pipe:1",
        ]
    )
    if code != 0 or not out:
        raise RuntimeError((err or b"").decode(errors="replace")[-300:] or "no frame decoded")

    stderr = (err or b"").decode(errors="replace")
    match = _PTS_RE.search(stderr)
    actual = round(float(match.group(1)), 3) if match else None

    # ffmpeg emits PNG so the single lossy encode happens here, at a known quality,
    # rather than compounding with an intermediate JPEG.
    img = PILImage.open(io.BytesIO(out))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue(), actual, img.width, img.height


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
        is_video = meta.get("type") == "VIDEO"
        payload: Dict[str, Any] = {
            "asset_id": meta.get("id"),
            "filename": meta.get("originalFileName"),
            "type": "video" if is_video else "image",
            "taken_at": taken,
            "days_ago": _days_ago(taken),
            "mime_type": meta.get("originalMimeType"),
            "width": exif.get("exifImageWidth"),
            "height": exif.get("exifImageHeight"),
            "file_size_bytes": exif.get("fileSizeInByte"),
            "camera": exif.get("model"),
            "albums": [{"album_id": a.get("id"), "name": a.get("albumName")} for a in albums],
        }

        # Immich reports a duration for videos and leaves exifInfo empty, so fps,
        # rotation and audio have to be read out of the container itself. Without
        # duration here, immich_get_video_frames would have to burn a call just to
        # discover how long the clip is before it could choose timestamps.
        if is_video:
            try:
                probe = await _probe_video(asset_id)
                payload["video"] = probe
                payload["width"] = payload["width"] or probe.get("width")
                payload["height"] = payload["height"] or probe.get("height")
                payload["next"] = (
                    "Use immich_get_video_frames with count=6 for an overview, then explicit "
                    "timestamps to narrow in on whatever changed."
                )
            except FFmpegMissing:
                payload["video"] = {
                    "error": "ffmpeg is not installed in this container; rebuild the image.",
                    "duration_seconds": round(meta["duration"] / 1000, 3)
                    if isinstance(meta.get("duration"), (int, float)) else None,
                }
            except Exception as exc:  # noqa: BLE001 - metadata is still worth returning
                payload["video"] = {"error": f"could not probe the file: {exc}"}

        return json.dumps(payload, ensure_ascii=False, indent=1)
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@mcp.tool(
    name="immich_list_unfiled_assets",
    annotations={
        "title": "List Immich photos not in any album",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def immich_list_unfiled_assets(
    limit: Annotated[int, Field(description="Max assets to return (1-200).", ge=1, le=200)] = 20,
    offset: Annotated[int, Field(description="Assets to skip, for pagination.", ge=0)] = 0,
    order: Annotated[AssetOrder, Field(description="newest or oldest by capture date.")] = AssetOrder.NEWEST,
) -> str:
    """List photographs in the library that are not in any album yet - the inbox.

    This is where a phone's automatic backup leaves things, so it answers "what is
    new and unsorted?" - a question the album-based tools cannot ask.

    The intended loop: read this list, call immich_get_image on each entry to see
    what it actually is, then immich_add_to_album to file it. Pass a small
    max_dimension to immich_get_image while sorting; recognising a subject needs a
    fraction of the detail a full-size fetch returns.

    Returns:
        str: JSON with schema
        {
          "total": int, "count": int, "offset": int,
          "has_more": bool, "next_offset": int|null,
          "assets": [
            {"asset_id": str, "filename": str, "taken_at": str,
             "days_ago": int, "type": "image"|"video"}
          ]
        }
        An empty list means everything in the library has been filed.
    """
    try:
        # isNotInAlbum is Immich's own filter, so the "is it filed?" question is
        # answered by the database rather than by listing every album and
        # subtracting - which would be O(albums) requests and race with uploads.
        data = await _post_json("/api/search/metadata", {"isNotInAlbum": True, "size": 1000})
        items = (data.get("assets") or {}).get("items") or []
        items.sort(
            key=lambda a: _parse_ts(a.get("fileCreatedAt")) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=(order == AssetOrder.NEWEST),
        )
        total = len(items)
        page = items[offset : offset + limit]
        consumed = offset + len(page)
        return json.dumps(
            {
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
    name="immich_get_video_frames",
    annotations={
        "title": "Extract still frames from an Immich video",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def immich_get_video_frames(
    asset_id: Annotated[str, Field(description="Video asset id from immich_list_assets.")],
    timestamps: Annotated[Optional[List[float]], Field(
        description="Seconds from the start of the clip. Up to 8. Mutually exclusive with count."
    )] = None,
    count: Annotated[Optional[int], Field(
        description="Alternative to timestamps: N evenly spaced frames across the clip, 2-8. "
                    "Use for a first pass when the timeline is unknown.", ge=2, le=MAX_VIDEO_FRAMES
    )] = None,
    max_dimension: Annotated[int, Field(
        description="Longest edge in px (128-1440).", ge=128, le=PREVIEW_LONG_EDGE
    )] = 768,
) -> list:
    """Pull still frames out of a video at chosen moments, as real images.

    Video cannot be watched through this interface, but the events that matter in
    a short clip are usually seconds apart, so a handful of stills at the right
    moments carries nearly all of the diagnostic content.

    The intended loop is iterative narrowing: start with count=6 for an overview,
    read what changed and where, then come back with explicit timestamps to
    bracket the moment - [9.5, 10, 10.5, 11]. That only works because the
    timestamp reported for each frame is the one actually decoded, not the one
    requested; where the two differ by more than 0.2s the text block says so.

    For images use immich_get_image instead. Use immich_get_asset_metadata first
    to learn the duration, otherwise choosing timestamps is guesswork.

    Returns:
        list: for each frame in ascending time order, a TextContent naming the
        frame index and its true timestamp, followed by an ImageContent (JPEG);
        then a final TextContent with the clip duration, capture date and which
        rendition was decoded. On failure, a single TextContent explaining why.
    """
    try:
        if (timestamps is None) == (count is None):
            return [TextContent(type="text", text=(
                "Pass exactly one of timestamps or count. Use count for a first look at a "
                "clip whose timeline you do not know yet, then timestamps to narrow in."
            ))]

        meta = await _get_json(f"/api/assets/{asset_id}")
        if meta.get("type") != "VIDEO":
            return [TextContent(type="text", text=(
                f"{meta.get('originalFileName')} is an image, not a video. "
                "Use immich_get_image for still photographs; this tool decodes video only."
            ))]

        probe = await _probe_video(asset_id)
        duration = probe.get("duration_seconds")

        if timestamps is not None:
            if not timestamps:
                return [TextContent(type="text", text="timestamps was empty — pass at least one.")]
            if len(timestamps) > MAX_VIDEO_FRAMES:
                return [TextContent(type="text", text=(
                    f"{len(timestamps)} frames requested; the limit is {MAX_VIDEO_FRAMES} per call. "
                    "Eight frames is already a large amount of context — narrow the range and "
                    "call again rather than widening this one."
                ))]
            wanted = sorted(float(t) for t in timestamps)
            if wanted[0] < 0:
                return [TextContent(type="text", text="Timestamps must be zero or greater.")]
            if duration and wanted[-1] > duration:
                return [TextContent(type="text", text=(
                    f"Timestamp {wanted[-1]:.2f}s is past the end of this {duration:.2f}s clip."
                ))]
        else:
            if not duration:
                return [TextContent(type="text", text=(
                    "Could not read this clip's duration, so evenly spaced frames cannot be "
                    "placed. Pass explicit timestamps instead."
                ))]
            # Span the clip from its first moment to just short of the last, rather
            # than the centres of equal segments: the opening instant is usually the
            # baseline you compare everything else against, and the final frame is
            # where a decode is most likely to fall off the end of the file.
            span = duration * 0.98
            wanted = [round(span * i / (count - 1), 3) for i in range(count)]

        out: List[Any] = []
        drifted = 0
        for index, at in enumerate(wanted, start=1):
            try:
                jpeg, actual, width, height = await _extract_frame(
                    asset_id, at, max_dimension, "original"
                )
            except FFmpegMissing:
                return [TextContent(type="text", text=(
                    "ffmpeg is not installed in this server's container, so video frames "
                    "cannot be decoded. Rebuild the image — the Dockerfile installs it."
                ))]
            except Exception as exc:  # noqa: BLE001 - one bad frame must not lose the rest
                out.append(TextContent(type="text", text=(
                    f"Frame {index} of {len(wanted)} at {at:.2f}s could not be decoded: {exc}"
                )))
                continue

            shown = actual if actual is not None else at
            note = ""
            if actual is not None and abs(actual - at) > SEEK_DRIFT_TOLERANCE:
                drifted += 1
                note = (f" — NOTE: asked for {at:.2f}s, this frame is {actual:.2f}s "
                        f"({actual - at:+.2f}s); the nearest decodable frame was not where "
                        f"it was requested, so read the time on the frame, not the request")
            out.append(TextContent(type="text", text=(
                f"Frame {index} of {len(wanted)} — t={shown:.2f}s, {width}x{height}{note}"
            )))
            out.append(ImageContent(
                type="image", data=base64.b64encode(jpeg).decode("ascii"), mimeType="image/jpeg"
            ))

        taken = meta.get("fileCreatedAt")
        ago = _days_ago(taken)
        tail = (
            f"{meta.get('originalFileName')} — {duration:.2f}s"
            if duration else f"{meta.get('originalFileName')}"
        )
        tail += f", {probe.get('fps')} fps" if probe.get("fps") else ""
        tail += f", filmed {taken}" + (f" ({ago} days ago)" if ago is not None else "")
        tail += ". Decoded from the original file, not a transcode, so fine texture is intact."
        if probe.get("rotation"):
            tail += f" Rotation metadata of {probe['rotation']}° was applied."
        if drifted:
            tail += (f" {drifted} of {len(wanted)} frames landed more than "
                     f"{SEEK_DRIFT_TOLERANCE}s from the requested moment — see the notes above.")
        out.append(TextContent(type="text", text=tail))
        return out
    except Exception as exc:  # noqa: BLE001
        return [TextContent(type="text", text=_error(exc))]


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


# This server has exactly two routes. Everything else is answered 404, and that
# matters more than it looks.
#
# A client registering this server first probes for an OAuth authorization server
# at /.well-known/oauth-protected-resource and friends. Under the MCP auth spec a
# 401 means "authenticate, here is where to start" — so answering a discovery
# probe with 401 sends the client hunting for an OAuth server that does not
# exist. It then fails to register a client and reports that *this server's*
# sign-in service is broken, which points the blame at the wrong component
# entirely. A 404 says plainly: no OAuth here, use the credential you were given.
#
# Whitelisting the two real routes, rather than blacklisting /.well-known/,
# survives a reverse proxy that strips a path prefix before forwarding — in which
# case the probe arrives as /oauth-protected-resource with no /.well-known/ on it
# and a prefix test would miss.
#
# The same trap exists outside this process and cannot be fixed from in here: a
# SPA like Immich answers any unknown path with its index page and HTTP 200, so
# if it holds the root of the hostname it will answer these probes with HTML that
# the client tries to parse as OAuth metadata. Route the discovery paths to this
# server, or give it its own hostname.
OPEN_PATHS = frozenset({"/healthz"})


class BearerAuth(BaseHTTPMiddleware):
    """The tunnel URL is guessable in principle; this makes knowing it insufficient."""

    async def dispatch(self, request, call_next):
        path = request.url.path
        if path in OPEN_PATHS:
            return await call_next(request)

        # Not the MCP endpoint => nothing lives here. Say so, and say it before
        # the auth check, so a discovery probe never sees a 401.
        if path.rstrip("/") != MCP_PATH:
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
