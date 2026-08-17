#!/usr/bin/env python3
"""Self-test container: probes an Immich MCP server and writes a JSON report to a
bind-mounted folder, so the result can be read by a person or by an agent without
anyone pasting terminal output.

It answers three questions about the image tool, from the returned bytes rather
than from the server's own claims: is it a real MCP image block, is it small enough
to be worth sending, and is it something a client can actually decode. The image is
written to disk beside the report so it can be opened and looked at.

Deliberately implementation-agnostic — it recognises the tool names used by other
Immich MCP servers as well, so it can be pointed at one for comparison.

Env:
  MCP_URL           server endpoint (default http://immich-mcp-ro:3001/mcp)
  MCP_BEARER_TOKEN  bearer token, if the server requires one
  IMMICH_URL        Immich itself, used only to pick a real test asset
  IMMICH_API_KEY    ditto
  PROBE_UPLOAD=1    also exercise the upload path; off by default, because it puts
                    a 1x1 JPEG in the library
  OUT_PATH          report location (default /out/report.json)
  WAIT_SECONDS      delay before starting, to let servers finish booting

Runs once per `docker compose up`, then exits.
"""

import base64
import json
import os
import time
from datetime import datetime, timezone

import httpx

OUT = os.environ.get("OUT_PATH", "/out/report.json")
TOKEN = os.environ.get("MCP_BEARER_TOKEN", "")
IMMICH_URL = os.environ.get("IMMICH_URL", "http://immich-server:2283")
IMMICH_KEY = os.environ.get("IMMICH_API_KEY", "")
MCP_URL = os.environ.get("MCP_URL", "http://immich-mcp-ro:3001/mcp")
TEST_UPLOAD = os.environ.get("PROBE_UPLOAD", "").lower() in ("1", "true", "yes")

TINY_JPEG = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEYIx8lJCIfIiEmKzcvJik0KSEiMEExNDk7Pj4+JS5ESUM8SDc9Pjv/2wBDAQoLCw4NDhwQEBw7KCIoOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozv/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDFooorwz5w/9k="

TARGETS = [
    {"name": "immich-mcp", "url": MCP_URL, "token": TOKEN},
]


def sse(text: str):
    for line in text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    return json.loads(text) if text.strip() else None


class Probe:
    def __init__(self, url, token):
        self.url, self.token, self.session = url, token, None

    def rpc(self, payload, timeout=60.0):
        h = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if self.session:
            h["Mcp-Session-Id"] = self.session
        r = httpx.post(self.url, headers=h, json=payload, timeout=timeout)
        if r.headers.get("Mcp-Session-Id") and not self.session:
            self.session = r.headers["Mcp-Session-Id"]
        if r.status_code >= 400:
            return {"__http": r.status_code, "__body": r.text[:400]}
        return sse(r.text)

    def handshake(self):
        res = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                   "clientInfo": {"name": "selftest", "version": "1"}}})
        if res and not res.get("__http"):
            self.rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return res

    def tools(self):
        r = self.rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        if not r or r.get("__http"):
            return [], r
        return [t["name"] for t in r.get("result", {}).get("tools", [])], None

    def call(self, name, args, timeout=90.0):
        return self.rpc({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                         "params": {"name": name, "arguments": args}}, timeout)


def pick_asset():
    """Grab one real HEIC asset id straight from Immich - the hardest case."""
    if not IMMICH_KEY:
        return None, "IMMICH_API_KEY not set"
    try:
        h = {"x-api-key": IMMICH_KEY, "Content-Type": "application/json"}
        albums = httpx.get(f"{IMMICH_URL}/api/albums", headers=h, timeout=30).json()
        for a in albums:
            r = httpx.post(f"{IMMICH_URL}/api/search/metadata", headers=h,
                           json={"albumIds": [a["id"]], "size": 50}, timeout=30).json()
            for it in (r.get("assets") or {}).get("items", []):
                if it.get("type") == "IMAGE" and "heic" in (it.get("originalMimeType") or ""):
                    return it, None
        return None, "no HEIC asset found"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def ok_result(res):
    """A tools/call succeeded only if there is a result AND isError is not set."""
    return bool(res) and not res.get("__http") and bool(res.get("result")) \
        and not res["result"].get("isError")


def brief(res):
    """Trim a response down to something readable in a report."""
    if not res:
        return None
    if res.get("__http"):
        return {"http": res["__http"], "body": res.get("__body", "")[:200]}
    out = []
    for c in (res.get("result") or {}).get("content", []):
        if c.get("type") == "text":
            out.append(c["text"][:200])
        else:
            out.append(f"<{c.get('type')} {c.get('mimeType')}>")
    return {"isError": (res.get("result") or {}).get("isError", False), "content": out}


MAGIC = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"RIFF": "webp",
}


def sniff(raw: bytes):
    """Identify the bytes by their actual header, not by what the server claims."""
    for sig, name in MAGIC.items():
        if raw.startswith(sig):
            return name
    if raw[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1"):
        return "heic"
    return "unknown"


def dimensions(raw: bytes):
    """Pull width/height straight out of the JPEG SOF marker. No image library needed."""
    if not raw.startswith(b"\xff\xd8"):
        return None
    i = 2
    while i < len(raw) - 9:
        if raw[i] != 0xFF:
            i += 1
            continue
        marker = raw[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            h = int.from_bytes(raw[i + 5:i + 7], "big")
            w = int.from_bytes(raw[i + 7:i + 9], "big")
            return {"width": w, "height": h}
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        i += 2 + int.from_bytes(raw[i + 2:i + 4], "big")
    return None


def judge(content):
    """Apply the three checks to a tools/call result: real image block, small
    enough to send, and decodable by an ordinary client."""
    v = {"check1_image_block": False, "check2_resized": False, "check3_not_heic": False,
         "detail": []}
    for c in content or []:
        t = c.get("type")
        if t == "image":
            raw = base64.b64decode(c.get("data", "") + "==")
            n = len(raw)
            v["check1_image_block"] = True
            v["bytes"] = n
            v["kb"] = round(n / 1024)
            v["mime"] = c.get("mimeType")
            v["check2_resized"] = n < 900_000
            v["check3_not_heic"] = "heic" not in (c.get("mimeType") or "").lower()

            # The claim is the mimeType; this is the evidence. Sniff the real header,
            # read the real dimensions, and drop the bytes on disk so a human (or
            # Claude) can open the thing and confirm it is a photograph.
            v["actual_format"] = sniff(raw)
            v["dimensions"] = dimensions(raw)
            v["header_matches_mimetype"] = (
                v["actual_format"] in (c.get("mimeType") or "").lower()
            )
            try:
                p = os.path.join(os.path.dirname(OUT), "sample.jpg")
                with open(p, "wb") as fh:
                    fh.write(raw)
                v["saved_to"] = p
            except Exception as e:
                v["saved_to"] = f"failed: {e}"

            v["detail"].append(f"image block {v['kb']}KB {v['mime']} "
                               f"(header says {v['actual_format']}, {v['dimensions']})")
        elif t == "text":
            txt = c.get("text", "")
            v["detail"].append("text: " + txt[:200])
    return v


def main():
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "targets": []}

    asset, err = pick_asset()
    report["test_asset"] = ({"id": asset["id"], "file": asset["originalFileName"],
                             "mime": asset["originalMimeType"]} if asset else {"error": err})

    for t in TARGETS:
        entry = {"name": t["name"], "url": t["url"]}
        p = Probe(t["url"], t["token"])
        try:
            entry["initialize"] = p.handshake()
            names, err = p.tools()

            # Gateway-mode servers advertise only two tools until categories are
            # enabled. Do that, then re-list, or every later lookup finds nothing.
            if "immich_tools_enable" in names:
                entry["gateway_mode"] = True
                p.call("immich_tools_enable", {"categories": ["all"]})
                names, err = p.tools()

            entry["tools"] = names
            if err:
                entry["tools_error"] = err

            img_tool = next((n for n in names if n in
                             ("immich_get_image", "immich_assets_download_thumbnail",
                              "immich_assets_download_original")), None)
            entry["image_tool"] = img_tool
            entry["writes_present"] = [n for n in names
                                       if any(w in n for w in ("upload", "delete", "create", "update", "remove"))]

            if img_tool and asset:
                for argname in ("asset_id", "id", "assetId"):
                    res = p.call(img_tool, {argname: asset["id"]})
                    # isError=True still arrives inside "result" - treating that as
                    # success is what made the last run report a bogus verdict.
                    if ok_result(res):
                        entry["verdict"] = judge(res["result"].get("content"))
                        entry["arg_used"] = argname
                        break
                    entry.setdefault("call_errors", []).append({argname: brief(res)})

            # Upload path, if this server exposes one. A 1x1 JPEG - enough to prove
            # the round trip without cluttering the library with anything visible.
            # Still opt-in: it does write to a real library.
            up_tool = next((n for n in names if "upload" in n and "authorize" not in n
                            and "init" not in n and "status" not in n
                            and "from_path" not in n), None)
            entry["upload_tool"] = up_tool
            if up_tool and not TEST_UPLOAD:
                entry["upload_ok"] = "skipped (set PROBE_UPLOAD=1 to test)"
            elif up_tool:
                shapes = [
                    {"image_base64": TINY_JPEG, "filename": "immich-mcp-selftest.jpg"},
                    {"fileContent": TINY_JPEG, "fileName": "immich-mcp-selftest.jpg"},
                ]
                for s in shapes:
                    res = p.call(up_tool, s)
                    if ok_result(res):
                        entry["upload_ok"] = True
                        entry["upload_args"] = sorted(s)
                        entry["upload_result"] = brief(res)
                        break
                    entry.setdefault("upload_errors", []).append(brief(res))
                entry.setdefault("upload_ok", False)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        report["targets"].append(entry)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    time.sleep(float(os.environ.get("WAIT_SECONDS", "12")))  # let servers finish booting
    main()
