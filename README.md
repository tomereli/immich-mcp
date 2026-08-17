# immich-mcp

An MCP server over a self-hosted [Immich](https://immich.app) library that hands a model **actual pixels** — a real MCP image content block, resized and JPEG-encoded — instead of a URL it cannot fetch.

Read-only by default. Nothing in it can delete, in any mode.

```
you: "has the hoya put out new growth since June?"
     -> immich_list_albums    -> immich_list_assets(order=oldest)
     -> immich_get_image      -> the model is looking at the photograph
```

---

## Tools

| Tool | Purpose | Mode |
|---|---|---|
| `immich_list_albums` | album id, name, asset count, most recent asset + `days_ago` | read |
| `immich_list_assets` | assets in an album, `newest`/`oldest`, paginated, each with `days_ago` | read |
| `immich_get_image` | one photo as an MCP **image content block**, resized, JPEG | read |
| `immich_get_asset_metadata` | date, dimensions, size, camera, album membership — no pixels | read |
| `immich_upload_image` | add an image and file it under an album **by name**, created if absent | full |
| `immich_create_album` | create an empty album | full |
| `immich_add_to_album` | add existing assets to an album | full |
| `immich_update_album` | rename an album or change its description | full |
| `immich_remove_from_album` | take assets out of an album; they stay in the library | full |
| `immich_delete_album` | delete an album; its photographs stay in the library | full |
| `immich_trash_assets` | move photos to Immich's trash — recoverable | full |
| `immich_restore_assets` | bring them back out of the trash | full |

`MCP_MODE=read` is the default and the write tools are **not registered** — they are absent from `tools/list`, so there is nothing for a client to talk itself into.

### Nothing in `full` mode can destroy a photograph

That is a property of the code, not a promise about prompting. Immich's asset-delete API takes a `force` flag that bypasses the trash and erases immediately; **this server never sends it**, and no tool exposes an argument that could reach it. The strongest available action is a move to the trash, which Immich retains for the period configured on the server — 30 days out of the box — and `immich_restore_assets` reverses.

Deleting an album deletes the label, not the contents; Immich albums are tags rather than folders, so the assets outlive them. Emptying the trash for real is left to Immich's own interface, where a human does it deliberately.

The reasoning: an agent can be talked into things, and "clean that up" is genuinely ambiguous. Making the worst case cost a click to undo is cheaper than making the tool descriptions eloquent.

Every entry carries `days_ago` alongside the timestamp, so a model comparing dates never has to do arithmetic it might get wrong.

---

## The one thing worth stealing from this repo

**You do not need a HEIC decoder.** Immich already built one.

Immich generates renditions of every asset at import, and the `preview` rendition is **already JPEG**, at roughly 1440 px on the long edge. Reading from that endpoint means no `libheif`, no `pillow-heif`, no native build step, and nothing to break when Immich changes how it stores originals:

```
GET /api/assets/{id}/thumbnail?size=thumbnail   -> image/webp, ~12 KB
GET /api/assets/{id}/thumbnail?size=preview     -> image/jpeg, ~1440 px long edge
GET /api/assets/{id}/original                   -> whatever the camera wrote, HEIC on iPhone
```

Measured by the self-test in this repo, one iPhone photo, default settings:

| | bytes | mimeType | header sniff | decodable by an ordinary client |
|---|---|---|---|---|
| Immich original | 2.6 MB | `image/heic` | heic | no |
| **what this server returns** | **210 KB** | `image/jpeg` | jpeg | yes |

Roughly 13× smaller and universally decodable, with no image format work done anywhere in this codebase beyond a resize.

The visible consequence is that `max_dimension` is **capped at 1440**. That is not a limitation worth removing — going above it would mean decoding HEIC to gain detail nobody needs.

One subtlety that is easy to get wrong: return the mimeType that matches the **bytes you actually send**, not the asset's original mimeType. A JPEG labelled `image/heic` fails to render and looks like a server bug.

---

## Install

Clone next to the `docker-compose.yml` that already runs Immich:

```bash
git clone https://github.com/tomereli/immich-mcp.git
```

Add the two secrets to Immich's existing `.env` (see [.env.example](.env.example)):

```
IMMICH_API_KEY=<Immich: avatar -> Account Settings -> API Keys>
MCP_BEARER_TOKEN=<anything long and random>
```

Append the services in [docker-compose.example.yml](docker-compose.example.yml) to that compose file's `services:` block, then:

```bash
docker compose up -d --build
```

```bash
curl http://localhost:3000/healthz
```

`{"ok":true,...}` means it is up. `/healthz` is deliberately the **only** unauthenticated route.

To run it somewhere other than Immich's own compose project, drop `depends_on` and point `IMMICH_URL` at a reachable address.

---

## Exposing it

The intended client is a hosted assistant, which only connects to remote URL-based MCP servers — so the endpoint has to be reachable from the internet. A [Tailscale Funnel](https://tailscale.com/kb/1223/funnel) does that without opening a port:

```bash
tailscale funnel --bg --set-path=/immichmcp 3000
```

The server mounts its endpoint at `/mcp`, so with the path above the public URL is `https://<your-machine>.<tailnet>.ts.net/immichmcp/mcp`.

Register that as a custom connector with the header:

```
Authorization: Bearer <MCP_BEARER_TOKEN>
```

**If the connector UI will not let you set a header**, there is a fallback. Set `MCP_ALLOW_QUERY_TOKEN=1` and the server also accepts `?token=<MCP_BEARER_TOKEN>` on the URL. It is **off by default and should stay off if the header works** — tokens in query strings end up in proxy and server logs in a way headers do not. Use it only if the alternative is not shipping.

### Why the token matters more than it looks

The Immich API key stays on the host and is never sent to a client. The bearer token is the only thing standing between the open internet and your library, because a tunnel URL is guessable in principle. That is also why capability lives on **this** server rather than in the API key: `MCP_MODE` governs what the exposed endpoint can do, independently of what the key could do.

---

## Self-test

`selftest/` is a container that runs once per `docker compose up`, picks a real HEIC asset out of your library, calls the image tool, and writes `selftest-out/report.json` plus the image it got back. Point an agent at the report and it can confirm the thing works without you pasting any terminal output.

It judges the returned bytes rather than the server's claims — sniffing the file header and reading the dimensions straight out of the JPEG SOF marker:

```json
"verdict": {
  "check1_image_block": true,
  "check2_resized": true,
  "check3_not_heic": true,
  "bytes": 215201, "kb": 210, "mime": "image/jpeg",
  "actual_format": "jpeg", "header_matches_mimetype": true,
  "dimensions": { "width": 768, "height": 1024 }
}
```

It also recognises the tool names used by other Immich MCP servers, so you can point `MCP_URL` at one and compare like for like.

The upload path is **skipped unless you set `PROBE_UPLOAD=1`**, because testing it means writing a 1×1 JPEG into a real library.

---

## Things that cost me time

### `GET /api/albums/{id}` returns no assets in Immich v3

In Immich v3.1.0 that endpoint gives you album metadata and no assets array at all. The way to list an album's contents is search:

```
POST /api/search/metadata  {"albumIds": ["<id>"], "size": 1000}
```

and read `assets.items`. Sort by `fileCreatedAt` yourself — the returned order is not chronological.

### FastMCP's DNS-rebinding protection rejects every legitimate caller here

It refuses any `Host` header not on an allowlist, which breaks all three real callers: another container reaching the service by its compose name, a tunnel presenting the public `.ts.net` name, and a local probe using `localhost`. It is disabled deliberately (`TransportSecuritySettings(enable_dns_rebinding_protection=False)`). Nothing is lost, because the bearer token is what actually gates access — and unlike a `Host` header, an attacker cannot set it.

### The `mcp` SDK version is pinned on purpose

`FastMCP` moved out of `mcp.server.fastmcp` in a later SDK release, so an unpinned `>=1.9.0` happily resolves to a version where the import in this file does not exist. `mcp==1.27.0` is verified working.

### Videos need an explicit refusal

`immich_get_image` cannot render a `.MOV`, and most albums have one. Rather than fail obscurely it names the file and points at `immich_list_assets` to find a still in the same album — a model reading that recovers on its own.

### Name the machine in the "unreachable" message

A self-hosted library goes down because the machine under it is off. Set `IMMICH_HOST_HINT` and every failure message tells its reader which box to go and switch on, instead of surfacing a connection error nobody can act on.

---

## Verified against

Immich **v3.1.0**, `mcp` **1.27.0**, Python 3.12, Docker Desktop on Windows 11. The measurements above come from the self-test in this repo run against a live library, not from documentation.

## Related

[barryw/ImmichMCP](https://github.com/barryw/ImmichMCP) — a much larger C#/.NET Immich MCP server, ~49 tools. Worth a look if you want broad API coverage rather than a small read-only surface; at time of writing its image tools return originals and it documents no endpoint authentication of its own.

Not affiliated with or endorsed by the Immich project.

## License

MIT — see [LICENSE](LICENSE).
