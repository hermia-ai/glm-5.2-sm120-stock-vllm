# Using GLM-5.2 on vLLM as an Anthropic-API backend

This page covers driving the GLM-5.2-504B-NVFP4 server described in this repo from
Anthropic-SDK clients (Claude Code, `anthropic` Python/TS SDKs, anything that speaks the
Messages API).

Everything below was probed live against **stock vLLM 0.25.1** serving
`0xSero/GLM-5.2-504B-Nvidia` with the flags from this repo's launch script. HTTP status
codes and token counts are measured, not assumed.

Throughout, `$HOST` is wherever you run the server (`localhost` if that's the same box) and
`$PORT` is whatever you passed to `--port`.

---

## 1. vLLM serves `/v1/messages` natively

vLLM 0.25.1 implements the Anthropic Messages API directly. For raw API calls you do
**not** need a translation proxy, LiteLLM, or any OpenAI→Anthropic shim.

```bash
curl -s http://$HOST:$PORT/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -H 'x-api-key: dummy' \
  -d '{
    "model": "glm-5.2",
    "max_tokens": 1024,
    "system": "You are concise.",
    "messages": [
      {"role": "user", "content": "What is the capital of Japan? One word."}
    ]
  }'
```

Response shape is the real thing: `"type": "message"`, `"role": "assistant"`,
a `content` array, and a populated `usage` object. The top-level `system` field is honored.

> `x-api-key` is only present to satisfy SDK clients that insist on sending one. Use
> vLLM's `--api-key` if you want it to actually mean something.

The model name must match what the engine registered — check `GET /v1/models` and use that
string verbatim.

**Token counting** works too:

```bash
curl -s http://$HOST:$PORT/v1/messages/count_tokens \
  -H 'content-type: application/json' \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"hello"}]}'
```

**Streaming** works and emits all six SSE event types in the correct order:
`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`,
`message_delta`, `message_stop`.

---

## 2. Support matrix

| Feature | Status | Notes |
|---|---|---|
| `POST /v1/messages` | ✅ works | `type=message`, `role=assistant`, top-level `system` honored, `usage` populated |
| Streaming SSE | ✅ works | all 6 events emitted |
| Tool use | ✅ works | `stop_reason=tool_use`, content `[thinking, text, tool_use]`, correct arguments |
| Native thinking blocks | ✅ works | real Anthropic-shaped `thinking` content blocks |
| `POST /v1/messages/count_tokens` | ✅ works | |
| OpenAI-style `tool_calls` (on `/v1/chat/completions`) | ✅ works | |
| Tool search (`tool_search_tool_regex_20251119`, `defer_loading`) | ❌ HTTP 400 | `"missing input_schema"` — vLLM treats every tool as a custom tool |
| Memory tool (`memory_20250818`) | ❌ HTTP 400 | same cause: server-side tool types aren't recognized |
| Context editing (`context_management.edits` / `clear_tool_uses_20250919`) | ⚠️ no-op | returns **200** but echoes nothing back; nothing is edited |
| `cache_control` blocks | ⚠️ silently ignored | accepted, no error, no effect |
| Claude Code **WebSearch** | ❌ HTTP 400 | same `missing input_schema` cause — CC's nested sub-request declares the *server* tool `web_search_20250305`. **400 = 1 attempt, no retry, no hang.** Fixable in the shim: see [§4 web search](#web-search-why-it-400s-and-why-that-is-fine) |
| Claude Code **WebFetch** | ✅ works | 100% client-side — CC fetches the URL and converts HTML→markdown itself, then asks the model to summarize. No server tool is ever declared. **Don't "fix" it.** |
| `image` content blocks | ❌ HTTP **500** | `"glm-5.2 is not a multimodal model"` — the model is text-only. **500 → Claude Code retries → hang.** Fixable in the shim: see [§4 vision bridge](#the-vision-bridge-giving-a-text-only-model-eyes) |

### On tool search

Because tool search returns 400, Claude Code's `ENABLE_TOOL_SEARCH` is a no-op at best and
a hard failure at worst against vLLM. Don't enable it.

### On prompt caching — read this before you copy Anthropic advice

`cache_control` blocks are accepted and **silently ignored**. You cannot even measure a
cache hit: `usage` comes back with only `input_tokens` and `output_tokens` — there is no
`cache_creation_input_tokens` and no `cache_read_input_tokens`. Anthropic prompt-caching
economics simply do not apply here.

This is the general lesson: **most published advice for "reducing Claude Code context
cost" is Anthropic-API-specific and does not transfer to vLLM.** Cache breakpoints, tool
search, memory tools, context editing — all of it targets server-side behavior that vLLM
does not implement.

The local equivalent is **vLLM automatic prefix caching**, which is `enable_prefix_caching=True`
by default. It gives you the compute benefit (skipped prefill on a shared prefix) for free,
with no client-side annotation at all. And it wants KV headroom to retain history — which
is exactly what the DCP patch in this repo buys you: the KV pool goes from 164,160 to
554,759 tokens, giving prefix caching far more room to keep prefixes resident instead of
evicting them.

So: delete your `cache_control` blocks, don't replace them with anything, and spend the
effort on KV capacity instead.

---

## 3. THE BIG ONE: `max_tokens` must be ≥ 1024

GLM-5.2 **always emits a thinking block first.** If `max_tokens` is small, the entire
budget is consumed by thinking and you get a **silently empty response with no error**.

Measured, same prompt ("What is the capital of Japan? One word."), only `max_tokens` changed:

| `max_tokens` | `stop_reason` | content blocks | visible text |
|---|---|---|---|
| 64 | `max_tokens` | `[(thinking, 193 chars)]` | `""` ← **empty, no error** |
| 1024 | `end_turn` | `[(thinking, 353), (text, 6)]` | `"Tokyo."` |

Note the thinking block at `max_tokens=64` is *longer than the budget* — it's the parsed
reasoning stream, and it ran until the cap cut it off before any `text` block ever started.

**Rule: set `max_tokens >= 1024` for every request.** If your client returns empty strings
and you're about to debug the server, check this first.

### The general form of this bug (it will bite you twice)

This is not a GLM quirk — it is what **any** `--reasoning-parser` does. The parser strips
thinking out of `content`. So when a reasoning model runs out of budget while still thinking:

> **a too-small `max_tokens` produces EMPTY output, not truncated output.**

There is no error, no warning, and `reasoning_content` is often not populated either — you
get `finish_reason: "length"` and `content: ""`. Every downstream layer then misreports it
as something else entirely: a proxy sees "no description", a client sees "the model refused",
a user sees *"I can't see the image"*.

**We shipped this bug ourselves.** The vision bridge in §5 below called a reasoning-enabled
vision model with `max_tokens: 800`. Small test images with a short prompt fit the budget and
passed. Real screenshots with the real (thorough) prompt did not — the model thought past 800,
returned empty content, the bridge substituted its "backend unavailable" placeholder, and the
text-only model dutifully told the user it couldn't see the image. The bridge had run
*correctly* and logged success. **The whole failure was 800 vs 2000 tokens.**

Two defences, use both:

```jsonc
{
  "chat_template_kwargs": {"enable_thinking": false},  // thinking off where you don't need it:
                                                       // measured 603 -> 155 completion tokens
  "max_tokens": 2000                                   // and a budget well clear of the cap
}
```

And log `finish_reason` whenever you get empty content — `length` names this bug instantly:

```python
if not txt:
    log("empty content; finish=%s" % choice.get("finish_reason"))   # "length" == this bug
```

**Diagnostic tell:** if it works on your small test fixture and fails on real input, suspect
the token budget before you suspect the model. Test with the *real* prompt, not a short one.

---

## 4. Claude Code cannot talk to vLLM directly — you need a normalizing shim

Claude Code (verified with 2.1.207) sends three things that stock vLLM rejects. Pointing
`ANTHROPIC_BASE_URL` straight at vLLM will appear to hang.

### What breaks

**(a) A `role: "system"` message inside `messages[]`.**
The Anthropic API takes `system` as a *top-level* field. Claude Code additionally places a
system-role entry in the `messages` array. vLLM responds **400**, or the request hangs.

**(b) `output_config: {"effort": "xhigh"}`.**
vLLM returns **500**:

```
reasoning_effort error : xhigh, should be no_think/low/high
```

Claude Code then retries roughly 8 times. From the UI this is indistinguishable from a
hang — you see a spinner, not an error.

**(c) An image, if your model is text-only.** GLM-5.2 is text-only. Paste or attach an
image in Claude Code and it sends a real Anthropic `image` content block:

```
POST /v1/messages  →  HTTP 500
{"type":"error","error":{"type":"internal_error","message":"glm-5.2 is not a multimodal model"}}
```

Note the status. The **OpenAI** route returns a clean `400` for the same input, but
`/v1/messages` returns **500** — which lands in Claude Code's retry loop and becomes a
hang, exactly like (b). You get a spinner, not "this model can't see images."

> **An MCP vision tool does not fix this**, and it's worth understanding why before you
> build one. A `describe_image(path)` MCP tool only helps when the model *chooses to call
> it* about a file on disk. A **pasted** image is never a tool call — it arrives as an
> `image` block in `messages[]` and is rejected before the model is invoked at all. The
> model never gets the chance to route around it. The fix has to sit in the one layer that
> sees every request: the shim.

### What the shim must do

Write a small reverse proxy that sits in front of vLLM and normalizes requests. It is
maybe 100 lines (150 with vision). It must do these things:

1. **Hoist non-`user`/`assistant` messages out of `messages[]` into the top-level `system`
   field.** Concatenate onto any existing `system` rather than overwriting it.
2. **Clamp the effort knob.** Map `reasoning_effort` and `output_config.effort` to a value
   vLLM accepts (`no_think` / `low` / `high`). Anything else — `xhigh`, `max`, future
   values — must be clamped, not passed through.
3. **Stream responses back as HTTP/1.1 chunked.** If you terminate the response by simply
   closing the connection, Node/undici (which Claude Code uses) treats it as a truncated
   response and retries. Send proper chunked framing and a clean terminating chunk.
4. **Substitute images with a description** (only if your model is text-only — see the
   vision bridge below).

Pseudocode for the request path:

```python
def normalize(body: dict) -> dict:
    # 1. hoist stray system-role messages to the top-level field
    kept, hoisted = [], []
    for m in body.get("messages", []):
        if m.get("role") in ("user", "assistant"):
            kept.append(m)
        else:
            hoisted.append(as_text(m.get("content")))
    if hoisted:
        existing = body.get("system")
        prefix = as_text(existing) if existing else ""
        body["system"] = "\n\n".join(x for x in [prefix, *hoisted] if x)
    body["messages"] = kept

    # 2. clamp effort to what vLLM accepts
    ALLOWED = {"no_think", "low", "high"}
    if body.get("reasoning_effort") not in (None, *ALLOWED):
        body["reasoning_effort"] = "high"
    oc = body.get("output_config")
    if isinstance(oc, dict) and oc.get("effort") not in (None, *ALLOWED):
        oc["effort"] = "high"

    # 3. enforce the thinking-block floor from section 3
    if body.get("max_tokens", 0) < 1024:
        body["max_tokens"] = 1024

    return body
```

Then forward to `http://$HOST:$PORT/v1/messages`, and relay the SSE stream back with
`Transfer-Encoding: chunked`.

Point `ANTHROPIC_BASE_URL` at the shim, not at vLLM.

> This repo deliberately does not ship an implementation — the normalization above is the
> whole specification, and your proxy of choice (a FastAPI app, a Caddy/nginx Lua handler,
> a few dozen lines of Node) will differ. What matters is that you normalize these three
> things; anything that does is sufficient.

Plain Anthropic SDK usage that doesn't do (a) or (b) needs no shim at all — call vLLM
directly.

---


### The vision bridge: giving a text-only model eyes

If your model is text-only, add a fourth transform. When a request contains an `image`
block, don't forward it — send the image to *any* multimodal model you can reach, and
replace the block with the text it returns. The text-only model then receives a normal
text request and never knows an image was involved.

This works for pasted images, screenshots, drag-and-drop — anything, because it operates
below the model and requires no cooperation from it.

```python
VISION_ROUTES = {"my-text-only-route"}      # ONLY these get rewritten
VISION_URL    = "http://<vision-host>:<port>/v1"   # any OpenAI-shaped multimodal server
VISION_MODEL  = "<a-multimodal-model>"
VISION_PROMPT = ("Describe this image thoroughly and objectively. Transcribe any text, "
                 "code, or error messages verbatim. Note UI layout, diagrams, and data. "
                 "Be specific: a text-only model will rely entirely on your description.")

def describe(ref: str) -> str | None:
    """ref is a `data:<mime>;base64,<...>` URI or a plain URL. Returns text, or None."""
    try:
        r = post(f"{VISION_URL}/chat/completions", {
            "model": VISION_MODEL, "max_tokens": 800, "temperature": 0.2,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": ref}},
                {"type": "text", "text": VISION_PROMPT}]}],
        }, timeout=180)
        return r["choices"][0]["message"]["content"].strip() or None
    except Exception:
        return None            # never propagate — see the fallback rule below

def substitute_images(body: dict, route: str) -> dict:
    if route not in VISION_ROUTES:
        return body                       # multimodal routes keep their image blocks
    for m in body.get("messages", []):
        c = m.get("content")
        if not isinstance(c, list):
            continue
        out = []
        for b in c:
            if isinstance(b, dict) and b.get("type") == "image":
                src = b.get("source") or {}
                if src.get("type") == "base64":
                    ref = f"data:{src.get('media_type','image/png')};base64,{src['data']}"
                elif src.get("type") == "url":
                    ref = src["url"]
                else:
                    ref = None
                desc = describe(ref) if ref else None
                out.append({"type": "text", "text":
                    f"[Image described by {VISION_MODEL}]\n{desc}" if desc else
                    "[Image omitted: this model is text-only and the vision backend "
                    "was unavailable. Ask the user to describe it, or retry.]"})
            else:
                out.append(b)
        m["content"] = out
    return body
```

**Four rules that matter more than the code:**

1. **Gate it per-route.** Never rewrite images for a route whose model *is* multimodal —
   you'd be throwing away real image input and replacing it with a lossy paraphrase.
2. **Never let a vision failure become a 500.** If the vision backend is down, substitute a
   placeholder string. A placeholder produces a model that says "I can't see the image";
   a propagated 500 produces a *hang*. The whole point of this transform is removing a
   hang, so don't reintroduce one on the error path.
3. **Ship a kill switch** (e.g. `VISION_DISABLE=1`). This runs on every request; you want
   to turn it off without redeploying.
4. **Stay byte-identical when there's no image.** Don't re-serialize requests you didn't
   change — it makes the transform trivially safe to leave on and easy to debug.

**Cost:** one extra round-trip, only when an image is actually present. Measured ~5 s
end-to-end (vision description + a short GLM-5.2 answer) with a ~35B vision model on a LAN
host. There's no cost at all on text-only requests.

**Verified:** pasting a solid RGB(0,128,255) PNG at GLM-5.2 through a shim doing exactly
this returned `HTTP 200` in 4.9 s with `stop_reason: end_turn` and the answer *"The color
of the image is bright blue."* — from a model that cannot see. The identical request
without the bridge returns `HTTP 500` and hangs the client.

**What this does *not* replace:** a `describe_image(path)` MCP tool is still worth having.
It's better when the image is a file on disk the model can reason about deliberately
("look at ./failing-test.png"), it costs nothing when unused, and the model controls the
question it asks. The bridge covers the case the tool structurally cannot: an image the
user pastes.

### Web search: why it 400s, and why that is fine

`web_search_20250305` is an Anthropic **server tool**. It has `type` + `name` and **no
`input_schema`** — so vLLM's `tools[]` model rejects it at body-parse, in about a
millisecond, before the GPU is touched. Identical cause to the tool-search and memory-tool
rows above. Version doesn't help: `web_search_20260209` and `web_fetch_20260209` fail the same way.

**The architecture is not what it looks like, and this trips people up.** Claude Code
(verified on 2.1.207) does *not* put the server tool in your main request:

| Request | What's in `tools[]` | vLLM |
|---|---|---|
| **Main** conversation request | `WebSearch` as an ordinary **client** tool, with a valid `input_schema` | ✅ 200 |
| **Nested sub-request** fired from inside `WebSearch.call()` | the **server** tool `{"type":"web_search_20250305","name":"web_search","max_uses":8}`, `tools:[]`, forced `tool_choice` | ❌ 400 |

So the blast radius is **that one sub-request**, not your whole turn. Your other tools are
fine. We initially measured this wrong — a synthetic request with a server tool mixed in
alongside client tools *does* 400 the entire body, which looks like "you lose all tools."
Real Claude Code never sends that shape.

**Crucially, a 400 does not hang.** Measured against a mock endpoint:

| Status | Claude Code behaviour |
|---|---|
| **400** | **exactly 1 attempt**, ~1s, surfaced as `API Error: 400 …` |
| **500** | ~11 attempts, exponential backoff capped at 32s → **~160s+ stall** |

So the model simply sees the error and falls back to whatever other search tool you've
given it (an MCP search server, say). **Search still works.** The real cost is one doomed
round-trip and a red error in the transcript on every search.

> ### ⚠️ Never map an unsupported feature to a 5xx
> This is the single most important line in this document. That 400 is *load-bearing*. If
> your proxy "helpfully" rewrites unknown-tool errors to 500, you convert a 1-second
> graceful failure into a ~3-minute stall. Same for a body containing
> `"type":"overloaded_error"` or a `x-should-retry: true` header — both get retried
> **even on a 4xx**. And if a `fallbackModel` is configured, a 400 can *silently switch
> models* instead of erroring.

**The fix, if you want it clean:** strip `WebSearch` from `tools[]` in your shim **by name**,
and add a system line pointing the model at your MCP search tool. The model then never calls
it, the sub-request never fires, and you go straight to a tool that works. Measured on our
box: **0 × 400, and 21s vs 32s** for the same prompt — a third faster, because the wasted
round-trip is gone.

```python
# in normalize_body(), gated to your local routes + a kill switch
kept = [t for t in j["tools"] if not (isinstance(t, dict) and t.get("name") == "WebSearch")]
if len(kept) != len(j["tools"]):
    j["tools"] = kept or None          # avoid tools:[] edge cases
    tc = j.get("tool_choice")          # never leave a forced choice naming a removed tool
    if isinstance(tc, dict) and tc.get("type") == "tool" and tc.get("name") == "WebSearch":
        j["tool_choice"] = {"type": "auto"} if kept else None
    # + append a system hint naming your MCP search tool (idempotently)
```

**Strip by NAME — not by "missing `input_schema`".** It is tempting to write the general rule
"drop any tool without a schema." Don't. That rule fires on the *sub-request*, whose forced
`tool_choice` still names `web_search` — you get a dangling forced choice, and at best another
400, at worst a reply with no scrapeable results: **a silent empty search instead of a loud
error.** Loud beats silent. Let that path keep 400ing; once the name-strip works it is
unreachable anyway.

**Leave WebFetch alone.** It is entirely client-side and already works on any endpoint. An
over-broad "strip the web tools" rule breaks something that isn't broken.

**What we deliberately did *not* build:** full server-tool emulation — having the shim run the
search itself and synthesize `server_tool_use` / `web_search_tool_result` blocks so CC's native
WebSearch "just works." It's seductive and it's a trap: the shim's response path is an opaque
byte relay with no SSE producer, the real block shapes are undocumented and we never captured
them, and the entire payoff is a *label* — the MCP tool already returns correct, cited results.
If you attempt it anyway, get two measurements first: confirm the sub-request actually sets
`stream:true`, and capture a real `web_search_tool_result` from `api.anthropic.com`. Without
both, you're reverse-engineering a wire format into your hot path.

---

## 5. Sampling parameters

The `0xSero/GLM-5.2-504B-Nvidia` model card asks for:

```
min_p = 0.05
repetition_penalty = 1.05
```

This matters because this checkpoint is a **REAP-pruned cut that has not been Router-KD
recovered**. Those two settings are the card's compensation for that; treat them as
defaults rather than tuning knobs.

Note the interaction with speculative decoding: **MTP disables `min_p`.** vLLM prints

```
WARNING __init__.py:204 min_p and logit_bias parameters will not work with speculative decoding
```

That's one more reason MTP is a bad trade on this setup — see the MTP section of the main
README, where it also turns out not to fit alongside DCP4 at 250K context.

Anthropic-SDK clients typically expose `temperature` / `top_p` / `top_k` but **not**
`min_p` or `repetition_penalty`. To get them applied, either set them server-side, or have
your shim inject them into every forwarded request body — vLLM accepts both fields on
`/v1/messages` as extensions.
