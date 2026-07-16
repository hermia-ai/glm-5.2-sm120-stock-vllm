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

---

## 4. Claude Code cannot talk to vLLM directly — you need a normalizing shim

Claude Code (verified with 2.1.207) sends two things that stock vLLM rejects. Pointing
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

### What the shim must do

Write a small reverse proxy that sits in front of vLLM and normalizes requests. It is
maybe 100 lines. It must do exactly three things:

1. **Hoist non-`user`/`assistant` messages out of `messages[]` into the top-level `system`
   field.** Concatenate onto any existing `system` rather than overwriting it.
2. **Clamp the effort knob.** Map `reasoning_effort` and `output_config.effort` to a value
   vLLM accepts (`no_think` / `low` / `high`). Anything else — `xhigh`, `max`, future
   values — must be clamped, not passed through.
3. **Stream responses back as HTTP/1.1 chunked.** If you terminate the response by simply
   closing the connection, Node/undici (which Claude Code uses) treats it as a truncated
   response and retries. Send proper chunked framing and a clean terminating chunk.

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
