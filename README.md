# GLM-5.2-504B (NVFP4) on 4× RTX PRO 6000 — 250K context, stock vLLM

**GLM-5.2-504B NVFP4 serving at 250,000-token context on 4× RTX PRO 6000 Blackwell (sm_120), on stock vLLM 0.25.1.
No fork. No Docker. No CUDA 13.2. No torch 2.12. No b12x.** One ~126-line patch to a single vLLM file, one deleted
key in the checkpoint's `config.json`, and one pip version bump. That's the whole delta.

This repo exists because three separate walls sit between a stock `pip install vllm` and a working GLM-5.2 server on
sm_120, and each one costs hours to diagnose: an OOM caused by an upstream `transformers` bug, a `TypeError` caused by
a version pin vLLM doesn't honour at runtime, and an `AssertionError` that tells you to switch to a backend that does
not exist on your hardware. All three are documented here with the exact error strings, the exact `file:line` causes,
and the fixes. Everything below is measured on real hardware on 2026-07-15, not estimated.

---

## Results

Model: [`0xSero/GLM-5.2-504B-Nvidia`](https://huggingface.co/0xSero/GLM-5.2-504B-Nvidia) — a REAP-pruned NVFP4 cut of
`nvidia/GLM-5.2-NVFP4`. 317.9 GB, 64 shards, `GlmMoeDsaForCausalLM` (DeepSeek-V3.2-style MLA + DSA sparse indexer),
78 layers, 168 routed experts, 1M native `max_position_embeddings`.

Hardware: 4× NVIDIA RTX PRO 6000 Blackwell, 96 GB each (94.97 GiB usable), sm_120, PCIe, **no NVLink**.

| Metric | Measured |
|---|---|
| Context served | **250,000 tokens** |
| KV cache pool | **554,759 tokens** |
| Max concurrency @ 250K | **2.22×** |
| Decode throughput | **~43 tok/s** single-stream (incl. thinking tokens, via `usage.completion_tokens`) |
| Weights | **79.1 GB / GPU** |
| Resident VRAM | ~92.5 GB / GPU |
| Needle-in-haystack | **PASS** at 11,058 / 44,058 / 88,058 / 107,309 / 121,058 / 178,809 / 233,809 input tokens |
| Cold start (first run) | 340 s engine init (FlashInfer JIT) · warm: 45–105 s |

### What the DCP patch buys

The KV pool is a **fixed byte budget** — whatever VRAM is left after weights. It does not grow when you raise
`--max-model-len`. Context and concurrency trade directly against each other. Decode context parallelism (DCP) shards
that budget across 4 ranks, so each GiB of KV holds 4× the tokens.

| | Without DCP | With `--decode-context-parallel-size 4` |
|---|---|---|
| KV pool | 164,160 tokens | **554,759 tokens** (**3.37×**) |
| KV density | 19,898 tok/GiB | **79,595 tok/GiB** (exactly **4.00×**) |
| KV memory available | 8.25 GiB | 6.96 GiB (DCP workspaces cost 1.29 GiB) |
| Concurrency @ 131,072 ctx | 1.25× | **4.23×** |
| Concurrency @ 250,000 ctx | **0.66× — could not fit one request** | **2.22×** |
| Decode | 52 tok/s | 43 tok/s (**−18%**) |

Net gain is 3.37× rather than the raw 4.00× because the DCP workspaces eat 8.25 → 6.96 GiB of the KV budget.

**The −18% decode is the cost, and it is real.** DCP adds a q all-gather plus an LSE reduce per layer per decode step
= **156 collectives per token** across 78 layers, and with `NCCL_P2P_DISABLE=1` (required — Blackwell PCIe P2P
allreduce deadlocks) these are host-staged through CPU bounce buffers. It is latency-bound, not bandwidth-bound
(~430 MB/s at 52 tok/s is trivial next to PCIe Gen5). If decode latency matters more than context to you, either skip
the patch or try `--dcp-comm-backend a2a`.

---

## Is this you?

If you pasted one of these into a search engine and landed here — yes. See **[docs/BUGS.md](docs/BUGS.md)** for the
full trace, cause, and fix of each.

**1. OOM during weight load, ~93.64 GiB/GPU**, with the log line
`[EP Rank 0/4] ... Local/global number of experts: 64/256` (it should say **42/168**), ending in
`torch.OutOfMemoryError` at `vllm/model_executor/layers/quantization/modelopt.py:1458` in `create_weights`.
→ Upstream `transformers` bug. The checkpoint ships both the true `n_routed_experts: 168` and a stale
`num_experts: 256` from the unpruned base, and
`src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py:161-163` unconditionally re-routes the
**deprecated** key over the **explicit** one. vLLM is innocent. **Fix: delete `num_experts` from the checkpoint's
`config.json`** (`scripts/fix-config.py`). The `--hf-overrides` workaround also works but silently forecloses MTP.

**2.** `TypeError: trtllm_batch_decode_with_kv_cache_mla() got an unexpected keyword argument 'kv_scale_format'`
→ vLLM 0.25.1 pins `flashinfer-python==0.6.13` in metadata but its own sm120 sparse-MLA call site passes
`kv_scale_format`, which only exists in 0.6.14. **Fix: install `flashinfer-python==0.6.14` and set
`FLASHINFER_DISABLE_VERSION_CHECK=1`.**

**3.** `AssertionError: Decode Context Parallelism (DCP) requires attention implementations to return the softmax LSE
during decode, but FlashInferMLASparseSM120Impl does not. Try a different backend by setting --attention-backend or
disable DCP.`
→ The suggestion is impossible: `FLASHINFER_MLA_SPARSE_SM120` is the **only** backend vLLM offers on sm_120
(`out of potential backends: [FLASHINFER_MLA_SPARSE_SM120]`). The file simply never declares
`can_return_lse_for_decode` (`flashinfer_mla_sparse_sm120.py:32`), so it inherits `False` from `backend.py:784` —
even though its sibling `FlashInferMLASparseImpl` already implements the entire DCP+LSE path
(`flashinfer_mla_sparse.py:373-374, :450-457, :508, :527`) and the FlashInfer 0.6.14 kernel already exposes
`lse`/`return_lse`. **Fix: `patches/flashinfer_mla_sparse_sm120-dcp.patch`.**

Also worth knowing before you file a bug: **a 5-minute silent cold start is not a hang.** First launch spends ~340 s
in FlashInfer JIT with GPUs at 0% / 15 W while the log repeats
`shm_broadcast: No available shared memory broadcast block found in 60 seconds`. That is nvcc compiling. Wait.

---

## Quickstart

Assumes a working NVIDIA driver, CUDA runtime, and the checkpoint already downloaded. Set these first:

```bash
export MODEL=/path/to/GLM-5.2-504B-Nvidia     # the checkpoint directory
export VENV=$HOME/venvs/glm52                 # a dedicated venv — do not reuse an existing one
export HOST=127.0.0.1
export PORT=8000
```

### 1. Create the venv (stock vLLM 0.25.1 + torch 2.11 + flashinfer 0.6.14)

```bash
bash scripts/setup-venv.sh "$VENV"
```

This installs stock `vllm==0.25.1` and then bumps `flashinfer-python` to `0.6.14` (see wall #2). **pip will print an
"incompatible" dependency warning about the 0.6.13 pin — it is safe to ignore for this path**; the pin is metadata
only and is not enforced at runtime. Note that `flashinfer-cubin` has **no 0.6.14 release on PyPI** (latest is
0.6.13), which is why `FLASHINFER_DISABLE_VERSION_CHECK=1` is mandatory — without it the version guard in
`flashinfer/jit/env.py` aborts at startup.

### 2. Fix the checkpoint config (wall #1)

```bash
python3 scripts/fix-config.py "$MODEL"
```

Deletes the stale `num_experts: 256` key (backing up `config.json` first). Verify:

```bash
python3 -c "from transformers import AutoConfig; import os; \
print(AutoConfig.from_pretrained(os.environ['MODEL'], trust_remote_code=True).n_routed_experts)"
# must print 168 — if it prints 256, the key is still there
```

For the record, this is why the obvious workarounds don't work:

```text
AutoConfig.from_pretrained(M)                       -> n_routed_experts=256   # JSON says 168!
AutoConfig.from_pretrained(M, num_experts=168)      -> n_routed_experts=256   # kwarg does NOT help
AutoConfig.from_pretrained(M, n_routed_experts=168) -> n_routed_experts=168
*** deleting num_experts from config.json           -> n_routed_experts=168   # the real fix
```

The `num_experts=168` kwarg fails because the clobber runs during `__post_init__`, before `from_pretrained` applies
kwargs.

### 3. Apply the DCP/LSE patch (wall #3)

```bash
SP=$("$VENV/bin/python" -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
python3 patches/apply_dcp_patch.py \
  "$SP/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
# -> PATCHED OK
```

The applier is anchor-based and verifies every edit landed; it refuses to run twice and exits non-zero rather than
half-patching. `patches/flashinfer_mla_sparse_sm120-dcp.patch` is the equivalent 126-line unified diff if you prefer
`git apply`. Re-run this after any `pip install -U vllm`.

### 4. Serve

```bash
bash scripts/serve-glm52.sh
```

Which is, in full:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NCCL_P2P_DISABLE=1          # Blackwell PCIe P2P allreduce deadlocks without this

"$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name glm-5.2 \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --decode-context-parallel-size 4 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8 \
  --max-model-len 250000 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --trust-remote-code
```

**Do not pass `--cp-kv-cache-interleave-size`** — it must stay at 1 or vLLM raises at `indexer.py:261-263`.

If it worked, the log auto-selects all the right sm_120 backends with no further flags:
`FLASHINFER_MLA_SPARSE_SM120` attention · `fp8_ds_mla` KV format · `FLASH_ATTN` MLA prefill ·
`FLASHINFER_CUTLASS` NvFp4 MoE (**not** Marlin) · vendored `vllm.third_party.deep_gemm`.

### Gotchas that will waste your afternoon

- **`max_tokens` must be ≥ 1024.** GLM-5.2 always emits a thinking block first, so a small budget returns a
  **silently empty response with no error**: `max_tokens=64` → `stop_reason=max_tokens`, blocks `[(thinking,193)]`,
  text `""`. `max_tokens=1024` → `stop_reason=end_turn`, `[(thinking,353),(text,6)]` → `"Tokyo."`
- **Parser names:** vLLM 0.25.1 registers `glm45` and `glm47` as aliases of the same `glm47_moe` class, for both tool
  and reasoning parsers. `glm47_moe` is the *module* name and is **not** a valid CLI value — it will fail.
- **Benchmark harnesses that count only `delta.content` produce garbage** on this model, because with
  `--reasoning-parser` the output streams into `delta.reasoning_content` first. Symptom: `gen_tokens ~1` and absurd
  rates (we measured "838,860 tok/s"). Count `usage.completion_tokens` (request `stream_options.include_usage`) or
  sum both fields. Tolerate `content=None`.
- **Rapid stop→start cycling fragments the KV pool.** A hurried relaunch gave 6.05 GiB / 481,576 KV (1.93×); a clean
  stop with a ~12 s settle and no residual compute apps restored 6.97 GiB / 554,759 KV (2.22×). If KV looks ~13% low,
  stop cleanly and relaunch.
- **Don't run 8-concurrent throughput benchmarks at low KV headroom.** Pre-DCP (1.25×) that thrashed all 4 GPUs to
  100% / 240–390 W to produce ~2 tok/s aggregate, and pushed VRAM to 95,619 of 97,887 MiB.

---

## MTP does not fit. Don't spend a day on it.

`--speculative-config {"method":"mtp","num_speculative_tokens":3}` with DCP4 @ 250K: the drafter consumes ~6.96 of the
6.97 GiB KV budget →

```text
Available KV cache memory: 0.01 GiB
ValueError: To serve at least one request with the model max seq len (250000), 3.19 GiB KV cache is needed,
which is larger than the available KV cache memory (0.01 GiB). Based on the available memory, the estimated
maximum model length is 256.
```

**MTP and DCP4/250K are mutually exclusive on 4×96 GB.** MTP buys ~+15% decode and costs all 250,000 tokens of
context. It also disables `min_p` (`WARNING __init__.py:204 min_p and logit_bias parameters will not work with
speculative decoding`) — and the 0xSero model card asks for `min_p=0.05` + `repetition_penalty=1.05` precisely because
this REAP cut is **not** Router-KD recovered. And `num_speculative_tokens > 1` reuses the single MTP layer
(`num_nextn_predict_layers: 1`) with, per vLLM's own warning, "lower acceptance rate".

There is a second reason to fix `config.json` rather than use `--hf-overrides`: MTP with the override fails at
`self.drafter.load_model(self.model)` with
`AssertionError: Attempted to load weight (torch.Size([168])) into parameter (torch.Size([256]))`, because vLLM
**deliberately** discards dict-form `hf_overrides` for the draft model. Verbatim from
`vllm/config/speculative.py:602`:

> "Callable overrides on the target are config-to-config transforms ... and must also reach the draft config ...
> Dict overrides are target-specific key patches and are not applied to the draft."

The CLI can only ever produce a dict. The workaround structurally forecloses MTP; fixing the checkpoint doesn't.

---

## Anthropic / Claude Code compatibility

vLLM 0.25.1 serves the Anthropic Messages API **natively** at `/v1/messages` — no translation proxy is needed for raw
calls. Probed live; full detail in **[docs/CLAUDE-CODE.md](docs/CLAUDE-CODE.md)**.

**Works:** `POST /v1/messages` (`type=message`, `role=assistant`, top-level `system` honoured, `usage` populated) ·
streaming SSE with all 6 events · `tool_use` (`stop_reason=tool_use`, content `[thinking, text, tool_use]`, correct
args) · native Anthropic thinking blocks · `/v1/messages/count_tokens` · OpenAI-style `tool_calls`.

**Does not work:** tool search (`tool_search_tool_regex_20251119` + `defer_loading`) → **HTTP 400 "missing
input_schema"**, so Claude Code's `ENABLE_TOOL_SEARCH` is a no-op against vLLM · memory tool (`memory_20250818`) →
HTTP 400, same cause · context editing (`context_management.edits` / `clear_tool_uses_20250919`) → returns 200 but
echoes nothing, treat as a no-op · `cache_control` blocks → accepted but **silently ignored**; `usage` returns only
`[input_tokens, output_tokens]` with no `cache_creation_input_tokens` / `cache_read_input_tokens`, so Anthropic
prompt-caching economics don't apply and hits can't be measured. vLLM's automatic prefix caching
(`enable_prefix_caching=True` by default) gives you the compute benefit for free instead — and DCP4 tripling the KV
pool gives it far more room to retain history.

**Claude Code (2.1.207) cannot talk to vLLM directly.** It sends two things vLLM rejects: (a) a `role:"system"`
message *inside* `messages[]` → 400 or hang; (b) `output_config:{"effort":"xhigh"}` → 500
(`reasoning_effort error : xhigh, should be no_think/low/high`) → Claude Code retries ~8× → looks like a hang. The
pattern that fixes it is a small request-normalizing reverse proxy in front of vLLM that (1) moves any
non-`user`/`assistant` message into the top-level `system` field and (2) clamps `reasoning_effort` /
`output_config.effort` to a value vLLM accepts. Stream responses back as HTTP/1.1 chunked — a plain connection close
makes Node/undici retry.

---

## What this is NOT

- **Not a fork.** Stock `pip install vllm==0.25.1`, plus one patched file you can re-apply in 2 seconds after an
  upgrade.
- **Not a new quantization.** The NVFP4 checkpoint is 0xSero's work, not ours.
- **Not a model release.** We host no weights.
- **Not a novel design.** The DCP patch is an independent reimplementation of
  [vLLM PR #47779](https://github.com/vllm-project/vllm/pull/47779) (open and unreviewed as of 2026-07-15). Credit for
  the approach belongs to that PR's author. What this repo adds is validation on sm_120 hardware and the surrounding
  documentation — **not an improvement on the upstream fix.**

### The complementary path: 0xSero/glm-5.2-sm120

[**0xSero/glm-5.2-sm120**](https://github.com/0xSero/glm-5.2-sm120) is a real, validated fork/Docker recipe for the
same model on the same hardware, and it goes further than this repo does: it reports **~710K KV and working MTP=3**
via the `B12X_MLA_SPARSE` backend. If you want maximum KV and speculative decoding, that is the path to take — this
repo cannot match those numbers.

The tradeoff is build cost: a 5-repo source build or a Docker image. Choose by what you're optimizing:

| | This repo | 0xSero/glm-5.2-sm120 |
|---|---|---|
| Install | `pip install vllm==0.25.1` + 1 patched file | 5-repo source build / Docker image |
| KV pool | 554,759 tokens | ~710K reported |
| MTP | does not fit | works (=3) |
| Attention backend | stock `FLASHINFER_MLA_SPARSE_SM120` | `B12X_MLA_SPARSE` (fork-only) |

Note that `pip install b12x` alone does **not** get you `--attention-backend B12X_MLA_SPARSE` on stock vLLM: there is
no such backend registered (`import b12x` → `ModuleNotFoundError`; no `b12x` anywhere in `vllm/attention`). That
backend exists only in the fork.

---

## Credits

- **[vLLM PR #47779](https://github.com/vllm-project/vllm/pull/47779)** — the DCP/LSE design our patch reimplements
  independently. Ours is not an improvement on it; the design credit is theirs.
- **DeepGEMM PR #324** (NVIDIA DevTech APAC) — added `sm120_fp4_mqa_logits`, which unblocked GLM-5.2's DSA indexer on
  sm_120.
- **FlashInfer PR #3395** — the SM120 sparse MLA kernels.
- **[0xSero](https://github.com/0xSero)** — the REAP-pruned NVFP4 checkpoint and the `glm-5.2-sm120` fork/Docker
  recipe.
- **lukealonso/b12x** — SM120 NVFP4 kernels.

See [NOTICE](NOTICE). Licensed Apache-2.0 ([LICENSE](LICENSE)).

---

## Docs

| | |
|---|---|
| **[docs/BUGS.md](docs/BUGS.md)** | The three walls in full: traces, `file:line` root causes, fixes. |
| **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** | KV/concurrency/decode numbers, needle-in-haystack, method. |
| **[docs/CLAUDE-CODE.md](docs/CLAUDE-CODE.md)** | Anthropic Messages API probe results and the proxy pattern. |
| **[docs/DEAD-ENDS.md](docs/DEAD-ENDS.md)** | W4A16, SGLang, KV shrinking, `--max-model-len`, b12x-on-stock. |
</content>
