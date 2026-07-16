# Bugs and blockers

Everything on this page was hit first-hand while bringing up
`0xSero/GLM-5.2-504B-Nvidia` (REAP-pruned NVFP4, 317.9 GB, 64 shards,
`GlmMoeDsaForCausalLM` / `glm_moe_dsa`) on **4x RTX PRO 6000 Blackwell 96 GB
(sm_120, PCIe, no NVLink)** with **stock vLLM 0.25.1 + torch 2.11 +
flashinfer-python 0.6.14**. No fork, no Docker.

Error strings are quoted exactly so they are searchable.

## Summary

| # | Blocker | Class | Fixed by |
|---|---------|-------|----------|
| 1 | `torch.OutOfMemoryError` at weight load, 93.64 GiB/GPU | **Upstream bug** in `transformers`, triggered by **checkpoint data** | Delete stale `num_experts` from `config.json` (`scripts/fix-config.py`) |
| 2 | `TypeError: ... unexpected keyword argument 'kv_scale_format'` | **Upstream packaging bug** in vLLM (pin vs call site) | `flashinfer-python==0.6.14` + `FLASHINFER_DISABLE_VERSION_CHECK=1` |
| 3 | `AssertionError: Decode Context Parallelism (DCP) requires attention implementations to return the softmax LSE` | **Upstream gap** in vLLM (missing capability flag) | `patches/flashinfer_mla_sparse_sm120-dcp.patch` |
| 4 | MTP: `Available KV cache memory: 0.01 GiB`, and `AssertionError: Attempted to load weight (torch.Size([168])) into parameter (torch.Size([256]))` | **Physics** (does not fit) + **intended-but-surprising** vLLM behaviour | Do not use MTP on 4x96 GB. See [MTP](#4-the-mtp-trap) |

---

## 1. OOM at weight load — the `n_routed_experts` clobber

**Class: upstream `transformers` bug, triggered by stale checkpoint data. vLLM is innocent.**

### Symptom

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate ... (GPU 0; 93.64 GiB ...)
```

raised inside:

```
vllm/model_executor/layers/quantization/modelopt.py:1458   create_weights
```

The tell is earlier in the log, before the OOM:

```
[EP Rank 0/4] ... Local/global number of experts: 64/256
```

**Correct is `42/168`.** If you see `64/256`, you have this bug. The model is
trying to allocate ~111 GB of expert weights per GPU instead of ~80 GB.

### Root cause

GLM-5.2 REAP checkpoints ship **both** keys in `config.json`:

* `n_routed_experts: 168` — the true, post-pruning count
* `num_experts: 256` — a stale leftover from the unpruned base model

`transformers` then does an **unconditional** backward-compat re-route in
`src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py`,
`__post_init__`, lines 161-163:

```python
# BC: re-route `num_experts` to `n_routed_experts`
if (num_experts := kwargs.get("num_experts")) is not None:
    self.n_routed_experts = num_experts
```

There is no guard. The **deprecated** key silently overwrites the **explicit**
one. vLLM then builds `FusedMoE` for 256 experts, 168 of which do not exist in
the checkpoint.

vLLM reads the config correctly — `vllm/model_executor/models/deepseek_v2.py`
uses `config.n_routed_experts`. It is handed a lie.

The checkpoint really does contain 168 contiguous routed experts (indices
`0..167`); verified directly from `model.safetensors.index.json`.

### Measured evidence

`AutoConfig` behaviour, on the unmodified checkpoint:

```python
AutoConfig.from_pretrained(M)                       -> n_routed_experts = 256   # JSON says 168!
AutoConfig.from_pretrained(M, num_experts=168)      -> n_routed_experts = 256   # kwarg does NOT help
AutoConfig.from_pretrained(M, n_routed_experts=168) -> n_routed_experts = 168
# deleting `num_experts` from config.json           -> n_routed_experts = 168   <-- the real fix
```

**Why passing `num_experts=168` as a kwarg does not help:** the clobber runs
during `__post_init__`, i.e. *during construction*, before `from_pretrained`
applies your kwargs on top. You are overwriting a value that has already done
its damage — and then it is re-read from `kwargs` anyway.

### Fix (recommended)

Delete the stale `num_experts` key from the checkpoint's `config.json`:

```bash
python scripts/fix-config.py "$MODEL"
```

This fixes the problem at the source, for every consumer, including the MTP
draft model. Expected log after the fix:

```
[EP Rank 0/4] ... Local/global number of experts: 42/168
```

and ~79.1 GB of weights per GPU.

### Fix (workaround — has a hidden cost)

```
--hf-overrides '{"n_routed_experts":168}'
```

This works for plain serving. **It silently forecloses MTP** — see
[bug 4](#4-the-mtp-trap). Prefer fixing `config.json`.

### Upstream status

Open as of 2026-07-15. The `transformers` re-route is written as
unconditional; a correct version would only apply `num_experts` when
`n_routed_experts` is absent. Until that lands, treat the presence of both keys
in any REAP/pruned MoE checkpoint as a landmine.

---

## 2. `kv_scale_format` — vLLM's flashinfer pin does not match its own call site

**Class: upstream packaging bug in vLLM 0.25.1.**

### Symptom

```
TypeError: trtllm_batch_decode_with_kv_cache_mla() got an unexpected keyword argument 'kv_scale_format'
```

### Root cause

vLLM 0.25.1 pins `flashinfer-python==0.6.13` in its package metadata, but its
own sm_120 sparse-MLA call site passes `kv_scale_format=`, a parameter that
**only exists in flashinfer 0.6.14**. The pin is pip metadata only and is not
enforced at runtime, so you get a clean install that cannot serve.

Signatures:

* **0.6.13** — argument list ends at `[..., sparse_mla_top_k, out, bmm1_scale]`. No `kv_scale_format`.
* **0.6.14** — `[..., lse, return_lse, cute_dsl_impl, kv_scale_format, cum_seq_lens_q, max_q_len]`.

0.6.14 is also what makes [bug 3](#3-dcp-refuses-to-run-missing-can_return_lse_for_decode)
fixable, because it is the version that exposes `lse` / `return_lse`.

### Fix

```bash
pip install flashinfer-python==0.6.14
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

pip will print a dependency-conflict warning about the 0.6.13 pin. **It is safe
to ignore for this path** — the whole point is that the call site wants 0.6.14.

`FLASHINFER_DISABLE_VERSION_CHECK=1` is **mandatory**, not cosmetic:
`flashinfer-cubin` has **no 0.6.14 release on PyPI** (latest is 0.6.13), so the
version guard in `flashinfer/jit/env.py` will find a version mismatch between
`flashinfer-python` and `flashinfer-cubin` and abort at import.

### Upstream status

Open as of 2026-07-15. The fix is trivial (bump the pin), but it is blocked in
practice by `flashinfer-cubin` not having a matching 0.6.14 artifact.

---

## 3. DCP refuses to run — missing `can_return_lse_for_decode`

**Class: upstream gap in vLLM. This is the one that costs you 3.37x of KV cache.**

### Symptom

```
AssertionError: Decode Context Parallelism (DCP) requires attention implementations to return the softmax LSE during decode, but FlashInferMLASparseSM120Impl does not. Try a different backend by setting --attention-backend or disable DCP.
```

The suggestion is **impossible to follow**. On sm_120, vLLM logs:

```
out of potential backends: [FLASHINFER_MLA_SPARSE_SM120]
```

There is exactly one backend. "Try a different backend" is not an option, so
without the patch your only choice is to drop DCP — which costs you the KV pool
(see `docs/BENCHMARKS.md`: 554,759 -> 164,160 tokens).

### Root cause

`vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py:32` — the class
`FlashInferMLASparseSM120Impl` never declares `can_return_lse_for_decode`, so
it inherits `False` from `backend.py:784`. `need_to_return_lse_for_decode` is
derived from that flag at `backend.py:849-850`, and the DCP assertion fires.

This is a **stripped-down variant of an already-DCP-capable class**. Its sibling
`FlashInferMLASparseImpl` in `flashinfer_mla_sparse.py` already implements the
entire DCP + LSE path:

* `:373-374` — capability flags
* `:450-457` — the DCP index-filter branch
* `:508`, `:527` — LSE plumbing and `_normalize_lse`

And the flashinfer 0.6.14 kernel already exposes `lse` / `return_lse`. Nothing
was missing at the kernel level. The sm_120 file simply never wired it up.

### Fix

```bash
python patches/apply_dcp_patch.py
```

which applies `patches/flashinfer_mla_sparse_sm120-dcp.patch`. It adds:

* `can_return_lse_for_decode = True`
* `lse_base_on_e = False` — **load-bearing, see below**
* a DCP branch using `triton_filter_and_convert_dcp_index(..., return_valid_counts=True)`
* `seq_lens=seq_lens` (was `seq_lens=None`)
* `num_query_heads = q.shape[1]` for the output buffer. Under DCP the q heads
  are all-gathered across the DCP group before `forward_mqa`
  (`mla_attention.py:807`), 16 -> 64. Sizing the output from `self.num_heads`
  gives a shape error.
* `return_lse` plumbing, reusing `FlashInferMLASparseImpl._normalize_lse`
* **empty-row masking** (`out=0`, `lse=-inf`) so rows whose top-k set is empty
  on a given rank do not poison the cross-rank softmax denominator.

### Why `lse_base_on_e = False` — the silent-corruption trap

Get this wrong and the model **still runs and still produces fluent text**. It
is just wrong. There is no error. This is the single most dangerous line in the
patch.

The sm_120 kernel emits LSE in **log2 space**, not natural log. This is
provable from the CUDA sources shipped inside the flashinfer wheel:

* `decode_dsv3_2_kernel.cuh:396-399` — pre-scales logits by `LOG2E`
* `decode_dsv3_2_kernel.cuh:420-423` — accumulates with `exp2f`
* `decode_dsv3_2_kernel.cuh:652` — writes `log2f(global_sum[0]) + global_max[0]`
* `common/online_softmax.cuh:64-66` — comment, verbatim:
  *"Compute LSE (log-sum-exp) in log2 space for the merge kernel"*
* the GLM path reuses the decode-dsv4 merge kernel
  (`sparse_mla_sm120_decode_dsv3_2.cu:7,19,113-121`), which writes
  `out_lse = log2f(total_sum) + gmax` at `decode_dsv4_kernel.cuh:865,912`
  — **with no base conversion**

Meanwhile vLLM's DCP reducer branches between `tl.exp` and `tl.exp2` on
`IS_BASE_E` (`v1/attention/ops/common.py:59-64,90`), and `backend.py:795`
**defaults `lse_base_on_e` to `True`**. Omit the line and every cross-rank
softmax denominator is computed with the wrong base.

### Upstream status

**vLLM PR #47779** is the upstream fix for this. As of **2026-07-15** it is
**OPEN and unreviewed**.

The patch in this repo is an **independent reimplementation** of that PR's
approach, not a copy — credit for the design belongs to that PR's author (see
`NOTICE`). What this repo contributes is independent validation on sm_120
hardware. If #47779 merges, drop the patch and use stock vLLM.

---

## 4. The MTP trap

MTP (multi-token prediction / speculative decoding) looks attractive: ~+15%
decode. **It does not fit on 4x96 GB, and trying to make it fit via the
[bug 1](#1-oom-at-weight-load--the-n_routed_experts-clobber) workaround hits a
second wall.** Documented so you do not spend a day on it.

### 4a. It does not fit (physics, not a bug)

With `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` plus
DCP4 at 250K context:

```
Available KV cache memory: 0.01 GiB
ValueError: To serve at least one request with the model max seq len (250000), 3.19 GiB KV cache is needed, which is larger than the available KV cache memory (0.01 GiB). Based on the available memory, the estimated maximum model length is 256.
```

The drafter consumes ~6.96 of the ~6.97 GiB KV budget. **MTP and DCP4/250K are
mutually exclusive on this hardware.** MTP buys ~+15% decode and costs all
250,000 tokens of context. Not a trade worth making.

### 4b. `--hf-overrides` never reaches the draft model

**Class: intended-but-surprising vLLM behaviour. Not a bug — but it is a trap.**

If you used the `--hf-overrides` workaround from bug 1 instead of fixing
`config.json`, and then try MTP:

```
self.drafter.load_model(self.model)
AssertionError: Attempted to load weight (torch.Size([168])) into parameter (torch.Size([256]))
```

The target model got 168 experts; the draft model did not. vLLM **deliberately
discards dict-form `hf_overrides` for the draft model**.
`vllm/config/speculative.py:602`, `compose_draft_hf_overrides` docstring,
verbatim:

> "Callable overrides on the target are config-to-config transforms ... and must
> also reach the draft config ... Dict overrides are target-specific key patches
> and are not applied to the draft."

```python
if not callable(target_hf_overrides):
    return SpeculativeConfig.hf_config_override
```

Only a **callable** override reaches the draft. **The CLI can only ever produce
a dict.** So `--hf-overrides` structurally forecloses MTP — you cannot get there
from the command line at all.

This is the strongest argument for fixing `config.json` at the source: the
workaround is not merely inelegant, it removes a capability.

### 4c. MTP also costs you the model card's sampling guardrails

```
WARNING __init__.py:204  min_p and logit_bias parameters will not work with speculative decoding
```

The `0xSero` model card asks for `min_p=0.05` and `repetition_penalty=1.05`,
because this REAP cut is **not Router-KD recovered**. MTP disables `min_p`.
You are trading a quality guardrail the checkpoint author explicitly asked for
against ~15% decode.

Additionally, vLLM warns that `num_speculative_tokens > 1` reuses the single MTP
layer (the config has `num_nextn_predict_layers: 1`) with a "lower acceptance
rate", so the ~15% is optimistic.

---

## Classification recap

* **Upstream bugs** (should be fixed by the projects, not you):
  bug 1 (`transformers` unconditional re-route), bug 2 (vLLM's pin vs its own
  call site), bug 3 (vLLM's missing capability flag — PR #47779 pending).
* **Checkpoint-data bug** (the trigger for bug 1): GLM-5.2 REAP checkpoints
  shipping a stale `num_experts: 256` alongside the true `n_routed_experts: 168`.
* **Intended-but-surprising** (documented behaviour, still a trap):
  bug 4b, dict `hf_overrides` not reaching the draft model.
* **Physics** (no fix exists): bug 4a, MTP not fitting in the KV budget.
