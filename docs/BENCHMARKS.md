# Benchmarks

All numbers measured 2026-07-15 on the configuration described below. Nothing
here is extrapolated or estimated unless the text says so explicitly. The
[What was NOT measured](#what-was-not-measured) section at the bottom is part of
the result — read it before citing anything here.

## Headline

| Metric | Value |
|--------|-------|
| Context | 250,000 tokens |
| KV cache pool | **554,759 tokens** |
| Max concurrency @ 250K | **2.22x** |
| Weights | 79.1 GB/GPU |
| Resident VRAM | ~92.5 GB/GPU |
| Decode, single stream | **~43 tok/s** (includes thinking tokens) |
| Needle-in-haystack | **PASS** at every depth tested, up to 233,809 input tokens |

The DCP patch is worth **3.37x the KV pool**. Without it, this model cannot fit
even one full-context request.

---

## System under test

* **GPUs:** 4x NVIDIA RTX PRO 6000 Blackwell, 96 GB each (94.97 GiB usable),
  sm_120, PCIe, **no NVLink**
* **Stack:** stock vLLM 0.25.1, torch 2.11, flashinfer-python 0.6.14, in a
  dedicated venv. No fork, no Docker, no CUDA 13.2, no torch 2.12, no b12x.
* **Model:** `0xSero/GLM-5.2-504B-Nvidia` — REAP-pruned NVFP4 cut of
  `nvidia/GLM-5.2-NVFP4`. 317.9 GB, 64 shards. `GlmMoeDsaForCausalLM` /
  `glm_moe_dsa` (DeepSeek-V3.2-style MLA + DSA sparse attention indexer).
* **Model config:** `num_hidden_layers=78`, `n_routed_experts=168`,
  `n_shared_experts=1`, `num_experts_per_tok=8`, `moe_intermediate_size=2048`,
  `hidden_size=6144`, `index_topk=2048`, `kv_lora_rank=512`,
  `qk_rope_head_dim=64`, `num_nextn_predict_layers=1`,
  `max_position_embeddings=1048576` (1M native).
* **Patches applied:** `scripts/fix-config.py` (removes stale `num_experts`),
  `patches/flashinfer_mla_sparse_sm120-dcp.patch`.

### Serve configuration

```
--tensor-parallel-size 4 --enable-expert-parallel --decode-context-parallel-size 4
--quantization modelopt_fp4 --kv-cache-dtype fp8 --max-model-len 250000
--gpu-memory-utilization 0.92
--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45
--trust-remote-code
```

with `FLASHINFER_DISABLE_VERSION_CHECK=1` and `NCCL_P2P_DISABLE=1`.

`NCCL_P2P_DISABLE=1` is not optional: Blackwell PCIe P2P allreduce deadlocks
without it. It also directly shapes the DCP decode cost measured below.

Do **not** pass `--cp-kv-cache-interleave-size`; it must stay at 1 or vLLM
raises at `indexer.py:261-263`.

### Auto-selected backends (all correct on sm_120, no override needed)

* attention: `FLASHINFER_MLA_SPARSE_SM120`
* KV format: `fp8_ds_mla`
* MLA prefill: `FLASH_ATTN`
* MoE: `FLASHINFER_CUTLASS` NvFp4 (**not** Marlin)
* `vllm.third_party.deep_gemm` (vendored)

### Startup

* **Cold: 340 s engine init.** flashinfer JIT-compiles from scratch — roughly
  5 minutes of silent `nvcc` with GPUs at 0% / 15 W, while the log repeats
  `shm_broadcast: No available shared memory broadcast block found in 60 seconds`.
  **That is compilation, not a hang.** Do not kill it.
* **Warm: ~45-105 s.**

---

## 1. KV cache and concurrency — with and without DCP

The single most important measurement in this repo.

| Config | Context | KV pool | Max concurrency |
|--------|---------|---------|-----------------|
| **No DCP** | 131,072 | 164,160 tok | **1.25x** |
| **No DCP** | 250,000 | 164,160 tok | **0.66x** — cannot fit one request |
| **DCP4** | 131,072 | 554,759 tok | **4.23x** |
| **DCP4** | 250,000 | 554,759 tok | **2.22x** |

Note the KV pool column: **164,160 in both no-DCP rows, 554,759 in both DCP4
rows.** That is the point of the next section.

### The KV pool is a fixed byte budget

The KV pool is whatever VRAM is left after weights, and it is **identical
regardless of `--max-model-len`**. Raising the context does not give you more KV
cache; it just divides the same pool into fewer, longer slots. Context and
concurrency trade directly against each other:

```
max concurrency = KV pool tokens / max-model-len
```

Check it: 164,160 / 131,072 = 1.25. 164,160 / 250,000 = 0.66. 554,759 / 250,000
= 2.22. 554,759 / 131,072 = 4.23. Every row above is that one division.

At 0.66x, vLLM cannot fit a single full-context request. **Without the DCP
patch, 250K context on this hardware is not merely slow — it is unavailable.**

At 131,072 with DCP4, vLLM prints verbatim:

```
Maximum concurrency for 131,072 tokens per request: 4.23x
```

(was `1.25x`).

### The 4.00x / 3.37x math

DCP4 shards the KV cache across 4 ranks, so **per-GiB capacity is exactly 4.00x**:

| | tokens per GiB |
|--|--|
| No DCP | 19,898 |
| DCP4 | **79,595** |

79,595 / 19,898 = **4.00x**. Exactly, as designed.

But the **net** gain is only **3.37x**, because DCP's workspaces eat into the
memory available for KV:

| | KV memory available | tok/GiB | KV pool |
|--|--|--|--|
| No DCP | 8.25 GiB | 19,898 | 164,160 |
| DCP4 | **6.96 GiB** | 79,595 | **554,759** |

554,759 / 164,160 = **3.37x**.

So: DCP quadruples the *density* and gives back ~1.29 GiB of the *budget* to its
own workspaces. You keep 3.37 of the theoretical 4.00. That is the honest number
and it is the one to quote.

---

## 2. Decode throughput — what DCP costs

| Config | Decode, single stream |
|--------|----------------------|
| No DCP | **52 tok/s** |
| DCP4 | **43 tok/s** |

Roughly **18% slower**. This is the price of the 3.37x KV pool, and for this
workload it is clearly worth paying — 18% decode against the difference between
"250K context works" and "250K context does not exist".

### Why: 156 collectives per token

DCP adds, **per layer per decode step**:

1. a **q all-gather** across the DCP group, and
2. an **LSE reduce** to merge each rank's partial softmax.

With `num_hidden_layers=78`, that is **2 x 78 = 156 collectives per decoded
token**. And because `NCCL_P2P_DISABLE=1` is mandatory on this hardware, those
collectives are **host-staged through CPU bounce buffers** rather than going
GPU-to-GPU.

**This is latency-bound, not bandwidth-bound.** At 52 tok/s the traffic is
~430 MB/s — trivial against PCIe Gen5. The cost is 156 round trips per token,
not the bytes. That is why the penalty is a flat ~18% rather than scaling with
sequence length.

**Escape hatch:** if decode latency matters more to you than context,
`--dcp-comm-backend a2a` changes the collective pattern. Not benchmarked here.

---

## 3. Needle-in-haystack recall

**Result: PASS at every depth tested.** The 250K context is real, not nominal.

| Input tokens | Result |
|-------------:|--------|
| 11,058 | PASS |
| 44,058 | PASS |
| 88,058 | PASS |
| 107,309 | PASS |
| 121,058 | PASS |
| 178,809 | PASS |
| 233,809 | PASS |

These are **real measured `input_tokens` values** from the API response, not
target sizes.

This is a **retrieval** check — it establishes that attention reaches the far
end of a 233K-token context and that the DCP patch (in particular
`lse_base_on_e = False`) is not silently corrupting the cross-rank softmax. A
wrong LSE base would show up here as degraded recall at depth. It is **not** a
general quality evaluation. See [What was NOT measured](#what-was-not-measured).

---

## 4. MTP rejection

`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` with DCP4 at
250K:

```
Available KV cache memory: 0.01 GiB
ValueError: To serve at least one request with the model max seq len (250000), 3.19 GiB KV cache is needed, which is larger than the available KV cache memory (0.01 GiB). Based on the available memory, the estimated maximum model length is 256.
```

The drafter consumes **~6.96 of the ~6.97 GiB** KV budget, leaving 0.01 GiB.
The estimated max model length collapses from 250,000 to **256**.

**MTP and DCP4/250K are mutually exclusive on 4x96 GB.** MTP buys ~+15% decode
and costs all 250,000 tokens of context. See `docs/BUGS.md` for the two further
traps (dict `hf_overrides` never reaching the draft model; MTP disabling
`min_p`).

---

## Methodology, and how to falsify this

### Measuring decode throughput correctly

**Any harness that counts only `delta.content` produces garbage for this model.**

With `--reasoning-parser` active, GLM-5.2's output streams into
`delta.reasoning_content` first — and this model *always* thinks first. A
content-only counter sees almost nothing.

**Symptom: `gen_tokens ~1` and an absurd rate. We measured 838,860 tok/s.** If
your benchmark reports a number like that, it is not a fast model, it is a
broken harness.

Do one of:

* request `stream_options.include_usage` and count **`usage.completion_tokens`**
  (what we did — this is why the 43 tok/s figure *includes* thinking tokens), or
* sum **both** `content` and `reasoning_content`.

Also tolerate `content=None`.

The 52 and 43 tok/s figures are single-stream, `usage.completion_tokens`-based,
and include thinking tokens. They are therefore a **conservative** measure of
useful output rate and a **fair** measure of total generation rate.

### Reading KV pool and concurrency

Both are printed by vLLM at startup — `Available KV cache memory: N GiB`, the
GPU KV cache size in tokens, and `Maximum concurrency for N tokens per request:
Nx`. Every table in sections 1 and 4 is read off those lines. No custom
instrumentation. To reproduce, launch with and without
`--decode-context-parallel-size 4` and read the log.

### Do NOT run 8-concurrent throughput benchmarks at low KV headroom

Pre-DCP (1.25x concurrency), an 8-concurrent run **thrashed all 4 GPUs to 100%
and 240-390 W while producing only ~2 tok/s aggregate**, and pushed VRAM to
**95,619 of 97,887 MiB**.

That is not a throughput measurement, it is a preemption storm — vLLM is
evicting and recomputing rather than generating. The number it produces is
meaningless and the thermal/power cost is real. Only run concurrency benchmarks
with headroom above the concurrency you are testing.

### Clean-restart discipline (affects your numbers by ~13%)

**Rapid stop -> start cycling fragments the KV pool.**

| Restart style | KV memory | KV pool | Concurrency @250K |
|--|--|--|--|
| Hurried relaunch | 6.05 GiB | 481,576 | 1.93x |
| Clean stop, ~12 s settle, no residual compute apps | **6.97 GiB** | **554,759** | **2.22x** |

That is a ~13% swing from nothing but restart hygiene. **If your KV pool looks
~13% low, you did not find a regression — stop cleanly and relaunch.** All
headline numbers on this page are from clean starts.

---

## What was NOT measured

Stated explicitly so nothing here is over-read:

* **No quality evaluation.** There is no benchmark suite run, and nothing here
  separates this checkpoint's output quality from any other model. The
  needle-in-haystack results are **retrieval only** — they show attention
  reaches 233K tokens and that the LSE base is correct. They say nothing about
  reasoning, coding, or knowledge quality.
* **The REAP cut is not quality-assessed.** This is a pruned checkpoint and, per
  its model card, **not Router-KD recovered**. The card asks for `min_p=0.05` and
  `repetition_penalty=1.05` as guardrails. **The guardrailed re-run was not
  done.** The measurements on this page were taken without characterising what
  those settings change.
* **`--dcp-comm-backend a2a` was not benchmarked.** It is named as an escape
  hatch on the strength of the latency-bound analysis, not on data.
* **No multi-user / sustained-load throughput number.** The only concurrency
  figure quoted is vLLM's own capacity calculation, not a measured aggregate
  tok/s under load. The one 8-concurrent run attempted was at insufficient
  headroom and is reported above as a warning, not a result.
* **MTP's ~+15% decode claim is not ours** — it never ran here, because it never
  fit. Treat it as the reason someone would try, not as a measurement.
* **No power/efficiency measurements** beyond the incidental 240-390 W observed
  during the thrash run.
