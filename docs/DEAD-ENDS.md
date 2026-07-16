# Dead ends

Things that look like they should work, and do not. All first-hand on
**4x RTX PRO 6000 Blackwell 96 GB (sm_120, PCIe)** with stock vLLM 0.25.1, as of
**2026-07-15**.

Each entry: **what you would hope**, **what actually happens**, **the one-line
reason**. Read this before you spend a day on any of them.

## Index

| Dead end | One-line reason |
|----------|-----------------|
| [W4A16 checkpoint](#1-0xseroglm-52-504b-w4a16--does-not-load) | vLLM hardcodes the fused `indexer.wk` layer as unquantized; the int4 weights are unreachable by any flag |
| [SGLang](#2-sglang--a-regression-not-an-alternative) | sm_120 forces bf16 KV, which doubles bytes/token = strictly less context than vLLM |
| [Shrinking KV per token](#3-shrinking-kv-bytes-per-token--nothing-to-shrink) | `--kv-cache-dtype fp8` is already a no-op; the 656 B/token/layer page size is hardcoded |
| [Raising `--max-model-len`](#4-raising---max-model-len-without-dcp--no-extra-kv) | The KV pool is a fixed byte budget; context and concurrency trade against each other |
| [`pip install b12x`](#5-pip-install-b12x--does-not-register-a-backend) | Stock vLLM has no `B12X_MLA_SPARSE` backend registered; it exists only in a fork |
| [MTP](#6-mtp--does-not-fit) | The drafter eats ~6.96 of the 6.97 GiB KV budget |

---

## 1. `0xSero/GLM-5.2-504B-W4A16` — does not load

**What you would hope:** 260.7 GB instead of 317.9 GB. That is **57 GB smaller**,
which is ~14 GB/GPU back, which is more KV cache — potentially a lot more, since
the KV pool is only ~7-8 GiB/GPU to begin with. On paper this is the single
biggest available win. It is very tempting.

**What actually happens:** it does not load at all.

```
KeyError: ...wk_weights_proj.qweight
```

**The one-line reason:** `indexer.wk` ships int4-packed, but vLLM fuses
`wk` + `weights_proj` into one layer that is **hardcoded unquantized**.

### The trace

The checkpoint ships `indexer.wk` as int4-packed — `.qweight`, `.qzeros`,
`.scales`, and **zero `.weight`** tensors. vLLM fuses `wk` and `weights_proj`
into a single layer and constructs it with a **literal** `quant_config=None`:

* `vllm/model_executor/models/deepseek_v2.py:677-683` — the fused
  `wk_weights_proj` layer, passed `quant_config=None` as a literal
* contrast `:669-674` — `wq_b`, which **does** receive `quant_config`

So the layer is built to accept unquantized weights only. Then:

1. `.qweight` fails `_try_load_fp8_indexer_wk` (`:821-839`), which accepts only
   `.weight` in fp8 or bf16
2. it falls through to `stacked_params_mapping` (`:1521-1522`)
3. which looks up `params_dict[...wk_weights_proj.qweight]`
4. **`KeyError`** — that parameter does not exist, because the layer was built
   unquantized

**There is no flag for this.** It is a literal `None` in the model definition,
not a config path. The only fix is a code change upstream.

### Two more reasons not to chase it

* It ships **zero MTP layers**, so even if it loaded, the speculative-decoding
  path is gone.
* Its **model card claims NVFP4/B200 while its `config.json` says int4**. The
  metadata is not trustworthy. Read the config, not the card.

---

## 2. SGLang — a regression, not an alternative

**What you would hope:** a different serving engine sidesteps vLLM's sm_120 gaps
(the missing `can_return_lse_for_decode`, the flashinfer pin mismatch). Maybe
SGLang just works.

**What actually happens:** it runs, and gives you **strictly less context** than
the vLLM path.

**The one-line reason:** on sm_120 SGLang requires `--kv-cache-dtype bf16`, and
bf16 KV doubles the bytes per token.

fp8 KV **garbles output** on sm_120 under SGLang — tracked as
`sgl-project/sglang#29562`. So you are forced to bf16, which **doubles
per-token KV bytes** versus the vLLM `fp8_ds_mla` path. The KV pool is a fixed
byte budget (see [#4](#4-raising---max-model-len-without-dcp--no-extra-kv)), so
doubling bytes/token halves your tokens.

Since context is the entire problem being solved on this hardware, an engine
that halves KV density is a regression regardless of any other merit. Not
pursued further.

---

## 3. Shrinking KV bytes per token — nothing to shrink

**What you would hope:** the KV pool is a fixed byte budget, so if you cannot
make the budget bigger, make each token cheaper. `--kv-cache-dtype fp8` looks
like the lever. Failing that, an fp4 indexer cache.

**What actually happens:** every lever is already pulled, welded, or gated off.

**The one-line reason:** `fp8` is already the silent default, and the page size
is a hardcoded constant.

Three separate walls:

* **`--kv-cache-dtype fp8` is already a no-op.** `mla_attention.py:322-336`
  silently promotes `auto` | `fp8` | `fp8_e4m3` -> `fp8_ds_mla`. You get the
  same thing whether you pass the flag or not. And
  `flashinfer_mla_sparse_sm120.py:65-67` **raises** on anything else — so there
  is exactly one legal value and you are already using it. (Pass it anyway for
  documentation value; it costs nothing.)
* **The page size is hardcoded.** **656 bytes/token/layer**, a literal at
  `kv_cache_interface.py:388`. Not a config, not a flag. A constant.
* **The fp4 indexer cache is SM100-gated.** Not available on sm_120.

There is no tuning knob here. The only way to get more KV tokens on this
hardware is **DCP**, which is why the patch in this repo exists.

---

## 4. Raising `--max-model-len` without DCP — no extra KV

**What you would hope:** ask for 250K context and get 250K context.

**What actually happens:** vLLM accepts the flag and reports
**0.66x max concurrency** — it cannot fit even one full-context request.

**The one-line reason:** the KV pool is a **fixed byte budget** (leftover VRAM
after weights) and is **identical regardless of `--max-model-len`**.

Measured, without DCP: **KV pool = 164,160 tokens at every context setting.**

```
max concurrency = KV pool / max-model-len
164,160 / 131,072 = 1.25x     <- fits
164,160 / 250,000 = 0.66x     <- fits nothing
```

`--max-model-len` does not allocate memory. It divides an already-fixed pool
into fewer, longer slots. **Context and concurrency trade directly against each
other**, and below 1.0x you have not made a slow server — you have made a server
that cannot serve the request at all.

The fix is to make the pool bigger, which means DCP (554,759 tokens = 2.22x at
250K). See `docs/BENCHMARKS.md` and `docs/BUGS.md` bug 3.

---

## 5. `pip install b12x` — does not register a backend

**What you would hope:** `b12x` provides SM120 NVFP4 kernels, and the
`B12X_MLA_SPARSE` backend sounds like exactly the sm_120 sparse-MLA path we
need. So `pip install b12x` + `--attention-backend B12X_MLA_SPARSE` should be a
one-line alternative to patching vLLM.

**What actually happens:** nothing. The backend does not exist in stock vLLM.

**The one-line reason:** `B12X_MLA_SPARSE` is registered only in a **fork**, not
in stock vLLM — the pip package alone cannot add a backend.

Verified two ways:

* `import b12x` -> `ModuleNotFoundError`
* there is **no reference to `b12x` anywhere in `vllm/attention`**

The backend lives only in the `local-inference-lab/vllm` fork.

### This one has a real destination

Unlike the other entries on this page, **this is a dead end for *this* repo's
approach, not a dead end in general.** The fork path is real and validated — see
**`0xSero/glm-5.2-sm120`**, which documents it and reports **~710K KV and MTP=3**
(both better than this repo's 554,759 KV and no-MTP).

The cost is a **5-repo source build / Docker image**. That is the entire trade:

| | This repo (stock vLLM) | The fork path |
|--|--|--|
| KV | 554,759 | ~710K |
| MTP | does not fit | =3 |
| Build | `pip install` + 2 small patches | 5-repo source build / Docker |

If you want the maximum numbers, go use the fork; it is a legitimate, working
answer and this repo says so plainly (see `NOTICE`). This repo exists for people
who want stock vLLM, no fork and no Docker, and are willing to give up ~22% of
the KV pool and MTP for that. **What does not exist is the middle option** —
`pip install b12x` is not a shortcut to the fork's results.

---

## 6. MTP — does not fit

**What you would hope:** `num_speculative_tokens=3` for ~+15% decode. The
checkpoint has `num_nextn_predict_layers: 1`, so the MTP layer is right there.

**What actually happens:**

```
Available KV cache memory: 0.01 GiB
ValueError: To serve at least one request with the model max seq len (250000), 3.19 GiB KV cache is needed, which is larger than the available KV cache memory (0.01 GiB). Based on the available memory, the estimated maximum model length is 256.
```

**The one-line reason:** the drafter consumes ~6.96 of the ~6.97 GiB KV budget.

**MTP and DCP4/250K are mutually exclusive on 4x96 GB.** MTP buys ~+15% decode
and costs all 250,000 tokens of context — max model length collapses to 256.

Two further traps, both documented in full in `docs/BUGS.md` bug 4:

* If you fixed the expert count with `--hf-overrides` instead of editing
  `config.json`, MTP additionally fails with
  `AssertionError: Attempted to load weight (torch.Size([168])) into parameter (torch.Size([256]))`,
  because vLLM deliberately does not apply dict-form `hf_overrides` to the draft
  model (`vllm/config/speculative.py:602`). The CLI can only produce a dict, so
  **that workaround forecloses MTP entirely.**
* MTP disables `min_p`
  (`WARNING __init__.py:204 min_p and logit_bias parameters will not work with speculative decoding`),
  and this REAP cut's model card explicitly asks for `min_p=0.05` +
  `repetition_penalty=1.05` because it is **not Router-KD recovered**.

So MTP costs context, and also costs a quality guardrail the checkpoint author
asked for. It is not a close call on this hardware.
