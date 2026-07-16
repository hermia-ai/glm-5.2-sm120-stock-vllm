#!/usr/bin/env python3
"""Apply the SM120 sparse-MLA DCP/LSE enablement edits.

Usage:  python3 apply_dcp_patch.py <target.py>

Idempotent-ish: refuses to run twice (checks for the marker).
Edits ONLY the file given -- it never searches for or touches any other venv.

The target is normally:
  $VENV/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py
"""
import sys

p = sys.argv[1]
s = open(p).read()

if "can_return_lse_for_decode" in s:
    print("ALREADY PATCHED — aborting")
    sys.exit(1)

orig = s

# ---- 1. imports ----------------------------------------------------------
s = s.replace(
    """from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)""",
    """from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseImpl,
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)""",
    1,
)

# ---- 2. class attributes -------------------------------------------------
s = s.replace(
    '''class FlashInferMLASparseSM120Impl(SparseMLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""
''',
    '''class FlashInferMLASparseSM120Impl(SparseMLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

    # The SM120 sparse kernel DOES compute and emit the softmax LSE. The GLM
    # (dsv3_2) path writes per-split mid_lse (decode_dsv3_2_kernel.cuh:652),
    # then reuses decode-dsv4's merge kernel
    # (sparse_mla_sm120_decode_dsv3_2.cu:113-121), which writes the final
    # out_lse at decode_dsv4_kernel.cuh:911-912.
    can_return_lse_for_decode: bool = True
    # The kernel emits LSE in LOG2 space, NOT natural log:
    #   decode_dsv4_kernel.cuh:865  sm_glse = log2f(total_sum) + gmax
    #   common/online_softmax.cuh:64-66 "Compute LSE ... in log2 space"
    # so the DCP reducer must use exp2 (ops/common.py:59-64 IS_BASE_E).
    lse_base_on_e: bool = False
''',
    1,
)

# ---- 3. DCP index filtering + output head count --------------------------
s = s.replace(
    """        topk_indices_physical = cast(
            torch.Tensor,
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            ),
        )

        output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )""",
    """        if self.dcp_world_size > 1:
            # Each DCP rank owns an interleaved slice of the KV cache: filter
            # the top-k rows down to this rank's shard and hand the per-row
            # valid counts to the kernel as seq_lens (topk_length).
            topk_indices_physical, seq_lens = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=(
                    attn_metadata.cp_kv_cache_interleave_size
                ),
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        else:
            topk_indices_physical, seq_lens = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )

        # Under DCP the q heads are all-gathered across the DCP group before
        # forward_mqa (mla_attention.py:807), so the kernel sees
        # dcp_world_size * self.num_heads. Size the output from the ACTUAL
        # tensor, never from self.num_heads.
        num_query_heads = q.shape[1]
        output = q.new_empty(
            (num_actual_toks, num_query_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )""",
    1,
)

# ---- 4. call site + LSE unpack -------------------------------------------
s = s.replace(
    "        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(",
    "        kernel_out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(",
    1,
)
s = s.replace("            seq_lens=None,\n", "            seq_lens=seq_lens,\n", 1)
s = s.replace(
    """            kv_scale_format=self.kv_scale_format,
        )
        return out.squeeze(1), None""",
    """            kv_scale_format=self.kv_scale_format,
            return_lse=self.need_to_return_lse_for_decode,
        )
        if self.need_to_return_lse_for_decode:
            assert isinstance(kernel_out, tuple)
            o, lse = kernel_out
        else:
            o = cast(torch.Tensor, kernel_out)
            lse = None

        out = o.squeeze(1)
        if lse is None:
            return out, None

        # The shared DCP reducer (cp_lse_ag_out_rs / dcp_a2a_lse_reduce) wants
        # (num_tokens, num_heads) float32. Reuse the sibling's normalizer.
        lse = FlashInferMLASparseImpl._normalize_lse(lse, out.shape[0], out.shape[1])
        # Rows whose top-k set is empty on this rank must not contribute to the
        # cross-rank softmax denominator.
        empty_rows = (topk_indices_physical == -1).all(dim=-1)
        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
        lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
        return out, lse""",
    1,
)

# ---- verify every edit landed -------------------------------------------
required = [
    "can_return_lse_for_decode: bool = True",
    "lse_base_on_e: bool = False",
    "triton_filter_and_convert_dcp_index",
    "FlashInferMLASparseImpl",
    "num_query_heads = q.shape[1]",
    "seq_lens=seq_lens,",
    "return_lse=self.need_to_return_lse_for_decode,",
    "_normalize_lse",
]
missing = [r for r in required if r not in s]
if missing or s == orig:
    print("EDIT FAILED — anchors did not match:", missing)
    sys.exit(2)

open(p, "w").write(s)
print("PATCHED OK ->", p)
