#!/usr/bin/env bash
# serve-glm52.sh — serve 0xSero/GLM-5.2-504B-Nvidia (REAP NVFP4, 317.9 GB) on
# 4x RTX PRO 6000 Blackwell (96 GB, sm_120) with STOCK vLLM 0.25.1.
#
# Prerequisites (do these first, once):
#   1. scripts/setup-venv.sh            -> vllm 0.25.1 + flashinfer 0.6.14
#   2. scripts/fix-config.py $MODEL     -> deletes the stale num_experts key
#                                          (else: OOM at ~93.64 GiB/GPU on load)
#   3. patches/apply_dcp_patch.py       -> DCP/LSE support in the sm120 backend
#        $VENV/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py
#      (else: "AssertionError: Decode Context Parallelism (DCP) requires attention
#       implementations to return the softmax LSE during decode")
#
# Usage:
#   MODEL=/path/to/GLM-5.2-504B-Nvidia ./serve-glm52.sh
#   MODEL=... VENV=/opt/glm52-venv PORT=8000 CTX=250000 UTIL=0.92 DCP=1 ./serve-glm52.sh
#   ./serve-glm52.sh --model /path/to/model --port 8000 --no-dcp
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (env, overridable by args)
# ---------------------------------------------------------------------------
MODEL="${MODEL:-}"
VENV="${VENV:-./venv}"
HOST="${HOST:-127.0.0.1}"      # bind loopback by default; this server has no auth
PORT="${PORT:-8000}"
CTX="${CTX:-250000}"           # 250K. Native max_position_embeddings is 1048576,
                               # but KV memory, not the model, is the limit here.
UTIL="${UTIL:-0.92}"           # 0.92 of 94.97 GiB. Higher risks OOM under load;
                               # lower wastes KV pool.
DCP="${DCP:-1}"                # 1 = decode context parallelism across 4 GPUs.
TP="${TP:-4}"                  # tensor parallel = number of GPUs
SERVED_NAME="${SERVED_NAME:-glm-5.2}"
LOG="${LOG:-./glm52-serve.log}"
DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-}"   # optional: "a2a" trades some KV-path
                                           # latency characteristics; see below.

while [ $# -gt 0 ]; do
    case "$1" in
        --model)   MODEL="$2"; shift 2 ;;
        --venv)    VENV="$2"; shift 2 ;;
        --host)    HOST="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        --ctx)     CTX="$2"; shift 2 ;;
        --util)    UTIL="$2"; shift 2 ;;
        --tp)      TP="$2"; shift 2 ;;
        --log)     LOG="$2"; shift 2 ;;
        --dcp)     DCP=1; shift ;;
        --no-dcp)  DCP=0; shift ;;
        --dcp-comm-backend) DCP_COMM_BACKEND="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0 ;;
        *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
done

if [ -z "${MODEL}" ]; then
    echo "ERROR: MODEL is not set. Pass --model /path/to/GLM-5.2-504B-Nvidia" >&2
    echo "       or export MODEL=/path/to/GLM-5.2-504B-Nvidia" >&2
    exit 2
fi
if [ ! -d "${MODEL}" ]; then
    echo "ERROR: model directory not found: ${MODEL}" >&2
    exit 2
fi

VLLM_BIN="${VENV}/bin/vllm"
if [ ! -x "${VLLM_BIN}" ]; then
    echo "ERROR: ${VLLM_BIN} not found. Run scripts/setup-venv.sh first," >&2
    echo "       or point VENV at your venv." >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Pre-flight: is the port already taken?
# This matters more than it looks. The readiness probe below polls
# http://HOST:PORT/v1/models. If some OTHER process already owns that port, the
# probe answers 200 from THAT process, and this script would report READY while
# our vLLM was actually dying in the background with "Address already in use".
# Fail loudly here instead.
# ---------------------------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    if python3 - "${HOST}" "${PORT}" <<'PYEOF'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.settimeout(2)
try:
    s.connect((host, port))
    sys.exit(0)      # something answered -> port is occupied
except OSError:
    sys.exit(1)      # nothing there -> free
finally:
    s.close()
PYEOF
    then
        echo "ERROR: something is ALREADY listening on ${HOST}:${PORT}." >&2
        echo "       Refusing to start: the readiness probe cannot tell your vLLM" >&2
        echo "       apart from the process already on that port, and vLLM would" >&2
        echo "       fail to bind anyway." >&2
        echo >&2
        echo "       Stop that process, or pick another port:  PORT=8001 $0 ..." >&2
        exit 4
    fi
fi

# ---------------------------------------------------------------------------
# Pre-flight: the stale num_experts key. Cheap to check, expensive to hit.
# ---------------------------------------------------------------------------
if command -v python3 >/dev/null 2>&1 && [ -f "${MODEL}/config.json" ]; then
    if python3 - "${MODEL}/config.json" <<'PYEOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
ne, nre = cfg.get("num_experts"), cfg.get("n_routed_experts")
sys.exit(0 if (ne is not None and nre is not None and ne != nre) else 1)
PYEOF
    then
        echo "ERROR: ${MODEL}/config.json still has a stale num_experts key that" >&2
        echo "       disagrees with n_routed_experts. transformers will re-route the" >&2
        echo "       stale value over the real one and you will OOM at ~93.64 GiB/GPU" >&2
        echo "       during weight load. Fix it first:" >&2
        echo >&2
        echo "         ${VENV}/bin/python scripts/fix-config.py ${MODEL}" >&2
        exit 3
    fi
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# flashinfer/jit/env.py refuses to run unless flashinfer-python and its companion
# flashinfer-cubin wheel match. We need flashinfer-python 0.6.14 (vLLM 0.25.1's
# sm120 sparse-MLA call site passes kv_scale_format=, which does not exist in
# 0.6.13), but flashinfer-cubin has NO 0.6.14 release on PyPI (latest 0.6.13).
# So the guard must be disabled or startup aborts.
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Blackwell + PCIe with no NVLink: NCCL's P2P allreduce path DEADLOCKS. Symptom is
# a hang with no error, usually during init or the first collective. Disabling P2P
# routes collectives through host-staged CPU bounce buffers -- slower per hop, but
# it actually completes. This is also why DCP costs ~18% decode throughput (below).
export NCCL_P2P_DISABLE=1

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
ARGS=(
    --model "${MODEL}"
    --served-model-name "${SERVED_NAME}"
    --host "${HOST}"
    --port "${PORT}"

    # 4-way tensor parallel across the 4 GPUs.
    --tensor-parallel-size "${TP}"

    # Expert parallel: shard the 168 routed experts across ranks (42/rank) instead
    # of replicating them. Non-negotiable at this size -- 317.9 GB of weights only
    # fit as 79.1 GB/GPU with EP on.
    --enable-expert-parallel

    # NVFP4 checkpoint. Selects the modelopt_fp4 path, which on sm_120 resolves to
    # the FLASHINFER_CUTLASS NvFp4 MoE kernel (NOT Marlin -- Marlin would be wrong
    # here and much slower).
    --quantization modelopt_fp4

    # fp8 KV. Note this is effectively a no-op knob for this model: vLLM silently
    # promotes auto|fp8|fp8_e4m3 -> fp8_ds_mla (mla_attention.py:322-336), and the
    # sm120 sparse backend raises on anything else (flashinfer_mla_sparse_sm120.py:65-67).
    # Stated explicitly so the intent is on the record.
    --kv-cache-dtype fp8

    --max-model-len "${CTX}"
    --gpu-memory-utilization "${UTIL}"

    # GLM-5.2 emits native thinking blocks and tool calls. In vLLM 0.25.1, glm45
    # and glm47 are both registered as ALIASES of the same glm47_moe parser class,
    # for tool AND reasoning parsers alike. Do NOT pass "glm47_moe" -- that is the
    # module name, not a valid CLI value, and it will fail.
    --enable-auto-tool-choice
    --tool-call-parser glm47
    --reasoning-parser glm45

    # glm_moe_dsa (GlmMoeDsaForCausalLM) needs the checkpoint's remote code.
    --trust-remote-code
)

if [ "${DCP}" = "1" ]; then
    # Decode Context Parallelism: shard the KV cache across the 4 GPUs during
    # decode. This is the whole point of the repo. Per-GiB KV capacity goes up
    # EXACTLY 4.00x (19,898 -> 79,595 tokens/GiB). Net gain is 3.37x, not 4.00x,
    # because DCP's workspaces eat into the pool (8.25 -> 6.96 GiB available).
    #
    # Requires patches/flashinfer_mla_sparse_sm120-dcp.patch. Without it:
    #   AssertionError: Decode Context Parallelism (DCP) requires attention
    #   implementations to return the softmax LSE during decode, but
    #   FlashInferMLASparseSM120Impl does not.
    # (and the assertion's advice to "try a different backend" is impossible to
    #  follow: FLASHINFER_MLA_SPARSE_SM120 is the ONLY backend vLLM offers on sm_120.)
    ARGS+=( --decode-context-parallel-size "${TP}" )

    # DELIBERATELY NOT PASSING --cp-kv-cache-interleave-size.
    # It must stay at its default of 1; any other value makes vLLM raise at
    # indexer.py:261-263.

    if [ -n "${DCP_COMM_BACKEND}" ]; then
        # Escape hatch. DCP adds a q all-gather + an LSE reduce per layer per
        # decode step = 156 collectives/token across 78 layers, and with
        # NCCL_P2P_DISABLE=1 each is host-staged. The cost is ~18% decode
        # (52 -> 43 tok/s). It is latency-bound, not bandwidth-bound (~430 MB/s at
        # 52 tok/s is trivial for PCIe Gen5). If decode latency matters more to you
        # than context, try: --dcp-comm-backend a2a
        ARGS+=( --dcp-comm-backend "${DCP_COMM_BACKEND}" )
    fi
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
cat <<EOF
==============================================================
 GLM-5.2-504B-NVFP4 / sm_120 / stock vLLM 0.25.1
==============================================================
  model      : ${MODEL}
  venv       : ${VENV}
  endpoint   : http://${HOST}:${PORT}
  context    : ${CTX}
  gpu util   : ${UTIL}
  TP         : ${TP}
  DCP        : $([ "${DCP}" = "1" ] && echo "${TP} (patched sm120 backend required)" || echo "OFF")
  log        : ${LOG}

  COLD START TAKES ~340s the first time. flashinfer JIT-compiles its kernels
  from scratch: ~5 minutes of silent nvcc with the GPUs at 0% / 15W, while the
  log repeats:
      "shm_broadcast: No available shared memory broadcast block found in 60 seconds"
  THAT IS COMPILATION, NOT A HANG. Do not kill it. Warm starts are ~45-105s.
==============================================================

EOF

: > "${LOG}"
"${VLLM_BIN}" serve "${ARGS[@]}" >>"${LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo
        echo "Stopping server (pid ${SERVER_PID}) ..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

echo "Server pid ${SERVER_PID}. Waiting for http://${HOST}:${PORT}/v1/models ..."

# Health-wait. 600s budget covers the ~340s cold JIT compile with headroom.
DEADLINE=$(( $(date +%s) + 600 ))
READY=0
while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo
        echo "SERVER EXITED before becoming ready. Last 40 log lines:"
        echo "--------------------------------------------------------------"
        tail -n 40 "${LOG}"
        echo "--------------------------------------------------------------"
        echo "Full log: ${LOG}"
        exit 1
    fi
    if curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
        # Trust a 200 only if OUR child is still the one running. Belt and
        # braces against answering for a foreign process on the same port.
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            READY=1
            break
        fi
    fi
    sleep 5
    printf '.'
done
echo

if [ "${READY}" != "1" ]; then
    echo "TIMED OUT waiting for readiness. Last 40 log lines:"
    tail -n 40 "${LOG}"
    cleanup
    exit 1
fi

# ---------------------------------------------------------------------------
# Go / no-go: did DCP actually engage?
# ---------------------------------------------------------------------------
cat <<'EOF'
==============================================================
 READY. KV cache report from the log:
==============================================================
EOF

grep -E "GPU KV cache size|Maximum concurrency|Available KV cache memory|number of experts" "${LOG}" || \
    echo "  (no matching lines found -- check ${LOG} manually)"

cat <<EOF

--------------------------------------------------------------
 HOW TO READ THAT (at --max-model-len 250000, DCP=4):
   EXPECT   ~554,759 KV tokens, max concurrency ~2.22x   -> DCP engaged.
   IF YOU SEE ~164,160 KV tokens / ~0.66x                -> DCP did NOT engage.
     0.66x means it cannot fit even ONE full-context request. Confirm the DCP
     patch is applied to the sm120 backend in this venv and that
     --decode-context-parallel-size is on the command line.
   IF YOU SEE ~481,576 KV tokens / ~1.93x (~13% low)     -> fragmented KV pool
     from rapid stop->start cycling. Stop cleanly, let it settle ~12s with no
     residual compute processes on the GPUs, and relaunch.
   Expert split should read "Local/global number of experts: 42/168".
     "64/256" means config.json still has the stale num_experts key.

 Note the KV pool is a FIXED BYTE BUDGET (leftover VRAM after weights). It does
 NOT grow with --max-model-len; context and concurrency trade against each other.
 At --max-model-len 131072 with DCP=4 you should see ~4.23x concurrency.
--------------------------------------------------------------

 Endpoints: http://${HOST}:${PORT}/v1/chat/completions   (OpenAI)
            http://${HOST}:${PORT}/v1/messages           (Anthropic, native)

 CLIENT GOTCHA: GLM-5.2 ALWAYS emits a thinking block first. Set max_tokens >= 1024
 or you get a SILENTLY EMPTY response with no error (the thinking block consumes
 the whole budget and stop_reason comes back as max_tokens with text="").

 Tail the log:  tail -f ${LOG}
 Stop:          kill ${SERVER_PID}
--------------------------------------------------------------
EOF

# Hand the terminal to the server: Ctrl-C now stops it via the trap.
wait "${SERVER_PID}"
