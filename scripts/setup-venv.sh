#!/usr/bin/env bash
# setup-venv.sh — build the stock-vLLM venv for GLM-5.2-504B-NVFP4 on sm_120.
#
# This creates a Python 3.12 venv containing:
#   vllm==0.25.1            (STOCK. No fork, no Docker, no source build.)
#   flashinfer-python==0.6.14
#
# Why 0.6.14 and not the 0.6.13 that vLLM pins:
#   vLLM 0.25.1's own sm120 sparse-MLA call site passes kv_scale_format= to
#   flashinfer's trtllm_batch_decode_with_kv_cache_mla(). That keyword only
#   exists in flashinfer 0.6.14. With the pinned 0.6.13 you get, at runtime:
#       TypeError: trtllm_batch_decode_with_kv_cache_mla() got an unexpected
#       keyword argument 'kv_scale_format'
#   The ==0.6.13 pin is pip metadata only; nothing enforces it at runtime.
#   So: install 0.6.14 deliberately and ignore pip's resolver complaint.
#
# Usage:
#   ./setup-venv.sh [VENV_PATH]
#   VENV=/opt/glm52-venv ./setup-venv.sh
#
set -euo pipefail

VENV="${1:-${VENV:-./venv}}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

VLLM_VERSION="0.25.1"
FLASHINFER_VERSION="0.6.14"

echo "=============================================================="
echo " GLM-5.2 sm_120 stock-vLLM venv setup"
echo "=============================================================="
echo "  venv path     : ${VENV}"
echo "  interpreter   : ${PYTHON_BIN}"
echo "  vllm          : ${VLLM_VERSION}"
echo "  flashinfer    : ${FLASHINFER_VERSION}"
echo

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON_BIN} not found on PATH."
    echo "       Install Python 3.12 or set PYTHON_BIN=/path/to/python3.12"
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Create the venv
# ---------------------------------------------------------------------------
if [ -d "${VENV}" ]; then
    echo "[1/5] venv already exists at ${VENV} — reusing it."
else
    echo "[1/5] Creating venv at ${VENV} ..."
    "${PYTHON_BIN}" -m venv "${VENV}"
fi

VPY="${VENV}/bin/python"
VPIP="${VENV}/bin/pip"

if [ ! -x "${VPY}" ]; then
    echo "ERROR: ${VPY} is missing or not executable — venv creation failed."
    exit 1
fi

echo "      -> $("${VPY}" --version)"

# ---------------------------------------------------------------------------
# 2. Baseline tooling
# ---------------------------------------------------------------------------
echo "[2/5] Upgrading pip/setuptools/wheel ..."
"${VPIP}" install --upgrade pip setuptools wheel >/dev/null
echo "      -> pip $("${VPIP}" --version | awk '{print $2}')"

# ---------------------------------------------------------------------------
# 3. vLLM (pulls torch 2.11 and flashinfer-python==0.6.13 as a dependency)
# ---------------------------------------------------------------------------
echo "[3/5] Installing vllm==${VLLM_VERSION} (this pulls torch and CUDA wheels; takes a while) ..."
"${VPIP}" install "vllm==${VLLM_VERSION}"

# ---------------------------------------------------------------------------
# 4. Upgrade flashinfer over the top of vLLM's pin
# ---------------------------------------------------------------------------
echo
echo "[4/5] Upgrading flashinfer-python to ${FLASHINFER_VERSION} ..."
echo
echo "      NOTE: pip is ABOUT TO PRINT A DEPENDENCY CONFLICT WARNING, roughly:"
echo
echo "        ERROR: pip's dependency resolver does not currently take into account"
echo "        all the packages that are installed. This behaviour is the source of"
echo "        the following dependency conflicts."
echo "        vllm ${VLLM_VERSION} requires flashinfer-python==0.6.13, but you have"
echo "        flashinfer-python ${FLASHINFER_VERSION} which is incompatible."
echo
echo "      THAT WARNING IS EXPECTED AND SAFE FOR THIS PATH. vLLM's own sm120"
echo "      sparse-MLA code calls trtllm_batch_decode_with_kv_cache_mla() with"
echo "      kv_scale_format=, a keyword that does not exist until 0.6.14. The pin"
echo "      is stale metadata; the runtime needs the newer wheel. Step 5 proves it"
echo "      by inspecting the real function signature."
echo
"${VPIP}" install "flashinfer-python==${FLASHINFER_VERSION}"

# ---------------------------------------------------------------------------
# 5. Verify — do not assume, check.
# ---------------------------------------------------------------------------
# FLASHINFER_DISABLE_VERSION_CHECK=1 is REQUIRED even just to import flashinfer
# here: flashinfer/jit/env.py compares flashinfer-python against the companion
# flashinfer-cubin wheel, and flashinfer-cubin has NO 0.6.14 release on PyPI
# (latest is 0.6.13). Without this env var the version guard aborts.
echo "[5/5] Verifying the install ..."
echo

FLASHINFER_DISABLE_VERSION_CHECK=1 "${VPY}" - <<'PYEOF'
import inspect
import sys

expected_vllm = "0.25.1"
expected_flashinfer = "0.6.14"

failures = []

# --- vllm version ---------------------------------------------------------
import vllm
print(f"  vllm.__version__       = {vllm.__version__}")
if vllm.__version__ != expected_vllm:
    failures.append(f"vllm is {vllm.__version__}, expected {expected_vllm}")

# --- flashinfer version ---------------------------------------------------
import flashinfer
print(f"  flashinfer.__version__ = {flashinfer.__version__}")
if flashinfer.__version__ != expected_flashinfer:
    failures.append(
        f"flashinfer is {flashinfer.__version__}, expected {expected_flashinfer}"
    )

# --- the signature check that is the WHOLE REASON for 0.6.14 --------------
# vLLM 0.25.1 passes kv_scale_format= to this function. If the parameter is not
# in the signature, serving dies with a TypeError the moment the first decode
# step runs -- i.e. long after startup looked fine. Catch it now.
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla as _fn

params = list(inspect.signature(_fn).parameters)
print(f"  trtllm_batch_decode_with_kv_cache_mla params:")
print(f"    {params}")

if "kv_scale_format" in params:
    print("  OK: kv_scale_format IS present in the signature.")
else:
    failures.append(
        "kv_scale_format is MISSING from "
        "trtllm_batch_decode_with_kv_cache_mla(). This is the 0.6.13 signature; "
        "vLLM 0.25.1 will raise TypeError at decode time."
    )

# --- torch / GPU sanity ---------------------------------------------------
import torch
print(f"  torch.__version__      = {torch.__version__}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"  CUDA devices           = {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        cc = f"sm_{props.major}{props.minor}"
        mem = props.total_memory / (1024 ** 3)
        print(f"    [{i}] {props.name}  {cc}  {mem:.2f} GiB")
        if cc != "sm_120":
            print(f"        WARNING: this repo targets sm_120; device {i} is {cc}.")
else:
    print("  CUDA devices           = NONE VISIBLE (ok if this is a build-only host)")

print()
if failures:
    print("VERIFICATION FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print("VERIFICATION PASSED.")
PYEOF

echo
echo "=============================================================="
echo " Setup complete."
echo "=============================================================="
echo
echo " Next steps:"
echo
echo "   1. Fix the checkpoint's stale num_experts key (otherwise you OOM at"
echo "      ~93.64 GiB/GPU during weight load):"
echo
echo "        ${VPY} scripts/fix-config.py /path/to/GLM-5.2-504B-Nvidia"
echo
echo "   2. Apply the DCP/LSE patch to the sm120 sparse-MLA backend (otherwise"
echo "      --decode-context-parallel-size 4 asserts and you are capped at"
echo "      ~164,160 KV tokens):"
echo
echo "        ${VPY} patches/apply_dcp_patch.py \\"
echo "          ${VENV}/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
echo
echo "   3. Serve:"
echo
echo "        VENV=${VENV} MODEL=/path/to/GLM-5.2-504B-Nvidia scripts/serve-glm52.sh"
echo
echo " Remember: every process that imports flashinfer needs"
echo " FLASHINFER_DISABLE_VERSION_CHECK=1 (serve-glm52.sh exports it for you)."
echo
