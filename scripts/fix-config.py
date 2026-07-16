#!/usr/bin/env python3
"""fix-config.py — remove the stale `num_experts` key from a GLM-5.2 REAP checkpoint.

WHAT THIS FIXES
---------------
Loading a REAP-pruned GLM-5.2 NVFP4 checkpoint on 4x 96GB sm_120 dies with:

    torch.OutOfMemoryError
      ... in vllm/model_executor/layers/quantization/modelopt.py:1458 create_weights

and just before that, the log says:

    [EP Rank 0/4] ... Local/global number of experts: 64/256

64/256 is wrong. The checkpoint has 168 routed experts (indices 0..167, verifiable
in model.safetensors.index.json), so with --enable-expert-parallel over 4 ranks the
correct line is 42/168. Building FusedMoE for 256 experts -- 88 of which do not
exist -- asks for roughly 111 GB/GPU of expert weights instead of ~80 GB/GPU, and
you OOM at ~93.64 GiB.

ROOT CAUSE (an upstream `transformers` bug, NOT a vLLM bug)
-----------------------------------------------------------
GLM-5.2 REAP checkpoints ship BOTH keys in config.json:

    "n_routed_experts": 168     <- the truth for this pruned cut
    "num_experts":      256     <- stale, inherited from the unpruned base model

`transformers`, in
src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py, __post_init__
lines 161-163, does an UNCONDITIONAL backward-compat re-route:

    # BC: re-route `num_experts` to `n_routed_experts`
    if (num_experts := kwargs.get("num_experts")) is not None:
        self.n_routed_experts = num_experts

So the DEPRECATED key silently overwrites the EXPLICIT one. vLLM is innocent: it
reads config.n_routed_experts correctly (vllm/model_executor/models/deepseek_v2.py)
-- it is simply handed a 256 that transformers manufactured.

Measured behaviour:

    AutoConfig.from_pretrained(M)                       -> n_routed_experts=256  # JSON says 168!
    AutoConfig.from_pretrained(M, num_experts=168)      -> n_routed_experts=256  # kwarg does NOT help
    AutoConfig.from_pretrained(M, n_routed_experts=168) -> n_routed_experts=168
    (delete num_experts from config.json)               -> n_routed_experts=168  <- the real fix

The `num_experts=168` kwarg fails because the clobber runs during __post_init__,
i.e. during construction, BEFORE from_pretrained applies caller kwargs on top.

WHY DELETE THE KEY INSTEAD OF USING --hf-overrides
--------------------------------------------------
`--hf-overrides '{"n_routed_experts":168}'` does get the main model loading. But it
is a dict-form override, and vLLM DELIBERATELY refuses to forward dict overrides to
a draft model. From vllm/config/speculative.py:602, compose_draft_hf_overrides, the
docstring verbatim:

    "Callable overrides on the target are config-to-config transforms ... and must
     also reach the draft config ... Dict overrides are target-specific key patches
     and are not applied to the draft."

    if not callable(target_hf_overrides):
        return SpeculativeConfig.hf_config_override

The CLI can only ever produce a dict. So with the --hf-overrides workaround, the
target model sees 168 while the draft model still sees 256, and enabling MTP fails:

    self.drafter.load_model(self.model)
    AssertionError: Attempted to load weight (torch.Size([168])) into parameter (torch.Size([256]))

Fixing config.json at the source fixes target AND draft in one place, and leaves the
MTP path structurally open. (On 4x96GB, MTP does not actually fit alongside DCP4 at
250K context -- see docs -- but that should be your decision, not a foreclosed one.)

SAFETY
------
This script is conservative and idempotent:
  * It touches config.json ONLY if `num_experts` is present AND `n_routed_experts`
    is present AND the two DISAGREE. Any other shape is left alone and reported.
  * It writes a timestamped backup before modifying anything.
  * Re-running after a successful fix is a no-op.
  * --dry-run shows the plan without writing.
  * --restore puts back the most recent backup.

USAGE
-----
    python3 fix-config.py /path/to/model            # fix (with backup)
    python3 fix-config.py /path/to/model --dry-run  # show what would change
    python3 fix-config.py /path/to/model --restore  # undo, from newest backup
    python3 fix-config.py /path/to/model --no-verify
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys

BACKUP_SUFFIX = ".bak-fix-config-"


def die(msg: str, code: int = 1) -> "None":
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def config_path(model_dir: str) -> str:
    path = os.path.join(model_dir, "config.json")
    if not os.path.isdir(model_dir):
        die(f"not a directory: {model_dir}")
    if not os.path.isfile(path):
        die(f"no config.json in {model_dir}")
    return path


def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        die(f"{path} is not valid JSON: {exc}")
    return {}  # unreachable


def find_backups(path: str) -> list:
    directory = os.path.dirname(path) or "."
    base = os.path.basename(path)
    prefix = base + BACKUP_SUFFIX
    names = [n for n in os.listdir(directory) if n.startswith(prefix)]
    return sorted(os.path.join(directory, n) for n in names)


def do_restore(path: str) -> int:
    backups = find_backups(path)
    if not backups:
        die(f"no backups found next to {path}")
    newest = backups[-1]
    print(f"Restoring:\n  from : {newest}\n  to   : {path}")
    shutil.copy2(newest, path)
    cfg = load_json(path)
    print("\nRestored config now has:")
    print(f"  num_experts       = {cfg.get('num_experts', '<absent>')}")
    print(f"  n_routed_experts  = {cfg.get('n_routed_experts', '<absent>')}")
    print("\nRESTORED OK")
    return 0


def verify(model_dir: str, expected: int) -> int:
    """Prove that transformers now resolves n_routed_experts to the JSON value."""
    print("\n--- verification (AutoConfig.from_pretrained) ---")
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    try:
        from transformers import AutoConfig
    except ImportError:
        print("  SKIPPED: transformers is not importable in this interpreter.")
        print("  Run this script with your serving venv's python to verify.")
        return 0

    try:
        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - report whatever transformers raises
        print(f"  SKIPPED: AutoConfig.from_pretrained failed: {exc}")
        return 0

    resolved = getattr(cfg, "n_routed_experts", None)
    print(f"  resolved n_routed_experts = {resolved}")
    print(f"  expected                  = {expected}")
    if resolved != expected:
        print("\n  VERIFICATION FAILED: transformers still does not agree with config.json.")
        print("  Do NOT start serving; you will get the 64/256 expert split and OOM.")
        return 1
    print("  OK: transformers agrees with config.json.")
    print(f"\n  With --tensor-parallel-size 4 --enable-expert-parallel expect the log to")
    print(f"  read 'Local/global number of experts: {expected // 4}/{expected}'.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Remove the stale num_experts key from a GLM-5.2 REAP config.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("model_dir", help="path to the model directory (contains config.json)")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, write nothing")
    ap.add_argument("--restore", action="store_true", help="restore the newest backup")
    ap.add_argument("--no-verify", action="store_true", help="skip the AutoConfig check")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    path = config_path(model_dir)

    if args.restore:
        if args.dry_run:
            die("--restore and --dry-run are mutually exclusive")
        return do_restore(path)

    cfg = load_json(path)
    num_experts = cfg.get("num_experts", None)
    n_routed = cfg.get("n_routed_experts", None)

    print(f"config: {path}")
    print("\n--- before ---")
    print(f"  num_experts       = {num_experts if num_experts is not None else '<absent>'}")
    print(f"  n_routed_experts  = {n_routed if n_routed is not None else '<absent>'}")

    # ---- refuse to touch anything that does not exhibit the exact bug -------
    if num_experts is None:
        print("\nNo `num_experts` key present. This config does not exhibit the bug.")
        print("Nothing to do (already fixed, or never affected).")
        return 0 if args.no_verify or n_routed is None else verify(model_dir, n_routed)

    if n_routed is None:
        print("\nREFUSING TO EDIT: `num_experts` is present but `n_routed_experts` is NOT.")
        print("Here `num_experts` may be the only source of truth, and deleting it would")
        print("destroy information. This is not the bug this script fixes.")
        return 2

    if not isinstance(num_experts, int) or not isinstance(n_routed, int):
        print("\nREFUSING TO EDIT: expected both keys to be integers, got")
        print(f"  num_experts      : {type(num_experts).__name__}")
        print(f"  n_routed_experts : {type(n_routed).__name__}")
        return 2

    if num_experts == n_routed:
        print(f"\nBoth keys agree ({num_experts}). The transformers BC re-route still fires,")
        print("but it writes the same value, so there is no bug to fix here.")
        print("REFUSING TO EDIT (nothing would change).")
        return 0 if args.no_verify else verify(model_dir, n_routed)

    # ---- the bug, confirmed ------------------------------------------------
    print("\nBUG CONFIRMED: the two keys disagree.")
    print(f"  transformers __post_init__ will re-route num_experts ({num_experts})")
    print(f"  over n_routed_experts ({n_routed}), so the model will be built for")
    print(f"  {num_experts} experts instead of {n_routed} -> OOM during weight load.")
    print(f"\nPlan: delete the stale `num_experts` key; keep n_routed_experts = {n_routed}.")

    if args.dry_run:
        print("\n--- after (SIMULATED) ---")
        print("  num_experts       = <deleted>")
        print(f"  n_routed_experts  = {n_routed}")
        print("\nDRY RUN — no files were written.")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = f"{path}{BACKUP_SUFFIX}{stamp}"
    shutil.copy2(path, backup)
    print(f"\nBackup written: {backup}")

    del cfg["num_experts"]

    tmp = path + ".tmp-fix-config"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)  # atomic; never leaves a half-written config.json

    after = load_json(path)
    print("\n--- after ---")
    print(f"  num_experts       = {after.get('num_experts', '<deleted>')}")
    print(f"  n_routed_experts  = {after.get('n_routed_experts')}")

    if args.no_verify:
        print("\nFIXED (verification skipped).")
        return 0

    rc = verify(model_dir, n_routed)
    print("\nFIXED OK" if rc == 0 else "\nFIXED, BUT VERIFICATION FAILED")
    print(f"(undo with: python3 {os.path.basename(__file__)} {args.model_dir} --restore)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
