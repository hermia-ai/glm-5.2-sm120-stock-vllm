#!/usr/bin/env python3
"""
needle_test.py -- needle-in-a-haystack long-context validation for GLM-5.2 on vLLM.

Builds a large filler document, buries a unique secret in the middle of it, and asks the
model to read it back. Reports the REAL prompt token count measured by the server, not our
guess.

Dependency-light on purpose: stdlib `urllib` only. No requests, no openai, no anthropic.

    python3 bench/needle_test.py --base-url http://localhost:8000 --model glm-5.2

REFERENCE RESULTS (4x RTX PRO 6000 Blackwell 96GB, sm_120, stock vLLM 0.25.1,
DCP4 patch applied, --max-model-len 250000, KV pool 554,759 tokens):

    measured prompt_tokens   result
    ----------------------   ------
                    11,058   PASS
                    44,058   PASS
                    88,058   PASS
                   107,309   PASS
                   121,058   PASS
                   178,809   PASS
                   233,809   PASS

Those are the server-reported prompt_tokens, which is why they are not round numbers.

WHY WE REPORT MEASURED TOKENS AND NOT THE TARGET
------------------------------------------------
Our first pass sized the filler with a chars-per-token estimate and trusted it. It was off
by roughly 1.38x -- we overshot the context limit and got a wall of HTTP 400s from vLLM
before we understood why. The estimate below is deliberately conservative, and every row of
the output table prints what the *server* said the prompt was, via usage.prompt_tokens.
Read that column. Treat --targets as a request, not a promise.

WHY THIS RUNS STRICTLY SEQUENTIALLY
-----------------------------------
See the comment above run_all(). Short version: do not make it concurrent.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# Conservative chars-per-token for the ASCII filler below. Deliberately low so we
# UNDERSHOOT the target rather than overshoot it into HTTP 400 territory. The real number
# is what the server reports; this constant only decides how much text we generate.
CHARS_PER_TOKEN = 3.2

SECRET = "PLUM-47-HORIZON-9134"

NEEDLE = (
    "\n\nIMPORTANT RECORD: The archive access code for this document set is "
    f"{SECRET}. Memorize it; it appears exactly once.\n\n"
)

QUESTION = (
    "Read the document above. Somewhere inside it there is exactly one line beginning "
    "'IMPORTANT RECORD:' that states an archive access code. Reply with that access code "
    "and nothing else."
)

# Filler with enough lexical variety that it does not compress into a trivially cached
# prefix, but no accidental digit-strings that could be mistaken for the needle.
FILLER_SENTENCES = [
    "The survey team catalogued the sediment layers along the eastern ridge before dusk.",
    "Maintenance logs were transcribed by hand and later collated into the central ledger.",
    "Wind conditions remained stable throughout the observation window that season.",
    "The auxiliary generator was serviced on schedule and returned to standby duty.",
    "Field notes describe an unremarkable morning with light cloud cover and no rainfall.",
    "Inventory reconciliation found no discrepancies worth escalating to the supervisor.",
    "The northern access road remained passable despite the previous week of weather.",
    "Radio checks were completed at the usual intervals and all stations reported clear.",
    "Samples were labelled, sealed, and moved to cold storage without incident.",
    "The rota was adjusted slightly to accommodate the visiting inspection party.",
]


def build_prompt(target_tokens: int) -> str:
    """Build a document of roughly target_tokens with the needle buried at the midpoint."""
    target_chars = int(target_tokens * CHARS_PER_TOKEN)

    body = []
    used = 0
    i = 0
    while used < target_chars:
        line = f"[{i:06d}] {FILLER_SENTENCES[i % len(FILLER_SENTENCES)]}\n"
        body.append(line)
        used += len(line)
        i += 1

    mid = len(body) // 2
    body.insert(mid, NEEDLE)
    return "".join(body)


def post_json(url: str, payload: dict, timeout: int):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_text(choice: dict) -> str:
    """
    Pull the answer out of a choice, tolerating both shapes.

    With --reasoning-parser (which this deployment uses), GLM-5.2 streams its thinking into
    `reasoning_content` and the answer into `content`. `content` can legitimately be None
    if the whole budget went to thinking -- see the max_tokens floor note below. Read both.
    """
    msg = choice.get("message") or {}
    parts = []
    for key in ("content", "reasoning_content"):
        val = msg.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
    return "\n".join(parts)


def run_one(base_url: str, model: str, target: int, max_tokens: int, timeout: int) -> dict:
    prompt = build_prompt(target)
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # The model card for this REAP cut asks for these; they are not Router-KD recovered.
        "min_p": 0.05,
        "repetition_penalty": 1.05,
        "messages": [
            {"role": "user", "content": prompt + "\n\n" + QUESTION},
        ],
    }

    url = base_url.rstrip("/") + "/v1/chat/completions"
    t0 = time.time()
    try:
        resp = post_json(url, payload, timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        return {
            "target": target,
            "prompt_tokens": None,
            "ok": False,
            "seconds": time.time() - t0,
            "note": f"HTTP {e.code}: {detail}",
        }
    except Exception as e:  # noqa: BLE001 - a bad URL / dead server should not traceback
        return {
            "target": target,
            "prompt_tokens": None,
            "ok": False,
            "seconds": time.time() - t0,
            "note": f"{type(e).__name__}: {e}",
        }

    elapsed = time.time() - t0
    usage = resp.get("usage") or {}
    choices = resp.get("choices") or [{}]
    text = extract_text(choices[0])
    ok = SECRET in text

    note = ""
    if not ok:
        if not text.strip():
            note = "empty reply -- raise --max-tokens (see the floor note)"
        else:
            note = "answer: " + " ".join(text.split())[:60]

    return {
        "target": target,
        # The ONLY number worth trusting. Our char estimate is not it.
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "ok": ok,
        "seconds": elapsed,
        "note": note,
    }


def run_all(base_url: str, model: str, targets, max_tokens: int, timeout: int):
    """
    Run every target STRICTLY SEQUENTIALLY. Never parallelize this.

    The KV pool on this deployment is a fixed byte budget of leftover VRAM -- ~554,759
    tokens with the DCP4 patch, and max concurrency at 250K context is only 2.22x. Two
    200K-token requests in flight do not fit. What actually happens when you overcommit is
    not a clean error: requests get preempted and recomputed, all four GPUs pin at 100% and
    240-390W, and aggregate throughput collapses (we measured ~2 tok/s aggregate at 8-way
    concurrency on a low-headroom pool). A sequential run measures the model. A concurrent
    run measures the scheduler thrashing.
    """
    results = []
    for t in targets:
        print(f"  -> requesting ~{t:,} tokens ...", end="", flush=True)
        r = run_one(base_url, model, t, max_tokens, timeout)
        measured = r["prompt_tokens"]
        shown = f"{measured:,}" if measured else "?"
        print(f" measured {shown} prompt_tokens, {r['seconds']:.1f}s, "
              f"{'PASS' if r['ok'] else 'FAIL'}")
        results.append(r)
    return results


def print_table(results):
    print()
    print("  target      measured    gen    secs   result  note")
    print("  " + "-" * 76)
    for r in results:
        measured = f"{r['prompt_tokens']:,}" if r["prompt_tokens"] else "-"
        gen = str(r.get("completion_tokens") or "-")
        print(f"  {r['target']:>9,}  {measured:>10}  {gen:>5}  {r['seconds']:>6.1f}  "
              f"{'PASS' if r['ok'] else 'FAIL':<6}  {r['note']}")
    print()
    passed = sum(1 for r in results if r["ok"])
    print(f"  {passed}/{len(results)} passed")
    print()
    print("  Compare the MEASURED column, not the target column. Reference PASS points:")
    print("  11,058 / 44,058 / 88,058 / 107,309 / 121,058 / 178,809 / 233,809")


def main():
    p = argparse.ArgumentParser(
        description="Needle-in-a-haystack long-context test for GLM-5.2 on vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-url", default="http://localhost:8000",
                   help="vLLM server root, e.g. http://localhost:8000")
    p.add_argument("--model", required=True,
                   help="Served model name. Must match GET /v1/models exactly.")
    p.add_argument(
        "--max-tokens", type=int, default=1024,
        # WHY THE FLOOR MATTERS: GLM-5.2 ALWAYS emits a thinking block before any answer.
        # If max_tokens is small the thinking eats the entire budget and you get a
        # SILENTLY EMPTY response -- no error, stop_reason=max_tokens, content="".
        # Measured on a one-word question: max_tokens=64 -> thinking(193 chars), text="";
        # max_tokens=1024 -> thinking(353), text(6) -> "Tokyo.". Do not lower this below
        # 1024 and then report that the model failed the needle test.
        help="Generation budget. Do NOT go below 1024: GLM-5.2 always thinks first, and a "
             "smaller budget yields a silently empty reply. (default: 1024)",
    )
    p.add_argument("--targets", type=int, nargs="+",
                   default=[11000, 44000, 88000, 107000, 121000, 178000, 233000],
                   help="Approximate input sizes in tokens. Approximate is the operative "
                        "word -- see the module docstring.")
    p.add_argument("--timeout", type=int, default=1800,
                   help="Per-request timeout in seconds (default: 1800). Long prefills at "
                        "230K tokens are slow; a cold engine is slower still.")
    args = p.parse_args()

    if args.max_tokens < 1024:
        print("WARNING: --max-tokens below 1024. GLM-5.2 emits a thinking block first; "
              "expect silently empty replies and spurious FAILs.", file=sys.stderr)

    print(f"\nneedle_test: {args.model} @ {args.base_url}")
    print(f"secret: {SECRET}   max_tokens: {args.max_tokens}   sequential: yes\n")

    results = run_all(args.base_url, args.model, sorted(args.targets),
                      args.max_tokens, args.timeout)
    print_table(results)
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
