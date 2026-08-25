"""
Red-team test for the input guardrail.

WHAT THIS DOES
Sends every prompt in attack_dataset.py through the same Lakera Guard check
used by the live RAG graph (guardrails.check_input_safety), counts how many
get flagged per category, and writes a results file + prints a summary table.

HOW TO RUN
    python red_team/run_red_team.py

That's it — no CI/CD, no scheduler, no monitoring stack needed for this to
count as "done" for a portfolio project. Run it manually whenever you change
the guardrail or want fresh numbers for your README. (If you ever do want it
automated, the only thing worth adding later is a single GitHub Actions step
that runs this script on push — see the note at the bottom of the file. Not
required.)

OUTPUT
    red_team/results.json   <- machine-readable results, timestamped
    stdout                  <- human-readable summary table

This script exits with code 1 if any category falls below its threshold in
attack_dataset.THRESHOLDS, so it can double as a pass/fail check if you ever
do wire it into CI — but running it manually and pasting the table into your
README is enough for this project.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from guardrails import check_input_safety
from attack_dataset import ATTACKS, THRESHOLDS


def run() -> dict:
    results = {}
    total_prompts = 0
    total_blocked = 0

    for category, prompts in ATTACKS.items():
        blocked = 0
        per_prompt = []

        for prompt in prompts:
            outcome = check_input_safety(prompt)
            was_blocked = bool(outcome["flagged"])
            blocked += was_blocked
            per_prompt.append({
                "prompt": prompt,
                "blocked": was_blocked,
                "categories": outcome.get("categories", []),
            })
            time.sleep(0.2)  # be polite to the API / avoid rate limits

        rate = blocked / len(prompts)
        results[category] = {
            "total": len(prompts),
            "blocked": blocked,
            "detection_rate": round(rate, 3),
            "threshold": THRESHOLDS[category],
            "passed": rate >= THRESHOLDS[category],
            "prompts": per_prompt,
        }
        total_prompts += len(prompts)
        total_blocked += blocked

    results["_summary"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_prompts": total_prompts,
        "total_blocked": total_blocked,
        "overall_detection_rate": round(total_blocked / total_prompts, 3),
    }
    return results


def print_table(results: dict):
    print("\n" + "=" * 60)
    print("RED TEAM RESULTS")
    print("=" * 60)
    print(f"{'Category':<20}{'Blocked':<12}{'Rate':<10}{'Threshold':<10}{'Status'}")
    print("-" * 60)
    for category, r in results.items():
        if category == "_summary":
            continue
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{category:<20}{r['blocked']}/{r['total']:<12}"
              f"{r['detection_rate']*100:>5.1f}%   {r['threshold']*100:>5.1f}%     {status}")
    print("-" * 60)
    s = results["_summary"]
    print(f"Overall: {s['total_blocked']}/{s['total_prompts']} blocked "
          f"({s['overall_detection_rate']*100:.1f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    results = run()
    print_table(results)

    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Full results written to {out_path}")

    all_passed = all(r["passed"] for k, r in results.items() if k != "_summary")
    sys.exit(0 if all_passed else 1)

# --- Optional, NOT required for this project ---
# If you later want this to run automatically on every push, a single step
# in .github/workflows/red-team.yml calling `python red_team/run_red_team.py`
# with LAKERA_API_KEY as a repo secret is all it takes. Skip this entirely
# unless you specifically want a CI badge.