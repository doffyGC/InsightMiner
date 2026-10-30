"""
Evaluate the agent against questions with known answers.

    python -m app.evals                          # every case, default model
    python -m app.evals --model openai/gpt-oss-20b --repeat 3
    python -m app.evals --case chart             # only cases whose name contains "chart"
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from app.agents.data_agent import create_data_agent
from app.core.llm import DEFAULT_MODEL
from app.evals.cases import CASES
from app.evals.runner import run_cases, to_json, totals

RESULTS_DIR = Path("outputs") / "evals"


def main():
    parser = argparse.ArgumentParser(description="Evaluate the InsightMiner agent.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Groq model id (default {DEFAULT_MODEL})")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case, to measure flakiness")
    parser.add_argument("--case", default="", help="only run cases whose name contains this text")
    parser.add_argument("--list", action="store_true", help="list the cases and exit")
    args = parser.parse_args()

    cases = [c for c in CASES if args.case.lower() in c.name.lower()]
    if args.list or not cases:
        for c in CASES:
            print(f"- {c.name}: {' -> '.join(c.turns)}")
        return

    print(f"Evaluating {args.model} on {len(cases)} cases x {args.repeat} run(s)\n")

    def show(result):
        mark = "⚠️" if result.error else "✅" if result.passed else "❌"
        detail = "" if result.passed else f"  ({result.failures()})"
        print(f"{mark} {result.case:<28} {result.seconds:5.1f}s {result.tokens:>7,} tokens{detail}")

    evaluation = run_cases(cases, lambda: create_data_agent(args.model), args.repeat, on_result=show)
    summaries = evaluation.summaries

    if evaluation.stopped_by:
        print(f"\n⛔ Stopped: the provider's rate limit was reached.\n   {evaluation.stopped_by}")
        print(f"   Not run: {', '.join(evaluation.not_run)}")
    if not summaries:
        return

    t = totals(summaries)
    if args.repeat > 1:
        print("\nPass rate per case:")
        for s in summaries:
            print(f"  {s.case:<28} {s.pass_rate:.0%}")
    print(
        f"\nPassed {t['passed']}/{t['runs']} runs ({t['pass_rate']:.0%}) · "
        f"{t['cases_always_passing']}/{t['cases']} cases always passed · "
        f"no invented numbers in {t['grounded_rate']:.0%} of answers · "
        f"{t['errors']} errors · {t['avg_seconds']:.1f}s and {t['tokens'] // max(1, t['runs']):,} tokens per run"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", args.model.lower()).strip("_")
    path = RESULTS_DIR / f"eval-{slug}-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(to_json(evaluation, args.model), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Details (answers, tool calls, checks): {path}")


if __name__ == "__main__":
    main()
