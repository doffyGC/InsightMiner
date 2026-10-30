import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import app.tools.create_chart as create_chart
from app.evals.cases import Case
from app.evals.checks import CheckResult, Turn
from app.services.dataframe_services import dataframe_service


class ProviderError(Exception):
    """The model provider failed (rate limit, outage); says nothing about the agent's quality."""


class RateLimited(ProviderError):
    pass


@dataclass
class RunResult:
    case: str
    passed: bool
    checks: list[CheckResult]
    turns: list[Turn]
    seconds: float
    error: str | None = None
    tokens: int = 0

    @property
    def grounded(self) -> bool:
        return all(c.passed for c in self.checks if c.name == "grounded")

    def failures(self) -> str:
        if self.error:
            return f"error: {self.error}"
        return "; ".join(f"{c.name}: {c.detail}" for c in self.checks if not c.passed)


@dataclass
class CaseSummary:
    case: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.runs) / len(self.runs)


def check_response(response) -> None:
    # agno doesn't raise on provider failures: it returns the error as the answer with status ERROR
    if str(getattr(response, "status", "")).upper().endswith("ERROR"):
        message = str(response.content or "")
        if "rate_limit" in message or "Rate limit" in message:
            raise RateLimited(message)
        raise ProviderError(message)


def run_case(case: Case, agent_factory: Callable) -> RunResult:
    df = dataframe_service.load_csv(case.dataset)
    agent = agent_factory()  # fresh conversation per run
    turns: list[Turn] = []
    tokens = 0
    start = time.perf_counter()
    try:
        for message in case.turns:
            response = agent.run(message)
            tokens += getattr(response.metrics, "total_tokens", 0) or 0
            check_response(response)
            tools = [(t.tool_name, dict(t.tool_args or {}), str(t.result)) for t in (response.tools or [])]
            turns.append(Turn(answer=str(response.content or ""), tools=tools))
    except RateLimited:
        raise
    except Exception as e:
        error = f"{type(e).__name__}: {e}"[:300]
        return RunResult(case.name, False, [], turns, time.perf_counter() - start, error, tokens)

    checks = [check.run(df, turns) for check in case.all_checks()]
    return RunResult(case.name, all(c.passed for c in checks), checks, turns, time.perf_counter() - start, tokens=tokens)


@dataclass
class Evaluation:
    summaries: list[CaseSummary]
    not_run: list[str] = field(default_factory=list)  # cases skipped after hitting the rate limit
    stopped_by: str | None = None


def run_cases(cases: list[Case], agent_factory: Callable, repeat: int = 1, on_result=None) -> Evaluation:
    evaluation = Evaluation([])
    previous_dir = create_chart.OUTPUT_DIR
    # charts drawn during the evaluation are not worth keeping
    with tempfile.TemporaryDirectory() as tmp:
        create_chart.OUTPUT_DIR = Path(tmp)
        try:
            for i, case in enumerate(cases):
                summary = CaseSummary(case.name)
                try:
                    for _ in range(repeat):
                        result = run_case(case, agent_factory)
                        summary.runs.append(result)
                        if on_result:
                            on_result(result)
                except RateLimited as e:
                    # every following call would fail too: stop instead of burning requests
                    evaluation.stopped_by = str(e)[:300]
                    evaluation.not_run = [c.name for c in cases[i:]]
                    if summary.runs:
                        evaluation.summaries.append(summary)
                        evaluation.not_run.remove(case.name)
                    break
                evaluation.summaries.append(summary)
        finally:
            create_chart.OUTPUT_DIR = previous_dir
    return evaluation


def totals(summaries: list[CaseSummary]) -> dict:
    runs = [r for s in summaries for r in s.runs]
    return {
        "tokens": sum(r.tokens for r in runs),
        "runs": len(runs),
        "passed": sum(r.passed for r in runs),
        "pass_rate": sum(r.passed for r in runs) / len(runs) if runs else 0,
        "cases_always_passing": sum(s.pass_rate == 1 for s in summaries),
        "cases": len(summaries),
        "grounded_rate": sum(r.grounded for r in runs if not r.error) / max(1, sum(not r.error for r in runs)),
        "errors": sum(bool(r.error) for r in runs),
        "avg_seconds": sum(r.seconds for r in runs) / len(runs) if runs else 0,
    }


def to_json(evaluation: Evaluation, model: str) -> dict:
    return {
        "model": model,
        "totals": totals(evaluation.summaries),
        "not_run": evaluation.not_run,
        "stopped_by": evaluation.stopped_by,
        "cases": [
            {"case": s.case, "pass_rate": s.pass_rate, "runs": [asdict(r) for r in s.runs]}
            for s in evaluation.summaries
        ],
    }
