import unicodedata
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from app.report.narrative import interpretations, numbers_in, unverified_numbers


@dataclass
class Turn:
    """What the agent did in one message: its answer and the tools it called."""
    answer: str
    tools: list[tuple[str, dict, str]]  # (tool name, arguments, result)

    def tool_text(self) -> str:
        return "\n".join(f"{args}\n{result}" for _, args, result in self.tools)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def normalize(text: str) -> str:
    """Lowercase without accents, so 'Região' matches 'regiao'."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def mentions_number(text: str, expected: float, tolerance: float) -> bool:
    return any(
        abs(value - expected) <= tolerance
        for token in numbers_in(text)
        for value in interpretations(token)
    )


@dataclass
class Numbers:
    """Every expected value appears in the answer, in any format (1,234.5 / 1.234,5 / 1.2K), within `tolerance`."""
    expected: Callable[[pd.DataFrame], list[float]]
    tolerance: float = 0.05
    name: str = "numbers"

    def run(self, df: pd.DataFrame, turns: list[Turn]) -> CheckResult:
        values = [float(v) for v in self.expected(df)]
        missing = [v for v in values if not mentions_number(turns[-1].answer, v, self.tolerance)]
        if missing:
            return CheckResult(self.name, False, f"missing {[round(v, 2) for v in missing]}")
        return CheckResult(self.name, True, f"found {[round(v, 2) for v in values]}")


@dataclass
class AnyOf:
    """Passes if any of `checks` passes: for questions with more than one defensible answer."""
    checks: list
    name: str = "any of"

    def run(self, df: pd.DataFrame, turns: list[Turn]) -> CheckResult:
        results = [check.run(df, turns) for check in self.checks]
        passed = next((r for r in results if r.passed), None)
        if passed:
            return CheckResult(self.name, True, passed.detail)
        return CheckResult(self.name, False, " / ".join(r.detail for r in results))


@dataclass
class Mentions:
    """The answer mentions at least one of `options` (case and accent insensitive)."""
    options: list[str]
    name: str = "mentions"

    def run(self, df: pd.DataFrame, turns: list[Turn]) -> CheckResult:
        answer = normalize(turns[-1].answer)
        found = [o for o in self.options if normalize(o) in answer]
        if found:
            return CheckResult(self.name, True, f"mentions '{found[0]}'")
        return CheckResult(self.name, False, f"none of {self.options}")


@dataclass
class CalledTool:
    """The last message called `tool` with arguments that satisfy `expect` (name -> value or predicate)."""
    tool: str
    expect: dict
    name: str = "tool call"

    def run(self, df: pd.DataFrame, turns: list[Turn]) -> CheckResult:
        calls = [args for name, args, _ in turns[-1].tools if name == self.tool]
        if not calls:
            return CheckResult(self.name, False, f"{self.tool} not called")
        for args in calls:
            wrong = {
                key: args.get(key) for key, want in self.expect.items()
                if not (want(args.get(key)) if callable(want) else args.get(key) == want)
            }
            if not wrong:
                return CheckResult(self.name, True, f"{self.tool}({args})")
        return CheckResult(self.name, False, f"unexpected arguments {wrong} in {calls[-1]}")


@dataclass
class Grounded:
    """Every number in the answer comes from a tool result of the conversation (no invented numbers)."""
    name: str = "grounded"

    def run(self, df: pd.DataFrame, turns: list[Turn]) -> CheckResult:
        # earlier messages count: follow-ups may repeat numbers computed before
        sources = "\n".join(turn.tool_text() for turn in turns)
        flagged = unverified_numbers([turns[-1].answer], sources)
        if flagged:
            return CheckResult(self.name, False, f"not in any tool result: {flagged}")
        return CheckResult(self.name, True)
