from types import SimpleNamespace

import pandas as pd
import pytest

from app.evals.cases import CASES, Case
from app.evals.checks import AnyOf, CalledTool, Grounded, Mentions, Numbers, Turn
from app.evals.runner import run_case, run_cases, totals

DF = pd.DataFrame({"regiao": ["Sul", "Sul", "Norte"], "receita": [100.0, 50.5, 20.0]})


def turn(answer, *tools):
    return Turn(answer=answer, tools=list(tools))


# --- checks ---------------------------------------------------------------------------

@pytest.mark.parametrize("answer", [
    "Sul: 150.5 e Norte: 20",
    "Sul soma 150,50; Norte, 20,00",
    "Norte 20 | Sul 150.50",
])
def test_numbers_accepts_any_format_and_order(answer):
    check = Numbers(lambda df: df.groupby("regiao")["receita"].sum().tolist())

    assert check.run(DF, [turn(answer)]).passed


def test_numbers_reports_what_is_missing():
    result = Numbers(lambda df: [150.5, 20]).run(DF, [turn("Sul: 150.5")])

    assert not result.passed and result.detail == "missing [20.0]"


def test_numbers_tolerance():
    assert Numbers(lambda df: [9.18], tolerance=0.6).run(DF, [turn("9%")]).passed
    assert not Numbers(lambda df: [9.18], tolerance=0.1).run(DF, [turn("9%")]).passed


def test_numbers_does_not_match_digits_inside_other_numbers():
    assert not Numbers(lambda df: [1], tolerance=0).run(DF, [turn("there are 11 rows")]).passed


def test_mentions_ignores_case_and_accents():
    assert Mentions(["Região Sul"]).run(DF, [turn("a REGIAO SUL lidera")]).passed
    assert not Mentions(["Norte"]).run(DF, [turn("a Sul lidera")]).passed


def test_checks_look_at_the_last_answer():
    turns = [turn("Sul 150.5"), turn("só no Norte: nada")]

    assert not Numbers(lambda df: [150.5]).run(DF, turns).passed


CHART = ("create_chart_tool", {"kind": "bar", "x": "regiao", "filter": "df['canal'] == 'Online'"}, "Chart saved to x.png")


def test_called_tool_with_values_and_predicates():
    check = CalledTool("create_chart_tool", {"kind": "bar", "filter": lambda f: "Online" in f})

    assert check.run(DF, [turn("ok", CHART)]).passed


def test_called_tool_wrong_arguments():
    result = CalledTool("create_chart_tool", {"kind": "line"}).run(DF, [turn("ok", CHART)])

    assert not result.passed and "unexpected arguments {'kind': 'bar'}" in result.detail


def test_called_tool_not_called():
    result = CalledTool("create_chart_tool", {}).run(DF, [turn("here is your chart")])

    assert not result.passed and result.detail == "create_chart_tool not called"


def test_grounded_accepts_numbers_from_any_previous_tool_result():
    first = turn("Sul 150.5", ("query_data_tool", {}, "Sul 150.5\nNorte 20.0"))
    follow_up = turn("Como eu disse, o Sul tem 150,50")

    assert Grounded().run(DF, [first, follow_up]).passed


def test_grounded_flags_invented_numbers():
    answer = turn("Sul é 88,3% do total", ("query_data_tool", {}, "Sul 150.5\nNorte 20.0"))

    result = Grounded().run(DF, [answer])

    assert not result.passed and "88,3" in result.detail


# --- runner ---------------------------------------------------------------------------

class FakeAgent:
    """Answers from a script: each message gets the next (answer, tools, status)."""

    def __init__(self, script):
        self.script = list(script)
        self.messages = []

    def run(self, message):
        self.messages.append(message)
        answer, tools, status = self.script.pop(0)
        return SimpleNamespace(
            content=answer, status=status, metrics=SimpleNamespace(total_tokens=100),
            tools=[SimpleNamespace(tool_name=n, tool_args=a, result=r) for n, a, r in tools],
        )


CASE = Case("sum", "data/vendas.csv", ["receita total?", "e no Sul?"],
            [Numbers(lambda df: [df.loc[df["regiao"] == "Sul", "receita"].sum()], tolerance=1)])
SUL = pd.read_csv("data/vendas.csv", sep=";").query("regiao == 'Sul'")["receita"].sum()


def test_run_case_passes_and_counts_tokens():
    agent = FakeAgent([
        ("total 321281", [("query_data_tool", {}, "321281.0")], "COMPLETED"),
        (f"Sul: {SUL:.2f}", [("query_data_tool", {}, f"{SUL:.2f}")], "COMPLETED"),
    ])

    result = run_case(CASE, lambda: agent)

    assert result.passed and result.tokens == 200
    assert agent.messages == CASE.turns


def test_run_case_fails_on_invented_number():
    agent = FakeAgent([("x", [], "COMPLETED"), (f"Sul: {SUL:.2f}", [], "COMPLETED")])

    result = run_case(CASE, lambda: agent)

    assert not result.passed
    assert [c.name for c in result.checks if not c.passed] == ["grounded"]


def test_provider_error_is_an_error_not_a_wrong_answer():
    agent = FakeAgent([('{"error": "service unavailable"}', [], "RunStatus.ERROR")])

    result = run_case(CASE, lambda: agent)

    assert result.error.startswith("ProviderError") and result.checks == []


def test_rate_limit_stops_the_evaluation():
    rate_limited = ('{"error":{"code":"rate_limit_exceeded"}}', [], "ERROR")
    agents = iter([
        FakeAgent([("total", [], "COMPLETED"), (f"Sul: {SUL:.2f}", [("q", {}, f"{SUL:.2f}")], "COMPLETED")]),
        FakeAgent([rate_limited]),
    ])
    other = Case("other", "data/vendas.csv", ["?"])

    evaluation = run_cases([CASE, other, other], lambda: next(agents))

    assert [s.case for s in evaluation.summaries] == ["sum"]
    assert evaluation.not_run == ["other", "other"]
    assert "rate_limit_exceeded" in evaluation.stopped_by
    assert totals(evaluation.summaries)["pass_rate"] == 1


def test_case_names_are_unique():
    names = [c.name for c in CASES]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_expected_answers_compute(case):
    """Every expected value can be computed from its dataset (catches typos in column names)."""
    df = pd.read_csv(case.dataset, sep=";")
    for check in case.checks:
        if isinstance(check, Numbers):
            assert all(pd.notna(v) for v in check.expected(df))


def test_any_of_accepts_either_answer():
    check = AnyOf([Numbers(lambda df: [119], tolerance=0), Numbers(lambda df: [116], tolerance=0)])

    assert check.run(DF, [turn("116 pedidos")]).passed
    assert check.run(DF, [turn("119 pedidos")]).passed
    assert not check.run(DF, [turn("120 pedidos")]).passed
