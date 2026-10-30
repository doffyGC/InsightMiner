import base64
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import app.report as report
import app.report.narrative as narrative
import app.tools.create_chart as create_chart
from app.report.charts import pick_charts
from app.report.narrative import Narrative, NarrativeResult, clean_bullet, interpretations, unverified_numbers
from app.report.render import column_values, localize
from app.tools.generate_insights import profile_dataset
from app.tools.generate_report import generate_report_tool
from app.tools.load_dataset import load_dataset_tool


@pytest.fixture(autouse=True)
def outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(create_chart, "OUTPUT_DIR", tmp_path / "outputs")
    return tmp_path / "outputs"


@pytest.fixture
def fake_narrative(monkeypatch):
    """Replace the LLM writer; records what it was asked to describe."""
    calls = []

    def write(facts, charts, language="en", model=None):
        calls.append(SimpleNamespace(facts=facts, charts=charts, language=language))
        return NarrativeResult(
            summary=["**Sul** leads with **171,951.71**", "<script>alert(1)</script>"],
            captions=[f"caption {i}" for i in range(len(charts))],
        )

    monkeypatch.setattr(report, "write_narrative", write)
    return calls


def vendas() -> pd.DataFrame:
    return pd.read_csv("data/vendas.csv", sep=";")


# --- number verification -------------------------------------------------------------

@pytest.mark.parametrize("token, value", [
    ("171,951.71", 171951.71),
    ("171.951,71", 171951.71),
    ("1 053,38", 1053.38),
    ("0,80", 0.80),
    ("172K", 172_000),
    ("172 mil", 172_000),
    ("1.2M", 1_200_000),
])
def test_interpretations(token, value):
    assert any(abs(v - value) < 1e-6 for v in interpretations(token))


def test_ambiguous_separator_has_both_meanings():
    assert interpretations("1.234") == {1.234, 1234.0}


FACTS = "Sul 171951.71, Norte 23976.53, share 53.6, mean 1053.38, 45%, r 0.80"


@pytest.mark.parametrize("text", [
    "Sul leads with 171,951.71",
    "Sul lidera com 171.951,71",
    "about 172K in the South",
    "cerca de 172 mil",
    "53.6% of the total and 45% of the rows",
    "3 regions in 2025, since 2025-01-01, see Chart 4",
])
def test_grounded_numbers_pass(text):
    assert unverified_numbers([text], FACTS) == []


@pytest.mark.parametrize("text, flagged", [
    ("a total of 195,928.24", ["195,928.24"]),
    ("Norte is 12% of revenue", ["12"]),
    ("27 rows in October", ["27"]),
])
def test_invented_numbers_are_flagged(text, flagged):
    assert unverified_numbers([text], FACTS) == flagged


@pytest.mark.parametrize("text, clean", [
    ("- Sul leads", "Sul leads"),
    ("• Sul leads", "Sul leads"),
    ("1. Sul leads", "Sul leads"),
    ("Sul leads - by far", "Sul leads - by far"),
])
def test_clean_bullet(text, clean):
    assert clean_bullet(text) == clean


class FakeAgent:
    """Stands in for the LLM: answers each run with the next prepared output."""
    outputs = []
    prompts = []

    def __init__(self, **kwargs):
        pass

    def run(self, prompt):
        FakeAgent.prompts.append(prompt)
        output = FakeAgent.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(content=output)


@pytest.fixture
def fake_agent(monkeypatch):
    FakeAgent.outputs, FakeAgent.prompts = [], []
    monkeypatch.setattr(narrative, "Agent", FakeAgent)
    return FakeAgent


def test_narrative_retries_once_without_invented_numbers(fake_agent):
    fake_agent.outputs = [
        Narrative(summary=["Total is 195,928.24"], captions=["c1"]),
        Narrative(summary=["- Sul leads with 171,951.71"], captions=["c1"]),
    ]

    result = narrative.write_narrative(FACTS, [("Chart", "Plotted data")], model=object())

    assert result.summary == ["Sul leads with 171,951.71"]
    assert result.unverified == []
    assert "195,928.24" in fake_agent.prompts[1]


def test_narrative_reports_numbers_still_invented_after_retry(fake_agent):
    fake_agent.outputs = [
        Narrative(summary=["Total is 195,928.24"], captions=[]),
        Narrative(summary=["Total is 195,928.24"], captions=[]),
    ]

    result = narrative.write_narrative(FACTS, [("Chart", "data")], model=object())

    assert result.unverified == ["195,928.24"]
    assert result.captions == [""]  # one caption per chart, even if the model wrote fewer


def test_narrative_failure_is_reported_not_raised(fake_agent):
    fake_agent.outputs = [RuntimeError("Groq is down")]

    result = narrative.write_narrative(FACTS, [], model=object())

    assert result.error == "RuntimeError: Groq is down"
    assert result.summary == []


def test_prompt_asks_for_language_and_number_format(fake_agent):
    fake_agent.outputs = [Narrative(summary=["ok"], captions=[])]

    narrative.write_narrative(FACTS, [], language="pt", model=object())

    assert "Brazilian Portuguese" in fake_agent.prompts[0]
    assert "1.053,38" in fake_agent.prompts[0]


# --- chart selection ------------------------------------------------------------------

def test_picks_charts_from_findings():
    df = vendas()

    plans = pick_charts(df, profile_dataset(df))

    assert [(p.kind, p.x, p.y, p.agg, p.group) for p in plans] == [
        ("line", "data", None, "count", None),            # date column
        ("scatter", "qtd", "receita", "count", None),     # strongest correlation
        ("bar", "regiao", "receita", "mean", None),       # largest difference between group means
        ("bar", "regiao", None, "count", "canal"),        # strongest category concentration
        ("bar", "canal", None, "count", None),            # remaining categorical column
        ("histogram", "receita", None, "count", None),    # skewed numeric column
    ]


def test_no_charts_for_identifiers_only():
    df = pd.DataFrame({"id": range(50), "name": [f"n{i}" for i in range(50)]})

    assert pick_charts(df, profile_dataset(df)) == []


def test_constant_columns_are_not_charted():
    df = pd.DataFrame({"zero": [0] * 40, "value": np.arange(40) * 1.5})

    plans = pick_charts(df, profile_dataset(df))

    assert [p.x for p in plans] == ["value"]


def test_at_most_six_charts():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"c{i}": rng.choice(list("abc"), 60) for i in range(6)}
                      | {"d": pd.date_range("2025-01-01", periods=60).strftime("%Y-%m-%d"),
                         "x": np.arange(60.0), "y": np.arange(60.0) * 2})

    assert len(pick_charts(df, profile_dataset(df))) <= 6


# --- the report -----------------------------------------------------------------------

def test_report_is_a_self_contained_html(outputs, fake_narrative):
    df = vendas()

    result = report.generate_report(df, "data/vendas.csv")

    html = result.path.read_text(encoding="utf-8")
    assert result.path.parent == outputs and result.path.name.startswith("report-vendas-")
    assert "<title>Data report: vendas.csv</title>" in html
    images = re.findall(r"src='data:image/png;base64,([^']+)'", html)
    assert len(images) == len(result.chart_titles) == 6
    assert base64.b64decode(images[0])[:8] == b"\x89PNG\r\n\x1a\n"
    assert not list(outputs.glob("*.png"))  # charts live inside the HTML only
    assert "caption 5" in html


def test_report_escapes_llm_text_and_keeps_bold(fake_narrative):
    html = report.generate_report(vendas(), "vendas.csv").path.read_text(encoding="utf-8")

    assert "<strong>Sul</strong> leads with <strong>171,951.71</strong>" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_report_gives_the_writer_facts_and_plotted_data(fake_narrative):
    report.generate_report(vendas(), "vendas.csv")

    call = fake_narrative[0]
    assert "Strong positive correlation between 'qtd' and 'receita'" in call.facts
    assert any("Plotted data:" in note for _, note in call.charts)
    assert any("Share %" in note for _, note in call.charts)


def test_report_in_portuguese(fake_narrative):
    result = report.generate_report(vendas(), "vendas.csv", language="pt")

    html = result.path.read_text(encoding="utf-8")
    assert '<html lang="pt-BR">' in html and "Relatório de dados: vendas.csv" in html
    assert "Linhas por 'canal'" in result.chart_titles
    assert "<div class='value'>1,1%</div>" in html
    assert fake_narrative[0].language == "pt"


def test_report_warns_about_unverified_numbers(monkeypatch):
    monkeypatch.setattr(report, "write_narrative",
                        lambda *a, **k: NarrativeResult(summary=["x"], captions=[], unverified=["88%"]))

    html = report.generate_report(vendas(), "vendas.csv").path.read_text(encoding="utf-8")

    assert "could not be matched to the computed facts, check them before sharing: 88%" in html


def test_report_survives_llm_failure(monkeypatch):
    monkeypatch.setattr(report, "write_narrative", lambda *a, **k: NarrativeResult(error="RuntimeError: down"))

    html = report.generate_report(vendas(), "vendas.csv").path.read_text(encoding="utf-8")

    assert "The written summary could not be generated (RuntimeError: down)" in html
    assert html.count("data:image/png;base64") == 6


def test_report_rejects_unknown_language():
    with pytest.raises(ValueError, match="Unknown language 'es'"):
        report.generate_report(vendas(), "vendas.csv", language="es")


def test_report_tool(fake_narrative):
    load_dataset_tool("data/vendas.csv")

    result = generate_report_tool()

    assert result.startswith("Report saved to ")
    assert "Summary:\n- **Sul** leads with **171,951.71**" in result


def test_report_tool_without_dataset():
    assert "No dataset loaded" in generate_report_tool()


@pytest.mark.parametrize("series, kind, language, expected", [
    (pd.Series([1500.0, 2_000_000.0]), "numeric", "en", "1,500 – 2M"),
    (pd.Series([12.5, 3000.25]), "numeric", "pt", "12,50 – 3.000"),
    (pd.Series([0, 0, 0]), "numeric", "en", "0"),
    (pd.Series(range(100)), "id", "en", "0 – 99"),
    (pd.Series(["01/02/2025", "15/03/2025"]), "date", "en", "2025-02-01 – 2025-03-15"),
    (pd.Series(list("aabbbcd")), "categorical", "en", "b, a, c, …"),
])
def test_column_values(series, kind, language, expected):
    assert column_values(series, kind, language) == expected


def test_localize():
    assert localize("1,234.5", "pt") == "1.234,5"
    assert localize("1,234.5", "en") == "1,234.5"


@pytest.mark.parametrize("text", ["3 doses de Covid-19", "Covid‑19 (non-breaking hyphen)", "H1N1 and COVID-2019"])
def test_numbers_inside_names_are_not_numbers(text):
    assert unverified_numbers([text], FACTS) == []


def test_whole_numbers_may_be_rounded_facts():
    assert unverified_numbers(["Sul is about 54% of revenue"], FACTS) == []  # 53.6 in the facts
    assert unverified_numbers(["Sul is about 55% of revenue"], FACTS) == ["55"]
    assert unverified_numbers(["Sul is 53.1% of revenue"], FACTS) == ["53.1"]  # decimals must match
