from pathlib import Path

import pandas as pd
import pytest
import app.tools.create_chart as create_chart
from app.tools.create_chart import (
    MISSING, OTHER, OTHER_COLOR, SERIES_COLORS, create_chart_tool, format_number, prepare_bar, prepare_line,
    series_colors,
)
from app.tools.load_dataset import load_dataset_tool


@pytest.fixture(autouse=True)
def charts_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(create_chart, "OUTPUT_DIR", tmp_path / "outputs")
    return tmp_path / "outputs"


def saved_path(result: str) -> Path:
    first_line = result.splitlines()[0]
    assert first_line.startswith("Chart saved to "), result
    return Path(first_line.removeprefix("Chart saved to "))


@pytest.mark.parametrize("value, expected", [
    (3, "3"),
    (2.5, "2.50"),
    (1234.567, "1,235"),
    (12_500, "12.5K"),
    (171_951.71, "172K"),
    (2_000_000, "2M"),
])
def test_format_number(value, expected):
    assert format_number(value) == expected


def test_format_number_forced_compact():
    assert format_number(8000, compact=True) == "8K"


@pytest.mark.parametrize("kwargs", [
    dict(kind="bar", x="regiao", y="receita", agg="sum"),
    dict(kind="bar", x="regiao", group="canal"),
    dict(kind="line", x="data", y="receita", agg="sum", group="regiao"),
    dict(kind="histogram", x="preco"),
    dict(kind="scatter", x="qtd", y="receita", group="regiao", title="Quantidade x receita"),
])
def test_every_kind_saves_a_png(charts_dir, kwargs):
    load_dataset_tool("data/vendas.csv")

    path = saved_path(create_chart_tool(**kwargs))

    assert path.parent == charts_dir
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_returns_plotted_data_for_grounding():
    load_dataset_tool("data/vendas.csv")

    result = create_chart_tool(kind="bar", x="regiao", y="receita", agg="sum")

    assert "Plotted data:" in result
    assert "Sul           171951.71" in result


def test_scatter_reports_correlation():
    load_dataset_tool("data/vendas.csv")

    result = create_chart_tool(kind="scatter", x="qtd", y="receita")

    assert "Points plotted: 305. Correlation between 'qtd' and 'receita': Pearson 0.09, Spearman (rank-based) 0.80." in result
    assert "outliers are distorting Pearson" in result


def test_scatter_without_outliers_has_no_warning(use_df):
    use_df(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.1, 5.9, 8.2, 9.9]}))

    result = create_chart_tool(kind="scatter", x="x", y="y")

    assert "Pearson 1.00, Spearman (rank-based) 1.00." in result
    assert "distorting" not in result


def test_same_chart_twice_does_not_overwrite():
    load_dataset_tool("data/vendas.csv")

    first = saved_path(create_chart_tool(kind="histogram", x="preco"))
    second = saved_path(create_chart_tool(kind="histogram", x="preco"))

    assert first != second
    assert first.exists() and second.exists()


@pytest.mark.parametrize("kwargs, message", [
    (dict(kind="pie", x="regiao"), "Unknown kind 'pie'"),
    (dict(kind="bar", x="regiao", agg="max", y="receita"), "Unknown agg 'max'"),
    (dict(kind="bar", x="inexistente"), "Column 'inexistente' not found"),
    (dict(kind="bar", x="regiao", agg="sum"), "'y' column is required"),
    (dict(kind="bar", x="regiao", y="canal", agg="sum"), "Column 'canal' must be numeric"),
    (dict(kind="histogram", x="regiao"), "Column 'regiao' must be numeric"),
    (dict(kind="line", x="regiao"), "must contain dates or numbers"),
    (dict(kind="line", x="data", time_unit="hour"), "Unknown time_unit 'hour'"),
])
def test_invalid_requests(charts_dir, kwargs, message):
    load_dataset_tool("data/vendas.csv")

    result = create_chart_tool(**kwargs)

    assert result.startswith("Error creating chart") and message in result
    assert not charts_dir.exists() or not any(charts_dir.iterdir())


def test_without_dataset():
    assert "No dataset loaded" in create_chart_tool(kind="bar", x="a")


def test_bar_sorted_with_long_tail_folded_into_other():
    df = pd.DataFrame({"cat": [f"c{i:02d}" for i in range(20) for _ in range(20 - i)]})

    data = prepare_bar(df, "cat", None, "count", None)

    assert len(data) == create_chart.MAX_BARS
    assert data.index[0] == "c00" and data.index[-1] == OTHER
    assert data["Rows"].iloc[:-1].is_monotonic_decreasing
    assert data["Rows"].sum() == len(df)


def test_bar_missing_values_get_their_own_label():
    df = pd.DataFrame({"regiao": ["Sul", "Sul", "Norte"], "canal": ["Loja", None, "Online"]})

    data = prepare_bar(df, "regiao", None, "count", "canal")

    assert MISSING in data.columns
    assert data.loc["Sul", MISSING] == 1


def test_line_fills_empty_periods():
    df = pd.DataFrame({"data": ["01/01/2025", "15/01/2025", "10/04/2025"], "valor": [1, 2, 3]})

    data, unit = prepare_line(df, "data", "valor", "sum", None, "auto")

    assert unit == "month"
    assert data["Sum of valor"].tolist() == [3, 0, 0, 3]


def test_line_picks_time_unit():
    days = pd.date_range("2025-01-01", periods=10, freq="D").strftime("%d/%m/%Y")
    data, unit = prepare_line(pd.DataFrame({"data": days}), "data", None, "count", None, "auto")

    assert unit == "day"
    assert len(data) == 10


def test_colors_follow_fixed_order_and_neutral_for_other():
    assert series_colors(["a", "b", OTHER]) == [SERIES_COLORS[0], SERIES_COLORS[1], OTHER_COLOR]
    assert series_colors(["x", MISSING]) == [SERIES_COLORS[0], OTHER_COLOR]


def test_scatter_groups_capped_at_three(charts_dir):
    df = pd.DataFrame({"x": range(40), "y": range(40), "g": list("abcde") * 8})
    create_chart.dataframe_service.df = df

    labels = create_chart.top_groups(df, "g", create_chart.MAX_SCATTER_GROUPS)

    assert labels.nunique() == 3 and OTHER in set(labels)
    saved_path(create_chart_tool(kind="scatter", x="x", y="y", group="g"))
