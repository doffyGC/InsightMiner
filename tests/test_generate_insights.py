import numpy as np
import pandas as pd
from app.tools.generate_insights import generate_insights_tool
from app.tools.load_dataset import load_dataset_tool


def insights_for(use_df, df: pd.DataFrame) -> str:
    use_df(df)
    return generate_insights_tool()


def test_empty_dataset(use_df):
    assert insights_for(use_df, pd.DataFrame({"a": []})) == "The dataset is empty."


def test_without_dataset():
    assert "No dataset loaded" in generate_insights_tool()


def test_overview_missing_and_duplicates(use_df):
    df = pd.DataFrame({"a": [1, 1, 2, None], "b": ["x", "x", "y", "z"]})

    result = insights_for(use_df, df)

    assert "The dataset has 4 rows and 2 columns." in result
    assert "There are 1 fully duplicated rows (25%)." in result
    assert "Missing values by column: 'a' 25%." in result


def test_no_missing_values(use_df):
    assert "There are no missing values." in insights_for(use_df, pd.DataFrame({"a": [1, 2]}))


def test_numeric_stats_skew_and_outliers(use_df):
    df = pd.DataFrame({"valor": [10] * 18 + [11, 1000]})

    result = insights_for(use_df, df)

    assert "'valor' (numeric): mean 59.55, median 10.00" in result
    assert "'valor' is skewed to the right" in result
    assert "'valor' has 1 outliers (5%), e.g. [1000]" in result


def test_outliers_with_spread_values(use_df):
    df = pd.DataFrame({"valor": [10.0, 12.0, 11.0, 13.0, 9.0, 10.5, 11.5, 12.5, 95.0]})

    assert "'valor' has 1 outliers (11%), e.g. [95.0]" in insights_for(use_df, df)


def test_constant_column(use_df):
    result = insights_for(use_df, pd.DataFrame({"desconto": [0, 0, 0], "x": [1, 2, 2]}))

    assert "'desconto' is constant (0)" in result


def test_identifiers_are_not_treated_as_metrics(use_df):
    df = pd.DataFrame({"id": list(range(40)) + [0], "nome": [f"P{i}" for i in range(41)], "x": [1, 2] * 20 + [1]})

    result = insights_for(use_df, df)

    assert "'id' has an almost unique value per row (1 repeated), so it is likely an identifier" in result
    assert "'nome' has an almost unique value per row" in result
    assert "'id' (numeric)" not in result


def test_distinct_integers_spread_out_are_metrics(use_df):
    salarios = [1500 + 137 * i for i in range(30)]

    result = insights_for(use_df, pd.DataFrame({"salario": salarios}))

    assert "'salario' (numeric)" in result
    assert "identifier" not in result


def test_date_column(use_df):
    df = pd.DataFrame({"data": ["01/01/2025", "15/01/2025", "20/01/2025", "10/02/2025"]})

    result = insights_for(use_df, df)

    assert "'data' (date): from 2025-01-01 to 2025-02-10 (40 days)." in result
    assert "'data' busiest month: 2025-01 with 3 records (75%)." in result


def test_categorical_distribution_and_dominance(use_df):
    df = pd.DataFrame({
        "status": ["A"] * 6 + ["B"] * 2 + ["C"] * 2,   # 60% of 3 values: dominated
        "canal": ["Loja"] * 6 + ["Online"] * 4,         # 60% of 2 values: not dominated
    })

    result = insights_for(use_df, df)

    assert "'status' (categorical, 3 values): A 60%, B 20%, C 20%." in result
    assert "'status' is dominated by 'A' (60%)." in result
    assert "'canal' is dominated" not in result


def test_strong_correlation(use_df):
    x = np.arange(20) * 3.5
    df = pd.DataFrame({"x": x, "y": x * 2 + 1.0, "z": [5.0, 1.0] * 10})

    result = insights_for(use_df, df)

    assert "Strong positive correlation between 'x' and 'y' (r = 1.00)." in result
    assert "'z'" not in result.split("## Relationships")[-1]


def test_no_correlation_on_few_rows(use_df):
    df = pd.DataFrame({"x": [1.5, 2.5, 3.5], "y": [2.0, 4.0, 6.0]})

    assert "correlation" not in insights_for(use_df, df)


def test_category_concentration(use_df):
    df = pd.DataFrame({
        "regiao": ["Norte"] * 10 + ["Sul"] * 10,
        "canal": ["Loja"] * 10 + ["Online"] * 10,
    })

    result = insights_for(use_df, df)

    assert "Among 'regiao' = 'Norte', 'canal' = 'Loja' is 100% vs 50% overall (10 records)." in result


def test_group_means_list_every_group(use_df):
    df = pd.DataFrame({
        "regiao": ["Sudeste", "Sul", "Norte"] * 4,
        "preco": [80.0, 40.0, 30.0] * 4,
    })

    result = insights_for(use_df, df)

    assert "Average 'preco' varies by 'regiao': 'Sudeste' 80.00, 'Sul' 40.00, 'Norte' 30.00." in result


def test_small_sample_caveat():
    load_dataset_tool("data/relatorio-notificacoes.csv")

    result = generate_insights_tool()

    assert "Only 11 rows: these relationships are indicative" in result


def test_sales_sample_finds_planted_patterns():
    load_dataset_tool("data/vendas.csv")

    result = generate_insights_tool()

    assert "There are 5 fully duplicated rows" in result
    assert "'pedido' has an almost unique value per row" in result
    assert "'receita' is skewed to the right" in result
    assert "'desconto' is constant (0)" in result
    assert "Among 'regiao' = 'Norte', 'canal' = 'Loja' is 96%" in result
    assert "Only" not in result
