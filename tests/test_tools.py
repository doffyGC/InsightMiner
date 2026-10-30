import pandas as pd
import pytest
from app.tools.load_dataset import load_dataset_tool
from app.tools.query_data import MAX_ROWS, query_data_tool


def test_load_dataset_returns_columns_and_types():
    result = load_dataset_tool("data/relatorio-notificacoes.csv")

    assert "Shape: (11, 6)" in result
    assert "- Público Alvo (object)" in result


def test_load_dataset_missing_file():
    assert load_dataset_tool("data/nao-existe.csv").startswith("Error loading dataset")


def test_query_uses_full_dataset(use_df):
    use_df(pd.DataFrame({"x": range(305)}))

    assert query_data_tool("len(df)") == "305"
    assert query_data_tool("df['x'].sum()") == str(sum(range(305)))


def test_query_aggregation():
    load_dataset_tool("data/relatorio-notificacoes.csv")

    result = query_data_tool('df[df["Status"] == "CONFIRMADA"].groupby("Tipo de Vacina").size()')

    assert "Covid-19    1" in result
    assert "Tetano      1" in result


def test_query_truncates_long_results(use_df):
    use_df(pd.DataFrame({"x": range(100)}))

    result = query_data_tool("df")

    assert f"(100 rows in total, showing first {MAX_ROWS})" in result


@pytest.mark.parametrize("code", [
    "().__class__.__bases__",
    "__import__('os')",
    "import os",
])
def test_query_blocks_unsafe_expressions(use_df, code):
    use_df(pd.DataFrame({"x": [1]}))

    assert query_data_tool(code) == "Error: expression not allowed."


def test_query_has_no_access_to_other_builtins(use_df):
    use_df(pd.DataFrame({"x": [1]}))

    assert "NameError" in query_data_tool("open('README.md').read()")


def test_query_rejects_statements(use_df):
    use_df(pd.DataFrame({"x": [1]}))

    assert "SyntaxError" in query_data_tool("y = 1")


def test_query_reports_errors(use_df):
    use_df(pd.DataFrame({"x": [1]}))

    assert query_data_tool("df['missing']") == "Error running query: KeyError: 'missing'"


def test_query_without_dataset():
    assert "No dataset loaded" in query_data_tool("len(df)")


@pytest.mark.parametrize("code, warning", [
    ("df[df['Status'] == 'Recusado'].shape[0]", "'Recusado' is not a value in the data; did you mean 'RECUSADA' (column 'Status')?"),
    ("df[df['Tipo de Vacina'] != \"covid\"].shape[0]", "'covid' is not a value in the data; did you mean 'Covid-19' (column 'Tipo de Vacina')?"),
    ("df[df['Status'].isin(['ENVIADA', 'Confirmado'])].shape[0]", "'Confirmado' is not a value in the data; did you mean 'CONFIRMADA'"),
    ("df[df['Status'] == 'xyz'].shape[0]", "'xyz' is not a value in the data. Fix the value"),
])
def test_query_warns_about_values_not_in_the_data(code, warning):
    load_dataset_tool("data/relatorio-notificacoes.csv")

    assert warning in query_data_tool(code)


@pytest.mark.parametrize("code", [
    "df[(df['Status'] == 'CONFIRMADA') | (df['Status'] == 'ENVIADA')].shape[0]",
    "df[df['Status'].isin(['CONFIRMADA', 'ENVIADA'])].shape[0]",
    "pd.to_datetime(df['Data da Dose'], format='%d/%m/%Y').dt.month.value_counts()",
    "df.groupby('Status').size()",
])
def test_query_does_not_warn_about_real_values_formats_or_columns(code):
    load_dataset_tool("data/relatorio-notificacoes.csv")

    assert "Warning" not in query_data_tool(code)
