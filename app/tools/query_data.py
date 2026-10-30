import pandas as pd
from app.services.dataframe_services import dataframe_service

MAX_ROWS = 50

# builtins the generated expressions are allowed to use
SAFE_BUILTINS = {
    "len": len, "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
    "sorted": sorted, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool, "range": range,
}


class UnsafeExpression(ValueError):
    pass


def evaluate(code: str, df: pd.DataFrame):
    """Evaluate a pandas expression over `df` with restricted builtins."""
    if "__" in code or "import" in code:
        raise UnsafeExpression("expression not allowed.")
    return eval(code, {"__builtins__": SAFE_BUILTINS, "pd": pd}, {"df": df})


def format_result(result) -> str:
    if isinstance(result, (pd.DataFrame, pd.Series)):
        text = result.head(MAX_ROWS).to_string()
        if len(result) > MAX_ROWS:
            text += f"\n... ({len(result)} rows in total, showing first {MAX_ROWS})"
        return text
    return str(result)


def query_data_tool(code: str) -> str:
    """
    Run a single pandas expression over the FULL loaded dataset and return its result.
    Use this to answer any question that needs counting, filtering, grouping or aggregating.

    The dataset is available as `df` and pandas as `pd`. Use the exact column names
    returned by load_dataset_tool. Only expressions are allowed (no assignments, imports or statements).

    Examples:
        df["Status"].value_counts()
        df[df["Status"] == "CONFIRMADA"].groupby("Tipo de Vacina").size()
        pd.to_datetime(df["Data da Dose"], dayfirst=True).dt.month.value_counts()

    args:
        code (str): A pandas expression using `df`.
    """
    try:
        return format_result(evaluate(code, dataframe_service.get_df()))

    except UnsafeExpression:
        return "Error: expression not allowed."
    except Exception as e:
        return f"Error running query: {type(e).__name__}: {str(e)}"
