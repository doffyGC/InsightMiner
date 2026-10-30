import re
from difflib import get_close_matches

import pandas as pd
from app.services.dataframe_services import dataframe_service

MAX_ROWS = 50
MAX_DISTINCT_FOR_HINTS = 1000  # skip free-text columns when suggesting values
STRING_LITERAL = re.compile(r"""(['"])(.*?)\1""")
# only literals compared with data are values: not formats ('%d/%m/%Y') or options (method="spearman")
COMPARED_LITERAL = re.compile(r"""(?:==|!=)\s*(['"])(.*?)\1|(['"])(.*?)\3\s*(?:==|!=)""")
ISIN_LIST = re.compile(r"""\.isin\(\s*[\[(]([^\])]*)[\])]""")

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


def unknown_values(code: str, df: pd.DataFrame) -> list[str]:
    """
    Text literals in `code` that are neither a column nor a value of a text column, with the
    closest real values. A filter on a misspelled value ('Recusado' for 'RECUSADA') silently
    returns nothing, and models tend to answer "none" instead of checking.
    """
    literals = {m.group(2) if m.group(1) else m.group(4) for m in COMPARED_LITERAL.finditer(code)}
    for match in ISIN_LIST.finditer(code):
        literals |= {m.group(2) for m in STRING_LITERAL.finditer(match.group(1))}
    literals -= set(map(str, df.columns))
    if not literals:
        return []
    values = {}  # real value -> column it comes from
    for column in df.select_dtypes(include=["object", "string", "category"]).columns:
        if df[column].nunique() <= MAX_DISTINCT_FOR_HINTS:
            for value in df[column].dropna().astype(str).unique():
                values.setdefault(value, column)

    warnings = []
    for literal in sorted(literals):
        if literal in values or not literal.strip():
            continue
        lowered = {v.lower(): v for v in values}
        close = get_close_matches(literal.lower(), list(lowered), n=3, cutoff=0.6)
        suggestions = ", ".join(f"'{lowered[c]}' (column '{values[lowered[c]]}')" for c in close)
        warnings.append(f"'{literal}' is not a value in the data" + (f"; did you mean {suggestions}?" if close else "."))
    return warnings


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
        df = dataframe_service.get_df()
        result = format_result(evaluate(code, df))
        warnings = unknown_values(code, df)
        if warnings:
            result += "\n\nWarning: " + " ".join(warnings) + " Fix the value and run the query again before answering."
        return result

    except UnsafeExpression:
        return "Error: expression not allowed."
    except Exception as e:
        return f"Error running query: {type(e).__name__}: {str(e)}"
