import warnings
import pandas as pd
from app.services.dataframe_services import dataframe_service

MAX_CATEGORIES = 20      # above this a text column is treated as free text / identifier
MIN_ROWS_FOR_CORR = 10   # correlations on fewer rows are mostly noise
STRONG_CORR = 0.7
SMALL_SAMPLE = 30      # below this, relationships are only indicative
MAX_GROUPS = 8         # groups listed when comparing a numeric column across categories
ID_UNIQUENESS = 0.95  # share of distinct values that makes a column look like an identifier
DATE_FORMATS = ["%d/%m/%Y", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"]


def pct(part, total) -> str:
    return f"{part / total:.0%}"


def parse_dates(series: pd.Series):
    """Return the series parsed as dates if at least 90% of its values parse, else None."""
    values = series.dropna().astype(str)
    if values.empty:
        return None
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(values, format=fmt, errors="coerce")
        if parsed.notna().mean() >= 0.9:
            return parsed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(values, dayfirst=True, errors="coerce")
    return parsed if parsed.notna().mean() >= 0.9 else None


def numeric_insights(name: str, series: pd.Series) -> list[str]:
    s = series.dropna()
    if s.empty:
        return []
    lines = [
        f"'{name}' (numeric): mean {s.mean():.2f}, median {s.median():.2f}, "
        f"std {s.std():.2f}, range {s.min():g} to {s.max():g}."
    ]

    skew = s.skew()
    if abs(skew) > 1:
        side = "right (a few high values pull the mean up)" if skew > 0 else "left (a few low values pull the mean down)"
        lines.append(f"'{name}' is skewed to the {side}; prefer the median.")

    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        outliers = s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]
        if len(outliers) > 0:
            lines.append(f"'{name}' has {len(outliers)} outliers ({pct(len(outliers), len(s))}), e.g. {outliers.head(3).tolist()}.")

    if s.nunique() == 1:
        lines.append(f"'{name}' is constant ({s.iloc[0]}) and carries no information.")
    return lines


def date_insights(name: str, dates: pd.Series) -> list[str]:
    d = dates.dropna()
    lines = [f"'{name}' (date): from {d.min():%Y-%m-%d} to {d.max():%Y-%m-%d} ({(d.max() - d.min()).days} days)."]

    by_month = d.dt.to_period("M").value_counts().sort_index()
    if len(by_month) > 1:
        top = by_month.idxmax()
        lines.append(f"'{name}' busiest month: {top} with {by_month.max()} records ({pct(by_month.max(), len(d))}).")

    by_weekday = d.dt.day_name().value_counts()
    if len(by_weekday) > 1:
        lines.append(f"'{name}' most common weekday: {by_weekday.idxmax()} ({by_weekday.max()} records).")
    return lines


def categorical_insights(name: str, series: pd.Series) -> list[str]:
    s = series.dropna()
    counts = s.value_counts()
    top = ", ".join(f"{value} {pct(count, len(s))}" for value, count in counts.head(5).items())
    lines = [f"'{name}' (categorical, {len(counts)} values): {top}."]

    top_share = counts.iloc[0] / len(s)
    if (top_share >= 0.7 and len(counts) > 1) or (top_share > 0.5 and len(counts) > 2):
        lines.append(f"'{name}' is dominated by '{counts.index[0]}' ({pct(counts.iloc[0], len(s))}).")

    rare = counts[counts == 1]
    if 0 < len(rare) < len(counts) and len(s) >= 20:
        lines.append(f"'{name}' has {len(rare)} values that appear only once, e.g. {rare.index[:3].tolist()}.")
    return lines


def looks_like_id(series: pd.Series) -> bool:
    s = series.dropna()
    return len(s) > 1 and s.nunique() >= ID_UNIQUENESS * len(s)


def text_insights(name: str, series: pd.Series) -> list[str]:
    s = series.dropna()
    duplicates = s.duplicated().sum()
    if looks_like_id(s):
        repeated = f" ({duplicates} repeated)" if duplicates else ""
        return [f"'{name}' has an almost unique value per row{repeated}, so it is likely an identifier (not useful for grouping)."]
    return [f"'{name}' (text): {s.nunique()} distinct values, {duplicates} repeated."]


def cross_insights(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    lines = []

    # strong correlations between numeric columns
    if len(numeric_cols) > 1 and len(df) >= MIN_ROWS_FOR_CORR:
        corr = df[numeric_cols].corr()
        for i, a in enumerate(numeric_cols):
            for b in numeric_cols[i + 1:]:
                r = corr.loc[a, b]
                if abs(r) >= STRONG_CORR:
                    direction = "positive" if r > 0 else "negative"
                    lines.append(f"Strong {direction} correlation between '{a}' and '{b}' (r = {r:.2f}).")

    # how each category concentrates the values of the other categorical columns
    for i, a in enumerate(categorical_cols):
        for b in categorical_cols[i + 1:]:
            table = pd.crosstab(df[a], df[b])
            share = table.div(table.sum(axis=1), axis=0)
            overall = df[b].value_counts(normalize=True)
            lift = share / overall
            # only report combinations backed by at least 2 rows
            lift = lift.where(table >= 2)
            if lift.stack().empty:
                continue
            (row, col), value = lift.stack().idxmax(), lift.stack().max()
            if value >= 1.5:
                lines.append(
                    f"Among '{a}' = '{row}', '{b}' = '{col}' is {pct(share.loc[row, col], 1)} "
                    f"vs {pct(overall[col], 1)} overall ({table.loc[row, col]} records)."
                )

    # numeric means that vary a lot across categories
    for cat in categorical_cols:
        for num in numeric_cols:
            means = df.groupby(cat)[num].mean().dropna().sort_values(ascending=False)
            if len(means) > 1 and means.min() > 0 and means.max() / means.min() >= 1.5:
                # list every group (not just the extremes) so the summary never has to guess the others
                groups = ", ".join(f"'{group}' {mean:.2f}" for group, mean in means.head(MAX_GROUPS).items())
                more = f" (+{len(means) - MAX_GROUPS} more groups)" if len(means) > MAX_GROUPS else ""
                lines.append(f"Average '{num}' varies by '{cat}': {groups}{more}.")
    return lines


def generate_insights_tool() -> str:
    """
    Profile the whole dataset and return factual insights: column types, distributions,
    dominant categories, date ranges and peaks, outliers, missing values, duplicates,
    correlations and relationships between columns.
    Use the result to write a clear summary for the user, and use query_data_tool to dig deeper.
    """
    try:
        df = dataframe_service.get_df()
        rows = len(df)
        if rows == 0:
            return "The dataset is empty."

        overview = [f"The dataset has {rows} rows and {df.shape[1]} columns."]

        duplicates = df.duplicated().sum()
        if duplicates > 0:
            overview.append(f"There are {duplicates} fully duplicated rows ({pct(duplicates, rows)}).")

        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            overview.append("There are no missing values.")
        else:
            details = ", ".join(f"'{col}' {pct(count, rows)}" for col, count in missing.items())
            overview.append(f"Missing values by column: {details}.")

        columns, numeric_cols, categorical_cols = [], [], []
        for name in df.columns:
            series = df[name]
            if series.isna().all():
                columns.append(f"'{name}' is completely empty.")
            elif pd.api.types.is_integer_dtype(series) and looks_like_id(series):
                columns.extend(text_insights(name, series))
            elif pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(name)
                columns.extend(numeric_insights(name, series))
            elif (dates := parse_dates(series)) is not None:
                columns.extend(date_insights(name, dates))
            elif series.nunique() <= MAX_CATEGORIES and series.nunique() < rows:
                categorical_cols.append(name)
                columns.extend(categorical_insights(name, series))
            else:
                columns.extend(text_insights(name, series))

        cross = cross_insights(df, numeric_cols, categorical_cols)
        if cross and rows < SMALL_SAMPLE:
            cross.append(f"Only {rows} rows: these relationships are indicative, not statistically reliable.")

        sections = [("Overview", overview), ("Columns", columns), ("Relationships", cross)]
        return "\n\n".join(
            f"## {title}\n" + "\n".join(f"- {line}" for line in lines)
            for title, lines in sections if lines
        )

    except Exception as e:
        return f"Error generating insights: {str(e)}"
