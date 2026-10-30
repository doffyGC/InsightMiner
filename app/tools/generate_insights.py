import warnings
from dataclasses import dataclass, field

import pandas as pd
from app.services.dataframe_services import dataframe_service

MAX_CATEGORIES = 20      # above this a text column is treated as free text / identifier
MIN_ROWS_FOR_CORR = 10   # correlations on fewer rows are mostly noise
STRONG_CORR = 0.7
CORR_DISAGREEMENT = 0.3  # Pearson vs Spearman gap that signals outliers distorting Pearson
SMALL_SAMPLE = 30      # below this, relationships are only indicative
MAX_GROUPS = 8         # groups listed when comparing a numeric column across categories
ID_UNIQUENESS = 0.95  # share of distinct values that makes a column look like an identifier
DATE_FORMATS = ["%d/%m/%Y", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"]


@dataclass
class Correlation:
    a: str
    b: str
    pearson: float
    spearman: float

    @property
    def strength(self) -> float:
        return max(abs(self.pearson), abs(self.spearman))


@dataclass
class Concentration:
    """Among rows where `column` == `value`, `other` == `other_value` is `lift` times more common than overall."""
    column: str
    value: str
    other: str
    other_value: str
    lift: float


@dataclass
class GroupMeans:
    """The average of `numeric` changes `ratio` times across the groups of `category`."""
    numeric: str
    category: str
    ratio: float


@dataclass
class Profile:
    """Facts about a dataset: text lines for the LLM plus structured findings for code."""
    rows: int
    overview: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    # column name -> "numeric", "date", "categorical", "id", "text" or "empty"
    kinds: dict[str, str] = field(default_factory=dict)
    correlations: list[Correlation] = field(default_factory=list)
    concentrations: list[Concentration] = field(default_factory=list)
    group_means: list[GroupMeans] = field(default_factory=list)
    irregular: list[str] = field(default_factory=list)  # numeric columns that are skewed or have outliers

    def columns_of(self, kind: str) -> list[str]:
        return [name for name, k in self.kinds.items() if k == kind]

    def text(self) -> str:
        sections = [("Overview", self.overview), ("Columns", self.columns), ("Relationships", self.relationships)]
        return "\n\n".join(
            f"## {title}\n" + "\n".join(f"- {line}" for line in lines)
            for title, lines in sections if lines
        )


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


def find_outliers(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        return s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]
    # most values are equal, so the IQR is 0: fall back to distance from the mean
    return s[(s - s.mean()).abs() > 3 * s.std()] if s.std() > 0 else s.iloc[:0]


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

    outliers = find_outliers(s)
    if len(outliers) > 0:
        lines.append(f"'{name}' has {len(outliers)} outliers ({pct(len(outliers), len(s))}), e.g. {outliers.head(3).tolist()}.")

    if s.nunique() == 1:
        lines.append(f"'{name}' is constant ({s.iloc[0]}) and carries no information.")
    return lines


def is_irregular(series: pd.Series) -> bool:
    s = series.dropna()
    return len(s) > 2 and (abs(s.skew()) > 1 or len(find_outliers(s)) > 0)


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
    if len(s) <= 1 or s.nunique() < ID_UNIQUENESS * len(s):
        return False
    if pd.api.types.is_integer_dtype(s):
        # numeric ids are dense sequences (0..299); distinct integer metrics are spread out
        return s.nunique() >= ID_UNIQUENESS * (s.max() - s.min() + 1)
    return True


def text_insights(name: str, series: pd.Series) -> list[str]:
    s = series.dropna()
    duplicates = s.duplicated().sum()
    if looks_like_id(s):
        repeated = f" ({duplicates} repeated)" if duplicates else ""
        return [f"'{name}' has an almost unique value per row{repeated}, so it is likely an identifier (not useful for grouping)."]
    return [f"'{name}' (text): {s.nunique()} distinct values, {duplicates} repeated."]


def cross_insights(df: pd.DataFrame, profile: Profile) -> None:
    numeric_cols, categorical_cols = profile.columns_of("numeric"), profile.columns_of("categorical")
    lines = profile.relationships

    # strong correlations between numeric columns; Spearman (rank-based) catches
    # relationships that a few outliers hide from Pearson
    if len(numeric_cols) > 1 and len(df) >= MIN_ROWS_FOR_CORR:
        pearson = df[numeric_cols].corr()
        spearman = df[numeric_cols].corr(method="spearman")
        for i, a in enumerate(numeric_cols):
            for b in numeric_cols[i + 1:]:
                corr = Correlation(a, b, pearson.loc[a, b], spearman.loc[a, b])
                if corr.strength >= STRONG_CORR:
                    p, s = corr.pearson, corr.spearman
                    strongest = p if abs(p) >= abs(s) else s
                    direction = "positive" if strongest > 0 else "negative"
                    line = f"Strong {direction} correlation between '{a}' and '{b}' (Pearson {p:.2f}, Spearman {s:.2f})."
                    if abs(p - s) >= CORR_DISAGREEMENT:
                        line += " Outliers distort Pearson here; Spearman reflects most rows."
                    lines.append(line)
                    profile.correlations.append(corr)

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
                profile.concentrations.append(Concentration(a, str(row), b, str(col), float(value)))

    # numeric means that vary a lot across categories
    for cat in categorical_cols:
        for num in numeric_cols:
            means = df.groupby(cat)[num].mean().dropna().sort_values(ascending=False)
            if len(means) > 1 and means.min() > 0 and means.max() / means.min() >= 1.5:
                # list every group (not just the extremes) so the summary never has to guess the others
                groups = ", ".join(f"'{group}' {mean:.2f}" for group, mean in means.head(MAX_GROUPS).items())
                more = f" (+{len(means) - MAX_GROUPS} more groups)" if len(means) > MAX_GROUPS else ""
                lines.append(f"Average '{num}' varies by '{cat}': {groups}{more}.")
                profile.group_means.append(GroupMeans(num, cat, float(means.max() / means.min())))


def profile_dataset(df: pd.DataFrame) -> Profile:
    rows = len(df)
    profile = Profile(rows=rows)
    profile.overview.append(f"The dataset has {rows} rows and {df.shape[1]} columns.")

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        profile.overview.append(f"There are {duplicates} fully duplicated rows ({pct(duplicates, rows)}).")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        profile.overview.append("There are no missing values.")
    else:
        details = ", ".join(f"'{col}' {pct(count, rows)}" for col, count in missing.items())
        profile.overview.append(f"Missing values by column: {details}.")

    for name in df.columns:
        series = df[name]
        if series.isna().all():
            profile.kinds[name] = "empty"
            profile.columns.append(f"'{name}' is completely empty.")
        elif pd.api.types.is_integer_dtype(series) and looks_like_id(series):
            profile.kinds[name] = "id"
            profile.columns.extend(text_insights(name, series))
        elif pd.api.types.is_numeric_dtype(series):
            profile.kinds[name] = "numeric"
            profile.columns.extend(numeric_insights(name, series))
            if is_irregular(series):
                profile.irregular.append(name)
        elif (dates := parse_dates(series)) is not None:
            profile.kinds[name] = "date"
            profile.columns.extend(date_insights(name, dates))
        elif series.nunique() <= MAX_CATEGORIES and series.nunique() < rows:
            profile.kinds[name] = "categorical"
            profile.columns.extend(categorical_insights(name, series))
        else:
            profile.kinds[name] = "id" if looks_like_id(series) else "text"
            profile.columns.extend(text_insights(name, series))

    cross_insights(df, profile)
    if profile.relationships and rows < SMALL_SAMPLE:
        profile.relationships.append(f"Only {rows} rows: these relationships are indicative, not statistically reliable.")
    return profile


def generate_insights_tool() -> str:
    """
    Profile the whole dataset and return factual insights: column types, distributions,
    dominant categories, date ranges and peaks, outliers, missing values, duplicates,
    correlations and relationships between columns.
    Use the result to write a clear summary for the user, and use query_data_tool to dig deeper.
    """
    try:
        df = dataframe_service.get_df()
        if len(df) == 0:
            return "The dataset is empty."
        return profile_dataset(df).text()

    except Exception as e:
        return f"Error generating insights: {str(e)}"
