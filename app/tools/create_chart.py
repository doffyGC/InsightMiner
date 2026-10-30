import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render to files only, no GUI window
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd

from app.services.dataframe_services import dataframe_service
from app.tools.generate_insights import parse_dates

OUTPUT_DIR = Path("outputs")
KINDS = ["bar", "line", "histogram", "scatter"]
AGGREGATIONS = ["count", "sum", "mean", "median"]
TIME_UNITS = ["auto", "day", "week", "month", "year"]
MAX_BARS = 15
MAX_TABLE_ROWS = 20

# Validated categorical palette (light mode), always assigned in this order.
# Only the first 3 stay distinguishable when every pair can touch (scatter), so scatter caps groups at 3.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MAX_SCATTER_GROUPS = 3
OTHER = "Other"
MISSING = "(missing)"
OTHER_COLOR = "#898781"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

STYLE = {
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "xtick.color": BASELINE,
    "ytick.color": BASELINE,
    "xtick.labelcolor": TEXT_MUTED,
    "ytick.labelcolor": TEXT_MUTED,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.75,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "legend.labelcolor": TEXT_SECONDARY,
}
FIGSIZE = (9, 5.4)
DPI = 150
BAR_MAX_PX = 24        # bar thickness cap
LINE_WIDTH = 1.5       # points (~2px)
MARKER_SIZE = 6        # points (~8px)
RING_WIDTH = 1.5       # surface ring around dots


class ChartError(ValueError):
    pass


def format_number(value: float, compact: bool | None = None) -> str:
    """Full numbers under 10,000 and compact (12.5K, 1.2M) above, unless `compact` forces a style."""
    if pd.isna(value):
        return ""
    if compact is None:
        compact = abs(value) >= 10_000
    if compact and abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M".replace(".0M", "M")
    if compact and abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K".replace(".0K", "K")
    if float(value).is_integer() or abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def require_column(df: pd.DataFrame, column: str | None, role: str) -> str:
    if not column:
        raise ChartError(f"'{role}' column is required for this chart.")
    if column not in df.columns:
        raise ChartError(f"Column '{column}' not found. Available columns: {list(df.columns)}")
    return column


def require_numeric(df: pd.DataFrame, column: str) -> None:
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ChartError(f"Column '{column}' must be numeric.")


def fold_to_other(values: pd.Series, keep: list) -> pd.Series:
    return values.where(values.isin(keep), OTHER)


def top_groups(df: pd.DataFrame, group: str, limit: int) -> pd.Series:
    """Group labels with everything past the `limit - 1` most frequent folded into 'Other'."""
    labels = df[group].astype(str).where(df[group].notna(), MISSING)
    counts = labels.value_counts()
    if len(counts) <= limit:
        return labels
    return fold_to_other(labels, counts.index[:limit - 1].tolist())


def aggregate(frame: pd.DataFrame, keys: list[str], y: str | None, agg: str) -> pd.Series:
    grouped = frame.groupby(keys, observed=True, sort=False)
    return grouped.size() if agg == "count" else grouped[y].agg(agg)


def describe_measure(y: str | None, agg: str) -> str:
    return "Rows" if agg == "count" else f"{agg.capitalize()} of {y}"


def pick_time_unit(dates: pd.Series, time_unit: str) -> str:
    if time_unit != "auto":
        return time_unit
    span = (dates.max() - dates.min()).days
    if span > 3 * 365:
        return "year"
    if span > 90:
        return "month"
    if span > 21:
        return "week"
    return "day"


def prepare_bar(df, x, y, agg, group) -> pd.DataFrame:
    frame = df.assign(_x=df[x].astype(str).where(df[x].notna(), MISSING))
    if group:
        frame["_group"] = top_groups(df, group, len(SERIES_COLORS))

    # fold the long tail of categories into 'Other', ranked by the plotted measure
    totals = aggregate(frame, ["_x"], y, agg).sort_values(ascending=False)
    if len(totals) > MAX_BARS:
        frame["_x"] = fold_to_other(frame["_x"], totals.index[:MAX_BARS - 1].tolist())

    keys = ["_x", "_group"] if group else ["_x"]
    data = aggregate(frame, keys, y, agg)
    data = data.unstack("_group") if group else data.to_frame(describe_measure(y, agg))

    order = aggregate(frame, ["_x"], y, agg).sort_values(ascending=False).index.tolist()
    if OTHER in order:
        order.remove(OTHER)
        order.append(OTHER)
    data = data.loc[order]
    if group:
        series_order = data.sum().sort_values(ascending=False).index.tolist()
        if OTHER in series_order:
            series_order.remove(OTHER)
            series_order.append(OTHER)
        data = data[series_order]
    data.index.name, data.columns.name = x, group
    return data


def prepare_line(df, x, y, agg, group, time_unit) -> tuple[pd.DataFrame, str]:
    if pd.api.types.is_numeric_dtype(df[x]):
        frame, unit = df.assign(_x=df[x]), None
    else:
        dates = parse_dates(df[x])
        if dates is None:
            raise ChartError(f"Column '{x}' must contain dates or numbers for a line chart.")
        unit = pick_time_unit(dates, time_unit)
        freq = {"day": "D", "week": "W", "month": "M", "year": "Y"}[unit]
        frame = df.loc[dates.index].assign(_x=dates.dt.to_period(freq))

    keys = ["_x"]
    if group:
        frame["_group"] = top_groups(frame, group, len(SERIES_COLORS))
        keys.append("_group")

    data = aggregate(frame, keys, y, agg)
    data = data.unstack("_group") if group else data.to_frame(describe_measure(y, agg))
    data = data.sort_index()

    if unit:
        # show empty periods as zero (count/sum) or gaps (mean/median) instead of skipping them
        full = pd.period_range(data.index.min(), data.index.max(), freq=data.index.freq)
        data = data.reindex(full, fill_value=0 if agg in ("count", "sum") else None)
        if agg in ("count", "sum"):
            data = data.fillna(0)
        data.index = data.index.to_timestamp()
    data.index.name, data.columns.name = x, group
    return data, unit


def style_value_axis(axis, integer: bool = False) -> None:
    axis.set_major_locator(MaxNLocator(nbins=6, integer=integer))
    # one style per axis: compact everywhere once the axis reaches 10,000
    axis.set_major_formatter(FuncFormatter(
        lambda value, _: format_number(value, compact=max(abs(v) for v in axis.get_view_interval()) >= 10_000)
    ))


def add_legend(ax, labels: list[str], colors: list[str], marker: str = "s") -> None:
    handles = [
        plt.Line2D([], [], marker=marker, linestyle="", markersize=8, markerfacecolor=color, markeredgewidth=0)
        for color in colors
    ]
    # figure-level legend, placed under the title by layout_header()
    ax.get_figure().legend(handles, labels, loc="upper left", ncol=min(len(labels), 4),
                           handletextpad=0.3, columnspacing=1.2, borderaxespad=0)


def series_colors(columns) -> list[str]:
    colors = SERIES_COLORS[:len(columns)]
    # 'Other' and missing values are not real categories, so they stay neutral gray
    return [OTHER_COLOR if col in (OTHER, MISSING) else color for col, color in zip(columns, colors)]


def layout_header(fig, title: str, subtitle: str | None) -> None:
    """Stack title, optional subtitle and legend at the top, then fit the plot below them."""
    left, y = 0.06, 0.975
    # plain fig.text, not suptitle: tight_layout would reserve the title's space a second time
    fig.text(left, y, title, ha="left", va="top", fontsize=14, fontweight="semibold", color=TEXT_PRIMARY)
    y -= 0.055
    if subtitle:
        fig.text(left, y + 0.005, subtitle, ha="left", va="top", fontsize=10, color=TEXT_SECONDARY)
        y -= 0.045
    for legend in fig.legends:
        legend.set_bbox_to_anchor((left - 0.01, y), transform=fig.transFigure)
        y -= 0.06
    fig.tight_layout(rect=(0.02, 0.02, 0.98, y))


def bar_thickness(ax, slots: int, horizontal: bool, fig) -> float:
    """Bar thickness in data units, capped at BAR_MAX_PX on screen."""
    bbox = ax.get_window_extent()
    length_px = (bbox.height if horizontal else bbox.width) * 100 / fig.dpi
    return min(0.7, BAR_MAX_PX / (length_px / slots))


def draw_bar(fig, ax, data: pd.DataFrame, integer: bool) -> None:
    labels = [str(label) for label in data.index]
    horizontal = len(labels) > 6 or max(len(label) for label in labels) > 12
    single = data.shape[1] == 1
    positions = range(len(labels))
    colors = series_colors(data.columns)

    thickness = bar_thickness(ax, len(labels), horizontal, fig)
    width = thickness / data.shape[1] if not single else thickness
    for i, (column, color) in enumerate(zip(data.columns, colors)):
        offset = (i - (data.shape[1] - 1) / 2) * width
        values = data[column].fillna(0)
        coords = [p + offset for p in positions]
        if horizontal:
            bars = ax.barh(coords, values, height=width, color=color, edgecolor=SURFACE, linewidth=1)
        else:
            bars = ax.bar(coords, values, width=width, color=color, edgecolor=SURFACE, linewidth=1)
        if single:
            # one series: value at the tip of each bar (also the relief for low-contrast colors)
            ax.bar_label(bars, labels=[format_number(v) for v in values], padding=3,
                         color=TEXT_SECONDARY, fontsize=9)

    value_axis = ax.xaxis if horizontal else ax.yaxis
    style_value_axis(value_axis, integer)
    if horizontal:
        ax.set_yticks(list(positions), labels)
        ax.invert_yaxis()
        ax.grid(axis="x")
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.tick_params(axis="y", length=0, labelcolor=TEXT_SECONDARY)
    else:
        ax.set_xticks(list(positions), labels)
        ax.grid(axis="y")
        ax.tick_params(axis="x", length=0, labelcolor=TEXT_SECONDARY)
    # leave just enough room past the longest bar for its value label
    low, high = min(0, data.min().min()), max(0, data.max().max())
    (ax.set_xlim if horizontal else ax.set_ylim)(low * 1.08, high * 1.08 or 1)
    if not single:
        add_legend(ax, [str(c) for c in data.columns], colors)


def end_labels_fit(data: pd.DataFrame) -> bool:
    """End labels only work when the series end apart from each other; otherwise rely on the legend."""
    if data.shape[1] > 4:
        return False
    ends = sorted(data[c][data[c].last_valid_index()] for c in data.columns if data[c].last_valid_index() is not None)
    span = (data.max().max() - min(0, data.min().min())) or 1
    return all(b - a >= 0.06 * span for a, b in zip(ends, ends[1:]))


def draw_line(ax, data: pd.DataFrame, integer: bool) -> None:
    colors = series_colors(data.columns)
    label_ends = end_labels_fit(data)
    for column, color in zip(data.columns, colors):
        values = data[column]
        ax.plot(data.index, values, color=color, linewidth=LINE_WIDTH,
                solid_capstyle="round", solid_joinstyle="round")
        last = values.last_valid_index()
        if last is not None:
            ax.plot([last], [values[last]], marker="o", markersize=MARKER_SIZE, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=RING_WIDTH)
            if label_ends:
                ax.annotate(format_number(values[last]), (last, values[last]), xytext=(6, 0),
                            textcoords="offset points", va="center", color=TEXT_SECONDARY, fontsize=9)
    style_value_axis(ax.yaxis, integer)
    ax.grid(axis="y")
    ax.set_ylim(bottom=min(0, data.min().min()))
    ax.margins(x=0.06)
    if isinstance(data.index, pd.DatetimeIndex):
        locator = mdates.AutoDateLocator(maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    else:
        style_value_axis(ax.xaxis)
    if data.shape[1] > 1:
        add_legend(ax, [str(c) for c in data.columns], colors, marker="o")


def draw_histogram(ax, values: pd.Series) -> pd.DataFrame:
    counts, edges, _ = ax.hist(values, bins="auto" if values.nunique() > 40 else min(values.nunique(), 20),
                               color=SERIES_COLORS[0], edgecolor=SURFACE, linewidth=1)
    style_value_axis(ax.yaxis, integer=True)
    style_value_axis(ax.xaxis)
    ax.grid(axis="y")
    ranges = [f"{format_number(a)} to {format_number(b)}" for a, b in zip(edges[:-1], edges[1:])]
    return pd.DataFrame({"Rows": counts.astype(int)}, index=pd.Index(ranges, name=values.name))


def draw_scatter(ax, frame: pd.DataFrame, x: str, y: str, group: str | None) -> None:
    if group:
        labels = top_groups(frame, group, MAX_SCATTER_GROUPS)
        names = labels.value_counts().index.tolist()
        if OTHER in names:
            names.remove(OTHER)
            names.append(OTHER)
    else:
        labels, names = pd.Series("all", index=frame.index), ["all"]
    colors = series_colors(names)
    for name, color in zip(names, colors):
        points = frame[labels == name]
        ax.scatter(points[x], points[y], s=MARKER_SIZE ** 2, color=color,
                   edgecolors=SURFACE, linewidths=RING_WIDTH, alpha=0.9)
    style_value_axis(ax.xaxis)
    style_value_axis(ax.yaxis)
    ax.grid(axis="both")
    ax.spines["left"].set_visible(True)
    if group:
        add_legend(ax, names, colors, marker="o")


def describe_correlation(frame: pd.DataFrame, x: str, y: str) -> str:
    pearson = frame[x].corr(frame[y])
    spearman = frame[x].corr(frame[y], method="spearman")
    text = f"Correlation between '{x}' and '{y}': Pearson {pearson:.2f}, Spearman (rank-based) {spearman:.2f}."
    if abs(spearman - pearson) >= 0.3:
        # Pearson is dragged by outliers; the rank-based value reflects most of the points
        text += (" They disagree a lot, so outliers are distorting Pearson: "
                 "use Spearman to judge whether the columns move together.")
    return text


def output_path(kind: str, *columns: str | None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = "-".join(re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in columns if c)
    base = f"{datetime.now():%Y%m%d-%H%M%S}-{kind}-{slug}"
    path, n = OUTPUT_DIR / f"{base}.png", 2
    while path.exists():
        path, n = OUTPUT_DIR / f"{base}-{n}.png", n + 1
    return path


def create_chart_tool(
    kind: str,
    x: str,
    y: str | None = None,
    agg: str = "count",
    group: str | None = None,
    title: str | None = None,
    time_unit: str = "auto",
) -> str:
    """
    Create a chart of the loaded dataset, save it as a PNG and return its path plus the plotted numbers.

    Pick the kind by the question:
    - "bar": compare a measure across the categories of `x` (e.g. revenue by region).
    - "line": how a measure changes over time; `x` is a date (or numeric) column.
    - "histogram": distribution of one numeric column `x`.
    - "scatter": relationship between two numeric columns `x` and `y`.

    args:
        kind (str): One of "bar", "line", "histogram", "scatter".
        x (str): Column for the x axis (categories for bar, dates for line).
        y (str): Numeric column to aggregate (bar/line) or the y axis (scatter). Not needed for count or histogram.
        agg (str): How to aggregate `y` for bar/line: "count", "sum", "mean" or "median". Default "count".
        group (str): Optional categorical column that splits the data into colored series (bar, line, scatter).
        title (str): Optional chart title. Defaults to a description of what is plotted.
        time_unit (str): For line charts over dates: "auto", "day", "week", "month" or "year".
    """
    try:
        df = dataframe_service.get_df()
        if kind not in KINDS:
            raise ChartError(f"Unknown kind '{kind}'. Use one of {KINDS}.")
        if agg not in AGGREGATIONS:
            raise ChartError(f"Unknown agg '{agg}'. Use one of {AGGREGATIONS}.")
        if time_unit not in TIME_UNITS:
            raise ChartError(f"Unknown time_unit '{time_unit}'. Use one of {TIME_UNITS}.")
        require_column(df, x, "x")
        if group:
            require_column(df, group, "group")
        if kind in ("bar", "line") and agg != "count":
            require_numeric(df, require_column(df, y, "y"))
        if kind == "histogram":
            require_numeric(df, x)
        if kind == "scatter":
            require_numeric(df, x)
            require_numeric(df, require_column(df, y, "y"))

        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=FIGSIZE)
            measure = describe_measure(y, agg)
            note = ""
            if kind == "bar":
                data = prepare_bar(df, x, y, agg, group)
                draw_bar(fig, ax, data, integer=agg == "count")
                subtitle = f"{measure} by {x}" + (f", split by {group}" if group else "")
            elif kind == "line":
                data, unit = prepare_line(df, x, y, agg, group, time_unit)
                draw_line(ax, data, integer=agg == "count")
                subtitle = f"{measure} per {unit or x}" + (f", split by {group}" if group else "")
            elif kind == "histogram":
                values = df[x].dropna()
                data = draw_histogram(ax, values)
                ax.set_xlabel(x)
                ax.set_ylabel("Rows")
                subtitle = f"Distribution of {x} ({len(values)} values)"
            else:
                frame = df[[x, y] + ([group] if group else [])].dropna(subset=[x, y])
                draw_scatter(ax, frame, x, y, group)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                subtitle = f"{y} vs {x} ({len(frame)} points)" + (f", colored by {group}" if group else "")
                data = None
                note = f"Points plotted: {len(frame)}. " + describe_correlation(frame, x, y)

            layout_header(fig, title or subtitle, subtitle if title else None)

            path = output_path(kind, x, y, group)
            fig.savefig(path, dpi=DPI, facecolor=SURFACE)
            plt.close(fig)

        if data is not None:
            table = data.round(2).head(MAX_TABLE_ROWS).to_string()
            if len(data) > MAX_TABLE_ROWS:
                table += f"\n... ({len(data)} rows in total)"
            note = f"Plotted data:\n{table}"
        return f"Chart saved to {path}\n{note}"

    except ChartError as e:
        return f"Error creating chart: {e}"
    except Exception as e:
        plt.close("all")
        return f"Error creating chart: {type(e).__name__}: {e}"
