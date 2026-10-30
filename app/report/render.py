import base64
import html
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.report.narrative import NarrativeResult
from app.tools.create_chart import format_number
from app.tools.generate_insights import Profile, parse_dates

# same tokens as the charts, so page and images read as one piece
CSS = """
:root {
  --surface: #fcfcfb; --page: #f9f9f7; --text: #0b0b0b; --text-2: #52514e; --muted: #898781;
  --hairline: #e1e0d9; --accent: #2a78d6; --warning-bg: #fff4dc; --warning-ink: #6b4a00;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text);
       font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 40px 16px 64px; }
h1 { font-size: 28px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 40px 0 12px; }
.meta { color: var(--text-2); margin: 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 24px; }
.tile { background: var(--surface); border: 1px solid var(--hairline); border-radius: 10px; padding: 14px 16px; }
.tile .label { color: var(--text-2); font-size: 13px; }
.tile .value { font-size: 26px; font-weight: 600; }
.card { background: var(--surface); border: 1px solid var(--hairline); border-radius: 10px; padding: 16px 20px; }
.summary li { margin: 6px 0; }
figure { margin: 0 0 20px; background: var(--surface); border: 1px solid var(--hairline);
         border-radius: 10px; padding: 12px; }
figure img { width: 100%; height: auto; display: block; }
figcaption { color: var(--text-2); padding: 8px 8px 0; }
details { margin: 8px 8px 0; }
summary { cursor: pointer; color: var(--text-2); font-size: 13px; }
pre { overflow-x: auto; font-size: 12px; background: var(--page); padding: 10px; border-radius: 6px; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; background: var(--surface); font-size: 14px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--hairline); vertical-align: top; }
th { color: var(--text-2); font-weight: 600; }
td.num { font-variant-numeric: tabular-nums; }
.note { color: var(--text-2); font-size: 13px; }
.warning { background: var(--warning-bg); color: var(--warning-ink); border-radius: 8px; padding: 10px 14px; }
footer { margin-top: 48px; color: var(--muted); font-size: 13px; }
"""


def rich(text: str) -> str:
    """Escape LLM text and keep only **bold**."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))


def localize(text: str, language: str) -> str:
    """English number punctuation (1,234.5) to Portuguese (1.234,5)."""
    if language != "pt":
        return text
    return text.translate(str.maketrans({",": ".", ".": ","}))


def value_range(low: str, high: str) -> str:
    return low if low == high else f"{low} – {high}"


def column_values(series: pd.Series, kind: str, language: str) -> str:
    s = series.dropna()
    if s.empty:
        return ""
    if kind == "numeric" or (kind == "id" and pd.api.types.is_numeric_dtype(s)):
        return localize(value_range(format_number(s.min()), format_number(s.max())), language)
    if kind == "date" and (dates := parse_dates(s)) is not None:
        return value_range(f"{dates.min():%Y-%m-%d}", f"{dates.max():%Y-%m-%d}")
    top = s.astype(str).value_counts().head(3).index.tolist()
    return ", ".join(top) + (", …" if s.nunique() > 3 else "")


def columns_table(df: pd.DataFrame, profile: Profile, labels: dict, language: str) -> str:
    rows = []
    for name in df.columns:
        series = df[name]
        kind = profile.kinds.get(name, "text")
        rows.append(
            f"<tr><td>{html.escape(str(name))}</td>"
            f"<td>{labels['kinds'].get(kind, '')}</td>"
            f"<td class='num'>{series.isna().mean():.0%}</td>"
            f"<td class='num'>{localize(f'{series.nunique():,}', language)}</td>"
            f"<td>{html.escape(column_values(series, kind, language))}</td></tr>"
        )
    header = "".join(f"<th>{labels[k]}</th>" for k in ("column", "type", "missing", "distinct", "values"))
    return f"<div class='table-wrap'><table><tr>{header}</tr>{''.join(rows)}</table></div>"


def render_html(
    df: pd.DataFrame,
    source: str,
    profile: Profile,
    charts: list[tuple[str, Path, str]],  # (title, png path, plotted data text)
    narrative: NarrativeResult,
    labels: dict,
    language: str,
) -> str:
    rows, cols = df.shape
    missing_cells = df.isna().sum().sum() / df.size if df.size else 0
    tiles = [
        (labels["rows"], localize(f"{rows:,}", language)),
        (labels["columns"], localize(f"{cols:,}", language)),
        (labels["missing_cells"], localize(f"{missing_cells:.1%}", language)),
        (labels["duplicated_rows"], localize(f"{df.duplicated().sum():,}", language)),
    ]
    tiles_html = "".join(
        f"<div class='tile'><div class='label'>{label}</div><div class='value'>{value}</div></div>"
        for label, value in tiles
    )

    if narrative.error:
        summary_html = f"<p class='warning'>{html.escape(labels['summary_unavailable'].format(error=narrative.error))}</p>"
    else:
        summary_html = "<ul class='card summary'>" + "".join(f"<li>{rich(s)}</li>" for s in narrative.summary) + "</ul>"
        if narrative.unverified:
            numbers = ", ".join(narrative.unverified)
            summary_html += f"<p class='warning'>{html.escape(labels['unverified'].format(numbers=numbers))}</p>"
        else:
            summary_html += f"<p class='note'>{labels['verified']}</p>"

    figures = []
    for i, (title, path, note) in enumerate(charts):
        image = base64.b64encode(Path(path).read_bytes()).decode()
        caption = narrative.captions[i] if i < len(narrative.captions) else ""
        figures.append(
            f"<figure><img src='data:image/png;base64,{image}' alt='{html.escape(title, quote=True)}'>"
            + (f"<figcaption>{rich(caption)}</figcaption>" if caption else "")
            + f"<details><summary>{labels['data']}</summary><pre>{html.escape(note)}</pre></details></figure>"
        )
    charts_html = f"<h2>{labels['charts']}</h2>" + "".join(figures) if figures else ""

    facts_html = (
        f"<details class='card'><summary>{labels['facts']}</summary>"
        f"<p class='note'>{labels['facts_hint']}</p><pre>{html.escape(profile.text())}</pre></details>"
    )
    title = f"{labels['report_title']}: {Path(source).name}"
    generated = f"{labels['generated']} {datetime.now():%Y-%m-%d %H:%M} · {localize(f'{rows:,} × {cols:,}', language)}"

    return f"""<!doctype html>
<html lang="{'pt-BR' if language == 'pt' else 'en'}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>{html.escape(title)}</h1>
<p class="meta">{generated}</p>
<div class="tiles">{tiles_html}</div>
<h2>{labels['summary']}</h2>
{summary_html}
{charts_html}
<h2>{labels['columns_section']}</h2>
{columns_table(df, profile, labels, language)}
{facts_html}
<footer>{labels['footer']}</footer>
</main>
</body>
</html>
"""
