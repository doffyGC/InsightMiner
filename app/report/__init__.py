import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

import app.tools.create_chart as create_chart
from app.report.charts import pick_charts
from app.report.labels import LABELS
from app.report.narrative import NarrativeResult, write_narrative
from app.report.render import render_html
from app.tools.generate_insights import profile_dataset


@dataclass
class Report:
    path: Path
    narrative: NarrativeResult
    chart_titles: list[str]


def report_path(source: str) -> Path:
    create_chart.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^a-z0-9]+", "_", Path(source).stem.lower()).strip("_") or "dataset"
    return create_chart.OUTPUT_DIR / f"report-{stem}-{datetime.now():%Y%m%d-%H%M%S}.html"


def generate_report(df: pd.DataFrame, source: str, language: str = "en", model=None) -> Report:
    """
    Build a self-contained HTML report: computed facts, charts picked from those facts and a
    summary written by the LLM whose numbers are checked against the facts.
    """
    if language not in LABELS:
        raise ValueError(f"Unknown language '{language}'. Use one of {list(LABELS)}.")
    if df.empty:
        raise ValueError("The dataset is empty.")
    labels = LABELS[language]
    profile = profile_dataset(df)

    charts = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, plan in enumerate(pick_charts(df, profile)):
            title = plan.title(labels)
            try:
                chart = create_chart.render_chart(
                    df, plan.kind, plan.x, plan.y, plan.agg, plan.group, title=title,
                    subtitle=plan.subtitle(labels), path=Path(tmp) / f"chart-{i}.png",
                    count_label=labels["count_label"], hidden_outliers_label=labels["hidden_outliers"],
                )
            except create_chart.ChartError:
                continue  # a chart that can't be drawn is skipped, not fatal for the report
            charts.append((title, chart.path, chart.note))

        narrative = write_narrative(profile.text(), [(t, note) for t, _, note in charts], language, model)
        html = render_html(df, source, profile, charts, narrative, labels, language)

    path = report_path(source)
    path.write_text(html, encoding="utf-8")
    return Report(path=path, narrative=narrative, chart_titles=[t for t, _, _ in charts])
