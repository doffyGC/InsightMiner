from app.report import generate_report
from app.services.dataframe_services import dataframe_service


def generate_report_tool(language: str = "en") -> str:
    """
    Generate a complete HTML report of the loaded dataset (summary, charts, column table and
    all computed facts) and save it in outputs/. Use it when the user asks for a report or a
    full analysis to share. Takes a while: it draws several charts and writes the summary.

    args:
        language (str): "en" (default, even if the user writes in Portuguese) or "pt" only when
            the user explicitly asks for the report in Portuguese.
    """
    try:
        df = dataframe_service.get_df()
        report = generate_report(df, dataframe_service.path or "dataset", language)
        lines = [f"Report saved to {report.path}", f"Charts: {', '.join(report.chart_titles) or 'none'}"]
        if report.narrative.error:
            lines.append(f"The written summary failed: {report.narrative.error}")
        else:
            lines.append("Summary:\n" + "\n".join(f"- {s}" for s in report.narrative.summary))
        if report.narrative.unverified:
            lines.append(f"Warning, numbers not matched to the facts: {', '.join(report.narrative.unverified)}")
        return "\n".join(lines)

    except Exception as e:
        return f"Error generating report: {type(e).__name__}: {e}"
