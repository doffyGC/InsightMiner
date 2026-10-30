from dataclasses import dataclass

import pandas as pd
from app.tools.generate_insights import Profile

MAX_CHARTS = 6
MAX_CATEGORY_CHARTS = 2


@dataclass
class ChartPlan:
    kind: str
    x: str
    y: str | None = None
    agg: str = "count"
    group: str | None = None
    title_key: str = ""
    subtitle_key: str = ""

    def key(self) -> tuple:
        return self.kind, self.x, self.y, self.agg, self.group

    def title(self, labels: dict) -> str:
        return labels[self.title_key].format(x=self.x, y=self.y, group=self.group)

    def subtitle(self, labels: dict) -> str:
        return labels[self.subtitle_key].format(x=self.x, y=self.y, group=self.group)


def pick_charts(df: pd.DataFrame, profile: Profile) -> list[ChartPlan]:
    """
    Choose the charts worth showing, most informative first. Each rule turns a finding of
    the profile into a chart, so the report is deterministic and only shows what the data supports.
    """
    numeric = [c for c in profile.columns_of("numeric") if df[c].nunique() > 1]
    categorical = profile.columns_of("categorical")
    dates = profile.columns_of("date")
    plans: list[ChartPlan] = []

    # activity over time
    if dates:
        plans.append(ChartPlan("line", dates[0], title_key="over_time", subtitle_key="over_time_sub"))

    # strongest relationship between two numeric columns
    if profile.correlations:
        corr = max(profile.correlations, key=lambda c: c.strength)
        plans.append(ChartPlan("scatter", corr.a, corr.b, title_key="relationship", subtitle_key="relationship_sub"))

    # the measure that differs the most between groups
    if profile.group_means:
        means = max(profile.group_means, key=lambda g: g.ratio)
        plans.append(ChartPlan("bar", means.category, means.numeric, "mean",
                               title_key="mean_by", subtitle_key="mean_by_sub"))

    # categories that combine unevenly
    if profile.concentrations:
        conc = max(profile.concentrations, key=lambda c: c.lift)
        plans.append(ChartPlan("bar", conc.column, group=conc.other,
                               title_key="split_by", subtitle_key="split_by_sub"))

    # distribution of the categories not shown yet, fewest values first (easiest to read)
    shown = {p.x for p in plans}
    for column in sorted((c for c in categorical if c not in shown), key=lambda c: df[c].nunique())[:MAX_CATEGORY_CHARTS]:
        plans.append(ChartPlan("bar", column, title_key="count_by", subtitle_key="count_by_sub"))

    # distribution of a numeric column, preferring one that is skewed or has outliers
    shown |= {p.y for p in plans}
    candidates = [c for c in profile.irregular if c in numeric] + [c for c in numeric if c not in shown]
    if candidates:
        plans.append(ChartPlan("histogram", candidates[0], title_key="distribution", subtitle_key="distribution_sub"))

    unique = {}
    for plan in plans:
        unique.setdefault(plan.key(), plan)
    return list(unique.values())[:MAX_CHARTS]
