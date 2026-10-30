import re
from dataclasses import dataclass, field

from agno.agent import Agent
from pydantic import BaseModel, Field

from app.core.llm import get_llm
from app.report.labels import LANGUAGES

# numbers as written in English (1,234.5) or Portuguese (1.234,5 / 1 234,5), optionally 12.5K / 3 mil;
# not after a letter or hyphen, so the 19 in "Covid-19" is part of a name, not a number
NUMBER = re.compile(
    r"(?<![\w.,\-‐‑])-?\d{1,3}(?:[ ,.  ]\d{3})+(?:[.,]\d+)?(?:\s?(?:[kKmM]\b|mil\b|mi\b))?"
    r"|(?<![\w.,\-‐‑])-?\d+(?:[.,]\d+)?(?:\s?(?:[kKmM]\b|mil\b|mi\b))?"
)
DATE = re.compile(r"\d{4}[-‑]\d{2}(?:[-‑]\d{2})?|\d{1,2}/\d{1,2}/\d{2,4}")
CHART_REF = re.compile(r"\b(?:Chart|Gráfico|Figure|Figura)\s+\d+", re.IGNORECASE)
SMALL_INT = 10  # tiny integers ("3 regions", "2 groups") are everywhere; not worth flagging
NUMBER_FORMAT = {"en": "1,053.38 and 0.80", "pt": "1.053,38 and 0,80"}
TOLERANCE = 0.006  # relative, so rounded values (172K for 171,951.71) still match


class Narrative(BaseModel):
    summary: list[str] = Field(description="3 to 6 bullet points with the most relevant findings")
    captions: list[str] = Field(description="One short caption per chart, in the same order as the charts")


@dataclass
class NarrativeResult:
    summary: list[str] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    error: str | None = None


def interpretations(token: str) -> set[float]:
    """All the values a written number can mean: '1.234' is 1234 in Portuguese and 1.234 in English."""
    text = token.strip().replace(" ", " ").replace(" ", " ")
    multiplier = 1
    if match := re.search(r"\s?(k|mil|m|mi)$", text, re.IGNORECASE):
        multiplier = 1_000 if match.group(1).lower() in ("k", "mil") else 1_000_000
        text = text[:match.start()]
    text = text.replace(" ", "")

    candidates = []
    if "," in text and "." in text:
        decimal = "," if text.rfind(",") > text.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        candidates.append(text.replace(thousands, "").replace(decimal, "."))
    elif "," in text or "." in text:
        sep = "," if "," in text else "."
        candidates.append(text.replace(sep, "."))  # decimal separator
        if re.fullmatch(rf"-?\d{{1,3}}(\{sep}\d{{3}})+", text):
            candidates.append(text.replace(sep, ""))  # thousands separator
    else:
        candidates.append(text)

    values = set()
    for candidate in candidates:
        try:
            values.add(float(candidate) * multiplier)
        except ValueError:
            pass
    return values


def numbers_in(text: str) -> list[str]:
    return NUMBER.findall(CHART_REF.sub(" ", DATE.sub(" ", text)))


def clean_bullet(text: str) -> str:
    """The page adds its own bullets, so drop the ones the model sometimes writes."""
    return re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text).strip()


def unverified_numbers(texts: list[str], facts: str) -> list[str]:
    """
    Numbers in `texts` that don't match (allowing for rounding) any number in `facts`.
    It catches invented numbers, not coincidences: a made-up 88% passes if the facts contain 88.07.
    """
    known = [value for token in numbers_in(facts) for value in interpretations(token)]

    def matches(value: float, rounded: bool) -> bool:
        # a whole number is often a rounded fact ("53%" for 53.5), so it may be off by less than 1
        slack = 0.999 if rounded else 0.051
        return any(abs(value - k) <= max(slack, TOLERANCE * abs(k)) for k in known)

    flagged = []
    for text in texts:
        for token in numbers_in(text):
            values = interpretations(token)
            if all(v.is_integer() and (abs(v) <= SMALL_INT or 1900 <= v <= 2100) for v in values):
                continue
            rounded = all(v.is_integer() for v in values)
            if not any(matches(v, rounded) for v in values) and token.strip() not in flagged:
                flagged.append(token.strip())
    return flagged


def build_prompt(facts: str, charts: list[tuple[str, str]], language: str) -> str:
    chart_list = "\n\n".join(f"Chart {i}: {title}\n{note}" for i, (title, note) in enumerate(charts, 1))
    return f"""Write the text of a data report in {LANGUAGES[language]}.

Rules:
- Use ONLY numbers that appear in the facts or chart data below. You may round them, but never
  compute new numbers (no new totals, percentages, differences or ratios).
- Point out caveats the facts mention (outliers, skew, missing values, small samples).
- Plain sentences, no markdown except **bold** for the key number or category of each point.
- Refer to columns by their exact names.
- Write numbers in the {LANGUAGES[language]} format, like {NUMBER_FORMAT[language]}.
- Don't add currency symbols or units: the data doesn't say what they are.

Write:
- summary: 3 to 6 bullet points with the most relevant findings for someone seeing this data for the first time.
- captions: exactly {len(charts)} captions, one per chart in order, each 1 or 2 sentences saying what the
  chart shows. Each caption sits right under its chart, so don't start with "Chart 1" or repeat the title.

# Facts
{facts}

# Charts
{chart_list}
"""


def write_narrative(facts: str, charts: list[tuple[str, str]], language: str = "en", model=None) -> NarrativeResult:
    """
    Ask the LLM for the summary and captions, then check every number against the facts.
    If some numbers don't match, ask once for a rewrite without them; whatever still doesn't
    match is returned in `unverified` so the report can warn about it.
    """
    agent = Agent(model=model or get_llm(), output_schema=Narrative, retries=2, delay_between_retries=1)
    prompt = build_prompt(facts, charts, language)
    sources = facts + "\n" + "\n".join(note for _, note in charts)

    try:
        narrative = agent.run(prompt).content
        if not isinstance(narrative, Narrative):
            raise ValueError(f"unexpected model output: {str(narrative)[:200]}")
        flagged = unverified_numbers(narrative.summary + narrative.captions, sources)
        if flagged:
            retry = agent.run(
                prompt + f"\n# Correction\nYour previous answer used numbers that are not in the facts: "
                f"{', '.join(flagged)}. Write it again using only numbers that appear above."
            ).content
            if isinstance(retry, Narrative):
                narrative = retry
                flagged = unverified_numbers(narrative.summary + narrative.captions, sources)
    except Exception as e:
        return NarrativeResult(error=f"{type(e).__name__}: {e}"[:300])

    captions = (narrative.captions + [""] * len(charts))[:len(charts)]
    return NarrativeResult(
        summary=[clean_bullet(s) for s in narrative.summary],
        captions=[clean_bullet(c) for c in captions],
        unverified=flagged,
    )
