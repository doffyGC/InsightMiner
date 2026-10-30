from uuid import uuid4

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from app.core.llm import get_llm
from app.services.dataframe_services import dataframe_service
from app.tools.load_dataset import load_dataset_tool
from app.tools.describe_data import describe_data_tool
from app.tools.generate_insights import generate_insights_tool
from app.tools.query_data import query_data_tool
from app.tools.create_chart import create_chart_tool

# previous exchanges kept in the context, so follow-ups like "and in the South?" work
HISTORY_RUNS = 6
HISTORY_TOOL_CALLS = 6  # tool results (profiles, tables) are long, keep only the latest ones

INSTRUCTIONS = """
You are a data analyst agent chatting with a user about a dataset.
Your job is to:
- Analyze datasets
- Generate insights
- Answer user questions clearly

Reply in the same language the user wrote their request in.

This is a conversation: use the previous messages to understand follow-ups such as
"and in the South?" or "now make a chart of that".

To generate insights, call generate_insights_tool and turn its result into a short,
clear summary of the most relevant findings. Don't repeat every line.

Every number you state must come from a tool result in this conversation. If you want
to mention something the tools haven't returned (another group, a breakdown, a
comparison), call query_data_tool to compute it first or leave it out.

To answer questions about the data, always use query_data_tool with a pandas
expression over `df`. Never guess numbers: base your answer on the tool result.
If the tool returns an error, fix the expression and try again.

When the user asks for a chart, plot or visualization, call create_chart_tool with
the dataset's column names. Use "bar" to compare categories, "line" for change over
time, "histogram" for the distribution of one numeric column and "scatter" for the
relationship between two numeric columns. Then tell the user where the chart was
saved and describe what it shows using the plotted data the tool returns.
"""


def dataset_context() -> str:
    """What is loaded right now. Rebuilt on every message, so it is never stale."""
    if dataframe_service.df is None:
        return (
            "No dataset is loaded yet. If the user asks about data, ask for the CSV path "
            "and load it with load_dataset_tool."
        )
    df = dataframe_service.df
    return (
        f"Currently loaded dataset: {dataframe_service.path} ({df.shape[0]} rows, {df.shape[1]} columns).\n"
        f"Columns:\n{dataframe_service.schema()}\n"
        "Use these exact column names. Load another file only if the user asks for it."
    )


def build_instructions() -> str:
    return f"{INSTRUCTIONS}\n{dataset_context()}"


def create_data_agent():
    return Agent(
        name="Data Analyst Agent",
        model=get_llm(),
        # Groq sometimes fails to parse gpt-oss tool calls (400 output_parse_failed);
        # the model layer won't retry 400s, so retry the whole run instead
        retries=2,
        delay_between_retries=1,
        tools=[
            load_dataset_tool,
            describe_data_tool,
            generate_insights_tool,
            query_data_tool,
            create_chart_tool
        ],
        instructions=build_instructions,
        # the conversation lives in memory, like the loaded dataset: both end with the CLI
        db=InMemoryDb(),
        session_id=str(uuid4()),
        add_history_to_context=True,
        num_history_runs=HISTORY_RUNS,
        max_tool_calls_from_history=HISTORY_TOOL_CALLS,
    )
