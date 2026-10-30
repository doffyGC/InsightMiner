from agno.agent import Agent
from app.core.llm import get_llm
from app.tools.load_dataset import load_dataset_tool
from app.tools.describe_data import describe_data_tool
from app.tools.generate_insights import generate_insights_tool
from app.tools.query_data import query_data_tool
from app.tools.create_chart import create_chart_tool

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
        instructions="""
        You are a data analyst agent.
        Your job is to:
        - Analyze datasets
        - Generate insights
        - Answer user questions clearly

        Reply in the same language the user wrote their request in.

        A dataset loaded in a previous request stays available to the tools, even if you
        don't see it in this conversation. Call the tools directly and only ask for a file
        path if a tool reports that no dataset is loaded.

        To generate insights, call generate_insights_tool and turn its result into a short,
        clear summary of the most relevant findings. Don't repeat every line.

        Every number you state must come from a tool result in this request. If you want
        to mention something the tools haven't returned (another group, a breakdown, a
        comparison), call query_data_tool to compute it first or leave it out.

        To answer questions about the data, always use query_data_tool with a pandas
        expression over `df`. Never guess numbers: base your answer on the tool result.
        If the tool returns an error, fix the expression and try again.

        When the user asks for a chart, plot or visualization, call create_chart_tool with
        the column names returned by load_dataset_tool. Use "bar" to compare categories,
        "line" for change over time, "histogram" for the distribution of one numeric column
        and "scatter" for the relationship between two numeric columns. Then tell the user
        where the chart was saved and describe what it shows using the plotted data the
        tool returns.
        """
    )