from agno.agent import Agent
from app.core.llm import get_llm
from app.tools.load_dataset import load_dataset_tool
from app.tools.describe_data import describe_data_tool
from app.tools.generate_insights import generate_insights_tool
from app.tools.query_data import query_data_tool

def create_data_agent():
    return Agent(
        name="Data Analyst Agent",
        model=get_llm(),
        tools=[
            load_dataset_tool,
            describe_data_tool,
            generate_insights_tool,
            query_data_tool
        ],
        instructions="""
        You are a data analyst agent.
        Your job is to:
        - Analyze datasets
        - Generate insights
        - Answer user questions clearly

        To answer questions about the data, always use query_data_tool with a pandas
        expression over `df`. Never guess numbers: base your answer on the tool result.
        If the tool returns an error, fix the expression and try again.
        """
    )