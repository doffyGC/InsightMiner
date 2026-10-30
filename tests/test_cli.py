from types import SimpleNamespace

import pytest
import app.cli.__main__ as cli


def tool(name: str, result: str):
    return SimpleNamespace(tool_name=name, result=result)


@pytest.fixture
def opened(monkeypatch):
    paths = []
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.os, "startfile", paths.append, raising=False)
    return paths


def test_opens_saved_charts(opened):
    response = SimpleNamespace(tools=[
        tool("query_data_tool", "Chart saved to fake.png"),
        tool("create_chart_tool", "Chart saved to outputs\a.png\nPlotted data:\n..."),
        tool("create_chart_tool", "Error creating chart: Unknown kind 'pie'"),
    ])

    cli.open_charts(response)

    assert opened == ["outputs\a.png"]


def test_no_tools(opened):
    cli.open_charts(SimpleNamespace(tools=None))

    assert opened == []
