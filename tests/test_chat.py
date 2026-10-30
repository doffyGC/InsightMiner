from types import SimpleNamespace

import pytest
import app.cli.__main__ as cli
from app.agents.data_agent import HISTORY_RUNS, build_instructions, create_data_agent, dataset_context
from app.services.dataframe_services import dataframe_service
from app.tools.load_dataset import load_dataset_tool


class FakeAgent:
    def __init__(self):
        self.messages = []

    def run(self, message):
        self.messages.append(message)
        return SimpleNamespace(content=f"answer to {message}", tools=[])


@pytest.fixture
def session():
    agents = []

    def factory():
        agents.append(FakeAgent())
        return agents[-1]

    chat = cli.ChatSession(agent_factory=factory)
    chat.agents = agents
    return chat


def test_dataset_context_without_dataset():
    assert "No dataset is loaded yet" in dataset_context()


def test_dataset_context_follows_loaded_dataset():
    load_dataset_tool("data/vendas.csv")

    context = dataset_context()

    assert "Currently loaded dataset: data/vendas.csv (305 rows, 8 columns)." in context
    assert "- receita (float64)" in context
    assert context in build_instructions()


def test_agent_keeps_conversation_history():
    agent = create_data_agent()

    assert agent.add_history_to_context and agent.num_history_runs == HISTORY_RUNS
    assert agent.db is not None and agent.session_id
    assert create_data_agent().session_id != agent.session_id
    # instructions are rebuilt on every message so the dataset info is never stale
    assert callable(agent.instructions)


def test_plain_text_goes_to_the_agent(session, capsys):
    assert session.handle("qual a receita total?") is True

    assert session.agent.messages == ["qual a receita total?"]
    assert "answer to qual a receita total?" in capsys.readouterr().out


def test_follow_ups_go_to_the_same_agent(session):
    session.handle("receita por região")
    session.handle("e só no Sul?")

    assert len(session.agents) == 1
    assert session.agent.messages == ["receita por região", "e só no Sul?"]


def test_new_starts_a_fresh_conversation_and_keeps_dataset(session):
    load_dataset_tool("data/vendas.csv")
    session.handle("receita por região")

    session.handle("/new")

    assert len(session.agents) == 2 and session.agent.messages == []
    assert dataframe_service.path == "data/vendas.csv"


def test_load_command_loads_without_the_agent(session, capsys):
    session.handle('/load "data/vendas.csv"')

    assert dataframe_service.path == "data/vendas.csv"
    assert session.agent.messages == []
    assert "Dataset loaded! Shape: (305, 8)" in capsys.readouterr().out


def test_load_without_path(session, capsys):
    session.handle("/load")

    assert "Usage: /load <path>" in capsys.readouterr().out


@pytest.mark.parametrize("line", ["/exit", "/quit", "  /exit  "])
def test_exit(session, line):
    assert session.handle(line) is False


def test_blank_line_is_ignored(session):
    assert session.handle("   ") is True
    assert session.agent.messages == []


def test_unknown_command(session, capsys):
    session.handle("/foo")

    assert "Unknown command /foo" in capsys.readouterr().out
    assert session.agent.messages == []


def test_help(session, capsys):
    session.handle("/help")

    assert "/load <path>" in capsys.readouterr().out
