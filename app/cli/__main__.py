import os
import subprocess
import sys

from app.agents.data_agent import create_data_agent
from app.tools.load_dataset import load_dataset_tool

HELP = """
Talk to the agent in natural language, for example:
  - quais insights você encontra nesse dataset?
  - qual a receita total por região?
  - e só no Sul?
  - faz um gráfico disso

Commands:
  /load <path>   load a CSV file
  /new           start a new conversation (keeps the loaded dataset)
  /help          show this message
  /exit          quit
"""


def open_charts(response):
    """Open every chart the agent saved while answering."""
    prefix = "Chart saved to "
    for tool in response.tools or []:
        result = str(tool.result or "")
        if tool.tool_name == "create_chart_tool" and result.startswith(prefix):
            path = result.splitlines()[0].removeprefix(prefix)
            print(f"📈 Chart: {path}")
            if sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", path], check=False)


def show(response):
    print("\n🤖", response.content)
    open_charts(response)


class ChatSession:
    def __init__(self, agent_factory=create_data_agent):
        self.agent_factory = agent_factory
        self.agent = agent_factory()

    def handle(self, line: str) -> bool:
        """Process one line typed by the user. Returns False when the chat should end."""
        line = line.strip()
        if not line:
            return True

        if line.startswith("/"):
            command, _, arg = line.partition(" ")
            arg = arg.strip().strip('"')
            if command in ("/exit", "/quit"):
                return False
            if command == "/help":
                print(HELP)
            elif command == "/new":
                self.agent = self.agent_factory()
                print("🧹 New conversation started.")
            elif command == "/load":
                if not arg:
                    print("Usage: /load <path>")
                else:
                    print(load_dataset_tool(arg))
            else:
                print(f"Unknown command {command}. Type /help to see the commands.")
            return True

        print("⏳ Thinking...")
        try:
            show(self.agent.run(line))
        except KeyboardInterrupt:
            print("\nInterrupted.")
        return True


def main():
    print("🚀 Insight Miner AI")
    session = ChatSession()

    if len(sys.argv) > 1:
        print(load_dataset_tool(sys.argv[1]))
    else:
        print("Load a dataset with /load <path> or just tell me which file to analyze.")
    print("Type /help for examples and commands.")

    while True:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not session.handle(line):
            break
    print("Finishing...")


if __name__ == "__main__":
    main()
