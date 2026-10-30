import os
import subprocess
import sys

from app.agents.data_agent import create_data_agent

def print_menu():
    print("\n📊 === Insight Miner AI ===")
    print("1. Load dataset")
    print("2. Describe dataset")
    print("3. Generate insights")
    print("4. Ask a question")
    print("5. Create a chart")
    print("6. Exit")


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


def main():
    agent = create_data_agent()

    print("🚀 Insight Miner AI Starting!")

    while True:
        print_menu()
        choice = input("\nChoose an option: ")

        if choice == "1":
            path = input("CSV path: ")

            if not path:
                print("Path cannot be empty.")
                continue

            response = agent.run(f"""
                You must call the tool load_dataset_tool.
                The argument is:
                path = "{path}"
            """)
            show(response)

        elif choice == "2":
            response = agent.run("Describe the dataset using describe_data_tool")
            show(response)

        elif choice == "3":
            response = agent.run("Generate insights about the dataset")
            show(response)

        elif choice == "4":
            question = input("Question: ")
            response = agent.run(f"Answer the following question about the dataset: {question}")
            show(response)

        elif choice == "5":
            request = input("What do you want to see? ")
            response = agent.run(f"Create a chart for this request: {request}")
            show(response)

        elif choice == "6":
            print("Finishing...")
            break

        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()
