from app.agents.data_agent import create_data_agent

def print_menu():
    print("\n📊 === Insight Miner AI ===")
    print("1. Load dataset")
    print("2. Describe dataset")
    print("3. Generate insights")
    print("4. Ask a question")
    print("5. Exit")


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
            print("\n🤖", response.content)

        elif choice == "2":
            response = agent.run("Describe the dataset using describe_data_tool")
            print("\n🤖", response.content)

        elif choice == "3":
            response = agent.run("Generate insights about the dataset")
            print("\n🤖", response.content)

        elif choice == "4":
            question = input("Question: ")
            response = agent.run(f"Answer the following question about the dataset: {question}")
            print("\n🤖", response.content)

        elif choice == "5":
            print("Finishing...")
            break

        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()