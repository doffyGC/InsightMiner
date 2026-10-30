# InsightMiner
An AI-powered data analysis agent that explores datasets, generates insights and answers questions in natural language. Designed to simplify data understanding by combining autonomous agents with interactive analysis.

Built with [Agno](https://github.com/agno-agi/agno) and [Groq](https://groq.com) (`openai/gpt-oss-120b`).

## How it works
The agent has a set of tools over a pandas DataFrame:

| Tool | What it does |
|---|---|
| `load_dataset_tool` | Loads a CSV and returns its shape, columns and types |
| `describe_data_tool` | Returns `df.describe()` statistics |
| `generate_insights_tool` | Returns basic insights (size, means, missing values) |
| `query_data_tool` | Runs a pandas expression written by the agent over the **full** dataset |

To answer a question, the agent writes a pandas expression such as
`df[df["Status"] == "CONFIRMADA"].groupby("Tipo de Vacina").size()`, runs it with `query_data_tool`
and answers based on the result, so numbers come from the data instead of being guessed.

## Setup
Requires Python 3.13+.

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your [Groq API key](https://console.groq.com/keys):

```
GROQ_API_KEY=your_key_here
```

## Usage
```bash
python -m app.cli
```

The CLI menu lets you load a dataset, describe it, generate insights and ask questions. A sample
dataset is available at `data/relatorio-notificacoes.csv`.

Example questions:
- *Quantas doses de Covid-19 foram confirmadas ou enviadas?*
- *Qual público alvo tem mais registros e quantos foram recusados?*

> CSV files are currently read with `;` as the separator.

## Project structure
```
app/
├── agents/     # Agno agent definition
├── cli/        # Interactive CLI (entry point)
├── core/       # LLM configuration
├── services/   # DataFrame state
└── tools/      # Tools available to the agent
data/           # Sample datasets
```

## Notes
`query_data_tool` evaluates model-generated expressions with restricted builtins. This guards against
mistakes, not malicious input, so it is meant for local use only.
