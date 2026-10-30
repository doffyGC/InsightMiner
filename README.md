# InsightMiner
An AI-powered data analysis agent that explores datasets, generates insights and answers questions in natural language. Designed to simplify data understanding by combining autonomous agents with interactive analysis.

Built with [Agno](https://github.com/agno-agi/agno) and [Groq](https://groq.com) (`openai/gpt-oss-120b`).

## How it works
The agent has a set of tools over a pandas DataFrame:

| Tool | What it does |
|---|---|
| `load_dataset_tool` | Loads a CSV and returns its shape, columns and types |
| `describe_data_tool` | Returns `df.describe()` statistics |
| `generate_insights_tool` | Profiles every column (numeric, date, categorical, identifier) and finds outliers, dominant categories, date peaks and relationships between columns |
| `query_data_tool` | Runs a pandas expression written by the agent over the **full** dataset |

To answer a question, the agent writes a pandas expression such as
`df[df["Status"] == "CONFIRMADA"].groupby("Tipo de Vacina").size()`, runs it with `query_data_tool`
and answers based on the result, so numbers come from the data instead of being guessed.
Insights work the same way: `generate_insights_tool` computes the facts with pandas and the LLM
only summarizes them, calling `query_data_tool` for anything beyond the profile.

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

The CLI menu lets you load a dataset, describe it, generate insights and ask questions. Sample
datasets are available in `data/`:
- `relatorio-notificacoes.csv`: vaccination notifications (categorical and date columns)
- `vendas.csv`: synthetic sales data (numeric columns, missing values, outliers and duplicated rows)

Example questions:
- *Quantas doses de Covid-19 foram confirmadas ou enviadas?*
- *Qual público alvo tem mais registros e quantos foram recusados?*

> The CSV separator (`;`, `,`, tab or `|`) and encoding (UTF-8 or Latin-1) are detected automatically.

## Tests
```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests cover CSV loading (separator and encoding detection) and the tools. They don't call the LLM,
so they run offline and don't need a Groq API key.

## Project structure
```
app/
├── agents/     # Agno agent definition
├── cli/        # Interactive CLI (entry point)
├── core/       # LLM configuration
├── services/   # DataFrame state
└── tools/      # Tools available to the agent
data/           # Sample datasets
tests/          # pytest suite
```

## Notes
`query_data_tool` evaluates model-generated expressions with restricted builtins. This guards against
mistakes, not malicious input, so it is meant for local use only.
