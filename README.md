# InsightMiner
An AI-powered data analysis agent that explores datasets, generates insights, creates charts and answers questions in natural language. Designed to simplify data understanding by combining autonomous agents with interactive analysis.

Built with [Agno](https://github.com/agno-agi/agno) and [Groq](https://groq.com) (`openai/gpt-oss-120b`).

## How it works
The agent has a set of tools over a pandas DataFrame:

| Tool | What it does |
|---|---|
| `load_dataset_tool` | Loads a CSV and returns its shape, columns and types |
| `describe_data_tool` | Returns `df.describe()` statistics |
| `generate_insights_tool` | Profiles every column (numeric, date, categorical, identifier) and finds outliers, dominant categories, date peaks and relationships between columns |
| `query_data_tool` | Runs a pandas expression written by the agent over the **full** dataset |
| `create_chart_tool` | Saves a bar, line, histogram or scatter chart (optionally filtered) as PNG in `outputs/` and returns the plotted numbers |

To answer a question, the agent writes a pandas expression such as
`df[df["Status"] == "CONFIRMADA"].groupby("Tipo de Vacina").size()`, runs it with `query_data_tool`
and answers based on the result, so numbers come from the data instead of being guessed.
Insights work the same way: `generate_insights_tool` computes the facts with pandas and the LLM
only summarizes them, calling `query_data_tool` for anything beyond the profile.

Charts follow the same idea: the agent only picks the chart type, columns and aggregation, and the
tool draws it with a fixed, colorblind-safe style. The tool also returns the numbers it plotted, so the
agent describes the chart from data, not from guessing what the image looks like.

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
python -m app.cli                   # start the chat
python -m app.cli data/vendas.csv   # start the chat with a dataset loaded
```

The CLI is a chat: ask for insights, questions and charts in natural language, in the same
conversation. The agent remembers the last messages, so follow-ups work:

```
> qual a receita total por região?
> e só no canal Online?
> faz um gráfico de barras disso
> agora mostra como isso evoluiu mês a mês
```

Charts are saved in `outputs/` and opened automatically. When a chart is about a subset of the
data, the filter is shown under its title.

| Command | What it does |
|---|---|
| `/load <path>` | Loads a CSV (you can also just ask the agent to load a file) |
| `/new` | Starts a new conversation, keeping the loaded dataset |
| `/help` | Shows examples and commands |
| `/exit` | Quits |

The conversation is kept in memory, like the loaded dataset, so both end when the CLI closes.

Sample datasets are available in `data/`:
- `relatorio-notificacoes.csv`: vaccination notifications (categorical and date columns)
- `vendas.csv`: synthetic sales data (numeric columns, missing values, outliers and duplicated rows)

More examples:
- *Quantas doses de Covid-19 foram confirmadas ou enviadas?* (`relatorio-notificacoes.csv`)
- *Como a receita evoluiu ao longo do ano em cada região?* (`vendas.csv`)
- *Existe relação entre quantidade e receita?* (`vendas.csv`)

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
outputs/        # Generated charts (not versioned)
tests/          # pytest suite
```

## Notes
`query_data_tool` evaluates model-generated expressions with restricted builtins. This guards against
mistakes, not malicious input, so it is meant for local use only.
