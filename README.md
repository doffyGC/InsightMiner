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
| `generate_report_tool` | Builds a shareable HTML report of the whole dataset (see [Reports](#reports)) |

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
| `/report [pt]` | Generates an HTML report, in English or in Portuguese with `pt` (you can also ask the agent for one) |
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

## Reports
`/report` (or asking the agent for a report) builds a single self-contained HTML file in `outputs/`,
with the charts embedded, ready to send to someone:

- **Key numbers**: rows, columns, missing cells and duplicated rows.
- **Summary** written by the LLM, plus a short caption under each chart.
- **Up to 6 charts**, chosen by rules from what the data shows: activity over time, the strongest
  correlation, the measure that differs the most between groups, categories that combine unevenly,
  category distributions and the distribution of a skewed numeric column. Each one has its data table.
- **Columns table** with type, missing values, distinct values and value ranges.
- **All computed facts** the text is based on.

Facts and charts are computed in code, so the same data always gives the same report. The LLM only
writes the text, and every number it writes is checked against the facts. If it uses a number that
isn't there (a new total or percentage, for example), it is asked to rewrite once; anything still
unmatched is listed in a warning on the report.

Reports are in English by default; `/report pt` writes them in Portuguese.

## Tests
```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests cover CSV loading (separator and encoding detection) and the tools. They don't call the LLM,
so they run offline and don't need a Groq API key.

## Evaluating the agent
The tests above check the code; the evaluation checks the **LLM**: does it get the answers right,
and does it invent numbers? It runs the real agent on questions whose correct answers are computed
with pandas:

```bash
python -m app.evals                                  # all cases with the default model
python -m app.evals --model openai/gpt-oss-20b       # compare another model
python -m app.evals --repeat 3                       # run each case 3 times to measure flakiness
python -m app.evals --case chart                     # only cases whose name contains "chart"
python -m app.evals --list                           # list the cases
```

Each case checks that the right numbers appear in the answer (in any format), that the right category
is named, that charts are called with the right columns and filters, and that every number in the
answer comes from a tool result. There are plain questions, follow-ups that need the conversation
memory, chart requests and traps: a question about a column that doesn't exist, and a correlation
hidden by outliers.

It prints a line per run and a summary (pass rate, answers without invented numbers, time and tokens
per run), and saves every answer and tool call to `outputs/evals/` for inspection. Run it before
switching models: it calls the real API and uses tokens, and if the provider's rate limit is reached
it stops and lists the cases that didn't run instead of counting them as failures.

## Project structure
```
app/
├── agents/     # Agno agent definition
├── cli/        # Interactive CLI (entry point)
├── core/       # LLM configuration
├── evals/      # Agent evaluation: cases with computed answers, checks, runner
├── report/     # HTML report: chart selection, LLM text with number checks, rendering
├── services/   # DataFrame state
└── tools/      # Tools available to the agent
data/           # Sample datasets
outputs/        # Generated charts and reports (not versioned)
tests/          # pytest suite
```

## Notes
`query_data_tool` evaluates model-generated expressions with restricted builtins. This guards against
mistakes, not malicious input, so it is meant for local use only.
