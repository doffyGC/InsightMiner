from dataclasses import dataclass, field

from app.evals.checks import AnyOf, CalledTool, Grounded, Mentions, Numbers

VENDAS = "data/vendas.csv"
VACINAS = "data/relatorio-notificacoes.csv"


@dataclass
class Case:
    name: str
    dataset: str
    turns: list[str]          # messages sent in order; checks look at the last answer
    checks: list = field(default_factory=list)
    # every case also checks that the answer has no invented numbers
    grounded: bool = True

    def all_checks(self) -> list:
        return self.checks + ([Grounded()] if self.grounded else [])


def receita_by_regiao(df, canal=None):
    rows = df if canal is None else df[df["canal"] == canal]
    return rows.groupby("regiao")["receita"].sum().tolist()


CASES = [
    # --- plain questions: the answer must contain the right numbers -----------------
    Case("total by group", VENDAS, ["Qual a receita total por região?"],
         [Numbers(receita_by_regiao, tolerance=1)]),
    # rows or distinct order ids: the dataset has duplicated rows, so both answers are defensible
    Case("filtered count", VENDAS, ["Quantos pedidos foram feitos pelo canal Online?"],
         [AnyOf([Numbers(lambda df: [(df["canal"] == "Online").sum()], tolerance=0),
                 Numbers(lambda df: [df.loc[df["canal"] == "Online", "pedido"].nunique()], tolerance=0)])]),
    Case("filtered mean", VENDAS, ["Qual o preço médio no Sudeste?"],
         [Numbers(lambda df: [df.loc[df["regiao"] == "Sudeste", "preco"].mean()])]),
    Case("argmax", VENDAS, ["Qual região tem a maior receita média por pedido?"],
         [Mentions(["Sul"]), Numbers(lambda df: [df.groupby("regiao")["receita"].mean().max()])]),
    Case("median", VENDAS, ["Qual a mediana da receita?"],
         [Numbers(lambda df: [df["receita"].median()])]),
    Case("duplicates", VENDAS, ["Tem linhas duplicadas no dataset? Quantas?"],
         [Numbers(lambda df: [df.duplicated().sum()], tolerance=0)]),
    Case("missing share", VENDAS, ["Qual a porcentagem de valores vazios na coluna canal?"],
         [Numbers(lambda df: [df["canal"].isna().mean() * 100], tolerance=0.6)]),
    Case("in english", VENDAS, ["Which sales channel has the most orders?"],
         [Mentions(["Loja"]), Numbers(lambda df: [df["canal"].value_counts().max()], tolerance=0)]),

    # --- traps: the easy answer is wrong -------------------------------------------
    # Pearson is 0.09 only because of outliers; Spearman (0.80) shows the real relationship
    Case("correlation with outliers", VENDAS, ["Existe relação entre quantidade e receita?"],
         [Numbers(lambda df: [df["qtd"].corr(df["receita"], method="spearman")], tolerance=0.01)]),
    # there is no profit column: any number in the answer would be invented
    Case("missing column", VENDAS, ["Qual o lucro total da empresa?"],
         [Mentions(["lucro", "profit"])]),

    # --- memory: follow-ups depend on the previous messages ------------------------
    Case("follow-up filter", VENDAS, ["Qual a receita total por região?", "E só no canal Loja?"],
         [Numbers(lambda df: receita_by_regiao(df, "Loja"), tolerance=1)]),
    Case("follow-up chart", VENDAS, ["Qual a receita total por região no canal Online?", "Faz um gráfico disso"],
         [CalledTool("create_chart_tool", {
             "kind": "bar", "x": "regiao", "y": "receita", "agg": "sum",
             "filter": lambda f: bool(f) and "Online" in f,
         })]),

    # --- charts --------------------------------------------------------------------
    Case("bar chart", VENDAS, ["Mostra um gráfico da receita total por região"],
         [CalledTool("create_chart_tool", {"kind": "bar", "x": "regiao", "y": "receita", "agg": "sum"})]),
    Case("line chart", VENDAS, ["Como o número de pedidos evoluiu ao longo do tempo? Faz um gráfico"],
         [CalledTool("create_chart_tool", {"kind": "line", "x": "data"})]),
    Case("scatter chart", VENDAS, ["Faz um gráfico da relação entre preço e quantidade"],
         [CalledTool("create_chart_tool", {"kind": "scatter", "x": lambda v: v in ("preco", "qtd"),
                                           "y": lambda v: v in ("preco", "qtd")})]),

    # --- second dataset ------------------------------------------------------------
    Case("count with two filters", VACINAS, ["Quantas doses de Covid-19 foram confirmadas ou enviadas?"],
         [Numbers(lambda df: [((df["Tipo de Vacina"] == "Covid-19")
                               & df["Status"].isin(["CONFIRMADA", "ENVIADA"])).sum()], tolerance=0)]),
    Case("two-part question", VACINAS, ["Qual público alvo tem mais registros e quantos deles foram recusados?"],
         [Mentions(["Infantil"]),
          Numbers(lambda df: [((df["Público Alvo"] == "Infantil") & (df["Status"] == "RECUSADA")).sum()], tolerance=0)]),
    Case("busiest month", VACINAS, ["Em qual mês houve mais doses?"],
         [Mentions(["novembro", "november", "2025-11", "11/2025"])]),
]
