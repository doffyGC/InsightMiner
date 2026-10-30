import pytest
from app.services.dataframe_services import dataframe_service, detect_encoding, detect_separator


@pytest.mark.parametrize("sep", [";", ",", "\t", "|"])
def test_detects_each_separator(write_csv, sep):
    text = "\n".join(sep.join(row) for row in [["a", "b", "c"], ["1", "2", "3"], ["4", "5", "6"]]) + "\n"
    path = write_csv(text)

    df = dataframe_service.load_csv(path)

    assert detect_separator(path, "utf-8-sig") == sep
    assert list(df.columns) == ["a", "b", "c"]
    assert df.shape == (2, 3)


def test_semicolon_with_decimal_comma(write_csv):
    path = write_csv("produto;preco;peso\nArroz;1,5;2,3\nFeijão;10,25;1,0\n")

    df = dataframe_service.load_csv(path)

    assert df.shape == (2, 3)
    assert df.loc[0, "preco"] == "1,5"


def test_comma_with_semicolon_inside_quotes(write_csv):
    path = write_csv('nome,obs\n"Ana","gosta; muito"\n"Bia","ok; ok"\n')

    df = dataframe_service.load_csv(path)

    assert df.shape == (2, 2)
    assert df.loc[0, "obs"] == "gosta; muito"


def test_single_column(write_csv):
    df = dataframe_service.load_csv(write_csv("valor\n1\n2\n3\n"))

    assert df.shape == (3, 1)


def test_trailing_blank_line(write_csv):
    df = dataframe_service.load_csv(write_csv("a,b\n1,2\n3,4\n\n"))

    assert df.shape == (2, 2)


def test_latin1_file(write_csv):
    path = write_csv("nome;cidade\nJoão;São Paulo\n", encoding="latin-1")

    df = dataframe_service.load_csv(path)

    assert detect_encoding(path) == "latin-1"
    assert df.loc[0, "cidade"] == "São Paulo"


def test_utf8_bom_is_stripped(write_csv):
    df = dataframe_service.load_csv(write_csv("﻿Nome;Idade\nAna;30\n"))

    assert list(df.columns) == ["Nome", "Idade"]


@pytest.mark.parametrize("path, shape", [
    ("data/relatorio-notificacoes.csv", (11, 6)),
    ("data/vendas.csv", (305, 8)),
])
def test_sample_datasets(path, shape):
    assert dataframe_service.load_csv(path).shape == shape


def test_get_df_without_dataset():
    with pytest.raises(ValueError, match="No dataset loaded"):
        dataframe_service.get_df()
