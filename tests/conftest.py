import pandas as pd
import pytest
from app.services.dataframe_services import dataframe_service


@pytest.fixture(autouse=True)
def reset_dataset():
    # the service is a module-level singleton, so clear it between tests
    dataframe_service.df = None
    yield
    dataframe_service.df = None


@pytest.fixture
def write_csv(tmp_path):
    """Write raw text to a CSV file and return its path."""
    def write(text: str, name: str = "data.csv", encoding: str = "utf-8") -> str:
        path = tmp_path / name
        path.write_text(text, encoding=encoding, newline="")
        return str(path)
    return write


@pytest.fixture
def use_df():
    """Make a DataFrame the currently loaded dataset."""
    def use(df: pd.DataFrame) -> pd.DataFrame:
        dataframe_service.df = df
        return df
    return use
