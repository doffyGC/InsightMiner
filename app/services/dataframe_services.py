import pandas as pd

class DataFrameService:
    def __init__(self):
        self.df = None

    def load_csv(self, path: str):
        self.df = pd.read_csv(path)
        return self.df

    def get_df(self):
        if self.df is None:
            raise ValueError("No dataset loaded.")
        return self.df

dataframe_service = DataFrameService()