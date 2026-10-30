import csv
import pandas as pd

CANDIDATE_SEPARATORS = [";", ",", "\t", "|"]
ENCODINGS = ["utf-8-sig", "latin-1"]  # latin-1 covers CSVs exported by Excel in Portuguese
SAMPLE_LINES = 50


def detect_encoding(path: str) -> str:
    for encoding in ENCODINGS:
        try:
            with open(path, encoding=encoding) as f:
                f.read()
            return encoding
        except UnicodeDecodeError:
            continue
    return ENCODINGS[-1]


def detect_separator(path: str, encoding: str) -> str:
    """
    Pick the separator that splits the first lines into the same number of columns (more than one).
    Ties go to the one producing more columns, then to the order in CANDIDATE_SEPARATORS.
    """
    with open(path, encoding=encoding, newline="") as f:
        sample = [line for _, line in zip(range(SAMPLE_LINES), f) if line.strip()]
    if not sample:
        return ","

    best, best_columns = ",", 1
    for sep in CANDIDATE_SEPARATORS:
        counts = {len(row) for row in csv.reader(sample, delimiter=sep)}
        if len(counts) == 1 and (columns := counts.pop()) > best_columns:
            best, best_columns = sep, columns
    return best


class DataFrameService:
    def __init__(self):
        self.df = None

    def load_csv(self, path: str):
        encoding = detect_encoding(path)
        sep = detect_separator(path, encoding)
        self.df = pd.read_csv(path, sep=sep, encoding=encoding)
        return self.df

    def get_df(self):
        if self.df is None:
            raise ValueError("No dataset loaded.")
        return self.df

dataframe_service = DataFrameService()
