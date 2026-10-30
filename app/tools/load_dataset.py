from app.services.dataframe_services import dataframe_service

def load_dataset_tool(path: str) -> str:
    """
    Load a dataset from the given path and return its shape and columns with their types.

    args:
        path (str): The file path to the dataset (e.g., CSV file).
    """
    try:
        df = dataframe_service.load_csv(path)
        return f"Dataset loaded! Shape: {df.shape}\nColumns:\n{dataframe_service.schema()}"
    except Exception as e:
        return f"Error loading dataset: {str(e)}"