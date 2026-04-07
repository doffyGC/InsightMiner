from app.services.dataframe_services import dataframe_service

def describe_data_tool() -> str:
    """
    Return basic statistics of the dataset.
    """
    try:
        df = dataframe_service.get_df()
        description = df.describe(include="all").to_string()
        return f"Dataset Statistics:\n{description}"
    except Exception as e:
        return f"Error describing dataset: {str(e)}"