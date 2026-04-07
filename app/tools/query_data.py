from app.services.dataframe_services import dataframe_service

def query_data_tool(question: str) -> str:
    """
    Query the dataset with a user question.
    """
    try:
        df = dataframe_service.get_df()

        # for simplicity, we just return the question and a sample of the data
        sample = df.head(10).to_string()

        return f"""
        Dataset (sample):
        {sample}

        User Question:
        {question}

        Respond with based on the data above.
        """

    except Exception as e:
        return f"Error querying dataset: {str(e)}"