from app.services.dataframe_services import dataframe_service

def generate_insights_tool() -> str:
    """
    Generate basic insights about the dataset.
    """
    try:
        df = dataframe_service.get_df()

        insights = []

        insights.append(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

        # numeric columns
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            insights.append(f"There are {len(numeric_cols)} numeric columns.")

            for col in numeric_cols[:3]:  # limit to avoid overflow
                mean = df[col].mean()
                insights.append(f"The mean of column '{col}' is {mean:.2f}.")

        # missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            insights.append(f"The dataset has {missing} missing values.")
        else:
            insights.append("There are no missing values in the dataset.")

        return "\n".join(insights)

    except Exception as e:
        return f"Error generating insights: {str(e)}"