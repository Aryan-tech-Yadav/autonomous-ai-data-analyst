import pandas as pd


def load_uploaded_file(uploaded_file):
    """Load and validate CSV or Excel business data."""

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Upload CSV or XLSX.")

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("The dataset has no columns.")

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    if any(not column for column in df.columns):
        raise ValueError("Column names cannot be empty.")

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"Duplicate column names found: {duplicates}"
        )

    return df
