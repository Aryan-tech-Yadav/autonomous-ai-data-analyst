from io import BytesIO

import pandas as pd
import pytest

from src.data.file_loader import load_uploaded_file


class UploadedFile(BytesIO):
    def __init__(self, name, content):
        super().__init__(content)
        self.name = name


def test_loads_csv_and_strips_column_names():
    file = UploadedFile(
        "sales.csv",
        b" Region ,Revenue\nNorth,100\nSouth,200\n",
    )

    df = load_uploaded_file(file)

    assert list(df.columns) == ["Region", "Revenue"]
    assert len(df) == 2


def test_loads_xlsx():
    buffer = BytesIO()
    pd.DataFrame({"Region": ["North"], "Revenue": [100]}).to_excel(
        buffer, index=False
    )

    file = UploadedFile("sales.xlsx", buffer.getvalue())
    df = load_uploaded_file(file)

    assert list(df.columns) == ["Region", "Revenue"]
    assert df.iloc[0]["Revenue"] == 100


def test_rejects_empty_csv():
    file = UploadedFile("empty.csv", b"Region,Revenue\n")

    with pytest.raises(ValueError, match="empty"):
        load_uploaded_file(file)


def test_rejects_unsupported_file_type():
    file = UploadedFile("sales.txt", b"hello")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_uploaded_file(file)


def test_rejects_empty_column_name(monkeypatch):
    def fake_read_csv(_file):
        return pd.DataFrame([[1, 2]], columns=["Region", ""])

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    file = UploadedFile("sales.csv", b"ignored")

    with pytest.raises(ValueError, match="Column names cannot be empty"):
        load_uploaded_file(file)


def test_rejects_duplicate_column_names(monkeypatch):
    def fake_read_csv(_file):
        return pd.DataFrame([[1, 2]], columns=["Region", "Region"])

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    file = UploadedFile("sales.csv", b"ignored")

    with pytest.raises(ValueError, match="Duplicate column names"):
        load_uploaded_file(file)
