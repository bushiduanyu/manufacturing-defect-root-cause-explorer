"""Dataset loading and validation helpers."""

from pathlib import Path

import pandas as pd

from src.feature_groups import EXPECTED_COLUMNS, NUMERIC_ANALYSIS_COLUMNS, TARGET_COLUMN


DEFAULT_DATA_PATH = Path("data/manufacturing_defect_dataset.csv")


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the manufacturing defect dataset."""
    return pd.read_csv(path)


def get_missing_columns(df: pd.DataFrame) -> list[str]:
    """Return expected columns that are not present in the dataset."""
    return sorted(set(EXPECTED_COLUMNS) - set(df.columns))


def get_non_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return analysis columns that are present but not numeric."""
    present_columns = [column for column in NUMERIC_ANALYSIS_COLUMNS if column in df.columns]
    return [
        column
        for column in present_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]


def is_binary_target(df: pd.DataFrame, target: str = TARGET_COLUMN) -> bool:
    """Return True when the target column contains only 0 and 1 values."""
    if target not in df.columns:
        return False

    values = set(df[target].dropna().unique())
    return values == {0, 1}


def validation_summary(df: pd.DataFrame) -> dict:
    """Return an app-friendly validation summary for the dataset."""
    missing_columns = get_missing_columns(df)
    non_numeric_columns = get_non_numeric_columns(df)
    target_is_binary = is_binary_target(df)
    missing_values = df.isna().sum()

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "expected_column_count": len(EXPECTED_COLUMNS),
        "missing_columns": missing_columns,
        "non_numeric_columns": non_numeric_columns,
        "target_column": TARGET_COLUMN,
        "target_is_binary": target_is_binary,
        "total_missing_values": int(missing_values.sum()),
        "missing_values_by_column": missing_values.to_dict(),
        "is_valid": not missing_columns and not non_numeric_columns and target_is_binary,
    }


def validate_dataset(df: pd.DataFrame) -> None:
    """Raise a ValueError if the dataset does not match notebook requirements."""
    summary = validation_summary(df)

    missing = summary["missing_columns"]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    non_numeric = summary["non_numeric_columns"]
    if non_numeric:
        raise ValueError(f"Dataset has non-numeric analysis columns: {non_numeric}")

    if not summary["target_is_binary"]:
        raise ValueError(f"{TARGET_COLUMN} must contain binary values 0 and 1")
