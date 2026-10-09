from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_disease.csv"
)

EXPECTED_COLUMNS = {
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
}


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Perform basic structural validation of the dataset.
    """

    missing_columns = EXPECTED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    target_values = set(df["target"].dropna().unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Unexpected target values: {target_values}"
        )

    print("Dataset validation passed.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    validate_dataset(df)


if __name__ == "__main__":
    main()