from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def download_heart_disease_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download the UCI Heart Disease dataset.

    Returns
    -------
    X : pd.DataFrame
        Feature dataframe.

    y : pd.DataFrame
        Target dataframe.
    """
    print("Downloading Heart Disease dataset from UCI...")

    dataset = fetch_ucirepo(id=45)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    print("Dataset downloaded successfully.")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y


def prepare_dataset(
    X: pd.DataFrame,
    y: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine features and target and convert the original UCI target
    into a binary heart-disease target.

    Original target:
        0 -> no heart disease
        1-4 -> heart disease present

    Binary target:
        0 -> no heart disease
        1 -> heart disease present
    """

    df = X.copy()

    # UCI target column is called "num"
    target = y["num"]

    df["target"] = (target > 0).astype(int)

    return df


def save_dataset(
    X: pd.DataFrame,
    y: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    """
    Save raw and prepared datasets.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve original feature and target data
    X.to_csv(RAW_DATA_DIR / "heart_disease_features.csv", index=False)
    y.to_csv(RAW_DATA_DIR / "heart_disease_target.csv", index=False)

    # Save combined dataset with binary target
    df.to_csv(
        PROCESSED_DATA_DIR / "heart_disease.csv",
        index=False,
    )

    print("\nFiles saved successfully:")
    print(
        RAW_DATA_DIR / "heart_disease_features.csv"
    )
    print(
        RAW_DATA_DIR / "heart_disease_target.csv"
    )
    print(
        PROCESSED_DATA_DIR / "heart_disease.csv"
    )


def main() -> None:
    X, y = download_heart_disease_data()

    df = prepare_dataset(X, y)

    print("\nPrepared dataset information:")
    print(f"Shape: {df.shape}")
    print("\nTarget distribution:")
    print(df["target"].value_counts().sort_index())

    save_dataset(X, y, df)


if __name__ == "__main__":
    main()