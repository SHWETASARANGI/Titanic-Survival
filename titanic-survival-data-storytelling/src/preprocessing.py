from pathlib import Path
import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Loads the data from the given path and returns a pandas DataFrame.
    """
    df = pd.read_csv(path)
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans column names by converting them to lowercase
    and replacing spaces with underscores.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values using domain-informed strategies.
    """
    df = df.copy()

    # Age: median by sex and class, then global median
    if {"age", "sex", "pclass"}.issubset(df.columns):
        df["age"] = (
            df.groupby(["sex", "pclass"])["age"]
            .transform(lambda x: x.fillna(x.median()))
        )
        df["age"] = df["age"].fillna(df["age"].median())

    # Embarked: most frequent
    if "embarked" in df.columns:
        df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # Fare: median
    if "fare" in df.columns:
        df["fare"] = df["fare"].fillna(df["fare"].median())

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds interpretable features for analysis and storytelling.
    """
    df = df.copy()

    # Family features
    if {"sibsp", "parch"}.issubset(df.columns):
        df["family_size"] = df["sibsp"] + df["parch"] + 1
        df["is_alone"] = (df["family_size"] == 1).astype(int)

    # Age groups
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 12, 60, np.inf],
            labels=["Child", "Adult", "Senior"]
        )

    # Log-transformed fare
    if "fare" in df.columns:
        df["log_fare"] = np.log1p(df["fare"])

    return df


def preprocess_data(raw_path: str, processed_path: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    Loads raw data, cleans columns, handles missing values,
    adds features, and saves the processed dataset.
    """
    df = load_data(raw_path)
    df = clean_columns(df)
    df = handle_missing_values(df)
    df = add_features(df)

    # Sanity check for storytelling / modeling
    is_training = "survived" in df.columns

    if is_training:
        print("Training data detected (target present)")
    else:
        print("Inference data detected (no target)")

    # Ensure output directory exists
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    return df
