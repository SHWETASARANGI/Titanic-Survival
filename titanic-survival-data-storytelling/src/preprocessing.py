import numpy as np
import pandas as pd

def load_data(path : str) -> pd.DataFrame:
    """
    Loads the data from the given path and returns a pandas DataFrame.
    
    Parameters:
    path (str): The path to the CSV file containing the data.
    
    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(path)

def handle_missing_values(df : pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the DataFrame by filling them with the median value of the respective columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame with potential missing values.
    
    Returns:
    pd.DataFrame: The DataFrame with missing values handled.
    """
    df = df.copy()

    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(lambda x: x.fillna(x.median()))

    df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Fare: median
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    return df

def add_features(df : pd.DataFrame) -> pd.DataFrame:
    """
    Adds new features to the DataFrame, such as 'FamilySize' and 'IsAlone'.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to which new features will be added.
    
    Returns:
    pd.DataFrame: The DataFrame with new features added.
    """
    df = df.copy()

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

     # Age groups for storytelling
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 60, np.inf],
        labels=["Child", "Adult", "Senior"]
    )

    # Log-transformed fare (reduces skew)
    if "Fare" in df.columns:
        df["LogFare"] = np.log1p(df["Fare"])

    return df

def clean_columns(df : pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the column names by converting them to lowercase and replacing spaces with underscores.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame with original column names.
    
    Returns:
    pd.DataFrame: The DataFrame with cleaned column names.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    return df

def preprocess_data(raw_path: str, processed_path: str) -> pd.DataFrame:
    """
    Preprocesses the data by loading it, handling missing values, adding new features, and cleaning column names.
    
    Parameters:
    raw_path (str): The path to the raw CSV file.
    processed_path (str): The path where the processed CSV file will be saved.
    
    Returns:
    pd.DataFrame: The preprocessed DataFrame ready for analysis or modeling.
    """
    df = load_data(raw_path)
    df = handle_missing_values(df)
    df = add_features(df)
    df = clean_columns(df)

    df.to_csv(processed_path, index=False)
    return df
