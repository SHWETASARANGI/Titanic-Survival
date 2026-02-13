import pandas as pd

def select_features(df : pd.DataFrame, features : list) -> pd.DataFrame:
    """
    Selects the specified features from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame from which features will be selected.
    features (list): A list of feature names to be selected.
    
    Returns:
    pd.DataFrame: The DataFrame containing only the selected features.
    """
    
    features = df.copy()

    # Encode Sex (interpretable binary)
    if "sex" in features.columns:
        features["sex"] = features["sex"].map({"male": 0, "female": 1})

    # Ensure Pclass is treated as ordinal
    if "pclass" in features.columns:
        features["pclass"] = features["pclass"].astype(int)

    selected_columns = [
        "sex",
        "pclass",
        "age",
        "logfare",
        "famil_ysize",
        "is_alone"
    ]

    return features[selected_columns]

def extract_target(df : pd.DataFrame) -> pd.Series:
    """
    Extracts the target variable from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame from which the target variable will be extracted.
    target_column (str): The name of the column containing the target variable.
    
    Returns:
    pd.Series: The extracted target variable as a pandas Series.
    """
    return df["survived"]