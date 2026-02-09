import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, 



def train_explanatory_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42
):
    """
    Train a logistic regression model for explanation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    test_size : float
    random_state : int

    Returns
    -------
    model : sklearn Pipeline
    metrics : dict
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return pipeline, metrics


def get_model_coefficients(model_pipeline, feature_names):
    """
    Extract logistic regression coefficients for interpretation.

    Parameters
    ----------
    model_pipeline : sklearn Pipeline
    feature_names : list

    Returns
    -------
    pd.DataFrame
        Coefficients with feature names.
    """
    coefficients = model_pipeline.named_steps["model"].coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    }).sort_values(by="coefficient", key=abs, ascending=False)

    return coef_df


def compute_odds_ratios(coef_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert coefficients to odds ratios.

    Parameters
    ----------
    coef_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    coef_df = coef_df.copy()
    coef_df["odds_ratio"] = coef_df["coefficient"].apply(lambda x: round(float(pd.np.exp(x)), 3))
    return coef_df