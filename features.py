import pandas as pd


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add basic calendar features."""
    df = df.copy()
    df["dow"] = df[date_col].dt.dayofweek
    df["dom"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, lags=None) -> pd.DataFrame:
    """Create lag columns for supervised learning."""
    if lags is None:
        lags = [1, 2, 3, 7, 14]
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def build_regression_frame(df: pd.DataFrame, date_col: str, target_col: str, lags=None) -> pd.DataFrame:
    """Full frame with time and lag features, dropping NA from lags."""
    df = add_time_features(df, date_col)
    df = add_lag_features(df, target_col, lags=lags)
    return df.dropna().reset_index(drop=True)




