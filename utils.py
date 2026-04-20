import pandas as pd


def ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Ensure the date column is datetime and sorted ascending."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).reset_index(drop=True)


def infer_freq(df: pd.DataFrame, date_col: str) -> str:
    """Infer frequency from the median delta; fallback to daily."""
    if len(df) < 3:
        return "D"
    deltas = df[date_col].diff().dropna()
    if deltas.empty:
        return "D"
    median_delta = deltas.median()
    return pd.tseries.frequencies.to_offset(median_delta).freqstr or "D"


def train_test_split_time(df: pd.DataFrame, test_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered dataframe into train and holdout."""
    if test_size <= 0 or test_size >= len(df):
        raise ValueError("test_size must be between 1 and len(df)-1")
    return df.iloc[:-test_size], df.iloc[-test_size:]


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    true = pd.Series(y_true).replace(0, eps)
    pred = pd.Series(y_pred)
    return ((true - pred).abs() / true.abs()).mean()




