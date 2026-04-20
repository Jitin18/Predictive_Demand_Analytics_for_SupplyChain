import pandas as pd
from sklearn.metrics import mean_absolute_error

from .models import available_models
from .utils import train_test_split_time, mape


def evaluate_all(df: pd.DataFrame, date_col: str, target_col: str, horizon: int, use_prophet: bool, ma_window: int):
    """Train/evaluate all models on a holdout horizon."""
    if horizon >= len(df):
        raise ValueError("Horizon must be smaller than dataset length.")

    df_train, df_test = train_test_split_time(df, horizon)
    results = []
    for model in available_models(use_prophet=use_prophet, ma_window=ma_window):
        try:
            res = model.evaluate(df_train, df_test, date_col, target_col)
            res["mape"] = mape(df_test[target_col], res["y_pred"])
            results.append(res)
        except Exception as exc:  # pragma: no cover - defensive for demo
            results.append({"model": model.name, "mae": float("inf"), "mape": float("inf"), "y_pred": [], "error": str(exc)})
    results = sorted(results, key=lambda x: x["mae"])
    return results, df_train, df_test


def best_result(results):
    return min(results, key=lambda x: x["mae"])




