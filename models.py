from __future__ import annotations

import warnings
import pandas as pd
from typing import List, Optional

from sklearn.metrics import mean_absolute_error

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None

from .features import build_regression_frame
from .utils import infer_freq

try:
    from prophet import Prophet
except Exception:  # pragma: no cover - optional dependency
    Prophet = None


class BaseForecaster:
    name: str = "base"

    def fit(self, df: pd.DataFrame, date_col: str, target_col: str):
        raise NotImplementedError

    def predict(self, horizon: int, dates: Optional[pd.Series] = None) -> List[float]:
        raise NotImplementedError

    def evaluate(self, df_train: pd.DataFrame, df_test: pd.DataFrame, date_col: str, target_col: str):
        self.fit(df_train, date_col, target_col)
        preds = self.predict(len(df_test), df_test[date_col])
        mae = mean_absolute_error(df_test[target_col], preds)
        return {"model": self.name, "mae": mae, "y_pred": preds}


class NaiveForecaster(BaseForecaster):
    name = "naive_last"

    def __init__(self):
        self.last_value: Optional[float] = None

    def fit(self, df: pd.DataFrame, date_col: str, target_col: str):
        self.last_value = df[target_col].iloc[-1]

    def predict(self, horizon: int, dates: Optional[pd.Series] = None) -> List[float]:
        if self.last_value is None:
            raise ValueError("Model not fitted")
        return [self.last_value] * horizon


class MovingAverageForecaster(BaseForecaster):
    name = "moving_average"

    def __init__(self, window: int = 3):
        self.window = window
        self.avg: Optional[float] = None

    def fit(self, df: pd.DataFrame, date_col: str, target_col: str):
        self.avg = df[target_col].tail(self.window).mean()

    def predict(self, horizon: int, dates: Optional[pd.Series] = None) -> List[float]:
        if self.avg is None:
            raise ValueError("Model not fitted")
        return [self.avg] * horizon


class XGBLagForecaster(BaseForecaster):
    name = "xgboost_lag"

    def __init__(self, lags=None):
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed")
        self.lags = lags or [1, 2, 3, 7, 14]
        self.model = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        self.target_col: Optional[str] = None
        self.date_col: Optional[str] = None
        self.train_values: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame, date_col: str, target_col: str):
        frame = build_regression_frame(df, date_col, target_col, lags=self.lags)
        feature_cols = [c for c in frame.columns if c not in [date_col, target_col]]
        self.model.fit(frame[feature_cols], frame[target_col])
        self.target_col = target_col
        self.date_col = date_col
        self.train_values = df[target_col].copy()

    def predict(self, horizon: int, dates: Optional[pd.Series] = None) -> List[float]:
        if self.train_values is None or self.target_col is None:
            raise ValueError("Model not fitted")
        history = self.train_values.tolist()
        preds = []
        for _ in range(horizon):
            row = {}
            for lag in self.lags:
                row[f"lag_{lag}"] = history[-lag] if len(history) >= lag else history[0]
            # calendar features default to 0 if dates missing
            row["dow"] = dates.iloc[len(preds)].dayofweek if dates is not None else 0
            row["dom"] = dates.iloc[len(preds)].day if dates is not None else 1
            row["month"] = dates.iloc[len(preds)].month if dates is not None else 1
            pred = float(self.model.predict(pd.DataFrame([row]))[0])
            preds.append(pred)
            history.append(pred)
        return preds


class ProphetForecaster(BaseForecaster):
    name = "prophet"

    def __init__(self):
        if Prophet is None:
            raise ImportError("prophet is not installed")
        self.model: Optional[Prophet] = None
        self.freq: Optional[str] = None

    def fit(self, df: pd.DataFrame, date_col: str, target_col: str):
        self.freq = infer_freq(df, date_col)
        prophet_df = df.rename(columns={date_col: "ds", target_col: "y"})
        self.model = Prophet()
        self.model.fit(prophet_df)

    def predict(self, horizon: int, dates: Optional[pd.Series] = None) -> List[float]:
        if self.model is None or self.freq is None:
            raise ValueError("Model not fitted")
        future = self.model.make_future_dataframe(periods=horizon, freq=self.freq, include_history=False)
        forecast = self.model.predict(future)
        return forecast["yhat"].tolist()


def available_models(use_prophet: bool, ma_window: int) -> List[BaseForecaster]:
    models: List[BaseForecaster] = [
        NaiveForecaster(),
        MovingAverageForecaster(window=ma_window),
    ]
    if XGBRegressor is None:
        warnings.warn("xgboost not installed; skipping.")
    else:
        models.append(XGBLagForecaster())
    if use_prophet:
        if Prophet is None:
            warnings.warn("Prophet not installed; skipping.")
        else:
            models.append(ProphetForecaster())
    return models

