# Supply Chain Demand Forecasting (Streamlit)

Lightweight, local-first demand forecasting workspace. Upload a CSV, compare multiple models, and visualize forecasts directly in Streamlit.

## Quick start

```bash
cd /Users/jitin/Documents/Cursor/supply-forecast
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## CSV format

- Required: one date column (daily/weekly/etc.) and one numeric demand column.
- Optional: other columns are ignored for now.
- Example: `data/sample_orders.csv`.

## Models included

- Naive last-value baseline
- Moving average baseline (configurable window)
- XGBoost regressor with lag features
- Prophet (if installed)

The app evaluates each model on a holdout (last N periods), picks the best by MAE, and plots the forecast.

## Project layout

- `app.py` — Streamlit UI
- `src/data.py` — CSV loading/parsing helpers
- `src/features.py` — lag feature builders
- `src/models.py` — model wrappers
- `src/evaluation.py` — train/test split and metrics
- `data/sample_orders.csv` — toy dataset




