import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.data import load_csv, load_multiple_csvs, merge_dataframes, prepare_dataframe
from src.evaluation import evaluate_all, best_result


st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("Supply Chain Demand Forecasting")
st.write("Upload one or multiple CSVs, compare models, and view the best forecast.")


@st.cache_data
def _load(file):
    return load_csv(file)


@st.cache_data
def _load_multiple(files):
    return load_multiple_csvs(files)


def plot_forecast(train_df, test_df, preds, date_col, target_col, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df[date_col], y=train_df[target_col], name="Train", mode="lines"))
    fig.add_trace(go.Scatter(x=test_df[date_col], y=test_df[target_col], name="Actual (holdout)", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=test_df[date_col], y=preds, name=f"Forecast ({model_name})", mode="lines+markers"))
    fig.update_layout(xaxis_title="Date", yaxis_title=target_col, legend=dict(orientation="h"))
    return fig


def main():
    st.sidebar.header("Configuration")
    ma_window = st.sidebar.slider("Moving average window", min_value=2, max_value=28, value=7, step=1)
    use_prophet = st.sidebar.checkbox("Use Prophet (requires install)", value=True)
    horizon = st.sidebar.number_input("Holdout horizon (periods)", min_value=3, max_value=60, value=14, step=1)

    uploaded_files = st.file_uploader("Upload CSV file(s)", type=["csv"], accept_multiple_files=True)
    if not uploaded_files:
        st.info("Upload one or more CSV files with a date column and demand column to begin.")
        st.caption("See sample at `data/sample_orders.csv`.")
        st.caption("💡 Tip: Upload multiple files to combine data from different sources, regions, or time periods.")
        return

    # Load all files
    with st.spinner(f"Loading {len(uploaded_files)} file(s)..."):
        loaded_files = _load_multiple(uploaded_files)
    
    # Check for failed loads
    failed_files = [name for name, df in loaded_files if df is None]
    if failed_files:
        st.error(f"Failed to load: {', '.join(failed_files)}")
    
    # Filter out failed loads
    valid_files = [(name, df) for name, df in loaded_files if df is not None]
    
    if not valid_files:
        st.error("No valid CSV files could be loaded.")
        return
    
    # Show loaded files info
    st.success(f"✅ Successfully loaded {len(valid_files)} file(s)")
    with st.expander("📁 View loaded files"):
        for name, df in valid_files:
            st.write(f"**{name}**: {len(df)} rows × {len(df.columns)} columns")
            st.caption(f"Columns: {', '.join(df.columns.tolist())}")
    
    # If multiple files, ask how to combine them
    merge_mode = None
    if len(valid_files) > 1:
        st.subheader("📊 Combine Multiple Files")
        merge_mode = st.radio(
            "How should we combine the files?",
            options=["concat", "merge"],
            format_func=lambda x: {
                "concat": "📥 Concatenate (stack rows) - Use when files have same columns but different time periods/products",
                "merge": "🔗 Merge by date (join columns) - Use when files have different metrics for the same dates"
            }[x],
            help="Concatenate: Stack all rows vertically (useful for different time periods or products). Merge: Join on date column horizontally (useful for different metrics)."
        )
    
    # Merge dataframes if multiple files
    if len(valid_files) > 1:
        dataframes = [df for _, df in valid_files]
        
        # For merge mode, we need to know the date column first
        if merge_mode == "merge":
            # Show columns from first file to help user identify date column
            st.info("For merge mode, please select the date column that exists in all files.")
            sample_df = dataframes[0]
            date_col_for_merge = st.selectbox(
                "Date column (for merging)", 
                options=sample_df.columns.tolist(),
                key="date_col_merge"
            )
            
            try:
                df = merge_dataframes(dataframes, merge_mode="merge", date_col=date_col_for_merge)
            except Exception as exc:
                st.error(f"Could not merge files: {exc}")
                return
        else:
            # Concatenate mode
            df = merge_dataframes(dataframes, merge_mode="concat")
    else:
        # Single file
        df = valid_files[0][1]

    if df.empty:
        st.warning("Combined dataset is empty.")
        return

    # Column selection
    st.subheader("📋 Select Columns")
    date_col = st.selectbox("Date column", options=df.columns.tolist(), key="date_col")
    target_col = st.selectbox("Demand/target column", options=df.columns.tolist(), key="target_col")

    if date_col == target_col:
        st.warning("Pick different columns for date and target.")
        return

    try:
        df = prepare_dataframe(df[[date_col, target_col]], date_col)
    except Exception as exc:
        st.error(f"Could not parse date column: {exc}")
        return

    if horizon >= len(df):
        st.error("Horizon must be smaller than dataset length.")
        return

    st.subheader("📈 Data Preview")
    st.write(f"Total rows: {len(df)} | Date range: {df[date_col].min()} to {df[date_col].max()}")
    st.dataframe(df.head(10))

    with st.spinner("Training and evaluating models..."):
        results, train_df, test_df = evaluate_all(df, date_col, target_col, horizon=horizon, use_prophet=use_prophet, ma_window=ma_window)

    best = best_result(results)

    st.subheader("Model comparison (lower is better)")
    st.dataframe(pd.DataFrame(results)[["model", "mae", "mape", "error"] if any("error" in r for r in results) else ["model", "mae", "mape"]])

    st.subheader(f"Best model: {best['model']} (MAE: {best['mae']:.2f})")
    fig = plot_forecast(train_df, test_df, best["y_pred"], date_col, target_col, best["model"])
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Holdout uses the last N periods as specified in the sidebar.")


if __name__ == "__main__":
    main()



