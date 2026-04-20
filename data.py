import io
import pandas as pd
from .utils import ensure_datetime


def load_csv(file) -> pd.DataFrame:
    """
    Load a CSV from a path, bytes, or file-like object.
    Streamlit uploader provides a BytesIO; we normalize to pandas.
    """
    if isinstance(file, (str, bytes, io.IOBase)):
        df = pd.read_csv(file)
    elif hasattr(file, "read"):
        df = pd.read_csv(file)
    else:
        raise ValueError("Unsupported file input")
    return df


def load_multiple_csvs(files) -> list[tuple[str, pd.DataFrame]]:
    """
    Load multiple CSV files and return a list of (filename, dataframe) tuples.
    """
    loaded = []
    for file in files:
        try:
            df = load_csv(file)
            filename = file.name if hasattr(file, "name") else str(file)
            loaded.append((filename, df))
        except Exception as e:
            # Continue loading other files even if one fails
            filename = file.name if hasattr(file, "name") else str(file)
            loaded.append((filename, None))  # Mark as failed
    return loaded


def merge_dataframes(dataframes: list[pd.DataFrame], merge_mode: str = "concat", date_col: str = None) -> pd.DataFrame:
    """
    Merge multiple dataframes based on the specified mode.
    
    Args:
        dataframes: List of dataframes to merge
        merge_mode: "concat" (vertical stack) or "merge" (horizontal merge on date)
        date_col: Required for "merge" mode to join on date column
    
    Returns:
        Merged dataframe
    """
    if not dataframes:
        return pd.DataFrame()
    
    if len(dataframes) == 1:
        return dataframes[0]
    
    if merge_mode == "concat":
        # Vertical concatenation - stack all rows
        # This is useful when files contain different time periods or different products
        return pd.concat(dataframes, ignore_index=True)
    
    elif merge_mode == "merge":
        # Horizontal merge on date column
        if date_col is None:
            raise ValueError("date_col is required for merge mode")
        
        # Ensure all dataframes have the date column as datetime
        dfs_processed = []
        for df in dataframes:
            df = ensure_datetime(df.copy(), date_col)
            dfs_processed.append(df)
        
        # Start with first dataframe
        result = dfs_processed[0]
        
        # Merge remaining dataframes on date column
        for i, df in enumerate(dfs_processed[1:], 1):
            # Add suffix to avoid column name conflicts (except date_col)
            suffix = f"_file{i}"
            result = pd.merge(
                result, 
                df, 
                on=date_col, 
                how="outer", 
                suffixes=("", suffix)
            )
        
        return result.sort_values(date_col).reset_index(drop=True)
    
    else:
        raise ValueError(f"Unknown merge_mode: {merge_mode}. Use 'concat' or 'merge'.")


def prepare_dataframe(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Clean and sort dataframe by date."""
    df = ensure_datetime(df, date_col)
    return df



