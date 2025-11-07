import pandas as pd

def filter_smpep2_out(df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean mask for rows where SMPEP2 is in the interval (0, 550].

    Args:
        df: DataFrame that must contain the 'SMPEP2' column.

    Returns:
        A pandas Series of boolean values, True for rows where 0 < SMPEP2 <= 550.
    """
    if 'SMPEP2' not in df.columns:
        raise KeyError("DataFrame must contain a 'SMPEP2' column")
    return (df['SMPEP2'] > 0) & (df['SMPEP2'] <= 550)
