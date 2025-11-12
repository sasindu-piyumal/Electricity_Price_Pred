import pandas as pd

def filter_smpep2(df):
    """
    Filter rows in DataFrame df, keeping only rows where SMPEP2 is greater than 0 and less than or equal to 550.
    """
    return df[(df['SMPEP2'] > 0) & (df['SMPEP2'] <= 550)]
