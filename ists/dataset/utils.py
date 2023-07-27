from typing import List, Literal
import calendar
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def move_to_end_of_week_or_month(dt: datetime.date, move_to: Literal['M', 'W']):
    if move_to == 'W':
        # Move to the end of the week
        dt = dt + timedelta(days=(6 - dt.weekday()))
    elif move_to == 'M':
        # Move to the end of the month
        _, last_day = calendar.monthrange(dt.year, dt.month)
        dt = dt.replace(day=last_day)
    else:
        raise ValueError("Invalid move_to option. Must be 'M', 'W'.")

    return dt


def insert_null_values(df: pd.DataFrame, pct_nan: float, cols: List[str]) -> pd.DataFrame:
    # Calculate the number of NaN values to insert per column
    num_nan_values = int(len(df) * pct_nan)

    # Create a copy of the original dataframe
    df_nan = df.copy()
    df_nan = df_nan.reset_index(drop=True)  # Reset the implicit index

    # Loop through the specified columns and insert NaN values at the random indices
    for col in cols:
        # Generate random indices to place NaN values
        indices_to_nan = np.random.choice(len(df), num_nan_values, replace=False)
        df_nan.loc[indices_to_nan, col] = np.nan

    # Assign the original index
    df_nan.index = df.index
    return df_nan
