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


def insert_nulls_max_consecutive_thr(time_series: np.ndarray, missing_rate: float, max_consecutive: int):
    n = len(time_series)
    pre_existing_nulls = np.isnan(time_series)
    total_pre_existing_nulls = pre_existing_nulls.sum()

    # Calculate the number of nulls to introduce
    total_missing = int(missing_rate * n)
    nulls_to_introduce = max(0, total_missing - total_pre_existing_nulls)
    # nulls_to_introduce = int(missing_rate * n)

    time_series = time_series.copy()
    missing_indices = set(np.where(pre_existing_nulls)[0])
    valid_indices = set(range(1, n-1)) - (missing_indices | {idx - 1 for idx in missing_indices if idx > 0} | {idx + 1 for idx in missing_indices if idx < n - 1})

    while len(missing_indices) - total_pre_existing_nulls < nulls_to_introduce and len(valid_indices) > 0:
        # Pick a random starting point
        # start_idx = np.random.randint(1, n-1)
        start_idx = np.random.choice(list(valid_indices))

        # Determine the consecutive missing length, bounded by max_consecutive
        consecutive_length = np.random.randint(1, max_consecutive + 1)

        # Generate the new interval and include bounds for gap validation
        end_idx = min(start_idx + consecutive_length, n)
        new_indices = set(range(start_idx, end_idx))
        new_indices_and_bounds = new_indices | {start_idx - 1} | {end_idx}  # Bounds to enforce gaps

        # Check for overlap or contiguity with existing intervals
        if not new_indices_and_bounds & missing_indices:
            # Add the new interval if it satisfies the constraints
            missing_indices.update(new_indices)
            valid_indices -= new_indices_and_bounds

        # Replace selected indices with NaN
    for idx in missing_indices - set(np.where(pre_existing_nulls)[0]):
        time_series[idx] = np.nan

    return time_series


def transpose_dict_of_dicts(original_dict):
    transposed_dict = {}
    all_keys = set(key for nested in original_dict.values() for key in nested)

    for outer_key, nested_dict in original_dict.items():
        for key in all_keys:
            if key not in transposed_dict:
                transposed_dict[key] = {}
            value = nested_dict.get(key, None)
            if value is not None:
                transposed_dict[key][outer_key] = value
    return transposed_dict


if __name__ == '__main__':
    # Create a time series with 1000 values
    ts = pd.Series(np.random.randn(1000))

    # Insert 10% of NaN values
    ts_nan = pd.Series(insert_nulls_max_consecutive_thr(ts.to_numpy(), 0.5, 12))

    print(ts_nan)
    print(ts_nan.isnull().sum())
    print(ts_nan.isnull().sum() / len(ts_nan))
