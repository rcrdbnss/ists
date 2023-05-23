from typing import Dict, List, Tuple, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm


def reindex_ts(ts: pd.DataFrame, freq: Literal['M', 'W']):
    min_dt, max_dt = ts.index.min(), ts.index.max()  # Read min max date
    # dt_name = ts.index.name  # Date Index name
    # Create a new monthly or week index
    new_index = pd.date_range(start=min_dt, end=max_dt, freq=freq).date
    # Check index constraint
    assert np.isin(ts.index, new_index).all()
    # Reindex the time-series DataFrame
    ts = ts.reindex(new_index)
    return ts


def null_distance_array(is_null: np.ndarray, method: Literal['log', 'lin'] = 'lin', max_dist: int = None) -> np.ndarray:
    # Initialize arrays
    distance_array = np.zeros(len(is_null))
    last_observed_index = -1
    for i, val in enumerate(is_null):
        if not val:
            # Not null value
            # Set distance_array to 0
            distance_array[i] = 0
            # Reset last_observed_index
            last_observed_index = i
        else:
            # Compute distance from last_observed_index
            if last_observed_index >= 0:
                if method == 'lin':
                    # Linear distance
                    distance_array[i] = i - last_observed_index
                    if max_dist:
                        distance_array[i] /= max_dist
                elif method == 'log':
                    # Log linear distance
                    distance_array[i] = np.log(1 + i - last_observed_index)
                    if max_dist:
                        distance_array[i] /= np.log(max_dist)
                else:
                    raise ValueError("Invalid method option. Must be 'log', 'lin'.")
            else:
                raise ValueError('Input array start with null values')

    return distance_array


def null_indicator(ts: pd.DataFrame, method: Literal['bool', 'log', 'lin'] = 'bool',
                   max_dist: int = None) -> np.ndarray:
    cond_null = pd.isnull(ts).any(axis=1).values
    if method == 'bool':
        null_array = cond_null.astype(int)
    elif method == 'lin':
        null_array = null_distance_array(cond_null, method=method, max_dist=max_dist)
    elif method == 'log':
        null_array = null_distance_array(cond_null, method=method, max_dist=max_dist)
    else:
        raise ValueError("Invalid method option. Must be 'log', 'lin', or 'bool.")

    if method == 'bool' or max_dist:
        null_array = 1 - null_array

    return null_array


def find_nearest_label(ts_label: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    label_isnull = ts_label.isnull().values
    distance_fw = null_distance_array(label_isnull, method='lin', max_dist=None)
    distance_bw = null_distance_array(label_isnull[::-1], method='lin', max_dist=None)[::-1]
    label_fw = ts_label.fillna(method='ffill').values
    label_bw = ts_label.fillna(method='bfill').values
    min_distance = np.where(distance_fw < distance_bw, distance_fw, distance_bw)
    nearest_label = np.where(distance_fw < distance_bw, label_fw, label_bw)

    return nearest_label, min_distance


def sliding_window(
        ts: pd.DataFrame,
        label_col: str,
        features: List[str],
        num_past: int,
        num_fut: int,
) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # # Make the label col the first in dataframe
    # cols = ts.columns[ts.columns != label_col].to_list()
    # ts = ts[[label_col] + cols]

    # Fix stride to 1
    stride = 1

    # Save date col name and transform it into column
    date_col = ts.index.name
    ts = ts.reset_index()

    # Find for each null label the nearest not null value and the distance from it in forward and backward
    new_label, dist_label = find_nearest_label(ts[label_col])
    ts['__LABEL__'] = new_label
    ts['__DISTANCE__'] = dist_label

    # Define window size
    window = num_past + num_fut
    rows, cols = ts.shape

    # Check minimum rows constraints
    if rows <= window:
        return None

    # New number of rows after the sliding window
    new_rows = 1 + (rows - window) // stride
    # Empty matrix to save sliding window segments
    matrix = np.zeros((new_rows, window, cols), dtype=np.object)

    # Sliding window process
    data = ts.values
    for i in range(new_rows):
        left = i * stride
        right = left + window
        matrix[i, :, :] = data[left:right, :]

    cond_x = np.isin(ts.columns, features)
    cond_y = ts.columns == '__LABEL__'
    cond_date = ts.columns == date_col
    cond_dist = ts.columns == '__DISTANCE__'

    x = matrix[:, :num_past, cond_x]
    y = matrix[:, window - 1, cond_y]
    time_mask = matrix[:, [0, num_past - 1, -1], cond_date]
    dist_y = matrix[:, window - 1, cond_dist]

    # y = np.squeeze(y, axis=2)
    # time_y = np.squeeze(time_y, axis=2)

    return x, y, time_mask, dist_y


def prepare_data(
        ts_dict: Dict[str, pd.DataFrame],
        num_past: int,
        num_fut: int,
        features: List[str],
        label_col: str,
        freq: Literal['M', 'W', 'D'] = 'D',
        null_feat: Literal['bool', 'log', 'lin'] = None,
        max_dist: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Empty array for saving x, y, time mask, and id for each record
    x_array = []
    y_array = []
    time_array = []
    dist_array = []
    id_array = []

    # Iterate each time-series
    for k, ts in tqdm(ts_dict.items()):
        # Reindex time index
        ts_new = reindex_ts(ts, freq=freq)

        # Null indicator feature
        if null_feat:
            ts_new['NullFeat'] = null_indicator(ts_new, method=null_feat, max_dist=max_dist)
            features += ['NullFeat']

        # Sliding window
        ts_new['Date'] = ts_new.index.values
        ts_new = ts_new.fillna(method='ffill')
        features += ['Date']
        blob = sliding_window(ts_new, label_col, features, num_past, num_fut)
        if blob is None:
            continue

        x_array += [blob[0]]
        y_array += [blob[1]]
        time_array += [blob[2]]
        dist_array += [blob[3]]
        id_array += [k] * len(blob[0])

    x_array = np.concatenate(x_array)
    y_array = np.concatenate(y_array)
    time_array = np.concatenate(time_array)
    dist_array = np.concatenate(dist_array)
    id_array = np.array(id_array)

    y_array = np.squeeze(y_array)
    dist_array = np.squeeze(dist_array)

    return x_array, y_array, time_array, dist_array, id_array
