from typing import List

import numpy as np
import pandas as pd


def get_time_max_sizes(codes: List[str]) -> List[int]:
    time_max_sizes = []
    for code in codes:
        if code == 'D':
            # Extract max day value
            val = 31
        elif code == 'DW':
            # Extract max day of the week value (Monday: 0, Sunday: 6)
            val = 7
        elif code == 'M':
            # Extract max month value
            val = 12
        elif 'WY' in codes:
            # Extract max week of the year value
            val = 56
        else:
            raise ValueError(f"Code {code} is not supported, it must be ['D', 'DW', 'WY', 'M']")

        time_max_sizes.append(val)

    return time_max_sizes


def time_encoding(df: pd.DataFrame, codes: List[str]) -> pd.DataFrame:
    # Create a copy of the input series
    df_new = df.copy()

    # Convert the date index series to datetime type
    datetime_series = pd.to_datetime(df.index)

    # Check code format
    for code in codes:
        if code not in ['D', 'DW', 'WY', 'M']:
            raise ValueError(f"Code {code} is not supported, it must be ['D', 'DW', 'WY', 'M']")

    if 'D' in codes:
        # Extract day value
        df_new['D'] = pd.Series(datetime_series.day).values - 1
    if 'DW' in codes:
        # Extract day of the week (Monday: 0, Sunday: 6)
        df_new['DW'] = pd.Series(datetime_series.dayofweek).values
    if 'M' in codes:
        # Extract month
        df_new['M'] = pd.Series(datetime_series.month).values - 1
    if 'WY' in codes:
        # Extract week of the year
        df_new['WY'] = pd.Series(datetime_series.isocalendar().week).values - 1

    return df_new


class StandardScalerBatch(object):
    def __init__(self, p1: float = None, p2: float = None):
        self.mean_array = None
        self.std_array = None
        self.p1 = p1
        self.p2 = p2

    def fit(self, x: np.ndarray):
        assert x.ndim == 3, 'Support only temporal data'
        self.mean_array = np.mean(x, axis=1)
        self.mean_array = self.mean_array[:, np.newaxis, :]
        self.std_array = np.std(x, axis=1)
        self.std_array = self.std_array[:, np.newaxis, :]

        # Manage constant time series scenario (with std 0) by transforming std into 1
        cond_mean = np.all((x - self.mean_array) == 0, axis=1)
        cond_std = np.squeeze(self.std_array == 0, axis=1)
        cond = (cond_mean & cond_std)[:, np.newaxis, :]
        self.std_array[cond] = 1

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 3 or x.ndim == 2, 'Support only temporal data and label'
        is_label = x.ndim == 2
        mean = self.mean_array if not is_label else self.mean_array[:, :, 0]
        std = self.std_array if not is_label else self.std_array[:, :, 0]

        if np.any(std == 0):
            raise ValueError('Invalid value encountered in standard scaler transform, std is 0')

        x_transformed = (x - mean) / std

        if is_label and self.p1 is not None and self.p2 is not None:
            th1, th2 = np.percentile(x_transformed, [self.p1, self.p2])
            x_transformed[x_transformed < th1] = th1
            x_transformed[x_transformed > th2] = th2

        return x_transformed

    def inverse_transform(self, x_transformed: np.ndarray) -> np.ndarray:
        assert x_transformed.ndim == 3 or x_transformed.ndim == 2, 'Support only temporal data and label'
        mean = self.mean_array if x_transformed.ndim == 3 else self.mean_array[:, :, 0]
        std = self.std_array if x_transformed.ndim == 3 else self.std_array[:, :, 0]
        x = (x_transformed * std) + mean
        return x

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


class MinMaxScalerBatch(object):
    def __init__(self):
        self.min, self.max = 0, 1
        self.max_array = None
        self.min_array = None

    def fit(self, x: np.ndarray):
        assert x.ndim == 3, 'Support only temporal data'
        self.max_array = np.max(x, axis=1)
        self.max_array = self.max_array[:, np.newaxis, :]
        self.min_array = np.min(x, axis=1)
        self.min_array = self.min_array[:, np.newaxis, :]

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 3 or x.ndim == 2, 'Support only temporal data and label'
        max_array = self.max_array if x.ndim == 3 else self.max_array[:, :, 0]
        min_array = self.min_array if x.ndim == 3 else self.min_array[:, :, 0]

        x_std = (x - min_array) / (max_array - min_array)
        x_scaled = x_std * (self.max - self.min) + self.min

        return x_scaled

    def inverse_transform(self, x_scaled: np.ndarray) -> np.ndarray:
        assert x_scaled.ndim == 3 or x_scaled.ndim == 2, 'Support only temporal data and label'
        max_array = self.max_array if x_scaled.ndim == 3 else self.max_array[:, :, 0]
        min_array = self.min_array if x_scaled.ndim == 3 else self.min_array[:, :, 0]

        x = ((x_scaled - self.min) / (self.max - self.min)) * (max_array - min_array) + min_array
        return x

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

# def main():
#     data1 = [[1, 1], [2, 0], [3, -1], [4, -2]]
#     data1 = np.array(data1)[np.newaxis, :]
#     data = np.vstack([data1, data1 * -1])
#     label_data = data[:, -1, 0][:, np.newaxis]
#
#     transformer = MinMaxScalerBatch()
#     transformer.fit(data)
#     data_transformed = transformer.transform(data)
#     data_back = transformer.inverse_transform(data_transformed)
#     label_transformed = transformer.transform(label_data)
#     label_back = transformer.inverse_transform(label_transformed)
#     print('Hello World!')
#
#
# if __name__ == '__main__':
#     main()
#     print('Hello World!')
