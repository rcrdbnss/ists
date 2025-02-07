from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

from ..piezo.read import create_ts_dict, read_context, create_spatial_matrix
from ..utils import insert_nulls_max_consecutive_thr
from ...preparation import reindex_ts


def read_french(filename: str, id_col: str, date_col: str, cols: List[str]) -> Dict[str, pd.DataFrame]:
    # Read french dataset
    df = pd.read_csv(filename)

    # Transform timestamp column into datetime object
    df[date_col] = pd.to_datetime(df[date_col]).dt.date

    # Split time-series based on date_col and keep only the selected cols
    ts_dict = create_ts_dict(df=df, id_col=id_col, date_col=date_col, cols=cols)

    # Reindex time-series with daily frequency
    ts_dict = {k: reindex_ts(df, 'D') for k, df in ts_dict.items()}

    return ts_dict


def load_frenchpiezo_data(
        ts_filename: str,
        ts_cols: List[str],
        exg_cols: List[str],
        context_filename: str,
        subset_filename: str = None,
        nan_percentage: float = 0,
        exg_cols_stn: List[str] = None,
        exg_cols_stn_scaler: str = 'standard',
        min_length = 0,
        max_null_th = float('inf')
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:

    label_col = ts_cols[0]
    cols = [label_col] + exg_cols

    # Read irregular time series
    ts_dict = read_french(
        filename=ts_filename,
        id_col='bss',
        date_col='time',
        cols=cols,
    )

    # Filter based on a minimum length
    ts_dict = {k: ts for k, ts in ts_dict.items() if len(ts) >= min_length}

    # Filter based on a subset if any
    if subset_filename:
        subset = pd.read_csv(subset_filename)['bss'].to_list()
        ts_dict = {k: ts_dict[k] for k in subset if k in ts_dict}

    # Read time series context (i.e. coordinates)
    if not exg_cols_stn: exg_cols_stn = []
    ctx_dict = read_context(context_filename, 'bss', 'x', 'y', 'masse_eau', *exg_cols_stn)

    # Remove time-series without context information
    ts_dict = {k: ts for k, ts in ts_dict.items() if k in ctx_dict}

    # Remove context information without time-series
    ctx_dict = {k: ctx for k, ctx in ctx_dict.items() if k in ts_dict}

    # Create distance matrix for each pair of irregular time series
    dist_matrix = create_spatial_matrix(ctx_dict, with_haversine=False)
    # Optimized version with numpy
    stations = list(ctx_dict.keys())
    num_stations = len(stations)
    dist_matrix = dist_matrix.to_numpy()
    water_bodies = [ctx_dict[stn]['masse_eau'] for stn in stations]

    # Iterate over pairs of stations to update the distance matrix
    for i in tqdm(range(num_stations), desc="Grouping stations by water body"):
        for j in range(i + 1, num_stations):
            if water_bodies[i] != water_bodies[j]:
                dist_matrix[i, j] = np.inf
                dist_matrix[j, i] = np.inf
    dist_matrix = pd.DataFrame(dist_matrix, index=stations, columns=stations)

    spt_dict = {}
    for k in ts_dict.keys():
        dists = dist_matrix.loc[k]
        dists = dists.drop(k)
        dists = dists[dists < np.inf]
        dists = dists.sort_values(ascending=True)
        spt_dict[k] = dists

    nan, tot = 0, 0
    for stn in ts_dict:
        for col in cols:
            nan += ts_dict[stn][col].isnull().sum()
            tot += len(ts_dict[stn][col])
    print(f"Missing values in the dataset: {nan}/{tot} ({nan/tot:.2%})")

    # Loop through the time-series and insert NaN values at random indices
    if nan_percentage > 0:
        nan, tot = 0, 0
        for stn in tqdm(ts_dict, desc='Injecting null values'):
            for col in cols:
                ts_dict[stn][col] = insert_nulls_max_consecutive_thr(ts_dict[stn][col].to_numpy(), nan_percentage, max_null_th)
                nan += ts_dict[stn][col].isnull().sum()
                tot += len(ts_dict[stn][col])
        print(f'Missing values after injection: {nan}/{tot} ({nan/tot:.2%})')

    # Station-level exogenous features, such as depth into the water body
    if exg_cols_stn:
        exg_cols_stn = {col: [] for col in exg_cols_stn}
        for col, data in exg_cols_stn.items():
            for stn in ts_dict.keys():
                data.append(ctx_dict[stn][col])

        if exg_cols_stn_scaler == 'standard':
            Scaler = StandardScaler
        elif exg_cols_stn_scaler == 'minmax':
            Scaler = MinMaxScaler
        for col, data in exg_cols_stn.items():
            scaler = Scaler()
            scaler.fit(np.reshape(data, (-1, 1)))
            for stn in ts_dict.keys():
                ts_dict[stn][col] = scaler.transform([[ctx_dict[stn][col]]])[0, 0]

    return ts_dict, spt_dict
