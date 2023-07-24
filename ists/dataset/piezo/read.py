from typing import TypedDict, Dict, List, Tuple
import math
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import haversine_distances
from pyproj import Transformer

from ...utils import move_to_end_of_week_or_month


class ContextType(TypedDict):
    x: float
    y: float


def create_ts_dict(df: pd.DataFrame, id_col: str, date_col: str, cols: List[str]) -> Dict[str, pd.DataFrame]:
    # Keep only selected cols
    df = df[[id_col, date_col] + cols]

    # Create a dictionary where for each id is associated its time-series
    ts_dict: Dict[str, pd.DataFrame] = dict(list(df.groupby(id_col)))
    for k, df_k in ts_dict.items():
        # Sort dates in ascending order
        df_k = df_k.sort_values(date_col, ascending=True)
        # Drop duplicated dates
        df_k = df_k.drop_duplicates(date_col, keep='last')
        # Set date index
        df_k = df_k.set_index(date_col, drop=True)
        # Order date index in ascending order
        df_k = df_k.sort_index(ascending=True)
        df_k.index.name = date_col  # rename index
        # Save time-series DataFrame
        ts_dict[k] = df_k[cols].copy()

    return ts_dict


def read_piezo(filename: str, id_col: str, date_col: str, cols: List[str]) -> Dict[str, pd.DataFrame]:
    """ Read piezo time-series """
    # Read excel or csv with piezo values
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    else:
        df = pd.read_excel(filename)
    # Transform date col into datetime object
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    df[date_col] = df[date_col].apply(lambda x: move_to_end_of_week_or_month(x, 'M'))

    ts_dict = create_ts_dict(df=df, id_col=id_col, date_col=date_col, cols=cols)

    return ts_dict


def read_context(filename: str, id_col: str, x_col: str, y_col: str) -> Dict[str, ContextType]:
    """ Read table in filename and extract context values """
    # Read excel or csv with context values
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    else:
        df = pd.read_excel(filename)

    # Create a dict with x and y coordinates for each id
    res = {
        row[id_col]: {'x': row[x_col], 'y': row[y_col]}
        for _, row in df.iterrows()
    }
    return res


def read_exogenous_series(filename) -> Dict[Tuple[float, float], pd.DataFrame]:
    """ Read exogenous data inside pickle file """
    with open(filename, "rb") as f:
        exg_dict = pickle.load(f)

    return exg_dict


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the distance between two points on the Earth's surface using the Haversine formula.
    """
    # Convert coordinates from degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    distance = 6371 * c  # Radius of the Earth in kilometers

    return distance


def search_nearest_long_lat(
        x_epsg_3035: float,
        y_epsg_3035: float,
        coords: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Find the nearest coordinates in an array of longitudes and latitudes based on a given EPSG:3035 pair.
    """

    # Transform x and y EPSG:3035 into longitude and latitude coordinates EPSG:4326
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326")
    y_lat, x_lon = transformer.transform(y_epsg_3035, x_epsg_3035)

    assert 42.2 <= y_lat <= 47.45
    assert 6.4 <= x_lon <= 14.4

    # Compute haversine distance for each pair of longitude and latitude
    dist_arr = [haversine(x_lon, y_lat, coord_lon, coord_lat) for (coord_lon, coord_lat) in coords]
    # Find and return the nearest pair
    nearest_id = np.argmin(dist_arr)
    nearest_coordinates = coords[nearest_id]
    return nearest_coordinates


def link_exogenous_series(
        ex_dict: Dict[Tuple[float, float], pd.DataFrame],
        coords_dict: Dict[str, ContextType]
) -> Dict[str, pd.DataFrame]:
    """
    Find the nearest exogenous series for each pair of input coordinates
    """
    res = {}  # empty dictionary for saving exogenous series for each pair of input coordinates
    coords = list(ex_dict.keys())  # Read the coordinates for each exogenous series
    for k, ctx in coords_dict.items():
        # Nearest long and lat
        long, lat = search_nearest_long_lat(ctx['x'], ctx['y'], coords)
        # Save
        res[k] = ex_dict[(long, lat)]
    return res


def create_spatial_matrix(coords_dict: Dict[str, ContextType], with_haversine: bool = False) -> pd.DataFrame:
    """ Create the spatial matrix """
    # Read id array
    ids = list(coords_dict.keys())
    # Extract x and y coords for each point
    xy_data = [{'x': val['x'], 'y': val['y']} for val in coords_dict.values()]
    xy_data = pd.DataFrame(xy_data).values
    if not with_haversine:
        # Compute pairwise euclidean distances for each pair of coords
        dist_matrix = pairwise_distances(xy_data)
    else:
        # Compute pairwise haversine distances for each pair of coords
        dist_matrix = haversine_distances(xy_data)

    dist_matrix = pd.DataFrame(dist_matrix, columns=ids, index=ids)
    return dist_matrix


def load_piezo_data(
        ts_filename: str,
        context_filename: str,
        ex_filename: str
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
    # Read irregular piezo time series
    ts_dict = read_piezo(ts_filename, id_col='Codice WISE stazione', date_col='Data', cols=['Piezometria (m)'])
    # Read time series context (i.e. coordinates)
    ctx_dict = read_context(context_filename, id_col='Codice WISE stazione', x_col='X EPSG:3035',
                            y_col='Y EPSG:3035')

    # Remove time series without context information
    keys = list(ts_dict.keys())
    for k in keys:
        if k not in ctx_dict:
            ts_dict.pop(k)

    # Remove context information without time-series
    keys = list(ctx_dict.keys())
    for k in keys:
        if k not in ts_dict:
            # print(k)
            ctx_dict.pop(k)

    # Read all exogenous series
    exg_dict = read_exogenous_series(ex_filename)
    # Link exogenous series with each irregular time series
    exg_dict = link_exogenous_series(exg_dict, ctx_dict)
    # Create distance matrix for each pair of irregular time series
    dist_matrix = create_spatial_matrix(ctx_dict, with_haversine=False)
    spt_dict = {}
    for k in ts_dict.keys():
        dists = dist_matrix.loc[k]
        dists = dists.drop(k)
        dists = dists.sort_values(ascending=True)
        spt_dict[k] = dists

    # exg_dict = {}
    return ts_dict, exg_dict, spt_dict
