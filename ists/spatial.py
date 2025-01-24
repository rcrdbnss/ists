from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from ists.dataset.utils import transpose_dict_of_dicts
from ists.preprocessing import time_encoding


def array_mapping(id_array: np.ndarray, time_array: np.ndarray, dist_array: np.ndarray) -> Dict[str, pd.DataFrame]:
    maps = pd.DataFrame({
        'Id': id_array,
        'Time': time_array,
        'Null': np.max(dist_array, axis=1),
        'Idx': np.arange(len(id_array))
    })
    maps = dict(list(maps.groupby('Id')))
    for k, df in maps.items():
        k_map = pd.DataFrame({'Idx': df['Idx'].values, 'Null': df['Null'].values}, index=df['Time'].values)
        maps[k] = k_map
    return maps


def prepare_spatial_data(
        x_array: np.ndarray,
        id_array: np.ndarray,
        time_array: np.ndarray,
        dist_x_array: np.ndarray,
        num_past: int,
        num_spt: int,
        spt_dict: Dict[str, pd.Series],
        max_dist_th: float = None,
        max_null_th: int = None,
        y_array: np.ndarray = None
) -> Tuple[List[np.array], np.ndarray]:
    max_null_th = max_null_th if max_null_th else float('inf')
    max_dist_th = max_dist_th if max_dist_th else float('inf')

    arr_mapping = array_mapping(id_array, time_array, dist_x_array)
    ids = np.unique(id_array)

    # for k, s in spt_dict.items():
    #     s = s[(s.index.isin(ids)) & (s < np.inf)]
    #     if len(s) >= num_spt:
    #         spt_dict[k] = s[:num_spt]
    #     else:
    #         spt_dict[k] = s[0:0]
    """spt_dict = {
        k: s[(s.index.isin(ids)) & (s < np.inf)] for k, s in spt_dict.items()
    }"""
    for k, s in spt_dict.items():
        s = s[(s.index.isin(ids)) & (s < np.inf)]
        if len(s) >= num_spt:
            spt_dict[k] = s[:num_spt]
        else:
            spt_dict[k] = s[0:0]

    id_array_chunks = {id_array[0]: [0, 1]}
    for i in range(1, len(id_array)):
        if id_array[i] != id_array[i - 1]:
            id_array_chunks[id_array[i]] = [i, i]
        id_array_chunks[id_array[i]][1] += 1

    spt_array = []
    mask = []
    # y_spt_array = []
    for rid, chunk in tqdm(id_array_chunks.items(), desc='Station'):
        dists = spt_dict[rid]
        if len(dists) < num_spt:
            mask.append([False] * (chunk[1] - chunk[0]))
            continue

        df = arr_mapping[rid]
        rows_mask = df['Null'] <= max_null_th
        # if sum(rows_mask) == 0:
        #     mask.append([False] * (chunk[1] - chunk[0]))
        #     continue

        time_index = pd.DataFrame(index=pd.Index(time_array[chunk[0]:chunk[1]]))
        df = df.loc[rows_mask]
        idx_df = time_index.merge(df[['Idx']].rename(columns={'Idx': rid}), how='left', left_index=True,
                                  right_index=True)
        null_df = time_index.merge(df[['Null']].rename(columns={'Null': rid}), how='left', left_index=True,
                                   right_index=True)

        if num_spt == 0:
            x0, x1, x2 = x_array.shape
            spt_array.append(np.zeros((0, 0, x1, x2)))
            mask.append(rows_mask.values)
            continue

        for stn in dists.index:
            df = arr_mapping[stn]
            rows_mask = df['Null'] <= max_null_th
            df = df.loc[rows_mask]
            idx_df = idx_df.merge(df[['Idx']].rename(columns={'Idx': stn}), how='left', left_index=True,
                                  right_index=True)
            null_df = null_df.merge(df[['Null']].rename(columns={'Null': stn}), how='left', left_index=True,
                                    right_index=True)
        idx_df, null_df = idx_df.values[:, 1:], null_df.values[:, 1:]

        matr_mask = ~np.isnan(null_df)
        # cols_mask = np.array([True] * idx_df.shape[1])
        # cols_mask[np.argpartition(-matr_mask.sum(axis=0), num_spt-1)[num_spt:]] = False
        # idx_df, null_df, matr_mask = idx_df[:, cols_mask], null_df[:, cols_mask], matr_mask[:, cols_mask]
        rows_mask = matr_mask.sum(axis=1) >= num_spt

        x_array_arg = idx_df[rows_mask]
        spt_array.append(x_array[x_array_arg.T.astype(int)])
        mask.append(rows_mask)
        # print(rid, np.concatenate(mask).sum() / np.concatenate(mask).shape[0])
        # y_spt_array_ = y_array[x_array_arg.T.astype(int)].transpose(1, 0, 2)  # (B, V, 1)
        # B, V, _ = y_spt_array_.shape
        # y_spt_array_ = y_spt_array_.reshape(B, V)
        # y_spt_array.append(y_spt_array_)

    spt_array = np.concatenate(spt_array, axis=1)[:, :, -num_past:, :]
    spt_array = [x for x in spt_array]
    mask = np.concatenate(mask)
    # y_spt_array = np.concatenate(y_spt_array, axis=0)
    # return spt_array, mask, y_spt_array
    return spt_array, mask, None


def prepare_spatial_data_xy(
        x_array: np.ndarray,
        id_array: np.ndarray,
        time_array: np.ndarray,
        dist_x_array: np.ndarray,
        num_past: int,
        num_spt: int,
        spt_dict: Dict[str, pd.Series],
        max_dist_th: float = None,
        max_null_th: int = None,
        y_array: np.ndarray = None
) -> Tuple[List[np.array], np.ndarray]:
    max_null_th = max_null_th if max_null_th else np.float('inf')
    max_dist_th = max_dist_th if max_dist_th else np.float('inf')

    arr_mapping = array_mapping(id_array, time_array, dist_x_array)
    ids = np.unique(id_array)

    spt_dict = {
        k: s[(s.index.isin(ids)) & (s <= max_dist_th)] for k, s in spt_dict.items()
    }

    id_array_chunks = {id_array[0]: [0, 1]}
    for i in range(1, len(id_array)):
        if id_array[i] != id_array[i - 1]:
            id_array_chunks[id_array[i]] = [i, i]
        id_array_chunks[id_array[i]][1] += 1

    spt_array = []
    mask = []
    for rid, chunk in tqdm(id_array_chunks.items(), desc='Station'):
        dists = spt_dict[rid]
        if len(dists) < num_spt:
            mask.append([False] * (chunk[1] - chunk[0]))
            continue

        df = arr_mapping[rid]
        rows_mask = df['Null'] <= max_null_th
        time_index = pd.DataFrame(index=pd.Index(time_array[chunk[0]:chunk[1]]))
        df = df.loc[rows_mask]
        idx_df = time_index.merge(df[['Idx']].rename(columns={'Idx': rid}), how='left', left_index=True,
                                  right_index=True)
        null_df = time_index.merge(df[['Null']].rename(columns={'Null': rid}), how='left', left_index=True,
                                   right_index=True)

        if num_spt == 0:
            x0, x1, x2 = x_array.shape
            spt_array.append(np.zeros((0, 0, x1, x2)))
            mask.append(rows_mask.values)
            continue

        for stn in dists.index:
            df = arr_mapping[stn]
            rows_mask = df['Null'] <= max_null_th
            df = df.loc[rows_mask]
            idx_df = idx_df.merge(df[['Idx']].rename(columns={'Idx': stn}), how='left', left_index=True,
                                  right_index=True)
            null_df = null_df.merge(df[['Null']].rename(columns={'Null': stn}), how='left', left_index=True,
                                    right_index=True)
        idx_df, null_df = idx_df.values[:, 1:], null_df.values[:, 1:]

        matr_mask = ~np.isnan(null_df)
        cols_mask = np.array([True] * idx_df.shape[1])
        cols_mask[np.argpartition(-matr_mask.sum(axis=0), num_spt-1)[num_spt:]] = False
        idx_df, null_df, matr_mask = idx_df[:, cols_mask], null_df[:, cols_mask], matr_mask[:, cols_mask]
        rows_mask = matr_mask.sum(axis=1) >= num_spt

        x_array_arg = idx_df[rows_mask]
        spt_array.append(x_array[x_array_arg.T.astype(int)])
        mask.append(rows_mask)

    spt_array = np.concatenate(spt_array, axis=1)[:, :, -num_past:, :]
    spt_array = [x for x in spt_array]
    mask = np.concatenate(mask)
    return spt_array, mask, None


def _prepare_spatial_data(
        x_array: np.ndarray,
        id_array: np.ndarray,
        time_array: np.ndarray,
        dist_x_array: np.ndarray,
        num_past: int,
        num_spt: int,
        spt_dict: Dict[str, pd.Series],
        max_dist_th: float = None,
        max_null_th: int = None,
) -> Tuple[List[np.array], np.ndarray]:
    # max_null_th = max_null_th if max_null_th else np.float('inf')
    # max_dist_th = max_dist_th if max_dist_th else np.float('inf')
    #
    # spt_array = [[] for _ in range(num_spt)]
    # spt_array_stations = [[] for _ in range(num_spt)]  # todo: remove
    # mask = []
    # arr_mapping = array_mapping(id_array, time_array, dist_x_array)
    # # Keep only spatial id that can be used
    # spt_dict = {k: s.index[(s.index.isin(id_array)) & (s <= max_dist_th)] for k, s in spt_dict.items()}
    # for rid, rtime in tqdm(zip(id_array, time_array)):
    #     # Keep only spatial id that can be used
    #     spt_ids = spt_dict[rid]
    #     spt_count = 0
    #     spt_mem = []
    #     stations = []  # todo: remove
    #     for spt_id in spt_ids:
    #         if rtime in arr_mapping[spt_id].index and arr_mapping[spt_id].loc[rtime, 'Null'] <= max_null_th:
    #             spt_count += 1
    #             spt_mem.append(arr_mapping[spt_id].loc[rtime, 'Idx'])
    #             stations.append(spt_id)  # todo: remove
    #
    #         if spt_count == num_spt:
    #             break
    #
    #     assert spt_count <= num_spt
    #     if spt_count == num_spt:
    #         mask.append(True)
    #         for i in range(num_spt):
    #             spt_array[i].append(x_array[spt_mem[i]])
    #             spt_array_stations[i].append(stations[i]) #  todo : remove
    #     else:
    #         mask.append(False)
    #
    # spt_res_array = []
    # for i in range(num_spt):
    #     spt_res_array.append(np.array(spt_array[i])[:, -num_past:, :])
    # return spt_res_array, np.array(mask)
    ...


def find_last_date_idx(df: pd.DataFrame, date: datetime.date, th: int = 7) -> int:
    df_slice = df[df.index <= date]
    if df_slice.empty:
        return -1

    last_date = df_slice.index[-1]
    last_date_idx = df.index.get_loc(last_date)

    if (date - last_date).days > th:
        return -1
    else:
        return last_date_idx


def prepare_exogenous_data(
        exg_dict: Dict[str, pd.DataFrame],
        features: List[str],
        time_feats: List[str],
        # null_feat: Literal['code_bool', 'code_lin', 'bool', 'log', 'lin'], # = None,  # DEPRECATED
        # null_max_dist: int, # = None,  # DEPRECATED
):
    exg_dict_feats = dict()
    for f in features:
        exg_dict_feats[f] = dict()
        for stn, df in exg_dict.items():
            exg_dict_feats[f][stn] = df[[f]].copy()

    for f, dt in exg_dict_feats.items():
        for stn, ts in dt.items():
            # ts = drop_first_nan_rows(ts, [f], reverse=False)  # todo: serve, se tengo solo null_indicator=code_bool?
            # ts = drop_first_nan_rows(ts, [f], reverse=True)  # todo: serve, se tengo solo null_indicator=code_bool?
            # ts['NullFeat'] = null_indicator(ts, method=null_feat, max_dist=null_max_dist)  # DEPRECATED
            ts['NullFeat'] = pd.isnull(ts[f]).astype(int)
            ts = time_encoding(ts, codes=time_feats)
            ts[[f, 'NullFeat']] = ts[[f, 'NullFeat']].ffill()
            # ts = ts.dropna()  # drop nan values at the beginning that could not be forward-filled
            # ts["NullDist"] = null_distance_array(ts.dropna()["NullFeat"])
            dt[stn] = ts

    return exg_dict_feats


def exg_sliding_window_arrays(
        exg_dict_feats: dict[str, dict[Any, pd.DataFrame]],
        id_array: np.ndarray,
        time_array: np.ndarray,
        num_past: int,
        num_fut: int = None,
        max_null_th: int = None,
):
    max_null_th = max_null_th if max_null_th else float('inf')

    id_array_chunks = {id_array[0]: [0, 1]}
    for i in range(1, len(id_array)):
        if id_array[i] != id_array[i - 1]:
            id_array_chunks[id_array[i]] = [i, i]
        id_array_chunks[id_array[i]][1] += 1

    # exg_dict = transpose_dict_of_dicts(exg_dict_feats)
    exg_array_feats, mask = {f: [] for f in exg_dict_feats.keys()}, []
    # y_exg_array = []

    for rid, chunk in tqdm(id_array_chunks.items(), desc='Station'):
        time_end = pd.to_datetime(time_array[chunk[0]:chunk[1]])
        time_start = time_end - pd.to_timedelta(num_past - 1, 'D')
        # time_target = time_end + pd.to_timedelta(num_fut, 'D')
        df_start, df_end = pd.DataFrame(index=time_start), pd.DataFrame(index=time_end)
        # df_y, df_y_null = pd.DataFrame(index=time_target), pd.DataFrame(index=time_target)
        # for f, df_ in exg_dict[rid].items():
        for f, exg_dict_f in exg_dict_feats.items():
            df_ = exg_dict_f[rid]
            df_start = df_start.merge(df_[[f]], how='left', left_index=True, right_index=True)
            df_end = df_end.merge(df_[[f]], how='left', left_index=True, right_index=True)
            # df_y = df_y.merge(df_[[f]], how='left', left_index=True, right_index=True)
            # df_y_null = df_y_null.merge(df_[['NullFeat']].rename(columns={'NullFeat': f}), how='left', left_index=True, right_index=True)
        if df_start.isna().any(axis=1).all() or df_end.isna().any(axis=1).all():
            mask.append([False] * (chunk[1] - chunk[0]))
            continue
        mask_end = (~df_end.isna()).all(axis=1).values
        df_end['time'] = df_end.index
        df_end.loc[~mask_end, 'time'] = np.nan
        df_end['time'] = df_end['time'].ffill(limit=7)
        mask_end = (~df_end['time'].isna()).values
        num_past_ = pd.to_timedelta(num_past - 1, 'D')

        mask_start = (~df_start.isna()).all(axis=1)
        mask_start[pd.NaT] = False
        mask_start = mask_start[(df_end['time'].ffill() - num_past_).values]
        mask_chunk = mask_start & mask_end
        mask_chunk = mask_chunk.values

        end = df_end['time'][mask_chunk]
        start = end - num_past_
        start = start.dt.date.values
        end = end.dt.date.values
        # start = df_start.index[mask_chunk].date
        # end = df_end.index[mask_chunk].date
        # start = df_start.index.date
        # end = df_end.index.date
        """exg_array_feats_ = {f: [] for f in exg_dict[rid].keys()}
        mask_exg_array_feats_ = {f: [] for f in exg_dict[rid].keys()}"""
        # exg_array_feats_ = {f: [] for f in exg_dict_feats.keys()}
        # mask_exg_array_feats_ = {f: [] for f in exg_dict_feats.keys()}
        for f, exg_dict_f in exg_dict_feats.items():
            df_ = exg_dict_f[rid]
        # for f, df_ in exg_dict[rid].items():
            for s, e in zip(start, end):
                exg_array_feats[f].append(df_.loc[s:e].values)
            """null_dist = df_["NullDist"]
            df_ = df_.drop("NullDist", axis=1)
            for i, (s, e, m) in enumerate(zip(start, end, mask_chunk)):
                window, null_dist_window = df_.loc[s:e], null_dist.loc[s:e]
                exg_array_feats_[f].append(window.values)
                if not m or null_dist_window.max() > max_null_th:
                    mask_exg_array_feats_[f].append(False)
                else:
                    mask_exg_array_feats_[f].append(True)"""
        """mask_exg_array_feats_ = pd.DataFrame.from_dict(mask_exg_array_feats_).all(axis=1).values
        for f in exg_dict[rid]:
        # for f in exg_dict_feats.keys():
            exg_array_feats[f].extend(np.array(exg_array_feats_[f], dtype="object")[mask_exg_array_feats_].tolist())
        mask_chunk = mask_chunk & mask_exg_array_feats_"""
        mask.append(mask_chunk)
        # y, y_null = df_y.values[mask_chunk], df_y_null.values[mask_chunk].astype(bool)
        # y[y_null] = np.nan
        # y_exg_array.append(y)

    mask = np.concatenate(mask)
    exg_array_feats = [np.array(arr, dtype=float) for arr in exg_array_feats.values()]
    # y_exg_array = np.concatenate(y_exg_array)
    return exg_array_feats, mask # y_exg_array


def exg_sliding_window_arrays_adbpo(
        id_array: np.ndarray,
        time_array: np.ndarray,
        # exg_dict: Dict[str, pd.DataFrame],
        exg_dict_feats,
        num_past: int,
        # features: List[str],
        # time_feats: List[str],
        max_null_th: int = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    max_null_th = max_null_th if max_null_th else float('inf')

    stations = set()
    for f in exg_dict_feats.keys():
        stations.update(exg_dict_feats[f].keys())

    exg_dict = dict()
    for stn in stations:
        df = list()
        for f in exg_dict_feats.keys():
            df_f_stn = exg_dict_feats[f][stn]
            df.append(df_f_stn.rename(columns={col: f'{f}_{col}' for col in df_f_stn.columns}))
        df = pd.concat(df, axis=1)
        exg_dict[stn] = df
    else:
        cols = exg_dict[stn].columns

    exg_array = []
    mask = []

    for rid, rtime in tqdm(zip(id_array, time_array)):
        df: pd.DataFrame = exg_dict[rid]
        # df = time_encoding(df, codes=time_feats)
        # df = df[features + time_feats]

        right = find_last_date_idx(df, date=rtime)
        if right != -1:  # [features]
            right = right + 1
            left = right - num_past
            if left >= 0:
                mask.append(True)
                exg_array.append(df.iloc[left:right].values)
                # window = df.iloc[left:right]
                # if window.loc[:, window.columns.str.contains("NullDist")].max().max() <= max_null_th:
                #     exg_array.append(window.values)
                #     mask.append(True)
                # else:
                #     mask.append(False)
            else:
                mask.append(False)
        else:
            mask.append(False)

    exg_array = np.array(exg_array, dtype=float)
    exg_array_feats = []
    for f in exg_dict_feats.keys():
        # exg_array_feats.append(exg_array[:, :, cols.str.startswith(f'{f}_') & ~cols.str.contains("NullDist")])
        exg_array_feats.append(exg_array[:, :, cols.str.startswith(f'{f}_')])
    return exg_array_feats, np.array(mask)
