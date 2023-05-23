from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def array_mapping(id_array: np.ndarray, time_array: np.ndarray) -> Dict[str, pd.Series]:
    maps = pd.DataFrame({
        'Id': id_array,
        'Time': time_array,
        'Idx': np.arange(len(id_array))
    })
    maps = dict(list(maps.groupby('Id')))
    for k, df in maps.items():
        k_map = pd.Series(df['Idx'].values, index=df['Time'].values)
        maps[k] = k_map
    return maps


def prepare_spatial_data(
        x_array: np.ndarray,
        id_array: np.ndarray,
        time_array: np.ndarray,
        num_past: int,
        num_spt: int,
        spt_dict: Dict[str, pd.Series]
) -> Tuple[List[np.array], np.ndarray]:
    spt_array = [[] for _ in range(num_spt)]
    mask = []
    arr_mapping = array_mapping(id_array, time_array)
    for rid, rtime in tqdm(zip(id_array, time_array)):
        spt_ids = spt_dict[rid].index.values
        spt_count = 0
        spt_mem = []
        for spt_id in spt_ids:
            if rtime in arr_mapping[spt_id]:
                spt_count += 1
                spt_mem.append(arr_mapping[spt_id][rtime])

            if spt_count == num_spt:
                break

        assert spt_count <= num_spt
        if spt_count == num_spt:
            mask.append(True)
            for i in range(num_spt):
                spt_array[i].append(x_array[spt_mem[i]])
        else:
            mask.append(False)

    spt_res_array = []
    for i in range(num_spt):
        spt_res_array.append(np.array(spt_array[i])[:, -num_past:, :])

    return spt_res_array, np.array(mask)


def prepare_exogenous_data(
        id_array: np.ndarray,
        time_array: np.ndarray,
        exg_dict: Dict[str, pd.DataFrame],
        num_past: int,
        # features: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    exg_array = []
    mask = []
    for rid, rtime in tqdm(zip(id_array, time_array)):
        df: pd.DataFrame = exg_dict[rid]
        if rtime in df.index:  # [features]
            right = df.index.get_loc(rtime) + 1
            left = right - num_past
            if left >= 0:
                mask.append(True)
                exg_array.append(df.iloc[left:right].values)
            else:
                mask.append(False)
        else:
            mask.append(False)

    exg_array = np.array(exg_array)
    return np.array(exg_array), np.array(mask)
