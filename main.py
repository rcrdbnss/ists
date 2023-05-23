from typing import Literal
import os

from ists.dataset.piezo.read import load_data
from ists.preparation import prepare_data
from ists.spatial import prepare_exogenous_data, prepare_spatial_data


def main():
    base_dir = '../../Dataset/AdbPo/Piezo/'
    ts_filename = os.path.join(base_dir, 'ts_er.xlsx')  # piezo time-series
    ctx_filename = os.path.join(base_dir, 'data_ext_er.xlsx')  # table with context information (i.e. coordinates...)
    ex_filename = os.path.join(base_dir, 'NetCDF', 'exg_tp_t2m.pickle')  # exogenous variable (i.e. temperature...)

    # Input time-series params
    features = ['Piezometria (m)']
    label_col = 'Piezometria (m)'
    num_past = 24
    num_fut = 12
    max_dist = 12
    freq: Literal['M', 'W', 'D'] = 'M'
    null_feat: Literal['bool', 'lin', 'log'] = 'log'

    # Exogenous time-series params
    exg_params = {
        'num_past': 36,
        # 'features': ['t2m']
    }

    spt_params = {
        'num_past': 12,
        'num_spt': 5
    }

    label_th = 2
    transform = 'std'  # 'minmax' 'std'

    # Load dataset
    ts_dict, exg_dict, spt_dict = load_data(
        ts_filename=ts_filename,
        context_filename=ctx_filename,
        ex_filename=ex_filename
    )

    # Prepare x, y, time, dist, id matrix
    x_array, y_array, time_array, dist_array, id_array = prepare_data(
        ts_dict=ts_dict,
        features=features,
        label_col=label_col,
        num_past=num_past,
        num_fut=num_fut,
        freq=freq,
        null_feat=null_feat,
        max_dist=max_dist
    )
    print(f'Num of records: {len(x_array)}')

    # Prepare spatial matrix
    spt_array, mask = prepare_spatial_data(
        x_array=x_array,
        id_array=id_array,
        time_array=time_array[:, 1],
        num_past=spt_params['num_past'],
        num_spt=spt_params['num_spt'],
        spt_dict=spt_dict,
    )
    x_array = x_array[mask]
    y_array = y_array[mask]
    time_array = time_array[mask]
    dist_array = dist_array[mask]
    id_array = id_array[mask]
    print(f'Num of records: {len(x_array)}')

    # Prepare exogenous matrix
    exg_array, mask = prepare_exogenous_data(
        id_array=id_array,
        time_array=time_array[:, 1],
        exg_dict=exg_dict,
        num_past=exg_params['num_past'],
        # features=exg_params['features']
    )
    x_array = x_array[mask]
    y_array = y_array[mask]
    time_array = time_array[mask]
    dist_array = dist_array[mask]
    id_array = id_array[mask]
    spt_array = spt_array[mask]
    print(f'Num of records: {len(x_array)}')

    mask = dist_array > label_th
    x_array = x_array[mask]
    y_array = y_array[mask]
    time_array = time_array[mask]
    dist_array = dist_array[mask]
    id_array = id_array[mask]
    spt_array = spt_array[mask]
    exg_array = exg_array[mask]

    print(f'Num of records: {len(x_array)}')

    print('Hello World!')


if __name__ == '__main__':
    main()
