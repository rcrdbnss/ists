import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ists.dataset.read import load_data
from ists.preparation import define_feature_mask, prepare_train_test, get_list_null_max_size, prepare_data, \
    sliding_window_arrays, filter_data
from ists.preprocessing import get_time_max_sizes
from ists.spatial import prepare_spatial_data, prepare_spatial_data_xy, prepare_exogenous_data, \
    exg_sliding_window_arrays_adbpo, exg_sliding_window_arrays
from ists.utils import IQRMasker


def parse_params():
    """ Parse input parameters. """

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='the path where the configuration is stored.')
    parser.add_argument('--dev', action='store_true', help='Run on development data')
    parser.add_argument('--cpu', action='store_true', help='Run on CPU')
    parser.add_argument('--num-past', type=int, default=0, help='Number of past values to consider')
    parser.add_argument('--num-fut', type=int, default=0, help='Number of future values to predict')
    parser.add_argument('--nan-percentage', type=float, default=-1., help='Percentage of NaN values to insert')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--force-data-step', action='store_true', help='Force data step')

    args = parser.parse_args()
    print(args)
    conf_file = args.file
    assert os.path.exists(conf_file), 'Configuration file does not exist'

    with open(conf_file, 'r') as f:
        conf = json.load(f)
    conf['path_params']['dev'] = args.dev
    conf['path_params']['force_data_step'] = args.force_data_step
    conf['model_params']['seed'] = args.seed
    if args.num_past > 0:
        conf['prep_params']['ts_params']['num_past'] = args.num_past
    if args.num_fut > 0:
        conf['prep_params']['ts_params']['num_fut'] = args.num_fut
    if args.nan_percentage >= 0:
        conf['path_params']['nan_percentage'] = args.nan_percentage
    if not conf['path_params']['ex_filename']:
        conf['path_params']['ex_filename'] = 'all'
    ex_name = conf['path_params']['ex_filename']

    if 'patience' not in conf['model_params']:
        conf['model_params']['patience'] = None

    if args.dev:
        ts_name, ts_ext = os.path.splitext(conf['path_params']['ts_filename'])
        conf['path_params']['ts_filename'] = f"{ts_name}_dev{ts_ext}"
        if ex_name == 'all':
            conf['path_params']['ex_filename'] = 'all_dev'
        else:
            ex_name, ex_ext = os.path.splitext(ex_name)
            conf['path_params']['ex_filename'] = f"{ex_name}_dev{ex_ext}"
        if conf['path_params']['ctx_filename']:
            ctx_name, ctx_ext = os.path.splitext(conf['path_params']['ctx_filename'])
            conf['path_params']['ctx_filename'] = f"{ctx_name}_dev{ctx_ext}"
        conf['model_params']['epochs'] = 3
        conf['model_params']['patience'] = 1
        # if conf['path_params']['type'] == 'french':
        #     conf['eval_params']['test_start'] = '2017-07-01'
        #     if 'valid_start' in conf['eval_params']:
        #         conf['eval_params']['valid_start'] = '2017-01-01'

    # if args.cpu:
    #     tf.config.set_visible_devices([], 'GPU')
    conf['model_params']['cpu'] = args.cpu

    return conf['path_params'], conf['prep_params'], conf['eval_params'], conf['model_params']


def apply_iqr_masker_by_stn(ts_dict, features_to_mask, train_end_excl):
    for stn in ts_dict:
        for f in features_to_mask:
            iqr_masker = IQRMasker()

            ts_train = ts_dict[stn].loc[ts_dict[stn].index < train_end_excl, f]
            if ts_train.isna().all(): continue
            ts_train = iqr_masker.fit_transform(ts_train.values.reshape(-1, 1))
            ts_dict[stn].loc[ts_dict[stn].index < train_end_excl, f] = ts_train.reshape(-1)

            ts_test = ts_dict[stn].loc[ts_dict[stn].index >= train_end_excl, f]
            if ts_test.isna().all(): continue
            ts_test = iqr_masker.transform(ts_test.values.reshape(-1, 1))
            ts_dict[stn].loc[ts_dict[stn].index >= train_end_excl, f] = ts_test.reshape(-1)

    return ts_dict


def apply_iqr_masker(ts_dict, features_to_mask, train_end_excl):
    for f in features_to_mask:
        ts_all_train = []
        for stn in ts_dict:
            ts = ts_dict[stn][f]
            ts_all_train.append(ts.loc[ts.index < train_end_excl].values)
        ts_all_train = np.concatenate(ts_all_train)
        iqr_masker = IQRMasker().fit(ts_all_train.reshape(-1, 1))

        for stn in ts_dict:
            ts_train = ts_dict[stn].loc[ts_dict[stn].index < train_end_excl, f]
            if ts_train.isna().all(): continue
            ts_train = iqr_masker.transform(ts_train.values.reshape(-1, 1))
            ts_dict[stn].loc[ts_dict[stn].index < train_end_excl, f] = ts_train.reshape(-1)

            ts_test = ts_dict[stn].loc[ts_dict[stn].index >= train_end_excl, f]
            if ts_test.isna().all(): continue
            ts_test = iqr_masker.transform(ts_test.values.reshape(-1, 1))
            ts_dict[stn].loc[ts_dict[stn].index >= train_end_excl, f] = ts_test.reshape(-1)

    return ts_dict


def apply_scaler_by_stn(ts_dict, features, train_end_excl, scaler_init):
    scalers = dict()
    for stn, ts in ts_dict.items():
        scalers[stn] = dict()
        for f in features:
            scaler = scaler_init()

            train_ts = ts.loc[ts.index < train_end_excl, f]
            if train_ts.isna().all(): continue
            train_ts = scaler.fit_transform(train_ts.values.reshape(-1, 1))
            ts.loc[ts.index < train_end_excl, f] = train_ts.reshape(-1)

            scalers[stn][f] = scaler

            test_ts = ts.loc[ts.index >= train_end_excl, f]
            if test_ts.isna().all(): continue
            test_ts = scaler.transform(test_ts.values.reshape(-1, 1))
            ts.loc[ts.index >= train_end_excl, f] = test_ts.reshape(-1)

    return ts_dict, scalers


def apply_scaler(ts_dict, features, train_end_excl, scaler_init):
    scalers = dict()
    for f in features:
        ts_all_train = []
        for stn in ts_dict:
            ts = ts_dict[stn][f]
            ts_all_train.append(ts.loc[ts.index < train_end_excl].values)
        ts_all_train = np.concatenate(ts_all_train)
        scaler = scaler_init().fit(ts_all_train.reshape(-1, 1))

        for stn in ts_dict:
            ts_train = ts_dict[stn].loc[ts_dict[stn].index < train_end_excl, f]
            if ts_train.isna().all(): continue
            ts_train = scaler.transform(ts_train.values.reshape(-1, 1))
            ts_dict[stn].loc[ts_dict[stn].index < train_end_excl, f] = ts_train.reshape(-1)

            ts_test = ts_dict[stn].loc[ts_dict[stn].index >= train_end_excl, f]
            if ts_test.isna().all(): continue
            ts_test = scaler.transform(ts_test.values.reshape(-1, 1))
            ts_dict[stn].loc[ts_dict[stn].index >= train_end_excl, f] = ts_test.reshape(-1)

        scalers[f] = scaler

    return ts_dict, scalers


def data_step(path_params: dict, prep_params: dict, eval_params: dict, keep_nan: bool = False, scaler_type=None):
    ts_params = prep_params["ts_params"]
    feat_params = prep_params["feat_params"]
    spt_params = prep_params["spt_params"]
    exg_params = prep_params["exg_params"]

    label_col = ts_params["label_col"]
    exg_cols = exg_params["features"]
    cols = [label_col] + exg_cols

    # Load dataset
    ts_dict, exg_dict, spt_dict = load_data(
        ts_filename=path_params['ts_filename'],
        context_filename=path_params['ctx_filename'],
        ex_filename=path_params['ex_filename'],
        data_type=path_params['type'],
        ts_features=[label_col],
        exg_features=exg_cols,
        nan_percentage=path_params['nan_percentage'],
        exg_cols_stn=exg_params['features_stn'] if 'features_stn' in exg_params else None,
        exg_cols_stn_scaler=scaler_type,
        num_past=ts_params['num_past'],
        num_future=ts_params['num_fut'],
        max_null_th=eval_params['null_th']
    )

    # ts_dict = {k: pd.concat([v, exg_dict[k]], axis=1) for k, v in ts_dict.items()}

    if 'valid_start' in eval_params and eval_params['valid_start']:
        train_end_excl = pd.to_datetime(eval_params['valid_start']).date()
    else:
        train_end_excl = pd.to_datetime(eval_params['test_start']).date()
        eval_params['valid_start'] = None

    scale_by_stn = {
        'french': True,
        'ushcn': True,
        'adbpo': True
    }[path_params['type']]
    if scale_by_stn:
        ts_dict = apply_iqr_masker_by_stn(ts_dict, ts_params['features'], train_end_excl)
        exg_dict = apply_iqr_masker_by_stn(exg_dict, exg_params['features'], train_end_excl)
    else:
        ts_dict = apply_iqr_masker(ts_dict, ts_params['features'], train_end_excl)
        exg_dict = apply_iqr_masker(exg_dict, exg_params['features'], train_end_excl)

    # train_end_excl = pd.to_datetime(eval_params["valid_start"]).date()

    # ts_dict = apply_iqr_masker(ts_dict, cols, train_end_excl)

    print(f'Null values after IQR masking')
    nan, tot = 0, 0
    for stn in ts_dict:
        nan += ts_dict[stn].isna().sum().sum()
        tot += ts_dict[stn].size
    print(f'  - Target: {nan}/{tot} ({nan/tot:.2%})')
    nan, tot = 0, 0
    for stn in exg_dict:
        nan += exg_dict[stn].isna().sum().sum()
        tot += exg_dict[stn][exg_params['features']].size
    print(f'  - Context: {nan}/{tot} ({nan/tot:.2%})')

    # nan, tot = 0, 0
    # for stn in ts_dict:
    #     for col in cols:
    #         nan += ts_dict[stn][col].isna().sum()
    #         tot += len(ts_dict[stn][col])
    # print(f"Null values after IQR masking: {nan}/{tot} ({nan/tot:.2%})")

    if scaler_type == "minmax":
        Scaler = MinMaxScaler
    elif scaler_type == "standard":
        Scaler = StandardScaler

    stns_no_data = list()
    for stn, ts in ts_dict.items():
        for f in ts_params['features']:
            ts_train = ts.loc[ts.index < train_end_excl, f]
            if ts_train.isna().all():
                stns_no_data.append(stn)
                continue

    for stn in stns_no_data:
        ts_dict.pop(stn)
        exg_dict.pop(stn)
        # spt_dict.pop(stn)

    if not scaler_type:
        ...
    elif scale_by_stn:
        print("Scaler:", scaler_type)
        ts_dict, spt_scalers = apply_scaler_by_stn(ts_dict, ts_params['features'], train_end_excl, Scaler)
        exg_dict, _ = apply_scaler_by_stn(exg_dict, exg_params['features'], train_end_excl, Scaler)
    else:
        print("Scaler:", scaler_type)
        ts_dict, spt_scalers = apply_scaler(ts_dict, ts_params['features'], train_end_excl, Scaler)
        spt_scalers_ = dict()
        for f, scaler in spt_scalers.items():
            for stn in ts_dict:
                if stn not in spt_scalers_:
                    spt_scalers_[stn] = dict()
                spt_scalers_[stn][f] = scaler
        spt_scalers = spt_scalers_
        exg_dict, _ = apply_scaler(exg_dict, exg_params['features'], train_end_excl, Scaler)

    # ts_dict, spt_scalers = apply_scaler_by_stn(ts_dict, cols, train_end_excl, Scaler)
    # spt_scalers = {
    #     stn: {label_col: vars(spt_scalers[stn][label_col])} for stn in spt_scalers
    # }

    ts_dict, new_features = prepare_data(
        ts_dict=ts_dict,
        features=ts_params['features'],
        label_col=ts_params['label_col'],
        freq=ts_params['freq'],
        # null_feat=feat_params['null_feat'],
        # null_max_dist=feat_params['null_max_dist'],
        time_feats=feat_params['time_feats'],
        with_fill=not keep_nan
    )

    x_array, y_array, time_array, dist_x_array, dist_y_array, id_array = sliding_window_arrays(
        ts_dict=ts_dict,
        num_past=ts_params['num_past'],
        num_fut=ts_params['num_fut'],
        features=ts_params['features'],
        new_features=new_features,
    )
    print(f'Num of records raw: {len(x_array)}')

    # exg_cols = exg_cols + (exg_params['features_stn'] if 'features_stn' in exg_params else [])
    # cols = [label_col] + exg_cols
    #
    time_feats = feat_params["time_feats"]
    # for stn, ts in ts_dict.items():
    #     for col in cols:
    #         ts[f"{col}_is_null"] = ts[col].isnull().astype(int)
    #     ts = time_encoding(ts, time_feats)
    #     ts[cols] = ts[cols].ffill()
    #     ts_dict[stn] = ts

    # Compute feature mask and time encoding max sizes
    x_feature_mask = define_feature_mask(
        base_features=[label_col],
        # null_feat=feat_params['null_feat'],
        null_feat="code_bool",
        time_feats=time_feats
    )
    x_time_max_sizes = get_time_max_sizes(time_feats)
    print(f'Feature mask: {x_feature_mask}')

    start_time = time.time()
    prepare_spatial_data_fn = {
        'french': prepare_spatial_data,
        'adbpo': prepare_spatial_data,
        'ushcn': prepare_spatial_data_xy
        # "french": link_spatial_data_water_body,
        # "ushcn": link_spatial_data,
    }[path_params['type']]
    spt_array, mask, y_spt_array = prepare_spatial_data_fn(
        x_array=x_array,
        id_array=id_array,
        time_array=time_array[:, 1],
        dist_x_array=dist_x_array,
        # num_past=spt_params['num_past'],
        num_past=ts_params['num_past'],
        num_spt=spt_params['num_spt'],
        spt_dict=spt_dict,
        max_dist_th=spt_params['max_dist_th'],
        max_null_th=eval_params['null_th'],
        y_array=y_array
    )
    # ts_dict = prepare_spatial_data_fn(
    #     ts_dict=ts_dict,
    #     label_col=label_col,
    #     exg_cols=exg_cols,
    #     num_spt=spt_params["num_spt"],
    #     spt_dict=spt_dict,
    #     max_dist_th=spt_params["max_dist_th"]
    # )
    end_time = time.time()
    print(f'Spatial augmentation execution time: {end_time - start_time:.2f} seconds')

    x_array = x_array[mask]
    y_array = y_array[mask]
    time_array = time_array[mask]
    dist_x_array = dist_x_array[mask]
    dist_y_array = dist_y_array[mask]
    id_array = id_array[mask]
    print(f'Num of records after spatial augmentation: {len(x_array)}')
    # y_array = np.concatenate([y_array, y_spt_array], axis=1)

    # Filter data before
    x_array, y_array, time_array, dist_x_array, dist_y_array, id_array, spt_array = filter_data(
        x_array=x_array,
        y_array=y_array,
        time_array=time_array,
        dist_x_array=dist_x_array,
        dist_y_array=dist_y_array,
        id_array=id_array,
        spt_array=spt_array,
        train_start=eval_params['train_start'],
        max_label_th=eval_params['label_th'],
        max_null_th=eval_params['null_th']
    )
    print(f'Num of records after null filter: {len(x_array)}')

    exg_dict = prepare_exogenous_data(
        exg_dict=exg_dict,
        features=exg_params['features'] + (exg_params['features_stn'] if 'features_stn' in exg_params else []),
        time_feats=exg_params['time_feats'],
        # null_feat=feat_params['null_feat'],
        # null_max_dist=feat_params['null_max_dist'],
    )

    exg_sliding_window_arrays_fn = {
        'french': exg_sliding_window_arrays,
        'adbpo': exg_sliding_window_arrays_adbpo,
        'ushcn': exg_sliding_window_arrays
    }[path_params['type']]

    # num_spt = spt_params["num_spt"]
    # cols = [label_col] + exg_cols + [f"spt{s}" for s in range(num_spt)]
    # for stn in list(ts_dict.keys()):
    #     if stn not in spt_dict:
    #         ts_dict.pop(stn)
    #         continue
    #
    #     ts = ts_dict[stn]
    #     for col in cols:
    #         ts[f"{col}_null_dist"] = null_distance_array(ts[f"{col}_is_null"])
    #     ts = ts.dropna(subset=cols)  # drop values that could not be forward filled
    #     ts_dict[stn] = ts

    start_time = time.time()
    exg_array, mask = exg_sliding_window_arrays_fn(
        exg_dict_feats=exg_dict,
        id_array=id_array,
        time_array=time_array[:, 1],
        num_past=ts_params['num_past'],
        max_null_th=eval_params['null_th'],
    )
    # x_array, exg_array, spt_array, y_array, time_array, id_array = extract_windows(
    #     ts_dict=ts_dict,
    #     label_col=label_col,
    #     exg_cols=exg_cols,
    #     num_spt=spt_params["num_spt"],
    #     time_feats=feat_params["time_feats"],
    #     num_past=ts_params["num_past"],
    #     num_fut=ts_params["num_fut"],
    #     max_null_th=eval_params["null_th"]
    # )
    end_time = time.time()
    print(f'Exogenous sliding window execution time: {end_time - start_time:.2f} seconds')
    # Compute exogenous feature mask and time encoding max sizes
    # exg_feature_mask = define_feature_mask(base_features=exg_params['features'], time_feats=exg_params['time_feats'])
    # exg_time_max_sizes = get_time_max_sizes(exg_params['time_feats'])

    x_array = x_array[mask]
    y_array = y_array[mask]
    time_array = time_array[mask]
    dist_x_array = dist_x_array[mask]
    dist_y_array = dist_y_array[mask]
    id_array = id_array[mask]
    spt_array = [arr[mask] for arr in spt_array]
    print(f'Num of records after exogenous augmentation: {len(x_array)}')
    # y_array = np.concatenate([y_array, y_exg_array], axis=1)

    # dist_x_array = np.zeros_like(x_array[:, :, 0])
    # dist_y_array = np.zeros_like(y_array[:, 0])

    res = prepare_train_test(
        x_array=x_array,
        y_array=y_array,
        time_array=time_array,
        dist_x_array=dist_x_array,
        dist_y_array=dist_y_array,
        id_array=id_array,
        spt_array=spt_array,
        exg_array=exg_array,
        test_start=eval_params['test_start'],
        valid_start=eval_params['valid_start'],
        spt_dict=spt_dict
    )
    print(f"X train: {len(res['x_train'])}")
    print(f"X valid: {len(res['x_valid'])}")
    print(f"X test: {len(res['x_test'])}")

    # Save extra params in train test dictionary
    # Save x and exogenous array feature mask
    res['x_feat_mask'] = x_feature_mask
    # res['exg_feat_mask'] = exg_feature_mask

    # Save null max size by finding the maximum between train and test if any
    arr_list = (
            [res['x_train']] + [res['x_test']] + [res['x_valid']] +
            res['spt_train'] + res['spt_test'] + res['spt_valid'] +
            res['exg_train'] + res['exg_test'] + res['exg_valid']
    )
    res['null_max_size'] = get_list_null_max_size(arr_list, x_feature_mask)

    # Save time max sizes
    res['time_max_sizes'] = x_time_max_sizes
    # res['exg_time_max_sizes'] = exg_time_max_sizes
    res['scalers'] = spt_scalers

    nan, tot = 0, 0
    for x in arr_list:
        nan += x[:, :, 1].sum().sum()
        tot += x[:, :, 1].size
    print(f"Null values in windows: {nan}/{tot} ({nan/tot:.2%})")

    return res


if __name__ == '__main__':
    path_params, prep_params, eval_params, model_params = parse_params()
    _seed = model_params['seed']
    if _seed is not None:
        random.seed(_seed)
        np.random.seed(_seed)

    # res_dir = './output/results'
    data_dir = './output/pickle' + ('_seed' + str(_seed) if _seed != 42 else '')
    # model_dir = './output/model' + ('_seed' + str(_seed) if _seed != 42 else '')

    # os.makedirs(res_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    # os.makedirs(model_dir, exist_ok=True)

    subset = path_params['ex_filename']
    if path_params['type'] == 'adbpo' and 'exg_w_tp_t2m' in subset:
        subset = os.path.basename(subset).replace('exg_w_tp_t2m', 'all').replace('.pickle', '')
    elif 'all' in subset:
        path_params['ex_filename'] = None
    else:
        subset = os.path.basename(subset).replace('subset_agg_', '').replace('.csv', '')
    nan_percentage = path_params['nan_percentage']
    num_past = prep_params['ts_params']['num_past']
    num_fut = prep_params['ts_params']['num_fut']
    num_spt = prep_params['spt_params']['num_spt']

    out_name = f"{path_params['type']}_{subset}_nan{int(nan_percentage * 10)}_np{num_past}_nf{num_fut}"
    print('out_name:', out_name)
    # results_path = os.path.join(res_dir, f"{out_name}.csv")
    pickle_path = os.path.join(data_dir, f"{out_name}.pickle")
    # checkpoint_path = os.path.join(model_dir, f"{out_name}")

    train_test_dict = data_step(
        path_params, prep_params, eval_params, keep_nan=False, scaler_type=model_params['transform_type']
    )

    with open(pickle_path, "wb") as f:
        print('Saving to', pickle_path, '...', end='', flush=True)
        pickle.dump(train_test_dict, f)
        print(' done!')
