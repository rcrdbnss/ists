import datetime
import os
import json
import pickle
import random
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ists import utils
from ists.dataset.read import load_data
from ists.model.encoder import SpatialExogenousEncoder
from ists.preparation import prepare_data, prepare_train_test, filter_data, sliding_window_arrays
from ists.preparation import define_feature_mask, get_list_null_max_size
from ists.preprocessing import get_time_max_sizes
from ists.spatial import prepare_exogenous_data, prepare_spatial_data, exg_sliding_window_arrays
from ists.model.wrapper import ModelWrapper
from ists.metrics import compute_metrics
from ists.utils import IQRMasker


def parse_params():
    """ Parse input parameters. """

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='the path where the configuration is stored.')
    parser.add_argument('--dev', action='store_true', help='Run on development data')
    parser.add_argument('--cpu', action='store_true', help='Run on CPU')
    parser.add_argument('--num-fut', type=int, default=0, help='Number of future values to predict')
    parser.add_argument('--nan-percentage', type=float, default=-1., help='Percentage of NaN values to insert')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    print(args)
    conf_file = args.file
    assert os.path.exists(conf_file), 'Configuration file does not exist'

    with open(conf_file, 'r') as f:
        conf = json.load(f)
    conf['path_params']['dev'] = args.dev
    conf['model_params']['seed'] = args.seed
    if args.num_fut > 0:
        conf['prep_params']['ts_params']['num_fut'] = args.num_fut
    if args.nan_percentage >= 0:
        conf['path_params']['nan_percentage'] = args.nan_percentage

    if args.dev:
        ts_name, ts_ext = os.path.splitext(conf['path_params']['ts_filename'])
        conf['path_params']['ts_filename'] = f"{ts_name}_dev{ts_ext}"
        ex_name, ex_ext = os.path.splitext(conf['path_params']['ex_filename'])
        conf['path_params']['ex_filename'] = f"{ex_name}_dev{ex_ext}"
        if conf['path_params']['ctx_filename']:
            ctx_name, ctx_ext = os.path.splitext(conf['path_params']['ctx_filename'])
            conf['path_params']['ctx_filename'] = f"{ctx_name}_dev{ctx_ext}"
        conf['model_params']['epochs'] = 2
        if conf['path_params']['type'] == 'french':
            conf['eval_params']['test_start'] = '2017-07-01'
            conf['eval_params']['valid_start'] = '2017-01-01'

    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')

    return conf['path_params'], conf['prep_params'], conf['eval_params'], conf['model_params']


def change_params(path_params: dict, base_string: str, new_string: str) -> dict:
    path_params['ts_filename'] = path_params['ts_filename'].replace(base_string, new_string, 1)
    path_params['ctx_filename'] = path_params['ctx_filename'].replace(base_string, new_string, 1)
    path_params['ex_filename'] = path_params['ex_filename'].replace(base_string, new_string, 1)

    return path_params


def data_step(path_params: dict, prep_params: dict, eval_params: dict, keep_nan: bool = False, scaler_type=None):
    ts_params = prep_params['ts_params']
    feat_params = prep_params['feat_params']
    spt_params = prep_params['spt_params']
    exg_params = prep_params['exg_params']

    # Load dataset
    ts_dict, exg_dict, spt_dict = load_data(
        ts_filename=path_params['ts_filename'],
        context_filename=path_params['ctx_filename'],
        ex_filename=path_params['ex_filename'],
        data_type=path_params['type'],
        nan_percentage=path_params['nan_percentage']
    )

    valid_start = pd.to_datetime(eval_params['test_start']).date()

    for k in ts_dict:
        for f in ts_params['features']:
            iqr_masker = IQRMasker()

            ts_train = ts_dict[k].loc[ts_dict[k].index < valid_start, f]
            if ts_train.isna().all(): continue
            ts_train = iqr_masker.fit_transform(ts_train.values.reshape(-1, 1))
            ts_dict[k].loc[ts_dict[k].index < valid_start, f] = ts_train.reshape(-1)

            ts_test = ts_dict[k].loc[ts_dict[k].index >= valid_start, f]
            if ts_test.isna().all(): continue
            ts_test = iqr_masker.transform(ts_test.values.reshape(-1, 1))
            ts_dict[k].loc[ts_dict[k].index >= valid_start, f] = ts_test.reshape(-1)

    for k in exg_dict:
        for f in exg_params['features']:
            iqr_masker = IQRMasker()

            ts_train = exg_dict[k].loc[exg_dict[k].index < valid_start, f]
            if ts_train.isna().all(): continue
            ts_train = iqr_masker.fit_transform(ts_train.values.reshape(-1, 1))
            exg_dict[k].loc[exg_dict[k].index < valid_start, f] = ts_train.reshape(-1)

            ts_test = exg_dict[k].loc[exg_dict[k].index >= valid_start, f]
            if ts_test.isna().all(): continue
            ts_test = iqr_masker.transform(ts_test.values.reshape(-1, 1))
            exg_dict[k].loc[exg_dict[k].index >= valid_start, f] = ts_test.reshape(-1)

    # spt_scalers = dict()
    for k, ts in ts_dict.items():
        # spt_scalers[k] = dict()
        for f in ts_params['features']:
            scaler = StandardScaler()
            # spt_scalers[k][f] = scaler

            train_ts = ts.loc[ts.index < valid_start, f]
            if train_ts.isna().all(): continue
            train_ts = scaler.fit_transform(train_ts.values.reshape(-1, 1))
            ts.loc[ts.index < valid_start, f] = train_ts.reshape(-1)

            test_ts = ts.loc[ts.index >= valid_start, f]
            if test_ts.isna().all(): continue
            test_ts = scaler.transform(test_ts.values.reshape(-1, 1))
            ts.loc[ts.index >= valid_start, f] = test_ts.reshape(-1)


    # exg_scalers = dict()
    for k, ts in exg_dict.items():
        # exg_scalers[k] = dict()
        for f in exg_params['features']:
            scaler = StandardScaler()
            # exg_scalers[k][f] = scaler

            train_ts = ts.loc[ts.index < valid_start, f]
            if train_ts.isna().all(): continue
            train_ts = scaler.fit_transform(train_ts.values.reshape(-1, 1))
            ts.loc[ts.index < valid_start, f] = train_ts.reshape(-1)

            test_ts = ts.loc[ts.index >= valid_start, f]
            if test_ts.isna().all(): continue
            test_ts = scaler.transform(test_ts.values.reshape(-1, 1))
            ts.loc[ts.index >= valid_start, f] = test_ts.reshape(-1)


    ts_dict_new, new_features = prepare_data(
        ts_dict=ts_dict,
        features=ts_params['features'],
        label_col=ts_params['label_col'],
        freq=ts_params['freq'],
        null_feat=feat_params['null_feat'],
        null_max_dist=feat_params['null_max_dist'],
        time_feats=feat_params['time_feats'],
        with_fill=not keep_nan
    )

    x_array, y_array, time_array, dist_x_array, dist_y_array, id_array = sliding_window_arrays(
        ts_dict=ts_dict_new,
        num_past=ts_params['num_past'],
        num_fut=ts_params['num_fut'],
        features=ts_params['features'],
        new_features=new_features,
    )
    print(f'Num of records raw: {len(x_array)}')
    # Compute feature mask and time encoding max sizes
    x_feature_mask = define_feature_mask(
        base_features=ts_params['features'],
        null_feat=feat_params['null_feat'],
        time_feats=feat_params['time_feats']
    )
    x_time_max_sizes = get_time_max_sizes(feat_params['time_feats'])
    print(f'Feature mask: {x_feature_mask}')

    # Prepare spatial matrix
    spt_array, mask = prepare_spatial_data(
        x_array=x_array,
        id_array=id_array,
        time_array=time_array[:, 1],
        dist_x_array=dist_x_array,
        num_past=spt_params['num_past'],
        num_spt=spt_params['num_spt'],
        spt_dict=spt_dict,
        max_dist_th=spt_params['max_dist_th'],
        max_null_th=spt_params['max_null_th']
    )
    x_array = x_array[mask]
    y_array = y_array[mask]
    time_array = time_array[mask]
    dist_x_array = dist_x_array[mask]
    dist_y_array = dist_y_array[mask]
    id_array = id_array[mask]
    print(f'Num of records after spatial augmentation: {len(x_array)}')

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

    exg_dict_new = prepare_exogenous_data(
        exg_dict=exg_dict,
        features=exg_params['features'],
        time_feats=exg_params['time_feats'],
        null_feat=feat_params['null_feat'],
        null_max_dist=feat_params['null_max_dist'],
    )

    exg_array, mask = exg_sliding_window_arrays(
        exg_dict_feats=exg_dict_new,
        id_array=id_array,
        time_array=time_array[:, 1],
        num_past=exg_params['num_past'],
    )
    # Compute exogenous feature mask and time encoding max sizes
    exg_feature_mask = define_feature_mask(base_features=exg_params['features'], time_feats=exg_params['time_feats'])
    exg_time_max_sizes = get_time_max_sizes(exg_params['time_feats'])

    x_array = x_array[mask]
    y_array = y_array[mask]
    time_array = time_array[mask]
    dist_x_array = dist_x_array[mask]
    dist_y_array = dist_y_array[mask]
    id_array = id_array[mask]
    spt_array = [arr[mask] for arr in spt_array]
    print(f'Num of records after exogenous augmentation: {len(x_array)}')

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
    )
    print(f"X train: {len(res['x_train'])}")
    print(f"X valid: {len(res['x_valid'])}")
    print(f"X test: {len(res['x_test'])}")

    # Save extra params in train test dictionary
    # Save x and exogenous array feature mask
    res['x_feat_mask'] = x_feature_mask
    res['exg_feat_mask'] = exg_feature_mask

    # Save null max size by finding the maximum between train and test if any
    res['null_max_size'] = get_list_null_max_size(
        [res['x_train']] + [res['x_test']] + [res['x_valid']] +
        res['spt_train'] + res['spt_test'] + res['spt_valid'] +
        res['exg_train'] + res['exg_test'] + res['exg_valid'],
        x_feature_mask
    )

    # Save time max sizes
    res['time_max_sizes'] = x_time_max_sizes
    res['exg_time_max_sizes'] = exg_time_max_sizes

    return res


def model_step(train_test_dict: dict, model_params: dict, checkpoint_dir: str) -> dict:
    model_type = model_params['model_type']
    transform_type = model_params['transform_type']
    nn_params = model_params['nn_params']
    loss = model_params['loss']
    lr = model_params['lr']
    epochs = model_params['epochs']
    batch_size = model_params['batch_size']

    # Insert data params in nn_params for building the correct model
    nn_params['feature_mask'] = train_test_dict['x_feat_mask']
    nn_params['exg_feature_mask'] = train_test_dict['exg_feat_mask']
    nn_params['spatial_size'] = len(train_test_dict['spt_train']) + 1 # target
    nn_params['exg_size'] = len(train_test_dict['exg_train']) + 1 # target
    nn_params['null_max_size'] = train_test_dict['null_max_size']
    nn_params['time_max_sizes'] = train_test_dict['time_max_sizes']
    nn_params['exg_time_max_sizes'] = train_test_dict['exg_time_max_sizes']
    if 'encoder_cls' in model_params:
        nn_params['encoder_cls'] = model_params['encoder_cls']

    transform_type = None  # todo: temporary

    model = ModelWrapper(
        checkpoint_dir=checkpoint_dir,
        model_type=model_type,
        model_params=nn_params,
        transform_type=transform_type,
        loss=loss,
        lr=lr,
        dev=train_test_dict['params']['path_params']['dev']
    )

    model.fit(
        x=train_test_dict['x_train'],
        spt=train_test_dict['spt_train'],
        exg=train_test_dict['exg_train'],
        y=train_test_dict['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
        id_array=train_test_dict['id_train'],
        # spt_scalers=train_test_dict['spt_scalers'], exg_scalers=train_test_dict['exg_scalers'],
        val_x=train_test_dict['x_valid'], val_spt=train_test_dict['spt_valid'], val_exg=train_test_dict['exg_valid'],
        val_y=train_test_dict['y_valid']
    )

    preds = model.predict(
        x=train_test_dict['x_test'],
        spt=train_test_dict['spt_test'],
        exg=train_test_dict['exg_test'],
    )

    res = {}
    res_test = compute_metrics(y_true=train_test_dict['y_test'], y_preds=preds)
    res_test = {f'test_{k}': val for k, val in res_test.items()}
    res.update(res_test)

    preds = model.predict(
        x=train_test_dict['x_train'],
        spt=train_test_dict['spt_train'],
        exg=train_test_dict['exg_train'],
    )
    res_train = compute_metrics(y_true=train_test_dict['y_train'], y_preds=preds)
    res_train = {f'train_{k}': val for k, val in res_train.items()}
    res.update(res_train)

    res['loss'] = model.history.history['loss']
    res['val_loss'] = model.history.history['val_loss']
    print(res_test)
    return res


def get_scalers(ts_dict_preproc, exg_dict_preproc, scaler_type, spt_feat):
    if scaler_type is None:
        spt_scalers = None
        exg_scalers = None
    else:
        data = np.concatenate([ts[spt_feat].values for k, ts in ts_dict_preproc.items()]).reshape(-1, 1)
        spt_scalers = {spt_feat: StandardScaler().fit(data)}
        exg_scalers = dict()
        for ef, exg_feat_dict in exg_dict_preproc.items():
            data = np.concatenate([ts[ef].values for k, ts in exg_feat_dict.items()]).reshape(-1, 1)
            exg_scalers[ef] = StandardScaler().fit(data)
    return spt_scalers, exg_scalers


def get_scalers_station(ts_dict_preproc, exg_dict_preproc, scaler_type, spt_feat):
    if scaler_type is None:
        spt_scalers = None
        exg_scalers = None
    else:
        spt_scalers = {k: dict() for k in ts_dict_preproc}
        for k, ts in ts_dict_preproc.items():
            f = spt_feat
            scaler = StandardScaler().fit(ts[f].values.reshape(-1, 1))
            spt_scalers[k][f] = scaler
        exg_scalers = dict()

        # transpose dictionary
        _exg_dict_preproc = dict()
        for k1, dt1 in exg_dict_preproc.items():
            for k2, dt2 in dt1.items():
                if k2 not in _exg_dict_preproc:
                    _exg_dict_preproc[k2] = dict()
                _exg_dict_preproc[k2][k1] = dt2
        exg_dict_preproc = _exg_dict_preproc

        for k, feats_dict in exg_dict_preproc.items():
            exg_scalers[k] = dict()
            for f, ts in feats_dict.items():
                scaler = StandardScaler().fit(ts[f].values.reshape(-1, 1))
                exg_scalers[k][f] = scaler
    return spt_scalers, exg_scalers


def main():
    path_params, prep_params, eval_params, model_params = parse_params()
    # path_params = change_params(path_params, '../../data', '../../Dataset/AdbPo')
    _seed = model_params['seed']
    if _seed is not None:
        random.seed(_seed)
        np.random.seed(_seed)
        tf.random.set_seed(_seed)

    res_dir = './output/results'
    data_dir = './output/pickle' + ('_seed' + str(_seed) if _seed != 42 else '')
    model_dir = './output/model' + ('_seed' + str(_seed) if _seed != 42 else '')

    subset = os.path.basename(path_params['ex_filename']).replace('subset_agg_', '').replace('.csv', '')
    nan_percentage = path_params['nan_percentage']
    num_fut = prep_params['ts_params']['num_fut']

    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    out_name = f"{path_params['type']}_{subset}_nan{int(nan_percentage * 10)}_nf{num_fut}"
    print('out_name:', out_name)
    results_path = os.path.join(res_dir, f"{out_name}.csv")
    pickle_path = os.path.join(data_dir, f"{out_name}.pickle")
    checkpoint_path = os.path.join(model_dir, f"{out_name}")

    # if os.path.exists(pickle_path):
    #     print('Loading from', pickle_path, '...', end='')
    #     with open(pickle_path, "rb") as f:
    #         train_test_dict = pickle.load(f)
    #     print(' done!')
    # else:
    if True:
        train_test_dict = data_step(
            path_params, prep_params, eval_params, keep_nan=False, scaler_type=model_params['transform_type'])

        train_test_dict['params'] = {
            'path_params': path_params,
            'prep_params': prep_params,
            'eval_params': eval_params,
            'model_params': model_params,
        }
        with open(pickle_path, "wb") as f:
            print('Saving to', pickle_path, '...', end='')
            pickle.dump(train_test_dict, f)
            print(' done!')

    if os.path.exists(results_path):
        results = pd.read_csv(results_path, index_col=0).T.to_dict()
    else:
        results = {}

    selected_model = train_test_dict['params']['model_params']['model_type'][:3].upper()

    results[selected_model] = model_step(train_test_dict, train_test_dict['params']['model_params'], checkpoint_path)

    pd.DataFrame(results).T.to_csv(results_path, index=True)

    print('Done!')


def main2():
    conf_file = 'data/params_ushcn.json'
    subsets = [
        'subset_agg_th1_0.csv',
        'subset_agg_th1_1.csv',
        'subset_agg_th1_2.csv',
        'subset_agg_th15_0.csv',
        'subset_agg_th15_1.csv',
        'subset_agg_th15_2.csv',
        'subset_agg_th15_3.csv'
    ]
    with open(conf_file, 'r') as f:
        conf = json.load(f)

    for nan_num in [0.0, 0.2, 0.5, 0.8]:
        for subset in subsets:
            print(f"\n {subset}")
            path_params, prep_params, eval_params, model_params = conf['path_params'], conf['prep_params'], conf[
                'eval_params'], conf['model_params']
            path_params["ex_filename"] = "../../data/USHCN/" + subset
            path_params["nan_percentage"] = nan_num
            path_params = change_params(path_params, '../../data', '../../Dataset/AdbPo')

            train_test_dict = data_step(path_params, prep_params, eval_params)

            with open(
                    f"output/{path_params['type']}_{subset.replace('subset_agg_', '').replace('.csv', '')}_nan{int(nan_num * 10)}.pickle",
                    "wb") as f:

                # List of keys to remove
                keys_to_remove = [
                    'dist_x_train',
                    'dist_x_test',

                    'dist_y_train',
                    'dist_y_test',

                    'spt_train',
                    'spt_test',

                    'exg_train',
                    'exg_test',
                ]

                # Remove keys
                for key in keys_to_remove:
                    train_test_dict.pop(key)

                train_test_dict['params'] = {
                    'path_params': path_params,
                    'prep_params': prep_params,
                    'eval_params': eval_params,
                    'model_params': model_params,
                }
                pickle.dump(train_test_dict, f)

    print('Hello World!')


if __name__ == '__main__':
    main()
