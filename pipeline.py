import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ists.metrics import compute_metrics
from ists.model.wrapper import ModelWrapper


def change_params(path_params: dict, base_string: str, new_string: str) -> dict:
    """path_params['ts_filename'] = path_params['ts_filename'].replace(base_string, new_string, 1)
    path_params['ctx_filename'] = path_params['ctx_filename'].replace(base_string, new_string, 1)
    path_params['ex_filename'] = path_params['ex_filename'].replace(base_string, new_string, 1)

    return path_params"""


def model_step(train_test_dict: dict, model_params: dict, checkpoint_dir: str) -> dict:
    model_type = model_params['model_type']
    transform_type = model_params['transform_type']
    nn_params = model_params['nn_params']
    loss = model_params['loss']
    lr = model_params['lr']
    epochs = model_params['epochs']
    patience = model_params['patience']
    batch_size = model_params['batch_size']

    # Insert data params in nn_params for building the correct model
    nn_params['feature_mask'] = train_test_dict['x_feat_mask']
    # nn_params['exg_feature_mask'] = train_test_dict['exg_feat_mask']
    nn_params['spatial_size'] = len(train_test_dict['spt_train']) + 1 # target
    nn_params['exg_size'] = len(train_test_dict['exg_train']) + 1 # target
    nn_params['null_max_size'] = train_test_dict['null_max_size']
    nn_params['time_max_sizes'] = train_test_dict['time_max_sizes']
    # nn_params['exg_time_max_sizes'] = train_test_dict['exg_time_max_sizes']
    if 'encoder_cls' in model_params:
        nn_params['encoder_cls'] = model_params['encoder_cls']
    if 'encoder_layer_cls' in model_params:
        nn_params['encoder_layer_cls'] = model_params['encoder_layer_cls']

    model = ModelWrapper(
        checkpoint_dir=checkpoint_dir,
        model_type=model_type,
        model_params=nn_params,
        loss=loss,
        lr=lr,
        dev=train_test_dict['params']['path_params']['dev']
    )

    valid_args = dict(
        val_x=train_test_dict['x_valid'],
        val_spt=train_test_dict['spt_valid'],
        val_exg=train_test_dict['exg_valid'],
        val_y=train_test_dict['y_valid']
    )
    test_args = {
        'test_x': train_test_dict['x_test'],
        'test_spt': train_test_dict['spt_test'],
        'test_exg': train_test_dict['exg_test'],
        'test_y': train_test_dict['y_test'],
    }

    model.fit(
        x=train_test_dict['x_train'],
        spt=train_test_dict['spt_train'],
        exg=train_test_dict['exg_train'],
        y=train_test_dict['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
        **valid_args,
        **test_args,
        early_stop_patience=patience,
    )

    res = {}
    scalers = train_test_dict['scalers']
    for id in scalers:
        for f in scalers[id]:
            if isinstance(scalers[id][f], dict):
                scaler = {
                    "standard": StandardScaler,
                    # "minmax": MinMaxScaler,
                }[transform_type]()
                for k, v in scalers[id][f].items():
                    setattr(scaler, k, v)
                scalers[id][f] = scaler

    preds = model.predict(
        x=train_test_dict['x_test'],
        spt=train_test_dict['spt_test'],
        exg=train_test_dict['exg_test'],
        # **predict_timedeltas
    )

    id_array = train_test_dict['id_test']
    y_true = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
                       for y_, id in zip(train_test_dict['y_test'], id_array)])
    y_preds = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
                        for y_, id in zip(preds, id_array)])
    # res_test = compute_metrics(y_true=train_test_dict['y_test'], y_preds=preds)
    res_test = compute_metrics(y_true=y_true, y_preds=y_preds)
    res_test = {f'test_{k}': val for k, val in res_test.items()}
    res.update(res_test)
    print(res_test)

    # preds = model.predict(
    #     x=train_test_dict['x_train'],
    #     spt=train_test_dict['spt_train'],
    #     exg=train_test_dict['exg_train'],
    #     id_array=train_test_dict['id_train'],
    # )
    #
    # id_array = train_test_dict['id_train']
    # y_true = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
    #                    for y_, id in zip(train_test_dict['y_train'], id_array)])
    # y_preds = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
    #                     for y_, id in zip(preds, id_array)])
    # # res_train = compute_metrics(y_true=train_test_dict['y_train'], y_preds=preds)
    # res_train = compute_metrics(y_true=y_true, y_preds=y_preds)
    # res_train = {f'train_{k}': val for k, val in res_train.items()}
    # res.update(res_train)
    # print(res_train)

    res['loss'] = model.history.history['loss']
    res['val_loss'] = model.history.history['val_loss']
    if 'test_loss' in model.history.history: res['test_loss'] = model.history.history['test_loss']
    res['epoch_times'] = model.epoch_times
    return res


def get_scalers(ts_dict_preproc, exg_dict_preproc, scaler_type, spt_feat):
    """if scaler_type is None:
        spt_scalers = None
        exg_scalers = None
    else:
        data = np.concatenate([ts[spt_feat].values for k, ts in ts_dict_preproc.items()]).reshape(-1, 1)
        spt_scalers = {spt_feat: StandardScaler().fit(data)}
        exg_scalers = dict()
        for ef, exg_feat_dict in exg_dict_preproc.items():
            data = np.concatenate([ts[ef].values for k, ts in exg_feat_dict.items()]).reshape(-1, 1)
            exg_scalers[ef] = StandardScaler().fit(data)
    return spt_scalers, exg_scalers"""


def get_scalers_station(ts_dict_preproc, exg_dict_preproc, scaler_type, spt_feat):
    """if scaler_type is None:
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
    return spt_scalers, exg_scalers"""


import tensorflow as tf
from data_step import parse_params, data_step


def main():
    path_params, prep_params, eval_params, model_params = parse_params()
    # path_params = change_params(path_params, '../../data', '../../Dataset/AdbPo')
    if model_params['cpu']:
        tf.config.set_visible_devices([], 'GPU')
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
        # from data_step import data_step
        train_test_dict = data_step(
            path_params, prep_params, eval_params, keep_nan=False, scaler_type=model_params['transform_type']
        )
        with open(pickle_path, "wb") as f:
            print('Saving to', pickle_path, '...', end='')
            pickle.dump(train_test_dict, f)
            print(' done!')

    train_test_dict['params'] = {
        'path_params': path_params,
        'prep_params': prep_params,
        'eval_params': eval_params,
        'model_params': model_params,
    }

    if os.path.exists(results_path):
        results = pd.read_csv(results_path, index_col=0).T.to_dict()
    else:
        results = {}

    selected_model = train_test_dict['params']['model_params']['model_type'][:3].upper()

    results[selected_model] = model_step(train_test_dict, train_test_dict['params']['model_params'], checkpoint_path)

    pd.DataFrame(results).T.to_csv(results_path, index=True)

    print('Done!')


def main2():
    """conf_file = 'data/params_ushcn.json'
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

            from data_step import data_step
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

    print('Hello World!')"""


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    main()
