import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ists_.metrics import compute_metrics
# from ists_.model.wrapper import ModelWrapper
from ists_.model.wrapper_pretrain import ModelWrapper


def model_step(train_test_dict: dict, model_params: dict, checkpoint_dir: str) -> (dict, dict):
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
    nn_params['spatial_size'] = len(train_test_dict['spt_train']) + 1 # target
    nn_params['exg_size'] = len(train_test_dict['exg_train']) + 1 # target
    nn_params["time_features"] = train_test_dict['params']["prep_params"]["feat_params"]['time_feats']
    if 'encoder_cls' in model_params:
        nn_params['encoder_cls'] = model_params['encoder_cls']
    if 'encoder_layer_cls' in model_params:
        nn_params['encoder_layer_cls'] = model_params['encoder_layer_cls']

    scalers = train_test_dict['scalers']
    f = train_test_dict['params']['prep_params']['ts_params']['label_col']
    for id in scalers:
        if isinstance(scalers[id][f], dict):
            scaler = {
                "standard": StandardScaler,
                "minmax": MinMaxScaler,
            }[transform_type]()
            for k, v in scalers[id][f].items():
                setattr(scaler, k, v)
            scalers[id][f] = scaler

    model = ModelWrapper(
        checkpoint_dir=checkpoint_dir,
        model_type=model_type,
        model_params=nn_params,
        loss=loss,
        lr=lr,
        run_eagerly=train_test_dict['params']['path_params']['dev']
    )

    static_args = {
        'exg_static': train_test_dict.get('exg_static_train', None),
        'spt_static': train_test_dict.get('spt_static_train', None),
        'val_exg_static': train_test_dict.get('exg_static_valid', None),
        'val_spt_static': train_test_dict.get('spt_static_valid', None),
    }

    metrics_pretr, curves_pretr = None, None
    if model_params['pretrain']:
        model.pretrain(
            x=train_test_dict['x_train'],
            spt=train_test_dict['spt_train'],
            exg=train_test_dict['exg_train'],
            epochs=(epochs if train_test_dict['params']['path_params']['dev'] else 100),
            batch_size=batch_size,
            verbose=1,
            val_x=train_test_dict['x_valid'],
            val_spt=train_test_dict['spt_valid'],
            val_exg=train_test_dict['exg_valid'],
            val_y=train_test_dict['y_valid'],
            **static_args,
            early_stop_patience=patience,
        )
        # model.load_pretrained_checkpoint(model.checkpoint_dir + "/pretr_encoder.weights.h5")
        metrics_pretr, curves_pretr = {}, {}
        curves_pretr.update({
            'loss': model.history.history['loss'],
            'mse_avg': model.history.history['mse_avg'] if 'mse_avg' in model.history.history else [],
            'mse': model.history.history['mse'],
            'mse_obs': model.history.history['mse_obs'],
            'val_loss': model.history.history['val_loss'],
            'val_mse_avg': model.history.history['val_mse_avg'] if 'val_mse_avg' in model.history.history else [],
            'val_mse': model.history.history['val_mse'],
            'val_mse_obs': model.history.history['val_mse_obs'],
        })

        if hasattr(model, 'pretrain_predict'):
            y_pred, y_true, y_fill, mask, aux_mask = model.pretrain_predict(
                x=train_test_dict['x_test'],
                spt=train_test_dict['spt_test'],
                exg=train_test_dict['exg_test'],
                exg_static=train_test_dict.get('exg_static_test', None), spt_static=train_test_dict.get('spt_static_test', None),
            )

            id_array = train_test_dict['id_test']
            # target feature only, for now
            y_pred, y_true, y_fill, mask, aux_mask = y_pred[:, 0], y_true[:, 0], y_fill[:, 0], mask[:, 0], aux_mask[:, 0]
            y_pred = np.array([scalers[id][f].inverse_transform([y_])[0] for y_, id in zip(y_pred, id_array)])
            y_true = np.array([scalers[id][f].inverse_transform([y_])[0] for y_, id in zip(y_true, id_array)])
            y_fill = np.array([scalers[id][f].inverse_transform([y_])[0] for y_, id in zip(y_fill, id_array)])
            print("# Test: Reconstruct held-out points #")
            div = np.sum(aux_mask)
            mae = np.sum(np.abs(y_true - y_pred) * aux_mask) / div
            mse = np.sum((y_true - y_pred) ** 2 * aux_mask) / div
            wmape = mae / np.mean(np.abs(y_true)) * 100
            metrics_pretr.update({'test_mae': mae, 'test_mse': mse, 'test_wMAPE': wmape})
            print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, wMAPE: {wmape:.2f}%")
            print("# Test: Reconstruct observed points")
            div = np.sum(mask)
            mae = np.sum(np.abs(y_true - y_pred) * mask) / div
            mse = np.sum((y_true - y_pred) ** 2 * mask) / div
            wmape = mae / np.mean(np.abs(y_true)) * 100
            print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, wMAPE: {wmape:.2f}%")
            print("# Baseline: original vs imputed held-out points #")
            div = np.sum(aux_mask)
            mae = np.sum(np.abs(y_true - y_fill) * aux_mask) / div
            mse = np.sum((y_true - y_fill) ** 2 * aux_mask) / div
            wmape = mae / np.mean(np.abs(y_true)) * 100
            # metrics_pretr.update({'mae_fill': mae, 'mse_fill': mse, 'wMAPE_fill': wmape})
            print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, wMAPE: {wmape:.2f}%")

    model.fit(
        x=train_test_dict['x_train'],
        spt=train_test_dict['spt_train'],
        exg=train_test_dict['exg_train'],
        y=train_test_dict['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        val_x=train_test_dict['x_valid'],
        val_spt=train_test_dict['spt_valid'],
        val_exg=train_test_dict['exg_valid'],
        val_y=train_test_dict['y_valid'],
        **static_args,
        early_stop_patience=patience,
    )

    metrics, curves = {}, {}

    y_true = train_test_dict['y_valid']
    y_pred = model.predict(
        x=train_test_dict['x_valid'],
        spt=train_test_dict['spt_valid'],
        exg=train_test_dict['exg_valid'],
        exg_static=train_test_dict.get('exg_static_valid', None), spt_static=train_test_dict.get('spt_static_valid', None),
    )
    id_array = train_test_dict['id_valid']
    y_true = np.array([scalers[id][f].inverse_transform([y_])[0] for y_, id in zip(y_true, id_array)])
    y_pred = np.array([scalers[id][f].inverse_transform([y_])[0] for y_, id in zip(y_pred, id_array)])
    res_valid = compute_metrics(y_true=y_true, y_preds=y_pred)
    res_valid = {f'val_{k}': val for k, val in res_valid.items()}
    metrics.update(res_valid)
    print(res_valid)

    y_true = train_test_dict['y_test']
    y_pred = model.predict(
        x=train_test_dict['x_test'],
        spt=train_test_dict['spt_test'],
        exg=train_test_dict['exg_test'],
        exg_static=train_test_dict.get('exg_static_test', None), spt_static=train_test_dict.get('spt_static_test', None),
    )
    id_array = train_test_dict['id_test']
    """y_true = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
                       for y_, id in zip(y_true, id_array)])
    y_pred = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
                       for y_, id in zip(y_pred, id_array)])"""
    y_true = np.array([scalers[id][f].inverse_transform([y_])[0] for y_, id in zip(y_true, id_array)])
    y_pred = np.array([scalers[id][f].inverse_transform([y_])[0] for y_, id in zip(y_pred, id_array)])
    res_test = compute_metrics(y_true=y_true, y_preds=y_pred)
    res_test = {f'test_{k}': val for k, val in res_test.items()}
    metrics.update(res_test)
    print(res_test)

    curves['loss'] = model.history.history['loss']
    curves['val_loss'] = model.history.history['val_loss']
    curves['mse'] = model.history.history['mse']
    curves['val_mse'] = model.history.history['val_mse']
    curves.update({
        'mse_avg': model.history.history['mse_avg'] if 'mse_avg' in model.history.history else [],
        'val_mse_avg': model.history.history['val_mse_avg'] if 'val_mse_avg' in model.history.history else [],
    })
    epoch_times = model.epoch_times
    if isinstance(epoch_times, dict):
        epoch_times = {f'epoch_times_{k}': v for k, v in epoch_times.items()}
    else:
        epoch_times = {'epoch_times': epoch_times}
    curves.update(epoch_times)

    if metrics_pretr is not None:
        for k, v in metrics_pretr.items():
            metrics[f'pretr_{k}'] = v
        for k, v in curves_pretr.items():
            curves[f'pretr_{k}'] = v

    return metrics, curves


import tensorflow as tf
from data_step import parse_params, data_step, get_conf_name


def main():
    path_params, prep_params, eval_params, model_params = parse_params()
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

    # subset = os.path.basename(path_params['ex_filename']).replace('subset_agg_', '').replace('.csv', '')
    nan_percentage = path_params['nan_percentage']
    num_past = prep_params['ts_params']['num_past']
    num_fut = prep_params['ts_params']['num_fut']
    num_spt = prep_params['spt_params']['num_spt']
    max_dist_th = prep_params['spt_params']['max_dist_th']

    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # out_name = f"{path_params['type']}_{subset}_nan{int(nan_percentage * 10)}_nf{num_fut}"
    out_name = get_conf_name(
        dataset=path_params['type'],
        nan_percentage=nan_percentage,
        num_past=num_past,
        num_fut=num_fut,
        num_spt=num_spt,
        max_dist_th=max_dist_th,
        seed=_seed,
        dev=path_params['dev']
    )
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
            path_params, prep_params, eval_params, scaler_type=model_params['transform_type']
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

    res = model_step(train_test_dict, train_test_dict['params']['model_params'], checkpoint_path)
    if isinstance(res['epoch_times'], dict):
        res.update(res['epoch_times'])
        del res['epoch_times']
    results[selected_model] = res

    pd.DataFrame(results).T.to_csv(results_path, index=True)

    print('Done!')


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
