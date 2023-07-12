import os
import json
import pickle
import argparse

from ists.dataset.piezo.read import load_data
from ists.preparation import prepare_data, prepare_train_test
from ists.preparation import define_feature_mask, get_list_null_max_size
from ists.preprocessing import get_time_max_sizes
from ists.spatial import prepare_exogenous_data, prepare_spatial_data
from ists.model.wrapper import ModelWrapper
from ists.metrics import compute_metrics


def parse_params():
    """ Parse input parameters. """

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='the path where the configuration is stored.')
    args = parser.parse_args()
    conf_file = args.file
    assert os.path.exists(conf_file), 'Configuration file does not exist'

    with open(conf_file, 'r') as f:
        conf = json.load(f)

    return conf['path_params'], conf['prep_params'], conf['eval_params'], conf['model_params']


def get_params():
    # Path params (i.e. time-series path, context table path)
    base_dir = '../../Dataset/AdbPo/Piezo/'
    path_params = {
        # main time-series (i.e. piezo time-series)
        'ts_filename': os.path.join(base_dir, 'ts_er.xlsx'),
        # table with context information (i.e. coordinates...)
        'ctx_filename': os.path.join(base_dir, 'data_ext_er.xlsx'),
        # dictionary of exogenous time-series (i.e. temperature...)
        'ex_filename': os.path.join(base_dir, 'NetCDF', 'exg_w_tp_t2m.pickle')
    }

    # Preprocessing params (i.e. num past, num future, sampling frequency, features....)
    prep_params = {
        'ts_params': {
            'features': ['Piezometria (m)'],
            'label_col': 'Piezometria (m)',
            'num_past': 48,
            'num_fut': 6,
            'freq': 'M',  # ['M', 'W', 'D']
        },
        'feat_params': {
            # Null Encoding
            'null_feat': 'code_lin',  # ['code_bool', 'code_lin', 'bool', 'lin', 'log']
            'null_max_dist': 12,
            # Time Encoding
            'time_feats': ['M']  # ['D', 'DW', 'WY', 'M']
        },
        'spt_params': {
            'num_past': 24,
            'num_spt': 5,
            'max_dist_th': 10000,
            'max_null_th': 13,
        },
        'exg_params': {
            'num_past': 72,
            'features': ['tp', 't2m_min', 't2m_max', 't2m_avg'],
            'time_feats': ['WY', 'M']
        },
    }

    # Evaluation train and test params
    eval_params = {
        'train_start': '2009-01-01',
        'test_start': '2019-01-01',
        'label_th': 1,
        'null_th': 13,
    }

    model_params = {
        'transform_type': 'standard',  # None 'minmax' 'standard'
        'model_type': 'sttransformer',  # 'sttransformer', 'dense', 'lstm', 'bilstm', 'lstm_base', 'bilstm_base'
        'nn_params': {
            'kernel_size': 3,
            'd_model': 128,
            'num_heads': 4,
            'dff': 128,
            'fff': 64,
            'num_layers': 4,
            'with_cross': False,
            'dropout_rate': 0.2
        },
        'lr': 0.0004,
        'loss': 'mse',
        'batch_size': 32,
        'epochs': 100
    }

    return path_params, prep_params, eval_params, model_params


def data_step(path_params: dict, prep_params: dict, eval_params: dict) -> dict:
    ts_params = prep_params['ts_params']
    feat_params = prep_params['feat_params']
    spt_params = prep_params['spt_params']
    exg_params = prep_params['exg_params']

    # Load dataset
    ts_dict, exg_dict, spt_dict = load_data(
        ts_filename=path_params['ts_filename'],
        context_filename=path_params['ctx_filename'],
        ex_filename=path_params['ex_filename']
    )

    # Prepare x, y, time, dist, id matrix
    x_array, y_array, time_array, dist_x_array, dist_y_array, id_array = prepare_data(
        ts_dict=ts_dict,
        features=ts_params['features'],
        label_col=ts_params['label_col'],
        num_past=ts_params['num_past'],
        num_fut=ts_params['num_fut'],
        freq=ts_params['freq'],
        null_feat=feat_params['null_feat'],
        null_max_dist=feat_params['null_max_dist'],
        time_feats=feat_params['time_feats'],
    )
    print(f'Num of records: {len(x_array)}')
    # Compute feature mask and time encoding max sizes
    x_feature_mask = define_feature_mask(
        base_features=ts_params['features'],
        null_feat=feat_params['null_feat'],
        time_feats=feat_params['time_feats']
    )
    x_time_max_sizes = get_time_max_sizes(feat_params['time_feats'])

    print(f'Record feature mask: {x_feature_mask}')

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
    print(f'Num of records: {len(x_array)}')

    # Prepare exogenous matrix
    exg_array, mask = prepare_exogenous_data(
        id_array=id_array,
        time_array=time_array[:, 1],
        exg_dict=exg_dict,
        num_past=exg_params['num_past'],
        features=exg_params['features'],
        time_feats=exg_params['time_feats']
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
    print(f'Num of records: {len(x_array)}')

    res = prepare_train_test(
        x_array=x_array,
        y_array=y_array,
        time_array=time_array,
        dist_x_array=dist_x_array,
        dist_y_array=dist_y_array,
        id_array=id_array,
        spt_array=spt_array,
        exg_array=exg_array,
        train_start=eval_params['train_start'],
        test_start=eval_params['test_start'],
        max_label_th=eval_params['label_th'],
        max_null_th=eval_params['null_th']
    )
    print(f"X train: {len(res['x_train'])}")
    print(f"X test: {len(res['x_test'])}")

    # Save extra params in train test dictionary
    # Save x and exogenous array feature mask
    res['x_feat_mask'] = x_feature_mask
    res['exg_feat_mask'] = exg_feature_mask

    # Save null max size by finding the maximum between train and test if any
    res['null_max_size'] = get_list_null_max_size(
        [res['x_train']] + [res['x_test']] + res['spt_train'] + res['spt_test'],
        x_feature_mask
    )

    # Save time max sizes
    res['time_max_sizes'] = x_time_max_sizes
    res['exg_time_max_sizes'] = exg_time_max_sizes

    return res


def model_step(train_test_dict: dict, model_params: dict) -> dict:
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
    nn_params['spatial_size'] = len(train_test_dict['spt_train'])
    nn_params['null_max_size'] = int(train_test_dict['null_max_size'])
    nn_params['time_max_sizes'] = train_test_dict['time_max_sizes']
    nn_params['exg_time_max_sizes'] = train_test_dict['exg_time_max_sizes']

    model = ModelWrapper(
        model_type=model_type,
        model_params=nn_params,
        transform_type=transform_type,
        loss=loss,
        lr=lr,
    )
    model.fit(
        x=train_test_dict['x_train'],
        spt=train_test_dict['spt_train'],
        exg=train_test_dict['exg_train'],
        y=train_test_dict['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=2,
        extra={
            'x': train_test_dict['x_test'],
            'spt': train_test_dict['spt_test'],
            'exg': train_test_dict['exg_test'],
            'y': train_test_dict['y_test']
        }
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


def main():
    path_params, prep_params, eval_params, model_params = get_params()
    # path_params, prep_params, eval_params, model_params = parse_params()

    train_test_dict = data_step(path_params, prep_params, eval_params)

    # with open('output/test.pickle', "wb") as f:
    #     pickle.dump(train_test_dict, f)
    # with open('output/test.pickle', "rb") as f:
    #     train_test_dict = pickle.load(f)

    _ = model_step(train_test_dict, model_params)

    print('Hello World!')


if __name__ == '__main__':
    main()
