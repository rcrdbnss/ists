import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

from ablation import apply_ablation_code
from ists.metrics import compute_metrics
from ists.model.wrapper import ModelWrapper


def fit_with_dummy_data(
        wrapper,
        spt,
        exg,
        batch_size: int = 32,
        # x_train_timedeltas=None,
):
    spt_shape, exg_shape = [*np.shape(spt)], [*np.shape(exg)]
    spt_shape[1], exg_shape[1] = spt_shape[0] + 1, exg_shape[0] + 1
    spt_shape[0], exg_shape[0] = batch_size, batch_size
    spt = np.random.randn(*spt_shape)
    exg = np.random.randn(*exg_shape)
    y = np.random.randn(batch_size, 1)
    X = [exg, spt]
    # if x_train_timedeltas is not None:
    #     x_train_timedeltas = x_train_timedeltas[:batch_size]
    #     X.append(x_train_timedeltas)
    X = tuple(X)

    wrapper.model.fit(
        x=X,
        y=y,
        epochs=1,
        batch_size=batch_size,
        validation_data=(X, y),
        verbose=1,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, help='Path to the model params file')
    # parser.add_argument('--dataset', required=True)
    parser.add_argument('--subset', default='all')
    parser.add_argument('--num-past', type=int, required=True, help='Number of past values to consider')
    parser.add_argument('--num-fut', type=int, required=True, help='Number of future values to predict')
    parser.add_argument('--nan-pct', type=float, required=True, help='Percentage of NaN values to insert')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dev', action='store_true', help='Run on development data')
    parser.add_argument('--abl-code', type=str, default='ES')
    parser.add_argument('-e', '--epochs', type=int, required=True)
    args = parser.parse_args()

    if args.dev:
        args.subset = args.subset + '_dev'

    with open(args.file, 'r') as f:
        conf = json.load(f)
    args.dataset = conf['path_params']['type']
    model_params = conf['model_params']

    res_dir = './output/results'
    data_dir = './output/pickle'
    model_dir = './output/model'
    out_name = f"{args.dataset}_{args.subset}_nan{int(args.nan_pct * 10)}_np{args.num_past}_nf{args.num_fut}"
    pickle_path = os.path.join(data_dir, out_name + '.pickle')

    results_path = os.path.join(res_dir, f"{out_name}.csv")
    checkpoint_dir = os.path.join(model_dir, out_name)
    # checkpoint_dir = os.path.join(checkpoint_basedir, 'best_model')
    # # checkpoint_path = os.path.join(checkpoint_basedir, 'best_model', 'cp.ckpt.data-00000-of-00001')
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model', 'cp.weights.h5')

    print('Loading from', pickle_path, '...', end='')
    with open(pickle_path, "rb") as f:
        train_test_dict = pickle.load(f)
    print(' done!')
    train_test_dict['params'] = {
        'model_params': model_params,
    }
    name, train_test_dict = apply_ablation_code(args.abl_code, train_test_dict)

    model_type = model_params['model_type']
    nn_params = model_params['nn_params']
    loss = model_params['loss']
    lr = model_params['lr']
    epochs = model_params['epochs']
    patience = model_params['patience'] if 'patience' in model_params else None
    batch_size = model_params['batch_size']

    # Insert data params in nn_params for building the correct model
    nn_params['feature_mask'] = train_test_dict['x_feat_mask']
    # nn_params['exg_feature_mask'] = train_test_dict['exg_feat_mask']
    nn_params['spatial_size'] = len(train_test_dict['spt_train']) + 1  # target
    nn_params['exg_size'] = len(train_test_dict['exg_train']) + 1  # target
    nn_params['null_max_size'] = train_test_dict['null_max_size']
    nn_params['time_max_sizes'] = train_test_dict['time_max_sizes']
    # nn_params['exg_time_max_sizes'] = train_test_dict['exg_time_max_sizes']
    if 'encoder_cls' in model_params:
        nn_params['encoder_cls'] = model_params['encoder_cls']
    if 'encoder_layer_cls' in model_params:
        nn_params['encoder_layer_cls'] = model_params['encoder_layer_cls']

    wrapper = ModelWrapper(
        checkpoint_dir=checkpoint_dir,
        model_type=model_type,
        model_params=nn_params,
        loss=loss,
        lr=lr,
        dev=args.dev
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

    # x_train_timedeltas = None
    # fit_timedeltas, predict_timedeltas = dict(), dict()
    # if 'x_train_timedeltas' in train_test_dict:
    #     x_train_timedeltas = train_test_dict['x_train_timedeltas']
    #     fit_timedeltas['x_train_timedeltas'] = x_train_timedeltas
    # if 'x_valid_timedeltas' in train_test_dict:
    #     fit_timedeltas['x_val_timedeltas'] = train_test_dict['x_valid_timedeltas']
    # if 'x_test_timedeltas' in train_test_dict:
    #     fit_timedeltas['x_test_timedeltas'] = train_test_dict['x_test_timedeltas']
    #     predict_timedeltas['x_timedeltas'] = train_test_dict['x_test_timedeltas']

    fit_with_dummy_data(
        wrapper,
        spt=train_test_dict['spt_train'],
        exg=train_test_dict['exg_train'],
        batch_size=batch_size,
        # x_train_timedeltas=x_train_timedeltas
    )

    wrapper.model.load_weights(checkpoint_path)

    val_loss_threshold = wrapper.evaluate(
        x=train_test_dict['x_valid'],
        spt=train_test_dict['spt_valid'],
        exg=train_test_dict['exg_valid'],
        y=train_test_dict['y_valid'],
    )['loss']
    print(f'Last checkpoint val loss: {val_loss_threshold}')

    if args.epochs > 0:
        wrapper.fit(
            x=train_test_dict['x_train'],
            spt=train_test_dict['spt_train'],
            exg=train_test_dict['exg_train'],
            y=train_test_dict['y_train'],
            epochs=args.epochs,
            batch_size=batch_size,
            # validation_split=0.2,
            verbose=1,
            **valid_args,
            **test_args,
            # **fit_timedeltas,
            early_stop_patience=patience,
            checkpoint_threshold=val_loss_threshold
        )

    res = {}
    scalers = train_test_dict['scalers']

    preds = wrapper.predict(
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
    res_test = compute_metrics(y_true=y_true, y_preds=y_preds)
    res_test = {f'test_{k}': val for k, val in res_test.items()}
    res.update(res_test)
    print(res_test)

    res['loss'] = wrapper.history.history['loss']
    res['val_loss'] = wrapper.history.history['val_loss']
    if 'test_loss' in wrapper.history.history: res['test_loss'] = wrapper.history.history['test_loss']

    if os.path.exists(results_path):
        results = pd.read_csv(results_path, index_col=0).T.to_dict()
    else:
        results = {}

    while name in results:
        name += '_'
    results[name] = res
    pd.DataFrame(results).T.to_csv(results_path, index=True)
