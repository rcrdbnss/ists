import datetime
import json
import os
import pickle
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf

from data_step import parse_params, data_step


def no_ablation(train_test_dict) -> dict:
    # train_test_dict['params']['model_params']['model_type'] = "sttransformer"
    return train_test_dict


def ablation_embedder_no_feat(train_test_dict, code) -> dict:
    for n in (['train', 'test'] + (['valid'] if train_test_dict['validation_data'] else [])):
        cond_x = [x != code for x in train_test_dict['x_feat_mask']]
        train_test_dict[f'x_{n}'] = train_test_dict[f'x_{n}'][:, :, cond_x]
        train_test_dict[f'spt_{n}'] = [x[:, :, cond_x] for x in train_test_dict[f'spt_{n}']]

        # cond_exg = [x != code for x in train_test_dict['exg_feat_mask']]
        # train_test_dict[f'exg_{n}'] = train_test_dict[f'exg_{n}'][:, :, cond_exg]
        train_test_dict[f'exg_{n}'] = [x[:, :, cond_x] for x in train_test_dict[f'exg_{n}']]

    train_test_dict['x_feat_mask'] = [x for x in train_test_dict['x_feat_mask'] if x != code]
    # train_test_dict['exg_feat_mask'] = [x for x in train_test_dict['exg_feat_mask'] if x != code]

    if code == 1:
        train_test_dict['null_max_size'] = None

    if code == 2:
        train_test_dict['time_max_sizes'] = []
        # train_test_dict['exg_time_max_sizes'] = []

    return train_test_dict


def ablation_embedder_no_time(train_test_dict) -> dict:
    train_test_dict = ablation_embedder_no_feat(train_test_dict, 2)
    return train_test_dict


def ablation_embedder_no_null(train_test_dict) -> dict:
    train_test_dict = ablation_embedder_no_feat(train_test_dict, 1)
    return train_test_dict


def ablation_embedder_no_time_null(train_test_dict) -> dict:
    train_test_dict = ablation_embedder_no_feat(train_test_dict, 1)
    train_test_dict = ablation_embedder_no_feat(train_test_dict, 2)
    return train_test_dict


def ablation_encoder_stt(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "sttransformer"
    return train_test_dict


def ablation_encoder_t(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "t"
    return train_test_dict


def ablation_encoder_s(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "s"
    return train_test_dict


def ablation_encoder_e(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "e"
    return train_test_dict


def ablation_encoder_ts(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "ts"
    return train_test_dict


def ablation_encoder_te(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "te"
    return train_test_dict


def ablation_encoder_se(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "se"
    return train_test_dict


def ablation_encoder_ts_fe(train_test_dict) -> dict:
    # Models TS by concatenating exogenous features E in the feature dimension to T.
    train_test_dict['params']['model_params']['model_type'] = "ts_fe"
    return train_test_dict


def ablation_encoder_ts_fe_nonull(train_test_dict) -> dict:
    # Models TS by concatenating exogenous features E in the feature dimension to T,
    # without null encoding.
    train_test_dict = ablation_encoder_ts_fe(train_test_dict)
    train_test_dict = ablation_embedder_no_null(train_test_dict)
    return train_test_dict


def ablation_encoder_ts_fe_nonull_notime(train_test_dict) -> dict:
    # Models TS by concatenating exogenous features E in the feature dimension to T,
    # without null and time encoding.
    train_test_dict = ablation_encoder_ts_fe(train_test_dict)
    train_test_dict = ablation_embedder_no_time_null(train_test_dict)
    return train_test_dict


def ablation_encoder_stt_se(train_test_dict) -> dict:
    # Models STT by integrating exogenous E and T similarly to the S module.
    train_test_dict['params']['model_params']['model_type'] = "stt_se"
    return train_test_dict


def ablation_encoder_stt_se_nonull(train_test_dict) -> dict:
    # Models STT by integrating exogenous E and T similarly to the S module,
    # without null encoding.
    train_test_dict = ablation_encoder_stt_se(train_test_dict)
    train_test_dict = ablation_embedder_no_null(train_test_dict)
    return train_test_dict


def ablation_encoder_se_se(train_test_dict) -> dict:
    # Models SE by integrating exogenous E and T similarly to the S module.
    train_test_dict['params']['model_params']['model_type'] = "se_se"
    return train_test_dict


def ablation_encoder_se_se_nonull(train_test_dict) -> dict:
    # Models SE by integrating exogenous E and T similarly to the S module,
    # without null encoding.
    train_test_dict = ablation_encoder_se_se(train_test_dict)
    train_test_dict = ablation_embedder_no_null(train_test_dict)
    return train_test_dict


def ablation_encoder_stt_mts_e(train_test_dict) -> dict:
    # Models STT with multivariate inputs in E.
    cond_x = [x == 0 for x in train_test_dict['x_feat_mask']]
    for n in ['train', 'test']:
        x = train_test_dict[f'x_{n}'][:, :, cond_x].copy()

        train_test_dict[f'exg_{n}'] = np.concatenate([train_test_dict[f'exg_{n}'], x], axis=2)

    x_feat_mask = [x for x in train_test_dict['x_feat_mask'] if x == 0]
    train_test_dict['exg_feat_mask'] = train_test_dict['exg_feat_mask'] + x_feat_mask

    return train_test_dict


def ablation_no_global_encoder(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "no_glb"
    return train_test_dict


def ablation_multivariate(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = 'stt_mv'
    # train_test_dict['params']['model_params']['univar_or_multivar'] = 'multivar'
    train_test_dict['params']['model_params']['multivar'] = True
    return train_test_dict


def ablation_multivariate_no_global_encoder(train_test_dict) -> dict:
    train_test_dict = ablation_multivariate(train_test_dict)
    train_test_dict = ablation_no_global_encoder(train_test_dict)
    train_test_dict['params']['model_params']['model_type'] = "mv_no_glb"
    return train_test_dict


def ablation_multivariate_no_null(train_test_dict) -> dict:
    train_test_dict = ablation_multivariate(train_test_dict)
    train_test_dict = ablation_embedder_no_null(train_test_dict)
    return train_test_dict


def ablation_multivariate_ts(train_test_dict) -> dict:
    train_test_dict = ablation_multivariate(train_test_dict)
    train_test_dict['params']['model_params']['model_type'] = 'mv_ts'
    return train_test_dict


def ablation_multivariate_te(train_test_dict) -> dict:
    train_test_dict = ablation_multivariate(train_test_dict)
    train_test_dict['params']['model_params']['model_type'] = 'mv_te'
    return train_test_dict


def ablation_multivariate_ts_no_null_no_global_encoder(train_test_dict) -> dict:
    train_test_dict = ablation_multivariate_no_null(train_test_dict)
    train_test_dict['params']['model_params']['model_type'] = "mv_ts_no_glb"
    return train_test_dict


def ablation_target_only(train_test_dict) -> dict:
    train_test_dict['spt_train'] = []
    train_test_dict['exg_train'] = []
    train_test_dict['spt_test'] = []
    train_test_dict['exg_test'] = []
    return train_test_dict


def ablation_stt_2(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "stt2"
    return train_test_dict


def apply_ablation_code(abl_code: str, train_test_dict):
    # suffix = None
    # if '#' in abl_code:
    #     abl_code, suffix = abl_code.split('#')

    T, S, E = 'T' in abl_code, 'S' in abl_code, 'E' in abl_code
    n, t = 'n' in abl_code, 't' in abl_code
    abl_1 = '1' in abl_code
    abl_2 = '2' in abl_code
    abl_3 = '3' in abl_code
    abl_4 = '4' in abl_code
    abl_5 = '5' in abl_code
    abl_6 = '6' in abl_code
    abl_7 = '7' in abl_code
    abl_8 = '8' in abl_code
    abl_9 = '9' in abl_code

    train_test_dict['params']['model_params']['model_type'] = "sttN"

    if S and E:
        T = False  # no need to force the target series in anymore
    if abl_1:
        train_test_dict['params']['model_params']['model_type'] = "baseline"
        n = False
    if abl_2:
        train_test_dict['params']['model_params']['nn_params']['do_emb'] = False
    if abl_3:
        train_test_dict['params']['model_params']['encoder_layer_cls'] = 'EncoderAttnMaskLayer'
    if abl_5:
        n = False
    if abl_6:
        train_test_dict['params']['model_params']['model_type'] = "baseline"
        n = False
        train_test_dict['params']['model_params']['nn_params']['do_emb'] = False
    if abl_7:
        train_test_dict['params']['model_params']['model_type'] = "baseline"
        train_test_dict['params']['model_params']['nn_params']['do_emb'] = False
    if abl_8:
        train_test_dict['params']['model_params']['model_type'] = "baseline"
    if abl_9:
        train_test_dict['params']['model_params']['model_type'] = "emb_gru"

    if not n:
        train_test_dict = ablation_embedder_no_feat(train_test_dict, 1)

    train_test_dict['params']['model_params']['nn_params']['do_exg'] = E
    train_test_dict['params']['model_params']['nn_params']['do_spt'] = S
    train_test_dict['params']['model_params']['nn_params']['force_target'] = T

    abl_code = []

    _abl_code = []
    if E: _abl_code.append('E')
    if S: _abl_code.append('S')
    if T: _abl_code.append('T')
    abl_code.append(''.join(_abl_code))

    _abl_code = []
    if n: _abl_code.append('n')
    if t: _abl_code.append('t')
    abl_code.append(''.join(_abl_code))

    if abl_1: abl_code.append('1')
    if abl_2: abl_code.append('2')
    if abl_3: abl_code.append('3')
    if abl_4: abl_code.append('4')
    if abl_5: abl_code.append('5')
    if abl_6: abl_code.append('6')
    if abl_7: abl_code.append('7')
    if abl_8: abl_code.append('8')
    if abl_9: abl_code.append('9')

    abl_code = '_'.join(abl_code)
    return abl_code, train_test_dict


from pipeline import model_step


def get_suffix(train_test_dict):
    defaults = {
        'num_layers': 1,
        'l2_reg': 0.01,
        'dropout_rate': 0.2,
        'time_feats': {
            'french': ('D',),
            'ushcn': ('WY',),
            'adbpo': ('M', 'WY'),
        },
        'epochs': 100,
        'patience': 20,
        'lr': {
            'french': 0.0004, 'ushcn': 0.00004, 'adbpo': 0.0004,
        },
        'tf': '2.17.0',
        'd_model': {
            'french': 64, 'ushcn': 64, 'adbpo': 32
        },
        'num_heads': { 'french': 4, 'ushcn': 4, 'adbpo': 2
        },
        'dff': {
            'french': 128, 'ushcn': 128, 'adbpo': 64,
        },
        'fff': {
            'french': 256, 'ushcn': 256, 'adbpo': 128
        },
    }

    dataset = train_test_dict['params']['path_params']['type']

    suffix = []

    num_layers = train_test_dict['params']['model_params']['nn_params']['num_layers']
    if num_layers != defaults['num_layers']:
        suffix.append(f'encs={num_layers}')

    d_model = train_test_dict['params']['model_params']['nn_params']['d_model']
    if d_model != defaults['d_model'][dataset]:
        suffix.append(f'd{d_model}')
    num_heads = train_test_dict['params']['model_params']['nn_params']['num_heads']
    if num_heads != defaults['num_heads'][dataset]:
        suffix.append(f'h{num_heads}')
    dff = train_test_dict['params']['model_params']['nn_params']['dff']
    if dff != defaults['dff'][dataset]:
        suffix.append(f'dff{dff}')
    fff = train_test_dict['params']['model_params']['nn_params']['fff']
    if fff != defaults['fff'][dataset]:
        suffix.append(f'fff{fff}')

    l2_reg = train_test_dict['params']['model_params']['nn_params']['l2_reg']
    if l2_reg != defaults['l2_reg']:
        l2_reg = str(l2_reg).replace('0.', '')
        suffix.append(f'reg{l2_reg}')

    dropout_rate = train_test_dict['params']['model_params']['nn_params']['dropout_rate']
    if dropout_rate != defaults['dropout_rate']:
        dropout_rate = str(dropout_rate).replace('0.', '')
        suffix.append(f'dro{dropout_rate}')

    epochs = train_test_dict['params']['model_params']['epochs']
    if epochs != defaults['epochs']:
        suffix.append(f'e{epochs}')

    patience = train_test_dict['params']['model_params']['patience']
    if patience != defaults['patience']:
        suffix.append(f'pat{patience}')

    lr = train_test_dict['params']['model_params']['lr']
    if lr != defaults['lr'][dataset]:
        suffix.append(f'lr{lr:.0e}')

    time_feats = train_test_dict['params']['prep_params']['feat_params']['time_feats']
    time_feats = tuple(sorted(time_feats))
    if time_feats != tuple(sorted(defaults['time_feats'][dataset])):
        suffix.append('_'.join(time_feats))

    if tf.__version__ != defaults['tf']:
        suffix.append(f'tf{tf.__version__.replace(".", "")}')

    return '_'.join(suffix)


def ablation(
        train_test_dict: dict,
        results_path: str,
        pickle_path: str,
        checkpoint_path: str,
        ablation_embedder: bool = True,
        ablation_encoder: bool = True,
        ablation_extra: dict = None
):
    if os.path.exists(results_path):
        results = pd.read_csv(results_path, index_col=0).T.to_dict()
    else:
        results = {}  # todo: non-grid results

    suffix = get_suffix(train_test_dict)

    ablations_mapping = [
        'E_nt',
        # 'E_t',
        # 'E_nt_1',
        # 'E_nt_2',
        # 'E_nt_3',
        # 'E_nt_4',
        # 'E_nt_6',
        # 'E_nt_7',
        # 'E_nt_8',
        # 'E_nt_9',
    ]

    for name in ablations_mapping:
        # Configure ablation test
        name, D = apply_ablation_code(name, deepcopy(train_test_dict))
        if suffix: name = f"{name}#{suffix}"
        if '#' not in name:
            name += '#'
        name += "_refactor"
        if name.endswith('#'):
            name = name[:-1]

        # Exec ablation test
        print(f"\n{name}: {D['params']['model_params']['model_type']}")
        if D['params']['model_params']['seed'] != 42:
            name += '_seed' + str(D['params']['model_params']['seed'])

        """timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        checkpoint_path += "/" + timestamp
        os.makedirs(checkpoint_path, exist_ok=True)"""
        model_res = model_step(D, D['params']['model_params'], checkpoint_path)
        results[name] = model_res  # todo: non-grid results
        """model_res["name"] = name
        model_res["params"] = D["params"]["model_params"]"""

        # Save results
        pd.DataFrame(results).T.to_csv(results_path, index=True)  # todo: non-grid results
        """results_path = results_path.replace('.csv', '/')
        os.makedirs(results_path, exist_ok=True)
        results_path += timestamp + '.json'
        with open(results_path, 'w') as f:
            model_res["params"]["nn_params"]["null_max_size"] = int(model_res["params"]["nn_params"]["null_max_size"])
            model_res["test_mae"] = float(model_res["test_mae"])
            model_res["test_mse"] = float(model_res["test_mse"])
            json.dump(model_res, f, indent=4)"""

    # return pd.DataFrame(results).T


def main():
    path_params, prep_params, eval_params, model_params = parse_params()
    if model_params['cpu']:
        tf.config.set_visible_devices([], 'GPU')
    _seed = model_params['seed']
    if _seed is not None:
        random.seed(_seed)
        np.random.seed(_seed)
        tf.random.set_seed(_seed)

    results_dir = './output/results'
    pickle_dir = './output/pickle' + ('_seed' + str(_seed) if _seed != 42 else '')
    model_dir = './output/model' + ('_seed' + str(_seed) if _seed != 42 else '')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(pickle_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

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

    conf_name = f"{path_params['type']}_{subset}_nan{int(nan_percentage * 10)}_np{num_past}_nf{num_fut}"
    print('configuration:', conf_name)
    results_file = os.path.join(results_dir, f"{conf_name}.csv")
    pickle_file = os.path.join(pickle_dir, f"{conf_name}.pickle")
    checkpoint_dir = os.path.join(model_dir, conf_name)

    if os.path.exists(pickle_file) and not path_params['force_data_step']:
        print('Loading from', pickle_file, '...', end='', flush=True)
        with open(pickle_file, "rb") as f:
            train_test_dict = pickle.load(f)
        print(' done!')
    else:
    # if True:
        train_test_dict = data_step(
            path_params, prep_params, eval_params, keep_nan=False, scaler_type=model_params['transform_type']
        )

        with open(pickle_file, "wb") as f:
            print('Saving to', pickle_file, '...', end='', flush=True)
            pickle.dump(train_test_dict, f)
            print(' done!')

    train_test_dict['params'] = {
        'path_params': path_params,
        'prep_params': prep_params,
        'eval_params': eval_params,
        'model_params': model_params,
    }

    ablation(
        train_test_dict=train_test_dict,
        results_path=results_file,
        pickle_path=pickle_file,
        checkpoint_path=checkpoint_dir,
        ablation_embedder=True,
        ablation_encoder=True,
        ablation_extra={
            'TS_FE': ablation_encoder_ts_fe,
            'STT_SE': ablation_encoder_stt_se,
            'SE_SE': ablation_encoder_se_se,
            'STT_MTS_E': ablation_encoder_stt_mts_e,
        }
    )

    print('Hello World!')


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
