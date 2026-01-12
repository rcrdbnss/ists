from datetime import datetime
import json
import os
import pickle
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf

from data_step import parse_params, data_step, get_conf_name
from ists_.preprocessing import TIME_N_VALUES


def no_ablation(train_test_dict) -> dict:
    return train_test_dict


def ablation_embedder_no_feat(train_test_dict, code) -> dict:
    for n in ['train', 'test', 'valid']:
        cond_x = [x != code for x in train_test_dict['x_feat_mask']]
        train_test_dict[f'x_{n}'] = train_test_dict[f'x_{n}'][:, :, cond_x]
        train_test_dict[f'spt_{n}'] = [x[:, :, cond_x] for x in train_test_dict[f'spt_{n}']]
        train_test_dict[f'exg_{n}'] = [x[:, :, cond_x] for x in train_test_dict[f'exg_{n}']]

    train_test_dict['x_feat_mask'] = [x for x in train_test_dict['x_feat_mask'] if x != code]

    if code == 1:
        train_test_dict['params']['model_params']['nn_params']['is_null_embedding'] = False

    if code == 2:
        train_test_dict['params']["prep_params"]["feat_params"]['time_feats'] = None

    return train_test_dict


def ablation_embedder_no_time(train_test_dict) -> dict:
    train_test_dict = ablation_embedder_no_feat(train_test_dict, 2)
    return train_test_dict


def ablation_embedder_no_null(train_test_dict) -> dict:
    """
    Instead of removing the attention mask, set it to 1 everywhere.
    """

    # train_test_dict = ablation_embedder_no_feat(train_test_dict, 1)

    null_id = np.where(np.array(train_test_dict['x_feat_mask']) == 1)[0]
    if len(null_id) == 0:
        return train_test_dict
    null_id = null_id[0]  # Assuming only one null feature for simplicity
    for n in ['train', 'test', 'valid']:
        X = train_test_dict[f'x_{n}']
        X[:, :, null_id] = 1.0  # Set the null feature to 1
        for i in range(len(train_test_dict[f'spt_{n}'])):
            X = train_test_dict[f'spt_{n}'][i]
            X[:, :, null_id] = 1.0
            train_test_dict[f'spt_{n}'][i] = X
        for i in range(len(train_test_dict[f'exg_{n}'])):
            X = train_test_dict[f'exg_{n}'][i]
            X[:, :, null_id] = 1.0
            train_test_dict[f'exg_{n}'][i] = X
    return train_test_dict


def ablation_embedder_no_time_null(train_test_dict) -> dict:
    # train_test_dict = ablation_embedder_no_feat(train_test_dict, 1)
    train_test_dict = ablation_embedder_no_null(train_test_dict)
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


def ablation_impute_mean(train_test_dict) -> dict:
    for n in ['train', 'test', 'valid']:
        X = train_test_dict[f'x_{n}']
        X[:, :, 0][X[:, :, 1].astype(bool)] = 0.
        for X in train_test_dict[f'spt_{n}']:
            X[:, :, 0][X[:, :, 1].astype(bool)] = 0.
        for X in train_test_dict[f'exg_{n}']:
            X[:, :, 0][X[:, :, 1].astype(bool)] = 0.
    return train_test_dict


def scale_time_features(x, feature_mask, time_features):
    if time_features is None:
        return x
    time_ids = np.where(feature_mask == 2)[0]
    assert len(time_ids) == len(time_features)
    for i, t in zip(time_ids, time_features):
        t_max = TIME_N_VALUES[t] - 1  # 0-indexed
        x[:, :, i] = x[:, :, i] / t_max - 0.5
        assert np.all(x[:, :, i] >= -0.5) and np.all(x[:, :, i] <= 0.5)
    return x


def apply_ablation_code(abl_code: str, D):
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

    def _scale_time_features():
        feature_mask = np.array(D['x_feat_mask'])
        time_features = D['params']["prep_params"]["feat_params"]['time_feats']
        for split in ['train', 'test', 'valid']:
            D[f'x_{split}'] = scale_time_features(D[f'x_{split}'], feature_mask, time_features)
            D[f'spt_{split}'] = [scale_time_features(x, feature_mask, time_features) for x in D[f'spt_{split}']]
            D[f'exg_{split}'] = [scale_time_features(x, feature_mask, time_features) for x in D[f'exg_{split}']]
        return D

    D['params']['model_params']['model_type'] = "sttN"

    if S and E:
        T = False  # no need to force the target series in anymore
    if abl_1:
        D['params']['model_params']['model_type'] = "baseline"
        n = False
    if abl_2:
        D['params']['model_params']['nn_params']['do_emb'] = False
        D['params']['model_params']['nn_params']['num_heads'] = 1
        D = _scale_time_features()
    if abl_3:
        D['params']['model_params']['encoder_layer_cls'] = 'MVEncoderLayer'
    if abl_4:
        D['params']['model_params']['nn_params']['predictor_cls'] = "PredictorFlatten"
    if abl_5:
        n = False
    if abl_6:
        D['params']['model_params']['model_type'] = "baseline"
        n = False
        D['params']['model_params']['nn_params']['do_emb'] = False
        D['params']['model_params']['nn_params']['num_heads'] = 1
        D = _scale_time_features()
    if abl_7:
        D['params']['model_params']['model_type'] = "baseline"
        D['params']['model_params']['nn_params']['do_emb'] = False
        D['params']['model_params']['nn_params']['num_heads'] = 1
        D = _scale_time_features()
    if abl_8:
        D['params']['model_params']['model_type'] = "baseline"
    if abl_9:
        D['params']['model_params']['model_type'] = "emb_gru"

    if not n:
        D = ablation_embedder_no_feat(D, 1)
    if not t:
        D = ablation_embedder_no_feat(D, 2)

    D['params']['model_params']['nn_params']['do_exg'] = E
    D['params']['model_params']['nn_params']['do_spt'] = S
    D['params']['model_params']['nn_params']['force_target'] = T

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
    return abl_code, D


from pipeline import model_step


"""def get_suffix(train_test_dict):
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
            'french': 0.0004, 'ushcn': 0.0004, 'adbpo': 0.0004,
        },
        'tf': '2.17.0',
        'd_model': {
            'french': 64, 'ushcn': 64, 'adbpo': 32
        },
        'num_heads': {
            'french': 4, 'ushcn': 4, 'adbpo': 2
        },
        'dff': {
            'french': 128, 'ushcn': 128, 'adbpo': 64,
        },
        'gru': {
            'french': 256, 'ushcn': 256, 'adbpo': 128
        },
        'fff': {
            'french': [256], 'ushcn': [256], 'adbpo': [128]
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
    gru = train_test_dict['params']['model_params']['nn_params']['gru']
    if gru != defaults['gru'][dataset]:
        suffix.append(f'gru{gru}')
    fff = train_test_dict['params']['model_params']['nn_params']['fff']
    if fff != defaults['fff'][dataset]:
        suffix.append(f'fff{"+".join([str(x) for x in fff])}')

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
        if lr == 0:
            suffix.append('lr0')
        else:
            suffix.append(f'lr{lr:.0e}')

    time_feats = train_test_dict['params']['prep_params']['feat_params']['time_feats']
    if time_feats:
        time_feats = tuple(sorted(time_feats))
        if time_feats != tuple(sorted(defaults['time_feats'][dataset])):
            suffix.append('+'.join(time_feats))

    if tf.__version__ != defaults['tf']:
        suffix.append(f'tf{tf.__version__.replace(".", "")}')

    if train_test_dict["params"]["model_params"]["nn_params"]["is_null_embedding"]:
        suffix.append("Nemb")

    return '_'.join(suffix)"""


def get_suffix(train_test_dict):
    def to_scientific_notation(number):
        mantissa, exponent = f"{number:.0e}".split("e")
        return float(mantissa), int(exponent)

    suffix = []

    num_layers = train_test_dict['params']['model_params']['nn_params']['num_layers']
    suffix.append(f'encs{num_layers}')
    d_model = train_test_dict['params']['model_params']['nn_params']['d_model']
    suffix.append(f'd{d_model}')
    num_heads = train_test_dict['params']['model_params']['nn_params']['num_heads']
    suffix.append(f'h{num_heads}')
    dff = train_test_dict['params']['model_params']['nn_params']['dff']
    suffix.append(f'dff{dff}')

    dropout_rate = train_test_dict['params']['model_params']['nn_params']['dropout_rate']
    suffix.append(f'dro{int(dropout_rate * 10)}')

    l2_reg = train_test_dict['params']['model_params']['nn_params']['l2_reg']
    m, e = to_scientific_notation(l2_reg)
    suffix.append(f'reg{int(m)}e{"+" if e > 0 else ""}{e}')
    lr = train_test_dict['params']['model_params']['lr']
    m, e = to_scientific_notation(lr)
    suffix.append(f'lr{int(m)}e{"+" if e > 0 else ""}{e}')

    epochs = train_test_dict['params']['model_params']['epochs']
    suffix.append(f'e{epochs}')
    patience = train_test_dict['params']['model_params']['patience']
    suffix.append(f'pat{patience}')

    time_feats = train_test_dict['params']['prep_params']['feat_params']['time_feats']
    if time_feats:
        time_feats = tuple(sorted(time_feats))
        suffix.append('+'.join(time_feats))

    return '_'.join(suffix)


def null_indicator_to_mask(train_test_dict):
    null_id = np.where(np.array(train_test_dict['x_feat_mask']) == 1)[0]
    if len(null_id) == 0:
        return train_test_dict
    for n in ['train', 'test', 'valid']:
        X = train_test_dict[f'x_{n}']
        X[:, :, null_id] = 1 - X[:, :, null_id]
        for X in train_test_dict[f'spt_{n}']:
            X[:, :, null_id] = 1 - X[:, :, null_id]
        for X in train_test_dict[f'exg_{n}']:
            X[:, :, null_id] = 1 - X[:, :, null_id]
    return train_test_dict


"""def sample_aux_mask(train_test_dict, rate=0.1):

    def _sample_aux_mask(mask):
        num_to_mask = max(1, round(len(mask) * rate))

        real_indices = np.where(mask == 1)[0]
        if len(real_indices) < num_to_mask:
            # Not enough real values to mask the required number
            selected = real_indices  # Mask all that are available
        else:
            selected = np.random.choice(real_indices, size=num_to_mask, replace=False)

        aux_mask = np.zeros_like(mask)
        aux_mask[selected] = 1
        return aux_mask

    def _apply_sample_aux_mask(X):
        mask = X[:, :, null_id]
        aux_mask = []
        for b in range(mask.shape[0]):
            aux_mask.append(_sample_aux_mask(mask[b]))
        aux_mask = np.array(aux_mask)[:, :, np.newaxis]
        X = np.concatenate([X, aux_mask], axis=2)
        return X

    null_id = np.where(np.array(train_test_dict['x_feat_mask']) == 1)[0]
    if len(null_id) == 0:
        return train_test_dict
    null_id = null_id[0]  # Assuming only one null feature for simplicity
    for split in ['train', 'test', 'valid']:
        X = train_test_dict[f'x_{split}']
        X = _apply_sample_aux_mask(X)
        train_test_dict[f'x_{split}'] = X

        for i in range(len(train_test_dict[f'spt_{split}'])):
            X = train_test_dict[f'spt_{split}'][i]
            X = _apply_sample_aux_mask(X)
            train_test_dict[f'spt_{split}'][i] = X

        for i in range(len(train_test_dict[f'exg_{split}'])):
            X = train_test_dict[f'exg_{split}'][i]
            X = _apply_sample_aux_mask(X)
            train_test_dict[f'exg_{split}'][i] = X
    return train_test_dict"""


def ablation(
        # train_test_dict: dict,
        pickle_file: str,
        results_file: str,
        checkpoint_basedir: str,
        path_params: dict,
        prep_params: dict,
        eval_params: dict,
        model_params: dict,
):
    ablations_mapping = [
        'E_nt',
        # 'E_n',
        # 'E_t',
        # 'E',
        # 'E_nt_1',
        # 'E_nt_2',
        # 'E_nt_3',
        # 'E_nt_4',
        # 'E_nt_6',
        # 'E_nt_7',
        # 'E_nt_8',
        # 'E_nt_9',
        # 'E_nt_A',
    ]

    for name in ablations_mapping:
        print('Loading from', pickle_file, '...', end='', flush=True)
        with open(pickle_file, "rb") as f:
            train_test_dict = pickle.load(f)
        print(' done!')
        train_test_dict['params'] = {
            'path_params': deepcopy(path_params),
            'prep_params': deepcopy(prep_params),
            'eval_params': deepcopy(eval_params),
            'model_params': deepcopy(model_params),
        }
        train_test_dict = null_indicator_to_mask(train_test_dict)

        name, train_test_dict = apply_ablation_code(name, train_test_dict)
        seed = model_params['seed']
        data_seed = prep_params['data_seed']
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        name += f"_d{data_seed}_s{seed}"

        suffix = get_suffix(train_test_dict)
        if suffix: name = f"{name}#{suffix}"
        if '#' not in name:
            name += '#'

        """train_test_dict['params']['model_params']['model_type'] = "istf_cls"
        name += '_CLS'"""

        """train_test_dict['params']['model_params']['model_type'] = "istf_interp_cls"
        name += '_PretrCLS'"""

        train_test_dict['params']['model_params']['model_type'] = "istf_attnpool"
        name += '_AttnPool' if train_test_dict['params']['model_params']['nn_params']['pooling'] == 'attn' else '_MeanPool'

        """train_test_dict['params']['model_params']['model_type'] = "istf_rope"
        name = name + (
            '_AttnPool' if train_test_dict['params']['model_params']['nn_params']['pooling'] == 'attn' else '_MeanPool'
        )
        name += '_RoPE'  # rotary positional embeddings"""

        with open(pickle_file.replace(".pickle", "_aux.pickle"), "rb") as f:
            train_test_dict_aux = pickle.load(f)
        keep_ids = np.where(np.isin(train_test_dict['x_feat_mask'], [0, 1]))[0]
        for n in ['train', 'test', 'valid']:
            def _f(D, D_aux, key, i=None):
                if i is not None:
                    X = D[key][i]
                    X_aux = D_aux[key][i][:, :, keep_ids]
                    D[key][i] = np.concatenate([X, X_aux], axis=2)
                    return D
                X = D[key]
                X_aux = D_aux[key][:, :, keep_ids]
                D[key] = np.concatenate([X, X_aux], axis=2)
                return D
            train_test_dict = _f(train_test_dict, train_test_dict_aux, f'x_{n}')
            for i in range(len(train_test_dict[f'spt_{n}'])):
                train_test_dict = _f(train_test_dict, train_test_dict_aux, f'spt_{n}', i)
            for i in range(len(train_test_dict[f'exg_{n}'])):
                train_test_dict = _f(train_test_dict, train_test_dict_aux, f'exg_{n}', i)

        # name += "+NoMask"
        # name += "+Mean"
        # name += "+bias"
        name += "_SW" if train_test_dict['params']['model_params']['nn_params']['shared_weights'] else ""  # shared weights
        # name += "_IV"  # I: shared weights + embedder w/o regularizing small layer, II: shared weights, III: shared weights + no CLS in global attention
        # name += "_iqr"
        name += "_sk500"  # scheduler options: sk4000, sk6e, skNoam1K
        # name += "_Mean"
        # name += "_Intp5"
        # name += "_Recn"
        # name += "_Aux_SF_dro"
        # name += "_Aux1PF"  # _Aux1PF
        # name += '_NoStatic'
        # name += "_Static2"  # StaticEmb, Static2
        # name += "_TFW"
        # name += "_R1"  # 1 regressor for multiple outputs
        # name += "_KMask"  # +GAwithCLS, +GAnoCLS, +CLSAttn
        name += "_LVEmbS+"  # Learnable VarEmbs initialized as regular simplex + learnable scale
        # name += "_PredAll"
        # name += "_Add2Enc"
        # name += "_LGSW"  # Local Global Shared Weights
        # name += "_PtMean"  # Pretrain Mean
        # name += "_H"  # multi-head attention like torch
        # name += "_k1"
        # name += "_regEmb" + str(train_test_dict['params']['model_params']['nn_params']['l2_reg'] * 10).replace('0.', '')
        # name += "_noWu"  # no warmup
        # name += "_CLSnoPE"  # no positional encoding for the CLS token
        # name += "_Pt100eFIX" if train_test_dict['params']['model_params']['pretrain'] else "_Ft"
        name += "_Pt" if train_test_dict['params']['model_params']['pretrain'] else ""
        # name += "Load"
        name += "+Loss0.3obs+0.1avg"  # loss weights
        # name += "_wu10e"
        name += "_Ft"
        name += "+lr=1e-4"  # 1e-4
        # name += "+Loss0.1avg"
        # name += "_FtNoEmb"  # Do not fine-tune the embedding layer
        # name += "_reg+"  # apply regularization to task-specific heads too
        # name += "_tanh" if train_test_dict['params']['model_params']['nn_params']['fff'] else ""
        # name += "_W15"  # window 15, + optimized version
        # name += "_"
        # name += "_minmax"
        name += "_b1000"  # sinusoidal embedding with base=100
        # name += "_Prenorm"  # include norm before heads
        # name += "_EmbScaleD"
        # name += "_TPEmbD"  # time features and position embedded together
        # name += "_TembPrd"  # time features embedded as periodic
        name += "_P|T"

        """train_test_dict['params']['model_params']['encoder_cls'] = "ParallelEncoder"
        name += '_P'"""

        if name.endswith('#'):
            name = name[:-1]
        # train_test_dict = ablation_impute_mean(train_test_dict)

        print(f"\n{name}: {train_test_dict['params']['model_params']['model_type']}")

        os.makedirs(checkpoint_basedir, exist_ok=True)
        # run_id = len(os.listdir(checkpoint_basedir)) + 1
        run_id = 1
        while os.path.exists(checkpoint_basedir + "/" + f'run{run_id:04d}'):
            run_id += 1
        # run_id = 35
        print('Run ID:', run_id)
        checkpoint_dir = checkpoint_basedir + "/" + f'run{run_id:04d}'
        # checkpoint_dir = checkpoint_basedir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # if dataset is "french", treat the last exogenous variable differently
        if path_params['type'] == 'french':
            # remove last exogenous variable from exogenous inputs
            # for n in ['train', 'test', 'valid']:
            #     train_test_dict[f'exg_{n}'] = train_test_dict[f'exg_{n}'][:-1]

            # specify its index as static features
            train_test_dict["params"]["model_params"]["nn_params"]["static_feats_ids"] = [3]

            """# extract static features
            for split in ['train', 'test', 'valid']:
                exg = train_test_dict[f'exg_{split}']  # (v, b, t, f)
                static_ids = [2]
                static = [exg[s][:, 0, 0] for s in static_ids]  # (num_static, B)
                exg = [exg[s] for s in range(len(exg)) if s not in static_ids]
                train_test_dict[f'exg_{split}'] = exg
                static = np.stack(static, axis=1)  # (B, num_static)
                exg_static = static[:, np.newaxis, :]  # (B, 1, num_static)
                spt_static = None  # fixme
                train_test_dict[f'spt_static_{split}'] = spt_static
                train_test_dict[f'exg_static_{split}'] = exg_static"""

            ...

        with open(checkpoint_dir + "/model_params.json", "w") as f:
            _model_params = deepcopy(train_test_dict['params']['model_params'])
            _model_params["name"] = name
            json.dump(_model_params, f, indent=4)

        met, cur = model_step(train_test_dict, train_test_dict['params']['model_params'], checkpoint_dir)

        # non-grid results
        if os.path.exists(results_file):
            results = pd.read_csv(results_file, index_col=0).to_dict(orient='index')
        else:
            results = {}
        met["run_id"] = run_id
        results[name] = met
        # pd.DataFrame(results).T.to_csv(results_file, index=True)
        columns = (
            "run_id,test_r2,test_mae,test_mse,test_wMAPE,valid_r2,val_mae,val_mse,val_wMAPE,"
            "pretr_test_mae,pretr_test_mse,pretr_test_wMAPE,"
        ).split(',')
        columns = [c for c in columns if c in met]
        pd.DataFrame.from_dict(results, orient='index')[columns].to_csv(results_file, index=True)

        curves_path = results_file.replace('.csv', '')
        os.makedirs(curves_path, exist_ok=True)
        # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        curves_path = curves_path + "/" + "curves_" + f'run{run_id:04d}' + ".pickle"
        cur["name"] = name
        cur["params"] = train_test_dict["params"]["model_params"]
        with open(curves_path, 'wb') as f:
            pickle.dump(cur, f)

        """# grid results
        import json
        res["name"] = name
        res["params"] = train_test_dict["params"]["model_params"]
        results_path = results_file.replace('.csv', '/')
        os.makedirs(results_path, exist_ok=True)
        results_path += timestamp + '.json'
        with open(results_path, 'w') as f:
            # res["params"]["nn_params"]["null_max_size"] = int(res["params"]["nn_params"]["null_max_size"])
            res["test_mae"] = float(res["test_mae"])
            res["test_mse"] = float(res["test_mse"])
            json.dump(res, f, indent=4)"""


def main():
    path_params, prep_params, eval_params, model_params = parse_params()
    if model_params['cpu']:
        tf.config.set_visible_devices([], 'GPU')
    seed = model_params['seed']
    data_seed = prep_params['data_seed']

    results_dir = './output/results'
    pickle_dir = './output/pickle'
    model_dir = './output/model'

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(pickle_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # subset = path_params['ex_filename']
    # if path_params['type'] == 'adbpo' and 'exg_w_tp_t2m' in subset:
    #     subset = os.path.basename(subset).replace('exg_w_tp_t2m', 'all').replace('.pickle', '')
    # elif 'all' in subset:
    #     path_params['ex_filename'] = None
    # else:
    #     subset = os.path.basename(subset).replace('subset_agg_', '').replace('.csv', '')
    path_params['ex_filename'] = None
    nan_percentage = path_params['nan_percentage']
    num_past = prep_params['ts_params']['num_past']
    num_fut = prep_params['ts_params']['num_fut']
    num_spt = prep_params['spt_params']['num_spt']
    max_dist_th = prep_params['spt_params']['max_dist_th']

    # conf_name = f"{path_params['type']}_{subset}_nan{int(nan_percentage * 10)}_np{num_past}_nf{num_fut}"
    # conf_name += "_iqr"
    conf_name = get_conf_name(
        dataset=path_params['type'],
        nan_percentage=nan_percentage,
        num_past=num_past,
        num_fut=num_fut,
        num_spt=num_spt,
        max_dist_th=max_dist_th,
        seed=data_seed,
        dev=path_params['dev']
    )
    print('configuration:', conf_name)
    results_file = os.path.join(results_dir, f"{conf_name}.csv")
    # results_file = os.path.join(results_dir, f"{conf_name}_mse.csv")
    pickle_file = os.path.join(pickle_dir, f"{conf_name}.pickle")
    checkpoint_dir = os.path.join(model_dir, conf_name)

    if path_params['force_data_step'] or not os.path.exists(pickle_file):
        random.seed(data_seed)
        np.random.seed(data_seed)
        train_test_dict = data_step(
            path_params, prep_params, eval_params, scaler_type=model_params['transform_type']
        )
        train_test_dict, train_test_dict_aux = train_test_dict
        pickle_aux_path = pickle_file.replace('.pickle', '_aux.pickle')
        with open(pickle_aux_path, "wb") as f:
            print('Saving to', pickle_aux_path, '...', end='', flush=True)
            pickle.dump(train_test_dict_aux, f)
            print(' done!')
        del train_test_dict_aux
        with open(pickle_file, "wb") as f:
            print('Saving to', pickle_file, '...', end='', flush=True)
            pickle.dump(train_test_dict, f)
            print(' done!')
        del train_test_dict

    ablation(
        pickle_file=pickle_file,
        results_file=results_file,
        checkpoint_basedir=checkpoint_dir,
        path_params=path_params,
        prep_params=prep_params,
        eval_params=eval_params,
        model_params=model_params,
    )

    # os.remove(pickle_file)  # remove pickle file to save space

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
