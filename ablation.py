import os
import pickle
import random
from collections import namedtuple
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf

from ists.model.encoder import SpatialExogenousEncoder
from pipeline import data_step, model_step, parse_params, change_params, get_scalers, get_scalers_station


def no_ablation(train_test_dict) -> dict:
    # train_test_dict['params']['model_params']['model_type'] = "sttransformer"
    return train_test_dict


def ablation_embedder_no_feat(train_test_dict, code) -> dict:
    for n in ['train', 'test', 'valid']:
        cond_x = [x != code for x in train_test_dict['x_feat_mask']]
        train_test_dict[f'x_{n}'] = train_test_dict[f'x_{n}'][:, :, cond_x]
        train_test_dict[f'spt_{n}'] = [x[:, :, cond_x] for x in train_test_dict[f'spt_{n}']]

        # cond_exg = [x != code for x in train_test_dict['exg_feat_mask']]
        # train_test_dict[f'exg_{n}'] = train_test_dict[f'exg_{n}'][:, :, cond_exg]
        train_test_dict[f'exg_{n}'] = [x[:, :, cond_x] for x in train_test_dict[f'exg_{n}']]

    train_test_dict['x_feat_mask'] = [x for x in train_test_dict['x_feat_mask'] if x != code]
    train_test_dict['exg_feat_mask'] = [x for x in train_test_dict['exg_feat_mask'] if x != code]

    if code == 1:
        train_test_dict['null_max_size'] = None

    if code == 2:
        train_test_dict['time_max_sizes'] = []
        train_test_dict['exg_time_max_sizes'] = []

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
    train_test_dict['params']['model_params']['encoder_cls'] = SpatialExogenousEncoder
    return train_test_dict


def apply_ablation_code(abl_code: str, train_test_dict):
    suffix = None
    if '#' in abl_code:
        abl_code, suffix = abl_code.split('#')

    T, S, E = 'T' in abl_code, 'S' in abl_code, 'E' in abl_code
    G = 'G' in abl_code
    n, t = 'n' in abl_code, 't' in abl_code
    M = 'M' in abl_code
    if S and E:
        T = False  # no need to force the target series in anymore

    if M:
        train_test_dict['params']['model_params']['nn_params']['multivar'] = True
    if not n:
        train_test_dict = ablation_embedder_no_feat(train_test_dict, 1)
    if not t:
        train_test_dict = ablation_embedder_no_feat(train_test_dict, 2)
    train_test_dict['params']['model_params']['nn_params']['do_exg'] = E
    train_test_dict['params']['model_params']['nn_params']['do_spt'] = S
    train_test_dict['params']['model_params']['nn_params']['force_target'] = T
    train_test_dict['params']['model_params']['nn_params']['do_glb'] = G

    _abl_code = (
            ''
            + ('E' if E else '')
            + ('S' if S else '')
            + ('T' if T else '')
            + ('_G' if G else '')
            + ('_' if n or t else '')
            + ('n' if n else '')
            + ('t' if t else '')
            + ('_M' if M else '')
            + ('\n' + suffix if suffix else '')
    )
    return _abl_code, train_test_dict


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
        results = {}

    selected_model = train_test_dict['params']['model_params']['model_type'][:3].upper() #9

    # ablations_mapping = {
    #     selected_model: no_ablation,
    # }
    #
    # if ablation_embedder:
    #     ablations_mapping.update({
    #         f'{selected_model} w/o time enc': ablation_embedder_no_time,
    #         f'{selected_model} w/o null enc': ablation_embedder_no_null,
    #         f'{selected_model} w/o time null enc': ablation_embedder_no_time_null,
    #     })
    #
    # if ablation_encoder:
    #     ablations_mapping.update({
    #         'T': ablation_encoder_t,
    #         'S': ablation_encoder_s,
    #         'E': ablation_encoder_e,
    #         'TE': ablation_encoder_te,
    #         'TS': ablation_encoder_ts,
    #         'SE': ablation_encoder_se,
    #     })
    #
    # if ablation_extra:
    #     ablations_mapping.update(ablation_extra)

    # ablations_mapping = {
    #     # selected_model: no_ablation,
    #     # 'NO_GLB': ablation_no_global_encoder,
    #     # 'S': ablation_encoder_s,
    #     # 'TS': ablation_encoder_ts,
    #     # f'{selected_model} w/o time enc': ablation_embedder_no_time,
    #     'STT_MV': ablation_multivariate,
    #     # 'MV_no_null': ablation_multivariate_no_null,
    #     # 'MV_TS': ablation_multivariate_ts,
    #     # 'MV_TE': ablation_multivariate_te,
    #     # 'MV_NO_GLB': ablation_multivariate_no_global_encoder,
    #     # 'MV_TS_no_null_no_GLB': ablation_multivariate_ts_no_null_no_global_encoder,
    #     # 'STT_MV_null_sum': ablation_multivariate,
    #     # 'STT2': ablation_stt_2,
    # }

    ablations_mapping = [
        # 'ES__G_nt_M#null_sum_norm_iqr',
        'ES____nt#null_sum_norm_iqr',
        # 'ES__G__t_M#norm_iqr',
        # '_ST_G__t_M#norm_iqr',
        # '_ST_G_nt_M#null_sum_norm_iqr',
        # '__T___nt',
        # '_S____nt_M',
        # '_ST_G_nt',
        # '_ST_G__t',
        # '_S____nt'
    ]

    for name in ablations_mapping:
        # func = ablations_mapping[name]

        # Configure ablation test
        # train_test_dict = func(deepcopy(train_test_dict))
        name, train_test_dict = apply_ablation_code(name, deepcopy(train_test_dict))

        # Exec ablation test
        print(f"\n{name}: {train_test_dict['params']['model_params']['model_type']}")
        if train_test_dict['params']['model_params']['seed'] != 42:
            name += '_seed' + str(train_test_dict['params']['model_params']['seed'])
        results[name] = model_step(train_test_dict, train_test_dict['params']['model_params'], checkpoint_path)

        # Save results
        pd.DataFrame(results).T.to_csv(results_path, index=True)

    return pd.DataFrame(results).T


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
            path_params, prep_params, eval_params, keep_nan=False, scaler_type=model_params['transform_type']
        )

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

    ablation(
        train_test_dict=train_test_dict,
        results_path=results_path,
        pickle_path=pickle_path,
        checkpoint_path=checkpoint_path,
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
    main()
