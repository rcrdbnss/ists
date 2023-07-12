import os
import json
import pandas as pd
from sklearn.model_selection import ParameterGrid

from pipeline import data_step, model_step, get_params, parse_params


def my_model_search(path_params, prep_params, eval_params, model_params):
    grid_search_params = {
        'nn_params': [
            {
                'kernel_size': 3,
                'd_model': 32,
                'num_heads': 8,
                'dff': 64,
                'fff': 32,
                'dropout_rate': 0.2
            },
            {
                'kernel_size': 3,
                'd_model': 128,
                'num_heads': 8,
                'dff': 256,
                'fff': 128,
                'dropout_rate': 0.2
            },
        ],
        "transform_type": ["standard"],  # , "minmax"],
        "epochs": [100],  # , 100],
        "loss": ['mse'],  # , "mae"],
        "exg_num_past": [60, 72],
        "spt_num_past": [24, 36],
        "spt_num_spt": [2, 5],
        "x_num_past": [24, 36, 48],
    }

    configs = []
    params = list(ParameterGrid(grid_search_params))
    for param in ParameterGrid(grid_search_params):
        # Prep params
        prep_params['ts_params']['num_past'] = param['x_num_past']
        prep_params['spt_params']['num_past'] = param['spt_num_past']
        prep_params['spt_params']['num_spt'] = param['spt_num_spt']
        prep_params['exg_params']['num_past'] = param['exg_num_past']

        # Model params
        model_params['transform_type'] = param['transform_type']
        model_params['epochs'] = param['epochs']
        model_params['loss'] = param['loss']
        model_params['nn_params'] = param['nn_params']

        config = (
            path_params,
            prep_params,
            eval_params,
            model_params
        )
        configs.append(config)
    return configs, params


def main():
    output_dir = './output/stt_model'
    # Read input base params
    # path_params, prep_params, eval_params, model_params = get_params()
    path_params, prep_params, eval_params, model_params = parse_params()
    # Extract all possibilities
    configs, params = my_model_search(path_params, prep_params, eval_params, model_params)

    results = []
    for i in range(len(configs)):
        print('\n\n' + '-' * 50)
        print(i)
        param = params[i]
        print(param)

        # Extract pipeline configuration params
        path_params, prep_params, eval_params, model_params = configs[i]
        # Data step
        train_test_dict = data_step(path_params, prep_params, eval_params)
        # Model step
        res = model_step(train_test_dict, model_params)
        # Extract tuned params
        res.update(param)
        results.append(res)

        print(param)


    results = pd.DataFrame(results)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    results_filename = os.path.join(output_dir, 'results.csv')
    results.to_csv(results_filename, index=False)

    params_filename = os.path.join(output_dir, 'params.json')
    with open(params_filename, 'w') as f:
        json.dump(params, f)

    print('Hello World!')


if __name__ == '__main__':
    main()
