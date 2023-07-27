from typing import Dict, Tuple
import numpy as np
import pandas as pd

from .piezo.read import load_piezo_data
from .ushcn.read import load_ushcn_data
from .french.read import load_frenchpiezo_data


def load_data(
        ts_filename: str,
        context_filename: str,
        ex_filename: str,
        data_type: str,
        nan_percentage: float = 0,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
    # Set a fixed seed for reproducibility
    fixed_seed = 42
    np.random.seed(fixed_seed)

    if data_type == 'adbpo':
        return load_piezo_data(
            ts_filename=ts_filename,
            context_filename=context_filename,
            ex_filename=ex_filename,
            nan_percentage=nan_percentage
        )
    elif data_type == 'french':
        return load_frenchpiezo_data(
            ts_filename=ts_filename,
            context_filename=context_filename,
            subset_filename=ex_filename,
            nan_percentage=nan_percentage
        )
    elif data_type == 'ushcn':
        return load_ushcn_data(
            ts_filename=ts_filename,
            subset_filename=ex_filename,
            nan_percentage=nan_percentage
        )
    else:
        raise ValueError(f'Dataset {data_type} is not supported, it must be: adbpo, french, or ushcn')
