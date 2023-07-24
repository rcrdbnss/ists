from typing import Dict, Tuple
import pandas as pd

from .piezo.read import load_piezo_data
from .ushcn.read import load_ushcn_data
from .french.read import load_frenchpiezo_data


def load_data(
        ts_filename: str,
        context_filename: str,
        ex_filename: str,
        data_type: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
    if data_type == 'adbpo':
        return load_piezo_data(ts_filename=ts_filename, context_filename=context_filename, ex_filename=ex_filename)
    elif data_type == 'french':
        return load_frenchpiezo_data(ts_filename=ts_filename, context_filename=context_filename)
    elif data_type == 'ushcn':
        return load_ushcn_data(ts_filename=ts_filename)
    else:
        raise ValueError(f'Dataset {data_type} is not supported, it must be: adbpo, french, or ushcn')
