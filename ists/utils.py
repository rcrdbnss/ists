from typing import TypedDict, Literal

import calendar
from datetime import datetime, timedelta

import pandas as pd


# class TS(TypedDict):
#     ts: pd.DataFrame
#     exogenous: pd.DataFrame
#     distances: pd.Series


def move_to_end_of_week_or_month(dt: datetime.date, move_to: Literal['M', 'W']):
    if move_to == 'W':
        # Move to the end of the week
        dt = dt + timedelta(days=(6 - dt.weekday()))
    elif move_to == 'M':
        # Move to the end of the month
        _, last_day = calendar.monthrange(dt.year, dt.month)
        dt = dt.replace(day=last_day)
    else:
        raise ValueError("Invalid move_to option. Must be 'M', 'W'.")

    return dt
