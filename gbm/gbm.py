import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import jarque_bera
from scipy.interpolate import interp1d
from collections import namedtuple
import pandas as pd


def get_time_dependent_volatility(excel_file_path: str) -> interp1d:
    """

    Calculates the time dependent volatilities using interpolation of a volatility term structure.

    :param excel_file_path: The path of the file. In other words, the file path where the Excel file is.
    :return: Returns the time dependent volatility.
    """
    excel_records = pd.read_excel(excel_file_path)
    excel_records_df = excel_records.loc[:, ~excel_records.columns.str.contains('^Unnamed')]
    tenors: list[float] = list(map(float, excel_records_df.Tenors))
    vols: list[float] = list(map(float, excel_records_df.Quotes))
    squared_vols: list[float] = list(map(lambda x: pow(x, 2), vols))
    new_vols = []
    for dt1, dt2 in zip(squared_vols, tenors):
        new_vols.append(dt1 * dt2)
    interpolated_volatility: interp1d = interp1d(tenors, new_vols, kind='linear', fill_value='extrapolate')
    return interpolated_volatility


class GBM:
    notional: float
    drift: float
    time_to_maturity: float
    number_of_paths: int
    number_of_time_steps: int
    volatility: float
    excel_file_path: str
    volatility_interpolator: interp1d

    def __init__(self, drift: float, excel_file_path: str):
        self.drift = drift
        self.volatility_interpolator = get_time_dependent_volatility(excel_file_path)

    def generate_gbm_paths_with_time_dependent_vols(
            self,
            number_of_paths: int,
            number_of_time_steps: int,
            notional: float,
            initial_spot: float,
            time_to_maturity) -> np.ndarray:
        paths: np.ndarray = np.array(np.zeros((number_of_paths, number_of_time_steps + 1)))
        paths[:, 0] = initial_spot * notional
        dt: float = time_to_maturity / number_of_time_steps

        for j in range(1, number_of_time_steps + 1):
            z: float = norm.ppf(np.random.uniform(0, 1, number_of_paths))
            volatility: float = self.volatility_interpolator(j * dt) / 100
            paths[:, j] = \
                paths[:, j - 1] * np.exp((self.drift - 0.5 * volatility ** 2) * dt + volatility *
                                         math.sqrt(dt) * z)

        return paths

