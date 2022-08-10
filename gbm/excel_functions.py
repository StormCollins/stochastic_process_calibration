import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from collections import namedtuple

VolData = namedtuple('VolData', ['Moneyness', 'Tenors', 'VolSurface'])


def read_vol_surface(
        file_path: str,
        sheet_name: str,
        data_date: datetime.date) -> VolData:
    data: pd.DataFrame = pd.read_excel(file_path, sheet_name=sheet_name)

    moneyness_indices: list[float] =\
        [i for i in range(len(data.columns.values))
         if isinstance(data.columns.values[i], int) or isinstance(data.columns.values[i], float)]
    moneyness: np.ndarray = np.array(data.columns[moneyness_indices].values)

    end_dates = data['End Date']
    tenors: np.ndarray = np.empty(len(end_dates))
    for i, end_date in enumerate(end_dates):
        tenors[i] = (end_date.date() - data_date).days / 365.0

    vols = data[moneyness].values.tolist()
    vol_data = VolData(moneyness, tenors, vols)
    return vol_data


def get_vol(moneyness: list[float], tenor: list[float], vol_surface: VolData) -> list[float]:
    """
    Returns an interpolated volatility from a volatility surface for a given moneyness and tenor.

    :param vol_surface: The volatilities.
    :param moneyness: The moneyness strikes.
    :param tenor: The length of time remaining in the contract.
    :return: Interpolated volatility.
    """

    bi_linear_interpolation\
        = interp2d(vol_surface.Moneyness, vol_surface.Tenors, vol_surface.VolSurface, kind='linear')
    return bi_linear_interpolation(moneyness, tenor)


file_path: str = r'C:\GitLab\stochastic_process_calibration_2022\gbm\vol-surface-data-2022-06-30.xlsx'
sheet_name: str = 'S&P500 Clean Vol Surface'
vol_surface: VolData = read_vol_surface(file_path, sheet_name, datetime.date(2022, 6, 30))
tenor: list[float] = [0.8, 0.9]
moneyness: list[float] = [1.1, 1.2]

print(f'bi-linear interpolation is:'
      f'{get_vol(moneyness, tenor, vol_surface)}')

# print(data['End Date'])

# Create the volatility surface

# def plot3D(X, Y, Z):
#     fig = plt.figure()
#     ax = Axes3D(fig, azim=-29, elev=50)
#
#     ax.plot(X, Y, Z, 'o')
#
#     plt.xlabel("expiry")
#     plt.ylabel("strike")
#     plt.show()
#
#     return plot3D(X,Y,Z)
