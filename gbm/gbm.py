import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
import pandas as pd


class GBM:
    notional: float
    drift: float
    time_to_maturity: float
    number_of_paths: int
    number_of_time_steps: int
    volatility: float
    excel_file_path: str
    variance_interpolator: interp1d

    def __init__(self, drift: float, volatility: float, excel_file_path: str, sheet_name: str):
        self.drift = drift
        self.volatility = volatility
        # Here 'variance' means 'sigma**2 * time'.
        self.variance_interpolator = self.setup_variance_interpolator(excel_file_path, sheet_name)

    def get_gbm_paths(
            self,
            number_of_paths: int,
            number_of_time_steps: int,
            notional: float,
            initial_spot: float,
            time_to_maturity: float,
            is_time_dependent: bool) -> np.ndarray:
        """

        Generates the GBM paths used to price various instruments. The volatility used can be time dependent or
        time-independent.

        :param number_of_paths: Number of the current value.
        :param number_of_time_steps: Number of time steps.
        :param notional: The notional amount.
        :param initial_spot: Initial spot price.
        :param time_to_maturity: Time to maturity (in years).
        :param is_time_dependent: Indicates whether a time-dependent or time-independent volatility
                is being used.
        :return: The simulated GBM paths.

        """
        paths: np.ndarray = np.array(np.zeros((number_of_paths, number_of_time_steps + 1)))
        paths[:, 0] = initial_spot * notional
        dt: float = time_to_maturity / number_of_time_steps

        if is_time_dependent:
            for j in range(1, number_of_time_steps + 1):
                z: float = norm.ppf(np.random.uniform(0, 1, number_of_paths))
                volatility: float = self.variance_interpolator(j * dt)/100
                # print(f'volatility is equal to {volatility}')
                paths[:, j] = \
                    paths[:, j - 1] * \
                    np.exp((self.drift - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * z)
            return paths

        else:
            for j in range(1, number_of_time_steps + 1):
                z: float = norm.ppf(np.random.uniform(0, 1, number_of_paths))
                paths[:, j] = \
                    paths[:, j - 1] * \
                    np.exp((self.drift - 0.5 * self.volatility ** 2) * dt + self.volatility * np.sqrt(dt) * z)
            return paths

    @staticmethod
    def setup_variance_interpolator(excel_file_path: str, sheet_name: str) -> interp1d:
        """
        Calculates the time dependent volatilities using interpolation of a volatility term structure.

        :param excel_file_path: The path of the file. In other words, the file path where the Excel file is.
        :param sheet_name: The sheet name in the Excel file housing the relevant data.
        :return: Returns the time dependent volatility.
        """
        excel_records = pd.read_excel(excel_file_path, sheet_name=sheet_name)
        excel_records_df = excel_records.loc[:, ~excel_records.columns.str.contains('^Unnamed')]
        tenors: list[float] = list(map(float, excel_records_df.Tenors))
        vols: list[float] = list(map(float, excel_records_df.Quotes))
        squared_vols: list[float] = list(map(lambda x: pow(x, 2), vols))
        new_vols = []
        for dt1, dt2 in zip(squared_vols, tenors):
            new_vols.append(dt1 * dt2)
        interpolated_volatility: interp1d = interp1d(tenors, new_vols, kind='linear', fill_value='extrapolate')
        return interpolated_volatility

    def get_time_dependent_vol(self, tenor: float) -> float:
        return np.sqrt(self.variance_interpolator(tenor) / tenor) / 100
