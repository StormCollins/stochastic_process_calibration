import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
import pandas as pd


class TimeDependentGBM:
    """
    Class for generating GBM paths where volatility is time-independent.
    """

    def __init__(
            self,
            drift: float,
            excel_file_path: str,
            sheet_name: str,
            notional: float,
            initial_spot: float,
            time_to_maturity: float):
        """
        Class constructor.

        :param drift: Drift.
        :param excel_file_path: Path to Excel file containing the ATM volatility term structure.
        :param sheet_name: The sheet name of the Excel file containing the ATM volatility term structure.
        :param notional: Notional.
        :param initial_spot: Initial spot.
        :param time_to_maturity: Time to maturity.
        """
        self.drift: float = drift
        # Here 'variance' means 'sigma**2 * time'.
        self.variance_interpolator: interp1d = self.setup_variance_interpolator(excel_file_path, sheet_name)
        self.notional: float = notional
        self.initial_spot: float = initial_spot
        self.time_to_maturity: float = time_to_maturity

    def get_gbm_paths(
            self,
            number_of_paths: int,
            number_of_time_steps: int,
            notional: float,
            initial_spot: float,
            time_to_maturity: float) -> np.ndarray:
        """
        Generates the GBM paths used to price various instruments.

        :param number_of_paths: Number of the current value.
        :param number_of_time_steps: Number of time steps.
        :param notional: The notional amount.
        :param initial_spot: Initial spot price.
        :param time_to_maturity: Time to maturity (in years).
        :return: The simulated GBM paths.
        """
        paths: np.ndarray = np.array(np.zeros((number_of_paths, number_of_time_steps + 1)))
        paths[:, 0] = initial_spot * notional
        dt: float = time_to_maturity / number_of_time_steps

        for j in range(1, number_of_time_steps + 1):
            z: float = norm.ppf(np.random.uniform(0, 1, number_of_paths))
            volatility: float = self.get_time_dependent_vol(j * dt)
            # print(f 'volatility is equal to {volatility}')
            paths[:, j] = \
                paths[:, j - 1] * \
                np.exp((self.drift - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * z)

        return paths

    @staticmethod
    def setup_variance_interpolator(excel_file_path: str, sheet_name: str) -> interp1d:
        """
        Sets up the interpolator for the variance i.e., 'volatility**2 * time'.

        :param excel_file_path: The path of the file. In other words, the file path where the Excel file is.
        :param sheet_name: The sheet name in the Excel file housing the relevant data.
        :return: Returns the time dependent variance interpolator.
        """
        excel_records = pd.read_excel(excel_file_path, sheet_name=sheet_name)
        excel_records_df = excel_records.loc[:, ~excel_records.columns.str.contains('^Unnamed')]
        tenors: list[float] = list(map(float, excel_records_df.Tenors))
        vols: list[float] = list(map(float, excel_records_df.Quotes))
        squared_vols: list[float] = list(map(lambda x: pow(x, 2), vols))
        new_vols = []

        for dt1, dt2 in zip(squared_vols, tenors):
            new_vols.append(dt1 * dt2)

        variance_interpolator: interp1d = interp1d(tenors, new_vols, kind='linear', fill_value='extrapolate')
        return variance_interpolator

    def get_time_dependent_vol(self, tenor: float) -> float:
        return np.sqrt(self.variance_interpolator(tenor) / tenor) / 100
