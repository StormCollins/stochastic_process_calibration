"""
Contains a class for generating GBM paths where volatility is time-independent.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import jarque_bera
from scipy.stats import norm
from src.enums_and_named_tuples.path_statistics import PathStatistics
from src.utils.plot_utils import PlotUtils


class TimeDependentGBM:
    """
    Class for generating GBM paths where volatility is time-independent.
    """

    def __init__(
            self,
            drift: float,
            excel_file_path: str,
            sheet_name: str,
            initial_spot: float):
        """
        Class constructor.

        :param drift: Drift.
        :param excel_file_path: Path to Excel file containing the ATM volatility term structure.
        :param sheet_name: The sheet name of the Excel file containing the ATM volatility term structure.
        :param initial_spot: Initial spot.
        """
        self.drift: float = drift
        # Here 'variance' means 'sigma**2 * time'.
        self.variance_interpolator: interp1d = self.setup_variance_interpolator(excel_file_path, sheet_name)
        self.bootstrapped_vol_interpolator: interp1d = \
            self.setup_bootstrapped_volatility_interpolator(excel_file_path, sheet_name)

        self.initial_spot: float = initial_spot

    def get_paths(self, number_of_paths: int, number_of_time_steps: int, time_to_maturity: float) -> np.ndarray:
        """
        Generates the GBM paths used to price various instruments.

        :param number_of_paths: Number of the current value.
        :param number_of_time_steps: Number of time steps.
        :param time_to_maturity: Time to maturity.
        :return: The simulated GBM paths.
        """
        dt: float = time_to_maturity / number_of_time_steps
        time_steps: np.ndarray = np.linspace(0, time_to_maturity, number_of_time_steps + 1)
        volatility = self.get_time_dependent_vol(time_steps[1:])
        z = np.random.normal(0, 1, (number_of_paths, number_of_time_steps))
        paths = \
            self.initial_spot * \
            np.cumprod(np.exp((self.drift - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * z), 1)

        paths = np.insert(paths, 0, np.tile(self.initial_spot, number_of_paths), axis=1)
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
        variances = [v**2 * t for v, t in zip(vols, tenors)]
        variance_interpolator: interp1d = interp1d(tenors, variances, kind='linear', fill_value='extrapolate')

        return variance_interpolator

    @staticmethod
    def setup_bootstrapped_volatility_interpolator(excel_file_path: str, sheet_name: str) -> interp1d:
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
        vols = [v / 100 for v in vols]
        variances = [v**2 * t for v, t in zip(vols, tenors)]
        bootstrapped_variances = []
        bootstrapped_vols = []
        for i in range(0, len(variances)):
            if i == 0 or i == 1:
                b_v = vols[i]
            else:
                b_v = np.sqrt((variances[i] - bootstrapped_variances[i - 1])/(tenors[i] - tenors[i - 1]))
            bootstrapped_vols.append(b_v)

            if i == 0 or i == 1:
                bootstrapped_variances.append(variances[i])
            else:
                bootstrapped_variances.append(bootstrapped_variances[i - 1] + b_v**2 * (tenors[i] - tenors[i - 1]))

        bootstrapped_variance_interpolator: interp1d = \
            interp1d(tenors, bootstrapped_vols, kind='previous', fill_value='extrapolate')

        return bootstrapped_variance_interpolator

    def get_time_dependent_vol(self, tenor: float, use_bootstrapped_variances=True) -> float:
        """
        Gets the time-dependent volatility at the given tenor.

        :param tenor: The tenor at which we want to extract the given volatility.
        :param use_bootstrapped_variances: Toggles the use of bootstrapped variances.
        :return: The time-dependent volatility at the given tenor.
        """
        if use_bootstrapped_variances:
            return self.bootstrapped_vol_interpolator(tenor)
        else:
            return float(np.sqrt(self.variance_interpolator(tenor) / tenor)/100)

    def create_plots(self, paths: np.ndarray, time_to_maturity: float) -> None:
        """
        Plots different figures such as:

        1. The current_value of the Geometric Brownian Motion,
        2. The histogram of the log-returns, including the theoretical PDF of a normal distribution.
           This plot shows that the Geometric Brownian Motion log-returns are normally distributed.
        """
        time_steps = np.linspace(0, time_to_maturity, paths.shape[1])
        maturity_volatility: float = self.get_time_dependent_vol(time_to_maturity)
        PlotUtils.plot_monte_carlo_paths(time_steps, paths, 'Time-Dependent GBM Paths', self.drift)
        log_returns = np.log(paths[:, -1] / paths[:, 0])
        mu: float = (self.drift - 0.5 * maturity_volatility ** 2) * time_to_maturity
        sigma: float = maturity_volatility * np.sqrt(time_to_maturity)
        PlotUtils.plot_normal_histogram(
            data=log_returns,
            histogram_title='Time-Dependent GBM Log-Returns vs. Normal PDF',
            histogram_label='Log-returns histogram',
            mean=mu,
            variance=sigma)

        returns: np.ndarray = paths[:, -1] / paths[:, 0]
        mu: float = (self.drift - 0.5 * maturity_volatility ** 2) * time_to_maturity
        sigma: float = maturity_volatility * np.sqrt(time_to_maturity)
        PlotUtils.plot_lognormal_histogram(
            data=returns,
            histogram_title='Time-Dependent GBM Returns vs. Log-Normal PDF',
            histogram_label='Returns Histogram',
            mean=mu,
            variance=sigma)

    def get_path_statistics(self, paths: np.ndarray, time_to_maturity: float) -> PathStatistics:
        """
        Tests if the log-returns of the GBM paths normally distributed.

        :param paths: The GBM simulated Monte Carlo paths.
        :param time_to_maturity: Time to maturity.
        :return: None.
        """
        maturity_volatility: float = self.get_time_dependent_vol(time_to_maturity)
        log_returns: np.ndarray = np.log(paths[:, -1] / paths[:, 0])
        log_returns_mean: float = np.average(log_returns)
        log_returns_standard_deviation: float = float(np.std(log_returns))

        theoretical_mean: float = self.initial_spot * np.exp(self.drift * time_to_maturity)
        theoretical_standard_deviation: float = \
            theoretical_mean * np.sqrt((np.exp(maturity_volatility ** 2 * time_to_maturity) - 1))

        empirical_mean: float = float(np.mean(paths[:, -1]))
        empirical_standard_deviation: float = float(np.std(paths[:, -1]))
        pfe: float = \
            self.initial_spot * \
            np.exp(self.drift * time_to_maturity +
                   norm.ppf(0.95) * maturity_volatility * np.sqrt(time_to_maturity))

        print('\n')
        print(f' Time-Dependent Statistics of GBM')
        print(f' ----------------------------------')
        print(f' Statistics of Underlying')
        print(f'  Mean: {theoretical_mean}')
        print(f'  Standard Deviation: {theoretical_standard_deviation}')
        print(f'  95% PFE: {pfe}')
        print(f' Statistics of Returns')
        print(f'  Mean: {log_returns_mean}')
        print(f'  Standard Deviation: {log_returns_standard_deviation}')

        jarque_bera_test: [float, float] = jarque_bera(log_returns)
        print(f'  Jarque-Bera Test Results:')
        print(f'     p-value: {jarque_bera_test[0]}')

        if jarque_bera_test[0] > 0.05:
            print('     GBM log-returns are normally distributed.')
        else:
            print('     GBM log-returns are not normally distributed.')

        return PathStatistics(
            theoretical_mean,
            empirical_mean,
            theoretical_standard_deviation,
            empirical_standard_deviation)
