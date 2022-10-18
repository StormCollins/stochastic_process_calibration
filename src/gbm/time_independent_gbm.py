"""
Contains a class for generating GBM paths where volatility is time-independent.
"""
import numpy as np
from scipy.stats import jarque_bera
from scipy.stats import norm
from src.enums_and_named_tuples.path_statistics import PathStatistics
from src.utils.plot_utils import PlotUtils


class TimeIndependentGBM:
    """
    Class for generating GBM paths where volatility is time-independent.
    """

    def __init__(self, drift: float, volatility: float, initial_spot: float):
        """
        Class constructor.

        :param drift: Drift.
        :param volatility: Volatility.
        :param initial_spot: Initial spot.
        """
        self.drift: float = drift
        self.volatility: float = volatility
        self.initial_spot: float = initial_spot

    def get_paths(self, number_of_paths: int, number_of_time_steps: int, time_to_maturity: float) -> np.ndarray:
        """
        Generates the GBM paths used to price various instruments.

        :param number_of_paths: Number of the current value.
        :param number_of_time_steps: Number of time steps.
        :param time_to_maturity: Time to maturity.
        :return: The simulated GBM paths.
        """
        paths: np.ndarray = np.array(np.zeros((number_of_paths, number_of_time_steps + 1)))
        paths[:, 0] = self.initial_spot
        dt: float = time_to_maturity / number_of_time_steps
        z = np.random.normal(0, 1, (number_of_paths, number_of_time_steps))

        paths = \
            self.initial_spot * \
            np.cumprod(np.exp((self.drift - 0.5 * self.volatility ** 2) * dt + self.volatility * np.sqrt(dt) * z), 1)

        paths = np.insert(paths, 0, np.tile(self.initial_spot, number_of_paths), axis=1)
        return paths

    def create_plots(self, paths: np.ndarray, time_to_maturity: float) -> None:
        """
        Plots different figures such as:

        1. The current_value of the Geometric Brownian Motion,
        2. The histogram of the log-returns, including the theoretical PDF of a normal distribution.
           This plot shows that the Geometric Brownian Motion log-returns are normally distributed.
        """
        time_steps = np.linspace(0, time_to_maturity, paths.shape[1])
        PlotUtils.plot_monte_carlo_paths(time_steps, paths, 'Time-Independent GBM Paths', self.drift)

        log_returns = np.log(paths[:, -1] / paths[:, 0])
        mu: float = (self.drift - 0.5 * self.volatility ** 2) * time_to_maturity
        sigma: float = self.volatility * np.sqrt(time_to_maturity)
        PlotUtils.plot_normal_histogram(
            data=log_returns,
            histogram_title='Time-Independent GBM Log-Returns vs. Normal PDF',
            histogram_label='Log-returns histogram',
            mean=mu,
            variance=sigma)

        returns: np.ndarray = paths[:, -1] / paths[:, 0]
        mu: float = (self.drift - 0.5 * self.volatility ** 2) * time_to_maturity
        sigma: float = self.volatility * np.sqrt(time_to_maturity)
        PlotUtils.plot_lognormal_histogram(
            data=returns,
            histogram_title='Time-Independent GBM Returns vs. Log-Normal PDF',
            histogram_label='Returns Histogram',
            mean=mu,
            variance=sigma)

    def get_path_statistics(self, paths: np.ndarray, time_to_maturity: float) -> PathStatistics:  # dict[str, float]:
        """
        Tests if the log-returns of the GBM paths normally distributed.

        :param paths: The GBM simulated Monte Carlo paths.
        :param time_to_maturity: Time to maturity.
        :return: None.
        """
        log_returns: np.ndarray = np.log(paths[:, -1] / paths[:, 0])
        theoretical_mean: float = self.initial_spot * np.exp(self.drift * time_to_maturity)
        theoretical_standard_deviation: float = \
            theoretical_mean * np.sqrt((np.exp(self.volatility ** 2 * time_to_maturity) - 1))

        empirical_mean: float = float(np.mean(paths[:, -1]))
        empirical_standard_deviation: float = float(np.std(paths[:, -1]))
        pfe: float = \
            self.initial_spot * \
            np.exp(self.drift * time_to_maturity +
                   norm.ppf(0.95) * self.volatility * np.sqrt(time_to_maturity))

        print('\n')
        print(f' Time-Independent Statistics of GBM')
        print(f' ----------------------------------')
        print(f'  Mean: {theoretical_mean}')
        print(f'  Standard Deviation: {theoretical_standard_deviation}')
        print(f'  95% PFE: {pfe}')
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
