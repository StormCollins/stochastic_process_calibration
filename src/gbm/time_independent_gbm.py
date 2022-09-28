import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import jarque_bera
from scipy.stats import lognorm
from scipy.stats import norm


class TimeIndependentGBM:
    """
    Class for generating GBM paths where volatility is time independent.
    """
    def __init__(self, drift: float, volatility: float, notional: float, initial_spot: float, time_to_maturity: float):
        """
        Class constructor.
        :param drift: Drift.
        :param volatility: Volatility.
        """
        self.drift: float = drift
        self.volatility: float = volatility
        self.notional: float = notional
        self.initial_spot: float = initial_spot
        self.time_to_maturity: float = time_to_maturity

    def get_paths(self, number_of_paths: int, number_of_time_steps: int) -> np.ndarray:
        """
        Generates the GBM paths used to price various instruments.

        :param number_of_paths: Number of the current value.
        :param number_of_time_steps: Number of time steps.
        :return: The simulated GBM paths.
        """
        paths: np.ndarray = np.array(np.zeros((number_of_paths, number_of_time_steps + 1)))
        paths[:, 0] = self.initial_spot * self.notional
        dt: float = self.time_to_maturity / number_of_time_steps

        for j in range(1, number_of_time_steps + 1):
            z: float = norm.ppf(np.random.uniform(0, 1, number_of_paths))
            paths[:, j] = \
                paths[:, j - 1] * \
                np.exp((self.drift - 0.5 * self.volatility ** 2) * dt + self.volatility * np.sqrt(dt) * z)

        return paths

    def create_plots(self, paths: np.ndarray) -> None:
        """
        Plots different figures such as:

        1. The current_value of the Geometric Brownian Motion,
        2. The histogram of the log-returns, including the theoretical PDF of a normal distribution.
           This plot shows that the Geometric Brownian Motion log-returns are normally distributed.
        """

        time = np.linspace(0, self.time_to_maturity, paths.shape[1])

        # Path plot
        indices_sorted_by_path_averages = np.argsort(np.average(paths, 1))
        sorted_paths = np.transpose(paths[indices_sorted_by_path_averages])
        sns.set_palette(sns.color_palette('dark:purple', paths.shape[0]))
        fig1, ax1 = plt.subplots()
        ax1.plot(time, sorted_paths)
        ax1.grid(True)
        ax1.set_facecolor('#AAAAAA')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.set_xlim([0, self.time_to_maturity])

        # Histogram of log-returns
        plt.style.use('ggplot')
        fig2, ax2 = plt.subplots(ncols=1, nrows=1)
        ax2.set_facecolor('#AAAAAA')
        ax2.grid(False)
        log_returns = np.log(paths[:, -1] / paths[:, 0])
        (values, bins, _) = ax2.hist(log_returns, bins=75, density=True, label='Histogram of samples', color='#6C3D91')
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        mu: float = (self.drift - 0.5 * self.volatility ** 2) * self.time_to_maturity
        sigma: float = self.volatility * np.sqrt(self.time_to_maturity)
        pdf = norm.pdf(x=bin_centers, loc=mu, scale=sigma)
        ax2.plot(bin_centers, pdf, label='PDF', color='black', linewidth=3)
        ax2.set_title('Comparison of GBM log-returns to normal PDF')
        ax2.legend()
        plt.show()

        # Histogram of the returns
        plt.style.use('ggplot')
        fig3, ax3 = plt.subplots(ncols=1, nrows=1)
        ax3.set_facecolor('#AAAAAA')
        ax3.grid(False)
        normal_returns = paths[:, -1] / paths[:, 0]
        (values, bins, _) = ax3.hist(normal_returns, bins=75, density=True, label='Histogram of samples', color='#6C3D91')
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        mu: float = (self.drift - 0.5 * self.volatility ** 2) * self.time_to_maturity
        sigma: float = self.volatility * np.sqrt(self.time_to_maturity)
        pdf = lognorm.pdf(x=bin_centers, s=sigma, scale=np.exp(mu))
        ax3.plot(bin_centers, pdf, label='PDF', color='black', linewidth=3)
        ax3.set_title('Comparison of GBM normal-returns to log-normal PDF')
        ax3.legend()
        plt.show()

    def get_path_statistics(self, paths: np.ndarray) -> None:
        """
        Tests if the log-returns of the GBM paths normally distributed.

        :param paths: The GBM simulated Monte Carlo paths.
        :return: None.
        """
        log_returns = np.log(paths[:, -1] / paths[:, 0])
        mean: float = self.initial_spot * np.exp(self.drift + (self.volatility ** 2 / 2))

        standard_deviation: float = \
            self.initial_spot * \
            np.sqrt((np.exp(self.volatility ** 2) - 1) * np.exp((2 * self.drift + self.volatility ** 2)))

        pfe: float = \
            self.initial_spot * \
            np.exp(self.drift * self.time_to_maturity +
                   norm.ppf(0.95) * self.volatility * np.sqrt(self.time_to_maturity))

        print('\n')
        print(f' Time-Independent Statistics of GBM')
        print(f' ----------------------------------')
        print(f'  Mean: {mean}')
        print(f'  Standard Deviation: {standard_deviation}')
        print(f'  95% PFE: {pfe}')
        jarque_bera_test: [float, float] = jarque_bera(log_returns)
        print(f'  Jarque-Bera Test Results:')
        print(f'     p-value: {jarque_bera_test[0]}')

        if jarque_bera_test[0] > 0.05:
            print('     GBM log-returns are normally distributed.')
        else:
            print('     GBM log-returns are not normally distributed.')
