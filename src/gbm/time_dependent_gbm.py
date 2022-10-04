import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.stats import jarque_bera
from scipy.stats import lognorm
from scipy.stats import norm


class TimeDependentGBM:
    """
    Class for generating GBM paths where volatility is time-independent.
    """

    def __init__(
            self,
            drift: float,
            excel_file_path: str,
            sheet_name: str,
            initial_spot: float,
            time_to_maturity: float):
        """
        Class constructor.

        :param drift: Drift.
        :param excel_file_path: Path to Excel file containing the ATM volatility term structure.
        :param sheet_name: The sheet name of the Excel file containing the ATM volatility term structure.
        :param initial_spot: Initial spot.
        :param time_to_maturity: Time to maturity.
        """
        self.drift: float = drift
        # Here 'variance' means 'sigma**2 * time'.
        self.variance_interpolator: interp1d = self.setup_variance_interpolator(excel_file_path, sheet_name)
        self.initial_spot: float = initial_spot
        self.time_to_maturity: float = time_to_maturity

    def get_gbm_paths(
            self,
            number_of_paths: int,
            number_of_time_steps: int,
            initial_spot: float,
            time_to_maturity: float) -> np.ndarray:
        """
        Generates the GBM paths used to price various instruments.

        :param number_of_paths: Number of the current value.
        :param number_of_time_steps: Number of time steps.
        :param initial_spot: Initial spot price.
        :param time_to_maturity: Time to maturity (in years).
        :return: The simulated GBM paths.
        """
        paths: np.ndarray = np.array(np.zeros((number_of_paths, number_of_time_steps + 1)))
        paths[:, 0] = initial_spot
        dt: float = time_to_maturity / number_of_time_steps

        for j in range(1, number_of_time_steps + 1):
            z: float = norm.ppf(np.random.uniform(0, 1, number_of_paths))
            volatility: float = self.get_time_dependent_vol(j * dt)
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
        """
        Gets the time-dependent volatility at the given tenor.

        :param tenor: The tenor at which we want to extract the given volatility.
        :return: The time-dependent volatility at the given tenor.
        """
        return np.sqrt(self.variance_interpolator(tenor) / tenor) / 100

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

        maturity_volatility: float = self.get_time_dependent_vol(self.time_to_maturity)

        # Histogram of log-returns
        plt.style.use('ggplot')
        fig2, ax2 = plt.subplots(ncols=1, nrows=1)
        ax2.set_facecolor('#AAAAAA')
        ax2.grid(False)
        log_returns = np.log(paths[:, -1] / paths[:, 0])
        (values, bins, _) = ax2.hist(log_returns, bins=75, density=True, label='Histogram of samples', color='#6C3D91')
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        mu: float = (self.drift - 0.5 * maturity_volatility ** 2) * self.time_to_maturity
        sigma: float = maturity_volatility * np.sqrt(self.time_to_maturity)
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
        mu: float = (self.drift - 0.5 * maturity_volatility ** 2) * self.time_to_maturity
        sigma: float = maturity_volatility * np.sqrt(self.time_to_maturity)
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
        terminal_volatility: float = self.get_time_dependent_vol(self.time_to_maturity)
        log_returns = np.log(paths[:, -1] / paths[:, 0])
        mean: float = self.initial_spot * np.exp(self.drift + (terminal_volatility ** 2 / 2))

        standard_deviation: float = \
            self.initial_spot * \
            np.sqrt((np.exp(terminal_volatility ** 2) - 1) * np.exp((2 * self.drift + terminal_volatility ** 2)))

        pfe: float = \
            self.initial_spot * \
            np.exp(self.drift * self.time_to_maturity +
                   norm.ppf(0.95) * terminal_volatility * np.sqrt(self.time_to_maturity))

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
