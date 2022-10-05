import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class PlotUtils:
    @staticmethod
    def plot_histogram(
            variable_to_plot: np.ndarray,
            histogram_title: str,
            histogram_label: str):
        """
        Plots the histogram of the returns, including the theoretical PDF of a normal distribution.

        :param variable_to_plot: Specifies what will be plotted. For example, the log-returns or the normal returns.
        :param histogram_title: Indicates the title of the histogram based on what is being plotted.
        :param histogram_label: Indicates the labels of the histogram (legend) based on what is being plotted.
        :return: The histogram of the returns, including the theoretical PDF of a normal distribution.
        """
        plt.style.use('ggplot')
        fig2, ax2 = plt.subplots(ncols=1, nrows=1)
        ax2.set_facecolor('#AAAAAA')
        ax2.grid(False)
        (values, bins, _) = ax2.hist(variable_to_plot, bins=75, density=True, label=histogram_label, color='#6C3D91')
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        # mu: float = (self.drift - 0.5 * self.volatility ** 2) * self.time_to_maturity
        # sigma: float = self.volatility * np.sqrt(self.time_to_maturity)
        pdf = norm.pdf(x=bin_centers, loc=mu, scale=sigma)
        ax2.plot(bin_centers, pdf, label='PDF', color='black', linewidth=3)
        ax2.set_title(histogram_title)
        ax2.legend()
        plt.show()
