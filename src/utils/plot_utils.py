"""
This module is contains a single class, PlotUtils, with static methods for plotting data.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import lognorm
from scipy.stats import norm


class PlotUtils:
    """
    A collection of static methods for plotting data such as histograms and
    """
    @staticmethod
    def plot_normal_histogram(
            data: np.ndarray,
            histogram_title: str,
            histogram_label: str,
            mean: float,
            variance: float) -> None:
        """
        Plots a histogram of the given data as well as overlaying a PDF of a normal distribution for the given mean
        and volatility.

        :param data: Data for which to plot a histogram.
        :param histogram_title: Histogram title.
        :param histogram_label: Histogram label.
        :param mean: The mean of the normal distribution PDF.
        :param variance: The volatility of the normal distribution PDF.
        :return: None.
        """
        plt.style.use('ggplot')
        plt.rcParams['font.family'] = 'calibri'
        fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.set_facecolor('white')
        ax.grid(False)
        (values, bins, _) = \
            ax.hist(data.flatten(), bins=75, density=True, label=histogram_label, color='#86BC25')

        ax.annotate(
            f'{len(data):,} Sims',
            xy=(0.65, 0.5),
            xycoords='axes fraction',
            bbox=dict(boxstyle='round,pad=0.3', fc='#E3E48D', lw=0))

        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        pdf = norm.pdf(x=bin_centers, loc=mean, scale=variance)
        ax.plot(bin_centers, pdf, label='PDF', color='#00A3E0', linewidth=2)
        ax.set_title(histogram_title)
        ax.legend()
        plt.show()

    @staticmethod
    def plot_lognormal_histogram(
            data: np.ndarray,
            histogram_title: str,
            histogram_label: str,
            mean: float,
            variance: float) -> None:
        """
        Plots a histogram of the given data as well as overlaying a PDF of a log-normal distribution for the given mean
        and volatility .

        :param data: Data for which to plot a histogram.
        :param histogram_title: Histogram title.
        :param histogram_label: Histogram label.
        :param mean: The mean of the log-normal distribution PDF.
        :param variance: The volatility of the log-normal distribution PDF.
        :return: None.
        """
        plt.style.use('ggplot')
        plt.rcParams['font.family'] = 'calibri'
        fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.set_facecolor('white')
        ax.grid(False)

        (values, bins, _) = \
            ax.hist(data.flatten(), bins=75, density=True, label=histogram_label, color='#86BC25')

        ax.annotate(
            f'{len(data):,} Sims',
            xy=(0.6, 0.5),
            xycoords='axes fraction',
            bbox=dict(boxstyle='round,pad=0.3', fc='#E3E48D', lw=0))

        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        pdf = lognorm.pdf(x=bin_centers, s=variance, scale=np.exp(mean))
        ax.plot(bin_centers, pdf, label='PDF', color='#00A3E0', linewidth=2)
        ax.set_title(histogram_title)
        ax.legend()
        plt.show()

    @staticmethod
    def plot_monte_carlo_paths(time_steps: np.ndarray, paths: np.ndarray, title: str):
        indices_sorted_by_path_averages = np.argsort(np.average(paths, 1))
        sorted_paths = np.transpose(paths[indices_sorted_by_path_averages])
        sns.set_palette(sns.dark_palette('#86BC25', n_colors=paths.shape[0], as_cmap=False))
        fig, ax = plt.subplots()
        plt.style.use(['ggplot', 'fast'])
        plt.rcParams['path.simplify_threshold'] = 1.0
        ax.plot(time_steps, sorted_paths)
        ax.grid(True)
        ax.set_facecolor('white')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_xlim([0, time_steps[-1]])
        ax.annotate(
            f'{paths.shape[0]:,} Sims\n'
            f'{paths.shape[1] - 1:,} Time Steps',
            xy=(0.05, 0.85),
            xycoords='axes fraction',
            bbox=dict(boxstyle='round,pad=0.3', fc='#E3E48D', lw=0))

        ax.set_title(title)
        plt.show(block=False)
