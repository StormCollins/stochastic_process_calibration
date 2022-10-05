import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm
from scipy.stats import norm


class PlotUtils:
    @staticmethod
    def plot_normal_histogram(
            data: np.ndarray,
            histogram_title: str,
            histogram_label: str,
            mean: float,
            variance: float) -> None:
        """
        Plots a histogram of the given data as well as overlaying a PDF of a normal distribution for the given mean (mu)
        and volatility (sigma).

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
