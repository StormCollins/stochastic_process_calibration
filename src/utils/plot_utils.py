"""
Contains a class with static methods for plotting data.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import lognorm
from scipy.stats import norm


colors_green: str = '#86BC25'
colors_bright_green: str = '#C4D600'
colors_light_green: str = '#E3E48D'
colors_teal: str = '#00A3E0'


class PlotUtils:
    """
    A collection of static methods for plotting data such as histograms and Monte Carlo paths.
    """

    @staticmethod
    def plot_normal_histogram(
            data: np.ndarray,
            histogram_title: str,
            histogram_label: str,
            mean: float,
            variance: float,
            additional_annotation: str = None) -> None:
        """
        Plots a histogram of the given data as well as overlaying a PDF of a normal distribution for the given mean
        and volatility.

        :param data: Data for which to plot a histogram.
        :param histogram_title: Histogram title.
        :param histogram_label: Histogram label.
        :param mean: The mean of the normal distribution PDF.
        :param variance: The volatility of the normal distribution PDF.
        :param additional_annotation: Any additional annotation to add to the plot. Default = None.
        :return: None.
        """
        plt.style.use(['ggplot', 'fast'])
        plt.rcParams['font.family'] = 'calibri'
        fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.set_facecolor('white')
        ax.grid(False)
        (values, bins, _) = \
            ax.hist(data.flatten(), bins=75, density=True, label=histogram_label, color=colors_green)

        ax.annotate(
            f'{len(data):,} Sims',
            fontsize=8,
            xy=(0.75, 0.8),
            xycoords='axes fraction',
            bbox=dict(boxstyle='round,pad=0.3', fc=colors_light_green, lw=0))

        if additional_annotation is not None:
            ax.annotate(
                f'{additional_annotation}',
                fontsize=8,
                xy=(0.05, 0.05),
                xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.3', fc=colors_light_green, lw=0))

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.5)
            ax.spines[axis].set_color('black')

        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        pdf = norm.pdf(x=bin_centers, loc=mean, scale=variance)
        ax.plot(bin_centers, pdf, label='Normal PDF', color=colors_teal, linewidth=2)
        ax.set_title(histogram_title)
        ax.set_xlim([bins[0], bins[-1]])
        ax.set_xlabel('Return Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.show()

    @staticmethod
    def plot_lognormal_histogram(
            data: np.ndarray,
            histogram_title: str,
            histogram_label: str,
            mean: float,
            variance: float,
            additional_annotation: str = None) -> None:
        """
        Plots a histogram of the given data as well as overlaying a PDF of a log-normal distribution for the given mean
        and volatility .

        :param data: Data for which to plot a histogram.
        :param histogram_title: Histogram title.
        :param histogram_label: Histogram label.
        :param mean: The mean of the log-normal distribution PDF.
        :param variance: The volatility of the log-normal distribution PDF.
        :param additional_annotation: Any additional annotation to add to the plot. Default = None.
        :return: None.
        """
        plt.style.use(['ggplot', 'fast'])
        plt.rcParams['font.family'] = 'calibri'
        fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.set_facecolor('white')
        ax.grid(True)
        (values, bins, _) = \
            ax.hist(data.flatten(), bins=75, density=True, label=histogram_label, color=colors_green)

        ax.annotate(
            f'{len(data):,} Sims',
            fontsize=8,
            xy=(0.75, 0.8),
            xycoords='axes fraction',
            bbox=dict(boxstyle='round,pad=0.3', fc=colors_light_green, lw=0))

        if additional_annotation is not None:
            ax.annotate(
                f'{additional_annotation}',
                fontsize=8,
                xy=(0.05, 0.05),
                xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.3', fc=colors_light_green, lw=0))

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.5)
            ax.spines[axis].set_color('black')

        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        pdf = lognorm.pdf(x=bin_centers, s=variance, scale=np.exp(mean))
        ax.plot(bin_centers, pdf, label='Log-normal PDF', color=colors_teal, linewidth=2)
        ax.set_xlim([bins[0], bins[-1]])
        ax.set_xlabel('Log-Return Value')
        ax.set_ylabel('Frequency')
        ax.set_title(histogram_title)
        ax.legend()
        plt.show()

    @staticmethod
    def plot_monte_carlo_paths(
            time_steps: np.ndarray,
            paths: np.ndarray,
            title: str,
            drift: float = None,
            additional_annotation: str = None) -> None:
        """
        Used to plot Monte Carlo paths.

        :param time_steps: The time steps of the Monte Carlo simulation.
        :param paths: The paths simulated during the Monte Carlo run.
        :param title: The title of the plot.
        :param drift: The (constant) drift, if applicable, of the process. This just corresponds to the standard drift
        in GBM. Currently, Hull-White is not supported.
        :param additional_annotation: Any additional annotation to add to the plot. Default = None.
        :return:
        """
        plt.style.use(['ggplot', 'fast'])
        plt.rcParams['font.family'] = 'calibri'
        plt.rcParams['path.simplify_threshold'] = 1.0
        indices_sorted_by_path_averages = np.argsort(np.average(paths, 1))
        sorted_paths = np.transpose(paths[indices_sorted_by_path_averages])
        sns.set_palette(sns.dark_palette(colors_green, n_colors=paths.shape[0], as_cmap=False))
        fig, ax = plt.subplots()
        ax.plot(time_steps, sorted_paths)
        empirical_path_means: np.ndarray = np.mean(paths, 0)
        initial_spot: float = paths[0, 0]

        if drift is not None:
            theoretical_path_means = initial_spot * np.exp(drift * time_steps)
            ax.plot(
                time_steps,
                theoretical_path_means,
                label='Theoretical Path Average',
                linestyle='solid',
                linewidth='3',
                color=colors_bright_green)

        ax.plot(
            time_steps,
            empirical_path_means,
            label='Empirical Path Average',
            linestyle='dashed',
            linewidth='1',
            color=colors_teal)

        ax.grid(False)
        ax.set_facecolor('white')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_xlim([0, time_steps[-1]])
        ax.annotate(
            f'{paths.shape[0]:,} Sims\n'
            f'{paths.shape[1] - 1:,} Time Steps',
            fontsize=8,
            xy=(0.05, 0.75),
            xycoords='axes fraction',
            bbox=dict(boxstyle='round,pad=0.3', fc=colors_light_green, lw=0))

        if additional_annotation is not None:
            ax.annotate(
                f'{additional_annotation}',
                fontsize=8,
                xy=(0.05, 0.05),
                xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.3', fc=colors_light_green, lw=0))

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.5)
            ax.spines[axis].set_color('black')

        ax.legend()
        ax.set_title(title)
        plt.show(block=False)

    @staticmethod
    def plot_curves(
            title: str,
            time_steps: np.ndarray,
            curves: list[tuple[str, np.ndarray]],
            additional_annotation: str = None) -> None:
        """
        Plots curves that have been passed as a tuple of ['legend label', 'data'] and which have the same time_steps.

        :param title: Title of the plot.
        :param time_steps: Time steps.
        :param curves: The curves to plot consisting of tuples of the form ['legend label', 'data'].
        :param additional_annotation: Any additional annotation to add to the plot. Default = None.
        :return:
        """
        plt.style.use(['ggplot', 'fast'])
        plt.rcParams['font.family'] = 'calibri'
        plt.rcParams['path.simplify_threshold'] = 1.0
        fig, ax = plt.subplots()
        ax.set_facecolor('white')
        ax.set_title(title)
        ax.grid(False)
        curve_colors: list[str] = [colors_teal, colors_green]
        y_min: float = np.min(curves[0][1])
        y_max: float = np.max(curves[0][1])
        for i, curve in enumerate(curves):
            ax.plot(time_steps, curve[1], color=curve_colors[i], label=curve[0])
            y_min = np.min([y_min, np.min(curve[1])])
            y_max = np.max([y_max, np.max(curve[1])])

        if additional_annotation is not None:
            ax.annotate(
                f'{additional_annotation}',
                fontsize=8,
                xy=(0.05, 0.05),
                xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.3', fc=colors_light_green, lw=0))

        ax.set_xlabel('$t$ (years)')
        ax.set_ylabel('$P(0, t)$')
        ax.set_xlim([0, time_steps[-1]])
        ax.set_ylim([y_min, y_max])
        ax.legend()

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.5)
            ax.spines[axis].set_color('black')

        plt.show()
