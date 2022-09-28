import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import lognorm
from collections import namedtuple
from src.gbm.gbm_simulation import GBM
from src.call_or_put import CallOrPut


MonteCarloResult = namedtuple('MonteCarloResult', ['price', 'error'])
VolData = namedtuple('VolData', ['Tenors', 'VolSurface'])




def create_gbm_plots(paths, interest_rate: float, volatility: float, time_to_maturity: float) -> None:
    """

    This function plots different figures such as:
    1. The current_value of the Geometric Brownian Motion,
    2. The histogram of the log-returns, including the theoretical PDF of a normal distribution.
       This plot shows that the Geometric Brownian Motion log-returns are normally distributed.
    """

    time = np.linspace(0, time_to_maturity, paths.shape[1])

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
    ax1.set_xlim([0, time_to_maturity])

    # Histogram of log-returns
    plt.style.use('ggplot')
    fig2, ax2 = plt.subplots(ncols=1, nrows=1)
    ax2.set_facecolor('#AAAAAA')
    ax2.grid(False)
    log_returns = np.log(paths[:, -1] / paths[:, 0])
    (values, bins, _) = ax2.hist(log_returns, bins=75, density=True, label='Histogram of samples', color='#6C3D91')
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    mu: float = (interest_rate - 0.5 * volatility ** 2) * time_to_maturity
    sigma: float = volatility * np.sqrt(time_to_maturity)
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
    mu: float = (interest_rate - 0.5 * volatility ** 2) * time_to_maturity
    sigma: float = volatility * np.sqrt(time_to_maturity)
    pdf = lognorm.pdf(x=bin_centers, s=sigma, scale=np.exp(mu))
    ax3.plot(bin_centers, pdf, label='PDF', color='black', linewidth=3)
    ax3.set_title('Comparison of GBM normal-returns to log-normal PDF')
    ax3.legend()
    plt.show()


