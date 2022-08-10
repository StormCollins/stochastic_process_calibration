import math
import numpy as np
from scipy.stats import norm
import matplotlib as cm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple

MonteCarloResult = namedtuple('MonteCarloResult', ['price', 'error', 'mean', 'standard_deviation', 'percentile'])


def fx_forward_monte_carlo_pricer(
        initial_spot: float,
        strike: float,
        domestic_interest_rate: float,
        foreign_interest_rate: float,
        volatility: float,
        time_to_maturity: float,
        number_of_paths: int,
        number_of_time_steps: int,
        plot_paths: bool = True) -> [MonteCarloResult | str]:
    """
    Returns the price for an FX forward using monte carlo simulations.

    :param initial_spot: Initial spot price for the FX option.
    :param strike: Strike price for the FX option.
    :param domestic_interest_rate: Domestic interest rate.
    :param foreign_interest_rate: Foreign interest rate.
    :param volatility: Volatility of the FX rate.
    :param time_to_maturity: Time to maturity (in years) of the FX option.
    :param number_of_paths: Number of paths for the FX option.
    :param number_of_time_steps: Number of time steps for the FX option.
    :param plot_paths: If set to True plots the paths.
    :return: Monte Carlo price for an FX forward.
    """

    paths: np.ndarray = np.array(np.zeros((number_of_paths, number_of_time_steps)))

    paths[:, 0] = initial_spot

    dt: float = time_to_maturity / (number_of_time_steps - 1)

    for j in range(1, number_of_time_steps):
        z: float = norm.ppf(np.random.uniform(0, 1, number_of_paths))
        paths[:, j] = \
            paths[:, j - 1] * \
            np.exp(
                (domestic_interest_rate - foreign_interest_rate - 0.5 * volatility ** 2) * dt +
                volatility * math.sqrt(dt) * z)

    if plot_paths:
        plt.plot(np.transpose(paths))
        plt.grid(True)
        plt.xlabel('Number of time steps')
        plt.ylabel('Number of paths')

    payoffs = (paths[:, -1] - strike) * np.exp(-domestic_interest_rate * time_to_maturity)
    price: float = np.average(payoffs)
    mu = np.mean(payoffs)
    sigma = np.std(payoffs)
    mean: float = math.exp(mu + (sigma ** 2 / 2))
    standard_deviation: float = np.sqrt((math.exp(sigma ** 2) - 1 ) * math.exp((
                          2 * mu + sigma ** 2)))
    percentile: float = initial_spot * math.exp(
        mu * time_to_maturity + norm.ppf(0.95) * volatility * math.sqrt(time_to_maturity))
    error = norm.ppf(0.95) * sigma / np.sqrt(number_of_paths)
    return MonteCarloResult(price, error, mean, standard_deviation, percentile)


initial_spot: float = 14.6038
strike: float = 17.2531
domestic_interest_rate: float = 0.061339421
foreign_interest_rate: float = 0.020564138
volatility: float = 15.4
time_to_maturity: float = 1
number_of_paths: int = 100_000
number_of_time_steps: int = 50

print(f'Monte Carlo FX Forward Price: ' +
      str(fx_forward_monte_carlo_pricer(
          initial_spot,
          strike,
          domestic_interest_rate,
          foreign_interest_rate,
          volatility,
          time_to_maturity,
          number_of_paths,
          number_of_time_steps)))
