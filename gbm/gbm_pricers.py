import math
import numpy as np
from scipy.stats import norm
import time
import matplotlib as cm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D


#TODO: Rename to slow_equity_european_option_monte_carlo_pricer
#TODO: Put a description in the comments.
def slow_european_option_monte_carlo_pricer(
        initial_spot: float,
        strike: float,
        interest_rate: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: str,
        number_of_paths: int,
        number_of_time_steps: int) -> [float | str, time]:
    """

    :param initial_spot: The initial spot price for the option.
    :param strike: The strike price for the option.
    :param interest_rate:  The interest rate/drift used for the option.
    :param volatility: The option volatility.
    :param time_to_maturity: The time (in years) at which the option expires.
    :param call_or_put: Indicates whether the option is a 'CALL' or a 'PUT'.
    :param number_of_paths: Number of paths to simulate for the option.
    :param number_of_time_steps: Number of time steps for the option.
    :return: Slow Monte Carlo price for an option.
    """
    start_time: time = time.time()

    paths: np.ndarray = np.array(np.zeros((number_of_paths, number_of_time_steps)))

    for i in range(0, number_of_paths):
        paths[i, 0] = initial_spot

    dt: float = time_to_maturity / (number_of_time_steps - 1)

    # Actual Monte Carlo
    for j in range(1, number_of_time_steps):
        for i in range(0, number_of_paths):
            z: float = norm.ppf(np.random.uniform(0, 1))
            paths[i, j] = paths[i, j - 1] * math.exp(
                (interest_rate - 0.5 * volatility ** 2) * dt + volatility * math.sqrt(dt) * z)

    if str.upper(call_or_put) == 'CALL':
        payoffs: np.ndarray = np.zeros(number_of_paths)
        for i in range(0, number_of_paths):
            payoffs[i] = np.max([paths[i, -1] - strike, 0]) * math.exp(-interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        end_time: time = time.time()
        return price, end_time - start_time

    elif str.upper(call_or_put) == 'PUT':
        payoffs: np.ndarray = np.zeros(number_of_paths)
        for i in range(0, number_of_paths):
            payoffs[i] = np.max([strike - paths[i, -1], 0]) * math.exp(-interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        end_time: time = time.time()
        return price, end_time - start_time

    else:
        return f'Unknown option type: {call_or_put}'


#TODO: Rename to fast_equity_european_option_monte_carlo_pricer
def fast_european_option_monte_carlo_pricer(
        initial_spot: float,
        strike: float,
        interest_rate: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: str,
        number_of_paths: int,
        number_of_time_steps: int,
        plot_paths: bool = False) -> [float | str, time]:
    """
    This function uses 'vectorisation' unlike the slow_european_option_monte_carlo_pricer function thus speeding up
    performance.

    :param initial_spot:The initial spot price of the option.
    :param strike: The option strike price.
    :param interest_rate: The interest rate/drift used for the option.
    :param volatility: The option volatility.
    :param time_to_maturity: The time (in years) at which the option expires.
    :param call_or_put: Indicates whether the option is a 'CALL' or a 'PUT'.
    :param number_of_paths: Number of paths to simulate for the option.
    :param number_of_time_steps: Number of time steps for the option.
    :param plot_paths: If set to True plots the paths.
    :return: Fast Monte Carlo price for an option.
    """
    start_time: time = time.time()
    paths: np.ndarray = np.array(np.zeros((number_of_paths, number_of_time_steps)))

    paths[:, 0] = initial_spot

    dt: float = time_to_maturity / (number_of_time_steps - 1)

    # Actual Monte Carlo
    for j in range(1, number_of_time_steps):
        z: float = norm.ppf(np.random.uniform(0, 1, number_of_paths))
        paths[:, j] = paths[:, j - 1] * np.exp(
            (interest_rate - 0.5 * volatility ** 2) * dt + volatility * math.sqrt(dt) * z)

    # Plot the paths
    if plot_paths:
        plt.plot(np.transpose(paths))
        plt.grid(True)
        plt.xlabel('Number of time steps')
        plt.ylabel('Number of paths')

    if str.upper(call_or_put) == 'CALL':
        payoffs = np.maximum(paths[:, -1] - strike, 0) * np.exp(-interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        end_time: time = time.time()
        return price, end_time - start_time

    elif str.upper(call_or_put) == 'PUT':
        payoffs = np.maximum(strike - paths[:, -1], 0) * np.exp(-interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        end_time: time = time.time()
        return price, end_time - start_time

    else:
        return f'Unknown option type: {call_or_put}'


# TODO: Make the name all lower case
# TODO: Add description in comments below
# TODO: Rename 'rd' and 'rf'
def FX_option_monte_carlo_pricer(
        initial_spot: float,
        strike: float,
        rd: float,
        rf: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: str,
        number_of_paths: int,
        number_of_time_steps: int,
        plot_paths: bool = False) -> [float | str]:

    """


    :param initial_spot: Initial spot price for the FX option.
    :param strike: Strike price for the FX option.
    :param rd: Domestic interest rate.
    :param rf: Foreign interest rate.
    :param volatility: Volatility of the FX rate.
    :param time_to_maturity: Time to maturity (in years) of the FX option.
    :param call_or_put: Indicates whether the option is a 'CALL' or a 'PUT'.
    :param number_of_paths: Number of paths for the FX option.
    :param number_of_time_steps: Number of time steps for the FX option.
    :param plot_paths: If set to True plots the paths.
    :return: Monte Carlo price for an FX Option.
    """

    paths: np.ndarray = np.array(np.zeros((number_of_paths, number_of_time_steps)))

    paths[:, 0] = initial_spot

    dt: float = time_to_maturity / (number_of_time_steps - 1)

    # Actual Monte Carlo for the FX option
    for j in range(1, number_of_time_steps):
        z: float = norm.ppf(np.random.uniform(0, 1, number_of_paths))
        paths[:, j] = paths[:, j - 1] * np.exp(
            (rd - rf - 0.5 * volatility ** 2) * dt + volatility * math.sqrt(dt) * z)

    # Plot the FX paths
    if plot_paths:
        plt.plot(np.transpose(paths))
        plt.grid(True)
        plt.xlabel('Number of time steps')
        plt.ylabel('Number of paths')

    if str.upper(call_or_put) == 'CALL':
        payoffs = np.maximum(paths[:, -1] - strike, 0) * np.exp(-rd * time_to_maturity)
        price: float = np.average(payoffs)
        return price

    elif str.upper(call_or_put) == 'PUT':
        payoffs = np.maximum(strike - paths[:, -1], 0) * np.exp(-rd * time_to_maturity)
        price: float = np.average(payoffs)
        return price

    else:
        return f'Unknown option type: {call_or_put}'



#
# #Create the volatility surface
# def plot3D(X, Y, Z):
#     fig = plt.figure()
#     ax = Axes3D(fig, azim=-29, elev=50)
#
#     ax.plot(X, Y, Z, 'o')
#
#     plt.xlabel("expiry")
#     plt.ylabel("strike")
#     plt.show()
#
#     return plot3D(X,Y,Z)