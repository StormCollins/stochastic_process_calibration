import math
import numpy as np
from scipy.stats import norm
import time


# Code up Black-Scholes
def black_scholes(
        initial_spot: float,
        strike: float,
        interest_rate: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: str) -> float | str:
    """
    Returns the standard Black-Scholes price for a 'CALL' or 'PUT' option (does not take into account whether you
    are 'long' or 'short' the option.

    :param initial_spot: The initial spot price of the option.
    :param strike: The option strike price.
    :param interest_rate: The interest rate/drift used for the option.
    :param volatility: The option volatility.
    :param time_to_maturity: The time (in years) at which the option expires.
    :param call_or_put: Indicates whether the option is a 'CALL' or a 'PUT'
    :return: Black-Scholes price for an option.
    """
    d_1: float = (np.log(initial_spot / strike) + ((interest_rate + 0.5 * volatility ** 2) * time_to_maturity)) / \
                 (volatility * math.sqrt(time_to_maturity))
    d_2: float = d_1 - volatility * math.sqrt(time_to_maturity)

    if str.upper(call_or_put) == 'CALL':
        return initial_spot * norm.cdf(d_1) - strike * math.exp(-interest_rate * time_to_maturity) * norm.cdf(d_2)
    elif str.upper(call_or_put) == 'PUT':
        return - initial_spot * norm.cdf(-d_1) + strike * math.exp(-interest_rate * time_to_maturity) * norm.cdf(-d_2)
    else:
        return f'Unknown option type: {call_or_put}'


# Code up Monte Carlo for option.
def slow_european_option_monte_carlo_pricer(
        initial_spot: float,
        strike: float,
        interest_rate: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: str,
        number_of_paths: int,
        number_of_time_steps: int) -> [float | str, time]:
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

    payoffs: np.ndarray = np.zeros(number_of_paths)
    for i in range(0, number_of_paths):
        payoffs[i] = np.max([paths[i, -1] - strike, 0]) * math.exp(-interest_rate * time_to_maturity)

    price: float = np.average(payoffs)
    end_time: time = time.time()
    return price, end_time - start_time


def fast_european_option_monte_carlo_pricer(
        initial_spot: float,
        strike: float,
        interest_rate: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: str,
        number_of_paths: int,
        number_of_time_steps: int) -> [float | str, time]:
    """
    This function uses 'vectorisation' unlike the slow_european_option_monte_carlo_pricer function thus speeding up
    performance.

    :param initial_spot:
    :param strike:
    :param interest_rate:
    :param volatility:
    :param time_to_maturity:
    :param call_or_put:
    :param number_of_paths:
    :param number_of_time_steps:
    :return:
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

    payoffs = np.maximum(paths[:, -1] - strike, 0) * np.exp(-interest_rate * time_to_maturity)

    price: float = np.average(payoffs)
    end_time: time = time.time()
    return price, end_time - start_time


initial_spot: float = 50
strike: float = 52
interest_rate: float = 0.1
volatility: float = 0.4
time_to_maturity: float = 5 / 12
number_of_paths: int = 100_000
number_of_time_steps: int = 2

# Compare Monte Carlo to Black-Scholes
slow_price, slow_time = slow_european_option_monte_carlo_pricer(initial_spot, strike, interest_rate, volatility, time_to_maturity, "call", number_of_paths, number_of_time_steps)
fast_price, fast_time = fast_european_option_monte_carlo_pricer(initial_spot, strike, interest_rate, volatility, time_to_maturity, "call", number_of_paths, number_of_time_steps)
print(f'Black-Scholes Price: {black_scholes(initial_spot, strike, interest_rate, volatility, time_to_maturity, "call")}')
print(f'Slow European Option Price: {slow_price} (time taken {slow_time})')
print(f'Fast European Option Price: {fast_price} (time taken {fast_time})')

print(f'\nSlow Monte Carlo / Fast Monte Carlo: {slow_time / fast_time}')

