import math
import numpy as np
from scipy.stats import norm
import time
import matplotlib as plt
import matplotlib.pyplot as plt


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

    :param initial_spot:The initial spot price of the option.
    :param strike: The option strike price.
    :param interest_rate: The interest rate/drift used for the option.
    :param volatility: The option volatility.
    :param time_to_maturity: The time (in years) at which the option expires.
    :param call_or_put: Indicates whether the option is a 'CALL' or a 'PUT'.
    :param number_of_paths: Number of paths to simulate for the option.
    :param number_of_time_steps: Number of time steps for the option.
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

    #Plot the paths

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


# TEST THE FUNCTIONS

initial_spot: float = 50
strike: float = 52
interest_rate: float = 0.1
volatility: float = 0.4
time_to_maturity: float = 5 / 12
number_of_paths: int = 10000
number_of_time_steps: int = 2

# Compare Monte Carlo to Black-Scholes
slow_price, slow_time = slow_european_option_monte_carlo_pricer(initial_spot, strike, interest_rate, volatility,
                                                                time_to_maturity, "put", number_of_paths,
                                                                number_of_time_steps)
fast_price, fast_time = fast_european_option_monte_carlo_pricer(initial_spot, strike, interest_rate, volatility,
                                                                time_to_maturity, "put", number_of_paths,
                                                                number_of_time_steps)
print(f'Black-Scholes Price: {black_scholes(initial_spot, strike, interest_rate, volatility, time_to_maturity, "put")}')
print(f'Slow European Option Price: {slow_price} (time taken {slow_time})')
print(f'Fast European Option Price: {fast_price} (time taken {fast_time})')

# print(f'\nSlow Monte Carlo / Fast Monte Carlo: {slow_time / fast_time}')

# FX Forward
#why is it complaining?
def FX_Forward(
        initial_spot: float,
        rd: float,
        rf: float,
        time_of_contract: float) -> float:
    """
    :param initial_spot: The initial forward spot price.
    :param rd: The domestic currency interest rate.
    :param rf: The foreign currency interest rate.
    :param time_of_contract: Time of the contract in years.
    :return: The FX Forward price of the forward.
    """
    return initial_spot * math.exp((rd - rf) * time_of_contract)

initial_fx: float = 50
rd: float = 0.1
rf: float = 0
time_of_contract: float = 5/12

print(f'FX Forward Price: {FX_Forward(initial_spot, rd, rf, time_of_contract)}')

#Analytical Pricer for FX forward

#Black Scholes Pricer for FX option [Garman-Kohlhagen]

# Code up Black-Scholes
def black_scholes_FX(
        initial_spot: float,
        strike: float,
        rd: float,
        rf: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: str) -> float | str:

    """

    :param initial_spot: Initial spot rate of the FX option.
    :param strike: Strike price of the FX option.
    :param rd: Domestic interest rate.
    :param rf: Foreign interest rate.
    :param volatility: Volatility of the FX rate.
    :param time_to_maturity: Time to maturity (in years) of the FX option.
    :param call_or_put: Indicates whether the option is a 'CALL' or a 'PUT'.
    :return: Black Scholes price for an FX option.
    """

    d_1: float = (np.log(initial_spot / strike) + ((rd - rf + 0.5 * volatility ** 2) * time_to_maturity)) / \
                 (volatility * math.sqrt(time_to_maturity))
    d_2: float = d_1 - volatility * math.sqrt(time_to_maturity)

    if str.upper(call_or_put) == 'CALL':
        return initial_spot * math.exp(rf * time_to_maturity) * norm.cdf(d_1) - strike * math.exp(-rd * time_to_maturity) * norm.cdf(d_2)
    elif str.upper(call_or_put) == 'PUT':
        return - initial_spot * math.exp(rf * time_to_maturity) * norm.cdf(-d_1) + strike * math.exp(-rd * time_to_maturity) * norm.cdf(-d_2)
    else:
        return f'Unknown option type: {call_or_put}'

#GBM pricer for FX option
def FX_option_monte_carlo_pricer(
        initial_spot: float,
        strike: float,
        rd: float,
        rf: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: str,
        number_of_paths: int,
        number_of_time_steps: int) -> [float | str]:

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
    :return: Monte Carlo price for an FX Opiton.
    """

    paths: np.ndarray = np.array(np.zeros((number_of_paths, number_of_time_steps)))

    paths[:, 0] = initial_spot

    dt: float = time_to_maturity / (number_of_time_steps - 1)

    # Actual Monte Carlo for the FX option
    for j in range(1, number_of_time_steps):
        z: float = norm.ppf(np.random.uniform(0, 1, number_of_paths))
        paths[:, j] = paths[:, j - 1] * np.exp(
            (rd - rf - 0.5 * volatility ** 2) * dt + volatility * math.sqrt(dt) * z)

    if str.upper(call_or_put) == 'CALL':
        payoffs = np.maximum(paths[:, -1] - strike, 0) * np.exp(-(rd-rf) * time_to_maturity)
        price: float = np.average(payoffs)
        return price

    elif str.upper(call_or_put) == 'PUT':
        payoffs = np.maximum(strike - paths[:, -1], 0) * np.exp(-(rd-rf) * time_to_maturity)
        price: float = np.average(payoffs)
        return price

    else:
        return f'Unknown option type: {call_or_put}'

initial_spot: float = 50
strike: float = 52
rd: float = 0.2
rf: float = 0.1
volatility: float = 0.4
time_to_maturity: float = 5 / 12
number_of_paths: int = 100_000
number_of_time_steps: int = 2

print(f'Black Scholes FX option price: { black_scholes_FX(initial_spot, strike, rd, rf, volatility, time_of_contract, "call")}')
print(f'Monte Carlo FX option price: {FX_option_monte_carlo_pricer(initial_spot, strike, rd, rf, volatility, time_of_contract, "call", number_of_paths, number_of_time_steps)}')
