import math
import numpy as np
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
from collections import namedtuple

MonteCarloResult = namedtuple('MonteCarloResult', ['price', 'error', 'mean', 'standard_deviation', 'percentile'])


def slow_equity_european_option_monte_carlo_pricer(
        notional: float,
        initial_spot: float,
        strike: float,
        interest_rate: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: str,
        number_of_paths: int,
        number_of_time_steps: int) -> [MonteCarloResult | str]:
    """
    Returns the price  for a 'CALL' or 'PUT' equity european option using monte carlo simulations.
    This is the slow equity european option monte carlo pricer, because it takes a longer time to run with more
    simulations.

    :param notional: The notional of the FX forward denominated in the foreign currency
        i.e. we exchange the notional amount in the foreign currency for
        strike * notional amount in the domestic currency
        e.g. if strike = 17 USDZAR and notional = 1,000,000
        then we are exchanging USD 1,000,000 for ZAR 17,000,000.
    :param initial_spot: The initial spot price for the option.
    :param strike: The strike price for the option.
    :param interest_rate:  The interest rate/drift used for the option.
    :param volatility: The option volatility.
    :param time_to_maturity: The time (in years) at which the option expires.
    :param call_or_put: Indicates whether the option is a 'CALL' or a 'PUT'.
    :param number_of_paths: Number of paths to simulate for the option.
    :param number_of_time_steps: Number of time steps for the option.
    :return: Slow Monte Carlo price for an equity european option.
    """

    paths: np.ndarray = np.array(np.zeros((number_of_paths, number_of_time_steps)))

    for i in range(0, number_of_paths):
        paths[i, 0] = notional * initial_spot

    dt: float = time_to_maturity / (number_of_time_steps - 1)

    # Actual Monte Carlo
    for j in range(1, number_of_time_steps):
        for i in range(0, number_of_paths):
            z: float = norm.ppf(np.random.uniform(0, 1))
            paths[i, j] = paths[i, j - 1] * math.exp(
                (interest_rate - 0.5 * volatility ** 2) * dt + volatility * math.sqrt(dt) * z)

    # Path statistics
    mean: float = initial_spot * math.exp(interest_rate + (volatility ** 2 / 2))
    standard_deviation: float = initial_spot * \
                                np.sqrt(
                                    (math.exp(volatility ** 2) - 1) * math.exp((2 * interest_rate + volatility ** 2)))
    percentile: float = initial_spot * math.exp(
        interest_rate * time_to_maturity + norm.ppf(0.95) * volatility * math.sqrt(time_to_maturity))

    if str.upper(call_or_put) == 'CALL':
        payoffs: np.ndarray = np.zeros(number_of_paths)
        for i in range(0, number_of_paths):
            payoffs[i] = np.max([paths[i, -1] - notional * strike, 0]) * math.exp(-interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
        return MonteCarloResult(price, error, mean, standard_deviation, percentile)

    elif str.upper(call_or_put) == 'PUT':
        payoffs: np.ndarray = np.zeros(number_of_paths)
        for i in range(0, number_of_paths):
            payoffs[i] = np.max([notional * strike - paths[i, -1], 0]) * math.exp(-interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        # Brandimarte, Numerical Methods in Finance and Economics, pg 265, eq 4.5
        error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
        return MonteCarloResult(price, error, mean, standard_deviation, percentile)

    else:
        return f'Unknown option type: {call_or_put}'


def fast_equity_european_option_monte_carlo_pricer(
        notional: float,
        initial_spot: float,
        strike: float,
        interest_rate: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: str,
        number_of_paths: int,
        number_of_time_steps: int,
        plot_paths: bool = False) -> [MonteCarloResult | str]:
    """
    Returns the price for a 'CALL' or 'PUT' equity european option using monte carlo simulations
    (does not take into account whether you are 'long' or 'short' the option).
    This function uses 'vectorisation' unlike the slow_equity_european_option_monte_carlo_pricer function thus
    speeding up performance.

    :param notional: The notional of the FX forward denominated in the foreign currency
        i.e. we exchange the notional amount in the foreign currency for
        strike * notional amount in the domestic currency
        e.g. if strike = 17 USDZAR and notional = 1,000,000
        then we are exchanging USD 1,000,000 for ZAR 17,000,000.
    :param initial_spot:The initial spot price of the option.
    :param strike: The option strike price.
    :param interest_rate: The interest rate/drift used for the option.
    :param volatility: The option volatility.
    :param time_to_maturity: The time (in years) at which the option expires.
    :param call_or_put: Indicates whether the option is a 'CALL' or a 'PUT'.
    :param number_of_paths: Number of paths to simulate for the option.
    :param number_of_time_steps: Number of time steps for the option.
    :param plot_paths: If set to True plots the paths.
    :return: Fast Monte Carlo price for an equity european option.
    """

    paths: np.ndarray = \
        generate_gbm_paths(number_of_paths, number_of_time_steps, notional, initial_spot, interest_rate, volatility, time_to_maturity)

    # Path statistics
    mean: float = initial_spot * math.exp(interest_rate + (volatility ** 2 / 2))
    standard_deviation: float = initial_spot * \
                                np.sqrt(
                                    (math.exp(volatility ** 2) - 1) * math.exp((2 * interest_rate + volatility ** 2)))
    percentile: float = initial_spot * math.exp(
        interest_rate * time_to_maturity + norm.ppf(0.95) * volatility * math.sqrt(time_to_maturity))

    if plot_paths:
        create_gbm_plots(paths, interest_rate, volatility, time_to_maturity)

    if str.upper(call_or_put) == 'CALL':
        payoffs = np.maximum(paths[:, -1] - notional * strike, 0) * np.exp(-interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        # Brandimarte, Numerical Methods in Finance and Economics, pg 265, eq 4.5
        error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
        return MonteCarloResult(price, error, mean, standard_deviation, percentile)

    elif str.upper(call_or_put) == 'PUT':
        payoffs = np.maximum(notional * strike - paths[:, -1], 0) * np.exp(-interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        # Brandimarte, Numerical Methods in Finance and Economics, pg 265, eq 4.5
        error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
        return MonteCarloResult(price, error, mean, standard_deviation, percentile)

    else:
        return f'Unknown option type: {call_or_put}'


def fx_option_monte_carlo_pricer(
        notional: float,
        initial_spot: float,
        strike: float,
        domestic_interest_rate: float,
        foreign_interest_rate: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: str,
        number_of_paths: int,
        number_of_time_steps: int,
        plot_paths: bool = True) -> [MonteCarloResult | str]:
    """
    Returns the price for a 'CALL' or 'PUT' FX option using monte carlo simulations (does not take into account whether
    you are 'long' or 'short' the option).

    :param notional: The notional of the FX forward denominated in the foreign currency
        i.e. we exchange the notional amount in the foreign currency for
        strike * notional amount in the domestic currency
        e.g. if strike = 17 USDZAR and notional = 1,000,000
        then we are exchanging USD 1,000,000 for ZAR 17,000,000.
    :param initial_spot: Initial spot price for the FX option.
    :param strike: Strike price for the FX option.
    :param domestic_interest_rate: Domestic interest rate.
    :param foreign_interest_rate: Foreign interest rate.
    :param volatility: Volatility of the FX rate.
    :param time_to_maturity: Time to maturity (in years) of the FX option.
    :param call_or_put: Indicates whether the option is a 'CALL' or a 'PUT'.
    :param number_of_paths: Number of paths for the FX option.
    :param number_of_time_steps: Number of time steps for the FX option.
    :param plot_paths: If set to True plots the paths.
    :return: Monte Carlo price for an FX Option.
    """
    drift: float = domestic_interest_rate - foreign_interest_rate
    paths: np.ndarray = \
        generate_gbm_paths(number_of_paths, number_of_time_steps, notional, initial_spot, drift, volatility, time_to_maturity)

    # Path statistics
    mean: float = initial_spot * math.exp(domestic_interest_rate - foreign_interest_rate + (volatility ** 2 / 2))
    standard_deviation: float = initial_spot * \
                                np.sqrt((math.exp(volatility ** 2) - 1) * math.exp(
                                    (2 * domestic_interest_rate - foreign_interest_rate + volatility ** 2)))
    percentile: float = initial_spot * math.exp(
        domestic_interest_rate - foreign_interest_rate * time_to_maturity + norm.ppf(0.95) * volatility * math.sqrt(
            time_to_maturity))

    if plot_paths:
        create_gbm_plots(paths, domestic_interest_rate - foreign_interest_rate, volatility, time_to_maturity)

    if str.upper(call_or_put) == 'CALL':
        payoffs = np.maximum(paths[:, -1] - notional * strike, 0) * np.exp(-domestic_interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
        return MonteCarloResult(price, error, mean, standard_deviation, percentile)

    elif str.upper(call_or_put) == 'PUT':
        payoffs = np.maximum(notional * strike - paths[:, -1], 0) * np.exp(-domestic_interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
        return MonteCarloResult(price, error, mean, standard_deviation, percentile)

    else:
        return f'Unknown option type: {call_or_put}'


def fx_forward_monte_carlo_pricer(
        notional: float,
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

    :param notional: The notional of the FX forward denominated in the foreign currency
        i.e. we exchange the notional amount in the foreign currency for
        strike * notional amount in the domestic currency
        e.g. if strike = 17 USDZAR and notional = 1,000,000
        then we are exchanging USD 1,000,000 for ZAR 17,000,000.
    :param initial_spot: Initial spot price for the FX forward.
    :param strike: Strike price for the FX forward.
    :param domestic_interest_rate: Domestic interest rate.
    :param foreign_interest_rate: Foreign interest rate.
    :param volatility: Volatility of the FX rate.
    :param time_to_maturity: Time to maturity (in years) of the FX forward.
    :param number_of_paths: Number of paths for the FX forward.
    :param number_of_time_steps: Number of time steps for the FX forward.
    :param plot_paths: If set to True plots the paths.
    :return: Monte Carlo price for an FX forward in the domestic currency.
    """

    drift: float = domestic_interest_rate - foreign_interest_rate
    paths: np.ndarray = \
        generate_gbm_paths(number_of_paths, number_of_time_steps, notional, initial_spot, drift, volatility, time_to_maturity)


    # Path statistics
    mean: float = initial_spot * math.exp(domestic_interest_rate - foreign_interest_rate + (volatility ** 2 / 2))
    standard_deviation: float = initial_spot * \
                                np.sqrt((math.exp(volatility ** 2) - 1) * math.exp(
                                    (2 * domestic_interest_rate - foreign_interest_rate + volatility ** 2)))
    percentile: float = initial_spot * math.exp(
        domestic_interest_rate - foreign_interest_rate * time_to_maturity + norm.ppf(0.95) * volatility * math.sqrt(
            time_to_maturity))

    if plot_paths:
        create_gbm_plots(paths, domestic_interest_rate - foreign_interest_rate, volatility, time_to_maturity)

    payoffs = (paths[:, -1] - notional * strike) * np.exp(-domestic_interest_rate * time_to_maturity)
    price: float = np.average(payoffs)
    error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
    return MonteCarloResult(price, error, mean, standard_deviation, percentile)


def generate_gbm_paths(
        number_of_paths: int,
        number_of_time_steps: int,
        notional: float,
        initial_spot: float,
        drift: float,
        volatility: float,
        time_to_maturity) -> np.ndarray:
    # TODO: Add function description.
    paths: np.ndarray = np.array(np.zeros((number_of_paths, number_of_time_steps)))
    paths[:, 0] = initial_spot * notional
    dt: float = time_to_maturity / (number_of_time_steps - 1)

    for j in range(1, number_of_time_steps):
        z: float = norm.ppf(np.random.uniform(0, 1, number_of_paths))
        paths[:, j] = \
            paths[:, j - 1] * np.exp((drift - 0.5 * volatility ** 2) * dt + volatility * math.sqrt(dt) * z)

    return paths


def create_gbm_plots(paths, interest_rate: float, volatility: float, time_to_maturity: float) -> None:
    # Todo: Add function description.

    # Path plot
    indices_sorted_by_path_averages = np.argsort(np.average(paths, 1))
    sorted_paths = np.transpose(paths[indices_sorted_by_path_averages])

    # sns.set_palette(sns.cubehelix_palette(paths.shape[0], start=.5, rot=-.75))
    sns.set_palette(sns.color_palette('dark:purple', paths.shape[0]))
    fig1, ax1 = plt.subplots()
    ax1.plot(sorted_paths)
    ax1.grid(True)
    ax1.set_xlabel('Number of time steps')
    ax1.set_ylabel('Number of paths')

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
    ax2.set_title('Comparison of GBM log-returns to normal PDF');
    ax2.legend()
    plt.show()
