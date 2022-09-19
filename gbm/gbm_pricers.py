import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import jarque_bera
import excel_file.excel_vol_surface_function
from excel_file.excel_vol_surface_function import *

MonteCarloResult = namedtuple('MonteCarloResult', ['price', 'error'])


def slow_equity_european_option_monte_carlo_pricer(
        notional: float,
        initial_spot: float,
        strike: float,
        interest_rate: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: str,
        number_of_paths: int,
        number_of_time_steps: int,
        show_stats: bool = False) -> [MonteCarloResult | str]:
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
    :param number_of_paths: Number of current_value to simulate for the option.
    :param number_of_time_steps: Number of time steps for the option.
    :show_stats: Displays the mean, standard deviation, 95% PFE and normality test.
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

    if show_stats:
        statistics(paths, initial_spot, interest_rate, volatility, time_to_maturity)

    if str.upper(call_or_put) == 'CALL':
        payoffs: np.ndarray = np.zeros(number_of_paths)
        for i in range(0, number_of_paths):
            payoffs[i] = np.max([paths[i, -1] - notional * strike, 0]) * math.exp(-interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
        return MonteCarloResult(price, error)

    elif str.upper(call_or_put) == 'PUT':
        payoffs: np.ndarray = np.zeros(number_of_paths)
        for i in range(0, number_of_paths):
            payoffs[i] = np.max([notional * strike - paths[i, -1], 0]) * math.exp(-interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        # Brandimarte, Numerical Methods in Finance and Economics, pg 265, eq 4.5
        error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
        return MonteCarloResult(price, error)

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
        plot_paths: bool = False,
        show_stats: bool = False) -> [MonteCarloResult | str]:
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
    :param number_of_paths: Number of current_value to simulate for the option.
    :param number_of_time_steps: Number of time steps for the option.
    :param plot_paths: If set to True plots the current_value.
    :show_stats: Displays the mean, standard deviation, 95% PFE and normality test.
    :return: Fast Monte Carlo price for an equity european option.
    """

    paths: np.ndarray = \
        generate_gbm_paths(
            number_of_paths,
            number_of_time_steps,
            notional,
            initial_spot,
            interest_rate,
            volatility,
            time_to_maturity)

    if plot_paths:
        create_gbm_plots(paths, interest_rate, volatility, time_to_maturity)

    if show_stats:
        statistics(paths, initial_spot, interest_rate, volatility, time_to_maturity)

    if str.upper(call_or_put) == 'CALL':
        payoffs = np.maximum(paths[:, -1] - notional * strike, 0) * np.exp(-interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        # Brandimarte, Numerical Methods in Finance and Economics, pg 265, eq 4.5
        error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
        return MonteCarloResult(price, error)

    elif str.upper(call_or_put) == 'PUT':
        payoffs = np.maximum(notional * strike - paths[:, -1], 0) * np.exp(-interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        # Brandimarte, Numerical Methods in Finance and Economics, pg 265, eq 4.5
        error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
        return MonteCarloResult(price, error)

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
        plot_paths: bool = True,
        show_stats: bool = False) -> [MonteCarloResult | str]:
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
    :param number_of_paths: Number of current_value for the FX option.
    :param number_of_time_steps: Number of time steps for the FX option.
    :param plot_paths: If set to True plots the current_value.
    :show_stats: If set to TruDisplays the mean, standard deviation, 95% PFE and normality test.
    :return: Monte Carlo price for an FX Option.
    """
    drift: float = domestic_interest_rate - foreign_interest_rate
    paths: np.ndarray = \
        generate_gbm_paths(number_of_paths, number_of_time_steps, notional, initial_spot, drift, volatility,
                           time_to_maturity)

    if plot_paths:
        create_gbm_plots(paths, domestic_interest_rate - foreign_interest_rate, volatility, time_to_maturity)

    if show_stats:
        statistics(paths, initial_spot, drift, volatility, time_to_maturity)

    if str.upper(call_or_put) == 'CALL':
        payoffs = np.maximum(paths[:, -1] - notional * strike, 0) * np.exp(-domestic_interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
        return MonteCarloResult(price, error)

    elif str.upper(call_or_put) == 'PUT':
        payoffs = np.maximum(notional * strike - paths[:, -1], 0) * np.exp(-domestic_interest_rate * time_to_maturity)
        price: float = np.average(payoffs)
        error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
        return MonteCarloResult(price, error)

    else:
        return f'Unknown option type: {call_or_put}'


def fx_forward_monte_carlo_pricer(
        notional: float,
        initial_spot: float,
        strike: float,
        domestic_interest_rate: float,
        foreign_interest_rate: float,
        time_to_maturity: float,
        number_of_paths: int,
        number_of_time_steps: int,
        volatility,
        plot_paths: bool = True,
        show_stats: bool = True) -> [MonteCarloResult | str]:
    """
    Returns the price for an FX forward (FEC) using monte carlo simulations.

    :param show_stats: Displays the mean, standard deviation, 95% PFE and normality test.
    :param volatility: Time-dependent FEC volatility
    :param notional: The notional of the FX forward denominated in the foreign currency
        i.e. we exchange the notional amount in the foreign currency for
        strike * notional amount in the domestic currency
        e.g. if strike = 17 USDZAR and notional = 1,000,000
        then we are exchanging USD 1,000,000 for ZAR 17,000,000.
    :param initial_spot: Initial spot price for the FX forward.
    :param strike: Strike price for the FX forward.
    :param domestic_interest_rate: Domestic interest rate.
    :param foreign_interest_rate: Foreign interest rate.
    :param time_to_maturity: Time to maturity (in years) of the FX forward.
    :param number_of_paths: Number of current_value for the FX forward.
    :param number_of_time_steps: Number of time steps for the FX forward.
    :param plot_paths: If set to True plots the current_value.
    :return: Monte Carlo price for an FX forward in the domestic currency.
    """
    # volatility = generate_time_dependent_volatilities(number_of_paths, number_of_time_steps, time_to_maturity)
    drift: float = domestic_interest_rate - foreign_interest_rate
    paths: np.ndarray = \
        generate_gbm_paths(number_of_paths, number_of_time_steps, notional, initial_spot, drift, volatility,
                           time_to_maturity)

    if plot_paths:
        create_gbm_plots(paths, domestic_interest_rate - foreign_interest_rate, volatility, time_to_maturity)

    if show_stats:
        statistics(paths, initial_spot, drift, volatility, time_to_maturity)

    payoffs = (paths[:, -1] - notional * strike) * np.exp(-domestic_interest_rate * time_to_maturity)
    price: float = np.average(payoffs)
    error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
    return MonteCarloResult(price, error)


# TODO SU-LISE - MY CODE. HY LOOP NIE VAN J=1 TOT 50 NIE.
def generate_time_dependent_volatilities(
        number_of_time_steps: int,
        time_to_maturity: float):
    """
    Returns the time dependent volatilities.

    :param number_of_time_steps:
    :param time_to_maturity:
    :return: The time dependent volatilities.
    """
    dt: float = time_to_maturity / number_of_time_steps
    # time_dependent_volatility_surface: np.ndarray = np.zeros((number_of_paths, number_of_time_steps + 1))
    # time_dependent_volatility_surface[:, 0] = excel_records_df.Quotes[0]
    # time_steps: np.ndarray = np.linspace(0, time_to_maturity, time_dependent_volatility.shape[1])
    # count = 0
    for j in range(1, number_of_time_steps + 1):
        # USING THE DICTIONARY FROM excel_vol_surface_function
        time_steps: np.ndarray = np.arange(0, j, dt)
        x = list(map(float, excel_records_df.Tenors))
        y = list(map(float, excel_records_df.Quotes))
        interpolated_volatility: interp1d = interp1d(x, y, kind='previous')
        time_dependent_volatility_surface = interpolated_volatility(time_steps) / 100
        time_dependent_volatility: interp1d = interp1d(time_steps, time_dependent_volatility_surface, kind='previous')
        time_dependent_volatility: float = time_dependent_volatility(time_to_maturity)
        # return time_steps, time_dependent_volatility_surface, time_dependent_volatility
        # count = count + 1
        return j, time_steps, time_dependent_volatility


# TODO: CODE EINDIG HIER


def generate_gbm_paths(
        number_of_paths: int,
        number_of_time_steps: int,
        notional: float,
        initial_spot: float,
        drift: float,
        volatility: float,
        time_to_maturity) -> np.ndarray:
    """
    Returns the monte carlo simulated Geometric Brownian Motion Paths.

    :param number_of_paths:
    :param number_of_time_steps:
    :param notional:
    :param initial_spot:
    :param drift:
    :param volatility:
    :param time_to_maturity:
    :return: The monte carlo simulated Geometric Brownian Motion Paths.
    """
    paths: np.ndarray = np.array(np.zeros((number_of_paths, number_of_time_steps + 1)))
    paths[:, 0] = initial_spot * notional
    dt: float = time_to_maturity / number_of_time_steps

    for j in range(1, number_of_time_steps + 1):
        z: float = norm.ppf(np.random.uniform(0, 1, number_of_paths))
        paths[:, j] = \
            paths[:, j - 1] * np.exp((drift - 0.5 * volatility ** 2) * dt + volatility *
                                     math.sqrt(dt) * z)

    return paths


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
    ax2.set_title('Comparison of GBM log-returns to normal PDF');
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
    ax3.set_title('Comparison of GBM normal-returns to log-normal PDF');
    ax3.legend()
    plt.show()


def statistics(paths, initial_spot, drift, volatility, time_to_maturity) -> float:
    log_returns = np.log(paths[:, -1] / paths[:, 0])
    jarque_bera_test = jarque_bera(log_returns)
    print(f'p-value: {jarque_bera_test.pvalue}')
    if jarque_bera_test.pvalue > 0.05:
        print('GBM current_value are normally distributed.')
    else:
        print('GBM current_value are not normally distributed.')

    # Path statistics
    mean: float = initial_spot * math.exp(drift + (volatility ** 2 / 2))
    standard_deviation: float = \
        initial_spot * np.sqrt((math.exp(volatility ** 2) - 1) * math.exp((2 * drift + volatility ** 2)))
    percentile: float = initial_spot * math.exp(
        drift * time_to_maturity + norm.ppf(0.95) * volatility * math.sqrt(time_to_maturity))

    print(f'Mean: {mean}')
    print(f'Standard Deviation: {standard_deviation}')
    print(f'95% PFE: {percentile}')
