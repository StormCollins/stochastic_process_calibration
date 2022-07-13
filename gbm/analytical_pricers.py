import math
import numpy as np
from scipy.stats import norm


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


# TODO: Add description of function in comments below.
# TODO: Rename 'rd' to domestic_interest_rate
# TODO: Rename 'rf' to foreign_interest_rate
# TODO: Rename time_of_contract to time_to_maturity.
def fx_forward(
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


# TODO: Rename function to garman_kohlhagen.
# TODO: Add a description of the function in the comments.
# TODO: Rename rd and rf to be the same as fx_forward inputs above.
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
        return initial_spot * math.exp(-rf * time_to_maturity) * norm.cdf(d_1) - \
               strike * math.exp(-rd * time_to_maturity) * norm.cdf(d_2)
    elif str.upper(call_or_put) == 'PUT':
        return - initial_spot * math.exp(-rf * time_to_maturity) * norm.cdf(-d_1) + \
               strike * math.exp(-rd * time_to_maturity) * norm.cdf(-d_2)
    else:
        return f'Unknown option type: {call_or_put}'
