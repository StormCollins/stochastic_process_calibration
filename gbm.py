import math
import struct

import numpy as np
from scipy.stats import norm


# Code up Black-Scholes
def black_scholes(
        spot: float,
        strike: float,
        interest_rate: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: str) -> float | str:
    """
    Returns the standard Black-Scholes price for a 'CALL' or 'PUT' option (does not take into account whether you
    are 'long' or 'short' the option.

    :param spot: The option spot price.
    :param strike: The option strike price.
    :param interest_rate: The interest rate/drift used for the option.
    :param volatility: The option volatility.
    :param time_to_maturity: The time (in years) at which the option expires.
    :param call_or_put: Indicates whether the option is a 'CALL' or a 'PUT'
    :return: Black-Scholes price for an option.
    """
    d_1: float = (np.log(spot / strike) + ((interest_rate + 0.5 * volatility ** 2) * time_to_maturity)) /\
          (volatility * math.sqrt(time_to_maturity))
    d_2: float = d_1 - volatility * math.sqrt(time_to_maturity)

    if str.upper(call_or_put) == 'CALL':
        return spot * norm.cdf(d_1) - strike * math.exp(-interest_rate * time_to_maturity) * norm.cdf(d_2)
    elif str.upper(call_or_put) == 'PUT':
        return - spot * norm.cdf(-d_1) + strike * math.exp(-interest_rate * time_to_maturity) * norm.cdf(-d_2)
    else:
        return f'Unknown option type: {call_or_put}'


print(black_scholes(50, 52, 0.1, 0.4, 5 / 12, 'put'))

# Code up Monte Carlo for option

# Compare Monte Carlo to Black-Scholes