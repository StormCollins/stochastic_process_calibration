import math
import numpy as np
from src.call_or_put import CallOrPut
from scipy.stats import norm


def garman_kohlhagen(
        notional: float,
        initial_spot: float,
        strike: float,
        domestic_interest_rate: float,
        foreign_interest_rate: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: CallOrPut) -> float | str:

    """
    The Garman-Kohlahgen model is an analytical model for valuing European options on foreign exchange.
    This model is a modification to the Black-Scholes option pricing model such that he model can deal with
    two interest rates, the domestic interest rate and the foreign interest rate.
    Returns the Black-Scholes (Garman-Kohlahgen) price for a 'CALL' or 'PUT' FX option (does not take into account
    whether you are 'long' or 'short' the option..

    :param notional: Notional of the FX forward denominated in the foreign currency
        i.e. we exchange the notional amount in the foreign currency for
        strike * notional amount in the domestic currency
        e.g. if strike = 17 USDZAR and notional = 1,000,000
        then we are exchanging USD 1,000,000 for ZAR 17,000,000.
    :param initial_spot: Initial spot rate of the FX option.
    :param strike: Strike price of the FX option.
    :param domestic_interest_rate: Domestic interest rate.
    :param foreign_interest_rate: Foreign interest rate.
    :param volatility: Volatility of the FX rate.
    :param time_to_maturity: Time to maturity (in years) of the FX option.
    :param call_or_put: Indicates whether the option is a 'CALL' or a 'PUT'.
    :return: Garman_kohlhagen price for an FX option.
    """
    initial_spot = initial_spot * notional
    strike = strike * notional

    d_1: float = (np.log(initial_spot / strike) +
                  ((domestic_interest_rate - foreign_interest_rate + 0.5 * volatility ** 2) * time_to_maturity)) / \
                 (volatility * math.sqrt(time_to_maturity))

    d_2: float = d_1 - volatility * math.sqrt(time_to_maturity)

    if call_or_put == CallOrPut.CALL:
        return initial_spot * math.exp(-1 * foreign_interest_rate * time_to_maturity) * norm.cdf(d_1) - \
               strike * math.exp(-1 * domestic_interest_rate * time_to_maturity) * norm.cdf(d_2)
    elif call_or_put == CallOrPut.PUT:
        return -1 * initial_spot * math.exp(-1 * foreign_interest_rate * time_to_maturity) * norm.cdf(-1 * d_1) + \
               strike * math.exp(-domestic_interest_rate * time_to_maturity) * norm.cdf(-1 * d_2)
    else:
        return f'Unknown option type: {call_or_put}'
