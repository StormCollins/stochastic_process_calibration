import math
import numpy as np
from scipy.stats import norm


def black_scholes(
        notional: float,
        initial_spot: float,
        strike: float,
        interest_rate: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: str) -> float | str:
    """
    Returns the standard Black-Scholes price for a 'CALL' or 'PUT' option (does not take into account whether you
    are 'long' or 'short' the option.

    :param notional: The notional of the FX forward denominated in the foreign currency
        i.e. we exchange the notional amount in the foreign currency for
        strike * notional amount in the domestic currency
        e.g. if strike = 17 USDZAR and notional = 1,000,000
        then we are exchanging USD 1,000,000 for ZAR 17,000,000.
    :param initial_spot: The initial spot price of the option.
    :param strike: The option strike price.
    :param interest_rate: The interest rate/drift used for the option.
    :param volatility: The option volatility.
    :param time_to_maturity: The time (in years) at which the option expires.
    :param call_or_put: Indicates whether the option is a 'CALL' or a 'PUT'
    :return: Black-Scholes price for an option.
    """
    initial_spot = initial_spot * notional
    strike = strike * notional

    d_1: float = (np.log(initial_spot / strike) + ((interest_rate + 0.5 * volatility ** 2) * time_to_maturity)) / \
                 (volatility * math.sqrt(time_to_maturity))
    d_2: float = d_1 - volatility * math.sqrt(time_to_maturity)

    if str.upper(call_or_put) == 'CALL':
        return initial_spot * norm.cdf(d_1) - strike * math.exp(-interest_rate * time_to_maturity) * norm.cdf(d_2)
    elif str.upper(call_or_put) == 'PUT':
        return - initial_spot * norm.cdf(-d_1) + strike * math.exp(-interest_rate * time_to_maturity) * norm.cdf(-d_2)
    else:
        return f'Unknown option type: {call_or_put}'


def fx_forward(
        notional: float,
        initial_spot: float,
        strike: float,
        domestic_interest_rate: float,
        foreign_interest_rate: float,
        time_to_maturity: float) -> float:
    """
    Returns the FX forward price. We follow the convention of quoting domestic currency per foreign currency i.e.
    FORDOM e.g. USDZAR = 17 means ZAR is the domestic currency, USD is the foreign it is 17 domestic (ZAR) to 1 foreign
    (USD).

    :param notional: The notional of the FX forward denominated in the foreign currency
        i.e. we exchange the notional amount in the foreign currency for
        strike * notional amount in the domestic currency
        e.g. if strike = 17 USDZAR and notional = 1,000,000
        then we are exchanging USD 1,000,000 for ZAR 17,000,000.
    :param initial_spot: The initial forward spot price in terms of domestic currency per foreign currency.
    :param strike: The strike in terms of domestic currency per foreign currency.
    :param domestic_interest_rate: The domestic currency interest rate.
    :param foreign_interest_rate: The foreign currency interest rate.
    :param time_to_maturity: The time (in years) at which the option expires.
    :return: The FX Forward price of the forward.
    """
    payoff: float = notional * initial_spot * np.exp(
        (domestic_interest_rate - foreign_interest_rate) * time_to_maturity) - notional * strike
    discount_payoff: float = payoff * np.exp(-domestic_interest_rate * time_to_maturity)
    return discount_payoff


def garman_kohlhagen(
        notional: float,
        initial_spot: float,
        strike: float,
        domestic_interest_rate: float,
        foreign_interest_rate: float,
        volatility: float,
        time_to_maturity: float,
        call_or_put: str) -> float | str:

    """

    The Garman-Kohlahgen model is an analytical model for valuing European options on currencies in the
    spot foreign exchange.
    This model is a modification to the Black-Scholes option pricing model such that he model can deal with
    two interest rates, the domestic interest rate and the foreign interest rate.
    Returns the Black-Scholes (Garman-Kohlahgen) price for a 'CALL' or 'PUT' FX option (does not take into account
    whether you are 'long' or 'short' the option..

    :param notional: The notional of the FX forward denominated in the foreign currency
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

    d_1: float = (np.log(initial_spot / strike) + ((domestic_interest_rate - foreign_interest_rate + 0.5 * volatility ** 2) * time_to_maturity)) / \
                 (volatility * math.sqrt(time_to_maturity))
    d_2: float = d_1 - volatility * math.sqrt(time_to_maturity)

    if str.upper(call_or_put) == 'CALL':
        return initial_spot * math.exp(-foreign_interest_rate * time_to_maturity) * norm.cdf(d_1) - \
               strike * math.exp(-domestic_interest_rate * time_to_maturity) * norm.cdf(d_2)
    elif str.upper(call_or_put) == 'PUT':
        return - initial_spot * math.exp(-foreign_interest_rate * time_to_maturity) * norm.cdf(-d_1) + \
               strike * math.exp(-domestic_interest_rate * time_to_maturity) * norm.cdf(-d_2)
    else:
        return f'Unknown option type: {call_or_put}'
