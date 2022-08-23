import math
from scipy.stats import norm
from scipy.interpolate import interp1d
import numpy as np
from curves.curve import Curve


# Time Dependant Hull-White


def swap_rate(
        swaption_tenor: float,
        swap_tenor: float,
        m: float) -> float:
    """
    This function calculates the at-the-money swap rate for a given swaption tenor and a given swap tenor.

    :param swaption_tenor: The length of time remaining before the swaption expires.
    :param swap_tenor: The length of time remaining before the swap expires.
    :param m: Number of payments per year under the swap (i.e. the frequency of the swap payments,
            m = 4 if payments are quarterly.
    :return: At-the-money swap rate.

    """
    # df_interpolator = interp1d(curve_tenors, curve_discount_factors, kind='linear', fill_value='extrapolate')
    # check if the discount curve goes far enough for given swaption
    if (swaption_tenor + swap_tenor) < Curve.tenors[-1]:
        numerator: float = Curve.interpolator(swaption_tenor) - \
                           Curve.interpolator(swaption_tenor + swap_tenor)
        number_of_payments: float = swap_tenor * m

        denominator = 0
        for i in range(1, number_of_payments + 1):  # this ensures that we have an i for each payment
            denominator += Curve.interpolator(swaption_tenor + i * 1 / m) * 1 / m

        return numerator / denominator
    else:
        print("Error: Need a longer discount curve")


def swaption(notional: int, swaption_tenor: float, swap_tenor: float, volatility: float, m: int) -> float:
    """
    This function calculates the price of an at-the-money (ATM) swaption.

    :param notional: The notional principal amount.
    :param swaption_tenor: The length of time remaining before the swaption expires.
    :param swap_tenor: The length of time remaining before the swap expires.
    :param volatility: The swaption volatility.
    :param m: Number of payments per year under the swap (i.e. the frequency of the swap payments,
            m = 4 if payments are quarterly.
    :return: At-the-money swaption price.

    """

    d1: float = (volatility * math.sqrt(swaption_tenor)) / 2
    d2: float = -d1

    number_of_payments: float = swap_tenor * m
    # A can be regarded as the discount factor for m x swap_tenor payoffs. We now calculate the sum function of A.

    A = 0
    for i in range(1, number_of_payments + 1):
        A += 1 / m * Curve.interpolator(swaption_tenor + i * 1 / m)

    atm_swap_rate: float = swap_rate(swaption_tenor, swap_tenor, m)

    return notional * A * (atm_swap_rate * norm.cdf(d1) - atm_swap_rate * norm.cdf(d2))


def caplet(
        notional: float,
        delta: float,
        forward_interest_rate: float,
        volatility: float,
        interest_rate: float,
        time_to_maturity: float) -> float:
    d1 = (np.log(forward_interest_rate / interest_rate)
          + 0.5 * (volatility ** 2 * time_to_maturity)) / (volatility * math.sqrt(time_to_maturity))
    d2 = d1 - volatility * math.sqrt(time_to_maturity)
