from curves.curve import *
from hullwhite.hullwhite import *


class Fra:
    """
    Used to encapsulate a FRA (forward rate agreement) with all it's parameters and functions.

    """
    forward_rate_start_tenor: float
    forward_rate_end_tenor: float
    strike: float

    def __init__(self, forward_rate_start_tenor, forward_rate_end_tenor, strike):
        self.forward_rate_start_tenor = forward_rate_start_tenor
        self.forward_rate_end_tenor = forward_rate_end_tenor
        self.strike = strike

    def calculate_fair_forward_rate(self, curve: Curve):
        return curve.get_forward_rate(
            start_tenor=self.forward_rate_start_tenor,
            end_tenor=self.forward_rate_end_tenor,
            compounding_convention=CompoundingConvention.Simple) - self.strike

    # Use the discount factors curve to value a FRA
    def value_a_fra(self, dt, tenors, alpha, sigma, s, r):
        b = (1 - math.exp(- alpha * (tenors - s))) / alpha
        a = Curve.get_discount_factor(tenors) / Curve.get_discount_factor(s) * math.exp(
            -b * (math.log(Curve.get_discount_factor(s)) -
                  math.log(Curve.get_discount_factor(s - 1))) / dt - (sigma ** 2 * ((math.exp(
                -alpha * tenors)) - (math.exp(-alpha * s))) ** 2 * (math.exp(
                2 * alpha * s) - 1)) / 4 * alpha ** 3)
        discount_factors: np.ndarray = a * math.exp(-b * r)

        forward_rates_at_payment_dates = -1 / dt * (
                discount_factors / HullWhiteCurve.get_discount_curve(tenors, dt, r, s)) - 1  # Luke's Code
        return forward_rates_at_payment_dates - self.strike
