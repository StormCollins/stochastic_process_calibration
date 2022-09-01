import numpy as np

from curves.curve import *
from hullwhite.hullwhite import *


class Fra:
    """
    Used to encapsulate a FRA (forward rate agreement) with all it's parameters and functions.

    """
    notional: float
    forward_rate_start_tenor: float
    forward_rate_end_tenor: float
    strike: float

    def __init__(
            self,
            notional: float,
            forward_rate_start_tenor: float,
            forward_rate_end_tenor: float,
            strike: float):
        self.notional = notional
        self.forward_rate_start_tenor = forward_rate_start_tenor
        self.forward_rate_end_tenor = forward_rate_end_tenor
        self.strike = strike

    def get_fair_forward_rate(self, curve: Curve, current_time: float = 0) -> float:
        return curve.get_forward_rate(
            start_tenor=self.forward_rate_start_tenor - current_time,
            end_tenor=self.forward_rate_end_tenor - current_time,
            compounding_convention=CompoundingConvention.Simple)

    def get_value(self, curve: Curve, current_time: float = 0) -> float:
        return self.notional * \
               (self.get_fair_forward_rate(curve, current_time) - self.strike) * \
               (self.forward_rate_end_tenor - self.forward_rate_start_tenor) * \
               curve.get_discount_factor(np.array([self.forward_rate_start_tenor - current_time]))[0]

    def get_monte_carlo_value(self, hw: HullWhite):
        """
        1. Simulate the short rate
        2. Get the discount curve at each time step of the simulation
        3. Value the FRA at each time step using the relevant discount curve from step 2.

        :param hw:
        :return:
        """
        maturity: float = self.forward_rate_start_tenor
        number_of_paths: int = 1
        number_of_time_steps: int = 4
        time_steps, short_rates = hw.simulate(maturity, number_of_paths, number_of_time_steps)
        current_time_step: float

        # TODO: Turn into a loop.
        current_time_step: float = time_steps[1]
        current_short_rate: float = short_rates[0][1]
        current_discount_curve: Curve = \
            hw.get_discount_curve(maturity, number_of_time_steps, current_short_rate, current_time_step)
        current_value: float = self.get_value(current_discount_curve, current_time_step)
        # TODO: Try to plot the values.
        print(current_value)



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




