from hullwhite.hullwhite import *


class Fra:
    def __init__(self, notional: float, strike: float, start_tenor: float, end_tenor: float):
        self.notional: float = notional
        self.strike: float = strike
        self.start_tenor: float = start_tenor
        self.end_tenor: float = end_tenor

    def get_monte_carlo_value(
            self,
            alpha: float,
            sigma: float,
            curve: Curve,
            number_of_paths: int,
            number_of_time_steps: int,
            short_rate_tenor: float = 0.01):
        hw: HullWhite = HullWhite(alpha, sigma, curve, short_rate_tenor)
        tenors, short_rates = \
            hw.simulate(self.start_tenor, number_of_paths, number_of_time_steps, SimulationMethod.SLOWANALYTICAL)
        discount_factors = \
            hw.a_function(self.start_tenor, np.array([self.end_tenor])) * \
            np.exp(-1 * short_rates[:, -1] * hw.b_function(self.start_tenor, np.array([self.end_tenor])))

        forward_rates = (1/(self.end_tenor - self.start_tenor)) * ((1 / discount_factors) - 1)
        dt = self.start_tenor / number_of_time_steps
        stochastic_discount_factors = np.prod(np.exp(-1 * short_rates * dt), 1)
        return np.mean(
            self.notional *
            (forward_rates - self.strike) *
            (self.end_tenor - self.start_tenor) *
            stochastic_discount_factors)

    def get_fair_forward_rate(self, curve: Curve):
        return curve.get_forward_rates(
            np.array([self.start_tenor]),
            np.array([self.end_tenor]),
            CompoundingConvention.Simple)

    def get_value(self, curve: Curve):
        """
        Gets the fair value of the (long) FRA using the given input curve.

        :param curve: Discount curve.
        :return: FRA fair value.
        """
        return self.notional * (self.get_fair_forward_rate(curve) - self.strike) * (self.end_tenor - self.start_tenor)
