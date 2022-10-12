"""
Contains a class for representing and pricing a FRA (Forward Rate Agreement).
"""
import numpy as np

from src.hullwhite.hullwhite import *


class Fra:
    """
    A class for representing and pricing a FRA (Forward Rate Agreement).
    """

    def __init__(self, notional: float, strike: float, start_tenor: float, end_tenor: float):
        """
        FRA constructor.

        :param notional: Notional.
        :param strike: Strike.
        :param start_tenor: The start tenor for the FRA e.g., for a 3x6 FRA this would be 3m.
        :param end_tenor: The end tenor for the FRA e.g., for a 3x6 FRA this would be 6m.
        """
        self.notional: float = notional
        self.strike: float = strike
        self.start_tenor: float = start_tenor
        self.end_tenor: float = end_tenor

    def get_monte_carlo_values(
            self,
            alpha: float,
            sigma: float,
            curve: Curve,
            number_of_paths: int,
            number_of_time_steps: int,
            short_rate_tenor: float = 0.01) -> [np.ndarray, np.ndarray]:
        """
        Gets the Monte Carlo values of the FRA at the various simulation time steps.

        :param alpha:
        :param sigma:
        :param curve:
        :param number_of_paths:
        :param number_of_time_steps:
        :param short_rate_tenor:
        :return: Monte Carlo (mean) values and error (standard deviation) per time step.
        """
        hw: HullWhite = HullWhite(alpha, sigma, curve, short_rate_tenor)

        tenors, short_rates, stochastic_dfs = \
            hw.simulate(
                self.start_tenor, number_of_paths, number_of_time_steps, HullWhiteSimulationMethod.SLOWANALYTICAL)

        start_discount_factors: np.ndarray = \
            hw.a_function(tenors, np.array([self.start_tenor])) * \
            np.exp(-1 * short_rates * hw.b_function(tenors, np.array([self.start_tenor])))

        end_discount_factors: np.ndarray = \
            hw.a_function(tenors, np.array([self.end_tenor])) * \
            np.exp(-1 * short_rates * hw.b_function(tenors, np.array([self.end_tenor])))

        # TODO: Generalise this into a function - get forward rates.
        forward_rates: np.ndarray = \
            (1 / (self.end_tenor - self.start_tenor)) * ((start_discount_factors / end_discount_factors) - 1)

        values: np.ndarray = \
            self.notional * (forward_rates - self.strike) * (self.end_tenor - self.start_tenor) * stochastic_dfs

        return np.mean(values, 0), np.sqrt(np.var(values, 0) / number_of_paths)

    def get_fair_forward_rate(self, curve: Curve) -> float:
        """
        Gets the FRA fair forward rate.

        :param curve:
        :return:
        """
        return curve.get_forward_rates(
            np.array([self.start_tenor]),
            np.array([self.end_tenor]),
            CompoundingConvention.Simple)

    def get_value(self, curve: Curve):
        """
        Gets the analytical fair value of the (long) FRA using the given input curve.

        :param curve: Discount curve.
        :return: FRA fair value.
        """
        return self.notional * (self.get_fair_forward_rate(curve) - self.strike) * (self.end_tenor - self.start_tenor)
