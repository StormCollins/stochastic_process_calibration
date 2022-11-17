"""
Contains a class for representing and pricing a FRA (Forward Rate Agreement).
"""
from src.hullwhite.hullwhite import *
from src.utils.plot_utils import PlotUtils


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
            initial_curve: Curve,
            number_of_paths: int,
            number_of_time_steps: int,
            short_rate_tenor: float = 0.01,
            plot_paths: bool = False,
            additional_annotation_for_plot: str = None) -> [np.ndarray, np.ndarray]:
        """
        Gets the Monte Carlo values of the FRA at the various simulation time steps.

        :param alpha: The mean reversion speed for the Hull-White simulation.
        :param sigma: The (constant) volatility for the Hull-White simulation.
        :param initial_curve: Initial curve.
        :param number_of_paths: Number of paths in Monte Carlo simulation.
        :param number_of_time_steps: Number of time steps in the Monte Carlo simulation.
        :param short_rate_tenor: The tenor of the underlying short rate to simulate.
        :param plot_paths: Plot the values of the FRA paths. Default = False.
        :param additional_annotation_for_plot: Additional annotation for the plot. Default = None.
        :return: Monte Carlo (mean) values and error (standard deviation) per time step.
        """
        hw: HullWhite = HullWhite(alpha, sigma, initial_curve, short_rate_tenor)

        simulation_tenors, short_rates, stochastic_dfs = \
            hw.simulate(
                maturity=self.start_tenor,
                number_of_paths=number_of_paths,
                number_of_time_steps=number_of_time_steps,
                method=HullWhiteSimulationMethod.DISCRETISED_INTEGRAL)

        start_discount_factors: np.ndarray = \
            hw.a_function(simulation_tenors, self.start_tenor) * \
            np.exp(-1 * short_rates * hw.b_function(simulation_tenors, self.start_tenor))

        end_discount_factors: np.ndarray = \
            hw.a_function(simulation_tenors, self.end_tenor) * \
            np.exp(-1 * short_rates * hw.b_function(simulation_tenors, self.end_tenor))

        # TODO: Generalise this into a function - get forward rates.
        forward_rates: np.ndarray = \
            (1 / (self.end_tenor - self.start_tenor)) * ((start_discount_factors / end_discount_factors) - 1)

        values: np.ndarray = \
            self.notional * (forward_rates - self.strike) * (self.end_tenor - self.start_tenor) * stochastic_dfs

        if plot_paths:
            PlotUtils.plot_monte_carlo_paths(
                time_steps=simulation_tenors,
                paths=values,
                title='Simulated FRA Values',
                additional_annotation=additional_annotation_for_plot)

        return np.mean(values, 0), np.sqrt(np.var(values, 0) / number_of_paths)

    def get_fair_forward_rate(self, curve: Curve) -> float:
        """
        Gets the FRA fair forward rate.

        :param curve:
        :return:
        """
        return curve.get_forward_rates(self.start_tenor, self.end_tenor, CompoundingConvention.Simple)

    def get_value(self, curve: Curve):
        """
        Gets the analytical fair value of the (long) FRA using the given input curve.

        :param curve: Discount curve.
        :return: FRA fair value.
        """
        return self.notional * (self.get_fair_forward_rate(curve) - self.strike) * (self.end_tenor - self.start_tenor)
