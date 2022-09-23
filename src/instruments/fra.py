from src.hullwhite.hullwhite import *


class Fra:
    """
    A class for creating and pricing a FRA (Forward Rate Agreement).

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
        hw: HullWhite = HullWhite(alpha, sigma, curve, short_rate_tenor)

        tenors, short_rates, stochastic_dfs = \
            hw.simulate(self.start_tenor, number_of_paths, number_of_time_steps, SimulationMethod.SLOWANALYTICAL)

        discount_factors = \
            hw.a_function(self.start_tenor, np.array([self.end_tenor])) * \
            np.exp(-1 * short_rates * hw.b_function(self.start_tenor, np.array([self.end_tenor])))

        # TODO: Replace '1 / discount_factors' with 'start_discount_factors / end_discount_factors'
        forward_rates = (1 / (self.end_tenor - self.start_tenor)) * ((1 / discount_factors) - 1)
        f = forward_rates[:, -1]

        # plt.style.use('ggplot')
        # fig, ax = plt.subplots(ncols=1, nrows=1)
        # ax.set_facecolor('#AAAAAA')
        # ax.grid(False)
        # rates: np.ndarray = short_rates[:, -1]
        # (values, bins, _) = ax.hist(f, bins=75, density=True, label='Histogram of $r(t)$', color='#6C3D91')
        # plt.show()
        # bin_centers = 0.5 * (bins[1:] + bins[:-1])

        values = self.notional * (forward_rates - self.strike) * (self.end_tenor - self.start_tenor) * stochastic_dfs
        return np.mean(values, 0), np.sqrt(np.var(values, 0) / number_of_paths)

    def get_fair_forward_rate(self, curve: Curve):
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
