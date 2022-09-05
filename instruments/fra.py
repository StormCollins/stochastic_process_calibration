import numpy as np

from curves.curve import *
from hullwhite.hullwhite import *


def plot_fra_values(current_value: float, maturity: float):
    time = np.linspace(0, maturity, current_value)

    # Path plot
    # indices_sorted_by_path_averages = np.argsort(np.average(current_value, 1))
    # sorted_fra_values = np.transpose(current_value[indices_sorted_by_path_averages])
    # sns.set_palette(sns.color_palette('dark:purple', current_value))
    fig1, ax1 = plt.subplots()
    ax1.plot(time, current_value)
    ax1.grid(True)
    ax1.set_facecolor('#AAAAAA')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value of FRA')
    ax1.set_xlim([0, maturity])


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
        # maturity: float

        # maturity: float = time_steps[1]
        # current_short_rate: float = short_rates[0][1]
        # current_discount_curve: Curve = \
        #     hw.get_discount_curve(maturity, number_of_time_steps, current_short_rate, maturity)
        # current_value: float = self.get_value(current_discount_curve, maturity)
        # # TODO: Try to plot the values.
        # plot_fra_values(maturity, current_value)
        # return current_value

        for i in range(1, number_of_time_steps + 1):
            current_time_step: float = time_steps[i]
            current_short_rate: float = short_rates[i - 1][i]
            current_discount_curve: Curve = \
                hw.get_discount_curve(maturity, number_of_time_steps, current_short_rate, current_time_step)
            current_values: float = self.get_value(current_discount_curve, current_time_step)
            # plot_fra_values(current_time_step, current_values)
            print(current_values)
            return current_values

