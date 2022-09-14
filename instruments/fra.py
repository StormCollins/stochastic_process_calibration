from hullwhite.hullwhite import *


class ValuationType(Enum):
    FUTUREVALUE = 1
    PRESENTVALUE = 2


# def plot_fra_values(current_value: float, maturity: float):
#     time = np.linspace(0, maturity, current_value)
#
#     # Path plot
#     # indices_sorted_by_path_averages = np.argsort(np.average(current_value, 1))
#     # sorted_fra_values = np.transpose(current_value[indices_sorted_by_path_averages])
#     # sns.set_palette(sns.color_palette('dark:purple', current_value))
#     fig1, ax1 = plt.subplots()
#     ax1.plot(time, current_value)
#     ax1.grid(True)
#     ax1.set_facecolor('#AAAAAA')
#     ax1.set_xlabel('Time')
#     ax1.set_ylabel('Value of FRA')
#     ax1.set_xlim([0, maturity])


class Fra:
    """
    Used to encapsulate a FRA (forward rate agreement) with all it's fields and methods.

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
        """
        FRA constructor.

        :param notional: The notional amount of the FRA.
        :param forward_rate_start_tenor: The start tenor of the FRA e.g., for a 3x6 FRA this would be 3m.
        :param forward_rate_end_tenor: The end tenor of the FRA e.g., for a 3x6 FRA this would be 6m.
        :param strike: The strike of the FRA.
        """
        self.notional = notional
        self.forward_rate_start_tenor = forward_rate_start_tenor
        self.forward_rate_end_tenor = forward_rate_end_tenor
        self.strike = strike

    def get_fair_forward_rate(
            self,
            curve: Curve,
            current_tenor: float = 0,
            compounding_convention: CompoundingConvention = CompoundingConvention.NACQ) -> np.ndarray:
        """
        Gets the fair forward rate for the FRA i.e., the strike that would cause the FRA to be valued to zero at the
        current tenor. Note a simple rate is used.

        :param curve: The curve for the current tenor.
        :param current_tenor: The current tenor (default value = 0).
        :param compounding_convention: Compounding convention (default value = NACQ since this is the most common for
            FRAs in South Africa),
        :return: The fair forward rate that would set the FRA value to zero.
        """
        return curve.get_forward_rates(
            start_tenors=np.array([self.forward_rate_start_tenor - current_tenor]),
            end_tenors=np.array([self.forward_rate_end_tenor - current_tenor]),
            compounding_convention=compounding_convention)

    def get_values(self, curve: Curve, current_time: float = 0) -> np.ndarray:
        """
        Calculates the value of the FRA at the current time using the given interest rate curve.

        :param curve: Interest rate curve at the current time.
        :param current_time: The current time - this may be some time step in a Monte Carlo simulation.
        :return: The value of the FRA.
        """
        return self.notional * \
            (self.get_fair_forward_rate(curve, current_time) - self.strike) * \
            (self.forward_rate_end_tenor - self.forward_rate_start_tenor)

    def get_monte_carlo_value(
            self,
            hw: HullWhite,
            number_of_paths: int,
            number_of_time_steps: int,
            method: str = 'fast_analytical',
            valuation_type: ValuationType = ValuationType.FUTUREVALUE) -> float:
        """
        1. Simulate the short rate
        2. Get the discount curve at each time step of the simulation
        3. Value the FRA at each time step using the relevant discount curve from step 2.

        :param valuation_type:
        :param hw:
        :return:
        """
        maturity: float = self.forward_rate_start_tenor
        time_steps, short_rates = hw.simulate(maturity, number_of_paths, number_of_time_steps, method)

        fra_values: np.ndarray = np.zeros((number_of_paths, number_of_time_steps + 1), float)
        initial_fra_value = self.get_values(hw.initial_curve, 0)
        fra_values[:, 0] = initial_fra_value

        step_wise_stochastic_discount_factors: np.ndarray = np.zeros((number_of_paths, number_of_time_steps))
        initial_stochastic_discount_factor: float = hw.initial_curve.get_discount_factors(np.array([time_steps[1]]))[0]
        step_wise_stochastic_discount_factors[:, 0] = initial_stochastic_discount_factor

        # for i in range(0, number_of_paths):
        #     for j in range(1, number_of_time_steps + 1):
        #         current_time_step: float = time_steps[j]
        #         current_short_rate: float = short_rates[i][j]
        #         current_discount_curve: Curve = \
        #             hw.get_discount_curve(current_short_rate, current_time_step)
        #         current_value = self.get_values(current_discount_curve, current_time_step)
        #         fra_values[i][j] = current_value
        #         if j < number_of_time_steps:
        #             step_wise_stochastic_discount_factors[i][j] = \
        #                 current_discount_curve.get_discount_factors(time_steps[j + 1])
                # plot_fra_values(current_time_step, current_values)

        for j in range(1, number_of_time_steps + 1):
            current_time_step: float = time_steps[j]
            current_discount_curves: Curve = \
                hw.get_discount_curve(short_rates[:, j], current_time_step)
            current_value = self.get_values(current_discount_curves, current_time_step)
            fra_values[:, j] = np.ndarray.flatten(current_value)
            if j < number_of_time_steps:
                step_wise_stochastic_discount_factors[:, j] = \
                    current_discount_curves.get_discount_factors(time_steps[j + 1])

        stochastic_discount_factors = np.prod(step_wise_stochastic_discount_factors, 1)
        fra_value: float = np.average(fra_values[:, -1] * stochastic_discount_factors)
        return fra_value
