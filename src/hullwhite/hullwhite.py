"""
Contains a class representing a Hull-White stochastic process.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import seaborn as sns
from scipy.stats import norm
from src.curves.curve import *
from src.utils.plot_utils import PlotUtils
from src.enums_and_named_tuples.hull_white_simulation_method import HullWhiteSimulationMethod


class HullWhite:
    """
    A class representing a Hull-White stochastic process.
    """
    current_discount_factor_interpolator: interp1d

    def __init__(
            self,
            alpha: float,
            sigma: float,
            initial_curve: Curve,
            short_rate_tenor: float):
        """
        HullWhite class constructor.

        :param alpha: The mean reversion speed.
        :param sigma: The volatility.
        :param initial_curve: The initial discount curve.
        :param short_rate_tenor: The short rate volatility_tenor (typically 0.25 i.e., 3m).
        """
        self.alpha = alpha
        self.sigma = sigma
        self.initial_curve = initial_curve
        # Small step used in numerical derivatives.
        self.numerical_derivative_step_size: float = 0.001
        self.theta_interpolator = self.calibrate_theta(initial_curve.tenors)
        self.short_rate_tenor = short_rate_tenor
        self.initial_short_rate = initial_curve.get_forward_rates(0, self.short_rate_tenor, CompoundingConvention.NACC)

    def calibrate_theta(self, theta_times: np.ndarray) -> interp1d:
        """
        Used to generate the theta function, used in the Hull-White model to represent the
        long term discount curve.

        :param theta_times: The tenors at which we calculate theta explicitly - the rest will be interpolated.
        :return: An interpolator for calculating theta for a given volatility_tenor.
        """
        theta_times = theta_times[theta_times > 0]
        discount_factors: np.ndarray = self.initial_curve.get_discount_factors(theta_times)
        zero_rates = -1 * np.log(discount_factors) / theta_times
        offset_discount_factors: np.ndarray = self.initial_curve.get_discount_factors(
            theta_times - self.numerical_derivative_step_size)
        offset_zero_rates = -1 * np.log(offset_discount_factors) / (theta_times - self.numerical_derivative_step_size)
        thetas: np.ndarray = \
            (zero_rates - offset_zero_rates) / self.numerical_derivative_step_size + \
            self.alpha * zero_rates + \
            self.sigma ** 2 / (2 * self.alpha) * (1 - np.exp(-2 * self.alpha * theta_times))
        # Given the discount curve like nature of theta, 'log-linear' interpolation seems the most reasonable.
        # TODO: Check extrapolation.
        theta_interpolator: interp1d = \
            interp1d(theta_times, np.log(thetas), kind='linear', fill_value='extrapolate')

        return theta_interpolator

    def theta(self, tenor: float) -> float:
        """
        Uses the theta_interpolator to return the log-linear interpolated value of theta.

        :param tenor: The volatility_tenor for which to calculate theta.
        :return: Theta for the given volatility_tenor.
        """
        return np.exp(self.theta_interpolator(tenor))

    def plot_paths(self, paths, maturity, drift):
        """
        Plots the monte carlo paths of Hull-White.

        :param paths: Short rate paths.
        :param maturity: Time to maturity.
        :param drift: Drift.
        :return:

        """

        time_steps = np.linspace(0, maturity, paths.shape[1])
        PlotUtils. \
            plot_monte_carlo_paths(time_steps, paths, f'$\\alpha$ = {self.alpha} & $\\sigma$ = {self.sigma}', drift)

    def simulate(
            self,
            maturity: float,
            drift: float,
            number_of_paths: int,
            number_of_time_steps: int,
            method: HullWhiteSimulationMethod = HullWhiteSimulationMethod.SLOWANALYTICAL,
            plot_results: bool = False) -> [np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates simulated short rates for the given Hull-White parameters.

        :param drift:
        :param plot_results:
        :param maturity: The maturity of the simulation.
        :param number_of_paths: The number of paths.
        :param number_of_time_steps: The number of time steps.
        :param method: Use the approximate, discretized Hull-White simulation method rather than the more
            accurate semi-analytical method. Default = False
        :return: Tuple of 3 arrays. Simulation tenors, short rates and stochastic discount factors.
        """
        short_rates: np.ndarray = np.zeros((number_of_paths, number_of_time_steps + 1))
        short_rates[:, 0] = self.initial_short_rate
        dt: float = maturity / number_of_time_steps
        # time_steps: np.ndarray = np.tile(np.linspace(0, maturity, short_rates.shape[1]), (number_of_paths, 1))
        time_steps: np.ndarray = np.linspace(0, maturity, short_rates.shape[1])

        if method == HullWhiteSimulationMethod.SLOWAPPROXIMATE:
            for j in range(number_of_time_steps):
                z: np.ndarray = norm.ppf(np.random.uniform(0, 1, number_of_paths))
                short_rates[:, j + 1] = \
                    short_rates[:, j] + (self.theta(j * dt) - self.alpha * short_rates[:, j]) * dt + \
                    self.sigma * z * np.sqrt(dt)
        elif method == HullWhiteSimulationMethod.SLOWANALYTICAL:
            for j in range(0, number_of_time_steps):
                short_rates[:, j + 1] = \
                    np.exp(-1 * self.alpha * dt) * short_rates[:, j] + \
                    scipy.integrate.quad(
                        lambda s: np.exp(self.alpha * (s - j * dt)) * self.theta(s), j * dt, (j + 1) * dt)[0] + \
                    self.sigma * np.exp(-1 * self.alpha * dt) * \
                    np.ndarray.flatten(self.exponential_stochastic_integral(j * dt, dt, number_of_paths))
        else:
            # TODO: Fix this.
            deterministic_part = \
                np.exp(-1 * self.alpha * time_steps) * self.initial_short_rate

            stochastic_part = np.zeros((number_of_paths, len(time_steps)))

            for j in range(0, number_of_time_steps):
                deterministic_part[j + 1] += \
                    scipy.integrate.quad(lambda s: np.exp(self.alpha * (s - j * dt)) * self.theta(s), 0, (j + 1) * dt)[
                        0]

                stochastic_part[:, j + 1] += \
                    self.sigma * np.exp(-1 * self.alpha * j * dt) * \
                    self.exponential_stochastic_integral(j * dt, dt, number_of_paths)

            short_rates = deterministic_part + stochastic_part

        if plot_results:
            self.plot_paths(short_rates, maturity, drift)

        stochastic_discount_factors: np.ndarray = np.cumprod(np.exp(-1 * short_rates * dt), 1)
        return time_steps, short_rates, stochastic_discount_factors

    def exponential_stochastic_integral(
            self,
            maturity: float,
            time_step_size: float,
            number_of_paths: int) -> np.ndarray:
        """
        Calculates the stochastic integral of the exponential term in the Hull-White analytical formula.

        :param maturity: The maturity/upper bound of the integral.
        :param time_step_size: The discretized step size for the integral.
        :param number_of_paths: The number of paths in the simulation.
        :return: An array of integral values for each path.
        """
        dt = time_step_size
        random_variables: np.ndarray = norm.ppf(np.random.uniform(0, 1, (number_of_paths, 1)))
        # TODO: Storm vs. Massi's approach - add toggle to switch between the two.
        # return np.sqrt((1/(2 * self.alpha)) * (np.exp(2 * self.alpha * maturity) - np.exp(2 * self.alpha * (maturity - dt)))) * random_variables
        return np.exp(self.alpha * maturity) * random_variables * np.sqrt(dt)

    def b_function(self, simulation_tenors: float | np.ndarray, end_tenors: float | np.ndarray) -> float | np.ndarray:
        """
        Used to calculate the 'B'-function commonly affiliated with Hull-White and used in the calculation of discount
        factors in the Hull-White simulation as per below.

        :math:`P(S,T) = A(S,T) e^{-r(S) B(S,T)}`

        :param simulation_tenors: The current tenor(s) in the simulation i.e., corresponding to S in P(S,T).
        :param end_tenors: Corresponds to T in P(S,T).
        :return: The 'B'-function value at the given tenor(s).
        """
        return (1 - np.exp(-self.alpha * (end_tenors - simulation_tenors))) / self.alpha

    def a_function(self, simulation_tenors: float | np.ndarray, tenors: float | np.ndarray) -> float | np.ndarray:
        """
        Used to calculate the 'A'-function commonly affiliated with Hull-White and used in the calculation of discount
        factors in the Hull-White simulation.

        :param simulation_tenors: The current tenor(s) in the simulation i.e., corresponding to S in P(S,T).
        :param tenors: Corresponding to T in P(S,T).
        :return: The 'A'-function value at the given tenors.
        """
        forward_discount_factors: np.ndarray = \
            self.initial_curve.get_discount_factors(tenors) / self.initial_curve.get_discount_factors(simulation_tenors)

        # If sigma is time-independent then we can replace the log discount factor derivatives
        # with the zero rate at that point.
        discount_factor_derivative: float = \
            -1 * self.b_function(simulation_tenors, tenors) * \
            self.initial_curve.get_log_discount_factor_derivatives(simulation_tenors)

        complex_factor: float = \
            self.sigma ** 2 * \
            (np.exp(-1 * self.alpha * tenors) - np.exp(-1 * self.alpha * simulation_tenors)) ** 2 * \
            (np.exp(2 * self.alpha * simulation_tenors) - 1) / \
            (4 * self.alpha ** 3)

        return forward_discount_factors * np.exp(discount_factor_derivative - complex_factor)

    # TODO: Check if this function is used.
    def get_discount_curve(
            self,
            short_rate: float,
            simulation_tenors: float | np.ndarray) -> Curve:
        """
        Gets the discount curve at the given time-step in the Hull-White simulation.

        :param short_rate: The short rate at the current point in the Hull-White simulation.
        :param simulation_tenors: The time point(s) in the Hull-White simulation.
        :return: A discount curve at the time point(s) in the Hull-White simulation.
        """
        tenors = self.initial_curve.tenors
        b = self.b_function(simulation_tenors, tenors)
        a = self.a_function(simulation_tenors, tenors)
        discount_factors: np.ndarray = a * np.transpose(np.exp(-np.outer(b, short_rate)))
        current_tenors = tenors[tenors >= simulation_tenors] - simulation_tenors
        current_discount_factors = discount_factors[:, (discount_factors.shape[1] - len(current_tenors)):]
        return Curve(current_tenors, current_discount_factors)

    # TODO: Check if this function is used.
    def get_discount_factors(self, tenors: float | np.ndarray) -> float | np.ndarray:
        """
        Gets the discount factors

        :param tenors: Tenors.
        :return:
        """
        return np.exp(self.current_discount_factor_interpolator(tenors) * tenors)

    def get_forward_discount_factors(self, start_tenors, end_tenors, short_rates):
        return self.a_function(start_tenors, end_tenors) * \
               np.exp(-1 * short_rates * self.b_function(start_tenors, end_tenors))

    def get_fixings(
            self,
            simulation_tenors: np.ndarray,
            simulated_short_rates: np.ndarray,
            fixing_period_start_tenors: np.ndarray,
            fixing_period_end_tenors: np.ndarray) -> np.ndarray:
        """
        Gets the simulated fixings (i.e., the reset rates) between the given start and end fixing tenors.

<<<<<<< HEAD
        :param simulation_tenors: The simulation tenors.
        :param simulated_short_rates: Simulated short rates.
        :param fixing_start_tenors: Fixing start tenors.
        :param fixing_end_tenors: Fixing end tenors.
        :return:
=======
        :param simulation_tenors: The tenors at which the short rate was simulated.
        :param simulated_short_rates: The simulated short rates.
        :param fixing_period_start_tenors: The start tenors of the fixing periods.
        :param fixing_period_end_tenors: The end tenors of the fixing periods.
        :return: The fixings for the given paths (rows) and start and end tenors (columns).
>>>>>>> be10d8f610c7cafaf63ea99ad9e6cdde413f4a53
        """
        fixing_start_tenor_indices = np.in1d(simulation_tenors, fixing_period_start_tenors)
        fixing_start_tenor_indices = \
            np.tile(fixing_start_tenor_indices, (simulated_short_rates.shape[0], 1))

        fixing_start_tenor_short_rates: np.ndarray = simulated_short_rates[fixing_start_tenor_indices]
        fixing_start_tenor_short_rates = \
            fixing_start_tenor_short_rates.reshape(fixing_start_tenor_indices.shape[0], len(fixing_period_start_tenors))

        forward_discount_factors: np.ndarray = \
            self.a_function(fixing_period_start_tenors, fixing_period_end_tenors) * \
            np.exp(-1 * fixing_start_tenor_short_rates *
                   self.b_function(fixing_period_start_tenors, fixing_period_end_tenors))

        fixings: np.ndarray = \
            (1 / (fixing_period_end_tenors - fixing_period_start_tenors)) * ((1 / forward_discount_factors) - 1)

        return fixings
