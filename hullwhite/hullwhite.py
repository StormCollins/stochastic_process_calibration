from curves.curve import *
import numpy as np
import scipy.integrate
from scipy.interpolate import interp1d
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import seaborn as sns


def plot_paths(paths, maturity: float):
    time = np.linspace(0, maturity, paths.shape[1])

    # Path plot
    indices_sorted_by_path_averages = np.argsort(np.average(paths, 1))
    sorted_paths = np.transpose(paths[indices_sorted_by_path_averages])
    sns.set_palette(sns.color_palette('dark:purple', paths.shape[0]))
    fig1, ax1 = plt.subplots()
    ax1.plot(time, sorted_paths)
    ax1.grid(True)
    ax1.set_facecolor('#AAAAAA')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.set_xlim([0, maturity])


class HullWhite:
    alpha: float
    sigma: float
    theta_interpolator: interp1d
    initial_curve: Curve
    short_rate_tenor: float
    initial_short_rate: float

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
        :param short_rate_tenor: The short rate tenor (typically 0.25 i.e., 3m).
        """
        self.alpha = alpha
        self.sigma = sigma
        self.initial_curve = initial_curve
        self.theta_interpolator = self.setup_theta(initial_curve.tenors)
        self.short_rate_tenor = short_rate_tenor
        self.initial_short_rate = \
            initial_curve.get_forward_rates(
                np.array([0]),
                np.array([self.short_rate_tenor]),
                CompoundingConvention.NACQ)

    def setup_theta(self, theta_times: np.ndarray) -> interp1d:
        """
        Used to generate the theta function, used in the Hull-White model to represent the
        long term discount curve.

        :param theta_times: The tenors at which we calculate theta explicitly - the rest will be interpolated.
        :return: An interpolator for calculating theta for a given tenor.
        """
        theta_times = theta_times[theta_times > 0]
        discount_factors: np.ndarray = self.initial_curve.get_discount_factors(theta_times)
        zero_rates = -1 * np.log(discount_factors) / theta_times
        thetas: np.ndarray = \
            (zero_rates[1:] - zero_rates[0:-1]) / (theta_times[1:] - theta_times[0:-1]) + \
            self.alpha * zero_rates[0:-1] + \
            self.sigma ** 2 / (2 * self.alpha) * (1 - np.exp(-2 * self.alpha * theta_times[0:-1]))
        # Given the discount curve like nature of theta, 'log-linear' interpolation seems the most reasonable.
        # TODO: Check extrapolation.
        theta_interpolator: interp1d = \
            interp1d(theta_times[:-1], np.log(thetas), kind='linear', fill_value='extrapolate')
        return theta_interpolator

    def theta(self, tenor: float):
        """
        Uses the theta_interpolator to return the log-linear interpolated value of theta.

        :param tenor: The tenor for which to calculate theta.
        :return: Theta for the given tenor.
        """
        return np.exp(self.theta_interpolator(tenor))

    def simulate(
            self,
            maturity: float,
            number_of_paths: int,
            number_of_time_steps: int,
            approximate_method: bool = False) -> [np.ndarray, np.ndarray]:
        """
        Generates simulated short rates for the given Hull-White parameters.

        :param maturity: The maturity of the simulation.
        :param number_of_paths: The number of paths.
        :param number_of_time_steps: The number of time steps.
        :param approximate_method: Use the approximate, discretized Hull-White simulation method rather than the more
            accurate semi-analytical method. Default = False
        :return:
        """
        short_rates: np.ndarray = np.zeros((number_of_paths, number_of_time_steps + 1))
        short_rates[:, 0] = self.initial_short_rate
        dt: float = maturity / number_of_time_steps
        time_steps: np.ndarray = np.linspace(0, maturity, short_rates.shape[1])

        if approximate_method:
            for j in range(number_of_time_steps):
                z: np.ndarray = norm.ppf(np.random.uniform(0, 1, number_of_paths))
                short_rates[:, j + 1] = \
                    short_rates[:, j] + (self.theta(j * dt) - self.alpha * short_rates[:, j]) * dt + \
                    self.sigma * z * math.sqrt(dt)
        else:
            for j in range(number_of_time_steps):
                short_rates[:, j + 1] = \
                    np.exp(-1 * self.alpha * j * dt) * short_rates[:, 0] + \
                    scipy.integrate.quad(lambda t: np.exp(self.alpha * (t - j * dt)) * self.theta(t), 0, j * dt) + \
                    self.sigma * np.exp(-1 * self.alpha * j * dt) * \
                    self.exponential_stochastic_integral(j * dt, dt, number_of_paths)
        plot_paths(short_rates, maturity)
        return time_steps, short_rates

    def exponential_stochastic_integral(self, maturity: float, time_step_size: float, number_of_paths: int):
        """
        Calculates the stochastic integral of the exponential term in the Hull-White analytical formula.

        :param maturity: The maturity/upper bound of the integral.
        :param time_step_size: The discretized step size for the integral.
        :param number_of_paths: The number of paths in the simulation.
        :return: An array of integral values for each path.
        """
        time_steps: np.ndarray = np.arange(0, maturity, time_step_size) + time_step_size
        output: np.ndarry = np.zeros(number_of_paths)
        for i in range(0, number_of_paths):
            random_variables: np.ndarray = norm.ppf(np.random.uniform(0, 1, len(time_steps)))
            output[i] = \
                sum(
                    [np.exp(self.alpha * t) * z * np.sqrt(time_step_size)
                     for t, z in zip(time_steps, random_variables)])
        return output

    def b_function(self, tenors, current_tenor):
        """
        Used to calculate the 'B'-function commonly affiliated with Hull-White and used in the calculation of discount
        factors in the Hull-White simulation.

        :param tenors: The tenors at which we'd like to calculate the 'B'-function.
        :param current_tenor: The current tenor in the simulation.
        :return: The 'B'-function value at the given tenors.
        """
        return (1 - np.exp(-self.alpha * (tenors - current_tenor))) / self.alpha

    def a_function(self, tenors: np.ndarray, current_tenor: float) -> np.ndarray:
        forward_discount_factors: np.ndarray(np.type(float)) = \
            self.initial_curve.get_discount_factors(tenors) / \
            self.initial_curve.get_discount_factors(np.array([current_tenor]))

        # TODO: Should this not just be '-b * zero rate'?
        discount_factor_derivative: float = \
            -1 * self.b_function(tenors, current_tenor) * \
            (np.log(self.initial_curve.get_discount_factors(np.array([current_tenor]))) -
             np.log(self.initial_curve.get_discount_factors(np.array([current_tenor - self.short_rate_tenor])))) / \
            self.short_rate_tenor

        complex_factor: float = \
            self.sigma ** 2 * \
            (np.exp(-1 * self.alpha * tenors) - np.exp(-1 * self.alpha * current_tenor)) ** 2 * \
            (np.exp(2 * self.alpha * current_tenor) - 1) / \
            (4 * self.alpha ** 3)

        return forward_discount_factors * np.exp(discount_factor_derivative - complex_factor)

    def get_discount_curve(
            self,
            short_rate: float,
            current_tenor: float) -> Curve:
        """
        Gets the discount curve at the given time-step in the Hull-White simulation.

        :param short_rate: The short rate at the current point in the Hull-White simulation.
        :param current_tenor: The current time point in the Hull-White simulation.
        :return: A discount curve at the current time point in the Hull-White simulation.
        """
        tenors = self.initial_curve.tenors
        b = self.b_function(tenors, current_tenor)
        a = self.a_function(tenors, current_tenor)
        discount_factors: np.ndarray = a * np.exp(-b * short_rate)
        current_tenors = tenors[tenors >= current_tenor] - current_tenor
        current_discount_factors = discount_factors[(len(discount_factors) - len(current_tenors)):]
        return Curve(current_tenors, current_discount_factors)
