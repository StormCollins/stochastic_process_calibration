from curves.curve import *
import numpy as np
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
    theta: interp1d
    initial_curve: Curve
    short_rate_tenor: float

    # TODO: Make theta_times the same as initial_curve tenors.
    def __init__(
            self,
            alpha: float,
            sigma: float,
            initial_curve: Curve,
            theta_times: np.ndarray,
            short_rate_tenor: float):
        self.alpha = alpha
        self.sigma = sigma
        self.initial_curve = initial_curve
        self.theta_times = theta_times
        self.theta = self.setup_theta(theta_times)
        self.short_rate_tenor = short_rate_tenor
        self.initial_short_rate = initial_curve.get_forward_rate(0, self.short_rate_tenor, CompoundingConvention.Simple)

    def simulate(self, maturity: float, number_of_paths: int, number_of_time_steps: int):
        """
        This function gives the simulated short rates.

        :param maturity: The maturity of the short rate.
        :param number_of_paths:
        :param number_of_time_steps:
        :return:

        """
        short_rates: np.ndarray = np.zeros((number_of_paths, number_of_time_steps + 1))
        short_rates[:, 0] = self.initial_short_rate
        dt: float = maturity / number_of_time_steps
        time_steps: np.ndarray = np.linspace(0, maturity, short_rates.shape[1])

        for j in range(number_of_time_steps):
            z: float = norm.ppf(np.random.uniform(0, 1, number_of_paths))
            short_rates[:, j + 1] = short_rates[:, j] + (
                    self.theta(j * dt) - self.alpha * short_rates[:, j]) * dt + self.sigma * z * math.sqrt(dt)

        plot_paths(short_rates, maturity)
        return time_steps, short_rates

    def setup_theta(self, theta_times: np.ndarray) -> interp1d:
        """
        Used to generate the setup_theta function, used in the Hull-White model to represent the
        long term discount curve.

        :param theta_times: The time points at which we'd like to calculate setup_theta.
        :return: An interpolator which allows one to calculate setup_theta for a given time.

        """
        discount_factors = self.initial_curve.get_discount_factor(theta_times)

        forward_rates = \
            1 / (theta_times[1:] - theta_times[0:-1]) * \
            np.log(discount_factors[0:-1] / discount_factors[1:])

        # We assume the initial, short rate is the same as the 3m rate hence the below.
        forward_rates = np.concatenate(([forward_rates[0]], forward_rates))

        thetas: np.ndarray = \
            (forward_rates[1:] - forward_rates[0:-1]) / (theta_times[1:] - theta_times[0:-1]) + \
            self.alpha * forward_rates[0:-1] + \
            self.sigma ** 2 / (2 * self.alpha) * (1 - np.exp(-2 * self.alpha * theta_times[0:-1]))
        theta_interpolator: interp1d = interp1d(theta_times[:-1], thetas, kind='cubic')
        return theta_interpolator

    # Hull-White calibration parameters from Josh's code
    # print(setup_theta(0.05, 0.01))

    def get_discount_curve(
            self,
            maturity: float,
            number_of_time_steps: int,
            r: float,
            s: float) -> Curve:
        tenors = self.initial_curve.tenors
        b = (1 - np.exp(- self.alpha * (tenors - s))) / self.alpha
        current_time_step: np.ndarray = np.array(s)

        dt: float = maturity / number_of_time_steps
        curve: Curve = self.initial_curve

        forward_discount_factor: np.ndarray = \
            curve.get_discount_factor(tenors) / curve.get_discount_factor(current_time_step)
        discount_differential: np.ndarray = \
            -b * (np.log(curve.get_discount_factor(current_time_step)) -
                  np.log(curve.get_discount_factor(current_time_step - dt))) / dt

        a = forward_discount_factor * \
            np.exp(discount_differential -
                   (self.sigma ** 2 *
                    (np.exp(-self.alpha * tenors) - (np.exp(-self.alpha * current_time_step))) ** 2 *
                    (np.exp(2 * self.alpha * current_time_step) - 1)) / 4 * self.alpha ** 3)
        discount_factors: np.ndarray = a * np.exp(-b * r)
        current_tenors = tenors[tenors >= s] - dt
        current_discount_factors = discount_factors[(len(discount_factors) - len(current_tenors)):]
        return Curve(current_tenors, current_discount_factors)


class HullWhiteCurve:
    alpha: float
    sigma: float
    theta: interp1d
    initial_curve: Curve
    short_rate_tenor: float

    def __init__(
            self,
            alpha: float,
            sigma: float,
            initial_curve: Curve,
            short_rate_tenor: float):
        self.alpha = alpha
        self.sigma = sigma
        self.initial_curve = initial_curve
        self.short_rate_tenor = short_rate_tenor
        self.initial_short_rate = initial_curve.get_forward_rate(0, self.short_rate_tenor, CompoundingConvention.Simple)

    def get_discount_curve(self, tenors: np.ndarray, dt: float, r: float, s: float) -> Curve:
        b = (1 - math.exp(- self.alpha * (tenors - s))) / self.alpha
        a = Curve.get_discount_factor(tenors) / Curve.get_discount_factor(s) * math.exp(
            -b * (math.log(Curve.get_discount_factor(s)) -
                  math.log(Curve.get_discount_factor(s - 1))) / dt - (self.sigma ** 2 * ((math.exp(
                -self.alpha * tenors)) - (math.exp(-self.alpha * s))) ** 2 * (math.exp(
                2 * self.alpha * s) - 1)) / 4 * self.alpha ** 3)
        discount_factors: np.ndarray = a * math.exp(-b * r)
        return Curve(tenors, discount_factors)
