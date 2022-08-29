from curves.curve import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import math
import matplotlib.pyplot as plt


class HullWhite:
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
        paths: np.ndarray = np.zeros((number_of_paths, number_of_time_steps + 1))
        paths[:, 0] = self.initial_short_rate
        dt: float = maturity / number_of_time_steps

        for j in range(number_of_paths):
            z: float = norm.ppf(np.random.uniform(0, 1, number_of_paths))
            paths[:, j + 1] = paths[:, j] + (
                    self.theta(j * dt) - self.alpha * paths[:, j]) * dt + self.sigma * z * math.sqrt(dt)

        # time, paths = self.simulate(maturity, number_of_paths, number_of_time_steps)
        # for i in range(number_of_paths):
        #     plt.plot(time, paths[i, :], lw=0.8, alpha=0.6)
        # plt.title("Hull-White Short Rate Simulation")
        # plt.show()
        return paths

    def setup_theta(
            self,
            theta_times: np.ndarray) -> interp1d:
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
