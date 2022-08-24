from curves.curve import Curve
import numpy as np
from scipy.interpolate import interp1d


class HullWhite:
    alpha: float
    sigma: float
    initial_curve: Curve

    def __init__(self, alpha: float, sigma: float, initial_curve: Curve):
        self.alpha = alpha
        self.sigma = sigma
        self.initial_curve = initial_curve


    def theta(
            self,
            theta_times: np.ndarray,
            curve_tenors: np.ndarray,
            curve_discount_factors: np.ndarray) -> interp1d:
        """
        Used to generate the theta function, used in the Hull-White model to represent the
        long term discount curve.

        :param theta_times: The time points at which we'd like to calculate theta.
        :param curve_tenors: The tenors of the discount curve.
        :param curve_discount_factors: The discount factors associated with the curve_tenors of the
        discount curve.
        :return:
        An interpolator which allows one to calculate theta for a given time.
        """
        discount_factor_interpolator = interp1d(curve_tenors, curve_discount_factors, kind='cubic')
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
        theta_interpolator: interp1d = interp1d(theta_times, thetas, kind='cubic')
        return theta_interpolator

    # Hull-White calibration parameters from Josh's code
    # print(theta(0.05, 0.01))
