import numpy as np
from scipy.interpolate import interp1d


def theta(
        alpha: float,
        sigma: float,
        theta_times: np.ndarray,
        curve_tenors: np.ndarray,
        curve_discount_factors: np.ndarray) -> interp1d:
    """
    Used to generate the theta function, used in the Hull-White model to represent the
    long term discount curve.

    :param alpha: The Hull-White mean reversion speed.
    :param sigma: The Hull-White volatility.
    :param theta_times: The time points at which we'd like to calculate theta.
    :param curve_tenors: The tenors of the discount curve.
    :param curve_discount_factors: The discount factors associated with the curve_tenors of the
    discount curve.
    :return:
    An interpolator which allows one to calculate theta for a given time.
    """
    discount_factor_interpolator = interp1d(curve_tenors, curve_discount_factors, kind='cubic')
    discount_factors = discount_factor_interpolator(theta_times)

    forward_rates = \
        1 / (theta_times[1:] - theta_times[0:-1]) * \
        np.log(discount_factors[0:-1] / discount_factors[1:])

    # We assume the initial, spot rate is the same as the 3m rate hence the below.
    forward_rates = np.concatenate(([forward_rates[0]], forward_rates))

    thetas: np.ndarray = (forward_rates[1:] - forward_rates[0:-1]) / (theta_times[1:] - theta_times[0:-1]) + \
                         alpha * forward_rates[0:-1] + \
                         sigma ** 2 / (2 * alpha) * (1 - np.exp(-2 * alpha * theta_times[0:-1]))
    theta_interpolator: interp1d = interp1d(theta_times, thetas, kind='cubic')
    return theta_interpolator

# Hull-White calibration parameters from Josh's code
# print(theta(0.05, 0.01))
