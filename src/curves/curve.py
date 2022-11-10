"""
Contains a class for representing a single discount curve or a set of discount curves.

In the case of a set of discount factors they are constructed using the same tenors but different discount factors for 
those tenors - this is useful in representing a curve being simulated during Monte Carlo.
"""
import numpy as np
from scipy.interpolate import interp1d
from src.enums_and_named_tuples.compounding_convention import CompoundingConvention


class Curve:
    """
    A class for representing a discount curve.
    """
    numerical_derivative_step_size: float = 0.000001
    tenors: np.ndarray
    discount_factors: np.ndarray
    discount_factor_interpolator: interp1d

    def __init__(self, tenors: np.ndarray, discount_factors: np.ndarray):
        """
        Curve constructor. Uses linear interpolation on the log of discount factors.

        :param tenors: Tenors.
        :param discount_factors: Discount factors.
        """
        self.tenors = tenors
        self.discount_factors = discount_factors
        if discount_factors.ndim == 1:
            self.discount_factor_interpolator = \
                interp1d(tenors, np.log(discount_factors), 'linear', fill_value='extrapolate')
        elif discount_factors.ndim == 2:
            self.discount_factor_interpolator = \
                interp1d(tenors, np.log(discount_factors), 'linear', fill_value='extrapolate', axis=1)
        else:
            raise ValueError(f'Discount factor dimensions should be 1 or 2 but received {discount_factors.ndim}.')

    def get_discount_factors(self, tenors: float | np.ndarray) -> float | np.ndarray:
        """
        Gets discount factors for a given set of tenors.

        :param tenors: Tenors.
        :return: An array of discount factors.
        """
        return np.exp(self.discount_factor_interpolator(tenors))

    def get_forward_discount_factor(self, start_tenors: np.ndarray, end_tenors: np.ndarray) -> np.ndarray:
        """
        Calculates the discount factor between times start_tenor and end_tenor.

        :param start_tenors: The volatility_tenor(s) we want to discount back to.
        :param end_tenors: The volatility_tenor(s) we want to discount back from.
        :return: An array of forward discount factor(s).
        """
        return self.discount_factor_interpolator(np.array(end_tenors)) / \
            self.discount_factor_interpolator(np.array(start_tenors))

    def get_forward_rates(
            self,
            start_tenors: float | np.ndarray,
            end_tenors: float | np.ndarray,
            compounding_convention: CompoundingConvention) -> float | np.ndarray:
        """
        Gets the forward rates for between a given array of start tenors and end tenors.

        :param start_tenors: Start tenors (starting tenors of the forward rates).
        :param end_tenors: End tenors (end tenors of the forward rates).
        :param compounding_convention: The compounding convention.
        :return: An array of forward rates.
        """
        start_discount_factors: np.ndarray = self.get_discount_factors(start_tenors)
        end_discount_factors: np.ndarray = self.get_discount_factors(end_tenors)
        if compounding_convention == compounding_convention.NACC:
            return (-1 / (end_tenors - start_tenors)) * np.log(end_discount_factors / start_discount_factors)
        elif compounding_convention == compounding_convention.NACQ:
            return 4 * ((start_discount_factors / end_discount_factors) ** (4 * (end_tenors - start_tenors)) - 1)
        elif compounding_convention == compounding_convention.Simple:
            return (1 / (end_tenors - start_tenors)) * (start_discount_factors / end_discount_factors - 1)

    def get_zero_rates(
            self,
            tenors: float | np.ndarray,
            compounding_convention: CompoundingConvention = CompoundingConvention.NACC) -> np.ndarray:
        """
        Gets the zero rates (NACC by default) for a specified array of tenors.

        :param tenors: Tenors.
        :param compounding_convention: Compounding convention (NACC by default).
        :return: An array of zero rates.
        """
        if compounding_convention == CompoundingConvention.NACC:
            return -1 / tenors * np.log(self.get_discount_factors(tenors))
        if compounding_convention == CompoundingConvention.NACQ:
            return 4 * (self.get_discount_factors(tenors) ** (-1 / (4 * tenors)) - 1)
        elif compounding_convention == CompoundingConvention.Simple:
            return 1 / tenors * (1 / self.get_discount_factors(tenors) - 1)
        else:
            raise ValueError(f'Invalid compounding convention: {compounding_convention}')

    def get_discount_factor_derivatives(self, tenors: float | np.ndarray):
        """
        Gets the derivative values of the discount factors at the given tenors.

        :param tenors: The tenors at which to calculate the derivative.
        :return: The derivative values.
        """
        return (self.get_discount_factors(tenors + self.numerical_derivative_step_size) -
                self.get_discount_factors(tenors - self.numerical_derivative_step_size)) / \
               (2 * self.numerical_derivative_step_size)

    def get_log_discount_factor_derivatives(self, tenors: np.ndarray):
        """
        Gets the derivative values of the log of the discount factors at the given tenors.
        
        :param tenors: The tenors at which to calculate the derivative.
        :return: The derivative values.
        """
        return (np.log(self.get_discount_factors(tenors + self.numerical_derivative_step_size)) -
                np.log(self.get_discount_factors(tenors - self.numerical_derivative_step_size))) / \
               (2 * self.numerical_derivative_step_size)
