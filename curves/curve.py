import numpy as np
from scipy.interpolate import interp1d
from enum import Enum


class CompoundingConvention(Enum):
    NACC = 1
    Simple = 2


class Curve:
    tenors: np.ndarray
    discount_factors: np.ndarray
    discount_factor_interpolator: interp1d

    def __init__(self, tenors: np.ndarray, discount_factors: np.ndarray):
        """
        Curve constructor. Uses cubic-spline interpolation.

        :param tenors: Tenors.
        :param discount_factors: Discount factors.
        """
        self.tenors = tenors
        self.discount_factors = discount_factors
        self.discount_factor_interpolator = interp1d(tenors, discount_factors, 'cubic')

    def get_discount_factors(self, tenors: np.ndarray) -> np.ndarray:
        """
        Gets discount factors for a given set of tenors.

        :param tenors: Tenors.
        :return: An array of discount factors.
        """
        return self.discount_factor_interpolator(np.array(tenors))

    def get_forward_discount_factor(self, start_tenors: np.ndarray, end_tenors: np.ndarray) -> np.ndarray:
        """
        Calculates the discount factor between times start_tenor and end_tenor.

        :param start_tenors: The tenor(s) we want to discount back to.
        :param end_tenors: The tenor(s) we want to discount back from.
        :return: An array of forward discount factor(s).
        """
        return self.discount_factor_interpolator(np.array(end_tenors)) / \
            self.discount_factor_interpolator(np.array(start_tenors))

    def get_forward_rates(
            self,
            start_tenors: np.ndarray,
            end_tenors: np.ndarray,
            compounding_convention: CompoundingConvention) -> float:
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
            return -1 / (end_tenors - start_tenors) * np.log(end_discount_factors / start_discount_factors)
        elif compounding_convention == compounding_convention.Simple:
            return 1 / (end_tenors - start_tenors) * (start_discount_factors / end_discount_factors - 1)

    def get_zero_rates(
            self,
            tenors: np.ndarray,
            compounding_convention: CompoundingConvention = CompoundingConvention.NACC) -> np.ndarray:
        """
        Gets the zero rates (NACC by default) for a specified array of tenors.

        :param tenors: Tenors.
        :param compounding_convention: Compounding convention (NACC by default).
        :return: An array of zero rates.
        """
        if compounding_convention == CompoundingConvention.Simple:
            return 1 / tenors * (1 / self.get_discount_factors(tenors) - 1)
        elif compounding_convention == CompoundingConvention.NACC:
            return -1 / tenors * np.log(self.get_discount_factors(tenors))
        else:
            raise ValueError(f'Invalid compounding convention: {compounding_convention}')
