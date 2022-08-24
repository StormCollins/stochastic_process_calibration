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
        self.tenors = tenors
        self.discount_factors = discount_factors
        self.discount_factor_interpolator = interp1d(tenors, discount_factors, 'cubic')

    def get_discount_factor(self, tenor: np.ndarray) -> np.ndarray:
        return self.discount_factor_interpolator(tenor)

    def get_forward_discount_factor(self, start_tenor: float, end_tenor: float) -> float:
        """
        Calculates the discount factor between times start_tenor and end_tenor.

        :param start_tenor: The time we want to discount back to.
        :param end_tenor: The time we want to discount back from.
        :return: The forward discount factor.
        """
        return self.discount_factor_interpolator(end_tenor) / self.discount_factor_interpolator(start_tenor)

    def get_forward_rate(
            self,
            start_tenor: float,
            end_tenor: float,
            compounding_convention: CompoundingConvention) -> float:
        start_discount_factor: float = self.get_discount_factor(np.array(start_tenor))[0]
        end_discount_factor: float = self.get_discount_factor(np.array(end_tenor))[0]
        if compounding_convention == compounding_convention.NACC:
            return -1 / (end_tenor - start_tenor) * np.log(end_discount_factor / start_discount_factor)
        elif compounding_convention == compounding_convention.Simple:
            return 1 / (end_tenor - start_tenor) * (start_discount_factor / end_discount_factor - 1)

    def get_zero_rate(self, tenor: float) -> float:
        return -1 / tenor * np.log(self.get_discount_factor(tenor)[0])
