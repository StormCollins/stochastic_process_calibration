import numpy as np
from scipy.interpolate import interp1d


class Curve:
    tenors: np.ndarray
    discount_factors: np.ndarray
    interpolator: interp1d

    def __init__(self, tenors, discount_factors):
        self.tenors = tenors
        self.discount_factors = discount_factors
        self.interpolator = interp1d(tenors, discount_factors, 'cubic')

    def get_discount_factor(self, tenor: float) -> float:
        return self.interpolator(tenor)

    def get_forward_rate(self, start_tenor: float, end_tenor: float) -> float:
        start_discount_factor: float = self.get_discount_factor(start_tenor)
        end_discount_factor: float = self.get_discount_factor(end_tenor)
        return -1 / (end_tenor - start_tenor) * np.log(end_discount_factor / start_discount_factor)

    def get_zero_rate(self, tenor: float) -> float:
        return -1 / tenor * np.log(self.get_discount_factor(tenor))
