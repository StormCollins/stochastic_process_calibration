from src.curves.curve import *


class Irs:
    def __init__(self, start_tenor: float, payment_tenors: np.ndarray):
        self.start_tenor: float = start_tenor
        self.payment_tenors: np.ndarray = payment_tenors

    def calculate_par_rate(self, curve: Curve):
        numerator: float = \
            curve.get_discount_factors(self.start_tenor) - curve.get_discount_factors(self.payment_tenors[-1])

        day_count_fractions: np.ndarray = np.array(self.payment_tenors[1:]) - np.array(self.payment_tenors[0:-1])
        np.insert(day_count_fractions, 0, self.payment_tenors[0] - self.start_tenor)
        discount_factors: np.ndarray = np.array([curve.get_discount_factors(t) for t in self.payment_tenors])
        denominator: float = sum([t * df for t, df in zip(day_count_fractions, discount_factors)])
        return numerator / denominator

