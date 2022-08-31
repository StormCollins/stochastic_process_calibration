from curves.curve import *


class Irs:
    start_tenor: float
    payment_tenors: list[float]

    def __init__(self, start_tenor: float, payment_tenors: list[float]):
        self.start_tenor = start_tenor
        self.payment_tenors = payment_tenors

    def calculate_par_rate(self, curve: Curve):
        numerator: float = \
            curve.get_discount_factor(np.array(self.start_tenor)) - curve.get_discount_factor(np.array(
                self.payment_tenors[-1]))

        day_count_fractions = list(np.array(self.payment_tenors[1:]) - np.array(self.payment_tenors[0:-1]))
        day_count_fractions.insert(0, self.payment_tenors[0] - self.start_tenor)
        discount_factors: list[float] = [curve.get_discount_factor(t) for t in self.payment_tenors]
        denominator: float = sum([t * df for t, df in zip(day_count_fractions, discount_factors)])

        return numerator / denominator
