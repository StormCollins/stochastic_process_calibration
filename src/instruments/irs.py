"""
Contains a class for representing Interest Rate Swaps (IRS).
"""
import numpy as np
from src.curves.curve import *
from src.long_or_short import LongOrShort


class Irs:
    """
    A class for representing Interest Rate Swaps (IRS) with identical pay and receive frequencies.
    """
    def __init__(
            self,
            notional: float,
            start_tenor: float,
            end_tenor: float,
            frequency: float,
            fixed_rate: float = None,
            curve: Curve = None):
        """
        Class constructor.

        :param notional: Notional.
        :param start_tenor: Start tenor. For a forwarding starting swap > 0.
        :param end_tenor: End tenor.
        :param frequency: The frequency of the pay and receive tenors, e.g., 3m = 0.25.
        :param fixed_rate: (Optional) Fixed rate. If set to 'None' and curve is provided then the par swap rate is
            calculated and used as the fixed rate.
        :param curve: (Optional) Discount curve. Used to calculate the par swap rate if the fixed_rate is not provided.
        """
        self.notional = notional
        self.start_tenor: float = start_tenor
        self.end_tenor: float = end_tenor
        self.frequency: float = frequency
        self.payment_tenors: np.ndarray = np.arange(self.start_tenor, self.end_tenor, self.frequency) + self.frequency
        self.receive_tenors: np.ndarray = self.payment_tenors

    def get_par_swap_rate(self, curve: Curve):
        numerator: float = \
            curve.get_discount_factors(self.start_tenor) - curve.get_discount_factors(self.payment_tenors[-1])

        day_count_fractions: np.ndarray = np.array(self.payment_tenors[1:]) - np.array(self.payment_tenors[0:-1])
        np.insert(day_count_fractions, 0, self.payment_tenors[0] - self.start_tenor)
        discount_factors: np.ndarray = np.array([curve.get_discount_factors(t) for t in self.payment_tenors])
        denominator: float = sum([t * df for t, df in zip(day_count_fractions, discount_factors)])
        return numerator / denominator

