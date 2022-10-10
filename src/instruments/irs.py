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
            long_or_short: LongOrShort,
            fixed_rate: float = None,
            curve: Curve = None):
        """
        Class constructor.

        :param notional: Notional.
        :param start_tenor: Start tenor. For a forwarding starting swap > 0.
        :param end_tenor: End tenor.
        :param frequency: The frequency of the pay and receive tenors, e.g., 3m = 0.25.
        :param long_or_short: The IRS is 'long' if we pay fixed and receive floating and vice versa for 'short'.
        :param fixed_rate: (Optional) Fixed rate. If set to 'None' and curve is not 'None', then the par swap rate is
            calculated and used as the fixed rate.
        :param curve: (Optional) Discount curve. Used to calculate the par swap rate if the fixed_rate is 'None'.
        """
        self.notional = notional
        self.start_tenor: float = start_tenor
        self.end_tenor: float = end_tenor
        self.frequency: float = frequency
        self.payment_tenors: np.ndarray = np.arange(self.start_tenor, self.end_tenor, self.frequency) + self.frequency
        self.receive_tenors: np.ndarray = self.payment_tenors

        if fixed_rate is not None:
            self.fixed_rate = fixed_rate
        elif fixed_rate is None and curve is not None:
            self.fixed_rate = self.get_par_swap_rate(curve)
        else:
            self.fixed_rate = None

    def get_par_swap_rate(self, curve: Curve) -> float:
        numerator: float = \
            curve.get_discount_factors(self.start_tenor) - curve.get_discount_factors(self.payment_tenors[-1])

        day_count_fractions: np.ndarray = self.payment_tenors - np.insert(self.start_tenor, 1, self.payment_tenors[:-1])
        np.insert(day_count_fractions, 0, self.payment_tenors[0] - self.start_tenor)
        discount_factors: np.ndarray = np.array([curve.get_discount_factors(t) for t in self.payment_tenors])
        denominator: float = sum([t * df for t, df in zip(day_count_fractions, discount_factors)])
        return numerator / denominator

    def get_fair_value(self, current_time_step: float, curve: Curve) -> float:
        # Calculate the floating leg fair value at current_time_step, using curve.
        # Calculate the fixed leg fair value at current_time_step, using curve.
        return 0

    def get_floating_leg_fair_value(self, current_time_step: float, curve: Curve) -> float:
        # Calculate the forward rates (whose end tenors are > current_time_step) using curve.
        # Calculate forward rates * day count fractions * discount factors
        # return the above.
        return 0

    def get_fixed_leg_fair_value(self, current_time_step: float, curve: Curve) -> float:
        # Calculate fixed rate * day count fractions * discount factors
        # return the above.
        return 0
