"""
Contains a class for representing an Interest Rate Swap (IRS).
"""
from src.curves.curve import *
from src.enums_and_named_tuples.long_or_short import LongOrShort


class Irs:
    """
    A class for representing Interest Rate Swap (IRS) with identical pay and receive frequencies.
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
        self.long_or_short: LongOrShort = long_or_short
        self.fixed_leg_reset_start_tenors: np.ndarray = np.arange(self.start_tenor, self.end_tenor, self.frequency)
        self.fixed_leg_reset_end_tenors: np.ndarray = \
            np.arange(self.start_tenor, self.end_tenor, self.frequency) + self.frequency
        self.floating_leg_reset_start_tenors: np.ndarray = self.fixed_leg_reset_start_tenors
        self.floating_leg_reset_end_tenors: np.ndarray = self.fixed_leg_reset_end_tenors

        if fixed_rate is not None:
            self.fixed_rate = fixed_rate
        elif fixed_rate is None and curve is not None:
            self.fixed_rate = self.get_par_swap_rate(curve)
        else:
            self.fixed_rate = None

    def get_par_swap_rate(self, curve: Curve) -> float:
        numerator: float = \
            curve.get_discount_factors(self.start_tenor) - curve.get_discount_factors(self.fixed_leg_reset_end_tenors[-1])

        day_count_fractions: np.ndarray = self.fixed_leg_reset_end_tenors - self.fixed_leg_reset_start_tenors
        discount_factors: np.ndarray = np.array([curve.get_discount_factors(t) for t in self.fixed_leg_reset_end_tenors])
        denominator: float = sum([t * df for t, df in zip(day_count_fractions, discount_factors)])
        return numerator / denominator

    def get_fair_value(self, current_time_step: float, curve: Curve) -> float:
        floating_leg_value = self.get_floating_leg_fair_value(current_time_step, curve)
        fixed_leg_value = self.get_fixed_leg_fair_value(current_time_step, curve)

        if self.long_or_short == LongOrShort.LONG:
            return floating_leg_value - fixed_leg_value
        else:
            return fixed_leg_value - floating_leg_value

    def get_floating_leg_fair_value(self, current_time_step: float, curve: Curve) -> float:
        # Calculate the forward rates (whose end tenors are > current_time_step) using curve.
        reset_end_tenors = self.floating_leg_reset_end_tenors[self.floating_leg_reset_end_tenors > current_time_step]
        forward_rates = Curve.get_forward_rates(curve, reset_end_tenors[:-1], reset_end_tenors, CompoundingConvention.NACQ)
        day_count_fractions: np.ndarray = self.floating_leg_reset_end_tenors - self.floating_leg_reset_start_tenors
        # Calculate forward rates * day count fractions * discount factors
        discount_factors: np.ndarray = np.array([curve.get_discount_factors(t) for t in reset_end_tenors])
        floating_leg = self.notional * forward_rates * day_count_fractions * discount_factors
        return floating_leg

    def get_fixed_leg_fair_value(self, current_time_step: float, curve: Curve) -> float:
        # Calculate fixed rate * day count fractions * discount factors
        reset_end_tenors = self.fixed_leg_reset_end_tenors[self.fixed_leg_reset_end_tenors > current_time_step]
        day_count_fractions: np.ndarray = self.fixed_leg_reset_end_tenors - self.fixed_leg_reset_start_tenors
        discount_factors: np.ndarray = np.array([curve.get_discount_factors(t) for t in reset_end_tenors[1:]])
        fixed_leg = self.notional * self.fixed_rate * day_count_fractions * discount_factors
        return fixed_leg

    def get_monte_carlo_fair_values(
            self,):
        return 0