from curves.curve import *


class Fra:
    """
    Used to encapsulate a FRA (forward rate agreement) with all it's parameters and functions.

    """
    forward_rate_start_tenor: float
    forward_rate_end_tenor: float
    strike: float

    def __init__(self, forward_rate_start_tenor, forward_rate_end_tenor, strike):
        self.forward_rate_start_tenor = forward_rate_start_tenor
        self.forward_rate_end_tenor = forward_rate_end_tenor
        self.strike = strike

    def calculate_fair_forward_rate(self, curve: Curve):
        return curve.get_forward_rate(
            start_tenor=self.forward_rate_start_tenor,
            end_tenor=self.forward_rate_end_tenor,
            compounding_convention=CompoundingConvention.Simple)