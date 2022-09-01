from curves.curve import Curve
from instruments.fra import Fra
from hullwhite.hullwhite import *
import pytest


class TestsFra:
    def test_fair_forward_rate(self):
        tenors = [0.00, 0.25, 0.50, 0.75, 1.00]
        # assumes a constant rate of 10%
        discount_factors = [1.000000, 0.975310, 0.951229, 0.927743, 0.904837]
        curve = Curve(tenors, discount_factors)
        fra = Fra(0.5, 0.75, 1)
        assert 0.10126 == pytest.approx(fra.get_fair_forward_rate(curve), 0.0001)

    def test_get_monte_carlo_value(self):
        tenors = np.array([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
        # assumes a constant rate of 10%
        discount_factors = np.array([1.000000, 0.975310, 0.951229, 0.927743, 0.904837, 0.882497, 0.860708])
        initial_curve = Curve(tenors, discount_factors)
        theta_times = np.array(tenors)
        short_rate_tenor = 0.25
        hw = HullWhite(0.1, 0.1, initial_curve, theta_times, short_rate_tenor)

        fra = Fra(1_000_000, 1.00, 1.25, 0.1)
        fra.get_monte_carlo_value(hw)
