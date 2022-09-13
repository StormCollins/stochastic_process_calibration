from curves.curve import Curve
import time
from instruments.fra import Fra
from hullwhite.hullwhite import *
import pytest


class TestsFra:
    def test_fair_forward_rate(self):
        tenors = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
        # assumes a constant rate of 10%
        discount_factors = np.array([1.000000, 0.975310, 0.951229, 0.927743, 0.904837])
        curve = Curve(tenors, discount_factors)
        fra = Fra(1, 0.5, 0.75, 1)
        assert 0.10126 == pytest.approx(fra.get_fair_forward_rate(curve), 0.0001)

    def test_get_monte_carlo_value(self):
        start_time = time.time()
        tenors = np.array([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75])
        # We assume a constant rate of 10%
        discount_factors = np.array([1.000000, 0.975310, 0.951229, 0.927743, 0.904837, 0.882497, 0.860708, 0.839457])
        initial_curve = Curve(tenors, discount_factors)
        short_rate_tenor = 0.01
        hw = HullWhite(0.1, 0.2, initial_curve, short_rate_tenor)

        fra = Fra(notional=1_0, forward_rate_start_tenor=1.25, forward_rate_end_tenor=1.50, strike=0.2)#0126)
        initial_fra_value = fra.get_value(curve=initial_curve, current_time=0)
        print()
        print(f'Initial FRA value: {initial_fra_value}')
        fra_value = fra.get_monte_carlo_value(hw, number_of_time_steps=30, number_of_paths=1000)
        print(f'Monte Carlo FRA Value (fast analytical): {fra_value}')
        print(time.time() - start_time)
        # plt.show()
