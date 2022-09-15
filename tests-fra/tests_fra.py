import time
from instruments.fra import *
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
        number_time_steps = 100
        short_rate_tenor = 1.25 / (number_time_steps + 1)
        hw = HullWhite(0.1, 0.2, initial_curve, short_rate_tenor)
        forward_rate_start_tenor = 1.25  # 1.25
        forward_rate_end_tenor = 1.5  # 1.50
        fra = Fra(1_000_000, forward_rate_start_tenor, forward_rate_end_tenor, strike=0.2)
        fra_future_value = \
            fra.get_values(curve=initial_curve, current_time=0, valuation_type=ValuationType.FUTUREVALUE)
        print()
        print(f'Analytical FRA future value: {fra_future_value}')
        fra_value = \
            fra.get_monte_carlo_value(
                hw,
                number_of_time_steps=number_time_steps,
                number_of_paths=10_000,
                valuation_type=ValuationType.FUTUREVALUE,
                method=SimulationMethod.SLOWANALYTICAL)
        print(f'Monte Carlo FRA future value: {fra_value}')
        print(f'Time taken: {time.time() - start_time}s')
        # plt.show()
