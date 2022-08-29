from curves.curve import Curve
from instruments.fra import Fra
import pytest


class TestsFra:
    def test_fair_forward_rate(self):
        tenors = [0.00, 0.25, 0.50, 0.75, 1.00]
        discount_factors = [1.000000, 0.975310, 0.951229, 0.927743, 0.904837]
        curve = Curve(tenors, discount_factors)
        fra = Fra(0.5, 0.75, 1)
        assert 0.10126 == pytest.approx(fra.calculate_fair_forward_rate(curve), 0.0001)
