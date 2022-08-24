from curves.curve import Curve
from instruments.irs import Irs
import pytest


class TestsIrs:
    def test_par_rate(self):
        tenors = [0.00, 0.25, 0.50, 0.75, 1.00]
        discount_factors = [1.000000, 0.975310, 0.951229, 0.927743, 0.904837]
        curve = Curve(tenors, discount_factors)
        irs = Irs(0.00, [0.25, 0.50, 0.75, 1.00])
        assert 0.10126 == pytest.approx(irs.calculate_par_rate(curve), 0.00001)
