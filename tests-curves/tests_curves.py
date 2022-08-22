import numpy as np
import pytest
from curves.curve import Curve


class TestCurves:
    def test_get_discount_factor(self):
        tenors: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
        discount_factors: np.ndarray = np.array([1.00, 0.95, 0.90, 0.85, 0.80])
        curve: Curve = Curve(tenors, discount_factors)
        assert curve.get_discount_factor(0.625) == 0.875

    def test_get_forward_rate(self):
        #TODO: Nicole
        tenors: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
        discount_factors: np.ndarray = np.array([1.00, 0.95, 0.90, 0.85, 0.80])
        curve: Curve = Curve(tenors, discount_factors)

    def test_get_zero_rate(self):
        #TODO: Nicole
        tenors: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
        discount_factors: np.ndarray = np.array([1.00, 0.95, 0.90, 0.85, 0.80])
        curve: Curve = Curve(tenors, discount_factors)
