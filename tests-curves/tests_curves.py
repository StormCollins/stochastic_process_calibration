import numpy as np
import pytest
import QuantLib as ql
from curves.curve import *


class TestCurves:
    #TODO: Setup tests initially.
    def test_get_single_discount_factor(self):
        tenors: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
        discount_factors: np.ndarray = np.array([1.00, 0.95, 0.90, 0.85, 0.80])
        curve: Curve = Curve(tenors, discount_factors)
        dates = \
            [ql.Date(1, 1, 2022),
             ql.Date(1, 4, 2022),
             ql.Date(30, 6, 2022),
             ql.Date(28, 9, 2022),
             ql.Date(27, 12, 2022)]

        ql_curve = ql.DiscountCurve(dates, list(discount_factors), ql.Actual360())
        assert curve.get_discount_factors(np.array([0.625]))[0] == ql_curve.discount(ql.Date(14, 8, 2022))

    def test_get_multiple_discount_factors(self):
        tenors: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
        discount_factors: np.ndarray = np.array([1.00, 0.95, 0.90, 0.85, 0.80])
        curve: Curve = Curve(tenors, discount_factors)
        actual: list[float] = list(curve.get_discount_factors(np.array([0.625, 0.875])))

        dates = \
            [ql.Date(1, 1, 2022),
             ql.Date(1, 4, 2022),
             ql.Date(30, 6, 2022),
             ql.Date(28, 9, 2022),
             ql.Date(27, 12, 2022)]

        ql_curve = ql.DiscountCurve(dates, list(discount_factors), ql.Actual360())
        expected: list[float] = [ql_curve.discount(ql.Date(14, 8, 2022)), ql_curve.discount(ql.Date(12, 11, 2022))]
        assert all([a == pytest.approx(b, 0.00001) for a, b in zip(actual, expected)])

    def test_get_single_forward_rate(self):
        tenors: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
        discount_factors: np.ndarray = np.array([1.00, 0.95, 0.90, 0.85, 0.80])
        curve: Curve = Curve(tenors, discount_factors)
        assert curve.get_forward_rates(np.array([0.00]), np.array([1.00]), CompoundingConvention.NACC) == \
               pytest.approx(0.22314355131421, 0.0000001)

    def test_get_single_zero_rate(self):
        tenors: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
        discount_factors: np.ndarray = np.array([1.00, 0.95, 0.90, 0.85, 0.80])
        curve: Curve = Curve(tenors, discount_factors)
        dates = \
            [ql.Date(1, 1, 2022),
             ql.Date(1, 4, 2022),
             ql.Date(30, 6, 2022),
             ql.Date(28, 9, 2022),
             ql.Date(27, 12, 2022)]

        ql_curve = ql.DiscountCurve(dates, list(discount_factors), ql.Actual360())
        expected = ql_curve.zeroRate(ql.Date(14, 8, 2022), ql.Actual360(), ql.Continuous).rate()
        assert curve.get_zero_rates(np.array([0.625]))[0] == pytest.approx(expected, 0.0000001)
