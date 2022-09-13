import numpy as np
import pytest
import QuantLib as ql
from curves.curve import *


@pytest.fixture
def curve():
    tenors: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    discount_factors: np.ndarray = np.array([1.00, 0.95, 0.90, 0.85, 0.80])
    return Curve(tenors, discount_factors)


@pytest.fixture
def ql_curve():
    dates = \
        [ql.Date(1, 1, 2022),
         ql.Date(1, 4, 2022),
         ql.Date(30, 6, 2022),
         ql.Date(28, 9, 2022),
         ql.Date(27, 12, 2022)]

    discount_factors: np.ndarray = np.array([1.00, 0.95, 0.90, 0.85, 0.80])
    return ql.DiscountCurve(dates, list(discount_factors), ql.Actual360())


def test_get_single_discount_factor(curve, ql_curve):
    assert curve.get_discount_factors(np.array([0.625]))[0] == ql_curve.discount(ql.Date(14, 8, 2022))


def test_get_multiple_discount_factors(curve, ql_curve):
    actual: list[float] = list(curve.get_discount_factors(np.array([0.625, 0.875])))
    expected: list[float] = [ql_curve.discount(ql.Date(14, 8, 2022)), ql_curve.discount(ql.Date(12, 11, 2022))]
    assert all([a == pytest.approx(b, 0.00001) for a, b in zip(actual, expected)])


def test_get_single_forward_rate(curve, ql_curve):
    assert curve.get_forward_rates(np.array([0.00]), np.array([1.00]), CompoundingConvention.NACC) == \
           pytest.approx(0.22314355131421, 0.0000001)


def test_get_single_zero_rate(curve, ql_curve):
    expected = ql_curve.zeroRate(ql.Date(14, 8, 2022), ql.Actual360(), ql.Continuous).rate()
    assert curve.get_zero_rates(np.array([0.625]))[0] == pytest.approx(expected, 0.0000001)


def test_multiple_discount_curves_get_discount_factors():
    tenors = np.array([0.00, 0.50, 1.00])
    discount_factors: np.ndarray = \
        np.array([[1.00, 0.99, 0.98],
                  [1.00, 0.97, 0.96],
                  [1.00, 0.95, 0.90]])

    curves = Curve(tenors, discount_factors)
    expected: np.ndarray = \
        np.exp(
            ((np.log(discount_factors[:, 1:]) - np.log(discount_factors[:, 0:-1]))/(tenors[1:] - tenors[0:-1])) *
            (np.array([0.25, 0.75]) - tenors[:-1]) +
            np.log(discount_factors[:, 0:-1]))
    actual: np.ndarray = curves.get_discount_factors(np.array([0.25, 0.75]))
    assert np.allclose(expected, actual)
