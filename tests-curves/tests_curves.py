import pytest
import QuantLib as ql
from curves.curve import *


@pytest.fixture
def single_curve():
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


@pytest.fixture
def multiple_curve_tenors():
    return np.array([0.00, 0.50, 1.00])


@pytest.fixture
def multiple_curve_discount_factors():
    return np.array([[1.00, 0.99, 0.98],
                     [1.00, 0.97, 0.96],
                     [1.00, 0.95, 0.90]])


@pytest.fixture
def multiple_curves(multiple_curve_tenors, multiple_curve_discount_factors):
    return Curve(multiple_curve_tenors, multiple_curve_discount_factors)


@pytest.fixture
def curve_tenors():
    return np.array([0.00, 0.25, 0.50, 0.75, 1.00])


@pytest.fixture
def flat_curve(curve_tenors):
    rate = 0.1
    discount_factors: np.ndarray(np.dtype(float)) = np.array([np.exp(-rate * t) for t in curve_tenors])
    return Curve(curve_tenors, discount_factors)


def test_get_single_discount_factor(single_curve, ql_curve):
    assert single_curve.get_discount_factors(np.array([0.625]))[0] == ql_curve.discount(ql.Date(14, 8, 2022))


def test_get_multiple_discount_factors(single_curve, ql_curve):
    actual: list[float] = list(single_curve.get_discount_factors(np.array([0.625, 0.875])))
    expected: list[float] = [ql_curve.discount(ql.Date(14, 8, 2022)), ql_curve.discount(ql.Date(12, 11, 2022))]
    assert all([a == pytest.approx(b, 0.00001) for a, b in zip(actual, expected)])


def test_get_single_forward_rate(single_curve, ql_curve):
    assert single_curve.get_forward_rates(np.array([0.00]), np.array([1.00]), CompoundingConvention.NACC) == \
           pytest.approx(0.22314355131421, 0.0000001)


def test_get_flat_curve_forward_rate(flat_curve):
    assert flat_curve.get_forward_rates(np.array([0.00]), np.array([1.00]), CompoundingConvention.NACC) == \
           pytest.approx(0.10, abs=0.01)
    assert flat_curve.get_forward_rates(np.array([0.00]), np.array([0.01]), CompoundingConvention.NACC) == \
           pytest.approx(0.10, abs=0.01)


def test_get_single_zero_rate(single_curve, ql_curve):
    expected = ql_curve.zeroRate(ql.Date(14, 8, 2022), ql.Actual360(), ql.Continuous).rate()
    assert single_curve.get_zero_rates(np.array([0.625]))[0] == pytest.approx(expected, 0.0000001)


def test_multiple_discount_curves_get_discount_factors(
        multiple_curve_tenors,
        multiple_curve_discount_factors,
        multiple_curves):
    t = multiple_curve_tenors
    dfs = multiple_curve_discount_factors
    expected: np.ndarray = \
        np.exp(
            ((np.log(dfs[:, 1:]) - np.log(dfs[:, 0:-1]))/(t[1:] - t[0:-1])) *
            (np.array([0.25, 0.75]) - t[:-1]) +
            np.log(dfs[:, 0:-1]))
    actual: np.ndarray = multiple_curves.get_discount_factors(np.array([0.25, 0.75]))
    assert np.allclose(expected, actual)


def test_get_discount_factor_derivatives(flat_curve):
    actual = flat_curve.get_discount_factor_derivatives(0.25)
    assert -0.1 * np.exp(-0.1 * 0.25) == pytest.approx(actual, 0.0001)


def test_get_log_discount_factor_derivatives(flat_curve):
    actual = flat_curve.get_log_discount_factor_derivatives(0.25)
    assert -0.1 == pytest.approx(actual, 0.0001)
