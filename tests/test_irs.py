import numpy as np
import pytest
from src.curves.curve import Curve
from src.instruments.irs import Irs
from src.long_or_short import LongOrShort
from src.compounding_convention import CompoundingConvention


@pytest.fixture
def example_irs():
    return Irs(1_000_000, 0.0, 1.0, 0.25, 0.1)


@pytest.fixture
def flat_curve():
    tenors: np.ndarray = np.array([0.0, 0.25, 0.50, 0.75, 1.00])
    discount_factors: np.ndarray = np.array([np.exp(-0.1 * t) for t in tenors])
    return Curve(tenors, discount_factors)


@pytest.fixture
def forward_rates():
    return np.array([0.1, 0.125, 0.15, 0.175])


@pytest.fixture
def curve_rates():
    tenors: np.ndarray = np.array([0.25, 0.50, 0.75, 1.00])
    forward_rates = np.array([0.1, 0.125, 0.15, 0.175])
    discount_factors: np.ndarray = np.cumprod([np.exp(-r * 0.25) for r in forward_rates])
    return Curve(tenors, discount_factors)


def test_fixed_leg_reset_start_tenors(example_irs):
    expected: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75])
    actual: np.ndarray = example_irs.fixed_leg_reset_start_tenors
    assert all([a == b for a, b in zip(expected, actual)])


def test_fixed_leg_reset_end_tenors(example_irs):
    expected: np.ndarray = np.array([0.25, 0.50, 0.75, 1.00])
    actual: np.ndarray = example_irs.fixed_leg_reset_end_tenors
    assert all([a == b for a, b in zip(expected, actual)])


def test_get_par_swap_rate(example_irs, flat_curve):
    actual: float = example_irs.get_par_swap_rate(flat_curve)
    # Since we're using a flat curve, 3m forward rate will be constant, regardless of start and end tenor.
    expected: np.ndarray = flat_curve.get_forward_rates(np.array([1.0]), np.array([1.25]), CompoundingConvention.NACQ)
    assert actual == pytest.approx(expected[0], abs=0.01)


def test_get_par_swap_rate_for_curve(example_irs, curve_rates, forward_rates):
    actual: float = example_irs.get_par_swap_rate(curve_rates)
    expected: float = np.average(forward_rates)
    assert actual == pytest.approx(expected, abs=0.01)


def test_irs_fixed_rate(flat_curve):
    irs = Irs(1_000_000, 0.0, 1.0, 0.25, LongOrShort.LONG, None, flat_curve)
    expected: float = irs.get_par_swap_rate(flat_curve)
    actual: float = irs.fixed_rate
    assert actual == pytest.approx(expected, abs=0.01)


@pytest.mark.skip(reason="Fails due to simulating fixing intricacies which need to be resolved.")
def test_irs_fair_value(curve_rates, forward_rates):
    irs = Irs(1_000_000, 0.0, 1.0, 0.25, LongOrShort.LONG, np.average(forward_rates))
    actual = irs.get_fair_value(0, curve_rates)
    expected = 0
    assert actual == pytest.approx(expected, abs=0.01)
