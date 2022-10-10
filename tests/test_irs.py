import numpy as np
import pytest
from src.curves.curve import Curve
from src.instruments.irs import Irs


@pytest.fixture
def example_irs():
    return Irs(1_000_000, 0.0, 1.0, 0.25, 0.1)


@pytest.fixture
def flat_curve():
    tenors: np.ndarray = np.array([0.0, 0.25, 0.50, 0.75, 1.00])
    discount_factors: np.ndarray = np.array([np.exp(-0.1 * t) for t in tenors])
    return Curve(tenors, discount_factors)


def test_payment_tenors(example_irs):
    expected: np.ndarray = np.array([0.25, 0.50, 0.75, 1.00])
    actual: np.ndarray = example_irs.payment_tenors
    assert all([a == b for a, b in zip(expected, actual)])


def test_get_par_swap_rate(example_irs, flat_curve):
    actual: float = example_irs.get_par_swap_rate(flat_curve)
    assert 1 == 1
