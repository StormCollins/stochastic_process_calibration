import numpy as np
import pytest
from src.instruments.irs import Irs


@pytest.fixture
def example_irs():
    return Irs(1_000_000, 0.0, 1.0, 0.25, 0.1)


def test_payment_tenors(example_irs):
    expected: np.ndarray = np.array([0.25, 0.50, 0.75, 1.00])
    actual: np.ndarray = example_irs.payment_tenors
    assert all([a == b for a, b in zip(expected, actual)])


def test_get_par_swap_rate():
    assert 1 == 1
