from src.gbm.analytical_pricers import *
import pytest


def test_garman_kohlhagen(self):
    notional: float = 1
    initial_spot: float = 50
    strike: float = 52
    domestic_interest_rate: float = 0.2
    foreign_interest_rate: float = 0.1
    volatility: float = 0.4
    time_to_maturity: float = 5 / 12
    actual_price: float = \
        garman_kohlhagen(
            notional,
            initial_spot,
            strike,
            domestic_interest_rate,
            foreign_interest_rate,
            volatility,
            time_to_maturity,
            "put")
    expected_price: float = 4.8620672089551995
    assert actual_price == pytest.approx(expected_price, 0.000000000001)
