from gbm.analytical_pricers import *
import pytest


class TestsAnalyticalPricers:
    def test_black_scholes(self):
        initial_spot: float = 50
        strike: float = 52
        interest_rate: float = 0.1
        volatility: float = 0.4
        time_to_maturity: float = 5 / 12
        actual_price: float = black_scholes(initial_spot, strike, interest_rate, volatility, time_to_maturity, "put")
        expected_price: float = 5.068933121521976
        assert actual_price == pytest.approx(expected_price, 0.000000000001)

# TODO:
# Move analytical pricers.
