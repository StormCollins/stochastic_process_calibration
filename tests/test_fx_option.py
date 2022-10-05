import pytest
from src.call_or_put import CallOrPut
from src.instruments.fx_option import FxOption
from src.long_or_short import LongOrShort
from src.monte_carlo_pricing_results import MonteCarloPricingResults


@pytest.fixture
def fx_option():
    notional: float = 1
    initial_spot: float = 50
    strike: float = 52
    domestic_interest_rate: float = 0.2
    foreign_interest_rate: float = 0.1
    volatility: float = 0.4
    time_to_maturity: float = 5 / 12
    put: CallOrPut = CallOrPut.PUT
    long: LongOrShort = LongOrShort.LONG

    return FxOption(
        notional=notional,
        initial_spot=initial_spot,
        strike=strike,
        domestic_interest_rate=domestic_interest_rate,
        foreign_interest_rate=foreign_interest_rate,
        volatility=volatility,
        time_to_maturity=time_to_maturity,
        call_or_put=put,
        long_or_short=long)


def test_get_garman_kohlhagen_price(fx_option):
    actual_price: float = fx_option.get_garman_kohlhagen_price()
    expected_price: float = 4.8620672089551995
    assert actual_price == pytest.approx(expected_price, 0.000000000001)


def test_get_time_independent_monte_carlo_price(fx_option):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 1_000

    actual: MonteCarloPricingResults = \
        fx_option.get_time_independent_monte_carlo_price(number_of_paths, number_of_time_steps, True, True)

    expected_price: float = fx_option.get_garman_kohlhagen_price()

    assert expected_price == pytest.approx(actual.price, actual.error)




