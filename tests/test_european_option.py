import pytest
from src.call_or_put import CallOrPut
from src.instruments.european_equity_option import EuropeanEquityOption
from src.long_or_short import LongOrShort
from src.monte_carlo_results import MonteCarloResults


@pytest.fixture
def european_equity_option():
    notional: float = 1
    initial_spot: float = 50
    strike: float = 52
    interest_rate: float = 0.1
    volatility: float = 0.4
    time_to_maturity: float = 5 / 12
    call: CallOrPut = CallOrPut.PUT
    long: LongOrShort = LongOrShort.LONG
    return EuropeanEquityOption(notional, initial_spot, strike, interest_rate, volatility, time_to_maturity, call, long)


def test_get_black_scholes_price(european_equity_option):
    actual_price: float = european_equity_option.get_black_scholes_price()
    expected_price: float = 5.068933121521976
    assert actual_price == pytest.approx(expected_price, 0.000000000001)


def test_fast_equity_european_option_monte_carlo_pricer(european_equity_option):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 1000

    actual: MonteCarloResults = \
        european_equity_option.get_time_independent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            plot_paths=True,
            show_stats=True)

    print()
    print(f' European Equity Option Prices')
    print(f' -----------------------------')
    print(f'  Monte Carlo Price: {actual.price:,.2f} ± {actual.error:,.2f}')
    expected_price: float = european_equity_option.get_black_scholes_price()
    print(f'  Black-Scholes Price: {expected_price:,.2f}')
    assert expected_price == pytest.approx(actual.price, actual.error)
