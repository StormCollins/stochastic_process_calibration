import pytest
from src.instruments.fx_forward import FxForward
from src.monte_carlo_pricing_results import MonteCarloPricingResults


@pytest.fixture
def fx_forward():
    notional: float = 1
    initial_spot: float = 50
    strike: float = 52
    domestic_interest_rate: float = 0.2
    foreign_interest_rate: float = 0.1
    time_to_maturity: float = 5 / 12
    return FxForward(notional, initial_spot, strike, domestic_interest_rate, foreign_interest_rate, time_to_maturity)


def test_fx_forward_get_analytical_price(fx_forward):
    actual_price: float = fx_forward.get_analytical_price()
    expected_price: float = 0.11716329473210145
    assert actual_price == pytest.approx(expected_price, 0.000000000001)


def test_fx_forward_get_time_independent_monte_carlo_pricer():
    """
    Note where these values come from:
    1. xvalite_fec_trade-data_2022-03-31.xlsx
        Strike - Forward Exchange Contracts Sheet
    2. xvalite_fec-and-fx-option_market-data_2022-03-31.xlsx
        initial_spot - FXHistories Sheet
        volatility - ATMFxOptions
    3. xvalite_fec-and-fx-option_discount-curves_2022-03-31.xlsx
        domestic_interest_rate - Discount Curves
        foreign_interest_rate - Discount Curves
    """
    notional: float = 1_000_000
    initial_spot: float = 14.6038
    strike: float = 17
    domestic_interest_rate: float = 0.061339421
    foreign_interest_rate: float = 0.020564138
    volatility: float = 0.149
    time_to_maturity: float = 1
    number_of_paths: int = 10_000
    number_of_time_steps: int = 50

    fec: FxForward = \
        FxForward(notional, initial_spot, strike, domestic_interest_rate, foreign_interest_rate, time_to_maturity)

    actual: MonteCarloPricingResults = \
        fec.get_time_independent_monte_carlo_price(number_of_paths, number_of_time_steps, volatility, True, True)

    print()
    print(f' FX Forward Prices')
    print(f' -----------------')
    print(f'  Monte Carlo Price: {actual.price:,.2f} ± {actual.error:,.2f}')
    expected: float = fec.get_analytical_price()
    print(f'  Analytical: {expected:,.2f}')
    assert expected == pytest.approx(actual.price, actual.error)
