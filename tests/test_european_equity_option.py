import numpy as np
import pytest
from src.call_or_put import CallOrPut
from src.instruments.european_equity_option import EuropeanEquityOption
from src.long_or_short import LongOrShort
from src.monte_carlo_pricing_results import MonteCarloPricingResults


@pytest.fixture
def option_for_constant_vol_tests() -> EuropeanEquityOption:
    notional: float = 1
    initial_spot: float = 50
    strike: float = 52
    interest_rate: float = 0.1
    volatility: float = 0.4
    time_to_maturity: float = 5 / 12
    call: CallOrPut = CallOrPut.PUT
    long: LongOrShort = LongOrShort.LONG
    return EuropeanEquityOption(notional, initial_spot, strike, interest_rate, volatility, time_to_maturity, call, long)


@pytest.fixture
def option_for_non_constant_vol_tests() -> EuropeanEquityOption:
    notional: float = 1
    initial_spot: float = 50
    strike: float = 52
    interest_rate: float = 0.1
    volatility: float = 0.1545
    time_to_maturity: float = 2
    call: CallOrPut = CallOrPut.PUT
    long: LongOrShort = LongOrShort.LONG
    return EuropeanEquityOption(notional, initial_spot, strike, interest_rate, volatility, time_to_maturity, call, long)


def test_get_black_scholes_price(option_for_constant_vol_tests):
    # TODO: Add reference to Paolo Brandiparte's book page etc.
    actual_price: float = option_for_constant_vol_tests.get_black_scholes_price()
    expected_price: float = 5.068933121521976
    assert actual_price == pytest.approx(expected_price, 0.000000000001)


def test_time_independent_gbm_monte_carlo_pricer(option_for_constant_vol_tests):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 100

    actual: MonteCarloPricingResults = \
        option_for_constant_vol_tests.get_time_independent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            plot_paths=True,
            show_stats=True)

    print()
    print(f' European Equity Option Prices')
    print(f' -----------------------------')
    print(f'  Monte Carlo Price: {actual.price:,.2f} ± {actual.error:,.2f}')
    expected_price: float = option_for_constant_vol_tests.get_black_scholes_price()
    print(f'  Black-Scholes Price: {expected_price:,.2f}')
    assert expected_price == pytest.approx(actual.price, actual.error)


def test_time_dependent_gbm_monte_carlo_pricer_for_constant_vol(option_for_constant_vol_tests):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 50
    excel_file_path = r'../tests/atm-volatility-surface.xlsx'

    actual: MonteCarloPricingResults = \
        option_for_constant_vol_tests.get_time_dependent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            volatility_excel_path=excel_file_path,
            volatility_excel_sheet_name='constant_vol_surface',
            plot_paths=True,
            show_stats=True)

    print()
    print(f' European Equity Option Prices')
    print(f' -----------------------------')
    print(f'  Monte Carlo Price: {actual.price:,.2f} ± {actual.error:,.2f}')
    expected_price: float = option_for_constant_vol_tests.get_black_scholes_price()
    print(f'  Black-Scholes Price: {expected_price:,.2f}')
    assert expected_price == pytest.approx(actual.price, actual.error)


def test_time_dependent_gbm_monte_carlo_pricer(option_for_non_constant_vol_tests):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 100
    excel_file_path = r'../tests/atm-volatility-surface.xlsx'
    np.random.seed(999)

    actual: MonteCarloPricingResults = \
        option_for_non_constant_vol_tests.get_time_dependent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            volatility_excel_path=excel_file_path,
            volatility_excel_sheet_name='vol_surface',
            plot_paths=True,
            show_stats=True)

    print()
    print(f' European Equity Option Prices')
    print(f' -----------------------------')
    print(f'  Monte Carlo Price: {actual.price:,.2f} ± {actual.error:,.2f}')
    expected_price: float = option_for_non_constant_vol_tests.get_black_scholes_price()
    print(f'  Black-Scholes Price: {expected_price:,.2f}')
    assert expected_price == pytest.approx(actual.price, 0.1)
