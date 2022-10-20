"""
Unit tests for European equity options.
"""
import numpy as np
import pytest
from src.enums_and_named_tuples.call_or_put import CallOrPut
from src.enums_and_named_tuples.long_or_short import LongOrShort
from src.enums_and_named_tuples.monte_carlo_pricing_results import MonteCarloPricingResults
from src.instruments.european_equity_option import EuropeanEquityOption
from src.utils.console_utils import ConsoleUtils
from tests_config import TestsConfig


@pytest.fixture
def excel_file_path() -> str:
    return r'tests/equity-atm-volatility-surface.xlsx'


@pytest.fixture
def option_for_constant_vol_tests() -> EuropeanEquityOption:
    """
    Option used for testing constant vol simulations.
    """
    notional: float = 1
    initial_spot: float = 50
    strike: float = 52
    interest_rate: float = 0.1
    volatility: float = 0.4  # 0.1545
    time_to_maturity: float = 2
    put: CallOrPut = CallOrPut.PUT
    long: LongOrShort = LongOrShort.LONG
    return EuropeanEquityOption(notional, initial_spot, strike, interest_rate, volatility, time_to_maturity, put, long)


@pytest.fixture
def inputs() -> EuropeanEquityOption:
    """
    Option used for testing constant vol simulations.
    """
    notional: float = 1
    initial_spot: float = 50
    strike: float = 52
    interest_rate: float = 0.1
    volatility: float = 0.1545  # 0.4
    time_to_maturity: float = 2
    put: CallOrPut = CallOrPut.PUT
    long: LongOrShort = LongOrShort.LONG
    return EuropeanEquityOption(notional, initial_spot, strike, interest_rate, volatility, time_to_maturity, put, long)


@pytest.fixture
def option_for_non_constant_vol_tests() -> EuropeanEquityOption:
    """
    Option used for testing non-constant vol simulations.
    """
    notional: float = 1
    initial_spot: float = 50
    strike: float = 52
    interest_rate: float = 0.1
    volatility: float = 0.1545
    time_to_maturity: float = 2
    put: CallOrPut = CallOrPut.PUT
    short: LongOrShort = LongOrShort.SHORT
    return EuropeanEquityOption(notional, initial_spot, strike, interest_rate, volatility, time_to_maturity, put, short)


def test_get_black_scholes_price(option_for_constant_vol_tests):
    """
    Reference: Paolo Brandimarte: 2.6.4 Black-Scholes model in MATLAB, page 113.
    Put BS price: 5.0689
    Call BS price: 5.1911
    """
    actual_price: float = option_for_constant_vol_tests.get_black_scholes_price()
    expected_price: float = 6.9968175753991

    # Call = 14.4228184153440
    # Put = 6.9968175753991
    assert actual_price == pytest.approx(expected_price, 0.000000000001)


def test_time_independent_gbm_monte_carlo_pricer(option_for_constant_vol_tests):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 20

    actual: MonteCarloPricingResults = \
        option_for_constant_vol_tests.get_time_independent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            plot_paths=TestsConfig.plots_on,
            show_stats=True)

    expected_price: float = option_for_constant_vol_tests.get_black_scholes_price()
    ConsoleUtils.print_monte_carlo_pricing_results(
        title='European Equity Option Prices',
        monte_carlo_price=actual.price,
        monte_carlo_price_error=actual.error,
        analytical_price=expected_price)

    assert expected_price == pytest.approx(actual.price, actual.error)


def test_time_dependent_gbm_monte_carlo_pricer_for_constant_vol(option_for_constant_vol_tests, excel_file_path):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 20
    np.random.seed(999)
    actual: MonteCarloPricingResults = \
        option_for_constant_vol_tests.get_time_dependent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            volatility_excel_path=excel_file_path,
            volatility_excel_sheet_name='constant_vol_surface',
            plot_paths=TestsConfig.plots_on,
            show_stats=True)

    expected_price: float = option_for_constant_vol_tests.get_black_scholes_price()
    ConsoleUtils.print_monte_carlo_pricing_results(
        title='European Equity Options Prices',
        monte_carlo_price=actual.price,
        monte_carlo_price_error=actual.error,
        analytical_price=expected_price)

    assert expected_price == pytest.approx(actual.price, abs=actual.error)


def test_time_dependent_gbm_monte_carlo_pricer(option_for_non_constant_vol_tests, excel_file_path):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 20
    np.random.seed(999)

    actual: MonteCarloPricingResults = \
        option_for_non_constant_vol_tests.get_time_dependent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            volatility_excel_path=excel_file_path,
            volatility_excel_sheet_name='vol_surface',
            plot_paths=TestsConfig.plots_on,
            show_stats=True)

    expected_price: float = option_for_non_constant_vol_tests.get_black_scholes_price()
    ConsoleUtils.print_monte_carlo_pricing_results(
        title='European Equity Option Prices',
        monte_carlo_price=actual.price,
        monte_carlo_price_error=actual.error,
        analytical_price=expected_price)

    assert expected_price == pytest.approx(actual.price, 0.1)


def test_time_independent_vs_time_dependent_price(inputs, excel_file_path):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 20
    np.random.seed(999)

    time_dependent: MonteCarloPricingResults = \
        inputs.get_time_dependent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            volatility_excel_path=excel_file_path,
            volatility_excel_sheet_name='vol_surface',
            plot_paths=TestsConfig.plots_on,
            show_stats=True)

    np.random.seed(999)

    time_independent: MonteCarloPricingResults = \
        inputs.get_time_independent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            plot_paths=TestsConfig.plots_on,
            show_stats=True)

    print('\n\n')
    print(f'Time-dependent Monte Carlo price: {time_dependent.price:,.2f} ± {time_dependent.error:,.2f}')
    print(f'Time-independent Monte Carlo price: {time_independent.price:,.2f} ± {time_independent.error:,.2f}')
    assert time_independent.price == pytest.approx(time_dependent.price, 0.1)
