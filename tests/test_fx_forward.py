"""
FX Forward unit tests.
"""
import numpy as np
import pytest
from src.enums_and_named_tuples.long_or_short import LongOrShort
from src.enums_and_named_tuples.monte_carlo_pricing_results import MonteCarloPricingResults
from src.instruments.fx_forward import FxForward
from src.utils.console_utils import ConsoleUtils
from test_config import TestsConfig
from test_utils import file_and_test_annotation


@pytest.fixture
def excel_file_path() -> str:
    """
    File path to the ATM (at-the-money) volatility term structure.
    """
    return r'tests/fec-atm-volatility-surface.xlsx'


@pytest.fixture
def fx_forward_constant_vol() -> FxForward:
    """
    FX forward for constant vol testing.
    """
    notional: float = 1
    initial_spot: float = 50
    strike: float = 52
    domestic_interest_rate: float = 0.2
    foreign_interest_rate: float = 0.1
    time_to_maturity: float = 5 / 12
    long: LongOrShort = LongOrShort.LONG
    return FxForward(
        notional=notional,
        initial_spot=initial_spot,
        strike=strike,
        domestic_interest_rate=domestic_interest_rate,
        foreign_interest_rate=foreign_interest_rate,
        time_to_maturity=time_to_maturity,
        long_or_short=long)


@pytest.fixture
def fx_forward_non_constant_vol() -> FxForward:
    """
    FX forward for non-constant vol testing.
    """
    notional: float = 1
    initial_spot: float = 50
    strike: float = 52
    domestic_interest_rate: float = 0.2
    foreign_interest_rate: float = 0.1
    time_to_maturity: float = 5 / 12
    long: LongOrShort = LongOrShort.LONG
    return FxForward(
        notional=notional,
        initial_spot=initial_spot,
        strike=strike,
        domestic_interest_rate=domestic_interest_rate,
        foreign_interest_rate=foreign_interest_rate,
        time_to_maturity=time_to_maturity,
        long_or_short=long)


@pytest.fixture
def fx_forward_xvalite():
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
    time_to_maturity: float = 1
    long: LongOrShort = LongOrShort.LONG
    return FxForward(
        notional=notional,
        initial_spot=initial_spot,
        strike=strike,
        domestic_interest_rate=domestic_interest_rate,
        foreign_interest_rate=foreign_interest_rate,
        time_to_maturity=time_to_maturity,
        long_or_short=long)


def test_fx_forward_get_analytical_price(fx_forward_constant_vol):
    actual_price: float = fx_forward_constant_vol.get_analytical_price()
    expected_price: float = 0.11716329473210145
    assert actual_price == pytest.approx(expected_price, 0.000000000001)


def test_fx_forward_get_time_independent_monte_carlo_pricer_constant_vol(fx_forward_xvalite):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 20
    volatility: float = 0.154
    np.random.seed(999)
    actual: MonteCarloPricingResults = \
        fx_forward_xvalite.get_time_independent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            volatility=volatility,
            plot_paths=TestsConfig.plots_on,
            show_stats=True,
            additional_annotation_for_plot=file_and_test_annotation())

    expected: float = fx_forward_xvalite.get_analytical_price()
    ConsoleUtils.print_monte_carlo_pricing_results(
        title='FX Forward Prices',
        monte_carlo_price=actual.price,
        monte_carlo_price_error=actual.error,
        analytical_price=expected)

    assert expected == pytest.approx(actual.price, abs=actual.error)


def test_fx_forward_get_time_dependent_monte_carlo_pricer_constant_vol(fx_forward_constant_vol, excel_file_path):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 20
    np.random.seed(999)
    actual: MonteCarloPricingResults = \
        fx_forward_constant_vol.get_time_dependent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            volatility_excel_path=excel_file_path,
            volatility_excel_sheet_name='vol_surface',
            plot_paths=TestsConfig.plots_on,
            show_stats=True,
            additional_annotation_for_plot=file_and_test_annotation())

    expected: float = fx_forward_constant_vol.get_analytical_price()
    ConsoleUtils.print_monte_carlo_pricing_results(
        title='FX Forward Prices',
        monte_carlo_price=actual.price,
        monte_carlo_price_error=actual.error,
        analytical_price=expected)

    assert expected == pytest.approx(actual.price, abs=actual.error)


def test_fx_forward_get_time_dependent_monte_carlo_pricer_non_constant_vol(fx_forward_non_constant_vol, excel_file_path):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 20
    np.random.seed(999)
    actual: MonteCarloPricingResults = \
        fx_forward_non_constant_vol.get_time_dependent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            volatility_excel_path=excel_file_path,
            volatility_excel_sheet_name='vol_surface',
            plot_paths=TestsConfig.plots_on,
            show_stats=True,
            additional_annotation_for_plot=file_and_test_annotation())

    expected: float = fx_forward_non_constant_vol.get_analytical_price()
    ConsoleUtils.print_monte_carlo_pricing_results(
        title='FX Forward Prices',
        monte_carlo_price=actual.price,
        monte_carlo_price_error=actual.error,
        analytical_price=expected)

    assert expected == pytest.approx(actual.price, abs=actual.error)


def test_xvalite_fx_forward_get_time_dependent_monte_carlo_pricer(fx_forward_xvalite, excel_file_path):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 20
    np.random.seed(999)
    actual: MonteCarloPricingResults = \
        fx_forward_xvalite.get_time_dependent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            volatility_excel_path=excel_file_path,
            volatility_excel_sheet_name='vol_surface',
            plot_paths=TestsConfig.plots_on,
            show_stats=True,
            additional_annotation_for_plot=file_and_test_annotation())

    expected: float = fx_forward_xvalite.get_analytical_price()
    ConsoleUtils.print_monte_carlo_pricing_results(
        title='FX Forward Prices',
        monte_carlo_price=actual.price,
        monte_carlo_price_error=actual.error,
        analytical_price=expected)

    assert expected == pytest.approx(actual.price, abs=actual.error)
