"""
FX option unit tests.
"""
import inspect
import numpy as np
import os
import QuantLib as ql
import pytest
from src.enums_and_named_tuples.call_or_put import CallOrPut
from src.enums_and_named_tuples.long_or_short import LongOrShort
from src.enums_and_named_tuples.monte_carlo_pricing_results import MonteCarloPricingResults
from src.instruments.fx_option import FxOption
from src.utils.console_utils import ConsoleUtils
from test_config import TestsConfig
from test_utils import file_and_test_annotation


@pytest.fixture
def fx_option_constant_vol() -> FxOption:
    """
    FX option for constant vol testing.
    """
    notional: float = 1_000_000
    initial_spot: float = 14.6
    strike: float = 17
    domestic_interest_rate: float = 0.05737
    foreign_interest_rate: float = 0.01227
    volatility: float = 0.154
    time_to_maturity: float = 0.5
    call: CallOrPut = CallOrPut.CALL
    long: LongOrShort = LongOrShort.LONG

    return FxOption(
        notional=notional,
        initial_spot=initial_spot,
        strike=strike,
        domestic_interest_rate=domestic_interest_rate,
        foreign_interest_rate=foreign_interest_rate,
        volatility=volatility,
        time_to_maturity=time_to_maturity,
        call_or_put=call,
        long_or_short=long)


@pytest.fixture
def fx_option_non_constant_vol() -> FxOption:
    """
        Note where these values come from:
        These values are for a USD/ZAR call option.
       1. xvalite_fx-option_trade-data_2022-03-31.xlsx
           Strike - Fx Options Sheet (Trade ID: 571494)
       2. xvalite_fx-option_market-data_2022-03-31.xlsx
           initial_spot - FX Histories Sheet
           volatility - FX Histories Sheet
           domestic_interest_rate - Discount Curves
           foreign_interest_rate - Discount Curves
    """
    notional: float = 1_000_000
    initial_spot: float = 14.6
    strike: float = 17
    domestic_interest_rate: float = 0.05737
    foreign_interest_rate: float = 0.01227
    time_to_maturity: float = 0.5
    volatility: float = 0.154
    put: CallOrPut = CallOrPut.PUT
    call: CallOrPut = CallOrPut.CALL
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


def test_get_garman_kohlhagen_price(fx_option_constant_vol):
    """
    We used QuantLib to validate the Garman-Kohlhagen pricing.
    See https://quant.stackexchange.com/questions/70258/quantlib-greeks-of-fx-option-in-python for an implementation of
    an FX option using QuantLib.
    """
    actual_price: float = fx_option_constant_vol.get_garman_kohlhagen_price()

    # QuantLib pricing.
    valuation_date: ql.Date = ql.Date(3, 1, 2022)
    calendar: ql.Calendar = ql.NullCalendar()
    period: ql.Period = ql.Period(f'{fx_option_constant_vol.time_to_maturity * 360}d')
    expiry_date: ql.Date = calendar.advance(valuation_date, period)
    ql.Settings.instance().evaluationDate = valuation_date
    calendar: ql.Calendar = ql.NullCalendar()
    day_count_convention: ql.DayCounter = ql.Actual360()
    domestic_rate_handle = ql.QuoteHandle(ql.SimpleQuote(fx_option_constant_vol.domestic_interest_rate))
    domestic_curve = \
        ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, domestic_rate_handle, day_count_convention))

    foreign_rate_handle = ql.QuoteHandle(ql.SimpleQuote(fx_option_constant_vol.foreign_interest_rate))
    foreign_curve = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, foreign_rate_handle, day_count_convention))
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, fx_option_constant_vol.strike)
    exercise = ql.EuropeanExercise(expiry_date)
    option = ql.VanillaOption(payoff, exercise)
    vol_handle = ql.QuoteHandle(ql.SimpleQuote(fx_option_constant_vol.volatility))
    volatility = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(valuation_date, calendar, vol_handle, day_count_convention))

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(fx_option_constant_vol.initial_spot))
    gk_process = ql.GarmanKohlagenProcess(spot_handle, foreign_curve, domestic_curve, volatility)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(gk_process))
    expected_price: float = fx_option_constant_vol.notional * option.NPV()
    assert actual_price == pytest.approx(expected_price, 0.000001)


def test_get_time_independent_monte_carlo_price_constant_vol(fx_option_constant_vol):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 100
    np.random.seed(999)
    actual: MonteCarloPricingResults = \
        fx_option_constant_vol.get_time_independent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            plot_paths=TestsConfig.plots_on,
            show_stats=False,
            additional_annotation_for_plot=file_and_test_annotation())

    np.random.seed(999)
    expected_price: float = fx_option_constant_vol.get_garman_kohlhagen_price()
    ConsoleUtils.print_monte_carlo_pricing_results(
        title='FX Option Prices',
        monte_carlo_price=actual.price,
        monte_carlo_price_error=actual.error,
        analytical_price=expected_price)

    assert expected_price == pytest.approx(actual.price, abs=actual.error)


@pytest.mark.skip(reason='Changed volatility from 0.4 to 0.154. Test only works for 0.4,'
                         'since the constant vol-surface is 0.4.')
def test_time_dependent_gbm_monte_carlo_pricer_constant_vol(fx_option_constant_vol):
    number_of_paths: int = 100_000
    number_of_time_steps: int = 100
    excel_file_path = r'tests/equity-atm-volatility-surface.xlsx'
    np.random.seed(999)
    actual: MonteCarloPricingResults = \
        fx_option_constant_vol.get_time_dependent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            volatility_excel_path=excel_file_path,
            volatility_excel_sheet_name='constant_vol_surface',
            plot_paths=TestsConfig.plots_on,
            additional_annotation_for_plot=file_and_test_annotation(),
            show_stats=True)

    expected_price: float = fx_option_constant_vol.get_garman_kohlhagen_price()

    ConsoleUtils.print_monte_carlo_pricing_results(
        title='FX Option Prices',
        monte_carlo_price=actual.price,
        monte_carlo_price_error=actual.error,
        analytical_price=expected_price)

    assert expected_price == pytest.approx(actual.price, abs=actual.error)


def test_fx_option_get_time_dependent_monte_carlo_pricer_non_constant_vol(fx_option_non_constant_vol):
    number_of_paths = 10_000
    number_of_time_steps = 20
    excel_file_path: str = r'tests/fx-option-atm-volatility-surface.xlsx'
    np.random.seed(999)
    additional_annotation: str = \
        f'File: {os.path.basename(__file__)}\n' \
        f'Test: {inspect.currentframe().f_code.co_name}' \
        if TestsConfig.show_test_location \
        else None

    actual: MonteCarloPricingResults = \
        fx_option_non_constant_vol.get_time_dependent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            volatility_excel_path=excel_file_path,
            volatility_excel_sheet_name='vol_surface',
            plot_paths=TestsConfig.plots_on,
            additional_annotation_for_plot=file_and_test_annotation(),
            show_stats=True)

    expected_price: float = fx_option_non_constant_vol.get_garman_kohlhagen_price()

    ConsoleUtils.print_monte_carlo_pricing_results(
        title='FX Option Prices',
        monte_carlo_price=actual.price,
        monte_carlo_price_error=actual.error,
        analytical_price=expected_price)

    assert expected_price == pytest.approx(actual.price, abs=actual.error)
