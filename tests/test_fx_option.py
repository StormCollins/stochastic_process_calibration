import pytest
import numpy as np
import QuantLib as ql
from src.call_or_put import CallOrPut
from src.instruments.fx_option import FxOption
from src.long_or_short import LongOrShort
from src.monte_carlo_pricing_results import MonteCarloPricingResults


@pytest.fixture
def fx_option_constant_vol():
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


@pytest.fixture
def fx_option_non_constant_vol():
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
    notional: float = 1
    initial_spot: float = 14.6
    strike: float = 17
    domestic_interest_rate: float = 0.05737
    foreign_interest_rate: float = 0.01227
    time_to_maturity: float = 0.5
    volatility: float = 0.154
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


def test_get_garman_kohlhagen_price(fx_option_constant_vol):
    """
    We used QuantLib to validate the Garman-Kohlhagen pricing.
    See https://quant.stackexchange.com/questions/70258/quantlib-greeks-of-fx-option-in-python for an implementation of
    an FX option using QuantLib.
    """
    actual_price: float = fx_option_constant_vol.get_garman_kohlhagen_price()
    valuation_date: ql.Date = ql.Date(3, 1, 2022)
    expiry_date: ql.Date = ql.Date(2, 6, 2022)
    ql.Settings.instance().evaluationDate = valuation_date
    calendar: ql.Calendar = ql.NullCalendar()
    day_count_convention: ql.DayCounter = ql.Actual360()
    domestic_rate_handle = ql.QuoteHandle(ql.SimpleQuote(fx_option_constant_vol.domestic_interest_rate))
    domestic_curve = \
        ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, domestic_rate_handle, day_count_convention))

    foreign_rate_handle = ql.QuoteHandle(ql.SimpleQuote(fx_option_constant_vol.foreign_interest_rate))
    foreign_curve = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, foreign_rate_handle, day_count_convention))
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, fx_option_constant_vol.strike)
    exercise = ql.EuropeanExercise(expiry_date)
    option = ql.VanillaOption(payoff, exercise)
    vol_handle = ql.QuoteHandle(ql.SimpleQuote(fx_option_constant_vol.volatility))
    volatility = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(valuation_date, calendar, vol_handle, day_count_convention))
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(fx_option_constant_vol.initial_spot))
    gk_process = ql.GarmanKohlagenProcess(spot_handle, foreign_curve, domestic_curve, volatility)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(gk_process))
    expected_price: float = option.NPV()
    assert actual_price == pytest.approx(expected_price, 0.000001)


def test_get_time_independent_monte_carlo_price_constant_vol(fx_option_constant_vol):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 50
    actual: MonteCarloPricingResults = \
        fx_option_constant_vol.get_time_independent_monte_carlo_price(number_of_paths, number_of_time_steps, True, True)

    expected_price: float = fx_option_constant_vol.get_garman_kohlhagen_price()
    assert expected_price == pytest.approx(actual.price, actual.error)


def test_time_dependent_gbm_monte_carlo_pricer_constant_vol(fx_option_constant_vol):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 50
    excel_file_path = r'tests/atm-volatility-surface.xlsx'
    np.random.seed(999)
    actual: MonteCarloPricingResults = \
        fx_option_constant_vol.get_time_dependent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            volatility_excel_path=excel_file_path,
            volatility_excel_sheet_name='constant_vol_surface',
            plot_paths=True,
            show_stats=True)

    print()
    print(f' FX Option Prices')
    print(f' -----------------------------')
    print(f'  Monte Carlo Price: {actual.price:,.2f} ± {actual.error:,.2f}')
    expected_price: float = fx_option_constant_vol.get_garman_kohlhagen_price()
    print(f'  Garman-Kohlhagen Price: {expected_price:,.2f}')
    assert expected_price == pytest.approx(actual.price, actual.error)


def test_fx_option_get_time_dependent_monte_carlo_pricer_non_constant_vol(fx_option_non_constant_vol):
    number_of_paths = 10_000
    number_of_time_steps = 50
    excel_file_path: str = r'tests/atm-volatility-surface.xlsx'
    np.random.seed(999)
    actual: MonteCarloPricingResults = \
        fx_option_non_constant_vol.get_time_dependent_monte_carlo_price(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            volatility_excel_path=excel_file_path,
            volatility_excel_sheet_name='vol_surface',
            plot_paths=True,
            show_stats=True)

    print()
    print(f' FX Option Prices')
    print(f' -----------------------------')
    print(f'  Monte Carlo Price: {actual.price:,.2f} ± {actual.error:,.2f}')
    expected_price: float = fx_option_non_constant_vol.get_garman_kohlhagen_price()
    print(f'  Garman-Kohlhagen Price: {expected_price:,.2f}')
    assert expected_price == pytest.approx(actual.price, actual.error)
