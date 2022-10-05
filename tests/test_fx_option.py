import pytest
import numpy as np
from src.call_or_put import CallOrPut
from src.instruments.fx_option import FxOption
from src.long_or_short import LongOrShort
from src.monte_carlo_results import MonteCarloResults


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
    actual_price: float = fx_option_constant_vol.get_garman_kohlhagen_price()
    expected_price: float = 4.8620672089551995
    assert actual_price == pytest.approx(expected_price, 0.000000000001)


def test_get_time_independent_monte_carlo_price_constant_vol(fx_option_constant_vol):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 1_000

    actual: MonteCarloResults = \
        fx_option_constant_vol.get_time_independent_monte_carlo_price(number_of_paths, number_of_time_steps, True, True)

    expected_price: float = fx_option_constant_vol.get_garman_kohlhagen_price()

    assert expected_price == pytest.approx(actual.price, actual.error)


def test_time_dependent_gbm_monte_carlo_pricer_constant_vol(fx_option_constant_vol):
    number_of_paths: int = 10_000
    number_of_time_steps: int = 1_000
    excel_file_path = r'tests/atm-volatility-surface.xlsx'
    np.random.seed(999)

    actual: MonteCarloResults = \
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
    number_of_time_steps = 1000
    excel_file_path: str = r'tests/atm-volatility-surface.xlsx'

    np.random.seed(999)

    actual: MonteCarloResults = \
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
