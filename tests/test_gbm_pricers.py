from src.gbm.analytical_pricers import *
from src.gbm.gbm_pricers import *
import pytest


def test_fast_equity_european_option_monte_carlo_pricer():
    notional: float = 1_000_000
    initial_spot: float = 50
    strike: float = 52
    interest_rate: float = 0.149
    volatility: float = 0.1
    time_to_maturity: float = 6 / 12
    number_of_paths: int = 10_000
    number_of_time_steps: int = 1000
    excel_file_path: str = r'atm-volatility-surface.xlsx'
    sheet_name: str = 'constant_vol_surface'

    actual: MonteCarloResult = \
        fast_equity_european_option_monte_carlo_pricer(
            notional=notional,
            initial_spot=initial_spot,
            strike=strike,
            interest_rate=interest_rate,
            volatility=volatility,
            time_to_maturity=time_to_maturity,
            call_or_put=CallOrPut.CALL,
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            excel_file_path=excel_file_path,
            sheet_name=sheet_name,
            plot_paths=True,
            show_stats=True)

    print(f'Monte Carlo price: {actual}')
    expected: float = \
        black_scholes(notional, initial_spot, strike, interest_rate, volatility, time_to_maturity, "call")

    print(f'Black-Scholes price: {expected}')
    assert expected == pytest.approx(actual.price, actual.error)


def test_slow_equity_european_option_monte_carlo_pricer():
    notional: float = 1_000_000
    initial_spot: float = 50
    strike: float = 52
    interest_rate: float = 0.1
    volatility: float = 0.1
    time_to_maturity: float = 6 / 12
    number_of_paths: int = 100_000
    number_of_time_steps: int = 50

    actual: MonteCarloResult = \
        slow_equity_european_option_monte_carlo_pricer(
            notional=notional,
            initial_spot=initial_spot,
            strike=strike,
            interest_rate=interest_rate,
            volatility=volatility,
            time_to_maturity=time_to_maturity,
            call_or_put=CallOrPut.PUT,
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            show_stats=True)

    expected: float = \
        black_scholes(notional, initial_spot, strike, interest_rate, volatility, time_to_maturity, CallOrPut.PUT)

    assert expected == pytest.approx(actual.price, actual.error)


def test_fx_option_monte_carlo_pricer():
    notional: float = 1_000_000
    initial_spot: float = 50
    strike: float = 52
    domestic_interest_rate: float = 0.06
    foreign_interest_rate: float = 0.02
    volatility: float = 0.149
    time_to_maturity: float = 6 / 12
    number_of_paths: int = 10_000
    number_of_time_steps: int = 1000
    excel_file_path: str = r'atm-volatility-surface.xlsx'
    sheet_name: str = 'constant_vol_surface'

    actual: MonteCarloResult = \
        fx_option_monte_carlo_pricer(
            notional=notional,
            initial_spot=initial_spot,
            strike=strike,
            domestic_interest_rate=domestic_interest_rate,
            foreign_interest_rate=foreign_interest_rate,
            volatility=volatility,
            time_to_maturity=time_to_maturity,
            call_or_put="put",
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            excel_file_path=excel_file_path,
            sheet_name=sheet_name,
            plot_paths=True,
            show_stats=True)

    expected: float = \
        garman_kohlhagen(
            notional=notional,
            initial_spot=initial_spot,
            strike=strike,
            domestic_interest_rate=domestic_interest_rate,
            foreign_interest_rate=foreign_interest_rate,
            volatility=volatility,
            time_to_maturity=time_to_maturity,
            call_or_put="put")
    assert expected == pytest.approx(actual.price, actual.error)


def test_fx_forward_monte_carlo_pricer():
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
    number_of_time_steps: int = 1000
    excel_file_path: str = r'atm-volatility-surface.xlsx'
    sheet_name: str = 'constant_vol_surface'

    actual: MonteCarloResult = \
        fx_forward_monte_carlo_pricer(
            notional=notional,
            initial_spot=initial_spot,
            strike=strike,
            domestic_interest_rate=domestic_interest_rate,
            foreign_interest_rate=foreign_interest_rate,
            time_to_maturity=time_to_maturity,
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            volatility=volatility,
            excel_file_path=excel_file_path,
            sheet_name=sheet_name,
            plot_paths=True,
            show_stats=True)

    print(f'FX Forward Monte Carlo Price: {actual}')

    expected: float = \
        fx_forward(
            notional=notional,
            initial_spot=initial_spot,
            strike=strike,
            domestic_interest_rate=domestic_interest_rate,
            foreign_interest_rate=foreign_interest_rate,
            time_to_maturity=time_to_maturity)

    print(f'FX Forward Analytical: {expected}')
    assert expected == pytest.approx(actual.price, actual.error)

