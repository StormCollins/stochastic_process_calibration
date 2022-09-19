import numpy as np

from gbm.analytical_pricers import *
from gbm.gbm_pricers import *
import pytest


class TestsGbmPricers:
    def test_fast_equity_european_option_monte_carlo_pricer(self):
        notional: float = 1_000_000
        initial_spot: float = 50
        strike: float = 52
        interest_rate: float = 0.1
        tenor: float = 0.25
        vol_surface: VolData = read_vol_surface('../excel_file/FEC_atm_vol_surface.xlsx')
        volatility: float = excel_file.excel_vol_surface_function.get_vol(tenor, vol_surface) / 100
        time_to_maturity: float = 6 / 12
        number_of_paths: int = 10_000
        number_of_time_steps: int = 1000
        actual: MonteCarloResult = \
            fast_equity_european_option_monte_carlo_pricer(
                notional,
                initial_spot,
                strike,
                interest_rate,
                volatility,
                time_to_maturity,
                "put",
                number_of_paths,
                number_of_time_steps,
                True,
                True)
        expected_price: float = black_scholes(notional, initial_spot, strike, interest_rate, volatility,
                                              time_to_maturity, "put")
        assert expected_price == pytest.approx(actual.price, actual.error)

    def test_slow_equity_european_option_monte_carlo_pricer(self):
        notional: float = 1_000_000
        initial_spot: float = 50
        strike: float = 52
        interest_rate: float = 0.1
        tenor: float = 0.25
        vol_surface: VolData = read_vol_surface('../excel_file/FEC_atm_vol_surface.xlsx')
        volatility: float = excel_file.excel_vol_surface_function.get_vol(tenor, vol_surface) / 100
        time_to_maturity: float = 6 / 12
        number_of_paths: int = 10_000
        number_of_time_steps: int = 2
        actual: MonteCarloResult = \
            slow_equity_european_option_monte_carlo_pricer(
                notional,
                initial_spot,
                strike,
                interest_rate,
                volatility,
                time_to_maturity,
                "put",
                number_of_paths,
                number_of_time_steps,
                True)
        expected_price: float = black_scholes(notional, initial_spot, strike, interest_rate, volatility,
                                              time_to_maturity, "put")
        assert expected_price == pytest.approx(actual.price, actual.error)

    def test_fx_option_monte_carlo_pricer(self):
        notional: float = 1_000_000
        initial_spot: float = 50
        strike: float = 52
        domestic_interest_rate: float = 0.2
        foreign_interest_rate: float = 0.1
        tenor: float = 0.25
        vol_surface: VolData = read_vol_surface('../excel_file/FEC_atm_vol_surface.xlsx')
        volatility: float = excel_file.excel_vol_surface_function.get_vol(tenor, vol_surface) / 100
        time_to_maturity: float = 5 / 12
        number_of_paths: int = 10_000
        number_of_time_steps: int = 100
        actual: MonteCarloResult = \
            fx_option_monte_carlo_pricer(
                notional,
                initial_spot,
                strike,
                domestic_interest_rate,
                foreign_interest_rate,
                volatility,
                time_to_maturity,
                "put",
                number_of_paths,
                number_of_time_steps,
                True,
                True)
        expected_price: float = \
            garman_kohlhagen(
                notional,
                initial_spot,
                strike,
                domestic_interest_rate,
                foreign_interest_rate,
                volatility,
                time_to_maturity,
                "put")
        assert expected_price == pytest.approx(actual.price, actual.error)

    def test_fx_forward_monte_carlo_pricer(self):
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
        tenor: float = 0.25
        vol_surface: VolData = read_vol_surface('../excel_file/FEC_atm_vol_surface.xlsx')
        volatility: float = excel_file.excel_vol_surface_function.get_vol(tenor, vol_surface) / 100
        time_to_maturity: float = 1
        number_of_paths: int = 10_000
        number_of_time_steps: int = 10
        actual: MonteCarloResult = \
            fx_forward_monte_carlo_pricer(
                notional,
                initial_spot,
                strike,
                domestic_interest_rate,
                foreign_interest_rate,
                volatility,
                time_to_maturity,
                number_of_paths,
                number_of_time_steps,
                True,
                True)
        expected_price: float = \
            fx_forward(
                notional,
                initial_spot,
                strike,
                domestic_interest_rate,
                foreign_interest_rate,
                time_to_maturity)
        assert expected_price == pytest.approx(actual.price, actual.error)

# TODO: SU-LISE - HIERDIE IS DIE UNIT TEST
    def test_generate_time_dependent_volatilities(self):
        number_of_time_steps: int = 52
        time_to_maturity = 5 / 12

        print(generate_time_dependent_volatilities(number_of_time_steps,
                                                   time_to_maturity))
