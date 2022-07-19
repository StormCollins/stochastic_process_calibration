from gbm.analytical_pricers import *
from gbm.gbm_pricers import *
import pytest


class TestsGbmPricers:
    def test_fast_equity_european_option_monte_carlo_pricer(self):
        initial_spot: float = 50
        strike: float = 52
        interest_rate: float = 0.1
        volatility: float = 0.4
        time_to_maturity: float = 5 / 12
        number_of_paths: int = 10_000
        number_of_time_steps: int = 2
        actual: MonteCarloResult = \
            fast_equity_european_option_monte_carlo_pricer(
                initial_spot,
                strike,
                interest_rate,
                volatility,
                time_to_maturity,
                "put",
                number_of_paths,
                number_of_time_steps,
                False)
        expected_price: float = black_scholes(initial_spot, strike, interest_rate, volatility, time_to_maturity, "put")
        assert expected_price == pytest.approx(actual.price, actual.error)

    def test_slow_equity_european_option_monte_carlo_pricer(self):
        initial_spot: float = 50
        strike: float = 52
        interest_rate: float = 0.1
        volatility: float = 0.4
        time_to_maturity: float = 5 / 12
        number_of_paths: int = 10_000
        number_of_time_steps: int = 2
        actual: MonteCarloResult = \
            slow_equity_european_option_monte_carlo_pricer(
                initial_spot,
                strike,
                interest_rate,
                volatility,
                time_to_maturity,
                "put",
                number_of_paths,
                number_of_time_steps)
        expected_price: float = black_scholes(initial_spot, strike, interest_rate, volatility, time_to_maturity, "put")
        assert expected_price == pytest.approx(actual.price, actual.error)

    def test_fx_option_monte_carlo_pricer(self):
        initial_spot: float = 50
        strike: float = 52
        domestic_interest_rate: float = 0.2
        foreign_interest_rate: float = 0.1
        volatility: float = 0.1
        time_to_maturity: float = 5 / 12
        number_of_paths: int = 10_000
        number_of_time_steps: int = 2
        actual: MonteCarloResult = \
            fx_option_monte_carlo_pricer(
                initial_spot,
                strike,
                domestic_interest_rate,
                foreign_interest_rate,
                volatility,
                time_to_maturity,
                "put",
                number_of_paths,
                number_of_time_steps,
                False)
        expected_price: float = garman_kohlhagen(
                                                initial_spot,
                                                strike,
                                                domestic_interest_rate,
                                                foreign_interest_rate,
                                                volatility,
                                                time_to_maturity,
                                                "put")
        assert expected_price == pytest.approx(actual.price, actual.error)

    def test_fx_forward_monte_carlo_pricer(self):
        initial_spot: float = 50
        strike: float = 52
        domestic_interest_rate: float = 0.2
        foreign_interest_rate: float = 0.1
        volatility: float = 0.1
        time_to_maturity: float = 5 / 12
        number_of_paths: int = 10_000
        number_of_time_steps: int = 2
        actual: MonteCarloResult = \
            fx_forward_monte_carlo_pricer(
                initial_spot,
                strike,
                domestic_interest_rate,
                foreign_interest_rate,
                volatility,
                time_to_maturity,
                number_of_paths,
                number_of_time_steps,
                False)
        expected_price: float = fx_forward(
                                            initial_spot,
                                            strike,
                                            domestic_interest_rate,
                                            foreign_interest_rate,
                                            time_to_maturity)
        assert expected_price == pytest.approx(actual.price, actual.error)
