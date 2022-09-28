from src.gbm.analytical_pricers import *
from src.gbm.gbm_pricers import *
import pytest




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


