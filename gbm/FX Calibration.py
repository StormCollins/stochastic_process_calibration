from gbm.gbm_pricers import *

notional: float = 50_000_000
initial_spot: float = 14.6
strike: float = 17.4
domestic_interest_rate: float = 0.065377245
foreign_interest_rate: float = 0.02566
volatility: float = 0.1595
time_to_maturity: float = 2.5
number_of_paths: int = 10_000
number_of_time_steps: int = 2

print(f'Monte Carlo FX Option Price: ' +
      str(fx_option_monte_carlo_pricer(
            notional,
            initial_spot,
            strike,
            domestic_interest_rate,
            foreign_interest_rate,
            volatility,
            time_to_maturity,
            "call",
            number_of_paths,
            number_of_time_steps,
            False)))

