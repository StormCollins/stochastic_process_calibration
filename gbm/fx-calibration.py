from gbm.gbm_pricers import *

notional: float = 1_000_000
initial_spot: float = 14.6
strike: float = 17
domestic_interest_rate: float = 0.05737
foreign_interest_rate: float = 0.01227
volatility: float = 0.154
time_to_maturity: float = 0.5
number_of_paths: int = 100_000
number_of_time_steps: int = 2
result = fx_option_monte_carlo_pricer(
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
            False)

print(f'Monte Carlo FX Option Price: {result.price:,.2f}')

