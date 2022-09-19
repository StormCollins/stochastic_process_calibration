from gbm.gbm_pricers import *


notional: float = 1_000_000
initial_spot: float = 14.6038
strike: float = 17
domestic_interest_rate: float = 0.061339421
foreign_interest_rate: float = 0.020564138
# tenor: float = 1
# vol_surface: VolData = read_vol_surface('../excel_file/FEC_atm_vol_surface.xlsx')
# volatility: float = excel_file.excel_vol_surface_function.get_vol(tenor, vol_surface)/100
time_to_maturity: float = 0.8
number_of_paths: int = 10_000
number_of_time_steps: int = 50
volatility = generate_time_dependent_volatilities(number_of_time_steps, time_to_maturity)

print(f'Monte Carlo FX Forward Price: ' +
      str(fx_forward_monte_carlo_pricer(
          notional,
          initial_spot,
          strike,
          domestic_interest_rate,
          foreign_interest_rate,
          time_to_maturity,
          number_of_paths,
          number_of_time_steps,
          volatility)))
