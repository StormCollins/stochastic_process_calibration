from gbm_pricers import *

notional: float = 1_000_000
initial_spot: float = 14.6038
strike: float = 17
domestic_interest_rate: float = 0.061339421
foreign_interest_rate: float = 0.020564138
time_to_maturity: float = 1.01
number_of_paths: int = 100_000
number_of_time_steps: int = 100
volatility: float = 0.154
excel_file_path: str = r'C:\GitLab\stochastic_process_calibration_2022\gbm\FEC_atm_vol_surface.xlsx'
sheet_name: str = 'constant_vol_surface'
np.random.seed(999)
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
          volatility,
          excel_file_path,
          sheet_name,
          False,
          False)))
