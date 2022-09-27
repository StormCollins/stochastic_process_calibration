from gbm.gbm_pricers import *

notional: float = 1_000_000
initial_spot: float = 14.6
strike: float = 17.11
domestic_interest_rate: float = 0.05737
foreign_interest_rate: float = 0.01227
time_to_maturity: float = 0.5
number_of_paths: int = 10_000
number_of_time_steps: int = 50
volatility: float = 0.154
excel_file_path: str = r'C:\GitLab\stochastic_process_calibration_2022\gbm\FX_option_atm_vol_surface.xlsx'
sheet_name: str = 'constant_vol_surface'
np.random.seed(999)

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
    excel_file_path,
    sheet_name)

print(f'Monte Carlo FX Option Price: {result.price:,.2f}')
