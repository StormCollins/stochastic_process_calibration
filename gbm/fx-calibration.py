from gbm.gbm_pricers import *

notional: float = 1_000_000
initial_spot: float = 14.6
strike: float = 17
domestic_interest_rate: float = 0.05737
foreign_interest_rate: float = 0.01227
tenor: float = 0.5
vol_surface: VolData = read_vol_surface('../excel_file/FX_option_atm_vol_surface.xlsx')
volatility: float = excel_file.excel_vol_surface_function.get_vol(tenor, vol_surface) / 100
time_to_maturity: float = 0.5
number_of_paths: int = 10_000
number_of_time_steps: int = 50
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
    number_of_time_steps)

print(f'Monte Carlo FX Option Price: {result.price:,.2f}')
