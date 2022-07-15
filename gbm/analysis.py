from analytical_pricers import *
from gbm_pricers import *

# TEST THE FUNCTIONS

# ----------------------------------------------------------------------------------------------------------------------
# Compare Monte Carlo to Black-Scholes for equity
initial_spot: float = 50
strike: float = 52
interest_rate: float = 0.1
volatility: float = 0.4
time_to_maturity: float = 5 / 12
number_of_paths: int = 10_000
number_of_time_steps: int = 2
slow_price = slow_equity_european_option_monte_carlo_pricer(initial_spot, strike, interest_rate, volatility,
                                                            time_to_maturity, "put", number_of_paths,
                                                            number_of_time_steps)
fast_price = fast_equity_european_option_monte_carlo_pricer(initial_spot, strike, interest_rate, volatility,
                                                            time_to_maturity, "put", number_of_paths,
                                                            number_of_time_steps)
print()
print('----------------------------------------------------------------------------------')
print('Equity Option Price Comparison')
print(f'Black-Scholes Price: {black_scholes(initial_spot, strike, interest_rate, volatility, time_to_maturity, "put")}')
print(f'Slow European Option Price: {slow_price}')
print(f'Fast European Option Price: {fast_price}')

# ----------------------------------------------------------------------------------------------------------------------
# Analytical FX Forward
initial_spot: float = 50
domestic_interest_rate: float = 0.2
foreign_interest_rate: float = 0.1
time_of_contract: float = 5 / 12
print()
print('----------------------------------------------------------------------------------')
print('FX Forward Price Comparison')
print(f'Analytical FX Forward Price: {fx_forward(initial_spot, domestic_interest_rate, foreign_interest_rate, time_of_contract)}')

# ----------------------------------------------------------------------------------------------------------------------
# FX Option
initial_spot: float = 50
strike: float = 52
domestic_interest_rate: float = 0.4
foreign_interest_rate: float = 0.1
volatility: float = 0.4
time_to_maturity: float = 5 / 12
number_of_paths: int = 10_000
number_of_time_steps: int = 2

print()
print('----------------------------------------------------------------------------------')
print('FX Option Price Comparison')
print(f'Black Scholes (Garman-Kohlhagen) FX option price: '
      f'{garman_kohlhagen(initial_spot, strike, domestic_interest_rate, foreign_interest_rate, volatility, time_of_contract, "put")}')
print(f'Monte Carlo FX option price: '
      f'{fx_option_monte_carlo_pricer(initial_spot, strike, domestic_interest_rate, foreign_interest_rate, volatility, time_of_contract, "put", number_of_paths, number_of_time_steps, True)}')

print()
print('----------------------------------------------------------------------------------')
print(f'FX Forward Price:'
      f'{fx_forward_monte_carlo_pricer(initial_spot,strike,domestic_interest_rate,foreign_interest_rate,volatility,time_to_maturity,number_of_paths,number_of_time_steps)}')

print()
print('----------------------------------------------------------------------------------')
print(f'Ensure GBM pricer for FX option gives same price as FX forward if vol is zero')
print(f'FX Forward Price:'
      f'{fx_forward_monte_carlo_pricer(initial_spot,strike,domestic_interest_rate,foreign_interest_rate,volatility,time_to_maturity,number_of_paths,number_of_time_steps)}')

initial_spot: float = 50
strike: float = 52
domestic_interest_rate: float = 0.4
foreign_interest_rate: float = 0.1
volatility: float = 0
time_to_maturity: float = 5 / 12
number_of_paths: int = 10_000
number_of_time_steps: int = 2

print(f'Monte Carlo FX option price: '
      f'{fx_option_monte_carlo_pricer(initial_spot, strike, domestic_interest_rate, foreign_interest_rate, volatility, time_of_contract, "put", number_of_paths, number_of_time_steps, True)}')






