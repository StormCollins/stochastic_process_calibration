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
slow_price, slow_time = slow_european_option_monte_carlo_pricer(initial_spot, strike, interest_rate, volatility,
                                                                time_to_maturity, "put", number_of_paths,
                                                                number_of_time_steps)
fast_price, fast_time = fast_european_option_monte_carlo_pricer(initial_spot, strike, interest_rate, volatility,
                                                                time_to_maturity, "put", number_of_paths,
                                                                number_of_time_steps)
print()
print('----------------------------------------------------------------------------------')
print('Equity Option Price Comparison')
print(f'Black-Scholes Price: {black_scholes(initial_spot, strike, interest_rate, volatility, time_to_maturity, "put")}')
print(f'Slow European Option Price: {slow_price} (time taken {slow_time})')
print(f'Fast European Option Price: {fast_price} (time taken {fast_time})')

# print(f'\nSlow Monte Carlo / Fast Monte Carlo: {slow_time / fast_time}')


# ----------------------------------------------------------------------------------------------------------------------
# FX Forward
initial_spot: float = 50
rd: float = 0.1
rf: float = 0
time_of_contract: float = 5/12
print()
print('----------------------------------------------------------------------------------')
print('FX Forward Price Comparison')
print(f'FX Forward Price: {fx_forward(initial_spot, rd, rf, time_of_contract)}')

# ----------------------------------------------------------------------------------------------------------------------
# FX Option
initial_spot: float = 50
strike: float = 52
rd: float = 0.2
rf: float = 0.1
volatility: float = 0.4
time_to_maturity: float = 5 / 12
number_of_paths: int = 10
number_of_time_steps: int = 50

print()
print('----------------------------------------------------------------------------------')
print('FX Option Price Comparison')
print(f'Black Scholes FX option price: '
      f'{ black_scholes_FX(initial_spot, strike, rd, rf, volatility, time_of_contract, "put")}')
print(f'Monte Carlo FX option price: '
      f'{FX_option_monte_carlo_pricer(initial_spot, strike, rd, rf, volatility, time_of_contract, "put", number_of_paths, number_of_time_steps, True)}')
