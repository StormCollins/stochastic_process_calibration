from analytical_pricers import *
from gbm_pricers import *
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# European Equity Option
# Comparison of Monte Carlo pricing to Black-Scholes pricing for a European equity option
notional: float = 1_000_000
initial_spot: float = 50
strike: float = 52
interest_rate: float = 0.1
volatility: float = 0.4
time_to_maturity: float = 5 / 12
number_of_paths: int = 10_000
number_of_time_steps: int = 2

slow_price: float =\
      slow_equity_european_option_monte_carlo_pricer(
          notional,
            initial_spot,
            strike,
            interest_rate,
            volatility,
            time_to_maturity,
            "call",
            number_of_paths,
            number_of_time_steps)

fast_price: float = \
    fast_equity_european_option_monte_carlo_pricer(notional, initial_spot, strike, interest_rate, volatility,
                                                   time_to_maturity, "put", number_of_paths, number_of_time_steps,,


print()
print('----------------------------------------------------------------------------------')
print('Equity Option Price Comparison')
print(f'Black-Scholes Price: {black_scholes(notional, initial_spot, strike, interest_rate, volatility, time_to_maturity, "put")}')
print(f'Slow European Option Price: {slow_price}')
print(f'Fast European Option Price: {fast_price}')

# ----------------------------------------------------------------------------------------------------------------------
# FX Forward
# Comparison of Monte Carlo pricing to analytical pricing for an FX Forward
notional: float = 1_000_000
initial_spot: float = 50
strike: float = 52
domestic_interest_rate: float = 0.2
foreign_interest_rate: float = 0.1
volatility: float = 0.1
time_to_maturity: float = 5 / 12
number_of_paths: int = 100_000
number_of_time_steps: int = 2

print()
print('----------------------------------------------------------------------------------')
print('FX Forward Price Comparison')
fx_forward_price: float = fx_forward(notional, initial_spot, strike, domestic_interest_rate, foreign_interest_rate, time_to_maturity)
print(f'Analytical FX Forward Price: '
      f'{fx_forward(notional, initial_spot, strike, domestic_interest_rate, foreign_interest_rate, time_to_maturity)}')
print(f'Monte Carlo FX Forward Price: ' +
      str(fx_forward_monte_carlo_pricer(
            notional,
            initial_spot,
            strike,
            domestic_interest_rate,
            foreign_interest_rate,
            volatility,
            time_to_maturity,
            number_of_paths,
            number_of_time_steps)))
#
# # Convergence comparison
# number_of_paths_list = [50_000 * x for x in range(1, 101)]
# monte_carlo_prices = list()
# for number_of_paths in number_of_paths_list:
#     monte_carlo_prices.append(
#           fx_forward_monte_carlo_pricer(
#             initial_spot,
#             strike,
#             domestic_interest_rate,
#             foreign_interest_rate,
#             volatility,
#             time_to_maturity,
#             number_of_paths,
#             number_of_time_steps))
#
# #plt.title('FX Forward Price Convergence for Monte Carlo')
# fx_forward_convergence_figure = plt.figure()
# plt.plot(number_of_paths_list, monte_carlo_prices, '-o')
# plt.plot([number_of_paths_list[0], number_of_paths_list[-1]], [fx_forward_price, fx_forward_price])



# ----------------------------------------------------------------------------------------------------------------------
# FX Option
# Comparison of Monte Carlo pricing to Garman-Kohlhagen pricing for an FX option
notional: float = 1_000_000
initial_spot: float = 50
strike: float = 52
domestic_interest_rate: float = 0.2
foreign_interest_rate: float = 0.1
volatility: float = 0.4
time_to_maturity: float = 5 / 12
number_of_paths: int = 10_000
number_of_time_steps: int = 2

print()
print('----------------------------------------------------------------------------------')
print('FX Option Price Comparison')
print(f'Black-Scholes (Garman-Kohlhagen) FX option price: ' +
      str(garman_kohlhagen(
            notional,
            initial_spot,
            strike,
            domestic_interest_rate,
            foreign_interest_rate,
            volatility,
            time_to_maturity,
            "put")))

# print(f'Monte Carlo FX option price: ' +
#       str(fx_option_monte_carlo_pricer(
#             initial_spot,
#             strike,
#             domestic_interest_rate,
#             foreign_interest_rate,
#             volatility,
#             time_to_maturity,
#             "put",
#             number_of_paths,
#             number_of_time_steps,
#             True)))


# print()
# print('----------------------------------------------------------------------------------')
# print(f'Comparison of FX Forward Price to FX Option with Zero Vol')

# initial_spot: float = 50
# strike: float = 52
# domestic_interest_rate: float = 0.4
# foreign_interest_rate: float = 0.1
# volatility: float = 0
# time_to_maturity: float = 5 / 12
# number_of_paths: int = 1
# number_of_time_steps: int = 2

# print(f'FX Forward Price: '
#       f'{fx_forward(initial_spot, strike, domestic_interest_rate, foreign_interest_rate, time_to_maturity)}')
# print('Monte Carlo FX option price: ' +
#       str(fx_option_monte_carlo_pricer(
#             initial_spot,
#             strike,
#             domestic_interest_rate,
#             foreign_interest_rate,
#             volatility,
#             time_to_maturity,
#             "call",
#             number_of_paths,
#             number_of_time_steps,
#             False)))

"""
This function returns the estimated Hull-White parameter setup_theta.

:param forward_rate: Continuously compounded forward rate.
:param interest_rate_sim_times: Times to maturity. (I don't really know about this one)
:param alpha: Calibration parameter from Josh Knipe's Jupyter Notebook.
:param sigma: Calibration parameter from Josh Knipe's Jupyter Notebook.
:return: Estimated Hull-White Theta parameter.

"""