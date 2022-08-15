import math
import numpy as np
from scipy.stats import norm
import matplotlib as cm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple
from gbm.gbm_pricers import *

notional: float = 1_000_000
initial_spot: float = 14.6038
strike: float = 17
domestic_interest_rate: float = 0.061339421
foreign_interest_rate: float = 0.020564138
volatility: float = 0.154
time_to_maturity: float = 1
number_of_paths: int = 10_000
number_of_time_steps: int = 2

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
