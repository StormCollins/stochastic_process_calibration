import pytest
from hullwhite.hullwhite import *


class TestsHullWhite:
    def test_theta(self):
        theta_times: np.ndarray = np.arange(0, 30.25, 0.25)
        alpha = 0.05
        sigma = 0.01

    def test_simulate(self):
        maturity = 5/12
        number_of_paths = 1_000
        number_of_time_steps = 2
        alpha = 0.05
        sigma = 0.01
