import matplotlib.pyplot as plt
import pytest
from hullwhite.hullwhite import *


class TestsHullWhite:
    def test_theta(self):
        theta_times: np.ndarray = np.arange(0, 30.25, 0.25)
        alpha = 2
        sigma = 0.01

    def test_simulate(self):
        theta_times: np.ndarray = np.arange(0, 30.25, 0.25)
        maturity = 5
        number_of_paths = 1_000
        number_of_time_steps = 2
        alpha = 2
        sigma = 0.01
        curve_tenors = np.array([0,
                                 0.00273972602739726,
                                 0.0821917808219178,
                                 0.249315068493151,
                                 0.495890410958904,
                                 0.747945205479452,
                                 1,
                                 1.24931506849315,
                                 1.4958904109589,
                                 1.74794520547945,
                                 2,
                                 2.9972602739726,
                                 4.0027397260274,
                                 5.0027397260274,
                                 6.0027397260274,
                                 7.0027397260274,
                                 8.0027397260274,
                                 9,
                                 10.0054794520548,
                                 12.0082191780822,
                                 15.0027397260274,
                                 20.0082191780822,
                                 25.013698630137,
                                 30.0191780821918])
        # Corresponding swap discount factors
        curve_discount_factors = np.array(
            [1,
             0.999907269575448,
             0.997260675265517,
             0.991717124230684,
             0.983808917763081,
             0.975717678662124,
             0.967523680878883,
             0.959082962515726,
             0.950458527887378,
             0.941229833843046,
             0.93164890952382,
             0.887225968658722,
             0.834895492942358,
             0.776717959564857,
             0.713405382605866,
             0.649354239063653,
             0.585176994918833,
             0.524323960782608,
             0.469243755978326,
             0.372526977084483,
             0.268633004028575,
             0.16274196335192,
             0.104570874377476,
             0.0717007182261533])
        initial_curve: Curve = Curve(curve_tenors, curve_discount_factors)
        hw = HullWhite(alpha, sigma, initial_curve, theta_times, 0.25)
        paths = hw.simulate(maturity, 1000, 50)
        # plot_paths(paths, maturity)
        plt.show()
        print(paths)

    def test_discount_curve(self):
        curve_tenors = np.array([0,
                                 0.00273972602739726,
                                 0.0821917808219178,
                                 0.249315068493151,
                                 0.495890410958904,
                                 0.747945205479452,
                                 1,
                                 1.24931506849315,
                                 1.4958904109589,
                                 1.74794520547945,
                                 2,
                                 2.9972602739726,
                                 4.0027397260274,
                                 5.0027397260274,
                                 6.0027397260274,
                                 7.0027397260274,
                                 8.0027397260274,
                                 9,
                                 10.0054794520548,
                                 12.0082191780822,
                                 15.0027397260274,
                                 20.0082191780822,
                                 25.013698630137,
                                 30.0191780821918])
        curve_discount_factors = np.array(
            [1,
             0.999907269575448,
             0.997260675265517,
             0.991717124230684,
             0.983808917763081,
             0.975717678662124,
             0.967523680878883,
             0.959082962515726,
             0.950458527887378,
             0.941229833843046,
             0.93164890952382,
             0.887225968658722,
             0.834895492942358,
             0.776717959564857,
             0.713405382605866,
             0.649354239063653,
             0.585176994918833,
             0.524323960782608,
             0.469243755978326,
             0.372526977084483,
             0.268633004028575,
             0.16274196335192,
             0.104570874377476,
             0.0717007182261533])
        number_of_time_steps = 50
        maturity = 5/12
        dt = maturity/number_of_time_steps
        alpha = 0.05
        sigma = 0.1
        theta_times: np.ndarray = np.arange(0, 30.25, 0.25)
        initial_curve: Curve = Curve(curve_tenors, curve_discount_factors)
        hw = HullWhiteCurve(alpha, sigma, initial_curve, 0.25)
        hw.get_discount_curve(curve_tenors, dt)

