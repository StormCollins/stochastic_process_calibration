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
        alpha = 0.1
        sigma = 0.1
        curve_tenors = \
            np.array([
                0.0000000,
                0.0027397,
                0.0821918,
                0.2493151,
                0.4958904,
                0.7479452,
                1.0000000,
                1.2493151,
                1.4958904,
                1.7479452,
                2.0000000,
                2.9972603,
                4.0027397,
                5.0027397,
                6.0027397,
                7.0027397,
                8.0027397,
                9.0000000,
                10.0054795,
                12.0082192,
                15.0027397,
                20.0082192,
                25.0136986,
                30.0191781])

        # TODO: Where did these discount factors come from?
        curve_discount_factors = np.array(
            [1.000000,
             0.999907,
             0.997261,
             0.991717,
             0.983809,
             0.975718,
             0.967524,
             0.959083,
             0.950459,
             0.941230,
             0.931649,
             0.887226,
             0.834895,
             0.776718,
             0.713405,
             0.649354,
             0.585177,
             0.524324,
             0.469244,
             0.372527,
             0.268633,
             0.162742,
             0.104571,
             0.071701])

        initial_curve: Curve = Curve(curve_tenors, curve_discount_factors)
        hw = HullWhite(alpha, sigma, initial_curve, theta_times, 0.25)
        paths = hw.simulate(maturity, number_of_paths=1_000, number_of_time_steps=50)
        plt.show()
        print(paths)

    def test_discount_curve(self):
        curve_tenors = \
            np.array([
                0.000000,
                0.002740,
                0.082192,
                0.249315,
                0.495890,
                0.747945,
                1.000000,
                1.249315,
                1.495890,
                1.747945,
                2.000000,
                2.997260,
                4.002740,
                5.002740,
                6.002740,
                7.002740,
                8.002740,
                9.000000,
                10.005490,
                12.008220,
                15.002740,
                20.008220,
                25.013699,
                30.019178])

        curve_discount_factors = \
            np.array([
                1.000000,
                0.999907,
                0.997261,
                0.991717,
                0.983809,
                0.975718,
                0.967524,
                0.959083,
                0.950459,
                0.941230,
                0.931649,
                0.887226,
                0.834895,
                0.776718,
                0.713405,
                0.649354,
                0.585177,
                0.524324,
                0.469244,
                0.372527,
                0.268633,
                0.162742,
                0.104571,
                0.071701])
        number_of_time_steps = 50
        maturity = 5/12
        dt = maturity/number_of_time_steps
        alpha = 0.05
        sigma = 0.1
        theta_times: np.ndarray = np.arange(0, 30.25, 0.25)
        initial_curve: Curve = Curve(curve_tenors, curve_discount_factors)
        hw = HullWhiteCurve(alpha, sigma, initial_curve, 0.25)
        hw.get_discount_curve(curve_tenors, dt)


