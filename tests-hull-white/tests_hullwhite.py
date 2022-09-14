# TODO: Setup more tests for theta with 'real' interest rate curves.

import matplotlib.pyplot as plt
import numpy as np
import pytest
from hullwhite.hullwhite import *


@pytest.fixture
def curve_tenors():
    return np.array([0.00, 0.25, 0.50, 0.75, 1.00])


@pytest.fixture
def flat_curve(curve_tenors):
    rate = 0.1
    discount_factors: np.ndarray(np.dtype(float)) = np.array([np.exp(-rate * t) for t in curve_tenors])
    return Curve(curve_tenors, discount_factors)


def test_theta_with_constant_zero_rates_and_zero_vol(flat_curve):
    alpha = 2
    sigma = 0
    hw = HullWhite(alpha, sigma, initial_curve=flat_curve, short_rate_tenor=0.25)
    actual: list[float] = [hw.theta(t) for t in [0.25, 0.375, 0.5, 0.625]]
    expected: list[float] = list(np.repeat(0.2, len(actual)))
    assert all([a == pytest.approx(b, 0.00001) for a, b in zip(actual, expected)])


def test_theta_with_constant_zero_rates(flat_curve, curve_tenors):
    alpha = 2
    sigma = 0.1
    hw: HullWhite = HullWhite(alpha, sigma, initial_curve=flat_curve, short_rate_tenor=0.25)
    test_tenors: list[float] = [0.25, 0.375, 0.5, 0.625]
    actual: list[float] = [hw.theta(t) for t in test_tenors]
    expected: list[float] = [alpha * 0.1 + (sigma ** 2) / (2 * alpha) * (1 - np.exp(-2 * alpha * t)) for t in
                             test_tenors]
    assert all([a == pytest.approx(b, 0.001) for a, b in zip(actual, expected)])


def test_b_function_large_alpha(flat_curve):
    alpha: float = 10_000
    sigma: float = 0
    hw: HullWhite = HullWhite(alpha, sigma, initial_curve=flat_curve, short_rate_tenor=0.25)
    actual = hw.b_function(np.array([0.25]), 0.00)[0]
    assert actual == pytest.approx(0.0, abs=0.0001)


def test_a_function_with_large_alpha_and_flat_curve(flat_curve):
    alpha = 1_000
    sigma = 0.1
    hw: HullWhite = HullWhite(alpha, sigma, initial_curve=flat_curve, short_rate_tenor=0.25)
    simulation_tenors = np.array([0.325, 0.500, 0.625, 0.750])
    current_tenor = 0.25
    expected = \
        flat_curve.get_discount_factors(simulation_tenors) / \
        flat_curve.get_discount_factors(np.array(current_tenor))
    actual: np.ndarray = hw.a_function(simulation_tenors, current_tenor=0.25)
    assert actual == pytest.approx(expected, abs=0.0001)


def test_a_function_with_flat_curve(flat_curve):
    alpha = 0.25
    sigma = 0.1
    hw: HullWhite = HullWhite(alpha, sigma, initial_curve=flat_curve, short_rate_tenor=0.25)
    simulation_tenors = np.array([0.325, 0.500, 0.625, 0.750])
    current_tenor = 0.25
    expected = \
        flat_curve.get_discount_factors(simulation_tenors) / \
        flat_curve.get_discount_factors(np.array(current_tenor)) * \
        np.exp(hw.b_function(simulation_tenors, current_tenor) * flat_curve.get_zero_rates(np.array([current_tenor])) -
               sigma ** 2 *
               (np.exp(-1 * alpha * simulation_tenors) - np.exp(-1 * alpha * current_tenor)) ** 2 *
               (np.exp(2 * alpha * current_tenor) - 1) /
               (4 * alpha ** 3))
    actual: np.ndarray = hw.a_function(simulation_tenors, current_tenor=0.25)
    assert actual == pytest.approx(expected, abs=0.0001)


def test_get_discount_factors_with_large_alpha_and_flat_curve(flat_curve):
    alpha = 1_000
    sigma = 0.1
    hw: HullWhite = HullWhite(alpha, sigma, initial_curve=flat_curve, short_rate_tenor=0.25)
    curve = hw.get_discount_curve(short_rate=0.1, current_tenor=0.25)
    tenors = np.array([0.250, 0.375, 0.500, 0.625, 0.700])
    actual = curve.get_discount_factors(tenors)
    current_tenor = 0.25
    expected = \
        flat_curve.get_discount_factors(tenors + current_tenor) / \
        flat_curve.get_discount_factors(np.array([current_tenor]))
    assert actual[0] == pytest.approx(expected, abs=0.00001)


def test_get_discount_factors_with_zero_vol(flat_curve):
    alpha = 0.1
    sigma = 0.0
    hw: HullWhite = HullWhite(alpha, sigma, initial_curve=flat_curve, short_rate_tenor=0.25)
    curve = hw.get_discount_curve(short_rate=0.1, current_tenor=0.25)
    tenors = np.array([0.250, 0.375, 0.500, 0.625, 0.700])
    actual = curve.get_discount_factors(tenors)
    current_tenor = 0.25
    expected = \
        flat_curve.get_discount_factors(tenors + current_tenor) / \
        flat_curve.get_discount_factors(np.array([current_tenor]))
    assert actual[0] == pytest.approx(expected, abs=0.0001)


def test_exponential_stochastic_integral_for_small_alpha(flat_curve):
    hw = HullWhite(0.0001, 0.1, flat_curve, 0.25)
    np.random.seed(999)
    x = hw.exponential_stochastic_integral(1.0, 0.1, 10_000)
    assert x.mean() == pytest.approx(0.0, abs=0.02)
    assert x.var() == pytest.approx(1.0, abs=0.02)


def test_simulate(flat_curve):
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
    curve_discount_factors = \
        np.array(
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
    hw = HullWhite(alpha, sigma, flat_curve, short_rate_tenor=0.25)
    paths = hw.simulate(maturity, number_of_paths=1_000, number_of_time_steps=50)
    plt.show()


@pytest.mark.skip(reason="Incomplete")
def test_discount_curve():
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
    maturity = 5 / 12
    dt = maturity / number_of_time_steps
    alpha = 0.05
    sigma = 0.1
    initial_curve: Curve = Curve(curve_tenors, curve_discount_factors)
    hw = HullWhiteCurve(alpha, sigma, initial_curve, 0.25)
    hw.get_discount_curve(curve_tenors, dt)
