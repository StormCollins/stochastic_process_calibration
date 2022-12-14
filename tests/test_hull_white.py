import pytest
import scipy.stats

from src.hullwhite.hullwhite import *


@pytest.fixture
def curve_tenors():
    return np.array([0.00, 0.25, 0.50, 0.75, 1.00])


@pytest.fixture
def flat_zero_rate_curve(curve_tenors):
    """
    Example flat zero rate curve (rate = 10%).

    :param curve_tenors: Curve tenors.
    :return: Flat zero rate curve.
    """
    rate = 0.1
    discount_factors: np.ndarray = np.array([np.exp(-rate * t) for t in curve_tenors])
    return Curve(curve_tenors, discount_factors)


@pytest.fixture
def real_zero_rate_curve():
    """
    Example real world zero rate curve.

    :return: Real world curve.
    """
    tenors: np.ndarray = \
        np.array([
            0.000,
            0.003,
            0.090,
            0.249,
            0.501,
            0.751,
            1.008,
            1.249,
            1.499,
            1.759,
            2.008,
            3.003,
            4.003,
            5.003,
            6.005,
            7.000,
            8.000,
            9.005,
            10.008,
            12.008,
            15.011,
            20.014,
            25.011,
            30.016])

    discount_factors: np.ndarray = \
        np.array([
            1.000000,
            0.999889,
            0.996165,
            0.989230,
            0.977003,
            0.963053,
            0.947840,
            0.932993,
            0.917552,
            0.901441,
            0.886044,
            0.825317,
            0.764745,
            0.704258,
            0.644889,
            0.588923,
            0.536582,
            0.488218,
            0.442883,
            0.367192,
            0.278862,
            0.176672,
            0.115827,
            0.103715])

    return Curve(tenors, discount_factors)


def test_theta_with_flat_zero_rate_curve(flat_zero_rate_curve, curve_tenors):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#theta_t
    """
    alpha = 2
    sigma = 0.1
    hw: HullWhite = HullWhite(alpha, sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
    test_tenors: list[float] = [0.25, 0.375, 0.5, 0.625]
    actual: list[float] = [hw.theta(t) for t in test_tenors]
    expected: list[float] = \
        [alpha * 0.1 + (sigma ** 2) / (2 * alpha) * (1 - np.exp(-2 * alpha * t)) for t in test_tenors]
    assert all([a == pytest.approx(b, 0.001) for a, b in zip(actual, expected)])


def test_theta_with_flat_zero_rate_curve_and_zero_vol(flat_zero_rate_curve):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#theta_t
    """
    alpha = 2
    sigma = 0
    hw = HullWhite(alpha, sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
    actual: list[float] = [hw.theta(t) for t in [0.25, 0.375, 0.5, 0.625]]
    expected: list[float] = list(np.repeat(0.2, len(actual)))
    assert all([a == pytest.approx(b, 0.00001) for a, b in zip(actual, expected)])


def test_theta_with_constant_zero_rates_and_small_alpha(flat_zero_rate_curve, curve_tenors):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#theta_t
    """
    alpha = 0.001
    sigma = 0.1
    hw: HullWhite = HullWhite(alpha, sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
    test_tenors: list[float] = [0.25, 0.375, 0.5, 0.625]
    actual: list[float] = [hw.theta(t) for t in test_tenors]
    expected: list[float] = list(np.zeros(len(actual)))
    assert all([a == pytest.approx(b, abs=0.01) for a, b in zip(actual, expected)])


# TODO: Finish test.
def test_theta_with_real_zero_rate_curve_and_time_tends_to_infinity(real_zero_rate_curve):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#theta_t
    """
    alpha: float = 1.0
    sigma: float = 0.1
    hw: HullWhite = HullWhite(alpha, sigma, initial_curve=real_zero_rate_curve, short_rate_tenor=0.01)
    actual = hw.theta(50)
    expected = alpha * 0.07549556 + sigma**2
    print(expected)


def test_b_function_large_alpha(flat_zero_rate_curve):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#b-function
    """
    alpha: float = 10_000
    sigma: float = 0
    hw: HullWhite = HullWhite(alpha, sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
    actual = hw.b_function(0.00, np.array([0.25]))[0]
    assert actual == pytest.approx(0.0, abs=0.0001)


def test_a_function_with_large_alpha_and_flat_curve(flat_zero_rate_curve):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#a-function
    """
    alpha = 1_000
    sigma = 0.1
    hw: HullWhite = HullWhite(alpha, sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
    simulation_tenors = np.array([0.325, 0.500, 0.625, 0.750])
    current_tenor = 0.25
    expected = \
        flat_zero_rate_curve.get_discount_factors(simulation_tenors) / \
        flat_zero_rate_curve.get_discount_factors(np.array(current_tenor))
    actual: np.ndarray = hw.a_function(current_tenor=0.25, tenors=simulation_tenors)
    assert actual == pytest.approx(expected, abs=0.0001)


def test_a_function_with_flat_curve(flat_zero_rate_curve):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#a-function
    """
    alpha = 0.25
    sigma = 0.1
    hw: HullWhite = HullWhite(alpha, sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
    simulation_tenors = np.array([0.325, 0.500, 0.625, 0.750])
    current_tenor = 0.25
    expected = \
        flat_zero_rate_curve.get_discount_factors(simulation_tenors) / \
        flat_zero_rate_curve.get_discount_factors(np.array(current_tenor)) * \
        np.exp(hw.b_function(current_tenor, simulation_tenors) * flat_zero_rate_curve.get_zero_rates(np.array([current_tenor])) -
               sigma ** 2 *
               (np.exp(-1 * alpha * simulation_tenors) - np.exp(-1 * alpha * current_tenor)) ** 2 *
               (np.exp(2 * alpha * current_tenor) - 1) /
               (4 * alpha ** 3))
    actual: np.ndarray = hw.a_function(current_tenor=0.25, tenors=simulation_tenors)
    assert actual == pytest.approx(expected, abs=0.0001)


def test_get_discount_factors_with_large_alpha_and_flat_curve(flat_zero_rate_curve):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#discount_factors
    """
    alpha = 1_000
    sigma = 0.1
    hw: HullWhite = HullWhite(alpha, sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
    curve = hw.get_discount_curve(short_rate=0.1, current_tenor=0.25)
    tenors = np.array([0.250, 0.375, 0.500, 0.625, 0.700])
    actual = curve.get_discount_factors(tenors)
    current_tenor = 0.25
    expected = \
        flat_zero_rate_curve.get_discount_factors(tenors + current_tenor) / \
        flat_zero_rate_curve.get_discount_factors(np.array([current_tenor]))
    assert actual[0] == pytest.approx(expected, abs=0.00001)


def test_get_discount_factors_with_zero_vol(flat_zero_rate_curve):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#discount_factors
    """
    alpha = 0.1
    sigma = 0.0
    hw: HullWhite = HullWhite(alpha, sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
    curve = hw.get_discount_curve(short_rate=0.1, current_tenor=0.25)
    tenors = np.array([0.250, 0.375, 0.500, 0.625, 0.700])
    actual = curve.get_discount_factors(tenors)
    current_tenor = 0.25
    expected = \
        flat_zero_rate_curve.get_discount_factors(tenors + current_tenor) / \
        flat_zero_rate_curve.get_discount_factors(np.array([current_tenor]))
    assert actual[0] == pytest.approx(expected, abs=0.0001)


def test_fit_to_initial_flat_zero_rate_curve(flat_zero_rate_curve):
    """
    https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#discount_factors
    """
    alpha: float = 0.1
    sigma: float = 0.1
    maturity: float = 5
    number_of_time_steps: int = 100
    number_of_paths: int = 100_000
    short_rate_tenor: float = maturity / (number_of_time_steps + 1)
    hw = HullWhite(alpha, sigma, flat_zero_rate_curve, short_rate_tenor)
    tenors, rates, stochastic_dfs = hw.simulate(maturity, number_of_paths, number_of_time_steps, method=SimulationMethod.SLOWANALYTICAL)
    stochastic_discount_factors: np.ndarray = \
        np.mean(np.cumprod(np.exp(-1 * rates * (maturity / (number_of_time_steps + 1))), 1), 0)
    stochastic_discount_factors = np.insert(stochastic_discount_factors, 0, 1)

    time_steps: np.ndarray = \
        np.arange(0, maturity * (1 + 2 / number_of_time_steps), maturity / number_of_time_steps)
    initial_curve_discount_factors: np.ndarray = flat_zero_rate_curve.get_discount_factors(time_steps)

    fig, ax = plt.subplots()
    ax.set_title('Comparison of Stochastic and Initial Curve Discount Factors')
    ax.plot(time_steps, initial_curve_discount_factors, color='#0D8390', label='Initial Curve')
    ax.plot(time_steps, stochastic_discount_factors, color='#86BC25', label='Stochastic Discount Factors')
    ax.grid(True)
    ax.set_facecolor('#AAAAAA')
    ax.set_xlabel('$t$ (years)')
    ax.set_ylabel('$P(0, t)$')
    ax.set_xlim([0, time_steps[-1]])
    ax.set_ylim([0, 1])
    ax.legend()
    plt.show()

    assert all([a == pytest.approx(b, 0.05)
                for a, b in zip(stochastic_discount_factors, initial_curve_discount_factors)])


def test_simulate_with_flat_curve_and_small_alpha_and_small_sigma(flat_zero_rate_curve):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#r_t
    Under these conditions the simulated short rate doesn't deviate from the initial short rate.
    """
    maturity = 5
    alpha = 0.00001
    sigma = 0.0
    hw: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, short_rate_tenor=0.1)
    tenors, short_rates, stochastic_dfs = \
        hw.simulate(maturity, number_of_paths=2, number_of_time_steps=5, method=SimulationMethod.SLOWANALYTICAL)
    for value in short_rates[0]:
        assert value == pytest.approx(hw.initial_short_rate, abs=0.00001)


def test_exponential_stochastic_integral_for_small_alpha(flat_zero_rate_curve):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#r_t
    """
    alpha = 0.0001
    sigma = 0.1
    hw = HullWhite(alpha, sigma, flat_zero_rate_curve, 0.25)
    np.random.seed(999)
    time_step_size = 0.01
    x = hw.exponential_stochastic_integral(maturity=1.0, time_step_size=time_step_size, number_of_paths=10_000)
    assert x.mean() == pytest.approx(0.0, abs=0.05)
    assert x.var() == pytest.approx(1 * time_step_size, abs=0.02)


def test_simulated_distribution_with_flat_curve_and_small_alpha(flat_zero_rate_curve):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#r_t
    """
    maturity = 1
    alpha = 0.1
    sigma = 0.1
    np.random.seed(999)
    hw: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, short_rate_tenor=0.001)
    tenors, short_rates, stochastic_dfs = \
        hw.simulate(
            maturity=maturity,
            number_of_paths=100_000,
            number_of_time_steps=1,
            method=SimulationMethod.SLOWANALYTICAL,
            plot_results=False)
    plt.style.use('ggplot')
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.set_facecolor('#AAAAAA')
    ax.grid(False)
    rates: np.ndarray = short_rates[:, -1]
    (values, bins, _) = ax.hist(rates, bins=75, density=True, label='Histogram of $r(t)$', color='#6C3D91')
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    normal_distribution_mean = \
        np.exp(-1 * alpha * maturity) * hw.initial_short_rate + \
        scipy.integrate.quad(lambda s: np.exp(alpha * (s - maturity)) * hw.theta(s), 0, maturity)[0]
    normal_distribution_std = np.sqrt(((sigma ** 2) / (2 * alpha)) * (1 - np.exp(-2 * alpha * maturity)))
    pdf = norm.pdf(x=bin_centers, loc=normal_distribution_mean, scale=normal_distribution_std)
    ax.plot(bin_centers, pdf, label='PDF', color='#00A3E0', linewidth=1, ls='solid')
    ax.set_title('Comparison of Hull-White $r(t)$ to normal PDF')
    ax.annotate(
        '$\mathcal{N}(e^{-\\alpha t}r(0) + \int_0^t e^{\\alpha(s-t)}\\theta(s)ds,\\frac{\\sigma^2}{2\\alpha}\\left(1 - e^{-2\\alpha t} \\right))$',
        xy=(0, 0.4),
        xytext=(-2, 0.2))
    ax.legend()
    plt.show()
    statistic, p_value = scipy.stats.normaltest(rates)
    assert p_value > 1e-3  # Null hypothesis (that rates are normal) cannot be rejected.
    assert rates.mean() == pytest.approx(normal_distribution_mean[0], abs=0.01)
    assert np.sqrt(rates.var()) == pytest.approx(normal_distribution_std, abs=0.01)


def test_simulated_distribution_with_flat_curve(flat_zero_rate_curve):
    maturity = 1
    alpha = 0.1
    sigma = 0.5
    np.random.seed(999)
    hw: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, short_rate_tenor=0.001)
    tenors, short_rates, stochastic_dfs = \
        hw.simulate(
            maturity=maturity,
            number_of_paths=100_000,
            number_of_time_steps=1,
            method=SimulationMethod.SLOWANALYTICAL,
            plot_results=False)

    rates: np.ndarray = short_rates[:, -1]
    plt.style.use('ggplot')
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.set_facecolor('#AAAAAA')
    ax.grid(False)
    (values, bins, _) = ax.hist(rates, bins=75, density=True, label='Histogram of $r(t)$', color='#6C3D91')
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    normal_distribution_mean = \
        np.exp(-1 * alpha * maturity) * hw.initial_short_rate + \
        scipy.integrate.quad(lambda s: np.exp(alpha * (s - maturity)) * hw.theta(s), 0, maturity)[0]
    normal_distribution_std = np.sqrt(((sigma ** 2) / (2 * alpha)) * (1 - np.exp(-2 * alpha * maturity)))
    pdf = norm.pdf(x=bin_centers, loc=normal_distribution_mean, scale=normal_distribution_std)
    ax.plot(bin_centers, pdf, label='PDF', color='#00A3E0', linewidth=1, ls='solid')
    ax.set_title('Comparison of Hull-White $r(t)$ to normal PDF')
    ax.annotate(
        '$\mathcal{N}(e^{-\\alpha t}r(0) + \int_0^t e^{\\alpha(s-t)}\\theta(s)ds,\\frac{\\sigma^2}{2\\alpha}\\left(1 - e^{-2\\alpha t} \\right))$',
        xy=(0, 0.4),
        xytext=(-2, 0.2))
    ax.legend()
    plt.show()
    statistic, pvalue = scipy.stats.normaltest(rates)
    assert pvalue > 1e-3
    assert rates.mean() == pytest.approx(normal_distribution_mean[0], abs=0.05)
    assert np.sqrt(rates.var()) == pytest.approx(normal_distribution_std, abs=0.05)


def test_initial_short_rate_for_flat_curve(flat_zero_rate_curve):
    alpha = 0.1
    sigma = 0.1
    hw_short_short_rate_tenor: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, short_rate_tenor=0.0001)
    hw_long_short_rate_tenor: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, short_rate_tenor=0.25)
    assert hw_short_short_rate_tenor.initial_short_rate == \
           pytest.approx(hw_long_short_rate_tenor.initial_short_rate, abs=0.0001)

