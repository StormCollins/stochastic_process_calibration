"""
Hull-White unit tests.
"""
import pytest
import QuantLib as ql
from scipy.stats import normaltest
from src.hullwhite.hullwhite import *
from src.utils.plot_utils import PlotUtils
from test_config import TestsConfig
from test_utils import file_and_test_annotation


@pytest.fixture
def flat_zero_rate_curve_tenors():
    """
    Curve tenors for flat zero rate curve.
    """
    return np.arange(0.00, 30.25, 0.25)


@pytest.fixture
def flat_zero_rate_curve(flat_zero_rate_curve_tenors):
    """
    Example flat zero rate curve (rate = 10%).
    """
    rate = 0.1
    discount_factors: np.ndarray = np.array([np.exp(-rate * t) for t in flat_zero_rate_curve_tenors])
    return Curve(flat_zero_rate_curve_tenors, discount_factors)


@pytest.fixture
def real_zero_rate_curve():
    """
    Example real world zero rate curve.
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


def test_theta_with_flat_zero_rate_curve(flat_zero_rate_curve, flat_zero_rate_curve_tenors):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#theta_t
    """
    alpha = 2
    sigma = 0.1
    hw: HullWhite = HullWhite(alpha=alpha, sigma=sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
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
    hw = HullWhite(alpha=alpha, sigma=sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
    actual: list[float] = [hw.theta(t) for t in [0.25, 0.375, 0.5, 0.625]]
    expected: list[float] = list(np.repeat(0.2, len(actual)))
    assert all([a == pytest.approx(b, 0.00001) for a, b in zip(actual, expected)])


def test_theta_with_constant_zero_rates_and_small_alpha(flat_zero_rate_curve, flat_zero_rate_curve_tenors):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#theta_t
    """
    alpha = 0.001
    sigma = 0.1
    hw: HullWhite = HullWhite(alpha=alpha, sigma=sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
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
    hw: HullWhite = HullWhite(alpha=alpha, sigma=sigma, initial_curve=real_zero_rate_curve, short_rate_tenor=0.01)
    actual = hw.theta(50)
    expected = alpha * 0.07549556 + sigma ** 2
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
    hw: HullWhite = HullWhite(alpha=alpha, sigma=sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
    simulation_tenors = np.array([0.325, 0.500, 0.625, 0.750])
    current_tenor = 0.25
    expected = \
        flat_zero_rate_curve.get_discount_factors(simulation_tenors) / \
        flat_zero_rate_curve.get_discount_factors(np.array(current_tenor))

    actual: np.ndarray = hw.a_function(simulation_tenors=0.25, tenors=simulation_tenors)
    assert actual == pytest.approx(expected, abs=0.0001)


def test_a_function_with_flat_curve(flat_zero_rate_curve):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#a-function
    """
    alpha = 0.25
    sigma = 0.1
    hw: HullWhite = HullWhite(alpha=alpha, sigma=sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
    simulation_tenors = np.array([0.325, 0.500, 0.625, 0.750])
    current_tenor = 0.25
    expected = \
        flat_zero_rate_curve.get_discount_factors(simulation_tenors) / \
        flat_zero_rate_curve.get_discount_factors(np.array(current_tenor)) * \
        np.exp(hw.b_function(current_tenor, simulation_tenors) *
               flat_zero_rate_curve.get_zero_rates(np.array([current_tenor])) -
               sigma ** 2 *
               (np.exp(-1 * alpha * simulation_tenors) - np.exp(-1 * alpha * current_tenor)) ** 2 *
               (np.exp(2 * alpha * current_tenor) - 1) /
               (4 * alpha ** 3))

    actual: np.ndarray = hw.a_function(simulation_tenors=0.25, tenors=simulation_tenors)
    assert actual == pytest.approx(expected, abs=0.0001)


def test_get_discount_factors_with_large_alpha_and_flat_curve(flat_zero_rate_curve):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#discount_factors
    """
    alpha = 1_000
    sigma = 0.1
    hw: HullWhite = HullWhite(alpha=alpha, sigma=sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
    curve = hw.get_discount_curve(short_rate=0.1, simulation_tenors=0.25)
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
    hw: HullWhite = HullWhite(alpha=alpha, sigma=sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
    curve = hw.get_discount_curve(short_rate=0.1, simulation_tenors=0.25)
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
    hw = HullWhite(alpha=alpha, sigma=sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=short_rate_tenor)
    tenors, rates, stochastic_dfs = \
        hw.simulate(
            maturity=maturity,
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            method=HullWhiteSimulationMethod.DISCRETISED_INTEGRAL)

    stochastic_discount_factors: np.ndarray = \
        np.mean(np.cumprod(np.exp(-1 * rates * (maturity / (number_of_time_steps + 1))), 1), 0)

    stochastic_discount_factors = np.insert(stochastic_discount_factors, 0, 1)
    time_steps: np.ndarray = \
        np.arange(0, maturity * (1 + 2 / number_of_time_steps), maturity / number_of_time_steps)

    initial_curve_discount_factors: np.ndarray = flat_zero_rate_curve.get_discount_factors(time_steps)
    # additional_annotation: str = \
    #     f'File: {os.path.basename(__file__)}\n' \
    #     f'Test: {inspect.currentframe().f_code.co_name}' \
    #     if TestsConfig.show_test_location \
    #     else None

    if TestsConfig.plots_on:
        PlotUtils.plot_curves(
            title='Mean Stochastic Discount Factors vs. Initial Discount Factors',
            time_steps=time_steps,
            curves=[('Initial Curve Discount Factors', initial_curve_discount_factors),
                    ('Stochastic Discount Factors', stochastic_discount_factors)],
            additional_annotation=file_and_test_annotation())

    assert all([a == pytest.approx(b, 0.05)
                for a, b in zip(stochastic_discount_factors, initial_curve_discount_factors)])


def test_simulate_with_flat_curve_and_small_alpha_and_small_sigma(flat_zero_rate_curve):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#r_t
    Under these conditions the simulated short rate doesn't deviate from the initial short rate.
    """
    maturity: float = 5
    alpha: float = 0.00001
    sigma: float = 0.0
    hw: HullWhite = HullWhite(alpha=alpha, sigma=sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.1)
    tenors, short_rates, stochastic_dfs = \
        hw.simulate(
            maturity=maturity,
            number_of_paths=2,
            number_of_time_steps=5,
            method=HullWhiteSimulationMethod.DISCRETISED_INTEGRAL)

    for value in short_rates[0]:
        assert value == pytest.approx(hw.initial_short_rate, abs=0.00001)


def test_exponential_stochastic_integral_for_small_alpha(flat_zero_rate_curve):
    """
    See https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#r_t
    """
    alpha = 0.0001
    sigma = 0.1
    hw = HullWhite(alpha=alpha, sigma=sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.25)
    np.random.seed(999)
    time_step_size = 0.01
    x = hw.exponential_stochastic_integral(upper_bound=1.0, time_step_size=time_step_size, number_of_paths=10_000)
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
    hw: HullWhite = HullWhite(alpha=alpha, sigma=sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.9)
    tenors, short_rates, stochastic_dfs = \
        hw.simulate(
            maturity=maturity,
            number_of_paths=100_000,
            number_of_time_steps=1,
            method=HullWhiteSimulationMethod.DISCRETISED_INTEGRAL)

    rates: np.ndarray = short_rates[:, -1]
    mean: float = \
        np.exp(-1 * alpha * maturity) * hw.initial_short_rate + \
        quad(lambda s: np.exp(alpha * (s - maturity)) * hw.theta(s), 0, maturity)[0]

    std: float = np.sqrt(((sigma ** 2) / (2 * alpha)) * (1 - np.exp(-2 * alpha * maturity)))
    if TestsConfig.plots_on:
        PlotUtils.plot_normal_histogram(rates, 'Hull-White $r(t)$ vs. Normal PDF', '$r(t)$', mean, std)

    statistic, p_value = normaltest(rates)
    # Null hypothesis (that rates are normal) cannot be rejected.
    assert p_value > 1e-3
    assert rates.mean() == pytest.approx(mean, abs=0.01)
    assert np.sqrt(rates.var()) == pytest.approx(std, abs=0.01)




def test_initial_short_rate_for_flat_curve(flat_zero_rate_curve):
    alpha = 0.1
    sigma = 0.1
    hw_short_short_rate_tenor: HullWhite = \
        HullWhite(
            alpha=alpha,
            sigma=sigma,
            initial_curve=flat_zero_rate_curve,
            short_rate_tenor=0.0001)

    hw_long_short_rate_tenor: HullWhite = \
        HullWhite(
            alpha=alpha,
            sigma=sigma,
            initial_curve=flat_zero_rate_curve,
            short_rate_tenor=0.25)

    assert hw_short_short_rate_tenor.initial_short_rate == \
           pytest.approx(hw_long_short_rate_tenor.initial_short_rate, abs=0.0001)


def test_get_fixings_with_flat_curve_and_zero_alpha_and_zero_vol(flat_zero_rate_curve):
    start_tenors: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75])
    end_tenors: np.ndarray = np.array([0.25, 0.50, 0.75, 1.00])
    forward_rates: np.ndarray = \
        flat_zero_rate_curve.get_forward_rates(
            start_tenors=start_tenors,
            end_tenors=end_tenors,
            compounding_convention=CompoundingConvention.NACQ)

    expected_forward_rate: float = 0.10126048209771543
    print(forward_rates)

    # alpha can't actually be set to zero since we divide by alpha in some expressions.
    alpha: float = 0.001
    sigma: float = 0.0
    hull_white_process: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, 0.01)
    np.random.seed(999)
    simulation_tenors, short_rates, stochastic_discount_factors = hull_white_process.simulate(1.00, 1_000, 100)
    fixings: np.ndarray = hull_white_process.get_fixings(simulation_tenors, short_rates, start_tenors, end_tenors)
    averaged_fixings: np.ndarray = np.average(fixings, 0)
    assert averaged_fixings == pytest.approx(expected_forward_rate, abs=0.00001)


def test_get_fixings_with_flat_curve_and_nonzero_alpha_and_zero_vol(flat_zero_rate_curve):
    start_tenors: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75])
    end_tenors: np.ndarray = np.array([0.25, 0.50, 0.75, 1.00])
    forward_rates: np.ndarray = \
        flat_zero_rate_curve.get_forward_rates(
            start_tenors=start_tenors,
            end_tenors=end_tenors,
            compounding_convention=CompoundingConvention.NACQ)

    expected_forward_rate: float = 0.10126048209771543
    print(forward_rates)

    alpha: float = 0.1
    sigma: float = 0.0
    hull_white_process: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, 0.01)
    np.random.seed(999)
    simulation_tenors, short_rates, stochastic_discount_factors = hull_white_process.simulate(1.00, 1_000, 100)
    fixings: np.ndarray = hull_white_process.get_fixings(simulation_tenors, short_rates, start_tenors, end_tenors)
    averaged_fixings: np.ndarray = np.average(fixings, 0)
    assert averaged_fixings == pytest.approx(expected_forward_rate, abs=0.00001)


def test_get_fixings_with_flat_curve_and_nonzero_alpha_and_nonzero_vol(flat_zero_rate_curve):
    start_tenors: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75])
    end_tenors: np.ndarray = np.array([0.25, 0.50, 0.75, 1.00])
    forward_rates: np.ndarray = \
        flat_zero_rate_curve.get_forward_rates(
            start_tenors=start_tenors,
            end_tenors=end_tenors,
            compounding_convention=CompoundingConvention.NACQ)

    expected_forward_rate: float = 0.10126048209771543
    print(forward_rates)

    alpha: float = 0.1
    sigma: float = 0.1
    hull_white_process: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, 0.01)
    np.random.seed(999)
    simulation_tenors, short_rates, stochastic_discount_factors = hull_white_process.simulate(1.00, 20_000, 100)
    fixings: np.ndarray = hull_white_process.get_fixings(simulation_tenors, short_rates, start_tenors, end_tenors)
    averaged_fixings: np.ndarray = np.average(fixings, 0)
    assert averaged_fixings == pytest.approx(expected_forward_rate, abs=0.005)


def test_get_fixings_with_flat_curve_and_nonzero_vol_and_zero_alpha(flat_zero_rate_curve):
    fixing_period_start_tenors: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75])
    fixing_period_end_tenors: np.ndarray = np.array([0.25, 0.50, 0.75, 1.00])
    forward_rates: np.ndarray = \
        flat_zero_rate_curve.get_forward_rates(
            start_tenors=fixing_period_start_tenors,
            end_tenors=fixing_period_end_tenors,
            compounding_convention=CompoundingConvention.NACQ)

    expected_forward_rate: float = 0.10126048209771543
    print(forward_rates)

    alpha: float = 0.001
    sigma: float = 0.1
    hull_white_process: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, 0.01)
    np.random.seed(999)
    simulation_tenors, short_rates, stochastic_discount_factors = hull_white_process.simulate(1.00, 1_000_000, 100)
    fixings: np.ndarray = hull_white_process.get_fixings(simulation_tenors, short_rates, fixing_period_start_tenors,
                                                         fixing_period_end_tenors)
    averaged_fixings: np.ndarray = np.average(fixings, 0)
    assert averaged_fixings == pytest.approx(expected_forward_rate, abs=0.005)


def test_simulate_short_term(flat_zero_rate_curve_tenors, flat_zero_rate_curve):
    alpha: float = 0.8
    sigma: float = 0.1
    hull_white_process: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, 0.01)
    np.random.seed(999)
    simulation_tenors, short_rates, stochastic_discount_factors = hull_white_process.simulate(1.00, 10_000, 20)
    PlotUtils.plot_monte_carlo_paths(
        time_steps=simulation_tenors,
        paths=short_rates,
        title='Hull-White Simulation',
        additional_annotation=file_and_test_annotation())


def test_convert_simulated_short_rates_to_curves_at_simulation_time_zero(flat_zero_rate_curve):
    alpha: float = 0.1
    sigma: float = 0.1
    hull_white_process: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, 0.01)
    np.random.seed(999)
    simulation_tenors, short_rates, stochastic_discount_factors = hull_white_process.simulate(1.00, 10, 20)

    curves: SimulatedCurves = \
        hull_white_process.convert_simulated_short_rates_to_curves(simulation_tenors, short_rates)

    # At simulation time 0 no randomness has been introduced yet, hence all discount factors, at simulation time 0, for
    # all tenors, should be identical.
    discount_factors: np.ndarray = curves.get_discount_factors(simulation_tenors[0], 0.9)
    for i in range(0, len(discount_factors) - 1):
        assert discount_factors[i] == discount_factors[i + 1]


def test_expected_curves(flat_zero_rate_curve):
    alpha: float = 0.1
    sigma: float = 0.1
    hull_white_process: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, 0.01)
    np.random.seed(999)
    simulation_tenors, short_rates, stochastic_discount_factors = hull_white_process.simulate(10.00, 100_000, 100)

    curves: SimulatedCurves = \
        hull_white_process.convert_simulated_short_rates_to_curves(simulation_tenors, short_rates)

    for i in range(0, len(simulation_tenors) - 1):
        expected_discount_factor: float = flat_zero_rate_curve.get_discount_factors(1) #simulation_tenors[i + 1])
        actual_discount_factor: float = \
            np.average(curves.get_discount_factors(simulation_tenors[i], 1.00))#simulation_tenors[i + 1]))
        print(simulation_tenors[i+1])
        assert actual_discount_factor == pytest.approx(expected_discount_factor, abs=0.1)


def test_expected_curves_with_zero_vol(flat_zero_rate_curve):
    alpha: float = 0.1
    sigma: float = 0.0
    hull_white_process: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, 0.01)
    np.random.seed(999)
    simulation_tenors, short_rates, stochastic_discount_factors = hull_white_process.simulate(10.00, 100_000, 1_000)
    curves: SimulatedCurves = \
        hull_white_process.convert_simulated_short_rates_to_curves(simulation_tenors, short_rates)

    for i in range(0, len(simulation_tenors) - 1):
        expected_discount_factor: float = flat_zero_rate_curve.get_discount_factors(simulation_tenors[i + 1])
        actual_discount_factor: float = \
            np.average(curves.get_discount_factors(simulation_tenors[i], simulation_tenors[i + 1]))

        assert actual_discount_factor == pytest.approx(expected_discount_factor, abs=0.0005)


def test_expected_forward_rates_for_flat_curve(flat_zero_rate_curve):
    alpha: float = 0.1
    sigma: float = 0.0
    hull_white_process: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, 0.01)
    np.random.seed(999)
    simulation_tenors, short_rates, stochastic_discount_factors = hull_white_process.simulate(1.00, 1, 20)

    curves: SimulatedCurves = \
        hull_white_process.convert_simulated_short_rates_to_curves(simulation_tenors, short_rates)

    actual_forward_rate_t2: float = np.average(curves.get_forward_rates(simulation_tenors[1], 0, 0.25))
    actual_forward_rate_t3: float = np.average(curves.get_forward_rates(simulation_tenors[2], 0, 0.25))
    actual_forward_rate_t4: float = np.average(curves.get_forward_rates(simulation_tenors[3], 0, 0.25))
    expected_forward_rate: float = flat_zero_rate_curve.get_forward_rates(0, 0.25, CompoundingConvention.NACQ)
    assert actual_forward_rate_t2 == pytest.approx(expected_forward_rate, abs=0.00001)
    assert actual_forward_rate_t3 == pytest.approx(expected_forward_rate, abs=0.00001)
    assert actual_forward_rate_t4 == pytest.approx(expected_forward_rate, abs=0.00001)


def test_simulate_short_mid_long_term(flat_zero_rate_curve):
    alpha: float = 0.01
    sigma: float = 0.01
    short_rate_tenor: float = 0.01
    maturity: float = 1.0
    number_of_paths: int = 1_000
    number_of_time_steps: int = 20
    hull_white_process_short_term: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, short_rate_tenor)
    simulation_tenors, short_rates, stochastic_discount_factors = \
        hull_white_process_short_term.simulate(maturity, number_of_paths, number_of_time_steps, plot_results=TestsConfig.plots_on)

    maturity = 5.0
    number_of_time_steps: int = 100
    hull_white_process_mid_term: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, short_rate_tenor)
    simulation_tenors, short_rates, stochastic_discount_factors = \
        hull_white_process_mid_term.simulate(maturity, number_of_paths, number_of_time_steps, plot_results=TestsConfig.plots_on)

    maturity = 10.0
    number_of_time_steps: int = 200
    hull_white_process_long_term: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, short_rate_tenor)
    simulation_tenors, short_rates, stochastic_discount_factors = \
        hull_white_process_long_term.simulate(maturity, number_of_paths, number_of_time_steps, plot_results=TestsConfig.plots_on)


def test_simulate_mid_term(flat_zero_rate_curve):
    alpha: float = 0.25
    sigma: float = 0.01
    short_rate_tenor: float = 0.01
    maturity: float = 5.0
    number_of_paths: int = 1_000
    number_of_time_steps: int = 20
    hull_white_process: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, short_rate_tenor)
    simulation_tenors, short_rates, stochastic_discount_factors = \
        hull_white_process.simulate(maturity, number_of_paths, number_of_time_steps, plot_results=TestsConfig.plots_on, method=HullWhiteSimulationMethod.DISCRETISED_SDE)


@pytest.mark.skip(reason='Long running.')
def test_plot_for_different_alphas_and_sigmas(real_zero_rate_curve):
    alpha = 0.1
    sigma = 0.1
    maturity = 2
    number_of_paths = 100_000
    number_of_time_steps = 100
    hw: HullWhite = HullWhite(alpha, sigma, real_zero_rate_curve, 0.1)
    hw.simulate(maturity, number_of_paths, number_of_time_steps, plot_results=True)


def test_simulate_long_term(flat_zero_rate_curve):
    alpha: float = 0.1
    sigma: float = 0.1
    short_rate_tenor: float = 0.0001
    maturity: float = 10.0
    number_of_paths: int = 1_000
    number_of_time_steps: int = 360
    hull_white_process: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, short_rate_tenor)
    simulation_tenors, short_rates, stochastic_discount_factors = \
        hull_white_process.simulate(
            maturity=maturity,
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            plot_results=TestsConfig.plots_on,
            method=HullWhiteSimulationMethod.DISCRETISED_SDE)


def test_simulated_vs_quantlib(flat_zero_rate_curve):
    alpha: float = 0.1
    sigma: float = 0.0
    timestep = 100
    maturity: float = 20.0
    forward_rate = 0.10126048209771543
    day_count = ql.Thirty360()
    todays_date = ql.Date(1, 1, 2020)
    ql.Settings.instance().evaluationDate = todays_date
    spot_curve = ql.FlatForward(todays_date, ql.QuoteHandle(ql.SimpleQuote(forward_rate)), day_count)
    spot_curve_handle = ql.YieldTermStructureHandle(spot_curve)
    hw_process = ql.HullWhiteProcess(spot_curve_handle, alpha, sigma)
    rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(timestep, ql.UniformRandomGenerator()))
    seq = ql.GaussianPathGenerator(hw_process, maturity, timestep, rng, False)

    num_paths = 1_000
    arr = np.zeros((num_paths, timestep+1))
    for i in range(num_paths):
        sample_path = seq.next()
        path = sample_path.value()
        time = [path.time(j) for j in range(len(path))]
        value = [path[j] for j in range(len(path))]
        arr[i, :] = np.array(value)

    time_steps = np.array(time)
    PlotUtils.plot_monte_carlo_paths(time_steps, arr, title='QuantLib Hull-White Simulation')


def test_simulate_medium_term(flat_zero_rate_curve_tenors, flat_zero_rate_curve):
    alpha: float = 0.1
    sigma: float = 0.1
    hull_white_process: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, 0.01)
    np.random.seed(999)
    maturity: float = 20.0
    number_of_time_steps: int = 100
    number_of_paths: int = 1_000
    simulation_tenors, short_rates, stochastic_discount_factors = \
        hull_white_process.simulate(maturity, number_of_paths, number_of_time_steps, HullWhiteSimulationMethod.DISCRETISED_INTEGRAL)
    PlotUtils.plot_monte_carlo_paths(
        time_steps=simulation_tenors,
        paths=short_rates,
        title='Hull-White Simulation',
        additional_annotation=file_and_test_annotation())


def test_simulated_distribution_with_flat_curve(flat_zero_rate_curve):
    maturity = 1
    alpha = 0.1
    sigma = 0.5
    np.random.seed(999)
    hw: HullWhite = HullWhite(alpha=alpha, sigma=sigma, initial_curve=flat_zero_rate_curve, short_rate_tenor=0.00001)
    tenors, short_rates, stochastic_dfs = \
        hw.simulate(
            maturity=maturity,
            number_of_paths=100_000,
            number_of_time_steps=100,
            method=HullWhiteSimulationMethod.DISCRETISED_INTEGRAL)

    rates: np.ndarray = short_rates[:, -1]
    mean: float = \
        np.exp(-1 * alpha * maturity) * hw.initial_short_rate + \
        quad(lambda s: np.exp(alpha * (s - maturity)) * hw.theta(s), 0, maturity)[0]

    std: float = np.sqrt(((sigma ** 2) / (2 * alpha)) * (1 - np.exp(-2 * alpha * maturity)))
    if TestsConfig.plots_on:
        PlotUtils.plot_normal_histogram(rates, 'Hull-White $r(t)$ vs. Normal PDF', '$r(t)$', mean, std)

    statistic, pvalue = normaltest(rates)
    assert pvalue > 1e-3
    assert rates.mean() == pytest.approx(mean, abs=0.05)
    assert np.sqrt(rates.var()) == pytest.approx(std, abs=0.05)
