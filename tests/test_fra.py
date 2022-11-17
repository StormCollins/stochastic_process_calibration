"""
FRA Unit tests.
"""
import pytest
from matplotlib import pyplot as plt
from src.instruments.fra import *
from test_config import TestsConfig
from test_utils import file_and_test_annotation


@pytest.fixture
def flat_zero_rate_curve_tenors():
    """
    Curve tenors for the flat zero rate curve.
    """
    return np.arange(0, 30.25, 0.25)


@pytest.fixture
def flat_zero_rate_curve(flat_zero_rate_curve_tenors):
    """
    Flat zero rate curve.
    """
    rate = 0.1
    discount_factors: np.ndarray(np.dtype(float)) = np.array([np.exp(-rate * t) for t in flat_zero_rate_curve_tenors])
    return Curve(flat_zero_rate_curve_tenors, discount_factors)


@pytest.fixture
def short_dated_atm_fra(flat_zero_rate_curve):
    """
    At-the-money (ATM) FRA.
    """
    notional: float = 1_000_000
    strike: float = 0.10126
    start_tenor = 1
    end_tenor = 1.25
    return Fra(notional, strike, start_tenor, end_tenor)


@pytest.fixture
def medium_dated_atm_fra(flat_zero_rate_curve):
    """
    At-the-money (ATM) FRA.
    """
    notional: float = 1_000_000
    strike: float = 0.10126
    start_tenor = 5
    end_tenor = 5.25
    return Fra(notional, strike, start_tenor, end_tenor)


@pytest.fixture
def long_dated_atm_fra(flat_zero_rate_curve):
    """
    At-the-money (ATM) FRA.
    """
    notional: float = 1_000_000
    strike: float = 0.10126
    start_tenor = 10.00
    end_tenor = 10.25
    return Fra(notional, strike, start_tenor, end_tenor)


@pytest.fixture
def hull_white_process(flat_zero_rate_curve):
    """
    Hull-White process for a flat zero rate curve.
    """
    short_rate_tenor: float = 0.01
    alpha: float = 0.8
    sigma: float = 0.1
    return HullWhite(alpha, sigma, flat_zero_rate_curve, short_rate_tenor)


def test_get_fair_forward_rate(flat_zero_rate_curve, short_dated_atm_fra):
    actual: float = short_dated_atm_fra.get_fair_forward_rate(flat_zero_rate_curve)
    expected: float = \
        float(flat_zero_rate_curve.get_forward_rates(
            start_tenors=short_dated_atm_fra.start_tenor,
            end_tenors=short_dated_atm_fra.end_tenor,
            compounding_convention=CompoundingConvention.Simple))

    assert actual == pytest.approx(expected, 0.0001)


def test_get_value(flat_zero_rate_curve, short_dated_atm_fra):
    actual: float = short_dated_atm_fra.get_value(flat_zero_rate_curve)
    expected: float = \
        short_dated_atm_fra.notional * \
        (flat_zero_rate_curve.get_forward_rates(short_dated_atm_fra.start_tenor, short_dated_atm_fra.end_tenor, CompoundingConvention.Simple) -
         short_dated_atm_fra.strike) * \
        (short_dated_atm_fra.end_tenor - short_dated_atm_fra.start_tenor)

    assert actual == pytest.approx(expected, 0.0001)


def test_get_value_long_dated_fra(flat_zero_rate_curve, long_dated_atm_fra):
    actual: float = long_dated_atm_fra.get_value(flat_zero_rate_curve)
    expected: float = \
        long_dated_atm_fra.notional * \
        (flat_zero_rate_curve.get_forward_rates(
            start_tenors=long_dated_atm_fra.start_tenor,
            end_tenors=long_dated_atm_fra.end_tenor,
            compounding_convention=CompoundingConvention.Simple) -
         long_dated_atm_fra.strike) * \
        (long_dated_atm_fra.end_tenor - long_dated_atm_fra.start_tenor)

    assert actual == pytest.approx(expected, 0.0001)


def test_get_monte_carlo_values_shorted_dated_fra(flat_zero_rate_curve, short_dated_atm_fra, hull_white_process):
    strike: float = 0.10126
    alpha: float = 0.05
    sigma: float = 0.1
    number_of_paths: int = 1_000_000
    number_of_time_steps: int = 100
    np.random.seed(999)
    actual, actual_std = \
        short_dated_atm_fra.get_monte_carlo_values(
            alpha=alpha,
            sigma=sigma,
            initial_curve=flat_zero_rate_curve,
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps)

    np.random.seed(999)
    tenors, short_rates, stochastic_dfs = \
        hull_white_process.simulate(
            short_dated_atm_fra.start_tenor,
            number_of_paths, number_of_time_steps, HullWhiteSimulationMethod.DISCRETISED_INTEGRAL)

    dt = short_dated_atm_fra.start_tenor / number_of_time_steps
    stochastic_discount_factors = np.prod(np.exp(-1 * short_rates * dt), 1)

    df = \
        hull_white_process.a_function(short_dated_atm_fra.start_tenor, np.array([short_dated_atm_fra.end_tenor])) * \
        np.exp(
            -1 * short_rates[:, -1] * hull_white_process.b_function(short_dated_atm_fra.start_tenor, np.array([short_dated_atm_fra.end_tenor])))

    forward_rates = (1 / (short_dated_atm_fra.end_tenor - short_dated_atm_fra.start_tenor)) * ((1 / df) - 1)

    expected: float = \
        float(np.mean(
            short_dated_atm_fra.notional *
            (forward_rates - strike) *
            (short_dated_atm_fra.end_tenor - short_dated_atm_fra.start_tenor) *
            stochastic_discount_factors))

    assert actual[-1] == pytest.approx(expected, abs=0.0000001)


def test_monte_carlo_vs_analytical_value_for_short_dated_fra(
        flat_zero_rate_curve,
        short_dated_atm_fra,
        hull_white_process):
    expected: float = short_dated_atm_fra.get_value(flat_zero_rate_curve)
    np.random.seed(999)
    actual, actual_std = \
        short_dated_atm_fra.get_monte_carlo_values(
            alpha=hull_white_process.alpha,
            sigma=hull_white_process.sigma,
            initial_curve=flat_zero_rate_curve,
            number_of_paths=100_000,
            number_of_time_steps=100,
            short_rate_tenor=hull_white_process.short_rate_tenor,
            additional_annotation_for_plot=file_and_test_annotation(),
            plot_paths=False)

    assert actual[-1] == pytest.approx(expected, abs=10)


def test_monte_carlo_vs_analytical_value_for_medium_dated_fra(
        flat_zero_rate_curve,
        medium_dated_atm_fra,
        hull_white_process):
    expected: float = medium_dated_atm_fra.get_value(flat_zero_rate_curve)
    np.random.seed(999)
    # actual, actual_std = \
    #     medium_dated_atm_fra.get_monte_carlo_values(
    #         alpha=hull_white_process.alpha,
    #         sigma=hull_white_process.sigma,
    #         initial_curve=flat_zero_rate_curve,
    #         number_of_paths=100_000,
    #         number_of_time_steps=10,
    #         short_rate_tenor=hull_white_process.short_rate_tenor,
    #         additional_annotation_for_plot=file_and_test_annotation(),
    #         plot_paths=True)

    simulation_tenors, short_rates, stochastic_discount_factors = hull_white_process.simulate(1000, 10_000, 5)
    hull_white_process.plot_paths(short_rates, 5)

    # assert actual[-1] == pytest.approx(expected, abs=1)


def test_get_monte_carlo_value_compared_to_analytical_long_dated_fra(
        flat_zero_rate_curve,
        long_dated_atm_fra,
        hull_white_process):
    expected: float = long_dated_atm_fra.get_value(flat_zero_rate_curve)
    np.random.seed(999)
    actual, actual_std = \
        long_dated_atm_fra.get_monte_carlo_values(
            alpha=hull_white_process.alpha,
            sigma=hull_white_process.sigma,
            initial_curve=flat_zero_rate_curve,
            number_of_paths=10_000,
            number_of_time_steps=20,
            short_rate_tenor=hull_white_process.short_rate_tenor,
            additional_annotation_for_plot=file_and_test_annotation(),
            plot_paths=TestsConfig.plots_on)

    assert actual[-1] == pytest.approx(expected, abs=1)


# @pytest.mark.skip(reason="Long running - move to Jupyter notebook.")
def test_fra_value_vs_alpha(flat_zero_rate_curve, short_dated_atm_fra):
    alphas = np.arange(0, 2, 0.1) + 0.1
    sigma = 0.1
    number_of_steps_list = [10, 20, 50, 100, 200]
    number_of_paths = 100_000
    for number_of_steps in number_of_steps_list:
        fra_values = list()
        for alpha in alphas:
            print(f'Current alpha: {alpha}')
            hw: HullWhite = HullWhite(alpha, sigma, flat_zero_rate_curve, 0.01)
            np.random.seed(999)
            actual, actual_std = \
                short_dated_atm_fra.get_monte_carlo_values(
                    alpha=hw.alpha,
                    sigma=hw.sigma,
                    initial_curve=flat_zero_rate_curve,
                    number_of_paths=number_of_paths,
                    number_of_time_steps=number_of_steps,
                    short_rate_tenor=hw.short_rate_tenor)
            fra_values.append(actual[-1])

        fig, ax = plt.subplots()
        ax.set_title(f'FRA Value vs. $\\alpha$ \n({number_of_paths:,} sims & $\\sigma$ = {sigma})')
        ax.plot(alphas, fra_values, color='#0D8390', label=f'Step size {1/number_of_steps}')
        ax.grid(True)
        ax.set_facecolor('#AAAAAA')
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('FRA Value')
        ax.set_xlim([0, alphas[-1]])
        ax.legend()
        plt.show()
