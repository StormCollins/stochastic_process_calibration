"""
FRA Unit tests.
"""
from src.instruments.fra import *
import pytest


@pytest.fixture
def curve_tenors():
    """
    Curve tenors.
    """
    return np.array([0.00, 0.25, 0.50, 0.75, 1.00])


@pytest.fixture
def flat_zero_rate_curve(curve_tenors):
    """
    Flat zero rate curve.
    """
    rate = 0.1
    discount_factors: np.ndarray(np.dtype(float)) = np.array([np.exp(-rate * t) for t in curve_tenors])
    return Curve(curve_tenors, discount_factors)


@pytest.fixture
def atm_fra(flat_zero_rate_curve):
    """
    At-the-money (ATM) FRA.
    """
    notional: float = 1_000_000
    strike: float = 0.10126
    start_tenor = 1
    end_tenor = 1.25
    return Fra(notional, strike, start_tenor, end_tenor)


@pytest.fixture
def hw(flat_zero_rate_curve):
    """
    Hull-White process for a flat zero rate curve.
    """
    short_rate_tenor: float = 0.01
    alpha: float = 0.1
    sigma: float = 0.1
    return HullWhite(alpha, sigma, flat_zero_rate_curve, short_rate_tenor)


def test_get_fair_forward_rate(flat_zero_rate_curve, atm_fra):
    actual: float = atm_fra.get_fair_forward_rate(flat_zero_rate_curve)
    expected: float = \
        float(flat_zero_rate_curve.get_forward_rates(
            start_tenors=atm_fra.start_tenor,
            end_tenors=atm_fra.end_tenor,
            compounding_convention=CompoundingConvention.Simple))

    assert actual == pytest.approx(expected, 0.0001)


def test_get_value(flat_zero_rate_curve, atm_fra):
    actual: float = atm_fra.get_value(flat_zero_rate_curve)
    expected: float = \
        atm_fra.notional * \
        (flat_zero_rate_curve.get_forward_rates(atm_fra.start_tenor, atm_fra.end_tenor, CompoundingConvention.Simple) -
         atm_fra.strike) * \
        (atm_fra.end_tenor - atm_fra.start_tenor)
    assert actual == pytest.approx(expected, 0.0001)


def test_get_monte_carlo_values(flat_zero_rate_curve, atm_fra, hw):
    strike: float = 0.10126
    alpha: float = 0.1
    sigma: float = 0.1
    number_of_paths: int = 10_000
    number_of_time_steps: int = 100

    np.random.seed(999)
    actual, actual_std = \
        atm_fra.get_monte_carlo_values(
            alpha=alpha,
            sigma=sigma,
            initial_curve=flat_zero_rate_curve,
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps)

    np.random.seed(999)
    tenors, short_rates, stochastic_dfs = \
        hw.simulate(
            atm_fra.start_tenor, number_of_paths, number_of_time_steps, HullWhiteSimulationMethod.SLOWANALYTICAL)

    dt = atm_fra.start_tenor / number_of_time_steps
    stochastic_discount_factors = np.prod(np.exp(-1 * short_rates * dt), 1)

    df = \
        hw.a_function(atm_fra.start_tenor, np.array([atm_fra.end_tenor])) * \
        np.exp(-1 * short_rates[:, -1] * hw.b_function(atm_fra.start_tenor, np.array([atm_fra.end_tenor])))

    forward_rates = (1/(atm_fra.end_tenor - atm_fra.start_tenor)) * ((1/df) - 1)

    expected: float = \
        float(np.mean(
            atm_fra.notional *
            (forward_rates - strike) *
            (atm_fra.end_tenor - atm_fra.start_tenor) *
            stochastic_discount_factors))

    assert actual[-1] == pytest.approx(expected, 0.0001)


def test_get_monte_carlo_value_compared_to_analytical(flat_zero_rate_curve, atm_fra, hw):
    expected: float = atm_fra.get_value(flat_zero_rate_curve)
    np.random.seed(999)
    actual, actual_std = \
        atm_fra.get_monte_carlo_values(
            alpha=hw.alpha,
            sigma=hw.sigma,
            initial_curve=flat_zero_rate_curve,
            number_of_paths=10_000,
            number_of_time_steps=20,
            short_rate_tenor=hw.short_rate_tenor,
            plot_paths=True)

    # TODO: Can we get the Monte Carlo value closer to zero?
    assert actual[-1] == pytest.approx(expected, abs=450)


@pytest.mark.skip(reason="Long running - move to Jupyter notebook.")
def test_fra_value_vs_alpha(flat_zero_rate_curve, atm_fra):
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
                atm_fra.get_monte_carlo_values(
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
