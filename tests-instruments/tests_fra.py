from instruments.fra import *
import pytest


@pytest.fixture
def curve_tenors():
    return np.array([0.00, 0.25, 0.50, 0.75, 1.00])


@pytest.fixture
def flat_curve(curve_tenors):
    rate = 0.1
    discount_factors: np.ndarray(np.dtype(float)) = np.array([np.exp(-rate * t) for t in curve_tenors])
    return Curve(curve_tenors, discount_factors)


@pytest.fixture
def atm_fra(flat_curve):
    notional: float = 1_000_000
    strike: float = 0.10126
    start_tenor = 1
    end_tenor = 1.25
    return Fra(notional, strike, start_tenor, end_tenor)


def test_get_monte_carlo_values(flat_curve, atm_fra):
    strike: float = 0.10126
    alpha: float = 0.1
    sigma: float = 0.1
    number_of_paths: int = 100_000
    number_of_time_steps: int = 100

    np.random.seed(999)
    actual: np.ndarray = \
        atm_fra.get_monte_carlo_values(
            alpha=alpha,
            sigma=sigma,
            curve=flat_curve,
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps)

    short_rate_tenor: float = 0.01
    hw: HullWhite = HullWhite(alpha, sigma, flat_curve, short_rate_tenor)
    np.random.seed(999)
    tenors, short_rates = \
        hw.simulate(atm_fra.start_tenor, number_of_paths, number_of_time_steps, SimulationMethod.SLOWANALYTICAL)

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


def test_get_fair_forward_rate(flat_curve, atm_fra):
    actual: float = atm_fra.get_fair_forward_rate(flat_curve)
    expected: float = flat_curve.get_forward_rates(atm_fra.start_tenor, atm_fra.end_tenor, CompoundingConvention.Simple)
    assert actual == pytest.approx(expected, 0.0001)


def test_get_value(flat_curve, atm_fra):
    actual: float = atm_fra.get_value(flat_curve)
    expected: float = \
        atm_fra.notional * \
        (flat_curve.get_forward_rates(atm_fra.start_tenor, atm_fra.end_tenor, CompoundingConvention.Simple) -
         atm_fra.strike) * \
        (atm_fra.end_tenor - atm_fra.start_tenor)
    assert actual == pytest.approx(expected, 0.0001)


def test_get_monte_carlo_value_compared_to_analytical(flat_curve, atm_fra):
    return 0