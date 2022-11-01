"""
Time-dependent GBM unit tests.
"""
import pytest
import inspect
import os
from test_config import TestsConfig
from src.gbm.time_dependent_gbm import TimeDependentGBM
from src.gbm.time_independent_gbm import TimeIndependentGBM
from src.enums_and_named_tuples.path_statistics import PathStatistics
from src.utils.plot_utils import *
from test_config import TestsConfig
from test_utils import file_and_test_annotation


@pytest.fixture
def excel_file_path() -> str:
    """
    Path to the Excel file containing the equity ATM (at-the-money) volatility term-structure.
    """
    return r'tests/equity-atm-volatility-surface.xlsx'


def test_get_time_dependent_volatility_for_constant_vol_term_structure(excel_file_path):
    drift: float = 0.1
    volatility: float = 0.4
    gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=drift,
            excel_file_path=excel_file_path,
            sheet_name='constant_vol_surface',
            initial_spot=0)

    np.random.seed(999)
    time_steps = np.arange(0, 10, 1) + 1
    actual_vols = [float(gbm.get_time_dependent_vol(t)) for t in time_steps]
    expected_vols = list(np.repeat(volatility, 10))
    assert actual_vols == pytest.approx(expected_vols, abs=0.00001)


def test_get_time_dependent_volatility_for_non_constant_vol_term_structure(excel_file_path):
    drift: float = 0.1
    volatility: float = 0.4
    gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=drift,
            excel_file_path=excel_file_path,
            sheet_name='constant_vol_surface',
            initial_spot=0)

    np.random.seed(999)
    time_steps = np.arange(0, 10, 1) + 1
    actual_vols = [float(gbm.get_time_dependent_vol(t)) for t in time_steps]
    expected_vols = list(np.repeat(volatility, 10))
    assert actual_vols == pytest.approx(expected_vols, abs=0.00001)


def test_get_time_dependent_gbm_paths_for_constant_vol_term_structure(excel_file_path):
    drift: float = 0.1
    volatility: float = 0.4
    number_of_paths: int = 10
    number_of_time_steps: int = 2
    initial_spot: float = 50
    time_to_maturity = 10
    np.random.seed(999)
    time_dependent_gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=drift,
            excel_file_path=excel_file_path,
            sheet_name='constant_vol_surface',
            initial_spot=initial_spot)

    np.random.seed(999)

    actual_paths = \
        time_dependent_gbm.get_paths(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            time_to_maturity=time_to_maturity)

    time_independent_gbm: TimeIndependentGBM = \
        TimeIndependentGBM(drift=drift, volatility=volatility, initial_spot=initial_spot)

    np.random.seed(999)
    expected_paths = \
        time_independent_gbm.get_paths(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            time_to_maturity=time_to_maturity)

    assert actual_paths == pytest.approx(expected_paths, abs=1.0)


def test_distribution(excel_file_path):
    np.random.seed(999)
    time_to_maturity: float = 1.0
    gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=0.0,
            excel_file_path=excel_file_path,
            sheet_name='constant_vol_surface',
            initial_spot=100)

    paths: np.ndarray = gbm.get_paths(10_000, 10, time_to_maturity)
    if TestsConfig.plots_on:
        gbm.create_plots(paths, 1.0, file_and_test_annotation())

    path_stats: PathStatistics = gbm.get_path_statistics(paths, time_to_maturity)
    assert path_stats.EmpiricalMean == pytest.approx(path_stats.TheoreticalMean, abs=1.00)
    assert path_stats.EmpiricalStandardDeviation == pytest.approx(path_stats.TheoreticalStandardDeviation, abs=1.00)


def test_bootstrapped_vols_for_non_constant_vol_term_structure(excel_file_path):
    drift: float = 0.1
    gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=drift,
            excel_file_path=excel_file_path,
            sheet_name='vol_surface',
            initial_spot=0)

    np.random.seed(999)
    tenors: list[float] = \
        [0.0001,
         0.0833,
         0.1667,
         0.2500,
         0.5000,
         0.7500,
         1.0000,
         2.0000,
         3.0000,
         5.0000,
         7.0000,
         10.0000]

    tenors: list[float] = [t - 0.0001 for t in tenors]
    actual: list[float] = [float(gbm.get_time_dependent_vol(t)) for t in tenors]
    expected: list[float] = \
        [0.12775,
         0.12775,
         0.13575,
         0.14942,
         0.15085,
         0.15242,
         0.15492,
         0.16267,
         0.15500,
         0.16721,
         0.16035,
         0.12827]

    assert actual == pytest.approx(expected, abs=0.001)


def test_bootstrapped_vs_original_volatilities_plot(excel_file_path):
    drift: float = 0.1
    gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=drift,
            excel_file_path=excel_file_path,
            sheet_name='vol_surface',
            initial_spot=0)

    tenors: list[float] = \
        [0.0000,
         0.0833,
         0.1667,
         0.2500,
         0.5000,
         0.7500,
         1.0000,
         2.0000,
         3.0000,
         5.0000,
         7.0000,
         10.0000]

    bootstrapped_volatilities: list[float] = [float(gbm.get_time_dependent_vol(t)) for t in tenors]
    original_volatilities: list[float] = [float(gbm.get_time_dependent_vol(t, False)) for t in tenors]

    additional_annotation: str = \
        f'File: {os.path.basename(__file__)}\n' \
        f'Test: {inspect.currentframe().f_code.co_name}' \
            if TestsConfig.show_test_location \
            else None

    PlotUtils.plot_bootstrap_volatilities(tenors,
                                          original_volatilities,
                                          bootstrapped_volatilities,
                                          'Bootstrapped volatilities vs. Original volatilities',
                                          additional_annotation)


def test_annual_bootstrapped_vs_original_volatilities_plot(excel_file_path):
    drift: float = 0.1
    gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=drift,
            excel_file_path=excel_file_path,
            sheet_name='annual_vols',
            initial_spot=0)

    tenors: list[float] = \
        [0.0000,
         1.0000,
         2.0000,
         3.0000,
         4.0000,
         5.0000,
         6.0000,
         7.0000,
         8.0000,
         9.0000,
         10.0000]

    bootstrapped_volatilities: list[float] = [float(gbm.get_time_dependent_vol(t)) for t in tenors]
    original_volatilities: list[float] = [float(gbm.get_time_dependent_vol(t, False)) for t in tenors]

    additional_annotation: str = \
        f'File: {os.path.basename(__file__)}\n' \
        f'Test: {inspect.currentframe().f_code.co_name}' \
            if TestsConfig.show_test_location \
            else None

    PlotUtils.plot_bootstrap_volatilities(
        tenors,
        original_volatilities,
        bootstrapped_volatilities,
        'Annual Bootstrapped volatilities vs. Annual Original volatilities',
        additional_annotation)


def test_simulate_time_dependent_gbm_with_annual_vols(excel_file_path):
    drift: float = 0.1
    time_to_maturity = 10
    gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=drift,
            excel_file_path=excel_file_path,
            sheet_name='annual_vols',
            initial_spot=50)
    np.random.seed(999)
    gbm_paths = gbm.get_paths(number_of_paths=10_000, number_of_time_steps=50, time_to_maturity=time_to_maturity)
    time_steps = np.linspace(0, time_to_maturity, gbm_paths.shape[1])

    additional_annotation: str = \
        f'File: {os.path.basename(__file__)}\n' \
        f'Test: {inspect.currentframe().f_code.co_name}' \
            if TestsConfig.show_test_location \
            else None

    PlotUtils.plot_monte_carlo_paths(time_steps,
                                     gbm_paths,
                                     'GBM Paths for annual volatilities stepping up at 0.1',
                                     drift,
                                     additional_annotation)


def test_period_bootstrapped_vs_original_volatilities_plot(excel_file_path):
    drift: float = 0.1
    gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=drift,
            excel_file_path=excel_file_path,
            sheet_name='stepwise_vols',
            initial_spot=0)

    tenors: list[float] = \
        [0.0000,
         0.1,
         0.2,
         0.3,
         0.4,
         0.5,
         0.6,
         0.7,
         0.8,
         0.9,
         1.0]

    bootstrapped_volatilities: list[float] = [float(gbm.get_time_dependent_vol(t)) for t in tenors]
    original_volatilities: list[float] = [float(gbm.get_time_dependent_vol(t, False)) for t in tenors]

    additional_annotation: str = \
        f'File: {os.path.basename(__file__)}\n' \
        f'Test: {inspect.currentframe().f_code.co_name}' \
            if TestsConfig.show_test_location \
            else None

    PlotUtils.plot_bootstrap_volatilities(
        tenors,
        original_volatilities,
        bootstrapped_volatilities,
        'Bootstrapped volatilities vs. Original volatilities stepping up at 0.1',
        additional_annotation)


def test_simulate_time_dependent_gbm_with_periodic_vols(excel_file_path):
    drift: float = 0.1
    time_to_maturity = 10
    gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=drift,
            excel_file_path=excel_file_path,
            sheet_name='stepwise_vols',
            initial_spot=50)
    np.random.seed(999)
    gbm_paths = gbm.get_paths(number_of_paths=100_000, number_of_time_steps=50, time_to_maturity=time_to_maturity)
    time_steps = np.linspace(0, time_to_maturity, gbm_paths.shape[1])

    additional_annotation: str = \
        f'File: {os.path.basename(__file__)}\n' \
        f'Test: {inspect.currentframe().f_code.co_name}' \
            if TestsConfig.show_test_location \
            else None

    PlotUtils.plot_monte_carlo_paths(time_steps,
                                     gbm_paths,
                                     'GBM Paths for stepwise volatilities stepping up at 0.1',
                                     drift,
                                     additional_annotation)


def test_bootstrapped_vols_for_extreme_original_vols(excel_file_path):
    drift: float = 0.1
    gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=drift,
            excel_file_path=excel_file_path,
            sheet_name='extreme_vols',
            initial_spot=0)

    np.random.seed(999)
    tenors: list[float] = \
        [0.0100,
         0.0833,
         0.1667,
         0.2500,
         0.5000,
         0.7500,
         1.0000,
         2.0000,
         3.0000,
         5.0000,
         7.0000,
         10.0000]

    tenors: list[float] = [t - 0.0001 for t in tenors]
    extreme_bootstrapped_vols: list[float] = [float(gbm.get_time_dependent_vol(t)) for t in tenors]
    original_volatilities: list[float] = [float(gbm.get_time_dependent_vol(t, False)) for t in tenors]

    additional_annotation: str = \
        f'File: {os.path.basename(__file__)}\n' \
        f'Test: {inspect.currentframe().f_code.co_name}' \
            if TestsConfig.show_test_location \
            else None

    PlotUtils.plot_bootstrap_volatilities(tenors,
                                          original_volatilities,
                                          extreme_bootstrapped_vols,
                                          'Extreme Bootstrapped volatilities vs. Extreme Original volatilities',
                                          additional_annotation)

    print(f'\n')
    print(f'Extreme Bootstrapped Volatilities: {extreme_bootstrapped_vols}')


def test_simulate_time_dependent_gbm_with_extreme_vols(excel_file_path):
    drift: float = 0.1
    time_to_maturity = 4
    gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=drift,
            excel_file_path=excel_file_path,
            sheet_name='extreme_vols',
            initial_spot=50)
    np.random.seed(999)
    gbm_paths = gbm.get_paths(number_of_paths=100_000, number_of_time_steps=100, time_to_maturity=time_to_maturity)
    time_steps = np.linspace(0, time_to_maturity, gbm_paths.shape[1])

    additional_annotation: str = \
        f'File: {os.path.basename(__file__)}\n' \
        f'Test: {inspect.currentframe().f_code.co_name}' \
            if TestsConfig.show_test_location \
            else None

    PlotUtils.plot_monte_carlo_paths(time_steps,
                                     gbm_paths,
                                     'Extreme GBM Paths',
                                     drift,
                                     additional_annotation)


def test_get_vols_from_file(excel_file_path):
    actual_tenors, actual_vols = TimeDependentGBM.get_vols_from_file(excel_file_path, 'vol_surface')
    expected_tenors: list[float] = \
        [0.0, 0.0833, 0.1667, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    expected_vols: list[float] = \
        [0.12775, 0.13575, 0.14275, 0.14550, 0.14900, 0.15100, 0.15400, 0.15450, 0.15885, 0.15945, 0.15120, 0.15000]

    assert actual_tenors == pytest.approx(expected_tenors, abs=0.001)
    assert actual_vols == pytest.approx(expected_vols, abs=0.001)


def test_linear_variance_interpolator(excel_file_path):
    gbm: TimeDependentGBM = TimeDependentGBM(0.0, excel_file_path, 'vol_surface', 100)
    tenors: list[float] = \
        [0.04167,
         0.08333,
         0.12500,
         0.16667,
         0.20833,
         0.25000,
         0.37500,
         0.50000,
         0.62500,
         0.75000,
         0.87500,
         1.00000,
         1.50000,
         2.00000,
         2.50000,
         3.00000,
         4.00000,
         5.00000,
         6.00000,
         7.00000,
         8.50000,
         10.00000]

    expected_vols: list[float] = \
        [0.13575,
         0.13575,
         0.14046,
         0.14275,
         0.14441,
         0.14550,
         0.14784,
         0.14900,
         0.15020,
         0.15100,
         0.15272,
         0.15400,
         0.15433,
         0.15450,
         0.15712,
         0.15885,
         0.15923,
         0.15945,
         0.15469,
         0.15120,
         0.15050,
         0.15000]

    actual_vols = [gbm.get_time_dependent_vol(t, False) for t in tenors]
    assert actual_vols == pytest.approx(expected_vols, abs=0.0001)


def test_bootstrapped_volatility_distribution2():
    np.random.seed(999)
    number_of_samples = 100_000
    z = np.random.normal(0, 1, number_of_samples) * np.sqrt(2)
    y1 = np.random.normal(0, 1, number_of_samples) * np.sqrt(1)
    y2 = np.random.normal(0, 1, number_of_samples) * np.sqrt(1)
    y = y1 + y2
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(z, bins=75, density=True, label='Single Variable', color=colors_green)
    ax.hist(y, bins=75, density=True, label='Sum of Variables', color=colors_green)
    plt.show()


def test_bootstrapped_volatility_distribution():
    number_of_samples = 1_000_000
    sigma_5 = 0.1490  # 0.142750  # 0.13575

    bootstrapped_sigma_1 = 0.12775
    bootstrapped_sigma_2 = 0.13575
    bootstrapped_sigma_3 = 0.14942
    bootstrapped_sigma_4 = 0.15085
    bootstrapped_sigma_5 = 0.15242

    np.random.seed(999)
    z = sigma_5 * np.random.normal(0, 1, number_of_samples) * np.sqrt(180 / 360)

    np.random.seed(999)
    y1 = bootstrapped_sigma_1 * np.random.normal(0, 1, number_of_samples) * np.sqrt(0 / 360)
    y2 = bootstrapped_sigma_2 * np.random.normal(0, 1, number_of_samples) * np.sqrt(30 / 360)
    y3 = bootstrapped_sigma_3 * np.random.normal(0, 1, number_of_samples) * np.sqrt(30 / 360)
    y4 = bootstrapped_sigma_4 * np.random.normal(0, 1, number_of_samples) * np.sqrt(30 / 360)
    y5 = bootstrapped_sigma_5 * np.random.normal(0, 1, number_of_samples) * np.sqrt(90 / 360)
    y = y1 + y2 + y3 + y4 + y5
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.histplot(z, bins=75, stat='density', element='step', legend=True, color=colors_teal)
    values, bins = np.histogram(z, bins=75, density=True)
    # y_dataframe = pd.DataFrame({'Summed Random Variables': y})
    sns.histplot(y, bins=75, stat='density', element='step', legend=True, color=colors_green)
    ax.legend(labels=['Z', 'Y'])
    plt.show()
