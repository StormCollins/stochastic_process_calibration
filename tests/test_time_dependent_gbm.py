"""
Time-dependent GBM unit tests.
"""
import numpy as np
import pytest
from src.gbm.time_dependent_gbm import TimeDependentGBM
from src.gbm.time_independent_gbm import TimeIndependentGBM
from src.enums_and_named_tuples.path_statistics import PathStatistics


def test_get_time_dependent_volatility_for_constant_vol():
    drift: float = 0.1
    volatility: float = 0.4
    excel_file_path = r'tests/equity-atm-volatility-surface.xlsx'

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


def test_get_time_dependent_volatility_for_non_constant_vol():
    drift: float = 0.1
    volatility: float = 0.4
    excel_file_path = r'tests/equity-atm-volatility-surface.xlsx'

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


def test_get_time_dependent_gbm_paths_for_constant_vols():
    drift: float = 0.1
    volatility: float = 0.4
    excel_file_path = 'tests/equity-atm-volatility-surface.xlsx'
    number_of_paths: int = 10
    number_of_time_steps: int = 2
    initial_spot: float = 50
    time_to_maturity = 1
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
        TimeIndependentGBM(drift, volatility, initial_spot)

    np.random.seed(999)
    expected_paths = time_independent_gbm.get_paths(number_of_paths, number_of_time_steps, time_to_maturity)
    assert actual_paths == pytest.approx(expected_paths, abs=1.0)


def test_distribution():
    np.random.seed(999)
    time_to_maturity: float = 1.0
    gbm: TimeDependentGBM = \
        TimeDependentGBM(0.0, 'tests/equity-atm-volatility-surface.xlsx', 'constant_vol_surface', 100)

    paths: np.ndarray = gbm.get_paths(10_000, 10, time_to_maturity)
    gbm.create_plots(paths, 1.0)
    path_stats: PathStatistics = gbm.get_path_statistics(paths, time_to_maturity)
    assert path_stats.EmpiricalMean == pytest.approx(path_stats.TheoreticalMean, abs=1.00)
    assert path_stats.EmpiricalStandardDeviation == pytest.approx(path_stats.TheoreticalStandardDeviation, abs=1.00)


def test_bootstrapped_vols_for_non_constant_vol_term_structure():
    drift: float = 0.1
    excel_file_path = r'tests/equity-atm-volatility-surface.xlsx'
    gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=drift,
            excel_file_path=excel_file_path,
            sheet_name='vol_surface',
            initial_spot=0)

    np.random.seed(999)
    tenors = [0.0000, 0.0833, 0.1667, 0.2500, 0.5000, 0.7500, 1.0000, 2.0000, 3.0000, 5.0000, 7.0000, 10.0000]
    tenors = [t - 0.0001 for t in tenors]
    actual = [float(gbm.get_time_dependent_vol(t)) for t in tenors]
    expected = [0.12775, 0.13575, 0.14942, 0.15085, 0.15242, 0.15492, 0.16267, 0.15500, 0.16721, 0.16035, 0.12827, 0.14716]
    assert actual == pytest.approx(expected, abs=0.00001)
