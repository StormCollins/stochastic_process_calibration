"""
Time-dependent GBM unit tests.
"""
import inspect
import numpy as np
import os.path
import pytest
from src.gbm.time_dependent_gbm import TimeDependentGBM
from src.gbm.time_independent_gbm import TimeIndependentGBM
from src.enums_and_named_tuples.path_statistics import PathStatistics
from tests_config import TestsConfig


@pytest.fixture
def excel_file_path() -> str:
    return r'tests/equity-atm-volatility-surface.xlsx'


def test_get_time_dependent_volatility_for_constant_vol(excel_file_path):
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


def test_get_time_dependent_volatility_for_non_constant_vol(excel_file_path):
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


def test_get_time_dependent_gbm_paths_for_constant_vols(excel_file_path):
    drift: float = 0.1
    volatility: float = 0.4
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
        TimeDependentGBM(drift=0.0,
                         excel_file_path=excel_file_path,
                         sheet_name='constant_vol_surface',
                         initial_spot=100)

    paths: np.ndarray = gbm.get_paths(10_000, 10, time_to_maturity)
    if TestsConfig.plots_on:
        gbm.create_plots(paths, 1.0, f'{os.path.basename(__file__)} : {inspect.currentframe().f_code.co_name}')

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
