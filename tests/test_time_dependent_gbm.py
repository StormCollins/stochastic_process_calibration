import numpy as np
import pytest
from src.gbm.time_dependent_gbm import TimeDependentGBM
from src.gbm.time_independent_gbm import TimeIndependentGBM
from src.path_statistics import PathStatistics


def test_get_time_dependent_volatility():
    drift: float = 0.1
    volatility: float = 0.4
    excel_file_path = r'tests/atm-volatility-surface.xlsx'

    gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=drift,
            excel_file_path=excel_file_path,
            sheet_name='constant_vol_surface',
            initial_spot=0)

    np.random.seed(999)
    time_steps = np.arange(0, 10, 1) + 1
    actual_vols = [gbm.get_time_dependent_vol(t) for t in time_steps]
    expected_vols = list(np.repeat(volatility, 10))
    assert actual_vols == expected_vols


def test_get_time_dependent_gbm_paths_for_constant_vols():
    drift: float = 0.1
    volatility: float = 0.4
    excel_file_path = 'tests/atm-volatility-surface.xlsx'
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
        TimeDependentGBM(0.0, 'tests/atm-volatility-surface.xlsx', 'constant_vol_surface', 100)

    paths: np.ndarray = gbm.get_paths(10_000, 10, time_to_maturity)
    gbm.create_plots(paths, 1.0)
    path_stats: PathStatistics = gbm.get_path_statistics(paths, time_to_maturity)
    assert path_stats.EmpiricalMean == pytest.approx(path_stats.TheoreticalMean, abs=1.00)
    assert path_stats.EmpiricalStandardDeviation == pytest.approx(path_stats.TheoreticalStandardDeviation, abs=1.00)
