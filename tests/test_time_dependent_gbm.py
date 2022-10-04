import numpy as np
import pytest
from src.gbm.time_dependent_gbm import TimeDependentGBM
from src.gbm.time_independent_gbm import TimeIndependentGBM


def test_get_time_dependent_volatility():
    drift: float = 0.1
    volatility: float = 0.4
    excel_file_path = r'tests/atm-volatility-surface.xlsx'

    gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=drift,
            excel_file_path=excel_file_path,
            sheet_name='constant_vol_surface',
            initial_spot=0,
            time_to_maturity=0)

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

    time_dependent_gbm: TimeDependentGBM = \
        TimeDependentGBM(
            drift=drift,
            excel_file_path=excel_file_path,
            sheet_name='constant_vol_surface',
            initial_spot=initial_spot,
            time_to_maturity=time_to_maturity)

    np.random.seed(999)

    actual_paths = \
        time_dependent_gbm.get_gbm_paths(
            number_of_paths=number_of_paths,
            number_of_time_steps=number_of_time_steps,
            initial_spot=initial_spot,
            time_to_maturity=time_to_maturity)

    time_independent_gbm: TimeIndependentGBM = \
        TimeIndependentGBM(drift, volatility, initial_spot, time_to_maturity)

    np.random.seed(999)
    expected_paths = time_independent_gbm.get_paths(number_of_paths, number_of_time_steps)

    assert actual_paths == pytest.approx(expected_paths, 0.001)
