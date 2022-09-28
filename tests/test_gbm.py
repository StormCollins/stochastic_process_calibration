import pytest
from src.gbm.gbm_simulation import GBM
import numpy as np


def test_get_time_dependent_volatility():
    drift: float = 0.1
    volatility: float = 0.10
    excel_file_path = r'atm-volatility-surface.xlsx'
    gbm: GBM = GBM(drift, volatility, excel_file_path, 'constant_vol_surface')
    np.random.seed(999)
    time_steps = np.arange(0, 10, 1) + 1
    actual_vols = [gbm.get_time_dependent_vol(t) for t in time_steps]
    expected_vols = list(np.repeat(volatility, 10))
    assert actual_vols == expected_vols


def test_get_time_dependent_gbm_paths_for_constant_vols():
    drift: float = 0.1
    volatility: float = 0.10
    excel_file_path = 'atm-volatility-surface.xlsx'
    number_of_paths: int = 10
    number_of_time_steps: int = 2
    notional: float = 1
    initial_spot: float = 50
    time_to_maturity = 1
    gbm: GBM = GBM(drift, volatility, excel_file_path, 'constant_vol_surface')
    np.random.seed(999)
    actual_paths = \
        gbm.get_gbm_paths(number_of_paths, number_of_time_steps, notional, initial_spot, time_to_maturity, True)
    np.random.seed(999)
    expected_paths = \
        gbm.get_gbm_paths(number_of_paths, number_of_time_steps, notional, initial_spot, time_to_maturity, False)
    assert actual_paths == pytest.approx(expected_paths, 0.001)


