import numpy as np
import pytest
from src.gbm.time_independent_gbm import TimeIndependentGBM
from src.enums_and_named_tuples.path_statistics import PathStatistics


def test_distribution():
    """
    The theoretical mean of the GBM paths is given by:

    :math:`\\mathbb{E}[S_t] = S_0 e^{\\sigma T}`

    The theoretical variance of the GBM paths is given by:

    :math:`\\mathbb{Var}[S_t] = S^2_0 e^{2\\mu T}\\left(e^{\\sigma^2 T} - 1\\right)`

    :return: None.
    """
    np.random.seed(999)
    time_to_maturity: float = 1.0
    gbm: TimeIndependentGBM = TimeIndependentGBM(drift=0.0, volatility=0.4, initial_spot=100)
    paths: np.ndarray = gbm.get_paths(number_of_paths=10_000,
                                      number_of_time_steps=20,
                                      time_to_maturity=time_to_maturity)
    gbm.create_plots(paths, time_to_maturity)
    path_stats: PathStatistics = gbm.get_path_statistics(paths, time_to_maturity)
    assert path_stats.EmpiricalMean == pytest.approx(path_stats.TheoreticalMean, abs=1.00)
    assert path_stats.EmpiricalStandardDeviation == pytest.approx(path_stats.TheoreticalStandardDeviation, abs=1.00)
