import numpy as np
import pytest
from src.gbm.time_independent_gbm import TimeIndependentGBM


def test_distribution():
    """
    The theoretical mean of the GBM paths is given by:

    :math:`\\mathbb{E}[S_t] = S_0 e^{\\sigma T}`

    The theoretical variance of the GBM paths is given by:

    :math:`\\mathbb{Var}[S_t] = S^2_0 e^{2\\mu T}\\left(e^{\\sigma^2 T} - 1\\right)`

    :return: None.
    """
    np.random.seed(999)
    gbm: TimeIndependentGBM = TimeIndependentGBM(0.0, 0.1, 100, 1.0)
    paths: np.ndarray = gbm.get_paths(100_000, 1)
    gbm.create_plots(paths)
    gbm.get_path_statistics(paths)
