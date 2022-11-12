import numpy as np
from src.curves.curve import Curve
from src.enums_and_named_tuples.compounding_convention import CompoundingConvention
from typing import Callable


class SimulatedCurves:
    def __init__(
            self,
            a_function: Callable[[float | np.ndarray, float | np.ndarray], float | np.ndarray],
            b_function: Callable[[float | np.ndarray, float | np.ndarray], float | np.ndarray],
            simulation_tenors: np.ndarray,
            short_rates: np.ndarray):
        self.a_function: Callable[[float | np.ndarray, float | np.ndarray], float | np.ndarray] = a_function
        self.b_function: Callable[[float | np.ndarray, float | np.ndarray], float | np.ndarray] = b_function
        self.simulation_tenors: np.ndarray = simulation_tenors
        self.short_rates: np.ndarray = short_rates

    def get_discount_factors(self, simulation_tenor: float, end_tenor: float) -> np.ndarray:
        current_short_rates: np.ndarray = self.short_rates[:, np.where(self.simulation_tenors == simulation_tenor)]
        return self.a_function(simulation_tenor, simulation_tenor + end_tenor) * \
            np.exp(-1 * current_short_rates * self.b_function(simulation_tenor, simulation_tenor + end_tenor))

    def get_forward_rates(
            self,
            simulation_tenor: float,
            start_tenor: float,
            end_tenor: float) -> np.ndarray:
        # TODO: Add functionality to get rates for different compounding conventions. Currently just does NACQ.
        near_discount_factors: np.ndarray = self.get_discount_factors(simulation_tenor, start_tenor)
        far_discount_factors: np.ndarray = self.get_discount_factors(simulation_tenor, end_tenor)
        return 4 * ((near_discount_factors / far_discount_factors)**(1 / (4 * (end_tenor - start_tenor))) - 1)
