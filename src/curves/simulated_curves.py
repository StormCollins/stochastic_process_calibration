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
        # We assume that the curves only go out to 30 years.
        # Furthermore, interpolation is done for tenors not multiples of 0.05.
        # This can be enhanced later.
        curve_tenors: np.ndarray = np.arange(0, 30.05, 0.05)
        self.curves: dict[float, Curve] = dict()

        for t in simulation_tenors:
            a_values = \
                np.tile(
                    self.a_function(simulation_tenors[0], simulation_tenors[0] + curve_tenors),
                    (short_rates.shape[0], 1))

            b_values = \
                np.tile(
                    self.b_function(simulation_tenors[0], simulation_tenors[0] + curve_tenors),
                    (short_rates.shape[0], 1))

            current_time_step_short_rates: np.ndarray = np.transpose(np.tile(short_rates[:, 0], (len(curve_tenors), 1)))
            discount_factors: np.ndarray = a_values * np.exp(-1 * current_time_step_short_rates * b_values)
            current_time_step_curves: Curve = Curve(curve_tenors, discount_factors)
            self.curves[t] = current_time_step_curves

    # TODO: Rewrite this function to use the a_function and b_function to be more accurate.
    def get_discount_factors(self, simulation_tenor: float, end_tenor: float) -> np.ndarray:
        return self.curves[simulation_tenor].get_discount_factors(end_tenor)

    def get_forward_rates(
            self,
            simulation_tenor: float,
            start_tenor: float,
            end_tenor: float,
            compounding_convention: CompoundingConvention) -> np.ndarray:
        return self.curves[simulation_tenor].get_forward_rates(start_tenor, end_tenor, compounding_convention)
