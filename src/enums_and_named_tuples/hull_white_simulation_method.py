"""
Contains an enum used to specify Hull-White simulation methods.
"""
from enum import Enum


class HullWhiteSimulationMethod(Enum):
    """
    Used to specify Hull-White simulation methods.
    """
    DISCRETISED_INTEGRAL = 1,
    DISCRETISED_SDE = 2


