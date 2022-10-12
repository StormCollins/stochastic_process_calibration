"""
Contains an enum used to specify Hull-White simulation methods.
"""
from enum import Enum


class HullWhiteSimulationMethod(Enum):
    """
    Used to specify Hull-White simulation methods.
    """
    SLOWANALYTICAL = 1
    FASTANALYTICAL = 2
    SLOWAPPROXIMATE = 3


