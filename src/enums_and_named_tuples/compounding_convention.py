"""
Contains an enum representing interest rate compounding conventions.
"""
from enum import Enum


class CompoundingConvention(Enum):
    """
    An enum representing interest rate compounding conventions.
    """
    Simple = 1
    NACC = 2
    NACQ = 3
