"""
Contains an enum for specifying whether the contract/option is taking a long or short position.
"""
from enum import Enum


class LongOrShort(Enum):
    """
    Enum for specifying whether the contract/option is taking a long or short position.
    """
    LONG = 0,
    SHORT = 1
