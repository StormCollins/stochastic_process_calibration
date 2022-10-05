from enum import Enum


class LongOrShort(Enum):
    """
    Class for specifying whether the contract/option is taking a long position or a short position.
    """
    LONG = 0,
    SHORT = 1
