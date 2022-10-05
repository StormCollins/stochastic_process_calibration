from enum import Enum


class CallOrPut(Enum):
    """
    Class for specifying whether the option is a call option or a put option.
    """

    CALL = 0
    PUT = 1
