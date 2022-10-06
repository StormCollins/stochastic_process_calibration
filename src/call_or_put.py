from enum import Enum


class CallOrPut(Enum):
    """
    Enum for specifying whether an option is a call or a put.
    """
    CALL = 0
    PUT = 1
