"""
Contains an enum used to specify the method of integration in the stochastic component of Hull-White.
"""
from enum import Enum


class HullWhiteStochasticIntegralMethod(Enum):
    """
    These specify the two, but for small enough time steps, equivalent approaches to calculating the
    stochastic integral

    :math:`\\int_0^t e^{\\alpha u} dW(u)`

    The first method is equivalent to a naive Riemann sum approach.
    The second is more accurate and uses Ito's isometry.

    See: https://wiki.fsa-aks.deloitte.co.za/doku.php?id=valuations:methodology:models:hull_white#an_aside_on_the_stochastic_integral
    for more details.

    """
    SIMPLE_RIEMANN_SUM = 1,
    ITO_ISOMETRY = 2,
