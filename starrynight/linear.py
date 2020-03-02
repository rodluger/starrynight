from .utils import *
from .special import pal
import numpy as np

TWOTHIRDS = 2.0 / 3.0


def dP2(bo, ro, k2, kappa, s1, s2, c1, F, E, PIprime):
    """
    Returns the difference of a pair (or pairs) of Pal integrals for the
    P2 (linear) term. Specifically, returns the sum of

        P2(bo, ro, kappa[i + 1]) - P2(bo, ro, kappa[i])

    for i = 0, 2, 4, ...

    """

    # Useful variables
    r2 = ro * ro
    b2 = bo * bo
    br = bo * ro
    bpr = bo + ro
    bmr = bo - ro
    d2 = r2 + b2 - 2 * br
    term = 0.5 / np.sqrt(br * k2)
    p0 = 4.0 - 7.0 * r2 - b2
    q2 = r2 + b2 - 2 * br * (1 - 2 * s2)

    # Special cases
    if bo == 0.0:

        # Analytic limit
        if ro < 1.0:
            return (1 - (1 - r2) * np.sqrt(1 - r2)) * pairdiff(kappa) / 3.0
        else:
            return pairdiff(kappa) / 3.0

    elif np.abs(bo - ro) < STARRY_PAL_BO_EQUALS_RO_TOL:

        # Solve numerically
        return pal(bo, ro, kappa)

    elif np.abs(bo - (ro - 1)) < STARRY_PAL_BO_EQUALS_RO_MINUS_ONE_TOL:

        # Solve numerically
        return pal(bo, ro, kappa)

    elif np.abs(bo - (1 - ro)) < STARRY_PAL_BO_EQUALS_ONE_MINUS_RO_TOL:

        # Solve numerically
        return pal(bo, ro, kappa)

    # Constant term
    if bo == ro:
        a0 = 0
    else:
        a0 = -pairdiff(
            np.arctan2(-bmr * c1, bpr * s1)
            + 2 * np.pi * np.sign(bmr) * (kappa > 3 * np.pi)
        )
    a1 = 0.5 * pairdiff(kappa)
    a2 = -2.0 * pairdiff(s1 * c1 * np.sqrt(1 - np.minimum(1.0, q2)))
    A = a0 + a1 + TWOTHIRDS * br * a2

    # Carlson RD term
    C = -2 * TWOTHIRDS * br * p0 * term * k2

    # Carlson RF term
    fac = -bpr / bmr
    B = (
        -((1.0 + 2.0 * r2 * r2 - 4.0 * r2) + TWOTHIRDS * br * (p0 + 5 * br) + fac)
        * term
    ) - C

    # Carlson PIprime term
    D = -TWOTHIRDS * fac / d2 * term * br

    return (A + B * F + C * E + D * PIprime) / 3.0
