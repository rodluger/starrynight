from .utils import *
import numpy as np


def dP2(bo, ro, k2, kappa, RF, RD, RJ, RF0, RD0, RJ0):
    """
    Returns the difference of a pair (or pairs) of Pal integrals for the
    P2 (linear) term. Specifically, returns the sum of

        P2(bo, ro, phi[i + 1]) - P2(bo, ro, phi[i])

    for i = 0, 2, 4, ...

    """
    # There's an offset between our angle and Pal's angle
    phi = (kappa - np.pi) % (2 * np.pi)

    # TODO: Solve this case separately
    if np.abs(bo - (ro - 1)) < 1e-8:
        s = np.sign(bo - (ro - 1))
        if s == 0:
            s = 1
        bo = ro - 1 + s * 1e-8

    # TODO: Solve this case separately
    if np.abs(bo - (1 - ro)) < 1e-8:
        s = np.sign(bo - (1 - ro))
        if s == 0:
            s = 1
        bo = 1 - ro + s * 1e-8

    # Compute all the antiderivatives
    res = pal(bo, ro, k2, RF, RD, RJ, phi)

    # The function in Pal (2012) is restricted to [0, 2pi]
    # but our domain is [-pi/2, 2pi + pi/2]. We need to
    # add an offset term to patch the discontinuous jumps.
    if np.any(kappa < np.pi) or np.any(kappa > 3 * np.pi):
        offset = pal(bo, ro, k2, RF0, RD0, RJ0, np.array([0, 2 * np.pi]))
        for i, kn in enumerate(kappa):
            if kn < np.pi:
                if i % 2 == 0:
                    res += offset
                else:
                    res -= offset
            elif kn > 3 * np.pi:
                if i % 2 == 0:
                    res -= offset
                else:
                    res += offset

    return res


def pal(bo, ro, k2, RF, RD, RJ, phi):
    """
    This is adapted from the `mttr_integral_primitive` function 
    in the MTTR code of Pal (2012). This is a vectorized function
    for an array of `phi` values.

    """

    # Special case
    if bo == 0.0:
        if ro < 1.0:
            return (1 - (1 - ro ** 2) * np.sqrt(1 - ro ** 2)) * pairdiff(phi) / 3.0
        else:
            return pairdiff(phi) / 3.0

    r2 = ro * ro
    b2 = bo * bo
    br = bo * ro
    bpr = bo + ro
    bmr = bo - ro
    d2 = r2 + b2 - 2 * br
    term = 0.5 / np.sqrt(br * k2)
    f23 = 2.0 / 3.0
    p0 = 4.0 - 7.0 * r2 - b2

    q2 = r2 + b2 + 2 * br * np.cos(phi)
    sx = np.sin(phi / 2)
    cx = np.cos(phi / 2)

    # Constant term
    a0 = pairdiff(-np.arctan2(bmr * sx, bpr * cx))
    a1 = 0.5 * pairdiff(phi)
    a2 = pairdiff(np.sin(phi) * np.sqrt(1 - np.minimum(1.0, q2)))
    A = a0 + a1 + f23 * br * a2

    # Carlson RF term
    B = ((1.0 + 2.0 * r2 * r2 - 4.0 * r2) + f23 * br * (p0 + 5 * br)) * term

    # Carlson RD term
    C = -f23 * f23 * br * p0 * term

    # Carlson RJ term
    if np.abs(bo - ro) > STARRY_PAL_BO_EQUALS_RO_TOL:

        p4 = -bpr / bmr
        B += p4 * term
        D = -f23 * p4 / d2 * term * br

    else:

        a3 = 2.0 * br * (np.cos(phi) + 1)
        a4 = pairdiff(a3 * cx / (q2 * np.sqrt(q2)))
        if bo < ro:
            A -= 2.0 * a0
        A -= 0.5 * np.pi * term * bpr * a4
        D = 0

    return (A + B * RF + C * RD + D * RJ) / 3.0
