from .utils import *
from .special import carlson_rf, carlson_rd, carlson_rj
from .special import el2, ellip
import numpy as np


def cxrf_el2(k2, phi):
    sx = np.sin(phi / 2)
    cx = np.cos(phi / 2)
    km2 = 1 / k2

    if km2 > 1:

        k = np.sqrt(km2)
        kc2 = 1 - 1 / km2
        kc = np.sqrt(kc2)
        arg = k * cx
        tanphi = arg / np.sqrt(1 - arg ** 2)
        tanphi[arg >= 1] = 1e30
        tanphi[arg <= -1] = -1e30
        F = el2(tanphi, kc, 1, 1) / k

    else:

        kc = np.sqrt(1 - km2)
        tanphi = cx / sx
        tanphi[(cx > 0) & (sx == 0)] = 1e30
        tanphi[(cx < 0) & (sx == 0)] = -1e30
        F = el2(tanphi, kc, 1, 1)
        F[(phi < 0) | (phi > 2 * np.pi)] *= -1

    return pairdiff(F)


def cx3rd_el2(k2, phi):
    sx = np.sin(phi / 2)
    cx = np.cos(phi / 2)
    km2 = 1 / k2

    if km2 > 1:

        k = np.sqrt(km2)
        kc2 = 1 - 1 / km2
        kc = np.sqrt(kc2)
        arg = k * cx
        tanphi = arg / np.sqrt(1 - arg ** 2)
        tanphi[arg >= 1] = 1e30
        tanphi[arg <= -1] = -1e30
        F = el2(tanphi, kc, 1, 1) / k
        E = el2(tanphi, kc, 1, kc2) / k
        D = (F - E) * 3

    else:

        kc = np.sqrt(1 - km2)
        tanphi = cx / sx
        tanphi[(cx > 0) & (sx == 0)] = 1e30
        tanphi[(cx < 0) & (sx == 0)] = -1e30
        F = el2(tanphi, kc, 1, 1)
        F[(phi < 0) | (phi > 2 * np.pi)] *= -1
        E = el2(tanphi, kc, 1, 1 - km2)
        E[(phi < 0) | (phi > 2 * np.pi)] *= -1
        D = (F - E) * 3 * k2

    return pairdiff(D)


def frj_el3(bo, ro, k2, phi):
    # TODO
    p = (ro * ro + bo * bo + 2 * ro * bo * np.cos(phi)) / (
        ro * ro + bo * bo - 2 * ro * bo
    )
    cx = np.cos(phi / 2)
    sx = np.sin(phi / 2)
    w = 1 - cx ** 2 / k2

    if np.abs(bo - ro) > STARRY_PAL_BO_EQUALS_RO_TOL:
        frj = (
            (np.cos(phi) + 1)
            * cx
            * np.array(
                [carlson_rj(w[i], sx[i] * sx[i], 1.0, p[i]) for i in range(len(w))]
            )
        )
    else:
        frj = np.zeros_like(phi)

    return pairdiff(frj)


def dP2(bo, ro, k2, kappa, RF, RD, RJ, RF0, RD0, RJ0):
    """
    Returns the difference of a pair (or pairs) of Pal integrals for the
    P2 (linear) term. Specifically, returns the sum of

        P2(bo, ro, phi[i + 1]) - P2(bo, ro, phi[i])

    for i = 0, 2, 4, ...

    """
    # There's an offset between our angle and Pal's angle
    phi = (kappa - np.pi) % (2 * np.pi)

    # Compute all the antiderivatives
    res = pal_indef(bo, ro, k2, RF, RD, RJ, phi)

    # The function in Pal (2012) is restricted to [0, 2pi]
    # but our domain is [-pi/2, 2pi + pi/2]. We need to
    # add an offset term to patch the discontinuous jumps.
    # Note that this term is a function of *complete* elliptic
    # integrals, should we ever want to speed this up.
    if np.any(kappa < np.pi) or np.any(kappa > 3 * np.pi):

        # RF0 = -2 * K0  # cxrf_el2(k2, np.array([0, 2 * np.pi]))
        # RD0 = 2 * (E0 - K0) * 3 * k2  # cx3rd_el2(k2, np.array([0, 2 * np.pi]))
        # RJ0 = frj_el3(bo, ro, k2, np.array([0, 2 * np.pi]))

        offset = pal_indef(bo, ro, k2, RF0, RD0, RJ0, np.array([0, 2 * np.pi]))

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


def pal_indef(bo, ro, k2, RF, RD, RJ, phi):
    """
    This is adapted from the `mttr_integral_primitive` function 
    in the MTTR code of Pal (2012). This is a vectorized function
    for an array of `phi` values.

    """
    if bo == 0.0:
        if ro < 1.0:
            return (1 - (1 - ro ** 2) * np.sqrt(1 - ro ** 2)) * pairdiff(phi) / 3.0
        else:
            return pairdiff(phi) / 3.0

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

    q2 = ro * ro + bo * bo + 2 * ro * bo * np.cos(phi)
    d2 = ro * ro + bo * bo - 2 * ro * bo
    sx = np.sin(phi / 2)
    cx = np.cos(phi / 2)
    r2 = ro * ro
    b2 = bo * bo
    br = bo * ro
    term = 0.5 / np.sqrt(br * k2)
    c0 = pairdiff(-np.arctan2((bo - ro) * sx, (bo + ro) * cx))
    c1 = 0.5 * pairdiff(phi)
    c2 = pairdiff(np.sin(phi) * np.sqrt(1 - np.minimum(1.0, q2)))
    f23 = 2.0 / 3.0
    p1 = 4.0 - 7.0 * r2 - b2

    res = (
        c0
        + c1
        + f23 * br * c2
        + (1.0 + 2.0 * r2 * r2 - 4.0 * r2) * term * RF
        + f23 * br * (p1 + 5 * br) * term * RF
        - f23 * f23 * br * p1 * term * RD
    )
    if np.abs(bo - ro) > STARRY_PAL_BO_EQUALS_RO_TOL:
        p2 = (ro + bo) / (ro - bo)
        res += p2 * term * RF
        res -= f23 * p2 / d2 * term * br * RJ
    else:

        foo = 2 * br * (np.cos(phi) + 1)
        c3 = pairdiff(foo * cx / (q2 * np.sqrt(q2)))
        if bo < ro:
            res -= 2.0 * c0
        res -= 0.5 * np.pi * term * (ro + bo) * c3

    return res / 3.0
