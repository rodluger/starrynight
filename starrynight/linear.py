import numpy as np
from scipy.integrate import quad
from .special import carlson_rf, carlson_rd, carlson_rj


def pal_indef(bo, ro, phi):
    """
    This is adapted from the `mttr_integral_primitive` function 
    in the MTTR code of Pal (2012).
    
    """
    q2 = ro * ro + bo * bo + 2 * ro * bo * np.cos(phi)
    d2 = ro * ro + bo * bo - 2 * ro * bo
    sx = np.sin(phi / 2)
    cx = np.cos(phi / 2)

    if q2 > 1.0:
        # TODO: solve the case q2 = 1 separately
        q2 = 1.0 - 1e-8

    if d2 >= 1.0:
        # TODO: solve the case d2 = 1 separately
        d2 = 1.0 - 1e-8

    w = (1 - q2) / (1 - d2)

    # Elliptic integrals
    rf = carlson_rf(w, sx * sx, 1.0)
    rd = carlson_rd(w, sx * sx, 1.0)
    if ro != bo:
        rj = carlson_rj(w, sx * sx, 1.0, q2 / d2)
    else:
        rj = 0.0

    # Equation (34) in Pal (2012)
    beta = np.arctan2((bo - ro) * sx, (bo + ro) * cx)
    w = cx / np.sqrt(1 - d2)
    iret = (
        -beta / 3.0
        + phi / 6.0
        + 2.0 / 9.0 * bo * ro * np.sin(phi) * np.sqrt(1 - q2)
        + 1.0 / 3.0 * (1 + 2 * ro * ro * ro * ro - 4 * ro * ro) * w * rf
        + 2.0 / 9.0 * ro * bo * (4 - 7 * ro * ro - bo * bo + 5 * ro * bo) * w * rf
        - 4.0 / 27.0 * ro * bo * (4 - 7 * ro * ro - bo * bo) * w * cx * cx * rd
    )
    if ro != bo:
        iret += 1.0 / 3.0 * w * (ro + bo) / (ro - bo) * (rf - (q2 - d2) / (3 * d2) * rj)
    else:
        iret -= 1.0 / 3.0 * w * (ro + bo) * (q2 - d2) * np.pi / (2 * q2 * np.sqrt(q2))

    return iret


def pal(bo, ro, phi1, phi2):
    """
    This is adapted from the `mttr_integral_definite` function 
    in the MTTR code of Pal (2012).

    """
    if bo == 0.0:
        if ro < 1.0:
            return (1 - (1 - ro ** 2) * np.sqrt(1 - ro ** 2)) * (phi2 - phi1) / 3.0
        else:
            return (phi2 - phi1) / 3.0

    # Preserve the integration order
    if phi1 > phi2:
        sgn = -1
    else:
        sgn = 1

    # Choose our quadrants intelligently
    x0 = phi1
    dx = phi2 - phi1
    if dx < 0.0:
        x0 += dx
        dx = -dx
    while x0 < 0.0:
        x0 += 2 * np.pi
    while 2 * np.pi <= x0:
        x0 -= 2 * np.pi
    ret = 0.0
    while 0.0 < dx:
        dc = 2 * np.pi - x0
        if dx < dc:
            dc = dx
            nx = x0 + dx
        else:
            nx = 0.0

        # Now actually compute the integral
        ret += sgn * (pal_indef(bo, ro, x0 + dc) - pal_indef(bo, ro, x0))

        x0 = nx
        dx -= dc

    return ret
