import numpy as np
from .utils import *


def on_dayside(b, theta, x, y):
    """Return True if a point is on the dayside."""
    if x ** 2 + y ** 2 > 1:
        raise ValueError("Point not on the unit disk.")
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    term = 1 - xr ** 2
    yt = b * np.sqrt(term)
    return yr >= yt


def sort_phi(b, theta, bo, ro, phi, tol=1e-7):
    # Sort a pair of `phi` angles according to the order
    # of the integration limits.
    phi1, phi2 = phi
    phi = np.array([phi1, phi2]) % (2 * np.pi)
    if phi[1] < phi[0]:
        phi[1] += 2 * np.pi
    x = ro * np.cos(phi[0] + tol)
    y = bo + ro * np.sin(phi[0] + tol)
    if (x ** 2 + y ** 2 > 1) or not on_dayside(b, theta, x, y):
        phi = np.array([phi2, phi1]) % (2 * np.pi)
    if phi[1] < phi[0]:
        phi[1] += 2 * np.pi
    return phi


def sort_xi(b, theta, bo, ro, xi, tol=1e-7):
    # Sort a pair of `xi` angles according to the order
    # of the integration limits.
    xi1, xi2 = xi
    xi = np.array([xi1, xi2]) % (2 * np.pi)
    if xi[0] < xi[1]:
        xi[0] += 2 * np.pi
    x = np.cos(theta) * np.cos(xi[1] + tol) - b * np.sin(theta) * np.sin(xi[1] + tol)
    y = np.sin(theta) * np.cos(xi[1] + tol) + b * np.cos(theta) * np.sin(xi[1] + tol)
    if x ** 2 + (y - bo) ** 2 > ro ** 2:
        xi = np.array([xi2, xi1]) % (2 * np.pi)
    if xi[0] < xi[1]:
        xi[0] += 2 * np.pi
    return xi


def sort_lam(b, theta, bo, ro, lam, tol=1e-7):
    # Sort a pair of `lam` angles according to the order
    # of the integration limits.
    lam1, lam2 = lam
    lam = np.array([lam1, lam2]) % (2 * np.pi)
    if lam[1] < lam[0]:
        lam[1] += 2 * np.pi
    x = np.cos(lam[0] + tol)
    y = np.sin(lam[0] + tol)
    if x ** 2 + (y - bo) ** 2 > ro ** 2:
        lam = np.array([lam2, lam1]) % (2 * np.pi)
    if lam[1] < lam[0]:
        lam[1] += 2 * np.pi
    return lam


def get_angles(b, theta, bo, ro, tol=1e-7):

    # Trivial cases
    if bo <= ro - 1:

        # Complete occultation
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            FLUX_ZERO,
        )

    elif bo >= 1 + ro:

        # No occultation
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            FLUX_SIMPLE_REFL,
        )

    # TODO: Use Sturm's theorem here to save time?

    # We'll solve for occultor-terminator intersections
    # in the frame where the semi-major axis of the
    # terminator ellipse is aligned with the x axis
    xo = bo * np.sin(theta)
    yo = bo * np.cos(theta)

    # Special case: b = 0
    if np.abs(b) < tol:

        x = np.array([])
        term = np.sqrt(ro ** 2 - yo ** 2)
        if np.abs(xo + term) < 1:
            x = np.append(x, xo + term)
        if np.abs(xo - term) < 1:
            x = np.append(x, xo - term)

    # Need to solve a quartic
    else:

        A = (1 - b ** 2) ** 2
        B = -4 * xo * (1 - b ** 2)
        C = -2 * (
            b ** 4
            + ro ** 2
            - 3 * xo ** 2
            - yo ** 2
            - b ** 2 * (1 + ro ** 2 - xo ** 2 + yo ** 2)
        )
        D = -4 * xo * (b ** 2 - ro ** 2 + xo ** 2 + yo ** 2)
        E = (
            b ** 4
            - 2 * b ** 2 * (ro ** 2 - xo ** 2 + yo ** 2)
            + (ro ** 2 - xo ** 2 - yo ** 2) ** 2
        )

        # Get all real roots `x` that satisfy `sgn(y(x)) = sgn(b)`.
        x = np.roots([A, B, C, D, E])
        x = np.array([xi.real for xi in x if np.abs(xi.imag) < tol])
        x = np.array(
            [
                xi
                for xi in x
                if np.abs(
                    (xi - xo) ** 2 + (b * np.sqrt(1 - xi ** 2) - yo) ** 2 - ro ** 2
                )
                < tol
            ]
        )

    # Get rid of any multiplicity
    x = np.array(list(set(x)))

    # Check that the number of roots is correct
    x_l = np.cos(theta)
    y_l = np.sin(theta)
    l1 = x_l ** 2 + (y_l - bo) ** 2 < ro ** 2
    l2 = x_l ** 2 + (-y_l - bo) ** 2 < ro ** 2
    if (l1 and not l2) or (l2 and not l1):
        if len(x) == 1:
            # All good
            pass
        else:
            # There should be one root!
            if len(x) == 0:
                raise RuntimeError(
                    "Unable to find the root. Try decreasing the tolerance."
                )
            elif len(x) == 2:
                # We likely have a rogue root that was included
                # because of the tolerance.
                # Pick the one with the smallest error
                x = np.array(
                    [
                        x[
                            np.argmin(
                                np.abs(
                                    (x - xo) ** 2
                                    + (b * np.sqrt(1 - x ** 2) - yo) ** 2
                                    - ro ** 2
                                )
                            )
                        ]
                    ]
                )

    # P-Q
    if len(x) == 0:

        # Trivial: use the standard starry algorithm

        if np.abs(1 - ro) <= bo <= 1 + ro:

            # The occultor intersects the limb at this point
            lam = np.arcsin((1 - ro ** 2 + bo ** 2) / (2 * bo))
            x = (1 - tol) * np.cos(lam)
            y = (1 - tol) * np.sin(lam)

            if on_dayside(b, theta, x, y):

                # This point is guaranteed to be on the night side
                # We're going to check if it's under the occultor or not
                x = (1 - tol) * np.cos(theta + 3 * np.pi / 2)
                y = (1 - tol) * np.sin(theta + 3 * np.pi / 2)

                if x ** 2 + (y - bo) ** 2 <= ro ** 2:

                    # The occultor is blocking some daylight
                    # and all of the night side
                    code = FLUX_SIMPLE_OCC

                else:

                    # The occultor is only blocking daylight
                    code = FLUX_SIMPLE_OCC_REFL

            else:

                # This point is guaranteed to be on the day side
                # We're going to check if it's under the occultor or not
                x = (1 - tol) * np.cos(theta + np.pi / 2)
                y = (1 - tol) * np.sin(theta + np.pi / 2)

                if x ** 2 + (y - bo) ** 2 <= ro ** 2:

                    # The occultor is blocking some night side
                    # and all of the day side
                    code = FLUX_ZERO

                else:

                    # The occultor is only blocking the night side
                    code = FLUX_SIMPLE_REFL
        else:

            # The occultor does not intersect the limb or the terminator
            if on_dayside(b, theta, 0, bo):

                # The occultor is only blocking daylight
                code = FLUX_SIMPLE_OCC_REFL

            else:

                # The occultor is only blocking the night side
                code = FLUX_SIMPLE_REFL

        return (
            np.array([]),
            np.array([]),
            np.array([]),
            code,
        )

    # P-Q-T
    if len(x) == 1:

        # PHI
        # ---

        # Angle of intersection with occultor
        phi_o = np.arcsin((1 - ro ** 2 - bo ** 2) / (2 * bo * ro))
        # There are always two points; always pick the one
        # that's on the dayside for definiteness
        if not on_dayside(
            b,
            theta,
            (1 - tol) * ro * np.cos(phi_o),
            (1 - tol) * (bo + ro * np.sin(phi_o)),
        ):
            phi_o = np.pi - phi_o

        # Angle of intersection with the terminator
        phi_t = theta + np.arctan2(b * np.sqrt(1 - x[0] ** 2) - yo, x[0] - xo)

        # Now ensure phi *only* spans the dayside.
        phi = sort_phi(b, theta, bo, ro, np.array([phi_o, phi_t]), tol=tol)

        # LAMBDA
        # ------

        # Angle of intersection with occultor
        lam_o = np.arcsin((1 - ro ** 2 + bo ** 2) / (2 * bo))
        # There are always two points; always pick the one
        # that's on the dayside for definiteness
        if not on_dayside(
            b, theta, (1 - tol) * np.cos(lam_o), (1 - tol) * np.sin(lam_o)
        ):
            lam_o = np.pi - lam_o

        # Angle of intersection with the terminator
        lam_t = theta
        # There are always two points; always pick the one
        # that's inside the occultor
        if np.cos(lam_t) ** 2 + (np.sin(lam_t) - bo) ** 2 > ro ** 2:
            lam_t = np.pi + theta

        # Now ensure lam *only* spans the inside of the occultor.
        lam = sort_lam(b, theta, bo, ro, np.array([lam_o, lam_t]), tol=tol)

        # XI
        # --

        # Angle of intersection with occultor
        xi_o = np.arctan2(np.sqrt(1 - x[0] ** 2), x[0])

        # Angle of intersection with the limb
        if (1 - xo) ** 2 + yo ** 2 < ro ** 2:
            xi_l = 0
        else:
            xi_l = np.pi

        # Now ensure xi *only* spans the inside of the occultor.
        xi = sort_xi(b, theta, bo, ro, np.array([xi_l, xi_o]), tol=tol)

        # In all cases, we're computing the dayside occulted flux
        code = FLUX_DAY_OCC

    # P-T
    elif len(x) == 2:

        # Angles are easy
        lam = []
        phi = np.sort(
            (theta + np.arctan2(b * np.sqrt(1 - x ** 2) - yo, x - xo)) % (2 * np.pi)
        )
        xi = np.sort(np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi))

        # Cases
        if bo <= 1 - ro:

            # No intersections with the limb (easy)
            phi = sort_phi(b, theta, bo, ro, phi, tol=tol)
            xi = sort_xi(b, theta, bo, ro, xi, tol=tol)
            code = FLUX_DAY_OCC

        else:

            # The occultor intersects the limb, so we need to
            # integrate along the simplest path.

            # 1. Rotate the points of intersection into a frame where the
            # semi-major axis of the terminator ellipse lies along the x axis
            # We're going to choose xi[0] to be the rightmost point in
            # this frame, so that the integration is counter-clockwise along
            # the terminator to xi[1].
            x = np.cos(theta) * np.cos(xi) - b * np.sin(theta) * np.sin(xi)
            y = np.sin(theta) * np.cos(xi) + b * np.cos(theta) * np.sin(xi)
            xr = x * np.cos(theta) + y * np.sin(theta)
            if xr[1] > xr[0]:
                xi = xi[::-1]

            # 2. Now we need the point corresponding to xi[1] to be the same as the
            # point corresponding to phi[0] in order for the path to be continuous
            x_xi1 = np.cos(theta) * np.cos(xi[1]) - b * np.sin(theta) * np.sin(xi[1])
            y_xi1 = np.sin(theta) * np.cos(xi[1]) + b * np.cos(theta) * np.sin(xi[1])
            x_phi = ro * np.cos(phi)
            y_phi = bo + ro * np.sin(phi)
            if np.argmin((x_xi1 - x_phi) ** 2 + (y_xi1 - y_phi) ** 2) == 1:
                phi = phi[::-1]

            # 3. Compare the *curvature* of the two sides of the
            # integration area. The curvatures are similar (i.e., same sign)
            # when cos(theta) < 0, in which case we must integrate *clockwise* along P.
            if np.cos(theta) < 0:
                # Integrate *clockwise* along P
                if phi[0] < phi[1]:
                    phi[0] += 2 * np.pi
            else:
                # Integrate *counter-clockwise* along P
                if phi[1] < phi[0]:
                    phi[1] += 2 * np.pi

            # 4. Determine the integration code. Let's identify the midpoint
            # along each integration path and average their (x, y)
            # coordinates to determine what kind of region we are
            # bounding.
            xi_mean = np.mean(xi)
            x_xi = np.cos(theta) * np.cos(xi_mean) - b * np.sin(theta) * np.sin(xi_mean)
            y_xi = np.sin(theta) * np.cos(xi_mean) + b * np.cos(theta) * np.sin(xi_mean)
            phi_mean = np.mean(phi)
            x_phi = ro * np.cos(phi_mean)
            y_phi = bo + ro * np.sin(phi_mean)
            x = 0.5 * (x_xi + x_phi)
            y = 0.5 * (y_xi + y_phi)
            if on_dayside(b, theta, x, y):
                if x ** 2 + (y - bo) ** 2 < ro ** 2:
                    # Dayside under occultor
                    code = FLUX_DAY_OCC
                    # We need to reverse the integration path, since
                    # the terminator is *under* the arc along the limb
                    # and we should instead start at the *leftmost* xi
                    # value.
                    phi = phi[::-1]
                    xi = xi[::-1]
                else:
                    # Dayside visible
                    code = FLUX_DAY_VIS
                    if b < 0:
                        phi = phi[::-1]
                        xi = xi[::-1]
            else:
                if x ** 2 + (y - bo) ** 2 < ro ** 2:
                    # Nightside under occultor
                    code = FLUX_NIGHT_OCC
                else:
                    # Nightside visible
                    code = FLUX_NIGHT_VIS

    # There's a pathological case with 3 roots
    elif len(x) == 3:

        # Pre-compute some angles
        x = np.sort(x)
        phi_l = np.arcsin((1 - ro ** 2 - bo ** 2) / (2 * bo * ro))
        lam_o = np.arcsin((1 - ro ** 2 + bo ** 2) / (2 * bo))

        # We need to do this case-by-case
        if b > 0:

            if (-1 - xo) ** 2 + yo ** 2 < ro ** 2:

                x = np.array([x[2], x[1], x[0]])
                phi_t = np.arctan2(b * np.sqrt(1 - x ** 2) - yo, x - xo)
                phi = np.append(theta + phi_t, phi_l,) % (2 * np.pi)
                for n in range(3):
                    while phi[n + 1] < phi[n]:
                        phi[n + 1] += 2 * np.pi
                xi_o = np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi)
                xi = np.append(xi_o, np.pi)
                xi = np.array([xi[1], xi[0], xi[3], xi[2]])
                lam = np.array([lam_o, np.pi + theta,]) % (2 * np.pi)
                if lam[1] < lam[0]:
                    lam[1] += 2 * np.pi

            else:

                x = np.array([x[1], x[0], x[2]])
                phi_t = np.arctan2(b * np.sqrt(1 - x ** 2) - yo, x - xo)
                phi = np.append(theta + phi_t, np.pi - phi_l,) % (2 * np.pi)
                phi[[2, 3]] = phi[[3, 2]]
                for n in range(3):
                    while phi[n + 1] < phi[n]:
                        phi[n + 1] += 2 * np.pi
                xi_o = np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi)
                xi = np.append(xi_o, 0.0)
                xi = np.array([xi[1], xi[0], xi[2], xi[3]])
                lam = np.array([theta, np.pi - lam_o,]) % (2 * np.pi)
                if lam[1] < lam[0]:
                    lam[1] += 2 * np.pi

            code = FLUX_TRIP_DAY_OCC

        else:

            if (-1 - xo) ** 2 + yo ** 2 < ro ** 2:

                x = np.array([x[1], x[2], x[0]])
                phi_t = np.arctan2(b * np.sqrt(1 - x ** 2) - yo, x - xo)
                phi = np.append(theta + phi_t, np.pi - phi_l,) % (2 * np.pi)
                phi[[2, 3]] = phi[[3, 2]]
                for n in range(3):
                    while phi[n + 1] < phi[n]:
                        phi[n + 1] += 2 * np.pi
                xi_o = np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi)
                xi = np.append(xi_o, np.pi)
                xi = np.array([xi[1], xi[0], xi[2], xi[3]])
                lam = np.array([np.pi + theta, np.pi - lam_o,]) % (2 * np.pi)
                if lam[1] < lam[0]:
                    lam[1] += 2 * np.pi

            else:

                phi_t = np.arctan2(b * np.sqrt(1 - x ** 2) - yo, x - xo)
                phi = np.append(theta + phi_t, phi_l,) % (2 * np.pi)
                for n in range(3):
                    while phi[n + 1] < phi[n]:
                        phi[n + 1] += 2 * np.pi
                xi_o = np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi)
                xi = np.append(xi_o, 0.0)
                xi = np.array([xi[1], xi[0], xi[3], xi[2]])
                lam = np.array([lam_o, theta,]) % (2 * np.pi)
                if lam[1] < lam[0]:
                    lam[1] += 2 * np.pi

            code = FLUX_TRIP_NIGHT_OCC

    # And a pathological case with 4 roots
    elif len(x) == 4:

        lam = []
        phi = np.sort(
            (theta + np.arctan2(b * np.sqrt(1 - x ** 2) - yo, x - xo)) % (2 * np.pi)
        )
        phi = np.array([phi[1], phi[0], phi[3], phi[2]])
        xi = np.sort(np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi))

        if b > 0:
            code = FLUX_QUAD_NIGHT_VIS
        else:
            xi = np.array([xi[1], xi[0], xi[3], xi[2]])
            code = FLUX_QUAD_DAY_VIS

    else:

        raise NotImplementedError("Unexpected branch.")

    return phi, lam, xi, code

