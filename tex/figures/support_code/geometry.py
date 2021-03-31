import numpy as np
from .utils import *


def on_dayside(b, theta, costheta, sintheta, x, y):
    """Return True if a point is on the dayside."""
    xr = x * costheta + y * sintheta
    yr = -x * sintheta + y * costheta
    term = 1 - xr ** 2
    yt = b * np.sqrt(term)
    return yr >= yt


def sort_phi(b, theta, costheta, sintheta, bo, ro, phi):
    """
    Sort a pair of `phi` angles.
    
    The basic rule here: the direction phi1 -> phi2 must 
    always span the dayside.
    """
    # First ensure the range is correct
    phi1, phi2 = phi
    phi = np.array([phi1, phi2]) % (2 * np.pi)
    if phi[1] < phi[0]:
        phi = np.array([phi[0], phi[1] + 2 * np.pi])

    # Now take the midpoint and check that it's on-planet and on the
    # dayside. If not, we swap the integration limits.
    phim = np.mean(phi)
    x = ro * np.cos(phim)
    y = bo + ro * np.sin(phim)
    if (x ** 2 + y ** 2 > 1) or not on_dayside(b, theta, costheta, sintheta, x, y):
        phi = np.array([phi2, phi1]) % (2 * np.pi)
    if phi[1] < phi[0]:
        phi = np.array([phi[0], phi[1] + 2 * np.pi])
    return phi


def sort_xi(b, theta, costheta, sintheta, bo, ro, xi):
    """Sort a pair of `xi` angles.
    
    The basic rule here: the direction xi2 --> xi1 must
    always span the inside of the occultor. (Note that
    the limits of the `T` integral are xi2 --> xi1, since
    we integrate *clockwise* along the arc.) Since xi
    is limited to the range [0, pi], enforcing this is
    actually trivial: we just need to make sure they are
    arranged in decreasing order.
    """
    xi1, xi2 = xi
    xi = np.array([xi1, xi2]) % (2 * np.pi)
    if xi[0] < xi[1]:
        xi = np.array([xi[1], xi[0]])
    return xi


def sort_lam(b, theta, costheta, sintheta, bo, ro, lam):
    """
    Sort a pair of `lam` angles.
    
    The basic rule here: the direction lam1 --> lam2
    must always span the inside of the occultor and the
    dayside.
    """
    # First ensure the range is correct
    lam1, lam2 = lam
    lam = np.array([lam1, lam2]) % (2 * np.pi)
    if lam[1] < lam[0]:
        lam = np.array([lam[0], lam[1] + 2 * np.pi])

    # Now take the midpoint and ensure it is inside
    # the occultor *and* on the dayside. If not, swap
    # the integration limits.
    lamm = np.mean(lam)
    x = np.cos(lamm)
    y = np.sin(lamm)
    if x ** 2 + (y - bo) ** 2 > ro ** 2 or not on_dayside(
        b, theta, costheta, sintheta, x, y,
    ):
        lam = np.array([lam2, lam1]) % (2 * np.pi)

    if lam[1] < lam[0]:
        lam = np.array([lam[0], lam[1] + 2 * np.pi])

    return lam


def get_roots(b, theta, costheta, sintheta, bo, ro, gradient=False):
    # We'll solve for occultor-terminator intersections
    # in the frame where the semi-major axis of the
    # terminator ellipse is aligned with the x axis
    xo = bo * sintheta
    yo = bo * costheta

    # Special case: b = 0
    if np.abs(b) < STARRY_B_ZERO_TOL:

        # Roots
        x = np.array([])
        term = np.sqrt(ro ** 2 - yo ** 2)
        if np.abs(xo + term) < 1:
            x = np.append(x, xo + term)
        if np.abs(xo - term) < 1:
            x = np.append(x, xo - term)
        x = np.array(list(set(x)))

        # Derivatives
        if yo < 0:
            s = 1
        else:
            s = -1
        dxdb = s * np.sqrt((1 - x ** 2) * (ro ** 2 - (x - xo) ** 2)) / (x - xo)
        dxdtheta = bo * (
            costheta - s * np.sqrt(ro ** 2 - (x - xo) ** 2) / (x - xo) * sintheta
        )
        dxdbo = sintheta + s * np.sqrt(ro ** 2 - (x - xo) ** 2) / (x - xo) * costheta
        dxdro = ro / (x - xo)

    # Need to solve a quartic
    else:

        # Get the roots (eigenvalue problem)
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
        roots = np.roots([A, B, C, D, E]) + 0j

        # Polish the roots using Newton's method on the *original*
        # function, which is more stable than the quartic expression.
        x = []
        dxdb = []
        dxdtheta = []
        dxdbo = []
        dxdro = []
        for n in range(len(roots)):

            # We're looking for the intersection of the function
            #
            #     y1 = b * sqrt(1 - x^2)
            #
            # and the function
            #
            #     y2 = yo +/- sqrt(ro^2 - (x - xo^2))
            #
            # Let's figure out which of the two cases (+/-) this
            # root is a solution to. We're then going to polish
            # the root by minimizing the function
            #
            #     f = y1 - y2
            #
            A = np.sqrt(1 - roots[n] ** 2)
            B = np.sqrt(ro ** 2 - (roots[n] - xo) ** 2)
            absfp = np.abs(b * A - yo + B)
            absfm = np.abs(b * A - yo - B)
            if np.argmin([absfp, absfm]) == 0:
                s = 1
                absf = absfp
            else:
                s = -1
                absf = absfm

            # Some roots may instead correspond to
            #
            #     y = -b * sqrt(1 - x^2)
            #
            # which is the wrong half of the terminator ellipse.
            # Let's only move forward if |f| is decently small.
            if absf < STARRY_ROOT_TOL_LOW:

                # Apply Newton's method to polish the root
                minf = np.inf
                for k in range(STARRY_ROOT_MAX_ITER):
                    A = np.sqrt(1 - roots[n] ** 2)
                    B = np.sqrt(ro ** 2 - (roots[n] - xo) ** 2)
                    f = b * A + s * B - yo
                    absf = np.abs(f)
                    if absf < minf:
                        minf = absf
                        minx = roots[n]
                        if minf <= STARRY_ROOT_TOL_HIGH:
                            break
                    df = -(b * roots[n] / A + s * (roots[n] - xo) / B)
                    roots[n] -= f / df

                # Only keep the root if the solver actually converged
                if minf < STARRY_ROOT_TOL_MED:

                    # Only keep the root if it's real
                    if (
                        np.abs(minx.imag) < STARRY_ROOT_TOL_HIGH
                        and np.abs(minx.real) <= 1
                    ):

                        # Discard the (tiny) imaginary part
                        minx = minx.real

                        # Check that we haven't included this root already
                        good = True
                        for xk in x:
                            if np.abs(xk - minx) < STARRY_ROOT_TOL_DUP:
                                good = False
                                break
                        if good:

                            # Store the root
                            x += [minx]

                            # Now compute its derivatives
                            q = (ro ** 2 - (minx - xo) ** 2) ** 0.5
                            p = (1 - minx ** 2) ** 0.5
                            v = (minx - xo) / q
                            w = b / p
                            t = 1 / (-w * minx - s * v)
                            dxdb += [-t * p]
                            dxdtheta += [-t * bo * (sintheta + s * v * costheta)]
                            dxdbo += [t * (costheta - s * v * sintheta)]
                            dxdro += [-t * s * ro / q]

    # Convert to arrays
    x = np.array(x)
    dxdb = np.array(dxdb)
    dxdtheta = np.array(dxdtheta)
    dxdbo = np.array(dxdbo)
    dxdro = np.array(dxdro)

    # Check if the extrema of the terminator ellipse are occulted
    e1 = costheta ** 2 + (sintheta - bo) ** 2 < ro ** 2 + STARRY_ROOT_TOL_HIGH
    e2 = costheta ** 2 + (sintheta + bo) ** 2 < ro ** 2 + STARRY_ROOT_TOL_HIGH

    # One is occulted, the other is not.
    # Usually we should have a single root, but
    # pathological cases with 3 roots (and maybe 4?)
    # are also possible.
    if (e1 and not e1) or (e2 and not e1):
        if (len(x) == 0) or (len(x) == 2):
            # TODO: Fix this case if it ever shows up
            print("ERROR: Solver did not find the correct number of roots.")
            breakpoint()

    # There is one root but none of the extrema are occulted.
    # This likely corresponds to a grazing occultation of the
    # dayside or nightside.
    if len(x) == 1:
        if not e1 and not e2:
            # TODO: Check this more rigorously? For now
            # we just delete the root.
            x = np.array([])
            dxdb = np.array([])
            dxdtheta = np.array([])
            dxdbo = np.array([])
            dxdro = np.array([])

    # Return the value of the roots
    if gradient:
        return x, (dxdb, dxdtheta, dxdbo, dxdro)
    else:
        return x


def get_angles(b, theta, costheta, sintheta, bo, ro):

    # Trivial cases
    if bo <= ro - 1 + STARRY_COMPLETE_OCC_TOL:

        # Complete occultation
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            FLUX_ZERO,
        )

    elif bo >= 1 + ro - STARRY_NO_OCC_TOL:

        # No occultation
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            FLUX_SIMPLE_REFL,
        )

    # Hack. This grazing configuration leads to instabilities
    # in the root solver. Let's avoid it.
    if 1 - ro < bo < 1 - ro + STARRY_GRAZING_TOL:
        bo = 1 - ro + STARRY_GRAZING_TOL

    # Get the points of intersection between the occultor & terminator
    # These are the roots to a quartic equation.
    xo = bo * sintheta
    yo = bo * costheta
    x = get_roots(b, theta, costheta, sintheta, bo, ro)

    # P-Q
    if len(x) == 0:

        # Trivial: use the standard starry algorithm
        phi = np.array([])
        lam = np.array([])
        xi = np.array([])

        if np.abs(1 - ro) <= bo <= 1 + ro:

            # The occultor intersects the limb at this point
            q = (1 - ro ** 2 + bo ** 2) / (2 * bo)
            x = (1 - STARRY_ANGLE_TOL) * np.sqrt(1 - q ** 2)
            y = (1 - STARRY_ANGLE_TOL) * q

            if on_dayside(b, theta, costheta, sintheta, x, y):

                # This point is guaranteed to be on the night side
                # We're going to check if it's under the occultor or not
                x = (1 - STARRY_ANGLE_TOL) * np.cos(theta + 3 * np.pi / 2)
                y = (1 - STARRY_ANGLE_TOL) * np.sin(theta + 3 * np.pi / 2)

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
                x = (1 - STARRY_ANGLE_TOL) * np.cos(theta + np.pi / 2)
                y = (1 - STARRY_ANGLE_TOL) * np.sin(theta + np.pi / 2)

                if x ** 2 + (y - bo) ** 2 <= ro ** 2:

                    # The occultor is blocking some night side
                    # and all of the day side
                    code = FLUX_ZERO

                else:

                    # The occultor is only blocking the night side
                    code = FLUX_SIMPLE_REFL
        else:

            # The occultor does not intersect the limb or the terminator
            if on_dayside(b, theta, costheta, sintheta, 0, bo):

                # The occultor is only blocking daylight
                code = FLUX_SIMPLE_OCC_REFL

            else:

                # The occultor is only blocking the night side
                code = FLUX_SIMPLE_REFL

    # P-Q-T
    elif len(x) == 1:

        # PHI
        # ---

        # Angle of intersection with limb
        phi_l = np.arcsin((1 - ro ** 2 - bo ** 2) / (2 * bo * ro))
        # There are always two points; always pick the one
        # that's on the dayside for definiteness
        if not on_dayside(
            b, theta, costheta, sintheta, ro * np.cos(phi_l), (bo + ro * np.sin(phi_l))
        ):
            phi_l = np.pi - phi_l

        # Angle of intersection with the terminator
        phi_t = theta + np.arctan2(b * np.sqrt(1 - x[0] ** 2) - yo, x[0] - xo)

        # Now ensure phi *only* spans the dayside.
        phi = sort_phi(b, theta, costheta, sintheta, bo, ro, np.array([phi_l, phi_t]),)

        # LAMBDA
        # ------

        # Angle of intersection with occultor
        lam_o = np.arcsin((1 - ro ** 2 + bo ** 2) / (2 * bo))
        # There are always two points; always pick the one
        # that's on the dayside for definiteness
        if not on_dayside(b, theta, costheta, sintheta, np.cos(lam_o), np.sin(lam_o)):
            lam_o = np.pi - lam_o

        # Angle of intersection with the terminator
        lam_t = theta
        # There are always two points; always pick the one
        # that's inside the occultor
        if np.cos(lam_t) ** 2 + (np.sin(lam_t) - bo) ** 2 > ro ** 2:
            lam_t = np.pi + theta

        # Now ensure lam *only* spans the inside of the occultor.
        lam = sort_lam(b, theta, costheta, sintheta, bo, ro, np.array([lam_o, lam_t]),)

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
        xi = sort_xi(b, theta, costheta, sintheta, bo, ro, np.array([xi_l, xi_o]),)

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
            phi = sort_phi(b, theta, costheta, sintheta, bo, ro, phi,)
            xi = sort_xi(b, theta, costheta, sintheta, bo, ro, xi,)
            code = FLUX_DAY_OCC

        else:

            # The occultor intersects the limb, so we need to
            # integrate along the simplest path.

            # 1. Rotate the points of intersection into a frame where the
            # semi-major axis of the terminator ellipse lies along the x axis
            # We're going to choose xi[0] to be the rightmost point in
            # this frame, so that the integration is counter-clockwise along
            # the terminator to xi[1].
            x = costheta * np.cos(xi) - b * sintheta * np.sin(xi)
            y = sintheta * np.cos(xi) + b * costheta * np.sin(xi)
            xr = x * costheta + y * sintheta
            if xr[1] > xr[0]:
                xi = np.array([xi[1], xi[0]])

            # 2. Now we need the point corresponding to xi[1] to be the same as the
            # point corresponding to phi[0] in order for the path to be continuous
            x_xi1 = costheta * np.cos(xi[1]) - b * sintheta * np.sin(xi[1])
            y_xi1 = sintheta * np.cos(xi[1]) + b * costheta * np.sin(xi[1])
            x_phi = ro * np.cos(phi)
            y_phi = bo + ro * np.sin(phi)
            if np.argmin((x_xi1 - x_phi) ** 2 + (y_xi1 - y_phi) ** 2) == 1:
                phi = np.array([phi[1], phi[0]])

            # 3. Compare the *curvature* of the two sides of the
            # integration area. The curvatures are similar (i.e., same sign)
            # when cos(theta) < 0, in which case we must integrate *clockwise* along P.
            if costheta < 0:
                # Integrate *clockwise* along P
                if phi[0] < phi[1]:
                    phi = np.array([phi[0] + 2 * np.pi, phi[1]])
            else:
                # Integrate *counter-clockwise* along P
                if phi[1] < phi[0]:
                    phi = np.array([phi[0], phi[1] + 2 * np.pi])

            # 4. Determine the integration code. Let's identify the midpoint
            # along each integration path and average their (x, y)
            # coordinates to determine what kind of region we are
            # bounding.
            xim = np.mean(xi)
            x_xi = costheta * np.cos(xim) - b * sintheta * np.sin(xim)
            y_xi = sintheta * np.cos(xim) + b * costheta * np.sin(xim)
            phi_mean = np.mean(phi)
            x_phi = ro * np.cos(phi_mean)
            y_phi = bo + ro * np.sin(phi_mean)
            x = 0.5 * (x_xi + x_phi)
            y = 0.5 * (y_xi + y_phi)
            if on_dayside(b, theta, costheta, sintheta, x, y):
                if x ** 2 + (y - bo) ** 2 < ro ** 2:
                    # Dayside under occultor
                    code = FLUX_DAY_OCC
                    # We need to reverse the integration path, since
                    # the terminator is *under* the arc along the limb
                    # and we should instead start at the *leftmost* xi
                    # value.
                    phi = np.array([phi[1], phi[0]])
                    xi = np.array([xi[1], xi[0]])
                else:
                    # Dayside visible
                    code = FLUX_DAY_VIS
                    if b < 0:
                        phi = np.array([phi[1], phi[0]])
                        xi = np.array([xi[1], xi[0]])
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

                # NOTE: Looks silly but simplest way to make jax-compliant
                while phi[1] < phi[0]:
                    phi = np.array([phi[0], phi[1] + 2 * np.pi, phi[2], phi[3]])
                while phi[2] < phi[1]:
                    phi = np.array([phi[0], phi[1], phi[2] + 2 * np.pi, phi[3]])
                while phi[3] < phi[2]:
                    phi = np.array([phi[0], phi[1], phi[2], phi[3] + 2 * np.pi])

                xi_o = np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi)
                xi = np.append(xi_o, np.pi)
                xi = np.array([xi[1], xi[0], xi[3], xi[2]])
                lam = np.array([lam_o, np.pi + theta,]) % (2 * np.pi)
                if lam[1] < lam[0]:
                    lam = np.array([lam[0], lam[1] + 2 * np.pi])

            else:

                x = np.array([x[1], x[0], x[2]])
                phi_t = np.arctan2(b * np.sqrt(1 - x ** 2) - yo, x - xo)
                phi = np.append(theta + phi_t, np.pi - phi_l,) % (2 * np.pi)
                phi = np.array([phi[0], phi[1], phi[3], phi[2]])

                # NOTE: Looks silly but simplest way to make jax-compliant
                while phi[1] < phi[0]:
                    phi = np.array([phi[0], phi[1] + 2 * np.pi, phi[2], phi[3]])
                while phi[2] < phi[1]:
                    phi = np.array([phi[0], phi[1], phi[2] + 2 * np.pi, phi[3]])
                while phi[3] < phi[2]:
                    phi = np.array([phi[0], phi[1], phi[2], phi[3] + 2 * np.pi])

                xi_o = np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi)
                xi = np.append(xi_o, 0.0)
                xi = np.array([xi[1], xi[0], xi[2], xi[3]])
                lam = np.array([theta, np.pi - lam_o,]) % (2 * np.pi)
                if lam[1] < lam[0]:
                    lam = np.array([lam[0], lam[1] + 2 * np.pi])

            code = FLUX_TRIP_DAY_OCC

        else:

            if (-1 - xo) ** 2 + yo ** 2 < ro ** 2:

                x = np.array([x[1], x[2], x[0]])
                phi_t = np.arctan2(b * np.sqrt(1 - x ** 2) - yo, x - xo)
                phi = np.append(theta + phi_t, np.pi - phi_l,) % (2 * np.pi)
                phi = np.array([phi[0], phi[1], phi[3], phi[2]])

                # NOTE: Looks silly but simplest way to make jax-compliant
                while phi[1] < phi[0]:
                    phi = np.array([phi[0], phi[1] + 2 * np.pi, phi[2], phi[3]])
                while phi[2] < phi[1]:
                    phi = np.array([phi[0], phi[1], phi[2] + 2 * np.pi, phi[3]])
                while phi[3] < phi[2]:
                    phi = np.array([phi[0], phi[1], phi[2], phi[3] + 2 * np.pi])

                xi_o = np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi)
                xi = np.append(xi_o, np.pi)
                xi = np.array([xi[1], xi[0], xi[2], xi[3]])
                lam = np.array([np.pi + theta, np.pi - lam_o,]) % (2 * np.pi)
                if lam[1] < lam[0]:
                    lam = np.array([lam[0], lam[1] + 2 * np.pi])

            else:

                phi_t = np.arctan2(b * np.sqrt(1 - x ** 2) - yo, x - xo)
                phi = np.append(theta + phi_t, phi_l,) % (2 * np.pi)

                # NOTE: Looks silly but simplest way to make jax-compliant
                while phi[1] < phi[0]:
                    phi = np.array([phi[0], phi[1] + 2 * np.pi, phi[2], phi[3]])
                while phi[2] < phi[1]:
                    phi = np.array([phi[0], phi[1], phi[2] + 2 * np.pi, phi[3]])
                while phi[3] < phi[2]:
                    phi = np.array([phi[0], phi[1], phi[2], phi[3] + 2 * np.pi])

                xi_o = np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi)
                xi = np.append(xi_o, 0.0)
                xi = np.array([xi[1], xi[0], xi[3], xi[2]])
                lam = np.array([lam_o, theta,]) % (2 * np.pi)
                if lam[1] < lam[0]:
                    lam = np.array([lam[0], lam[1] + 2 * np.pi])

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

    # Check
    if np.isnan(np.sum(phi)):
        # TODO: Fix this case if it ever shows up
        print("ERROR: At least one angle is NaN.")
        breakpoint()

    # Note that in starry, we use kappa = phi + pi / 2
    return (
        np.array(phi) + np.pi / 2,
        np.array(lam),
        np.array(xi),
        code,
    )
