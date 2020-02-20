from .special import hyp2f1, J, ellip
from .utils import *
from .vieta import Vieta
from .linear import dP2
import matplotlib.pyplot as plt
import numpy as np


__ALL__ = ["compute_P", "compute_Q", "comput_T"]


def compute_U(vmax, s1):
    """
    Given s1 = sin(0.5 * kappa), compute the integral of

        cos(x) sin^v(x)

    from 0.5 * kappa1 to 0.5 * kappa2 recursively and return an array 
    containing the values of this function from v = 0 to v = vmax.

    """
    U = np.empty(vmax + 1)
    U[0] = pairdiff(s1)
    term = s1 ** 2
    for v in range(1, vmax + 1):
        U[v] = pairdiff(term) / (v + 1)
        term *= s1
    return U


def compute_I(nmax, kappa, s1, c1):

    # Lower boundary
    I = np.empty(nmax + 1)
    I[0] = 0.5 * pairdiff(kappa)

    # Recurse upward
    s2 = s1 ** 2
    term = s1 * c1
    for v in range(1, nmax + 1):
        I[v] = (1.0 / (2 * v)) * ((2 * v - 1) * I[v - 1] - pairdiff(term))
        term *= s2

    return I


def _compute_W_indef(nmax, s2, q2, q3):
    """
    Compute the expression

        s^(2n + 2) (3 / (n + 1) * 2F1(-1/2, n + 1, n + 2, 1 - q^2) + 2q^3) / (2n + 5)

    evaluated at n = [0 .. nmax], where

        s = sin(1/2 kappa)
        q = (1 - s^2 / k^2)^1/2

    by either upward recursion (stable for |1 - q^2| > 1/2) or downward 
    recursion (always stable).

    """
    W = np.empty(nmax + 1)

    if np.abs(1 - q2) < 0.5:

        # Setup
        invs2 = 1 / s2
        z = (1 - q2) * invs2
        s2nmax = s2 ** nmax
        x = q2 * q3 * s2nmax

        # Upper boundary condition
        W[nmax] = (
            s2
            * s2nmax
            * (3 / (nmax + 1) * hyp2f1(-0.5, nmax + 1, nmax + 2, 1 - q2) + 2 * q3)
            / (2 * nmax + 5)
        )

        # Recurse down
        for b in range(nmax - 1, -1, -1):
            f = 1 / (b + 1)
            A = z * (1 + 2.5 * f)
            B = x * f
            W[b] = A * W[b + 1] + B
            x *= invs2
    else:

        # Setup
        z = s2 / (1 - q2)
        x = -2 * q3 * (z - s2) * s2

        # Lower boundary condition
        W[0] = (2 / 5) * (z * (1 - q3) + s2 * q3)

        # Recurse up
        for b in range(1, nmax + 1):
            f = 1 / (2 * b + 5)
            A = z * (2 * b) * f
            B = x * f
            W[b] = A * W[b - 1] + B
            x *= s2

    return W


def compute_W(nmax, s2, q2, q3):
    return pairdiff(
        np.array([_compute_W_indef(nmax, s2[i], q2[i], q3[i]) for i in range(len(s2))])
    )


def compute_J(nmax, k2, km2, kappa, s1, s2, c1, q2, dE, dF):
    """
    Return the array J[0 .. nmax], computed recursively using
    a tridiagonal solver and a lower boundary condition
    (analytic in terms of elliptic integrals) and an upper
    boundary condition (computed numerically).
    
    """
    # Boundary conditions
    z = s1 * c1 * np.sqrt(q2)
    resid = km2 * pairdiff(z)
    f0 = (1 / 3) * (2 * (2 - km2) * dE + (km2 - 1) * dF + resid)
    fN = J(nmax, k2, kappa)

    # Set up the tridiagonal problem
    a = np.empty(nmax - 1)
    b = np.empty(nmax - 1)
    c = np.empty(nmax - 1)
    term = k2 * z * q2 ** 2

    for i, v in enumerate(range(2, nmax + 1)):
        amp = 1.0 / (2 * v + 3)
        a[i] = -2 * (v + (v - 1) * k2 + 1) * amp
        b[i] = (2 * v - 3) * k2 * amp
        c[i] = pairdiff(term) * amp
        term *= s2

    # Add the boundary conditions
    c[0] -= b[0] * f0
    c[-1] -= fN

    # Construct the tridiagonal matrix
    A = np.diag(a, 0) + np.diag(b[1:], -1) + np.diag(np.ones(nmax - 2), 1)

    # Solve
    soln = np.linalg.solve(A, c)
    return np.concatenate(([f0], soln, [fN]))


def K(I, delta, u, v):
    """Return the integral K, evaluated as a sum over I."""
    return sum([Vieta(i, u, v, delta) * I[i + u] for i in range(u + v + 1)])


def L(J, k, delta, u, v, t):
    """Return the integral L, evaluated as a sum over J."""
    return k ** 3 * sum(
        [Vieta(i, u, v, delta) * J[i + u + t] for i in range(u + v + 1)]
    )


def compute_H(uvmax, xi, gradient=False):

    c = np.cos(xi)
    s = np.sin(xi)
    cs = c * s
    cc = c ** 2
    ss = s ** 2

    H = np.empty((uvmax + 1, uvmax + 1))
    dH = np.empty((uvmax + 1, uvmax + 1, len(xi)))
    H[0, 0] = pairdiff(xi)
    dH[0, 0] = 1
    H[1, 0] = pairdiff(s)
    dH[1, 0] = c
    H[0, 1] = -pairdiff(c)
    dH[0, 1] = s
    H[1, 1] = -0.5 * pairdiff(cc)
    dH[1, 1] = cs

    for u in range(2):
        for v in range(2, uvmax + 1 - u):
            H[u, v] = (-pairdiff(dH[u, v - 2] * cs) + (v - 1) * H[u, v - 2]) / (u + v)
            dH[u, v] = dH[u, v - 2] * ss

    for u in range(2, uvmax + 1):
        for v in range(uvmax + 1 - u):
            H[u, v] = (pairdiff(dH[u - 2, v] * cs) + (u - 1) * H[u - 2, v]) / (u + v)
            dH[u, v] = dH[u - 2, v] * cc

    if gradient:
        return H, dH
    else:
        return H


def _compute_T2_indef(b, xi):
    """
    Note: requires b >= 0.
    
    """
    s = np.sin(xi)
    c = np.cos(xi)
    t = s / c
    sgn = np.sign(s)
    bc = np.sqrt(1 - b ** 2)
    bbc = b * bc

    # Special cases
    if xi == 0:
        return -(np.arctan((2 * b ** 2 - 1) / (2 * bbc)) + bbc) / 3
    elif xi == 0.5 * np.pi:
        return (0.5 * np.pi - np.arctan(b / bc)) / 3
    elif xi == np.pi:
        return (0.5 * np.pi + bbc) / 3
    elif xi == 1.5 * np.pi:
        return (0.5 * np.pi + np.arctan(b / bc) + 2 * bbc) / 3

    # Figure out the offset
    if xi < 0.5 * np.pi:
        delta = 0
    elif xi < np.pi:
        delta = np.pi
    elif xi < 1.5 * np.pi:
        delta = 2 * bbc
    else:
        delta = np.pi + 2 * bbc

    # We're done
    return (
        np.arctan(b * t)
        - sgn * (np.arctan(((s / (1 + c)) ** 2 + 2 * b ** 2 - 1) / (2 * bbc)) + bbc * c)
        + delta
    ) / 3


def compute_P(ydeg, bo, ro, kappa):
    """Compute the P integral."""
    # Basic variables
    delta = (bo - ro) / (2 * ro)
    k2 = (1 - ro ** 2 - bo ** 2 + 2 * bo * ro) / (4 * bo * ro)
    k = np.sqrt(k2)
    km2 = 1.0 / k2
    fourbr15 = (4 * bo * ro) ** 1.5
    k3fourbr15 = k ** 3 * fourbr15
    tworo = np.empty(ydeg + 4)
    tworo[0] = 1.0
    for i in range(1, ydeg + 4):
        tworo[i] = tworo[i - 1] * 2 * ro

    # Pre-compute the helper integrals
    x = 0.5 * kappa
    s1 = np.sin(x)
    s2 = s1 ** 2
    c1 = np.cos(x)
    q2 = 1 - np.minimum(1.0, s2 / k2)
    q3 = q2 ** 1.5
    U = compute_U(2 * ydeg + 5, s1)
    I = compute_I(ydeg + 3, kappa, s1, c1)
    W = compute_W(ydeg, s2, q2, q3)

    # Compute the elliptic integrals
    if k2 == 1:

        # TODO: This is a special case in which the J integral reduces
        # to H[3, 2v]. However, the cosine term changes signs over the
        # interval, so we need to subdivide the integral into regions
        # where cosine has constant sign.
        #
        #   H = compute_H(3 + 2 * ydeg + 2, 0.5 * kappa)
        #   J = np.array([H[3, 2 * v] for v in range(ydeg + 2)])
        #
        # The expressions for the elliptic integrals also simplify.
        F, E, RF, RD, RJ, RF0, RD0, RJ0 = ellip(bo, ro, kappa, 1 + 1e-12)
        J = compute_J(ydeg + 1, k2, km2, kappa, s1, s2, c1, q2, E, F)

    else:

        F, E, RF, RD, RJ, RF0, RD0, RJ0 = ellip(bo, ro, kappa, k2)
        J = compute_J(ydeg + 1, k2, km2, kappa, s1, s2, c1, q2, E, F)

    # Now populate the P array
    P = np.zeros((ydeg + 1) ** 2)
    n = 0
    for l in range(ydeg + 1):
        for m in range(-l, l + 1):

            mu = l - m
            nu = l + m

            if (mu / 2) % 2 == 0:

                # Same as in starry
                P[n] = 2 * tworo[l + 2] * K(I, delta, (mu + 4) // 4, nu // 2)

            elif mu == 1:

                if l == 1:

                    # Same as in starry, but using expression from Pal (2012)
                    P[2] = dP2(bo, ro, k2, kappa, RF, RD, RJ, RF0, RD0, RJ0)

                elif l % 2 == 0:

                    # Same as in starry
                    P[n] = (
                        tworo[l - 1]
                        * fourbr15
                        * (
                            L(J, k, delta, (l - 2) // 2, 0, 0)
                            - 2 * L(J, k, delta, (l - 2) // 2, 0, 1)
                        )
                    )

                else:

                    # Same as in starry
                    P[n] = (
                        tworo[l - 1]
                        * fourbr15
                        * (
                            L(J, k, delta, (l - 3) // 2, 1, 0)
                            - 2 * L(J, k, delta, (l - 3) // 2, 1, 1)
                        )
                    )

            elif (mu - 1) / 2 % 2 == 0:

                # Same as in starry
                P[n] = (
                    2
                    * tworo[l - 1]
                    * fourbr15
                    * L(J, k, delta, (mu - 1) // 4, (nu - 1) // 2, 0)
                )

            else:

                """
                A note about these cases. In the original starry code, these integrals
                are always zero because the integrand is antisymmetric about the
                midpoint. Now, however, the integration limits are different, so 
                there's no cancellation in general.

                The cases below are just the first and fourth cases in equation (D25) 
                of the starry paper. We can re-write them as the first and fourth cases 
                in (D32) and (D35), respectively, but note that we pick up a factor
                of `sgn(cos(phi))`, since the power of the cosine term in the integrand
                is odd.
                
                The other thing to note is that `u` in the call to `K(u, v)` is now
                a half-integer, so our Vieta trick (D36, D37) doesn't work out of the box.
                """

                if nu % 2 == 0:

                    res = 0
                    u = int((mu + 4.0) // 4)
                    v = int(nu / 2)
                    for i in range(u + v + 1):
                        res += Vieta(i, u, v, delta) * U[2 * (u + i) + 1]
                    P[n] = 2 * tworo[l + 2] * res

                else:

                    res = 0
                    u = (mu - 1) // 4
                    v = (nu - 1) // 2
                    for i in range(u + v + 1):
                        res += Vieta(i, u, v, delta) * W[i + u]
                    P[n] = tworo[l - 1] * k3fourbr15 * res

            n += 1

    return P


def compute_Q(ydeg, lam, gradient=False):

    # Pre-compute H
    if gradient:
        H, dH = compute_H(ydeg + 2, lam, gradient=True)
    else:
        H = compute_H(ydeg + 2, lam)

    # Allocate
    Q = np.zeros((ydeg + 1) ** 2)
    dQdlam = np.zeros(((ydeg + 1) ** 2, len(lam)))

    # Note that the linear term is special
    Q[2] = pairdiff(lam) / 3
    dQdlam[2] = np.ones_like(lam) / 3

    # Easy!
    n = 0
    for l in range(ydeg + 1):
        for m in range(-l, l + 1):
            mu = l - m
            nu = l + m
            if nu % 2 == 0:

                Q[n] = H[(mu + 4) // 2, nu // 2]

                if gradient:
                    dQdlam[n] = dH[(mu + 4) // 2, nu // 2]

            n += 1

    # Enforce alternating signs for (lower, upper) limits
    dQdlam *= np.repeat([-1, 1], len(lam) // 2).reshape(1, -1)

    if gradient:
        return Q, dQdlam
    else:
        return Q


def compute_T(ydeg, b, theta, xi):

    # Pre-compute H
    H = compute_H(ydeg + 2, xi)

    # Vars
    ct = np.cos(theta)
    st = np.sin(theta)
    ttinvb = st / (b * ct)
    invbtt = ct / (b * st)
    b32 = (1 - b ** 2) ** 1.5
    bct = b * ct
    bst = b * st

    # Recurse
    T = np.zeros((ydeg + 1) ** 2)

    # Case 2 (special)
    T[2] = pairdiff([np.sign(b) * _compute_T2_indef(np.abs(b), x) for x in xi])

    # Special limit: sin(theta) = 0
    if np.abs(st) < STARRY_T_TOL:

        sgnct = np.sign(ct)
        n = 0
        for l in range(ydeg + 1):
            for m in range(-l, l + 1):
                mu = l - m
                nu = l + m
                if nu % 2 == 0:
                    T[n] = sgnct ** l * b ** (1 + nu // 2) * H[(mu + 4) // 2, nu // 2]
                else:
                    if mu == 1:
                        if (l % 2) == 0:
                            T[n] = -sgnct * b32 * H[l - 2, 4]
                        elif l > 1:
                            T[n] = -b * b32 * H[l - 3, 5]
                    else:
                        T[n] = sgnct ** (l - 1) * (
                            b32 * b ** ((nu + 1) // 2) * H[(mu - 1) // 2, (nu + 5) // 2]
                        )
                n += 1

        return T

    # Special limit: cos(theta) = 0
    elif np.abs(ct) < STARRY_T_TOL:

        sgnst = np.sign(st)
        n = 0
        for l in range(ydeg + 1):
            for m in range(-l, l + 1):
                mu = l - m
                nu = l + m
                if nu % 2 == 0:
                    T[n] = b ** ((mu + 2) // 2) * H[nu // 2, (mu + 4) // 2]
                    if sgnst == 1:
                        T[n] *= (-1) ** (mu // 2)
                    else:
                        T[n] *= (-1) ** (nu // 2)
                else:
                    if mu == 1:
                        if (l % 2) == 0:
                            T[n] = (
                                (-sgnst) ** (l - 1) * b ** (l - 1) * b32 * H[1, l + 1]
                            )
                        elif l > 1:
                            T[n] = b ** (l - 2) * b32 * H[2, l]
                            if sgnst == 1:
                                T[n] *= (-1) ** l
                            else:
                                T[n] *= -1
                    else:
                        T[n] = (
                            b32 * b ** ((mu - 3) // 2) * H[(nu - 1) // 2, (mu + 5) // 2]
                        )
                        if sgnst == 1:
                            T[n] *= (-1) ** ((mu - 1) // 2)
                        else:
                            T[n] *= (-1) ** ((nu - 1) // 2)
                n += 1

        return T

    # Cases 1 and 5
    jmax = 0
    Z0 = 1
    for nu in range(0, 2 * ydeg + 1, 2):
        kmax = 0
        Z1 = Z0
        for mu in range(0, 2 * ydeg - nu + 1, 2):
            l = (mu + nu) // 2
            n1 = l ** 2 + nu
            n5 = (l + 2) ** 2 + nu + 1
            Z2 = Z1
            for j in range(jmax + 1):
                Z_1 = -bst * Z2
                Z_5 = b32 * Z2
                for k in range(kmax + 1):
                    p = j + k
                    q = l + 1 - (j + k)
                    fac = -invbtt / (k + 1)
                    T[n1] += Z_1 * (bct * H[p + 1, q] - st * H[p, q + 1])
                    Z_1 *= (kmax + 1 - k) * fac
                    if n5 < (ydeg + 1) ** 2:
                        T[n5] += Z_5 * (bct * H[p + 1, q + 2] - st * H[p, q + 3])
                        Z_5 *= (kmax - k) * fac
                T[n1] += Z_1 * (bct * H[p + 2, q - 1] - st * H[p + 1, q])
                Z2 *= (jmax - j) / (j + 1) * ttinvb
            kmax += 1
            Z1 *= -bst
        jmax += 1
        Z0 *= bct

    # Cases 3 and 4
    Z0 = b32
    kmax = 0
    for l in range(2, ydeg + 1, 2):
        n3 = l ** 2 + 2 * l - 1
        n4 = (l + 1) ** 2 + 2 * l + 1
        Z = Z0
        for k in range(kmax + 1):
            p = k
            q = l + 1 - k
            T[n3] -= Z * (bst * H[p + 1, q] + ct * H[p, q + 1])
            if l < ydeg:
                T[n4] -= Z * (
                    bst * st * H[p + 2, q]
                    + bct * ct * H[p, q + 2]
                    + (1 + b ** 2) * st * ct * H[p + 1, q + 1]
                )
            Z *= -(kmax - k) / (k + 1) * invbtt
        kmax += 2
        Z0 *= bst ** 2

    return T
