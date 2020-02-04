from .special import hyp2f1, E, F, J
import numpy as np
import matplotlib.pyplot as plt


def _Windef(nmax, k2, kappa):
    """
    Compute the expression

        s^(2n + 2) (3 / (n + 1) * 2F1(-1/2, n + 1, n + 2, 1 - c^2) + 2c^3) / (2n + 5)

    evaluated at n = [0 .. nmax], where

        s = sin(1/2 kappa)
        c = (1 - s^2 / k^2)^1/2

    by either upward recursion (stable for |1 - c^2| > 1/2) or downward 
    recursion (always stable).

    """
    s2 = np.sin(0.5 * kappa) ** 2
    c2 = 1 - min(1.0, s2 / k2)
    c3 = c2 ** 1.5

    W = np.zeros(nmax + 1)
    if np.abs(1 - c2) < 0.5:

        invs2 = 1 / s2
        z = (1 - c2) * invs2
        s2nmax = s2 ** nmax
        x = c2 * c3 * s2nmax

        # Upper boundary condition
        W[nmax] = (
            s2
            * s2nmax
            * (3 / (nmax + 1) * hyp2f1(-0.5, nmax + 1, nmax + 2, 1 - c2) + 2 * c3)
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

        z = s2 / (1 - c2)
        x = -2 * c3 * (z - s2) * s2

        # Lower boundary condition
        W[0] = (2 / 5) * (z * (1 - c3) + s2 * c3)

        # Recurse up
        for b in range(1, nmax + 1):
            f = 1 / (2 * b + 5)
            A = z * (2 * b) * f
            B = x * f
            W[b] = A * W[b - 1] + B
            x *= s2

    return W


def compute_W(nmax, k2, kappa1, kappa2):
    return _Windef(nmax, k2, kappa2) - _Windef(nmax, k2, kappa1)


def compute_U(vmax, kappa1, kappa2):
    """
    Compute the integral of

        cos(x) sin^v(x)

    from 0.5 * kappa1 to 0.5 * kappa2 recursively and return an array 
    containing the values of this function from v = 0 to v = vmax.

    """
    U = np.empty(vmax + 1)
    s2 = np.sin(0.5 * kappa2)
    s1 = np.sin(0.5 * kappa1)
    U[0] = s2 - s1
    term2 = s2 ** 2
    term1 = s1 ** 2
    for v in range(1, vmax + 1):
        U[v] = (term2 - term1) / (v + 1)
        term2 *= s2
        term1 *= s1
    return U


def compute_I(nmax, kappa1, kappa2):

    # Useful quantities
    s1 = np.sin(0.5 * kappa1)
    s2 = np.sin(0.5 * kappa2)
    s12 = s1 ** 2
    s22 = s2 ** 2
    term1 = s1 * np.cos(0.5 * kappa1)
    term2 = s2 * np.cos(0.5 * kappa2)

    # Lower boundary
    I = np.empty(nmax + 1)
    I[0] = 0.5 * (kappa2 - kappa1)

    # Recurse upward
    for v in range(1, nmax + 1):
        term = 0.5 * (term2 - term1)
        I[v] = (1.0 / v) * (0.5 * (2 * v - 1) * I[v - 1] - term)
        term1 *= s12
        term2 *= s22

    return I


def compute_J(nmax, k, kappa1, kappa2):
    """
    Return the array J[0 .. nmax], computed recursively using
    a tridiagonal solver and a lower boundary condition
    (analytic in terms of elliptic integrals) and an upper
    boundary condition (computed numerically).
    
    """
    # Useful quantities
    k2 = k ** 2
    km2 = k ** -2
    x2 = 0.5 * kappa2
    x1 = 0.5 * kappa1
    s2 = np.sin(x2)
    s1 = np.sin(x1)
    s12 = s1 ** 2
    s22 = s2 ** 2
    z2 = s2 * np.cos(x2) * np.sqrt(max(0.0, 1 - km2 * s22))
    z1 = s1 * np.cos(x1) * np.sqrt(max(0.0, 1 - km2 * s12))

    # Boundary conditions
    dE = E(x2, km2) - E(x1, km2)
    dF = F(x2, km2) - F(x1, km2)
    resid = km2 * (z2 - z1)
    f0 = (1 / 3) * (2 * (2 - km2) * dE + (km2 - 1) * dF + resid)
    fN = J(nmax, k, kappa1, kappa2)

    # Set up the tridiagonal problem
    a = np.empty(nmax - 1)
    b = np.empty(nmax - 1)
    c = np.empty(nmax - 1)
    term2 = k2 * z2 * (1 - km2 * s22) ** 2
    term1 = k2 * z1 * (1 - km2 * s12) ** 2
    for i, v in enumerate(range(2, nmax + 1)):
        amp = 1.0 / (2 * v + 3)
        a[i] = -2 * (v + (v - 1) * k2 + 1) * amp
        b[i] = (2 * v - 3) * k2 * amp
        c[i] = (term2 - term1) * amp
        term2 *= s22
        term1 *= s12

    # Add the boundary conditions
    c[0] -= b[0] * f0
    c[-1] -= fN

    # Construct the tridiagonal matrix
    A = np.diag(a, 0) + np.diag(b[1:], -1) + np.diag(np.ones(nmax - 2), 1)

    # Solve
    soln = np.linalg.solve(A, c)
    return np.concatenate(([f0], soln, [fN]))
