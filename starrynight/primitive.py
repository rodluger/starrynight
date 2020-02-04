from .special import hyp2f1, E, F, J
from .utils import pairdiff
import numpy as np
import matplotlib.pyplot as plt


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
