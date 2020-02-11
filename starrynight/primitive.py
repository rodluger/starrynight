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


def T2_indef(b, xi):
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


class compute_H(object):
    def __init__(self, N, xi):

        c = np.cos(xi)
        s = np.sin(xi)
        self.cs = c * s
        self.cc = c ** 2
        self.ss = s ** 2

        self.H = np.zeros((N + 1, N + 1))
        self.T = np.zeros((N + 1, N + 1, len(xi)))

        self.H[0, 0] = pairdiff(xi)
        self.T[0, 0] = 1

        self.H[1, 0] = pairdiff(s)
        self.T[1, 0] = c

        self.H[0, 1] = -pairdiff(c)
        self.T[0, 1] = s

        self.H[1, 1] = -0.5 * pairdiff(self.cc)
        self.T[1, 1] = self.cs

        self.computed = np.zeros((N + 1, N + 1), dtype=bool)
        self.computed[:2, :2] = True

        for u in range(N + 1):
            for v in range(N + 1 - u):
                self.H[u, v], self.T[u, v] = self.compute(u, v)

    def compute(self, u, v):
        if self.computed[u, v]:
            return self.H[u, v], np.array(self.T[u, v])

        if u >= 2:

            H, T = self.compute(u - 2, v)
            term1 = pairdiff(T * self.cs)
            T *= self.cc
            term2 = (u - 1) * H

        else:

            H, T = self.compute(u, v - 2)
            term1 = -pairdiff(T * self.cs)
            T *= self.ss
            term2 = (v - 1) * H

        self.computed[u, v] = True
        return (term1 + term2) / (u + v), T


def compute_T(ydeg, b, theta, xi):

    # Pre-compute H
    H = compute_H(ydeg + 2, xi).H

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
    T[2] = pairdiff([np.sign(b) * T2_indef(np.abs(b), x) for x in xi])

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
