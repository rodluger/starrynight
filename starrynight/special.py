from scipy.special import hyp2f1
import numpy as np
import matplotlib.pyplot as plt


def compute_W0(imax, z):
    """
    Compute the expression

        W0(i, z) = 2F1(-1/2, i; i + 1; 1 - z)

    recursively and return an array containing the values of this function
    from i = 0 to i = imax.

    """

    term = z ** 1.5
    W = np.zeros(imax + 1)

    if np.abs(1 - z) < 0.5:

        # Recurse downward. We need to evaluate 2F1 once,
        # but it's unconditionally stable at all z.
        # TODO: Code up `hyp2f1`
        W[imax] = hyp2f1(-0.5, imax, imax + 1, 1 - z)
        for b in range(imax - 1, -1, -1):
            W[b] = (1 - z) * (b + 1.5) / (b + 1) * W[b + 1] + term
        return W

    else:

        # Recurse upward. This is faster but
        # it's unstable near z = 0.
        W[0] = 1.0
        for b in range(1, imax + 1):
            W[b] = b / ((1 - z) * (b + 0.5)) * (W[b - 1] - term)
        return W


def compute_Wold(nmax, k2, kappa1, kappa2):
    # TODO: Recurse directly in W, not W0
    W = np.zeros(nmax)
    s12 = np.sin(0.5 * kappa1) ** 2
    c12 = 1 - min(1.0, s12 / k2)
    c13 = c12 ** 1.5
    s22 = np.sin(0.5 * kappa2) ** 2
    c22 = 1 - min(1.0, s22 / k2)
    c23 = c22 ** 1.5

    W1 = compute_W0(nmax + 1, c12)
    W2 = compute_W0(nmax + 1, c22)

    for n in range(nmax):
        f1 = 3.0 / ((2 * n + 5) * (n + 1))
        f2 = 2.0 / (2 * n + 5)
        term1 = f1 * s12 ** (n + 1) * W1[n + 1] + f2 * s12 ** (n + 1) * c13
        term2 = f1 * s22 ** (n + 1) * W2[n + 1] + f2 * s22 ** (n + 1) * c23
        W[n] = term2 - term1
    return W


def compute_W(nmax, k2, kappa1, kappa2):

    s12 = np.sin(0.5 * kappa1) ** 2
    c12 = 1 - min(1.0, s12 / k2)
    c13 = c12 ** 1.5
    s22 = np.sin(0.5 * kappa2) ** 2
    c22 = 1 - min(1.0, s22 / k2)
    c23 = c22 ** 1.5

    W1 = np.zeros(nmax + 1)
    if np.abs(1 - c12) < 0.5:

        """
        W1[nmax] = (
            s12 ** (nmax + 1)
            * hyp2f1(-0.5, nmax + 1, nmax + 2, 1 - c12)
            * 3.0
            / ((2 * nmax + 5) * (nmax + 1))
        )
        for b in range(nmax - 1, -1, -1):
            A = (1 + 5 / (2 * (b + 1))) * (1 - c12) / s12
            B = 3.0 / ((2 * b + 5) * (b + 1)) * s12 ** (b + 1) * c12 ** 1.5
            W1[b] = A * W1[b + 1] + B

        n = np.arange(nmax + 1)
        W1 += 2.0 / (2 * n + 5) * s12 ** (n + 1) * c13
        """

        W1[nmax] = (
            s12 ** (nmax + 1)
            * hyp2f1(-0.5, nmax + 1, nmax + 2, 1 - c12)
            * 3.0
            / ((2 * nmax + 5) * (nmax + 1))
            + 2.0 / (2 * nmax + 5) * s12 ** (nmax + 1) * c13
        )
        for b in range(nmax - 1, -1, -1):
            A = (1 + 5 / (2 * b + 2)) * (1 - c12) / s12

            D = 3.0 / ((2 * b + 5) * (b + 1))
            C = 2.0 / (2 * b + 5)
            Cnp1 = 2.0 / (2 * b + 7) * s12
            B = c13 * s12 ** (b + 1) * (D + C - A * Cnp1)

            W1[b] = A * W1[b + 1] + B

    else:

        z = s12 / (1 - c12)
        W1[0] = (2 / 5) * (z * (1 - c13) + s12 * c13)
        for b in range(1, nmax + 1):
            A = z * (2 * b) / (2 * b + 5)
            B = -2 * c13 * (z - s12) * s12 ** b / (2 * b + 5)
            W1[b] = A * W1[b - 1] + B

    z = c22
    term = z ** 1.5
    W2 = np.zeros(nmax + 1)

    if np.abs(1 - z) < 0.5:

        W2[nmax] = s22 ** (nmax + 1) * hyp2f1(-0.5, nmax + 1, nmax + 2, 1 - z)
        for b in range(nmax - 1, -1, -1):
            W2[b] = (1 - z) * (b + 2.5) / (b + 2) / s22 * W2[b + 1] + s22 ** (
                b + 1
            ) * term
        n = np.arange(nmax + 1)
        W2 *= 3.0 / ((2 * n + 5) * (n + 1))
        W2 += 2.0 / (2 * n + 5) * s22 ** (n + 1) * c23

    else:

        W2[0] = 2 * s22 / (3 * (1 - z)) * (1 - term)
        for b in range(1, nmax + 1):
            fac = (b + 1) / ((1 - z) * (b + 1.5))
            W2[b] = s22 * fac * W2[b - 1] - fac * s22 ** (b + 1) * term

        n = np.arange(nmax + 1)
        W2 *= 3.0 / ((2 * n + 5) * (n + 1))
        W2 += 2.0 / (2 * n + 5) * s22 ** (n + 1) * c23

    return W2 - W1


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
