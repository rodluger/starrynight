from scipy.special import hyp2f1
import numpy as np
import matplotlib.pyplot as plt


def compute_W(imax, z):
    """
    Compute the expression

        W(i, z) = 2F1(-1/2, i; i + 1; z)

    recursively and return an array containing the values of this function
    from i = 0 to i = imax.

    """

    term = (1 - z) ** 1.5
    W = np.zeros(imax + 1)

    if np.abs(z) < 0.5:

        # Recurse downward. We need to evaluate 2F1 once,
        # but it's unconditionally stable at all z.
        # TODO: Code up `hyp2f1`
        W[imax] = hyp2f1(-0.5, imax, imax + 1, z)
        for b in range(imax - 1, -1, -1):
            W[b] = z * (b + 1.5) / (b + 1) * W[b + 1] + term
        return W

    else:

        # Recurse upward. This is faster but
        # it's unstable near z = 0.
        W[0] = 1.0
        for b in range(1, imax + 1):
            W[b] = b / (z * (b + 0.5)) * (W[b - 1] - term)
        return W


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
