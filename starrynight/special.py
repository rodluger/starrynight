from scipy.special import hyp2f1, poch, factorial
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
