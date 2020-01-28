from scipy.special import hyp2f1, gamma
import numpy as np


def compute_W(imax, z):
    """
    Compute the expression

        W(i, z) = 2F1(-1/2, i; i + 1; z)

    recursively and return an array containing the values of this function
    from i = 0 to i = imax.

    TODO: Code up the downward recursion for this. Unstable near z = 0 and z = 1!

    """

    # Special cases
    if z == 0:
        return np.ones(imax + 1)
    elif z == 1.0:
        W = np.zeros(imax + 1)
        W[0] = 1.0
        for i in range(1, imax + 1):
            W[i] = i / (i + 0.5) * W[i - 1]
        return W

    F = np.empty((imax + 2, imax + 2))

    # Initial conditions
    F[0, 1] = 1.0
    F[0, 2] = 1.0
    F[0, 3] = 1.0
    F[1, 1] = np.sqrt(1 - z)
    if np.abs(z) < 0.005:
        F[1, 2] = 1 - z / 4 - z ** 2 / 24 - z ** 3 / 64 - z ** 4 / 128
    else:
        F[1, 2] = 2 * (F[1, 1] * z - F[1, 1] + 1) / (3 * z)

    # Upward in c
    def upc(b, c):
        alpha = (c - 2) * (c - 1) * (z - 1)
        beta = (0.5 - c) * (b - c + 1) * z
        gamma = (c - 1) * (c + (b - 2 * c + 2.5) * z - 2)
        F[b, c] = -(gamma * F[b, c - 1] + alpha * F[b, c - 2]) / beta

    # Upward in b
    def upb(b, c):
        delta = b - c - 1
        eps = -(b - 1) * (z - 1)
        phi = c - 2 * b + (b - 0.5) * z + 2
        F[b, c] = -(phi * F[b - 1, c] + delta * F[b - 2, c]) / eps

    # Recurse
    for b in range(1, imax + 1):
        if b > 1:
            upb(b, b + 1)
        if b < imax:
            upc(b, b + 2)
        if b < imax - 1:
            upc(b, b + 3)
        upb(b + 1, b + 1)

    # We only care about the k + 1 diagonal
    return np.diagonal(F, 1)


# DEBUG
print(compute_W(10, 0.0))
