from scipy.special import hyp2f1, poch, factorial
import numpy as np

# Upward in c
def upc(F, z, b, c):
    alpha = (c - 2) * (c - 1) * (z - 1)
    beta = (0.5 - c) * (b - c + 1) * z
    gamma = (c - 1) * (c + (b - 2 * c + 2.5) * z - 2)
    F[b, c] = -(gamma * F[b, c - 1] + alpha * F[b, c - 2]) / beta


# Downward in c
def downc(F, z, b, c):
    alpha = c * (c + 1) * (z - 1)
    beta = -(1.5 + c) * (b - c + -1) * z
    gamma = (c + 1) * (c + 2 + (b - 2 * c - 1.5) * z - 2)
    F[b, c] = -(beta * F[b, c + 2] + gamma * F[b, c + 1]) / alpha


# Upward in b
def upb(F, z, b, c):
    delta = b - c - 1
    eps = -(b - 1) * (z - 1)
    phi = c - 2 * b + (b - 0.5) * z + 2
    F[b, c] = -(phi * F[b - 1, c] + delta * F[b - 2, c]) / eps


# Downward in b
def downb(F, z, b, c):
    delta = b - c + 1
    eps = -(b + 1) * (z - 1)
    phi = c - 2 * b + (b + 1.5) * z - 2
    F[b, c] = -(eps * F[b + 2, c] + phi * F[b + 1, c]) / delta


def compute_W(imax, z, method="up"):
    """
    Compute the expression

        W(i, z) = 2F1(-1/2, i; i + 1; z)

    recursively and return an array containing the values of this function
    from i = 0 to i = imax.

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

    if method == "up":

        # Initial conditions
        F = np.empty((imax + 2, imax + 2)) * np.nan
        F[0, 1] = 1.0
        F[0, 2] = 1.0
        F[0, 3] = 1.0
        F[1, 1] = np.sqrt(1 - z)
        F[1, 2] = 2 * (F[1, 1] * z - F[1, 1] + 1) / (3 * z)

        # Recurse
        for b in range(1, imax + 1):
            if b > 1:
                upb(F, z, b, b + 1)
            if b < imax:
                upc(F, z, b, b + 2)
            if b < imax - 1:
                upc(F, z, b, b + 3)
            upb(F, z, b + 1, b + 1)

        # We only care about the k + 1 diagonal
        return np.diagonal(F, 1)

    elif method == "down":

        # Initial conditions
        # TODO: Use a series solution
        F = np.empty((imax + 2, imax + 2)) * np.nan
        F[imax, imax + 1] = hyp2f1(-0.5, imax, imax + 1, z)
        F[imax, imax] = hyp2f1(-0.5, imax, imax, z)
        F[imax - 1, imax + 1] = hyp2f1(-0.5, imax - 1, imax + 1, z)
        F[imax - 1, imax] = hyp2f1(-0.5, imax - 1, imax, z)
        F[imax - 2, imax + 1] = hyp2f1(-0.5, imax - 2, imax + 1, z)

        # Recurse
        for b in range(imax - 1, 0, -1):
            downb(F, z, b - 1, b + 1)
            if b > 1:
                downb(F, z, b - 2, b + 1)
            downc(F, z, b, b)
            downc(F, z, b - 1, b)

        # We only care about the k + 1 diagonal
        return np.diagonal(F, 1)

    elif method == "taylor0":

        # TODO: Recurse for poch and factorial
        order = 20
        k = np.arange(order).reshape(1, -1)
        i = np.arange(imax + 1).reshape(-1, 1)
        C = poch(-0.5, k) * poch(i, k) / (factorial(k) * poch(1 + i, k))
        return np.dot(C, (z ** k).T).reshape(-1)

    elif method == "taylor1":

        # TODO: Recurse for poch and factorial
        order = 10
        k = np.arange(order).reshape(1, -1)
        i = np.arange(imax + 1).reshape(-1, 1)
        C = poch(-0.5, k) * poch(i, k) / (factorial(k) * poch(1 + i, k))
        return np.dot(C, (z ** k).T).reshape(-1)


if __name__ == "__main__":

    # Stability checks
    import matplotlib.pyplot as plt

    imax = 15
    z = np.linspace(-3, 1, 1000)
    up = np.zeros((imax + 1, len(z)))
    down = np.zeros_like(up)
    taylor0 = np.zeros_like(up)
    exact = np.zeros_like(up)
    for k, zk in enumerate(z):
        up[:, k] = compute_W(imax, zk, "up")
        down[:, k] = compute_W(imax, zk, "down")
        taylor0[:, k] = compute_W(imax, zk, "taylor0")
        exact[:, k] = [hyp2f1(-0.5, i, i + 1, zk) for i in range(imax + 1)]

    fig, ax = plt.subplots(4, 4, sharey=True, sharex=True)
    ax = ax.flatten()
    for j in range(imax + 1):
        ax[j].plot(z, np.log10(np.maximum(np.abs(exact[j] - up[j]), 1e-15)))
        ax[j].plot(z, np.log10(np.maximum(np.abs(exact[j] - down[j]), 1e-15)))
        ax[j].plot(z, np.log10(np.maximum(np.abs(exact[j] - taylor0[j]), 1e-15)))
        ax[j].set_ylim(-15, 0)
        ax[j].axhline(-8, color="k", ls="--")
    plt.show()
