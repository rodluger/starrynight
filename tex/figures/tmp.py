import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("MacOSX")


def get_S_exact(x, y, z, b):
    bc = np.sqrt(1 - b ** 2)
    ci = bc * y - b * z
    f1 = -b / z - ci
    f2 = -b / ci - z
    S = ci * np.maximum(0, np.minimum(f1, f2))
    return S


def fit(x, y, z, b, deg, Nb):
    # Construct the design matrix
    N = (deg + 1) ** 2
    n = 0
    X = np.zeros((len(y * z * b), N * Nb ** 2))
    bc = np.sqrt(1 - b ** 2)
    for n in range(N):

        # Get the nth term in the poly basis
        l = int(np.floor(np.sqrt(n)))
        m = n - l * l - l
        mu = l - m
        nu = l + m
        if nu % 2 == 0:
            i = mu // 2
            j = nu // 2
            k = 0
        else:
            i = (mu - 1) // 2
            j = (nu - 1) // 2
            k = 1
        term = x ** i * y ** j * z ** k

        for p in range(Nb):
            for q in range(Nb):
                X[:, n] = term * b ** p * bc ** q
                n += 1

    # Solve the linear problem
    w = np.linalg.solve(X.T.dot(X), X.T.dot(S))
    return w


def get_S(x, y, z, b, deg, Nb, w):

    # Compute the bbc basis
    bc = np.sqrt(1 - b ** 2)
    bbc = np.zeros(Nb ** 2)
    bbc[0] = 1
    for l in range(1, Nb):
        bbc[l] = bc * bbc[l - 1]
    for k in range(1, Nb):
        bbc[k * Nb] = b * bbc[(k - 1) * Nb]
        for l in range(1, Nb):
            bbc[k * Nb + l] = bc * bbc[k * Nb + l - 1]

    # coefficients in the polynomial basis
    N = (deg + 1) ** 2
    m = np.arange(Nb ** 2)
    c = np.zeros(N)
    for n in range(N):
        c[n] = w[m + Nb ** 2 * n].dot(bbc[m])

    breakpoint()

    return S


# Data grid
res = 100
bgrid = np.linspace(-1, 0, res)
xygrid = np.linspace(-1, 1, res)
x, y, b = np.meshgrid(xygrid, xygrid, bgrid)
z = np.sqrt(1 - x ** 2 - y ** 2)
idx = np.isfinite(z) & (y > b * np.sqrt(1 - x ** 2))
x = x[idx]
y = y[idx]
z = z[idx]
b = b[idx]
S = get_S_exact(x, y, z, b)

# Solve
deg = 5
Nb = 3
w = fit(x, y, z, b, deg, Nb)

# Visualize the fit
res = 300
grid = np.linspace(-1, 1, res)
x, y = np.meshgrid(grid, grid)
z = np.sqrt(1 - x ** 2 - y ** 2)
x = x.flatten()
y = y.flatten()
z = z.flatten()

bs = np.linspace(-1, 0, 10)[:-1]
fig, ax = plt.subplots(3, len(bs))
for i, b in enumerate(bs):
    night = y < b * np.sqrt(1 - x ** 2)
    Sapprox = get_S(x, y, z, b, deg, Nb, w)
    Sapprox[night] *= 0
    Sexact = get_S_exact(x, y, z, b)

    ax[0, i].imshow(
        Sexact.reshape(res, res), origin="lower", extent=(-1, 1, -1, 1), vmin=0, vmax=1,
    )
    ax[1, i].imshow(
        Sapprox.reshape(res, res),
        origin="lower",
        extent=(-1, 1, -1, 1),
        vmin=0,
        vmax=1,
    )

    ax[2, i].imshow(
        np.abs((Sexact - Sapprox)).reshape(res, res),
        origin="lower",
        extent=(-1, 1, -1, 1),
        vmin=0,
        vmax=0.1,
    )

    ax[0, i].axis("off")
    ax[1, i].axis("off")
    ax[2, i].axis("off")

plt.show()
