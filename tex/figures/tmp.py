import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("MacOSX")


def get_S_exact(y, z, b):
    bc = np.sqrt(1 - b ** 2)
    ci = bc * y - b * z
    f1 = -b / z - ci
    f2 = -b / ci - z
    S = ci * np.maximum(0, np.minimum(f1, f2))
    return S


def fit(y, z, b, Nyz, Nbbc):
    # Construct the design matrix
    n = 0
    X = np.zeros((len(y * z * b), Nyz ** 2 * Nbbc ** 2))
    bc = np.sqrt(1 - b ** 2)
    for i in range(Nyz):
        for j in range(Nyz):
            for k in range(Nbbc):
                for l in range(Nbbc):
                    X[:, n] = y ** i * z ** j * b ** k * bc ** l
                    n += 1

    # Solve the linear problem
    w = np.linalg.solve(X.T.dot(X), X.T.dot(S))
    return w


def get_S(y, z, b, Nyz, Nbbc, w):

    # bbc basis
    bc = np.sqrt(1 - b ** 2)
    bbc = np.zeros(Nbbc ** 2)
    bbc[0] = 1
    for l in range(1, Nbbc):
        bbc[l] = bc * bbc[l - 1]
    for k in range(1, Nbbc):
        bbc[k * Nbbc] = b * bbc[(k - 1) * Nbbc]
        for l in range(1, Nbbc):
            bbc[k * Nbbc + l] = bc * bbc[k * Nbbc + l - 1]

    # coefficients in the y-z basis
    n = 0
    m = np.arange(Nbbc ** 2)
    c = np.zeros(Nyz * Nyz)
    for i in range(Nyz):
        for j in range(Nyz):
            c[n] = w[m + Nbbc ** 2 * n].dot(bbc[m])
            n += 1

    # Easy way out
    n = 0
    S = 0
    for i in range(Nyz):
        for j in range(Nyz):
            S += c[n] * y ** i * z ** j
            n += 1
    return S

    # Construct `c` in actual polynomial basis: `p`
    # A3 transforms from c to p
    # sA3 = np.zeros(((2 * Nyz + 1) ** 2), (Nyz * Nyz))
    for i in range(Nyz):
        for j in range(Nyz):
            pass

    # Polynomial basis
    npts = len(y)
    deg = Nyz
    B = np.zeros((npts, (deg + 1) ** 2))
    for n in range((deg + 1) ** 2):
        if n == 0:
            B[:, n] = 1
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
        B[:, n] = x ** i * y ** j * z ** k

    return S


# Data grid
res = 100
bgrid = np.linspace(-1, 0, res)
xygrid = np.linspace(-1, 1, res)
x, y, b = np.meshgrid(xygrid, xygrid, bgrid)
z = np.sqrt(1 - x ** 2 - y ** 2)
idx = np.isfinite(z) & (y > b * np.sqrt(1 - x ** 2))
y = y[idx]
z = z[idx]
b = b[idx]
S = get_S_exact(y, z, b)

# Solve
Nyz = 3
Nbbc = 3
w = fit(y, z, b, Nyz, Nbbc)

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
    Sapprox = get_S(y, z, b, Nyz, Nbbc, w)
    Sapprox[night] *= 0
    Sexact = get_S_exact(y, z, b)

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
