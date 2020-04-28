import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("MacOSX")


def get_f_exact(x, y, z, b):
    r"""
    Return the expression

    .. math::

        f \equiv
        \cos\theta_i 
        \mathrm{max}\left(0, \cos(\phi_r - \phi_i)\right)
        \sin\alpha \tan\beta

    from Equation (30) in Oren & Nayar (1994) as a function
    of the Cartesian coordinates on the sky-projected sphere
    seen at a phase where the semi-minor axis of the terminator
    is `b`.

    """
    bc = np.sqrt(1 - b ** 2)
    ci = bc * y - b * z
    f1 = -b / z - ci
    f2 = -b / ci - z
    f = ci * np.maximum(0, np.minimum(f1, f2))
    return f


def poly_basis(x, y, z, deg):
    """Return the polynomial basis evaluated at `x`, `y`, `z`."""
    N = (deg + 1) ** 2
    B = np.zeros((len(x * y * z), N))
    for n in range(N):
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
    return B


def get_w6(deg=4, Nb=3, res=100, inv_var=1e-4):
    """
    Return the coefficients of the 6D fit to `f`
    in `x`, `y`, `z`, `b`, and `bc`.

    """
    # Construct a 3D grid in (x, y, b)
    bgrid = np.linspace(-1, 0, res)
    xygrid = np.linspace(-1, 1, res)
    x, y, b = np.meshgrid(xygrid, xygrid, bgrid)
    z = np.sqrt(1 - x ** 2 - y ** 2)
    idx = np.isfinite(z) & (y > b * np.sqrt(1 - x ** 2))
    x = x[idx]
    y = y[idx]
    z = z[idx]
    b = b[idx]

    # Compute the exact `f` function on this grid
    f = get_f_exact(x, y, z, b)

    # Construct the design matrix for fitting
    N = (deg + 1) ** 2
    u = 0
    X = np.zeros((len(y * z * b), N * Nb ** 2))
    bc = np.sqrt(1 - b ** 2)
    B = poly_basis(x, y, z, deg)
    for n in range(N):
        for p in range(Nb):
            for q in range(Nb):
                X[:, u] = B[:, n] * b ** p * bc ** q
                u += 1

    # Solve the linear problem
    w6 = np.linalg.solve(X.T.dot(X) + inv_var * np.eye(N * Nb ** 2), X.T.dot(f))
    return w6


def get_w(b, deg, Nb, w6):
    """
    Return the coefficients of the polynomial basis fit to `f`
    for a given value of `b`.

    """
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

    # Coefficients in the polynomial basis
    N = (deg + 1) ** 2
    m = np.arange(Nb ** 2)
    w = np.zeros(N)
    for n in range(N):
        w[n] = w6[m + Nb ** 2 * n].dot(bbc[m])

    return w


def get_I(x, y, z, b, sig, f):
    # Lambertian term
    I0 = np.maximum(0, np.sqrt(1 - b ** 2) * y - b * z)

    # Oren-Nayar term
    sig2 = sig ** 2
    A = 1 - 0.5 * sig2 / (sig2 + 0.33)
    B = 0.45 * sig2 / (sig2 + 0.09)
    I = A * I0 + B * f
    return I


# Solve
deg = 4
Nb = 3
w6 = get_w6(deg=deg, Nb=Nb)

# Visualize the fit
sig = 90 * np.pi / 180
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

    # Get the two `f` functions
    w = get_w(b, deg, Nb, w6)
    fapprox = poly_basis(x, y, z, deg).dot(w)
    fexact = get_f_exact(x, y, z, b)

    # Compute the corresponding intensity
    Iapprox = get_I(x, y, z, b, sig, fapprox)
    Iexact = get_I(x, y, z, b, sig, fexact)

    # View
    ax[0, i].imshow(
        Iexact.reshape(res, res), origin="lower", extent=(-1, 1, -1, 1), vmin=0, vmax=1,
    )
    ax[1, i].imshow(
        Iapprox.reshape(res, res),
        origin="lower",
        extent=(-1, 1, -1, 1),
        vmin=0,
        vmax=1,
    )

    ax[2, i].imshow(
        np.abs((Iexact - Iapprox)).reshape(res, res),
        origin="lower",
        extent=(-1, 1, -1, 1),
        vmin=0,
        vmax=0.1,
    )

    ax[0, i].axis("off")
    ax[1, i].axis("off")
    ax[2, i].axis("off")

plt.show()
