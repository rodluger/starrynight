from .configdefaults import config
from .utils import *
from mpmath import elliprf, elliprd, elliprj
from mpmath import ellipf, ellipe, ellipk
from scipy.special import hyp2f1 as scipy_hyp2f1
from scipy.integrate import quad


def carlson_rf(x, y, z):
    return float(elliprf(x, y, z).real)


def carlson_rd(x, y, z):
    return float(elliprd(x, y, z).real)


def carlson_rj(x, y, z, p):
    return float(elliprj(x, y, z, p).real)


def J(N, k2, kappa):
    # We'll need to solve this with gaussian quadrature
    func = (
        lambda x: config.np.sin(x) ** (2 * N)
        * (config.np.maximum(0, 1 - config.np.sin(x) ** 2 / k2)) ** 1.5
    )
    res = 0.0
    for i in range(0, len(kappa), 2):
        res += quad(
            func, 0.5 * kappa[i], 0.5 * kappa[i + 1], epsabs=1e-12, epsrel=1e-12,
        )[0]
    return res


def hyp2f1(a, b, c, z):
    term = a * b * z / c
    value = 1.0 + term
    n = 1
    while (config.np.abs(term) > STARRY_2F1_TOL) and (n < STARRY_2F1_MAXITER):
        a += 1
        b += 1
        c += 1
        n += 1
        term *= a * b * z / c / n
        value += term
    if n == STARRY_2F1_MAXITER:
        raise ValueError("Series for 2F1 did not converge.")
    return value


def el2(x, kc, a, b):
    """
    Vectorized implementation of the `el2` function from
    Bulirsch (1965). In this case, `x` is a *vector* of integration
    limits. The halting condition does not depend on the value of `x`,
    so it's much faster to evaluate all values of `x` at once!
    
    """
    if kc == 0:
        raise ValueError("Elliptic integral diverged because k = 1.")

    c = x * x
    d = 1 + c
    p = config.np.sqrt((1 + kc * kc * c) / d)
    d = x / d
    c = d / (2 * p)
    z = a - b
    i = a
    a = (b + a) / 2
    y = config.np.abs(1 / x)
    f = 0
    l = config.np.zeros_like(x)
    m = 1
    kc = config.np.abs(kc)

    for n in range(STARRY_EL2_MAX_ITER):

        b = i * kc + b
        e = m * kc
        g = e / p
        d = f * g + d
        f = c
        i = a
        p = g + p
        c = (d / p + c) / 2
        g = m
        m = kc + m
        a = (b / m + a) / 2
        y = -e / y + y

        y[y == 0] = config.np.sqrt(e) * c[y == 0] * b

        if config.np.abs(g - kc) > STARRY_EL2_CA * g:

            kc = config.np.sqrt(e) * 2
            l = l * 2
            l[y < 0] = 1 + l[y < 0]

        else:

            break

    if n == STARRY_EL2_MAX_ITER - 1:
        raise ValueError(
            "Elliptic integral failed to converge after {} iterations.".format(
                STARRY_EL2_MAX_ITER
            )
        )

    l[y < 0] = 1 + l[y < 0]
    e = (config.np.arctan(m / y) + config.np.pi * l) * a / m
    e[x < 0] = -e[x < 0]

    return e + c * z


def dF(phi, k2):
    """
    Returns the difference of a pair (or pairs) of incomplete elliptic integrals
    of the first kind. Specifically, returns the sum of

        F(phi[i + 1], k^2) - F(phi[i], k^2)

    for i = 0, 2, 4, ...

    """
    if k2 > 1:
        K = float(ellipk(1 / k2))  # = cel(kcinv, 1, 1, 1)
        K /= config.np.sqrt(k2)
    else:
        K = float(ellipk(k2))  # = cel(kc, 1, 1, 1)

    if k2 > 1:

        # Analytic continuation from (17.4.15) in Abramowitz & Stegun

        # Helper variables
        k = config.np.sqrt(k2)
        kc2 = 1 - 1 / k2
        kc = config.np.sqrt(kc2)
        arg = k * config.np.sin(phi)
        tanphi = arg / config.np.sqrt(1 - arg ** 2)
        tanphi[arg >= 1] = STARRY_HUGE_TAN
        tanphi[arg <= -1] = -STARRY_HUGE_TAN

        # Compute the elliptic integrals
        res = el2(tanphi, kc, 1, 1) / k

        # Add offsets to account for the limited domain of `el2`
        for i in range(len(phi)):
            if phi[i] > config.np.pi / 2:
                res[i] = 2 * K - res[i]
            if phi[i] > 3 * config.np.pi / 2:
                res[i] = 6 * K - res[i]

    else:

        # Helper variables
        kc2 = 1 - k2
        kc = config.np.sqrt(kc2)
        tanphi = config.np.tan(phi)

        # Compute the elliptic integrals
        res = el2(tanphi, kc, 1, 1)

        # Add offsets to account for the limited domain of `el2`
        for i in range(len(phi)):
            if phi[i] > config.np.pi / 2:
                res[i] += 2 * K
            if phi[i] > 3 * config.np.pi / 2:
                res[i] += 2 * K

    return pairdiff(res)


def dE(phi, k2):
    """
    Returns the difference of a pair (or pairs) of incomplete elliptic integrals
    of the second kind. Specifically, returns the sum of

        E(phi[i + 1], k^2) - E(phi[i], k^2)

    for i = 0, 2, 4, ...

    """
    if k2 > 1:
        K = float(ellipk(1 / k2))  # = cel(kcinv, 1, 1, 1)
        E = float(ellipe(1 / k2))  # = cel(kcinv, 1, 1, kcinv2)
        E = config.np.sqrt(k2) * (E - (1 - 1 / k2) * K)
    else:
        E = float(ellipe(k2))  # = cel(kc, 1, 1, kc2)

    if k2 > 1:

        # Analytic continuation from (17.4.16) in Abramowitz & Stegun
        # A better format is here: https://dlmf.nist.gov/19.7#ii

        # Helper variables
        k = config.np.sqrt(k2)
        k2inv = 1 / k2
        kcinv2 = 1 - k2inv
        kcinv = config.np.sqrt(kcinv2)
        arg = k * config.np.sin(phi)
        tanphi = arg / config.np.sqrt(1 - arg ** 2)
        tanphi[arg >= 1] = STARRY_HUGE_TAN
        tanphi[arg <= -1] = -STARRY_HUGE_TAN

        # Compute the elliptic integrals
        res = k * (el2(tanphi, kcinv, 1, kcinv2) - kcinv2 * el2(tanphi, kcinv, 1, 1))

        # Add offsets to account for the limited domain of `el2`
        for i in range(len(phi)):
            if phi[i] > config.np.pi / 2:
                res[i] = 2 * E - res[i]
            if phi[i] > 3 * config.np.pi / 2:
                res[i] = 6 * E - res[i]

    else:

        # Helper variables
        kc2 = 1 - k2
        kc = config.np.sqrt(kc2)
        tanphi = config.np.tan(phi)

        # Compute the elliptic integrals
        res = el2(tanphi, kc, 1, kc2)

        # Add offsets to account for the limited domain of `el2`
        for i in range(len(phi)):
            if phi[i] > config.np.pi / 2:
                res[i] += 2 * E
            if phi[i] > 3 * config.np.pi / 2:
                res[i] += 2 * E

    return pairdiff(res)
