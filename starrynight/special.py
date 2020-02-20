from .utils import *
from mpmath import elliprf, elliprd, elliprj
from mpmath import ellipf, ellipe, ellipk
from scipy.special import hyp2f1 as scipy_hyp2f1
from scipy.integrate import quad
import numpy as np


def carlson_rf(x, y, z):
    return float(elliprf(x, y, z).real)


def carlson_rd(x, y, z):
    return float(elliprd(x, y, z).real)


def carlson_rj(x, y, z, p):
    return float(elliprj(x, y, z, p).real)


def J(N, k2, kappa, gradient=False):
    # We'll need to solve this with gaussian quadrature
    func = (
        lambda x: np.sin(x) ** (2 * N) * (np.maximum(0, 1 - np.sin(x) ** 2 / k2)) ** 1.5
    )
    res = 0.0
    for i in range(0, len(kappa), 2):
        res += quad(
            func, 0.5 * kappa[i], 0.5 * kappa[i + 1], epsabs=1e-12, epsrel=1e-12,
        )[0]
    if gradient:
        # Deriv w/ respect to kappa is analytic
        dJdkappa = (
            0.5
            * (
                np.sin(0.5 * kappa) ** (2 * N)
                * (np.maximum(0, 1 - np.sin(0.5 * kappa) ** 2 / k2)) ** 1.5
            )
            * np.repeat([-1, 1], len(kappa) // 2).reshape(1, -1)
        )

        # Deriv w/ respect to k2 is tricky, need to integrate
        func = (
            lambda x: (1.5 / k2 ** 2)
            * np.sin(x) ** (2 * N + 2)
            * (np.maximum(0, 1 - np.sin(x) ** 2 / k2)) ** 0.5
        )
        dJdk2 = 0.0
        for i in range(0, len(kappa), 2):
            dJdk2 += quad(
                func, 0.5 * kappa[i], 0.5 * kappa[i + 1], epsabs=1e-12, epsrel=1e-12,
            )[0]

        return res, (dJdk2, dJdkappa)
    else:
        return res


def hyp2f1(a, b, c, z, gradient=False):
    term = a * b * z / c
    value = 1.0 + term
    n = 1
    while (np.abs(term) > STARRY_2F1_TOL) and (n < STARRY_2F1_MAXITER):
        a += 1
        b += 1
        c += 1
        n += 1
        term *= a * b * z / c / n
        value += term
    if n == STARRY_2F1_MAXITER:
        raise ValueError("Series for 2F1 did not converge.")
    if gradient:
        dFdz = a * b / c * hyp2f1(a + 1, b + 1, c + 1, z)
        return value, dFdz
    else:
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
    p = np.sqrt((1 + kc * kc * c) / d)
    d = x / d
    c = d / (2 * p)
    z = a - b
    i = a
    a = (b + a) / 2
    y = np.abs(1 / x)
    f = 0
    l = np.zeros_like(x)
    m = 1
    kc = np.abs(kc)

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

        y[y == 0] = np.sqrt(e) * c[y == 0] * b

        if np.abs(g - kc) > STARRY_EL2_CA * g:

            kc = np.sqrt(e) * 2
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
    e = (np.arctan(m / y) + np.pi * l) * a / m
    e[x < 0] = -e[x < 0]

    return e + c * z


def ellip(bo, ro, kappa, k2):

    # Helper variables
    k = np.sqrt(k2)
    k2inv = 1 / k2
    kinv = np.sqrt(k2inv)
    kc2 = 1 - k2
    kc = np.sqrt(kc2)
    kc2inv = 1 - k2inv
    kcinv = np.sqrt(kc2inv)

    if k2 < 1:
        K0 = float(ellipk(k2))
        E0 = float(ellipe(k2))
        E0 = np.sqrt(k2inv) * (E0 - (1 - k2) * K0)
        K0 *= np.sqrt(k2)
    else:
        K0 = float(ellipk(k2inv))
        E0 = float(ellipe(k2inv))

    if k2 < 1:

        # Analytic continuation from (17.4.15-16) in Abramowitz & Stegun
        # A better format is here: https://dlmf.nist.gov/19.7#ii

        # Helper variables
        arg = kinv * np.sin(kappa / 2)
        tanphi = arg / np.sqrt(1 - arg ** 2)
        tanphi[arg >= 1] = STARRY_HUGE_TAN
        tanphi[arg <= -1] = -STARRY_HUGE_TAN

        # Compute the elliptic integrals
        F = el2(tanphi, kc, 1, 1) * k
        E = kinv * (el2(tanphi, kc, 1, kc2) - kc2 * kinv * F)
        RF = -F
        RD = (E - F) * 3 * k2

        # Add offsets to account for the limited domain of `el2`
        for i in range(len(kappa)):
            if kappa[i] > 3 * np.pi:
                F[i] += 4 * K0
                E[i] += 4 * E0
            elif kappa[i] > np.pi:
                F[i] = 2 * K0 - F[i]
                E[i] = 2 * E0 - E[i]
                RF[i] *= -1
                RD[i] *= -1

    else:

        # Helper variables
        tanphi = np.tan(kappa / 2)

        # Compute the elliptic integrals
        F = el2(tanphi, kcinv, 1, 1)
        E = el2(tanphi, kcinv, 1, kc2inv)
        RF = -F
        RD = (E - F) * 3 * k2

        # Add offsets to account for the limited domain of `el2`
        for i in range(len(kappa)):
            if kappa[i] > 3 * np.pi:
                F[i] += 4 * K0
                E[i] += 4 * E0
            elif kappa[i] > np.pi:
                F[i] += 2 * K0
                E[i] += 2 * E0

    # TODO: Code up the integral of the third kind
    phi = (kappa - np.pi) % (2 * np.pi)
    p = (ro * ro + bo * bo + 2 * ro * bo * np.cos(phi)) / (
        ro * ro + bo * bo - 2 * ro * bo
    )
    cx = np.cos(phi / 2)
    sx = np.sin(phi / 2)
    w = 1 - cx ** 2 / k2
    if np.abs(bo - ro) > STARRY_PAL_BO_EQUALS_RO_TOL:
        RJ = (
            (np.cos(phi) + 1)
            * cx
            * np.array(
                [carlson_rj(w[i], sx[i] * sx[i], 1.0, p[i]) for i in range(len(w))]
            )
        )

        p = (ro * ro + bo * bo + 2 * ro * bo) / (ro * ro + bo * bo - 2 * ro * bo)
        RJ0 = -4.0 * carlson_rj(1 - 1 / k2, 0.0, 1.0, p)

    else:
        RJ = np.zeros_like(phi)
        RJ0 = 0.0

    RF0 = -2 * K0
    RD0 = 2 * (E0 - K0) * 3 * k2

    # Return the definite elliptic integrals
    return (
        pairdiff(F),
        pairdiff(E),
        pairdiff(RF),
        pairdiff(RD),
        pairdiff(RJ),
        RF0,
        RD0,
        RJ0,
    )

