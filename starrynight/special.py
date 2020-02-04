from mpmath import elliprf, elliprd, elliprj
from mpmath import ellipf, ellipe
from scipy.special import hyp2f1
from scipy.integrate import quad
import numpy as np


def carlson_rf(x, y, z):
    # TODO: Code this up in terms of E, F, Pi
    res = elliprf(x, y, z)
    return float(res.real)


def carlson_rd(x, y, z):
    # TODO: Code this up in terms of E, F, Pi
    res = elliprd(x, y, z)
    return float(res.real)


def carlson_rj(x, y, z, p):
    # TODO: Code this up in terms of E, F, Pi
    res = elliprj(x, y, z, p)
    return float(res.real)


def E(phi, k2):
    return float(ellipe(phi, k2).real)


def F(phi, k2):
    return float(ellipf(phi, k2).real)


def J(N, k, kappa1, kappa2):
    func = (
        lambda x: np.sin(x) ** (2 * N)
        * (np.maximum(0, 1 - np.sin(x) ** 2 / k ** 2)) ** 1.5
    )
    res, _ = quad(func, 0.5 * kappa1, 0.5 * kappa2, epsabs=1e-12, epsrel=1e-12,)
    return res
