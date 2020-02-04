from mpmath import elliprf, elliprd, elliprj
from mpmath import ellipf, ellipe
from scipy.special import hyp2f1
from scipy.integrate import quad
import numpy as np


def carlson_rf(x, y, z):
    return float(elliprf(x, y, z).real)


def carlson_rd(x, y, z):
    return float(elliprd(x, y, z).real)


def carlson_rj(x, y, z, p):
    return float(elliprj(x, y, z, p).real)


@np.vectorize
def E(phi, k2):
    return float(ellipe(phi, k2).real)


@np.vectorize
def F(phi, k2):
    return float(ellipf(phi, k2).real)


def J(N, k2, kappa):
    func = (
        lambda x: np.sin(x) ** (2 * N) * (np.maximum(0, 1 - np.sin(x) ** 2 / k2)) ** 1.5
    )
    res = 0.0
    for i in range(0, len(kappa), 2):
        res += quad(
            func, 0.5 * kappa[i], 0.5 * kappa[i + 1], epsabs=1e-12, epsrel=1e-12,
        )[0]
    return res
