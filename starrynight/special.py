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


def hyp2f1(a, b, c, z):
    return scipy_hyp2f1(a, b, c, z)


def el2(x, kc, a, b):
    D = 16
    ca = 10 ** (-D / 2)

    if x == 0:
        return 0
    elif kc == 0:
        raise ValueError("kc = 0")

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
    l = 0
    m = 1
    kc = np.abs(kc)

    while True:

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

        if y == 0:
            y = np.sqrt(e) * c * b

        if np.abs(g - kc) > ca * g:

            kc = np.sqrt(e) * 2
            l = l * 2
            if y < 0:
                l = 1 + l

        else:

            break

    if y < 0:
        l = 1 + l
    e = (np.arctan(m / y) + np.pi * l) * a / m
    if x < 0:
        e = -e

    return e + c * z


def del2(x, kc, a, b):
    D = 16
    ca = 10 ** (-D / 2)

    if kc == 0:
        # TODO
        raise ValueError("kc = 0")

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

    while True:

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

        if np.abs(g - kc) > ca * g:

            kc = np.sqrt(e) * 2
            l = l * 2
            l[y < 0] = 1 + l[y < 0]

        else:

            break

    l[y < 0] = 1 + l[y < 0]
    e = (np.arctan(m / y) + np.pi * l) * a / m
    e[x < 0] = -e[x < 0]

    res = e + c * z
    return sum(-np.array(res)[::2] + np.array(res)[1::2])


def dF(phi, k2):
    kc2 = 1 - k2
    kc = np.sqrt(kc2)
    res = del2(np.tan(phi), kc, 1, 1)

    for phi1, phi2 in zip(phi[::2], phi[1::2]):
        for x in [np.pi / 2, 3 * np.pi / 2]:
            if (phi2 > x) and (phi1 < x):
                res += 2 * float(ellipk(k2).real)  # = 2 * cel(kc, 1, 1, 1)

    return res


def dE(phi, k2):
    kc2 = 1 - k2
    kc = np.sqrt(kc2)
    res = del2(np.tan(phi), kc, 1, kc2)

    for phi1, phi2 in zip(phi[::2], phi[1::2]):
        for x in [np.pi / 2, 3 * np.pi / 2]:
            if (phi2 > x) and (phi1 < x):
                res += 2 * float(ellipe(k2).real)  # = 2 * cel(kc, 1, 1, kc2)

    return res


def bF(phi, k2):
    kc2 = 1 - k2
    kc = np.sqrt(kc2)
    res = el2(np.tan(phi), kc, 1, 1)

    if phi > np.pi / 2:
        offset = 2 * float(ellipk(k2).real)  # = 2 * cel(kc, 1, 1, 1)
        res += offset
        if phi > 3 * np.pi / 2:
            res += offset

    return res


def bE(phi, k2):
    kc2 = 1 - k2
    kc = np.sqrt(kc2)
    res = el2(np.tan(phi), kc, 1, kc2)

    if phi > np.pi / 2:
        offset = 2 * float(ellipe(k2).real)  # = 2 * cel(kc, 1, 1, kc2)
        res += offset
        if phi > 3 * np.pi / 2:
            res += offset

    return res


def test_el2():

    import matplotlib.pyplot as plt

    phi = np.linspace(0, 2 * np.pi, 100)
    k2 = np.linspace(0, 1, 50, endpoint=False)

    for k2i in k2:
        diffF = np.abs([F(phii, k2i) - bF(phii, k2i) for phii in phi])
        diffE = np.abs([E(phii, k2i) - bE(phii, k2i) for phii in phi])
        plt.plot(phi, diffE, ".", alpha=0.3, color="k")
        plt.plot(phi, diffF, ".", alpha=0.3, color="k")

    plt.yscale("log")
    plt.show()


def test_del2():

    import matplotlib.pyplot as plt

    k2 = [1 - 1e-10]

    for phi1 in np.linspace(0, 2 * np.pi, 15, endpoint=False):

        phi2 = np.linspace(phi1 + 0.1, 2 * np.pi, 15)

        for k2i in k2:

            # First kind
            y1 = np.array(
                [float((ellipf(phi2i, k2i) - ellipf(phi1, k2i)).real) for phi2i in phi2]
            )
            y2 = np.array([dF([phi1, phi2i], k2i) for phi2i in phi2])
            diffF = np.abs(y1 - y2)
            diffF[np.isnan(diffF)] = 1

            # Second kind
            y1 = np.array(
                [float((ellipe(phi2i, k2i) - ellipe(phi1, k2i)).real) for phi2i in phi2]
            )
            y2 = np.array([dE([phi1, phi2i], k2i) for phi2i in phi2])
            diffE = np.abs(y1 - y2)
            diffE[np.isnan(diffE)] = 1

            plt.plot(phi2, diffF, "k.", alpha=0.3)
            plt.plot(phi2, diffE, "k.", alpha=0.3)

    plt.yscale("log")
    plt.show()
