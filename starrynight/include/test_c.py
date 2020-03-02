from starrynight import c
from mpmath import ellipk, ellipf, ellipe, ellippi, elliprj
import numpy as np
import pytest


@pytest.mark.parametrize("phi,k2", [[1.0, 0.75], [0.5, 0.25], [-1.0, 0.5]])
def test_F(phi, k2):
    F1 = c.F(np.tan([phi]), k2)[0]
    F2 = float(ellipf(phi, k2).real)
    assert np.allclose(F1, F2)


@pytest.mark.parametrize("phi,k2", [[1.0, 0.75], [0.5, 0.25], [-1.0, 0.5]])
def test_dFdtanphi(phi, k2):
    F1 = c.dFdtanphi(np.tan([phi]), k2)[0]
    eps = 1e-8
    phi1 = np.arctan(np.tan(phi) - eps)
    phi2 = np.arctan(np.tan(phi) + eps)
    F2 = float((ellipf(phi2, k2) - ellipf(phi1, k2)).real / (2 * eps))
    assert np.allclose(F1, F2)


@pytest.mark.parametrize("phi,k2", [[1.0, 0.75], [0.5, 0.25], [-1.0, 0.5]])
def test_dFdk2(phi, k2):
    F1 = c.dFdk2(np.tan([phi]), k2)[0]
    eps = 1e-8
    F2 = float((ellipf(phi, k2 + eps) - ellipf(phi, k2 - eps)).real / (2 * eps))
    assert np.allclose(F1, F2)


@pytest.mark.parametrize("phi,k2", [[1.0, 0.75], [0.5, 0.25], [-1.0, 0.5]])
def test_E(phi, k2):
    F1 = c.E(np.tan([phi]), k2)[0]
    F2 = float(ellipe(phi, k2).real)
    assert np.allclose(F1, F2)


@pytest.mark.parametrize("phi,k2", [[1.0, 0.75], [0.5, 0.25], [-1.0, 0.5]])
def test_dEdtanphi(phi, k2):
    F1 = c.dEdtanphi(np.tan([phi]), k2)[0]
    eps = 1e-8
    phi1 = np.arctan(np.tan(phi) - eps)
    phi2 = np.arctan(np.tan(phi) + eps)
    F2 = float((ellipe(phi2, k2) - ellipe(phi1, k2)).real / (2 * eps))
    assert np.allclose(F1, F2)


@pytest.mark.parametrize("phi,k2", [[1.0, 0.75], [0.5, 0.25], [-1.0, 0.5]])
def test_dEdk2(phi, k2):
    F1 = c.dEdk2(np.tan([phi]), k2)[0]
    eps = 1e-8
    F2 = float((ellipe(phi, k2 + eps) - ellipe(phi, k2 - eps)).real / (2 * eps))
    assert np.allclose(F1, F2)


def test_PIprime():

    # Params (note this integral is used only when k2 > 1)
    bo = 0.5
    ro = 0.2
    kappa = np.linspace(0, 4 * np.pi, 500)

    # Helper vars
    n = -4 * bo * ro / (ro - bo) ** 2
    k2 = (1 - ro ** 2 - bo ** 2 + 2 * bo * ro) / (4 * bo * ro)
    kp2 = 1 / k2

    # Compute
    PIp1 = np.zeros_like(kappa)
    PIp2 = np.zeros_like(kappa)
    for k in range(len(kappa)):

        # Our version (note the - sign undoes the "pairdiff" op)
        _, _, PIp1[k] = -c.ellip(bo, ro, [kappa[k]])

        # Computed from the Legendre form
        PIp2[k] = (
            6
            / n
            * float((ellipf(kappa[k] / 2, kp2) - ellippi(n, kappa[k] / 2, kp2)).real)
        )

    assert np.allclose(PIp1, PIp2)


if __name__ == "__main__":

    # DEBUG
    k2 = 1.5
    kappa = np.linspace(0, 4 * np.pi, 100)
    phi = kappa / 2

    for k in range(len(kappa)):

        F1[k] = c.dFdtanphi(np.tan([phi]), k2)[0]

        eps = 1e-8
        phi1 = np.arctan(np.tan(phi) - eps)
        phi2 = np.arctan(np.tan(phi) + eps)
        F2[k] = float((ellipf(phi2, k2) - ellipf(phi1, k2)).real / (2 * eps))

    import matplotlib.pyplot as plt

    plt.plot(kappa, F1)
    plt.plot(kappa, F2)
    plt.show()
