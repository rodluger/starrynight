from starrynight import c
from mpmath import ellipf, ellipe, ellippi, elliprj
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
    # Params
    kappa = np.linspace(-np.pi, np.pi, 100)
    bo = 0.9
    ro = 0.2
    p = (ro * ro + bo * bo - 2 * ro * bo * np.cos(kappa)) / (
        ro * ro + bo * bo - 2 * ro * bo
    )
    k2 = (1 - ro ** 2 - bo ** 2 + 2 * bo * ro) / (4 * bo * ro)

    # Starry expression
    F1 = c.PIprime(kappa, k2, p)

    # Compute it from RJ
    phi = kappa / 2
    n = -4 * bo * ro / (ro - bo) ** 2
    kp2 = 1 / k2
    F2 = (
        -2.0
        * np.sin(phi) ** 3
        * np.array(
            [
                float(
                    elliprj(
                        np.cos(phi_i) ** 2,
                        1 - kp2 * np.sin(phi_i) ** 2,
                        1,
                        1 - n * np.sin(phi_i) ** 2,
                    ).real
                )
                for phi_i in phi
            ]
        )
    )

    # Compute it from PI and F
    F3 = np.array(
        [
            6 / n * float((ellipf(phi_i, kp2) - ellippi(n, phi_i, kp2)).real)
            for phi_i in phi
        ]
    )

    F1[1 - kp2 * np.sin(phi) ** 2 < 0] = np.nan
    F2[1 - kp2 * np.sin(phi) ** 2 < 0] = np.nan
    F3[1 - kp2 * np.sin(phi) ** 2 < 0] = np.nan

    assert np.allclose(F1, F2, equal_nan=True)
    assert np.allclose(F1, F3, equal_nan=True)
