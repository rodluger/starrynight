from starrynight import c
from mpmath import ellipf, ellipe
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
