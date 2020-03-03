import starrynight
from starrynight import c
from mpmath import ellipk, ellipf, ellipe, ellippi, elliprj
import numpy as np
import pytest


@pytest.mark.parametrize("bo,ro", [[0.5, 0.2], [0.95, 0.2]])
def test_ellip(bo, ro):

    # Evaluate over full range
    kappa = np.linspace(0, 4 * np.pi, 100)

    # Helper vars
    n = -4 * bo * ro / (ro - bo) ** 2
    k2 = (1 - ro ** 2 - bo ** 2 + 2 * bo * ro) / (4 * bo * ro)
    kp2 = 1 / k2

    # Compute
    F = np.zeros_like(kappa) * np.nan
    E = np.zeros_like(kappa) * np.nan
    PIp = np.zeros_like(kappa) * np.nan
    F_ = np.zeros_like(kappa) * np.nan
    E_ = np.zeros_like(kappa) * np.nan
    PIp_ = np.zeros_like(kappa) * np.nan
    for k in range(len(kappa)):

        # Only compute it if the answer is real
        if np.sin(kappa[k] / 2) ** 2 <= k2:

            # Our version
            (F[k], _), (E[k], _), (PIp[k], _) = c.ellip(bo, ro, [0.0, kappa[k]])

            # Computed from the Legendre form
            F_[k] = float(ellipf(kappa[k] / 2, kp2).real) - float(ellipf(0, kp2).real)
            E_[k] = float(ellipe(kappa[k] / 2, kp2).real) - float(ellipe(0, kp2).real)
            PIp_[k] = (6 / n) * (
                float((ellipf(kappa[k] / 2, kp2) - ellippi(n, kappa[k] / 2, kp2)).real)
                - float((ellipf(0, kp2) - ellippi(n, 0, kp2)).real)
            )

    assert np.allclose(F, F_, equal_nan=True)
    assert np.allclose(E, E_, equal_nan=True)
    assert np.allclose(PIp, PIp_, equal_nan=True)


@pytest.mark.parametrize("kappa", [0.25])
def test_ellip_deriv_bo(kappa):

    # Params
    bo = np.linspace(0.5, 1.0, 100)
    ro = 0.2

    # Compute analytic derivs
    F = np.zeros_like(bo) * np.nan
    E = np.zeros_like(bo) * np.nan
    PIp = np.zeros_like(bo) * np.nan
    dFdbo = np.zeros_like(bo) * np.nan
    dEdbo = np.zeros_like(bo) * np.nan
    dPIpdbo = np.zeros_like(bo) * np.nan
    for i in range(len(bo)):
        try:
            (
                (F[i], (dFdbo[i], _, _, _)),
                (E[i], (dEdbo[i], _, _, _)),
                (PIp[i], (dPIpdbo[i], _, _, _)),
            ) = c.ellip(bo[i], ro, [0.0, kappa])
        except:
            # Complex?
            pass

    # Numerical
    dFdbo_ = np.gradient(F, edge_order=2) / np.gradient(bo, edge_order=2)
    dEdbo_ = np.gradient(E, edge_order=2) / np.gradient(bo, edge_order=2)
    dPIpdbo_ = np.gradient(PIp, edge_order=2) / np.gradient(bo, edge_order=2)

    assert np.allclose(dFdbo, dFdbo_, equal_nan=True, atol=1e-4)
    assert np.allclose(dEdbo, dEdbo_, equal_nan=True, atol=1e-4)
    # TODO: assert np.allclose(dPIpdbo, dPIpdbo_, equal_nan=True, atol=1e-5)


@pytest.mark.parametrize("kappa", [0.25])
def test_ellip_deriv_ro(kappa):

    # Params
    bo = 0.2
    ro = np.linspace(0.5, 1.15, 1000)

    # Compute analytic derivs
    F = np.zeros_like(ro) * np.nan
    E = np.zeros_like(ro) * np.nan
    PIp = np.zeros_like(ro) * np.nan
    dFdro = np.zeros_like(ro) * np.nan
    dEdro = np.zeros_like(ro) * np.nan
    dPIpdro = np.zeros_like(ro) * np.nan
    for i in range(len(ro)):
        try:
            (
                (F[i], (_, dFdro[i], _, _)),
                (E[i], (_, dEdro[i], _, _)),
                (PIp[i], (_, dPIpdro[i], _, _)),
            ) = c.ellip(bo, ro[i], [0.0, kappa])
        except:
            # Complex?
            pass

    # Numerical
    dFdro_ = np.gradient(F, edge_order=2) / np.gradient(ro, edge_order=2)
    dEdro_ = np.gradient(E, edge_order=2) / np.gradient(ro, edge_order=2)
    dPIpdro_ = np.gradient(PIp, edge_order=2) / np.gradient(ro, edge_order=2)

    assert np.allclose(dFdro, dFdro_, equal_nan=True, atol=1e-4)
    assert np.allclose(dEdro, dEdro_, equal_nan=True, atol=1e-4)
    # TODO: assert np.allclose(dPIpdro, dPIpdro_, equal_nan=True, atol=1e-5)
