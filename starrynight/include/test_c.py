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


@pytest.mark.parametrize("kappa", [0.25, 1.25, 2.25])
def test_ellip_deriv_bo(kappa):

    # Params
    bo = np.linspace(0.5, 1.0, 300)
    ro = 0.2
    eps = 1e-8

    # Compute
    dFdbo = np.zeros_like(bo) * np.nan
    dEdbo = np.zeros_like(bo) * np.nan
    dPIpdbo = np.zeros_like(bo) * np.nan
    dFdbo_ = np.zeros_like(bo) * np.nan
    dEdbo_ = np.zeros_like(bo) * np.nan
    dPIpdbo_ = np.zeros_like(bo) * np.nan
    for i in range(len(bo)):
        try:
            # Analytic deriv
            (
                (_, (dFdbo[i], _, _, _)),
                (_, (dEdbo[i], _, _, _)),
                (_, (dPIpdbo[i], _, _, _)),
            ) = c.ellip(bo[i], ro, [0.0, kappa])

            # Numerical deriv
            ((F1, _), (E1, _), (PIp1, _),) = c.ellip(bo[i] - eps, ro, [0.0, kappa])
            ((F2, _), (E2, _), (PIp2, _),) = c.ellip(bo[i] + eps, ro, [0.0, kappa])
            dFdbo_[i] = (F2 - F1) / (2 * eps)
            dEdbo_[i] = (E2 - E1) / (2 * eps)
            dPIpdbo_[i] = (PIp2 - PIp1) / (2 * eps)

        except:
            # Complex?
            pass

    assert np.allclose(dFdbo[1:-1], dFdbo_[1:-1], equal_nan=True, atol=1e-6)
    assert np.allclose(dEdbo[1:-1], dEdbo_[1:-1], equal_nan=True, atol=1e-6)
    # TODO: assert np.allclose(dPIpdbo, dPIpdbo_, equal_nan=True, atol=1e-4)


@pytest.mark.parametrize("kappa", [0.25, 1.25, 2.25])
def test_ellip_deriv_ro(kappa):

    # Params
    bo = 0.2
    ro = np.linspace(0.5, 1.15, 1000)
    eps = 1e-8

    # Compute
    dFdro = np.zeros_like(ro) * np.nan
    dEdro = np.zeros_like(ro) * np.nan
    dPIpdro = np.zeros_like(ro) * np.nan
    dFdro_ = np.zeros_like(ro) * np.nan
    dEdro_ = np.zeros_like(ro) * np.nan
    dPIpdro_ = np.zeros_like(ro) * np.nan
    for i in range(len(ro)):
        try:
            # Analytic deriv
            (
                (_, (_, dFdro[i], _, _)),
                (_, (_, dEdro[i], _, _)),
                (_, (_, dPIpdro[i], _, _)),
            ) = c.ellip(bo, ro[i], [0.0, kappa])

            # Numerical deriv
            ((F1, _), (E1, _), (PIp1, _),) = c.ellip(bo, ro[i] - eps, [0.0, kappa])
            ((F2, _), (E2, _), (PIp2, _),) = c.ellip(bo, ro[i] + eps, [0.0, kappa])
            dFdro_[i] = (F2 - F1) / (2 * eps)
            dEdro_[i] = (E2 - E1) / (2 * eps)
            dPIpdro_[i] = (PIp2 - PIp1) / (2 * eps)

        except:
            # Complex?
            pass

    assert np.allclose(dFdro[1:-1], dFdro_[1:-1], equal_nan=True, atol=1e-6)
    assert np.allclose(dEdro[1:-1], dEdro_[1:-1], equal_nan=True, atol=1e-6)
    # TODO: assert np.allclose(dPIpdro, dPIpdro_, equal_nan=True, atol=1e-4)
