from starrynight import c
from mpmath import ellipf, ellipe, ellippi
from scipy.special import hyp2f1
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

        # Only compute it if the answer is real
        k2 = (1 - ro ** 2 - bo[i] ** 2 + 2 * bo[i] * ro) / (4 * bo[i] * ro)
        if np.sin(kappa / 2) ** 2 <= k2:

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

    assert np.allclose(dFdbo[1:-1], dFdbo_[1:-1], equal_nan=True, atol=1e-6)
    assert np.allclose(dEdbo[1:-1], dEdbo_[1:-1], equal_nan=True, atol=1e-6)
    assert np.allclose(dPIpdbo[1:-1], dPIpdbo_[1:-1], equal_nan=True, atol=1e-4)


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

        # Only compute it if the answer is real
        k2 = (1 - ro[i] ** 2 - bo ** 2 + 2 * bo * ro[i]) / (4 * bo * ro[i])
        if np.sin(kappa / 2) ** 2 <= k2:

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

    assert np.allclose(dFdro[1:-1], dFdro_[1:-1], equal_nan=True, atol=1e-6)
    assert np.allclose(dEdro[1:-1], dEdro_[1:-1], equal_nan=True, atol=1e-6)
    assert np.allclose(dPIpdro[1:-1], dPIpdro_[1:-1], equal_nan=True, atol=1e-4)


@pytest.mark.parametrize("bo,ro", [[0.5, 0.2], [0.95, 0.2]])
def test_ellip_deriv_kappa(bo, ro):

    # Evaluate over full range
    kappa = np.linspace(0, 4 * np.pi, 100)
    eps = 1e-8

    # Compute
    dFdkappa = np.zeros_like(kappa) * np.nan
    dEdkappa = np.zeros_like(kappa) * np.nan
    dPIpdkappa = np.zeros_like(kappa) * np.nan
    dFdkappa_ = np.zeros_like(kappa) * np.nan
    dEdkappa_ = np.zeros_like(kappa) * np.nan
    dPIpdkappa_ = np.zeros_like(kappa) * np.nan
    for i in range(len(kappa)):

        # Only compute it if the answer is real
        k2 = (1 - ro ** 2 - bo ** 2 + 2 * bo * ro) / (4 * bo * ro)
        if np.sin(kappa[i] / 2) ** 2 <= k2:

            # Analytic deriv
            (
                (_, (_, _, _, dFdkappa[i])),
                (_, (_, _, _, dEdkappa[i])),
                (_, (_, _, _, dPIpdkappa[i])),
            ) = c.ellip(bo, ro, [0.0, kappa[i]])

            # Numerical deriv
            ((F1, _), (E1, _), (PIp1, _),) = c.ellip(bo, ro, [0.0, kappa[i] - eps])
            ((F2, _), (E2, _), (PIp2, _),) = c.ellip(bo, ro, [0.0, kappa[i] + eps])
            dFdkappa_[i] = (F2 - F1) / (2 * eps)
            dEdkappa_[i] = (E2 - E1) / (2 * eps)
            dPIpdkappa_[i] = (PIp2 - PIp1) / (2 * eps)

    assert np.allclose(dFdkappa[1:-1], dFdkappa_[1:-1], equal_nan=True, atol=1e-6)
    assert np.allclose(dEdkappa[1:-1], dEdkappa_[1:-1], equal_nan=True, atol=1e-6)
    assert np.allclose(dPIpdkappa[1:-1], dPIpdkappa_[1:-1], equal_nan=True, atol=1e-4)


def test_quad():
    assert np.allclose(np.exp(1) - np.exp(0), c.quad(np.exp, 0.0, 1.0))


@pytest.mark.parametrize("bo,ro", [[0.5, 0.2], [0.9, 0.2]])
def test_P2_numerical(bo, ro):

    # NOTE: See comment in `iellip.h` about how our expression for
    # P2 isn't valid when successive terms in kappa span either side
    # of the discontinuities at pi and 3 pi. Let's just not worry about
    # that here.
    kappa = np.linspace(0, np.pi, 100, endpoint=False)

    # Helper variables
    k2 = (1 - ro ** 2 - bo ** 2 + 2 * bo * ro) / (4 * bo * ro)

    # Compute
    P2 = np.zeros_like(kappa) * np.nan
    P2_ = np.zeros_like(kappa) * np.nan
    dP2dbo = np.zeros_like(kappa) * np.nan
    dP2dro = np.zeros_like(kappa) * np.nan
    dP2dkappa = np.zeros_like(kappa) * np.nan
    dP2dbo_ = np.zeros_like(kappa) * np.nan
    dP2dro_ = np.zeros_like(kappa) * np.nan
    dP2dkappa_ = np.zeros_like(kappa) * np.nan
    for k in range(len(kappa)):

        # Only compute it if the answer is real
        if np.sin(kappa[k] / 2) ** 2 <= k2:
            P2[k], (dP2dbo[k], dP2dro[k], _, dP2dkappa[k]) = c.P2_numerical(
                bo, ro, np.array([0, kappa[k]])
            )
            P2_[k], (dP2dbo_[k], dP2dro_[k], _, dP2dkappa_[k]) = c.P2(
                bo, ro, np.array([0, kappa[k]])
            )

    assert np.allclose(P2, P2_, equal_nan=True)
    assert np.allclose(dP2dbo, dP2dbo_, equal_nan=True)
    assert np.allclose(dP2dro, dP2dro_, equal_nan=True)
    assert np.allclose(dP2dkappa, dP2dkappa_, equal_nan=True)


@pytest.mark.parametrize("N,bo,ro", [[20, 0.5, 0.2], [20, 0.9, 0.2]])
def test_J_numerical(N, bo, ro):

    # Strictly testing the derivatives here

    # Sample values
    kappa = np.array([0.3, 1.5])
    eps = 1e-8

    # Analytic
    _, (dJdbo, dJdro, _, dJdkappa) = c.J_numerical(N, bo, ro, kappa)

    # Numerical
    Q1, _ = c.J_numerical(N, bo, ro, kappa - np.array([0, eps]))
    Q2, _ = c.J_numerical(N, bo, ro, kappa + np.array([0, eps]))
    dJdkappa_num = (Q2 - Q1) / (2 * eps)

    Q1, _ = c.J_numerical(N, bo - eps, ro, kappa)
    Q2, _ = c.J_numerical(N, bo + eps, ro, kappa)
    dJdbo_num = (Q2 - Q1) / (2 * eps)
    assert np.allclose(dJdbo, dJdbo_num)

    Q1, _ = c.J_numerical(N, bo, ro - eps, kappa)
    Q2, _ = c.J_numerical(N, bo, ro + eps, kappa)
    dJdro_num = (Q2 - Q1) / (2 * eps)
    assert np.allclose(dJdro, dJdro_num)


def test_hyp2f1():
    # This expression is only used in the range z = [-0.5, 0.5]
    nmax = 20
    z = np.linspace(-0.5, 0.5, 1000)
    F = np.zeros_like(z)
    dFdz = np.zeros_like(z)
    for i in range(len(z)):
        F[i], dFdz[i] = c.hyp2f1(-0.5, nmax + 1, nmax + 2, z[i])
    assert np.allclose(F, hyp2f1(-0.5, nmax + 1, nmax + 2, z))
    assert np.allclose(
        dFdz, np.gradient(F, edge_order=2) / np.gradient(z, edge_order=2), atol=1e-6
    )


if __name__ == "__main__":
    test_hyp2f1()
