from starrynight import c
from mpmath import ellipf, ellipe, ellippi
from scipy.special import hyp2f1
from scipy.integrate import quad
import numpy as np
import pytest
from tqdm import tqdm
import matplotlib.pyplot as plt


def Glm(l, m):
    """A Green's basis antiderivative."""
    mu = l - m
    nu = l + m

    z = lambda x, y: np.maximum(1e-12, np.sqrt(np.abs(1 - x ** 2 - y ** 2)))

    if nu % 2 == 0:

        G = [lambda x, y: 0, lambda x, y: x ** (0.5 * (mu + 2)) * y ** (0.5 * nu)]

    elif (l == 1) and (m == 0):

        def G0(x, y):
            z_ = z(x, y)
            if z_ > 1 - 1e-8:
                return -0.5 * y
            else:
                return (1 - z_ ** 3) / (3 * (1 - z_ ** 2)) * (-y)

        def G1(x, y):
            z_ = z(x, y)
            if z_ > 1 - 1e-8:
                return 0.5 * x
            else:
                return (1 - z_ ** 3) / (3 * (1 - z_ ** 2)) * x

        G = [G0, G1]

    elif (mu == 1) and (l % 2 == 0):

        G = [lambda x, y: x ** (l - 2) * z(x, y) ** 3, lambda x, y: 0]

    elif (mu == 1) and (l % 2 != 0):

        G = [lambda x, y: x ** (l - 3) * y * z(x, y) ** 3, lambda x, y: 0]

    else:

        G = [
            lambda x, y: 0,
            lambda x, y: x ** (0.5 * (mu - 3)) * y ** (0.5 * (nu - 1)) * z(x, y) ** 3,
        ]

    return G


def Primitive(ydeg, x, y, dx, dy, theta1, theta2, epsabs=1e-12, epsrel=1e-12):
    """A general vectorized primitive integral computed numerically."""
    res = np.zeros((ydeg + 1) ** 2)
    n = 0
    for l in range(ydeg + 1):
        for m in range(-l, l + 1):
            G = Glm(l, m)
            func = lambda theta: G[0](x(theta), y(theta)) * dx(theta) + G[1](
                x(theta), y(theta)
            ) * dy(theta)
            res[n], _ = quad(func, theta1, theta2, epsabs=epsabs, epsrel=epsrel)
            n += 1
    return res


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
    assert np.allclose(dJdkappa, dJdkappa_num)

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


def test_J():

    nmax = 30
    bo = 0.5
    ro = 0.2
    kappa = np.array([0.0, 2.0])

    # Analytic
    J = c.J(nmax, bo, ro, kappa)

    # Numerical
    J_num = np.zeros(nmax + 1)
    for n in range(nmax + 1):
        J_num[n], _ = c.J_numerical(n, bo, ro, kappa)

    assert np.allclose(J, J_num)


def test_H():

    uvmax = 5
    xi = np.array([0.0, 2.0])
    eps = 1e-8

    # Analytic
    H, dHdxi0, dHdxi1 = c.H(uvmax, xi)

    # Numerical
    H_num = np.zeros((uvmax + 1, uvmax + 1))
    dH_numdxi0 = np.zeros((uvmax + 1, uvmax + 1))
    dH_numdxi1 = np.zeros((uvmax + 1, uvmax + 1))
    for u in range(uvmax + 1):
        for v in range(uvmax + 1 - u):
            H_num[u, v] = quad(lambda x: np.cos(x) ** u * np.sin(x) ** v, xi[0], xi[1])[
                0
            ]

            f1 = quad(lambda x: np.cos(x) ** u * np.sin(x) ** v, xi[0] - eps, xi[1])[0]
            f2 = quad(lambda x: np.cos(x) ** u * np.sin(x) ** v, xi[0] + eps, xi[1])[0]
            dH_numdxi0[u, v] = (f2 - f1) / (2 * eps)

            f1 = quad(lambda x: np.cos(x) ** u * np.sin(x) ** v, xi[0], xi[1] - eps)[0]
            f2 = quad(lambda x: np.cos(x) ** u * np.sin(x) ** v, xi[0], xi[1] + eps)[0]
            dH_numdxi1[u, v] = (f2 - f1) / (2 * eps)

    assert np.allclose(H, H_num)
    assert np.allclose(dHdxi0, dH_numdxi0, atol=1e-6)
    assert np.allclose(dHdxi1, dH_numdxi1, atol=1e-6)


@pytest.mark.parametrize(
    "b,theta", [[0.25, 1.0], [-0.25, 1.0], [0.25, np.pi / 2], [0.25, 0.0]]
)
def test_T(b, theta):

    ydeg = 5
    xi = np.array([1.0, 3.0])

    # Analytic
    T, _, _, _, _ = c.T(ydeg, b, theta, xi)

    # Numerically
    x = lambda xi: np.cos(theta) * np.cos(xi) - b * np.sin(theta) * np.sin(xi)
    y = lambda xi: np.sin(theta) * np.cos(xi) + b * np.cos(theta) * np.sin(xi)
    dx = lambda xi: -np.cos(theta) * np.sin(xi) - b * np.sin(theta) * np.cos(xi)
    dy = lambda xi: -np.sin(theta) * np.sin(xi) + b * np.cos(theta) * np.cos(xi)
    T_ = Primitive(ydeg, x, y, dx, dy, xi[0], xi[1])

    assert np.allclose(T, T_)


def test_Q():

    ydeg = 5
    lam = np.array([1.0, 3.0])

    # Analytic
    Q, _, _ = c.Q(ydeg, lam)

    # Numerically
    x = lambda lam: np.cos(lam)
    y = lambda lam: np.sin(lam)
    dx = lambda lam: -np.sin(lam)
    dy = lambda lam: np.cos(lam)
    Q_ = Primitive(ydeg, x, y, dx, dy, lam[0], lam[1])

    assert np.allclose(Q, Q_)


@pytest.mark.parametrize("bo,ro", [[0.5, 0.2], [0.95, 0.2]])
def test_P(bo, ro):

    ydeg = 3  # DEBUG 8
    kappa = np.array([0.0, 1.0])

    # Analytic
    P, _, _, _, _ = c.P(ydeg, bo, ro, kappa)

    # Numerically
    x = lambda phi: ro * np.cos(phi)
    y = lambda phi: bo + ro * np.sin(phi)
    dx = lambda phi: -ro * np.sin(phi)
    dy = lambda phi: ro * np.cos(phi)
    P_ = Primitive(ydeg, x, y, dx, dy, kappa[0] - np.pi / 2, kappa[1] - np.pi / 2)

    assert np.allclose(P, P_)


@pytest.mark.parametrize(
    "coeffs", [[2.0, -0.5, -7.0, 0.0, 1.0], [2.0, 1.0, -2.0, 0.0, 1.0]]
)
def test_eigen_roots(coeffs):
    r1 = np.sort(np.roots(coeffs) + 0j)
    r2 = np.sort(c.eigen_roots(coeffs))
    assert np.allclose(r1, r2)


def test_roots(plot=False):
    """Test the derivatives of the roots of the quartic equation."""

    def roots(b, theta, bo, ro):
        """Wrapper that sorts and pads the roots."""
        # Get the roots using the starry function
        x, dxdb, dxdtheta, dxdbo, dxdro = c.get_roots(b, theta, bo, ro)

        # Sort them
        i = np.argsort(x)
        x = np.array(x)[i]
        dxdb = np.array(dxdb)[i]
        dxdtheta = np.array(dxdtheta)[i]
        dxdbo = np.array(dxdbo)[i]
        dxdro = np.array(dxdro)[i]

        # Pad the arrays so they all have the same shape
        npad = 4 - len(x)
        x = np.pad(x, ((0, npad),), constant_values=np.nan)
        dxdb = np.pad(dxdb, ((0, npad),), constant_values=np.nan)
        dxdtheta = np.pad(dxdtheta, ((0, npad),), constant_values=np.nan)
        dxdbo = np.pad(dxdbo, ((0, npad),), constant_values=np.nan)
        dxdro = np.pad(dxdro, ((0, npad),), constant_values=np.nan)

        return x, dxdb, dxdtheta, dxdbo, dxdro

    # Number of data points
    ns = 1000

    # Fiducial values
    b0 = 0.25
    theta0 = np.pi / 2
    ro0 = 0.5
    bo0 = 0.25

    # Arrays
    b = np.linspace(-1, 1, ns)
    theta = np.linspace(0, 2 * np.pi, ns)
    ro = np.linspace(0, 2, ns)
    bo = np.linspace(0, 1 + ro0, ns)

    # Compute the analytic derivs
    x_b = np.zeros((ns, 4))
    x_theta = np.zeros((ns, 4))
    x_bo = np.zeros((ns, 4))
    x_ro = np.zeros((ns, 4))
    dxdb = np.zeros((ns, 4))
    dxdtheta = np.zeros((ns, 4))
    dxdbo = np.zeros((ns, 4))
    dxdro = np.zeros((ns, 4))
    for k in tqdm(range(ns)):
        x_b[k], dxdb[k], _, _, _ = roots(b[k], theta0, bo0, ro0)
        x_theta[k], _, dxdtheta[k], _, _ = roots(b0, theta[k], bo0, ro0)
        x_bo[k], _, _, dxdbo[k], _ = roots(b0, theta0, bo[k], ro0)
        x_ro[k], _, _, _, dxdro[k] = roots(b0, theta0, bo0, ro[k])

    # Compute the numerical derivs
    dxdb_num = np.transpose(
        [
            np.gradient(x_b[:, n], edge_order=2) / np.gradient(b, edge_order=2)
            for n in range(4)
        ]
    )
    dxdtheta_num = np.transpose(
        [
            np.gradient(x_theta[:, n], edge_order=2) / np.gradient(theta, edge_order=2)
            for n in range(4)
        ]
    )
    dxdbo_num = np.transpose(
        [
            np.gradient(x_bo[:, n], edge_order=2) / np.gradient(bo, edge_order=2)
            for n in range(4)
        ]
    )
    dxdro_num = np.transpose(
        [
            np.gradient(x_ro[:, n], edge_order=2) / np.gradient(ro, edge_order=2)
            for n in range(4)
        ]
    )

    # Compute the difference
    log_diff_b = np.log10(np.abs(dxdb_num - dxdb))
    log_diff_theta = np.log10(np.abs(dxdtheta_num - dxdtheta))
    log_diff_bo = np.log10(np.abs(dxdbo_num - dxdbo))
    log_diff_ro = np.log10(np.abs(dxdro_num - dxdro))

    # The derivatives wrt b and ro diverge at the extrema
    # so let's mask them. The derivative wrt theta is discontinuous
    # at theta = pi because of the order ambiguity of the roots, so
    # we mask that as well.
    log_diff_b[np.abs(dxdb_num) > 2] = np.nan
    log_diff_ro[np.abs(dxdro_num) > 2] = np.nan
    log_diff_theta[np.abs(theta - np.pi) < 0.1] = np.nan

    if plot:
        # Plot the derivatives
        fig, ax = plt.subplots(2, 2, figsize=(9, 6))
        fig.subplots_adjust(hspace=0.3)
        ax[0, 0].plot(b, dxdb_num, color="C1", lw=2)
        ax[0, 0].plot(b, dxdb, color="C0", lw=1)
        ax[0, 1].plot(theta, dxdtheta_num, color="C1", lw=2)
        ax[0, 1].plot(theta, dxdtheta, color="C0", lw=1)
        ax[1, 0].plot(bo, dxdbo_num, color="C1", lw=2)
        ax[1, 0].plot(bo, dxdbo, color="C0", lw=1)
        ax[1, 1].plot(ro, dxdro_num, color="C1", lw=2)
        ax[1, 1].plot(ro, dxdro, color="C0", lw=1)
        ax[0, 0].set_xlabel(r"$b$")
        ax[0, 1].set_xlabel(r"$\theta$")
        ax[1, 0].set_xlabel(r"$b_o$")
        ax[1, 1].set_xlabel(r"$r_o$")
        ax[0, 0].set_ylabel("derivative")
        ax[1, 0].set_ylabel("derivative")
        fig.savefig("test_quad_diff_derivs.pdf", bbox_inches="tight")

        # Plot the differences
        fig, ax = plt.subplots(2, 2, figsize=(9, 6))
        fig.subplots_adjust(hspace=0.3)
        ax[0, 0].plot(b, log_diff_b, "k-")
        ax[0, 1].plot(theta, log_diff_theta, "k-")
        ax[1, 0].plot(bo, log_diff_bo, "k-")
        ax[1, 1].plot(ro, log_diff_ro, "k-")
        ax[0, 0].set_xlabel(r"$b$")
        ax[0, 1].set_xlabel(r"$\theta$")
        ax[1, 0].set_xlabel(r"$b_o$")
        ax[1, 1].set_xlabel(r"$r_o$")
        ax[0, 0].set_ylabel("log difference")
        ax[1, 0].set_ylabel("log difference")
        fig.savefig("test_quad_diff_diff.pdf", bbox_inches="tight")

    # Check that it's small. The numerical gradient is not
    # very precise, so we're pretty lax here.
    assert np.all(log_diff_b[np.isfinite(log_diff_b)] < -3)
    assert np.all(log_diff_theta[np.isfinite(log_diff_theta)] < -3)
    assert np.all(log_diff_bo[np.isfinite(log_diff_bo)] < -3)
    assert np.all(log_diff_ro[np.isfinite(log_diff_ro)] < -3)


if __name__ == "__main__":

    # DEBUG ONLY
    import sys

    sys.path.insert(1, "../../tests")
    import cases
    import starrynight

    ydeg = 3
    S = starrynight.StarryNight(ydeg)

    for CASE in cases.CASE:

        for i, case in enumerate(tqdm(CASE)):

            # Check angles
            b, theta, bo, ro = case
            kappa, lam, xi, code = c.get_angles(b, theta, bo, ro)
            kappa_, lam_, xi_, code_ = starrynight.geometry.get_angles(
                b, theta, np.cos(theta), np.sin(theta), bo, ro
            )
            try:
                assert np.allclose(kappa, kappa_), "kappa"
                assert np.allclose(lam, lam_), "lam"
                assert np.allclose(xi, xi_), "xi"
                assert code == code_, "code"
            except:
                breakpoint()

            # Check integrals
            bs0T = np.zeros((ydeg + 2) ** 2)
            s0T, code, _, _, _, _ = c.s0T(ydeg, b, theta, bo, ro, bs0T)
            S.precompute(b, theta, bo, ro)
            if S.P is None:
                s0T_ = np.zeros((ydeg + 2) ** 2)
            else:
                s0T_ = S.P + S.Q + S.T
            try:
                assert np.allclose(s0T, s0T_)
            except:
                breakpoint()

            # Check illumination
            bI = np.zeros(((ydeg + 2) ** 2, (ydeg + 1) ** 2))
            I, _, _ = c.I(ydeg, b, theta, bI)
            I_ = S.illum()
            try:
                assert np.allclose(I, I_)
            except:
                breakpoint()
