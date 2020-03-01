import numpy as np
from starrynight.special import J, pal, EllipF, EllipE, EllipJ, el2, rj
from starrynight.primitive import compute_Q
from starrynight.geometry import get_roots
import matplotlib.pyplot as plt
from tqdm import tqdm

# DEBUG
if __name__ == "__main__":

    from mpmath import ellipf, ellipe, ellippi

    phi = 1.3
    bo = 0.95
    ro = 0.1

    # What starry computes anyways
    k2 = (1 - ro ** 2 - bo ** 2 + 2 * bo * ro) / (4 * bo * ro)
    k = np.sqrt(k2)
    kinv = 1 / k
    k2inv = 1 / k2
    kc2 = 1 - k2
    tanphi = np.tan([phi])
    if k2 < 1:
        F = EllipF(tanphi, k2)[0] * k
        E = kinv * (EllipE(tanphi, k2)[0] - kc2 * kinv * F)
    else:
        F = EllipF(tanphi, k2inv)[0]
        E = EllipE(tanphi, k2inv)[0]
    p = (ro * ro + bo * bo + 2 * ro * bo * np.cos(phi)) / (
        ro * ro + bo * bo - 2 * ro * bo
    )
    cx = np.cos(phi / 2)
    sx = np.sin(phi / 2)
    w = 1 - cx ** 2 / k2
    RJ = (np.cos(phi) + 1) * cx * rj(w, sx * sx, 1.0, p)

    #

    # Deriv with respect to n
    n = -4 * bo * ro / (ro - bo) ** 2
    phip = (phi - np.pi) / 2
    kp = np.sqrt(1 / k2)
    kp2 = 1 / k2
    tanphi_ = np.array([-cx / (k * np.sqrt(w))])
    F_ = k * EllipF(tanphi_, k2)[0]  # = ellipf(phip, kp2)
    E_ = kinv * (EllipE(tanphi_, k2)[0] - kc2 * kinv * F_)  # = ellipe(phip, kp2)
    PI_ = -n * RJ / 6 + F_  # = ellippi(n, phip, kp2)
    s2 = np.sin(phip) ** 2
    s_2 = np.sin(2 * phip)
    dPIdn = (
        1
        / (2 * (kp2 - n) * (n - 1))
        * (
            E_
            + (kp2 - n) / n * F_
            + (n * n - kp2) / n * PI_
            - n * (1 - kp2 * s2) ** 0.5 * s_2 / (2 * (1 - n * s2))
        )
    )
    eps = 1e-8
    dPIdn_num = float(
        ellippi(n + eps, phip, kp2).real - ellippi(n - eps, phip, kp2).real
    ) / (2 * eps)
    print(dPIdn, dPIdn_num)

    # Deriv with respect to k2
    m = kp2
    dPIdm = (
        1
        / (2 * (n - m))
        * (1 / (m - 1) * E_ + PI_ - m * s_2 / (2 * (m - 1) * np.sqrt(1 - m * s2)))
    )
    eps = 1e-8
    dPIdm_num = float(
        ellippi(n, phip, kp2 + eps).real - ellippi(n, phip, kp2 - eps).real
    ) / (2 * eps)
    print(dPIdm, dPIdm_num)

    # Deriv with respect to phip
    dPIdphip = (1 - m * s2) ** -0.5 * (1 - n * s2) ** -1
    eps = 1e-8
    dPIdphip_num = float(
        ellippi(n, phip + eps, kp2).real - ellippi(n, phip - eps, kp2).real
    ) / (2 * eps)
    print(dPIdphip, dPIdphip_num)

    quit()


def test_E():
    eps = 1e-8
    k2 = 0.75
    tanphi = np.array([0.75])

    # Analytic
    E, (dEdtanphi, dEdk2) = EllipE(tanphi, k2, gradient=True)

    # Numerical
    Q1 = EllipE(tanphi - eps, k2)
    Q2 = EllipE(tanphi + eps, k2)
    dEdtanphi_num = (Q2 - Q1) / (2 * eps)
    assert np.allclose(dEdtanphi, dEdtanphi_num)

    Q1 = EllipE(tanphi, k2 - eps)
    Q2 = EllipE(tanphi, k2 + eps)
    dEdk2_num = (Q2 - Q1) / (2 * eps)
    assert np.allclose(dEdk2, dEdk2_num)


def test_F():
    eps = 1e-8
    k2 = 0.75
    tanphi = np.array([0.75])

    # Analytic
    F, (dFdtanphi, dFdk2) = EllipF(tanphi, k2, gradient=True)

    # Numerical
    Q1 = EllipF(tanphi - eps, k2)
    Q2 = EllipF(tanphi + eps, k2)
    dFdtanphi_num = (Q2 - Q1) / (2 * eps)
    assert np.allclose(dFdtanphi, dFdtanphi_num)

    Q1 = EllipF(tanphi, k2 - eps)
    Q2 = EllipF(tanphi, k2 + eps)
    dFdk2_num = (Q2 - Q1) / (2 * eps)
    assert np.allclose(dFdk2, dFdk2_num)


def test_Q():
    ydeg = 3
    eps = 1e-8
    lam = np.array([0.3, 1.5])

    # Analytic
    Q, dQdlam = compute_Q(ydeg, lam, gradient=True)

    # Numerical
    dQdlam_num = np.zeros_like(dQdlam)
    Q1 = compute_Q(ydeg, lam - np.array([eps, 0]))
    Q2 = compute_Q(ydeg, lam + np.array([eps, 0]))
    dQdlam_num[:, 0] = (Q2 - Q1) / (2 * eps)
    Q1 = compute_Q(ydeg, lam - np.array([0, eps]))
    Q2 = compute_Q(ydeg, lam + np.array([0, eps]))
    dQdlam_num[:, 1] = (Q2 - Q1) / (2 * eps)

    assert np.allclose(dQdlam, dQdlam_num)


def test_pal():
    eps = 1e-8
    kappa = np.array([0.3, 1.5])
    bo = 0.25
    ro = 0.1

    # Analytic
    val, (dpaldbo, dpaldro, dpaldkappa) = pal(bo, ro, kappa, gradient=True)

    # Numerical
    dpaldkappa_num = np.zeros_like(dpaldkappa)
    Q1 = pal(bo, ro, kappa - np.array([eps, 0]))
    Q2 = pal(bo, ro, kappa + np.array([eps, 0]))
    dpaldkappa_num[:, 0] = (Q2 - Q1) / (2 * eps)
    Q1 = pal(bo, ro, kappa - np.array([0, eps]))
    Q2 = pal(bo, ro, kappa + np.array([0, eps]))
    dpaldkappa_num[:, 1] = (Q2 - Q1) / (2 * eps)
    assert np.allclose(dpaldkappa, dpaldkappa_num)

    Q1 = pal(bo - eps, ro, kappa)
    Q2 = pal(bo + eps, ro, kappa)
    dpaldbo_num = (Q2 - Q1) / (2 * eps)
    assert np.allclose(dpaldbo, dpaldbo_num)

    Q1 = pal(bo, ro - eps, kappa)
    Q2 = pal(bo, ro + eps, kappa)
    dpaldro_num = (Q2 - Q1) / (2 * eps)
    assert np.allclose(dpaldro, dpaldro_num)


def test_J():
    N = 5
    eps = 1e-8
    kappa = np.array([0.3, 1.5])
    k2 = 0.5

    # Analytic
    val, (dJdk2, dJdkappa) = J(N, k2, kappa, gradient=True)

    # Numerical
    dJdkappa_num = np.zeros_like(dJdkappa)
    Q1 = J(N, k2, kappa - np.array([eps, 0]))
    Q2 = J(N, k2, kappa + np.array([eps, 0]))
    dJdkappa_num[:, 0] = (Q2 - Q1) / (2 * eps)
    Q1 = J(N, k2, kappa - np.array([0, eps]))
    Q2 = J(N, k2, kappa + np.array([0, eps]))
    dJdkappa_num[:, 1] = (Q2 - Q1) / (2 * eps)
    assert np.allclose(dJdkappa, dJdkappa_num)

    Q1 = J(N, k2 - eps, kappa)
    Q2 = J(N, k2 + eps, kappa)
    dJdk2_num = (Q2 - Q1) / (2 * eps)
    assert np.allclose(dJdk2, dJdk2_num)


def test_roots(plot=False):
    """Test the derivatives of the roots of the quartic equation."""

    def roots(b, theta, bo, ro):
        """Wrapper that sorts and pads the roots."""
        # Get the roots using the starry function
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        x, (dxdb, dxdtheta, dxdbo, dxdro) = get_roots(
            b, theta, costheta, sintheta, bo, ro, gradient=True
        )

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

    # Check that it's small. The numerical gradient is not
    # very precise, so we're pretty lax here.
    assert np.all(log_diff_b[np.isfinite(log_diff_b)] < -3)
    assert np.all(log_diff_theta[np.isfinite(log_diff_theta)] < -3)
    assert np.all(log_diff_bo[np.isfinite(log_diff_bo)] < -3)
    assert np.all(log_diff_ro[np.isfinite(log_diff_ro)] < -3)

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
