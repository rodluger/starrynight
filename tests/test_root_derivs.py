import numpy as np
from starrynight.geometry import get_roots
from starrynight.utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm


def roots(b, theta, bo, ro):
    # Get the roots using the starry function
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    x, dxdb, dxdtheta, dxdbo, dxdro = get_roots(b, theta, costheta, sintheta, bo, ro)

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


def todo_root_derivs(plot=False):
    """Test the derivatives of the roots of the quartic equation."""
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
