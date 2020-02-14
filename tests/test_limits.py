"""
KNOWN SINGULARITIES
===================

TODO: root finding

    - bo = 0
    - bo = 0 and theta = 90 (only one root)
    - bo <~ 0.1 and theta = 90 (root finding fails I think)

    - bo = 1 - ro
    - bo = 1 + ro

"""

from numerical import Numerical
from starrynight import StarryNight
from starrynight.linear import pal
from starrynight.special import dE, dF
from starrynight.primitive import compute_J, compute_H
import numpy as np
import pytest
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad


def test_T_poles(tol=1e-10):
    """Test cases near the poles for theta."""
    # Settings
    ydeg = 10
    b = 0.25
    ro = 0.25
    n = 5

    # Compare
    S = StarryNight(ydeg)
    N = Numerical(ydeg)
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0]).reshape(-1, 1) * np.pi
    dx = np.concatenate(
        (-np.logspace(-15, -5, n)[::-1], [0], np.logspace(-15, -5, n))
    ).reshape(1, -1)
    theta = (x + dx).reshape(-1)
    bo = 0.35 * np.ones_like(theta)
    bo[np.abs(theta - np.pi) < 0.1] *= -1
    err = np.zeros_like(theta)
    for i in range(len(theta)):
        N.precompute(b, theta[i], bo[i], ro)
        S.precompute(b, theta[i], bo[i], ro)
        err[i] = np.max(np.abs(N.T - S.T))
    assert np.all(err < tol)


def test_Q_high_l(tol=1e-15):
    """Test Q for high degree."""
    # Settings
    ydeg = 20
    b = 0.25
    theta = np.pi / 3
    bo = 0.5
    ro = 0.75

    # Compare
    S = StarryNight(ydeg)
    N = Numerical(ydeg)
    N.precompute(b, theta, bo, ro)
    S.precompute(b, theta, bo, ro)
    assert np.all(np.abs(S.Q - N.Q) < tol)


"""
def test_grazing():

    b = 0.15
    theta = 1.57
    bo = 0.25
    ro = 0.75

    # Perturb the impact parameter about the given point
    dbo = np.logspace(-15, -1, 300)
    bo = np.abs(np.concatenate([[(1 - ro)], (1 - ro) + dbo]))

    # Compare
    S = StarryNight(1)
    P0 = np.zeros_like(bo)
    for i in range(len(bo)):
        S.precompute(b, theta, bo[i], ro)
        if hasattr(S, "P"):
            P0[i] = S.P[2]
            del S.P

    # DEBUG
    plt.plot(bo - (1 - ro), P0)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    assert False
"""


@pytest.mark.parametrize(
    "b,theta,bo,ro,tol,sign",
    [
        [0.15, 0, 0, 0.4, 1e-10, "both"],  # bo ~ 0
        [0.15, 0, 0.25, 0.25, 2e-6, "both"],  # bo ~ ro
        [0.15, 1.57, 0.25, 1.25, 1e-8, "pos"],  # bo ~ ro - 1
        [0.15, 1.57, 0.25, 0.75, 1e-7, "both"],  # bo ~ 1 - ro
    ],
)
def test_P2_edges(b, theta, bo, ro, tol, sign, minval=-15, maxval=-1, npts=100):
    """Test the Pal (2012) term near the known singularities.
    
    Those are:

        - ro = bo
        - bo = ro + 1
        - bo = ro - 1

    TODO: Come up with better expressions in these limits, then 
          decrease the test tolerance for these cases.

    """
    # Perturb the impact parameter about the given point
    dbo = np.logspace(minval, maxval, npts // 2)
    if sign == "neg":
        bo = np.abs(np.concatenate([bo - dbo[::-1], [bo]]))
    elif sign == "pos":
        bo = np.abs(np.concatenate([[bo], bo + dbo]))
    else:
        bo = np.abs(np.concatenate([bo - dbo[::-1], [bo], bo + dbo]))

    # Compare
    S = StarryNight(1)
    N = Numerical(1)
    err = np.zeros_like(bo)
    tmp = np.zeros_like(bo)
    for i in range(len(bo)):
        N.precompute(b, theta, bo[i], ro)
        S.precompute(b, theta, bo[i], ro)
        if hasattr(S, "P"):
            # Sometimes we might be so close to an edge case
            # that the root finder doesn't find the intersection.
            # This is fine; the answer will still be approximately correct.
            err[i] = np.abs(N.P[2] - S.P[2])
            tmp[i] = N.P[0]
            del N.P
            del S.P
        if np.isnan(err[i]):
            err[i] = 1.0
    assert np.all(err < tol), "{}".format(np.max(err))


def test_J_high_l(tol=1e-15):
    """Test the J helper integral when k is large and small."""
    # Settings
    ydeg = 20
    tol = 1e-12
    kappa = np.array([np.pi / 2, np.pi / 2 + 1])
    k2 = np.logspace(-2, 2, 100)

    # Compare to numerical integration
    erel = np.zeros_like(k2)
    for i in tqdm(range(len(k2))):
        km2 = 1 / k2[i]
        x = 0.5 * kappa
        s1 = np.sin(x)
        s2 = s1 ** 2
        c1 = np.cos(x)
        q2 = 1 - np.minimum(1.0, s2 / k2[i])
        dF_val = dF(x, km2)
        dE_val = dE(x, km2)
        J = compute_J(ydeg + 1, k2[i], km2, kappa, s1, s2, c1, q2, dE_val, dF_val)
        func = (
            lambda phi, v: np.sin(phi) ** (2 * v)
            * (max(0, 1 - km2 * np.sin(phi) ** 2)) ** 1.5
        )
        J_ = np.array(
            [
                quad(func, x[0], x[1], args=(v,), epsabs=1e-14, epsrel=1e-14)[0]
                for v in range(len(J))
            ]
        )
        erel[i] = max(1e-16, np.max(np.abs(J - J_)))
    assert np.all(erel < tol)


def test_P_high_l(tol=1e-13, s2tol=1e-8):
    """Test P for high degree."""
    # Settings
    ydeg = 20
    b = 0.25
    theta = np.pi / 3
    bo = 0.5
    ro = 0.75

    # Compare
    S = StarryNight(ydeg)
    N = Numerical(ydeg)
    N.precompute(b, theta, bo, ro)
    S.precompute(b, theta, bo, ro)

    # The s2 term has higher errors
    assert np.all(np.abs(S.P[:2] - N.P[:2]) < tol)
    assert np.abs(S.P[2] - N.P[2]) < s2tol
    assert np.all(np.abs(S.P[3:] - N.P[3:]) < tol)


@pytest.mark.parametrize(
    "b,theta,bo,ro,tol,sign",
    [
        [0.15, 0, 0, 0.4, 1e-14, "both"],  # bo ~ 0
        [0.15, 0, 0.25, 0.25, 1e-15, "both"],  # bo ~ ro
        [0.15, 1.57, 0.25, 1.25, 1e-10, "pos"],  # bo ~ ro - 1
        [0.15, 1.57, 0.25, 0.75, 1e-8, "both"],  # bo ~ 1 - ro
    ],
)
def test_P_edges(b, theta, bo, ro, tol, sign, minval=-15, maxval=-1, npts=100):
    """Test the P integral near the known singularities.
    
    Those are:

        - bo = 0
        - ro = bo
        - bo = ro + 1
        - bo = ro - 1

    We're doing this at ydeg = 10 for speed (the numerical integration is slow)
    but even at ydeg = 20 the results still hold, since the algorithm is stable!

    """
    # Perturb the impact parameter about the given point
    dbo = np.logspace(minval, maxval, npts // 2)
    if sign == "neg":
        bo = np.abs(np.concatenate([bo - dbo[::-1], [bo]]))
    elif sign == "pos":
        bo = np.abs(np.concatenate([[bo], bo + dbo]))
    else:
        bo = np.abs(np.concatenate([bo - dbo[::-1], [bo], bo + dbo]))

    # Compare
    ydeg = 10
    S = StarryNight(ydeg)
    N = Numerical(ydeg)
    err = np.zeros_like(bo)
    for i in tqdm(range(len(bo))):
        N.precompute(b, theta, bo[i], ro)
        S.precompute(b, theta, bo[i], ro)
        if hasattr(S, "P"):
            # Sometimes we might be so close to an edge case
            # that the root finder doesn't find the intersection.
            # This is fine; the answer will still be approximately correct.
            # Note that we're ignoring P[2], which is tested above
            err[i] = max(
                np.max(np.abs(N.P[:2] - S.P[:2])), np.max(np.abs(N.P[3:] - S.P[3:]))
            )
            del N.P
            del S.P
        if np.isnan(err[i]):
            err[i] = 1.0

    assert np.all(err < tol), "{}".format(np.max(err))
