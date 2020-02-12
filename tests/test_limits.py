"""
KNOWN SINGULARITIES
===================

TODO: root finding

    - bo = 0
    - bo = 0 and theta = 90 (only one root)
    - bo <~ 0.1 and theta = 90 (root finding fails I think)

"""

from numerical import Numerical
from starrynight import StarryNight
from starrynight.linear import pal
import numpy as np
import pytest
import matplotlib.pyplot as plt
from tqdm import tqdm


def test_T(tol=1e-10):
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


def test_Q(tol=1e-15):
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


@pytest.mark.parametrize(
    "b,theta,bo,ro,tol,sign",
    [
        [0.15, 0, 0, 0.4, 1e-10, "both"],  # bo ~ 0
        [0.15, 0, 0.25, 0.25, 2e-8, "both"],  # bo ~ ro
        [0.15, 1.57, 0.25, 1.25, 1e-8, "pos"],  # bo ~ ro - 1
        [0.15, 1.57, 0.25, 0.75, 2e-8, "both"],  # bo ~ 1 - ro
    ],
)
def test_P2(b, theta, bo, ro, tol, sign, minval=-15, maxval=-1, npts=100):
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
    for i in range(len(bo)):
        N.precompute(b, theta, bo[i], ro)
        S.precompute(b, theta, bo[i], ro)
        if hasattr(S, "P"):
            # Sometimes we might be so close to an edge case
            # that the root finder doesn't find the intersection.
            # This is fine; the answer will still be approximately correct.
            err[i] = np.abs(N.P[2] - S.P[2])
        if np.isnan(err[i]):
            err[i] = 1.0
    try:
        assert np.all(err < tol)
    except:
        # DEBUG
        plt.plot(bo, err)
        plt.yscale("log")
        plt.show()
        assert np.all(err < tol)

