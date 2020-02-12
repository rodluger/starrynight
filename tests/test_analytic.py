from cases import CASE
from numerical import Numerical
from starrynight import StarryNight
import numpy as np
import pytest


ydeg = 5
S = StarryNight(ydeg, tol=1e-7)
N = Numerical(ydeg, tol=1e-7)


args = CASE[1] + CASE[2] + CASE[3] + CASE[4] + CASE[5]


@pytest.mark.parametrize(
    "b,theta,bo,ro", args,
)
def test_P(b, theta, bo, ro):
    N.precompute(b, theta, bo, ro)
    S.precompute(b, theta, bo, ro)
    assert np.allclose(N.P, S.P)


@pytest.mark.parametrize(
    "b,theta,bo,ro", args,
)
def test_T(b, theta, bo, ro):
    N.precompute(b, theta, bo, ro)
    S.precompute(b, theta, bo, ro)
    assert np.allclose(N.T, S.T)


@pytest.mark.parametrize(
    "b,theta,bo,ro", args,
)
def test_Q(b, theta, bo, ro):
    N.precompute(b, theta, bo, ro)
    S.precompute(b, theta, bo, ro)
    assert np.allclose(N.Q, S.Q)
