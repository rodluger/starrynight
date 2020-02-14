from utils import compare
from cases import CASE
from numerical import Numerical
from starrynight import StarryNight
import numpy as np
import pytest

# Params
ydeg = 5
S = StarryNight(ydeg)
N = Numerical(ydeg)
args = CASE[1] + CASE[2] + CASE[3] + CASE[4] + CASE[5]


@pytest.mark.parametrize(
    "b,theta,bo,ro", args,
)
def test_P(b, theta, bo, ro):
    N.precompute(b, theta, bo, ro)
    S.precompute(b, theta, bo, ro)
    compare(N.P, S.P)
    del N.P
    del S.P


@pytest.mark.parametrize(
    "b,theta,bo,ro", args,
)
def test_T(b, theta, bo, ro):
    N.precompute(b, theta, bo, ro)
    S.precompute(b, theta, bo, ro)
    compare(N.T, S.T)
    del N.T
    del S.T


@pytest.mark.parametrize(
    "b,theta,bo,ro", args,
)
def test_Q(b, theta, bo, ro):
    N.precompute(b, theta, bo, ro)
    S.precompute(b, theta, bo, ro)
    compare(N.Q, S.Q)
    del N.Q
    del S.Q
