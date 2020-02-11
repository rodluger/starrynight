from starrynight import StarryNight
from starrynight.numerical import Numerical
from starrynight.geometry import get_angles
from starrynight.linear import pal
from scipy.integrate import quad
import numpy as np
import pytest


ydeg = 5
S = StarryNight(ydeg, tol=1e-7)
N = Numerical(ydeg, tol=1e-7)


args = [
    [0.4, np.pi / 3, 0.5, 0.7],
    [0.4, 2 * np.pi - np.pi / 3, 0.5, 0.7],
    [0.4, np.pi / 2, 0.5, 0.7],
    [0.4, np.pi / 2, 1.0, 0.2],
    [0.00001, np.pi / 2, 0.5, 0.7],
    [0, np.pi / 2, 0.5, 0.7],
    [0.4, -np.pi / 2, 0.5, 0.7],
    [-0.4, np.pi / 3, 0.5, 0.7],
    [-0.4, 2 * np.pi - np.pi / 3, 0.5, 0.7],
    [-0.4, np.pi / 2, 0.5, 0.7],
    [0.4, np.pi / 6, 0.3, 0.3],
    [0.4, np.pi + np.pi / 6, 0.1, 0.6],
    [0.4, np.pi + np.pi / 3, 0.1, 0.6],
    [0.4, np.pi / 6, 0.6, 0.5],
    [0.4, -np.pi / 6, 0.6, 0.5],
    [0.4, 0.1, 2.2, 2.0],
    [0.4, -0.1, 2.2, 2.0],
    [0.4, np.pi + np.pi / 6, 0.3, 0.8],
    [0.75, np.pi + 0.1, 4.5, 5.0],
    [-0.95, 0.0, 2.0, 2.5],
    [-0.1, np.pi / 6, 0.6, 0.75],
    [-0.5, np.pi, 0.8, 0.5],
    [-0.1, 0.0, 0.5, 1.0],
    [0.5488316824842527, 4.03591586925189, 0.34988513192814663, 0.7753986686719786,],
    [
        0.5488316824842527,
        2 * np.pi - 4.03591586925189,
        0.34988513192814663,
        0.7753986686719786,
    ],
    [
        -0.5488316824842527,
        4.03591586925189 - np.pi,
        0.34988513192814663,
        0.7753986686719786,
    ],
    [
        -0.5488316824842527,
        2 * np.pi - (4.03591586925189 - np.pi),
        0.34988513192814663,
        0.7753986686719786,
    ],
    [0.5, np.pi, 0.99, 1.5],
    [-0.5, 0.0, 0.99, 1.5],
    [0.5, np.pi, 1.0, 1.5],
    [0.5, 2 * np.pi - np.pi / 4, 0.4, 0.4],
    [0.5, 2 * np.pi - np.pi / 4, 0.3, 0.3],
]


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
