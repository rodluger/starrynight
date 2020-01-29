from starrynight import Numerical, Analytic
from starrynight.geometry import get_angles
from starrynight.linear import pal
from scipy.integrate import quad
import numpy as np
import pytest


ydeg = 5
A = Analytic(y=np.zeros((ydeg + 1) ** 2), tol=1e-7)
N = Numerical(y=np.zeros((ydeg + 1) ** 2), tol=1e-7)


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
    N.b, N.theta, N.bo, N.ro = b, theta, bo, ro
    A.b, A.theta, A.bo, A.ro = b, theta, bo, ro
    phi, _, _, _ = get_angles(b, theta, bo, ro)
    phi1, phi2 = phi[:2]
    for l in range(ydeg + 2):
        for m in range(-l, l + 1):
            P1 = A.P(l, m, phi1, phi2)
            P2 = N.P(l, m, phi1, phi2)
            assert np.allclose(P1, P2)


@pytest.mark.parametrize(
    "b,theta,bo,ro", args,
)
def test_linear(b, theta, bo, ro):
    N.b, N.theta, N.bo, N.ro = b, theta, bo, ro
    A.b, A.theta, A.bo, A.ro = b, theta, bo, ro
    phi, _, _, _ = get_angles(b, theta, bo, ro)
    phi1, phi2 = phi[:2]
    P1 = N.P(1, 0, phi1, phi2)
    P2 = pal(bo, ro, phi1 - np.pi / 2, phi2 - np.pi / 2)
    assert np.allclose(P1, P2)
