from starrynight import Numerical, Analytic
from starrynight.geometry import get_angles
import numpy as np
import pytest


ydeg = 3
A = Analytic(y=np.zeros((ydeg + 1) ** 2), tol=1e-7)


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
]


@pytest.mark.parametrize(
    "b,theta,bo,ro", args,
)
def test_P(b, theta, bo, ro):
    phi, _, _, _ = get_angles(b, theta, bo, ro)
    phi1, phi2 = phi[:2]
    for l in range(ydeg + 2):
        for m in range(-l, l + 1):
            P1 = A.P(l, m, phi1, phi2)
            P2 = A.Pnum(l, m, phi1, phi2)
            assert np.allclose(P1, P2)


if __name__ == "__main__":
    b, theta, bo, ro = args[0]
    phi, _, _, _ = get_angles(b, theta, bo, ro)
    phi1, phi2 = phi[:2]
    for l in range(ydeg + 2):
        for m in range(-l, l + 1):
            P1 = A.P(l, m, phi1, phi2)
            P2 = A.Pnum(l, m, phi1, phi2)
            print(
                "{:2d},{:2d}: {:13.10f} / {:13.10f} / {}".format(
                    l, m, P1, P2, np.allclose(P1, P2)
                )
            )
