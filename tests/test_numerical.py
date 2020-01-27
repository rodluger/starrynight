from starrynight import Numerical, Brute
import numpy as np
import pytest


N = Numerical(tol=1e-7)
B = Brute(tol=1e-7, res=4999)


def run(b, theta, bo, ro):
    N.b, N.theta, N.bo, N.ro = b, theta, bo, ro
    B.b, B.theta, B.bo, B.ro = b, theta, bo, ro
    assert np.allclose(N.flux(), B.flux(), atol=0.002)


@pytest.mark.parametrize(
    "b,theta,bo,ro",
    [
        [0.5, 0.1, 1.2, 0.1],
        [0.5, 0.1, 0.1, 1.2],
        [0.5, 0.1, 0.8, 0.1],
        [0.5, 0.1, 0.9, 0.2],
        [0.5, np.pi + 0.1, 0.8, 0.1],
        [0.5, np.pi + 0.1, 0.9, 0.2],
        [0.5, 0.1, 0.5, 1.25],
        [0.5, np.pi + 0.1, 0.5, 1.25],
    ],
)
def test_simple(b, theta, bo, ro):
    """Test cases where the occultor does not touch the terminator."""
    run(b, theta, bo, ro)


@pytest.mark.parametrize(
    "b,theta,bo,ro",
    [
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
    ],
)
def test_PQT(b, theta, bo, ro):
    """Test cases involving the three primitive integrals."""
    run(b, theta, bo, ro)


@pytest.mark.parametrize(
    "b,theta,bo,ro",
    [
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
    ],
)
def test_PT(b, theta, bo, ro):
    """Test cases involving only the P and T primitive integrals."""
    run(b, theta, bo, ro)


@pytest.mark.parametrize(
    "b,theta,bo,ro",
    [
        [
            0.5488316824842527,
            4.03591586925189,
            0.34988513192814663,
            0.7753986686719786,
        ],
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
    ],
)
def test_triple(b, theta, bo, ro):
    """Test cases with three points of intersection b/w occultor and terminator."""
    run(b, theta, bo, ro)


@pytest.mark.parametrize(
    "b,theta,bo,ro", [[0.5, np.pi, 0.99, 1.5], [-0.5, 0.0, 0.99, 1.5],],
)
def test_quadruple(b, theta, bo, ro):
    """Test cases with four points of intersection b/w occultor and terminator."""
    run(b, theta, bo, ro)


@pytest.mark.parametrize(
    "b,theta,bo,ro", [[0.5, np.pi, 1.0, 1.5],],
)
def test_edge_cases(b, theta, bo, ro):
    """Test various edge cases."""
    run(b, theta, bo, ro)
