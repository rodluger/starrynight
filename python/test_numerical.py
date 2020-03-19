from cases import CASE
from numerical import Numerical, Brute
import numpy as np
import pytest


ydeg = 1
y = [1, 1, 1, 1]
N = Numerical(ydeg)
B = Brute(ydeg, res=4999)


def run(b, theta, bo, ro):
    assert np.allclose(
        N.flux(y, b, theta, bo, ro), B.flux(y, b, theta, bo, ro), atol=0.002
    )


@pytest.mark.parametrize("b,theta,bo,ro", CASE[0])
def test_simple(b, theta, bo, ro):
    """Test cases where the occultor does not touch the terminator."""
    run(b, theta, bo, ro)


@pytest.mark.parametrize("b,theta,bo,ro", CASE[1])
def test_PQT(b, theta, bo, ro):
    """Test cases involving the three primitive integrals."""
    run(b, theta, bo, ro)


@pytest.mark.parametrize("b,theta,bo,ro", CASE[2])
def test_PT(b, theta, bo, ro):
    """Test cases involving only the P and T primitive integrals."""
    run(b, theta, bo, ro)


@pytest.mark.parametrize("b,theta,bo,ro", CASE[3])
def test_triple(b, theta, bo, ro):
    """Test cases with three points of intersection b/w occultor and terminator."""
    run(b, theta, bo, ro)


@pytest.mark.parametrize("b,theta,bo,ro", CASE[4])
def test_quadruple(b, theta, bo, ro):
    """Test cases with four points of intersection b/w occultor and terminator."""
    run(b, theta, bo, ro)


@pytest.mark.parametrize("b,theta,bo,ro", CASE[5])
def test_edge_cases(b, theta, bo, ro):
    """Test various edge cases."""
    run(b, theta, bo, ro)
