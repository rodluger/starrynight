from starrynight.numerical import Numerical, Brute
import numpy as np
import pytest

# Settings
seed = 0
nruns = 100
res = 999
atol = 1e-2
tol = 1e-7
res = 4999
ydeg = 1
y = [1, 1, 1, 1]

N = Numerical(ydeg, tol=tol)
B = Brute(ydeg, tol=tol, res=res)


def get_args():
    bo = 0
    ro = 2
    while (bo <= ro - 1) or (bo >= 1 + ro):
        if np.random.random() > 0.5:
            ro = np.random.random() * 10
            bo = np.random.random() * 20
        else:
            ro = np.random.random()
            bo = np.random.random() * 2
    theta = np.random.random() * 2 * np.pi
    b = 1 - 2 * np.random.random()
    return b, theta, bo, ro


np.random.seed(seed)
args = [get_args() for n in range(nruns)]


@pytest.mark.parametrize(
    "b,theta,bo,ro", args,
)
def test_mc(b, theta, bo, ro):
    assert np.allclose(
        N.flux(y, b, theta, bo, ro), B.flux(y, b, theta, bo, ro), atol=atol
    )
