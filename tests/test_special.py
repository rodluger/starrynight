from starrynight.special import compute_W
import numpy as np
from scipy.special import hyp2f1
import pytest


@pytest.mark.parametrize(
    "z", [0.1, 0.25, 0.5, 0.9],
)
def test_W(z, imax=10):
    W1 = compute_W(imax, z)
    W2 = np.array([hyp2f1(-0.5, i, i + 1, z) for i in range(imax + 1)])
    assert np.allclose(W1, W2)
