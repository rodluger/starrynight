from starrynight.special import compute_W
import numpy as np
from scipy.special import hyp2f1
import pytest


@pytest.mark.parametrize(
    "z", [0.0, 1e-5, 1e-2, 1e-1, 0.25, 0.5, 0.75, 1 - 1e-1, 1 - 1e-2, 1 - 1e-5, 1.0],
)
def test_W(z, imax=10):
    W1 = compute_W(imax, z)
    W2 = np.array([hyp2f1(-0.5, i, i + 1, z) for i in range(imax + 1)])
    assert np.allclose(W1, W2)
