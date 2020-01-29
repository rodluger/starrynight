from starrynight.special import compute_W
import numpy as np
from scipy.special import hyp2f1


def test_W():
    imax = 30
    z = np.linspace(-1, 1, 1000)
    W = np.array([compute_W(imax, zi) for zi in z])
    i = np.arange(imax + 1)
    exact = np.array([hyp2f1(-0.5, i, i + 1, zi) for zi in z])
    assert np.allclose(exact, W)
