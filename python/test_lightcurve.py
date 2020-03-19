from starrynight import StarryNight
from viz import visualize
from numerical import Brute
import numpy as np
import pytest
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


@pytest.mark.parametrize(
    "b,theta,ro",
    [
        [0.25, np.pi / 3, 0.3],
        [-0.25, np.pi / 3, 0.3],
        [0.25, -np.pi / 3, 0.3],
        [-0.25, -np.pi / 3, 0.3],
        [0.25, 2 * np.pi / 3, 0.3],
        [-0.25, 2 * np.pi / 3, 0.3],
        [0.25, 4 * np.pi / 3, 0.3],
        [-0.25, 4 * np.pi / 3, 0.3],
        [0.5, np.pi / 2, 1.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.1],
        [1.0 - 1e-3, 0.0, 0.5],
        [-1.0 + 1e-3, 0.0, 0.5],
        [-1.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [0.25, np.pi / 2, 0.5],
    ],
)
def test_lightcurve(b, theta, ro, y=[1], ns=1000, nb=50, res=999, plot=True):

    # Array over full occultation, including all singularities
    bo = np.linspace(0, 1 + ro, ns, endpoint=True)
    for pt in [ro, 1, 1 - ro, b + ro]:
        if pt >= 0:
            bo[np.argmin(np.abs(bo - pt))] = pt

    # Setup
    ydeg = int(np.sqrt(len(y)) - 1)
    flux = np.zeros_like(bo)
    flux_num = np.zeros_like(bo) * np.nan
    map = StarryNight(ydeg)
    map_num = Brute(ydeg, res=res)
    computed = np.zeros(ns, dtype=bool)

    # Compute
    for i, boi in tqdm(enumerate(bo), total=len(bo)):
        flux[i] = map.flux(y, b, theta, boi, ro)
        if (i == 0) or (i == ns - 1) or (i % (ns // nb) == 0):
            flux_num[i] = map_num.flux(y, b, theta, boi, ro)
            computed[i] = True

    # Fix baseline in numerical result
    map_num.res = 4999
    d = map_num.flux(y, b, theta, 0.0, ro) - flux_num[0]
    flux_num += d

    # Interpolate over numerical result
    f = interp1d(bo[computed], flux_num[computed], kind="cubic")
    flux_num_interp = f(bo)

    # Plot
    if plot:
        fig = plt.figure()
        plt.plot(bo, flux, "C0-", label="starry")
        plt.plot(bo, flux_num, "C1o", label="brute")
        plt.plot(bo, flux_num_interp, "C1-")
        plt.legend(loc="best")
        plt.xlabel("impact parameter")
        plt.ylabel("flux")
        fig.savefig(
            "test_lightcurve[{}-{}-{}].pdf".format(b, theta, ro), bbox_inches="tight"
        )

    # Compare with very lax tolerance; we're mostly looking
    # for gross outliers
    diff = np.abs(flux - flux_num_interp)
    assert np.max(diff) < 0.001
