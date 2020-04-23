"""Earth occultation example."""
import starry
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Disable lazy mode
starry.config.lazy = False

cmap = LinearSegmentedColormap(
    "cmap1",
    segmentdata={
        "red": [[0.0, 0.0, 0.122], [1.0, 1.0, 1.0]],
        "green": [[0.0, 0.0, 0.467], [1.0, 1.0, 1.0]],
        "blue": [[0.0, 0.0, 0.706], [1.0, 1.0, 1.0]],
    },
    N=256,
)
cmap.set_under("k")

for reflected in False, True:

    # Set up the plot
    nim = 12
    npts = 100
    nptsnum = 10
    res = 999
    fig = plt.figure(figsize=(12, 5))
    ax_im = [plt.subplot2grid((4, nim), (0, n)) for n in range(nim)]
    ax_lc = plt.subplot2grid((4, nim), (1, 0), colspan=nim, rowspan=3)

    # Instantiate the Earth
    map = starry.Map(25, reflected=reflected)
    map.load("earth", sigma=0.06)

    # Moon occultation params
    ro = 0.273
    yo = np.linspace(-0.5, 0.5, npts)
    yonum = np.linspace(-0.5, 0.5, nptsnum)
    xo = np.linspace(-1.5, 1.5, npts)
    xonum = np.linspace(-1.5, 1.5, nptsnum)

    # Say the occultation occurs over ~1 radian of the Earth's rotation
    # That's equal to 24 / (2 * pi) hours
    time = np.linspace(0, 24 / (2 * np.pi), npts)
    timenum = np.linspace(0, 24 / (2 * np.pi), nptsnum)
    theta0 = 0
    theta = np.linspace(theta0, theta0 + 180.0 / np.pi, npts, endpoint=True)
    thetanum = np.linspace(theta0, theta0 + 180.0 / np.pi, nptsnum, endpoint=True)

    # Compute and plot the flux *analytically*
    if reflected:
        # Position of the illumination source (the sun).
        # We'll assume it's constant for simplicity
        xs = -1.0 * np.ones_like(time)
        ys = 0.3 * np.ones_like(time)
        zs = 1.0 * np.ones_like(time)
        F = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, xs=xs, ys=ys, zs=zs)
    else:
        F = map.flux(theta=theta, xo=xo, yo=yo, ro=ro)
    maxF = np.max(F)
    F /= maxF
    ax_lc.plot(time, F, "k-", label="analytic")

    # Plot the earth images & compute the numerical flux
    res = 300
    t_num = np.zeros(nim)
    F_num = np.zeros(nim)
    for n in range(nim):
        i = int(np.linspace(0, npts - 1, nim)[n])
        if reflected:
            I = map.render(theta=theta[i], xs=xs[i], ys=ys[i], zs=zs[i], res=res)
        else:
            I = map.render(theta=theta[i], res=res)
        ax_im[n].imshow(I, origin="lower", cmap=cmap, extent=(-1, 1, -1, 1), vmin=1e-8)

        x = np.linspace(-1, 1, 1000)
        y = np.sqrt(1 - x ** 2)
        ax_im[n].plot(x, y, "k-", lw=1, zorder=10)
        ax_im[n].plot(x, -y, "k-", lw=1, zorder=10)

        xm = np.linspace(xo[i] - ro + 1e-5, xo[i] + ro - 1e-5, res)
        ax_im[n].fill_between(
            xm,
            yo[i] - np.sqrt(ro ** 2 - (xm - xo[i]) ** 2),
            yo[i] + np.sqrt(ro ** 2 - (xm - xo[i]) ** 2),
            color="grey",
            zorder=11,
            clip_on=False,
        )
        ax_im[n].plot(
            xm,
            yo[i] - np.sqrt(ro ** 2 - (xm - xo[i]) ** 2),
            "k-",
            lw=1,
            clip_on=False,
            zorder=12,
        )
        ax_im[n].plot(
            xm,
            yo[i] + np.sqrt(ro ** 2 - (xm - xo[i]) ** 2),
            "k-",
            lw=1,
            clip_on=False,
            zorder=12,
        )

        ax_im[n].axis("off")
        ax_im[n].set_xlim(-1.05, 1.05)
        ax_im[n].set_ylim(-1.05, 1.05)

        # Compute the numerical flux by discretely summing over the unocculted region
        x, y, z = map.ops.compute_ortho_grid(res)
        F_num[n] = (
            np.nansum(I.flatten()[(x - xo[i]) ** 2 + (y - yo[i]) ** 2 > ro ** 2])
            * 4
            / res ** 2
        )
        t_num[n] = time[i]

    # Plot the numerical flux
    F_num /= maxF
    ax_lc.plot(t_num, F_num, "C1o", label="numerical")

    # Appearance
    ax_lc.set_ylim(0.6, 1.05)
    ax_lc.set_xlabel("time [hours]", fontsize=16)
    ax_lc.set_ylabel("normalized flux", fontsize=16)
    for tick in ax_lc.get_xticklabels() + ax_lc.get_yticklabels():
        tick.set_fontsize(14)
    ax_lc.legend(loc="lower right", fontsize=12)

    # Save
    if reflected:
        fig.savefig("earthmoon.pdf", bbox_inches="tight", dpi=300)
    else:
        fig.savefig("earthmoon_emitted.pdf", bbox_inches="tight", dpi=300)
