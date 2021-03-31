"""Earth occultation example."""
import starry
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Disable lazy mode
starry.config.lazy = False

# Instantiate the Earth in emitted light
map_e = starry.Map(25)
map_e.load("earth", sigma=0.06)

# Instantiate the Earth in reflected light
map_r = starry.Map(25, reflected=True)
map_r.load("earth", sigma=0.06)

# Earth colormap
cmap = LinearSegmentedColormap(
    "earth",
    segmentdata={
        "red": [[0.0, 0.0, 0.122], [1.0, 1.0, 1.0]],
        "green": [[0.0, 0.0, 0.467], [1.0, 1.0, 1.0]],
        "blue": [[0.0, 0.0, 0.706], [1.0, 1.0, 1.0]],
    },
    N=256,
)

for reflected in (True, False):

    # Set up the plot
    nim = 12
    npts = 100
    nptsnum = 10
    res = 300
    fig = plt.figure(figsize=(12, 5))
    ax_im = [plt.subplot2grid((4, nim), (0, n)) for n in range(nim)]
    ax_lc = plt.subplot2grid((4, nim), (1, 0), colspan=nim, rowspan=3)

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
        F = map_r.flux(theta=theta, xo=xo, yo=yo, ro=ro, xs=xs, ys=ys, zs=zs)
    else:
        F = map_e.flux(theta=theta, xo=xo, yo=yo, ro=ro)
    maxF = np.max(F)
    F /= maxF
    ax_lc.plot(time, F, "C0-", label="analytic")

    # Plot the earth images & compute the numerical flux
    t_num = np.zeros(nim)
    F_num = np.zeros(nim)
    for n in range(nim):
        i = int(np.linspace(0, npts - 1, nim)[n])

        # Show the image
        if reflected:
            map_r.show(
                ax=ax_im[n],
                cmap=cmap,
                xs=xs[i],
                ys=ys[i],
                zs=zs[i],
                theta=theta[i],
                res=res,
                grid=False,
            )
        else:
            map_e.show(ax=ax_im[n], cmap=cmap, theta=theta[i], res=res, grid=False)

        # Outline
        x = np.linspace(-1, 1, 1000)
        y = np.sqrt(1 - x ** 2)
        f = 0.98
        ax_im[n].plot(f * x, f * y, "k-", lw=0.5, zorder=0)
        ax_im[n].plot(f * x, -f * y, "k-", lw=0.5, zorder=0)

        # Occultor
        x = np.linspace(xo[i] - ro + 1e-5, xo[i] + ro - 1e-5, res)
        y = np.sqrt(ro ** 2 - (x - xo[i]) ** 2)
        ax_im[n].fill_between(
            x,
            yo[i] - y,
            yo[i] + y,
            fc="#aaaaaa",
            zorder=1,
            clip_on=False,
            ec="k",
            lw=0.5,
        )
        ax_im[n].axis("off")
        ax_im[n].set_xlim(-1.05, 1.05)
        ax_im[n].set_ylim(-1.05, 1.05)
        ax_im[n].set_rasterization_zorder(0)

        # Compute the numerical flux by discretely summing over the unocculted region
        (lat, lon), (x, y, z) = map_e.ops.compute_ortho_grid(res)
        if reflected:
            img = map_r.render(xs=xs[i], ys=ys[i], zs=zs[i], theta=theta[i], res=res)
        else:
            img = map_e.render(theta=theta[i], res=res)
        F_num[n] = (
            np.nansum(img.flatten()[(x - xo[i]) ** 2 + (y - yo[i]) ** 2 > ro ** 2])
            * 4
            / res ** 2
        )
        t_num[n] = time[i]

    # Plot the numerical flux
    F_num /= maxF
    ax_lc.plot(t_num, F_num, "C1o", label="numerical")

    # Appearance
    ax_im[-1].set_zorder(-100)
    ax_lc.set_ylim(0.6, 1.05)
    ax_lc.set_xlabel("time [hours]", fontsize=16)
    ax_lc.set_ylabel("normalized flux", fontsize=16)
    for tick in ax_lc.get_xticklabels() + ax_lc.get_yticklabels():
        tick.set_fontsize(14)
    ax_lc.legend(loc="lower right", fontsize=12)

    # Save
    if reflected:
        fig.savefig("earthmoon.pdf", bbox_inches="tight", dpi=500)
    else:
        fig.savefig("earthmoon_emitted.pdf", bbox_inches="tight", dpi=500)
