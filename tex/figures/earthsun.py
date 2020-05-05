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
map_e.inc = 90 - 23.5

# Instantiate the Earth in reflected light
map_r = starry.Map(25, reflected=True)
map_r.load("earth", sigma=0.06)
map_r.inc = 90 - 23.5

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

    # Sun occultation params
    t1 = -342
    t2 = -332
    time = np.linspace(t1, t2, npts)  # in minutes
    timenum = np.linspace(t1, t2, nptsnum)
    ro = 6.957e8 / 6.3781e6  # 1 R_sun in R_earth
    yo = -0.5 * ro * np.ones(npts)
    yonum = -0.5 * ro * np.ones(nptsnum)
    a = 1.496e11 / 6.3781e6  # 1 AU in R_earth
    P = 365.25 * 24 * 60  # 1 year in minutes
    xo = a * np.sin(2 * np.pi * time / P)
    xonum = a * np.sin(2 * np.pi * timenum / P)
    zo = a * np.cos(2 * np.pi * time / P)
    zonum = a * np.cos(2 * np.pi * timenum / P)
    theta = 0

    # Compute and plot the flux *analytically*
    if reflected:
        F = map_r.flux(theta=theta, xo=xo, yo=yo, ro=ro, xs=xo, ys=yo, zs=zo)
    else:
        F = map_e.flux(theta=theta, xo=xo, yo=yo, ro=ro)
    maxF = np.max(F)
    F /= maxF
    ax_lc.plot(time - t1, F, "C0-", label="analytic")

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
                xs=xo[i],
                ys=yo[i],
                zs=zo[i],
                theta=theta,
                res=res,
                grid=False,
            )
        else:
            map_e.show(ax=ax_im[n], cmap=cmap, theta=theta, res=res, grid=False)

        # Outline
        x = np.linspace(-1, 1, 1000)
        y = np.sqrt(1 - x ** 2)
        f = 0.98
        ax_im[n].plot(f * x, f * y, "k-", lw=0.5, zorder=0)
        ax_im[n].plot(f * x, -f * y, "k-", lw=0.5, zorder=0)

        # Occultor
        x = np.linspace(-1.5, xo[i] + ro - 1e-5, res)
        y = np.sqrt(ro ** 2 - (x - xo[i]) ** 2)
        ax_im[n].fill_between(
            x, yo[i] - y, yo[i] + y, fc="w", zorder=1, clip_on=True, ec="k", lw=0.5,
        )
        ax_im[n].axis("off")
        ax_im[n].set_xlim(-1.05, 1.05)
        ax_im[n].set_ylim(-1.05, 1.05)
        ax_im[n].set_rasterization_zorder(0)

        # Compute the numerical flux by discretely summing over the unocculted region
        x, y, z = map_e.ops.compute_ortho_grid(res)
        if reflected:
            img = map_r.render(xs=xo[i], ys=yo[i], zs=zo[i], theta=theta, res=res)
        else:
            img = map_e.render(theta=theta, res=res)
        F_num[n] = (
            np.nansum(img.flatten()[(x - xo[i]) ** 2 + (y - yo[i]) ** 2 > ro ** 2])
            * 4
            / res ** 2
        )
        t_num[n] = time[i]

    # Plot the numerical flux
    F_num /= maxF
    ax_lc.plot(t_num - t1, F_num, "C1o", label="numerical")

    # Appearance
    ax_im[-1].set_zorder(-100)
    ax_lc.set_xlabel("time [minutes]", fontsize=16)
    ax_lc.set_ylabel("normalized flux", fontsize=16)
    for tick in ax_lc.get_xticklabels() + ax_lc.get_yticklabels():
        tick.set_fontsize(14)
    ax_lc.legend(loc="lower left", fontsize=12)

    # Save
    if reflected:
        fig.savefig("earthsun.pdf", bbox_inches="tight", dpi=500)
        Fref = np.array(F)
    else:
        fig.savefig("earthsun_emitted.pdf", bbox_inches="tight", dpi=500)
        Femi = np.array(F)

# Plot the difference
fig, ax = plt.subplots(1, figsize=(12, 2))
ax.plot(time - t1, 100 * (Fref - Femi), "C0-")
ax.set_xlabel("time [minutes]", fontsize=16)
ax.set_ylabel("difference", fontsize=16, labelpad=0)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontsize(14)
ax.set_ylim(-6.5, 6.5)
ax.set_yticks([-5, 0, 5])
ax.set_yticklabels(["-5%", "0%", "5%"])
fig.savefig("earthsun_diff.pdf", bbox_inches="tight", dpi=500)
