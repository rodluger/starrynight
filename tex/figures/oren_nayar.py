import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import starry

# Instantiate a starry map
starry.config.lazy = False
map = starry.Map(reflected=True)

# Settings
res = 300
theta_im = np.linspace(0, 360, 9)[1:-1] * np.pi / 180
theta_hr = np.linspace(0, 360, 1000) * np.pi / 180
theta_lr = theta_hr[5::10]
rs = [0, 15, 30, 45]

# Set up the figure
fig = plt.figure(figsize=(12, 8))
ax_im = np.array(
    [
        [
            plt.subplot2grid((4 * len(rs), len(theta_im)), (2 * i, j), rowspan=2)
            for j in range(len(theta_im))
        ]
        for i in range(len(rs))
    ]
)
ax_lc = plt.subplot2grid(
    (4 * len(rs), len(theta_im)),
    (2 * len(rs) + 1, 0),
    colspan=len(theta_im),
    rowspan=2 * len(rs) - 1,
)

# Illumination colormap
cmap = colors.LinearSegmentedColormap.from_list("illum", ["k", "k"], 256)
cmap._init()
alphas = np.linspace(1.0, 0.0, cmap.N + 3)
cmap._lut[:, -1] = alphas
cmap.set_under((0, 0, 0, 1))
cmap.set_over((0, 0, 0, 0))

# Plot the images & light curves
for i, roughness in enumerate(rs):

    # Set the roughness in degrees
    map.roughness = roughness

    # Plot the images for all phases
    img = map.render(
        xs=np.sin(theta_im), ys=0, zs=-np.cos(theta_im), res=res, on94_exact=False,
    )
    for j in range(len(theta_im)):
        if j < len(theta_im) - 1:
            ax_im[i, j].imshow(
                np.pi * img[j],
                origin="lower",
                extent=(-1, 1, -1, 1),
                cmap=cmap,
                vmin=0,
                vmax=1,
            )
            ax_im[i, j].add_artist(plt.Circle((0, 0), 1, ec="w", fc="none"))
        else:
            ax_im[i, j].plot([0, 1], [0, 0], "C{}".format(i), lw=3)
            ax_im[i, j].set_xlim(0, 2.5)
            ax_im[i, j].set_ylim(-1, 1)
            ax_im[i, j].annotate(
                r"$\sigma = {}^\circ$".format(roughness),
                xy=(1, 0),
                xycoords="data",
                xytext=(15, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=14,
                color="k",
            )
        ax_im[i, j].axis("off")

    # Plot the analytic light curve
    flux = map.flux(xs=np.sin(theta_hr), ys=0, zs=-np.cos(theta_hr))
    ax_lc.plot(
        theta_hr / (2 * np.pi), flux, color="C{}".format(i),
    )

    # Render the exact Oren-Nayar image and compute
    # the numerical flux based on their Equation (30)
    img = map.render(
        xs=np.sin(theta_lr), ys=0, zs=-np.cos(theta_lr), res=res, on94_exact=True,
    )
    flux_num = np.nansum(img, axis=(1, 2)) * 4 / res ** 2
    ax_lc.plot(
        theta_lr / (2 * np.pi), flux_num, "o", ms=2, color="C{}".format(i),
    )

    # Print the difference
    diff = flux[5::10] - flux_num
    print("std = {:.0f} ppm".format(np.std(diff) * 1e6))

ax_lc.set_xlabel("illumination phase", fontsize=18)
ax_lc.set_ylabel("observed intensity", fontsize=18)
ax_lc.axhline(2 / 3, color="C0", lw=1, ls="--")
ax_lc.annotate(
    "lambertian geometric albedo",
    xy=(0, 2 / 3),
    xycoords="data",
    xytext=(15, 5),
    textcoords="offset points",
    ha="left",
    va="bottom",
    fontsize=10,
    color="C0",
)
ax_lc.set_xlim(0, 1)
ax_lc.set_ylim(-0.05, 0.75)

fig.savefig("oren_nayar.pdf", bbox_inches="tight", dpi=300)
