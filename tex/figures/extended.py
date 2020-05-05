"""Extend illumination source."""
import starry
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# Config
starry.config.lazy = False

# Params from Wong et al. (2019)
aRs = 3.16
RpRs = 0.08229

# Get the two images
res = 999
map = starry.Map(reflected=True, source_npts=1)
img1 = map.render(projection="moll", xs=0, ys=0, zs=aRs / RpRs, res=res)
map = starry.Map(reflected=True, source_npts=300)
img2 = map.render(projection="moll", xs=0, ys=0, zs=aRs / RpRs, rs=1 / RpRs, res=res)
norm = np.nanmax(img2)
img1 /= norm
img2 /= norm

# Plot them. We'll instantiate a *regular* starry map and pass
# in the images so that the intensity gets plotted with the
# regular colormap (instead of with an alpha filter).
fig, ax = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
for axis in ax:
    axis.grid(False)
    axis.set_xticks([])
    axis.set_yticks([])
ax[0].set_title("point source")
ax[1].set_title("extended source")
norm = mpl.colors.Normalize(vmin=0, vmax=1)
map = starry.Map(1)
map.show(image=img1, projection="moll", ax=ax[0], norm=norm)
map.show(image=img2, projection="moll", ax=ax[1], norm=norm)

# Plot contours
X = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2), img2.shape[1])
Y = np.linspace(-np.sqrt(2), np.sqrt(2), img2.shape[0])
levels = np.array([0, 1, 5, 10, 30, 50, 75])
for i, axis, img in zip([0, 1], ax, [img1, img2]):
    cont = axis.contour(
        X,
        Y,
        img / np.nanmax(img) * 100,
        levels,
        colors="w",
        antialiased=True,
        linewidths=0.5,
        linestyles="dotted",
    )
    if i == 0:
        x = np.array([-90, -70, -60, -40]) * 2 * np.sqrt(2) / 180.0
    else:
        x = np.array([-106, -98, -92, -86, -70, -60, -40]) * 2 * np.sqrt(2) / 180.0
    y = np.zeros(len(x))
    manual = zip(x, y)
    fmt = {0: "0%", 1: "1", 5: "5", 10: "10", 30: "30", 50: "50", 75: "75%"}
    labels = axis.clabel(cont, inline=1, fontsize=6, fmt=fmt, manual=manual)
    for l in labels:
        l.set_rotation(0)

# Add a colorbar. We need to find the `imshow` instance from the
# second axis, since `map.show` doesn't return it!
im = [c for c in ax[1].get_children() if type(c) is mpl.image.AxesImage][0]
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="3%", pad=0.25)
cbar = fig.colorbar(im, cax=cax, shrink=0.5)
cbar.ax.tick_params(labelsize=10)
cbar.set_label(label="intensity", fontsize=12)

# Add an invisible colorbar to the first axis so the axis sizes match
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="3%", pad=0.25)
cax.set_visible(False)

# We're done.
fig.savefig("extended.pdf", bbox_inches="tight")
