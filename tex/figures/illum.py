import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.colors import LinearSegmentedColormap

fig, ax = plt.subplots(1, figsize=(6, 6))
ax.set_aspect(1)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis("off")
b = 0.25
theta = 0

# Illumination
cdict = {
    "red": [[0.0, 0.0, 0.122], [1.0, 1.0, 1.0]],
    "green": [[0.0, 0.0, 0.467], [1.0, 1.0, 1.0]],
    "blue": [[0.0, 0.0, 0.706], [1.0, 1.0, 1.0]],
}
cmap = LinearSegmentedColormap("testCmap", segmentdata=cdict, N=256)
cmap.set_under("k")
x0, y0 = np.meshgrid(np.linspace(-1, 1, 3000), np.linspace(-1, 1, 3000))
x = x0 * np.cos(theta) + y0 * np.sin(theta)
y = -x0 * np.sin(theta) + y0 * np.cos(theta)
z = np.sqrt(1 - x ** 2 - y ** 2)
I = np.sqrt(1 - b ** 2) * y - b * z
I[I < 0] = 0
ax.imshow(
    I,
    extent=(-1, 1, -1, 1),
    origin="lower",
    vmin=1e-8,
    vmax=1,
    cmap=cmap,
    alpha=0.75,
    zorder=-1,
)

# Planet
x = np.linspace(-1, 1, 1000)
yp = np.sqrt(1 - x ** 2)
ax.plot(x, yp, "k-", lw=1.5, zorder=98)
ax.plot(x, -yp, "k-", lw=1.5, zorder=98)

# Terminator
x0 = np.linspace(-1, 1, 1000)
y0 = b * np.sqrt(1 - x0 ** 2)
x = x0 * np.cos(theta) - y0 * np.sin(theta)
y = x0 * np.sin(theta) + y0 * np.cos(theta)
plt.plot(x, y, "k-", lw=2, zorder=99)
x = x0 * np.cos(theta) + y0 * np.sin(theta)
y = x0 * np.sin(theta) - y0 * np.cos(theta)
plt.plot(x, y, "k--", lw=1, zorder=1)

ax.plot([0, -b * np.sin(theta)], [0, b * np.cos(theta)], "w-", lw=1)
ax.annotate(
    r"$b$",
    xy=(0.55, 0.43),
    xycoords="axes fraction",
    ha="center",
    va="center",
    color="w",
    fontsize=14,
)
ax.annotate(
    r"$1$",
    xy=(0.75, 0.34),
    xycoords="axes fraction",
    ha="center",
    va="center",
    color="w",
    fontsize=14,
)
ax.plot([0, np.cos(theta)], [0, np.sin(theta)], "w-", lw=1)

if theta != 0:
    ax.annotate(
        r"$\theta$",
        xy=(0.6, 0.57),
        xycoords="axes fraction",
        ha="left",
        va="center",
        color="w",
        fontsize=18,
    )
    arc = Arc((0, 0), 0.4, 0.4, 0, 0, theta * 180 / np.pi, color="w", lw=1, zorder=3,)
    ax.add_patch(arc)

ax.plot([0, 0], [1, 1.5], "C1")
ax.plot(0, 1.5, "C1o")

ax.annotate(
    "$(0, y_s, z_s)$",
    xy=(-1.5 * np.sin(theta), 1.5 * np.cos(theta)),
    xycoords="data",
    xytext=(0, 15),
    textcoords="offset points",
    ha="center",
    va="center",
    fontsize=12,
)
ax.annotate(
    "case 0",
    xy=(0.5, 0.0),
    xycoords="axes fraction",
    xytext=(0, 0),
    textcoords="offset points",
    ha="center",
    va="top",
    fontsize=14,
)

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.75)
ax.set_rasterization_zorder(0)

# Save
fig.savefig("illum.pdf", bbox_inches="tight", dpi=300)
