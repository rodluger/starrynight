import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

fig, ax = plt.subplots(1, figsize=(6, 6))
ax.set_aspect(1)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis("off")
b = 0.25
theta = np.pi / 3

# Illumination
x0, y0 = np.meshgrid(np.linspace(-1, 1, 3000), np.linspace(-1, 1, 3000))

x = x0 * np.cos(theta) + y0 * np.sin(theta)
y = -x0 * np.sin(theta) + y0 * np.cos(theta)
z = np.sqrt(1 - x ** 2 - y ** 2)
I = np.sqrt(1 - b ** 2) * y - b * z
I[I < 0] = 0
cmap = plt.get_cmap("Greys_r")
ax.imshow(
    I, extent=(-1, 1, -1, 1), origin="lower", vmin=1e-8, vmax=np.nanmax(I), cmap=cmap
)

# Planet
x = np.linspace(-1, 1, 1000)
yp = np.sqrt(1 - x ** 2)
ax.plot(x, yp, "k-", lw=1, zorder=99)
ax.plot(x, -yp, "k-", lw=1, zorder=99)

# Terminator
x0 = np.linspace(-1, 1, 1000)
y0 = b * np.sqrt(1 - x0 ** 2)
x = x0 * np.cos(theta) - y0 * np.sin(theta)
y = x0 * np.sin(theta) + y0 * np.cos(theta)
plt.plot(x, y, "r-", lw=1)

ax.plot([0, -b * np.sin(theta)], [0, b * np.cos(theta)], "w--", lw=1)
ax.plot([0, np.cos(theta)], [0, np.sin(theta)], "w--", lw=1)
ax.plot([0, 1], [0, 0], "w--", lw=1)
ax.annotate(
    r"$b$",
    xy=(0.45, 0.57),
    xycoords="axes fraction",
    ha="left",
    va="center",
    color="w",
    fontsize=18,
)
ax.annotate(
    r"$1$",
    xy=(0.58, 0.73),
    xycoords="axes fraction",
    ha="left",
    va="center",
    color="w",
    fontsize=18,
)
ax.annotate(
    r"$\theta$",
    xy=(0.62, 0.57),
    xycoords="axes fraction",
    ha="left",
    va="center",
    color="w",
    fontsize=18,
)

arc = Arc((0, 0), 0.5, 0.5, 0, 0, theta * 180 / np.pi, color="w", lw=1, zorder=3,)
ax.add_patch(arc)

# Save
fig.savefig("illum.pdf", bbox_inches="tight")
