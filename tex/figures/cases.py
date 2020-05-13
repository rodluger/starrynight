from support_code import *
import numpy as np
import matplotlib.pyplot as plt

# Setup figure
fig, ax = plt.subplots(3, 4, figsize=(12, 8))
ax = ax.flatten()

# Legend
ax[11].plot([0, 1], [0, 0], color="C0", lw=8, alpha=0.75, label="dayside")
ax[11].plot(
    [0, 1],
    [0, 0],
    color=(0.314, 0.431, 0.514),
    lw=8,
    alpha=0.5,
    label="occulted dayside",
)
ax[11].plot([-1, 1], [0, 0], color="k", alpha=0.75, lw=8, label="nightside")
ax[11].plot([0, 1], [0, 0], color="k", alpha=0.5, lw=8, label="occulted nightside")
ax[11].plot([0, 1], [0, 0], color="k", lw=2, ls="--", label="occultor limb")
ax[11].plot([0, 1], [0, 0], color="C1", lw=2, label="integration boundary")
ax[11].set_ylim(1, 2)
ax[11].legend(loc="center", fontsize=10, frameon=False)

# Cases
visualize_simple(ax[0], 0.5, 75 * np.pi / 180, 0, 0.0)
visualize_simple(ax[1], 0.5, 75 * np.pi / 180, 0, 1.1)
visualize_simple(ax[2], -0.25, 0, 0.175, 1.05)
visualize_simple(ax[3], 0.25, np.pi, 0.175, 1.05)
visualize_simple(ax[4], -0.25, 190 * np.pi / 180, 0.8, 0.3)
visualize_simple(ax[5], 0.25, 10 * np.pi / 180, 0.8, 0.3)
visualize_simple(ax[6], 0.5, 75 * np.pi / 180, 0.5, 0.7)
visualize_simple(ax[7], 0.5, -10 * np.pi / 180, 0.5, 0.7)
visualize_simple(ax[8], -0.5, -190 * np.pi / 180, 0.5, 0.7)
visualize_simple(ax[9], 0.5, np.pi, 4.65, 5.0)
visualize_simple(ax[10], -0.5, 0, 4.65, 5.0)

# Appearance
for i, axis in enumerate(ax):
    axis.axis("off")
    if i < 11:
        axis.annotate(
            "case {}".format(i),
            xy=(0.5, 0.0),
            xycoords="axes fraction",
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=10,
        )
    axis.set_rasterization_zorder(0)

fig.savefig("cases.pdf", bbox_inches="tight", dpi=300)
