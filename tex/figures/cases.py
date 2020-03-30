from code import *
import numpy as np
import matplotlib.pyplot as plt


# Setup figure
fig, ax = plt.subplots(2, 4, figsize=(12, 5))
ax = ax.flatten()

# Labels
for i, axis in enumerate(ax):
    axis.axis("off")
    axis.annotate(
        "case {}".format(i + 1),
        xy=(0.5, 0.0),
        xycoords="axes fraction",
        xytext=(0, 0),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=10,
    )

visualize_simple(ax[0], 0.5, 75 * np.pi / 180, 0, 1.1)
visualize_simple(ax[1], 0.5, 75 * np.pi / 180, 0, 0.25)
visualize_simple(ax[2], 0.25, 10 * np.pi / 180, 0.8, 0.3)
visualize_simple(ax[3], -0.25, np.pi, 0.15, 1.05)
visualize_simple(ax[4], 0.5, 75 * np.pi / 180, 0.5, 0.7)
visualize_simple(ax[5], 0.5, -10 * np.pi / 180, 0.5, 0.7)
visualize_simple(ax[6], 0.5, np.pi, 4.65, 5.0)
visualize_simple(ax[7], -0.5, 0, 4.65, 5.0)

fig.savefig("cases.pdf", bbox_inches="tight")
