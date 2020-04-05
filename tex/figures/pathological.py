from code import *
import numpy as np
import matplotlib.pyplot as plt

# DEBUG
plt.switch_backend("MacOSX")

# Setup figure
fig, ax = plt.subplots(1, 4, figsize=(12, 3))

# Case 9
visualize_simple(
    ax[0], 0.5488316824842527, 4.03591586925189, 0.34988513192814663, 0.7753986686719786
)
axins0 = ax[0].inset_axes([1.15, 0.0, 1.0, 1.0])
visualize_simple(
    axins0,
    0.5488316824842527,
    4.03591586925189,
    0.34988513192814663,
    0.7753986686719786,
)
axins0.axis("on")
ax[1].set_visible(False)
x1, x2, y1, y2 = 0.6, 0.7, 0.7, 0.8
axins0.set_xlim(x1, x2)
axins0.set_ylim(y1, y2)
axins0.set_xticks([])
axins0.set_yticks([])
rectangle_patch, connector_lines = ax[0].indicate_inset_zoom(axins0)
rectangle_patch.set_ec("k")
for line in connector_lines:
    line.set_ec("k")

# Case 10
visualize_simple(ax[2], 0.5, np.pi, 0.99, 1.5)
axins1 = ax[2].inset_axes([1.15, 0.0, 1.0, 1.0])
visualize_simple(axins1, 0.5, np.pi, 0.99, 1.5)
axins1.axis("on")
ax[3].set_visible(False)
x1, x2, y1, y2 = 0.25, 1.0, -0.625, 0.125
axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)
axins1.set_xticks([])
axins1.set_yticks([])
rectangle_patch, connector_lines = ax[2].indicate_inset_zoom(axins1)
rectangle_patch.set_ec("k")
for line in connector_lines:
    line.set_ec("k")

# Appearance
for i, axis in enumerate([ax[0], ax[2]]):
    axis.axis("off")
    axis.annotate(
        "case {}".format(i + 9),
        xy=(0.5, 0.0),
        xycoords="axes fraction",
        xytext=(0, 0),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=10,
    )
    axis.set_rasterization_zorder(0)

fig.savefig("pathological.pdf", bbox_inches="tight", dpi=300)
