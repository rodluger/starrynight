from support_code import *
import numpy as np
import matplotlib.pyplot as plt

# Setup figure
fig, ax = plt.subplots(2, 4, figsize=(12, 6))
ax = ax.flatten()

# Case 11
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
    res=3999,
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

# Case 12
visualize_simple(
    ax[2],
    -0.5488316824842527,
    4.03591586925189 - np.pi,
    0.34988513192814663,
    0.7753986686719786,
)
axins1 = ax[2].inset_axes([1.15, 0.0, 1.0, 1.0])
visualize_simple(
    axins1,
    -0.5488316824842527,
    4.03591586925189 - np.pi,
    0.34988513192814663,
    0.7753986686719786,
    res=3999,
)
axins1.axis("on")
ax[3].set_visible(False)
x1, x2, y1, y2 = 0.6, 0.7, 0.7, 0.8
axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)
axins1.set_xticks([])
axins1.set_yticks([])
rectangle_patch, connector_lines = ax[2].indicate_inset_zoom(axins1)
rectangle_patch.set_ec("k")
for line in connector_lines:
    line.set_ec("k")

# Case 13
visualize_simple(ax[4], 0.5, np.pi, 0.99, 1.5)
axins2 = ax[4].inset_axes([1.15, 0.0, 1.0, 1.0])
visualize_simple(axins2, 0.5, np.pi, 0.99, 1.5)
axins2.axis("on")
ax[5].set_visible(False)
x1, x2, y1, y2 = 0.25, 1.0, -0.625, 0.125
axins2.set_xlim(x1, x2)
axins2.set_ylim(y1, y2)
axins2.set_xticks([])
axins2.set_yticks([])
rectangle_patch, connector_lines = ax[4].indicate_inset_zoom(axins2)
rectangle_patch.set_ec("k")
for line in connector_lines:
    line.set_ec("k")

# Case 14
visualize_simple(ax[6], -0.5, 0, 0.99, 1.5)
axins3 = ax[6].inset_axes([1.15, 0.0, 1.0, 1.0])
visualize_simple(axins3, -0.5, 0, 0.99, 1.5)
axins3.axis("on")
ax[7].set_visible(False)
x1, x2, y1, y2 = 0.25, 1.0, -0.625, 0.125
axins3.set_xlim(x1, x2)
axins3.set_ylim(y1, y2)
axins3.set_xticks([])
axins3.set_yticks([])
rectangle_patch, connector_lines = ax[6].indicate_inset_zoom(axins3)
rectangle_patch.set_ec("k")
for line in connector_lines:
    line.set_ec("k")

# Appearance
for i, axis in enumerate([ax[0], ax[2], ax[4], ax[6]]):
    axis.axis("off")
    axis.annotate(
        "case {}".format(i + 11),
        xy=(0.5, 0.0),
        xycoords="axes fraction",
        xytext=(0, 0),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=10,
    )
    axis.set_rasterization_zorder(0)

for axis in [axins0, axins1, axins2, axins3]:
    axis.set_rasterization_zorder(0)

fig.savefig("pathological.pdf", bbox_inches="tight", dpi=300)
