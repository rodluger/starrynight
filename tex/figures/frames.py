import matplotlib.pyplot as plt
import numpy as np
import starry
from starry._plotting import get_ortho_latitude_lines, get_ortho_longitude_lines
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch


# Config
starry.config.lazy = False
starry.config.quiet = True


def plot_grid(ax, inc, obl, theta):
    inc *= np.pi / 180
    obl *= np.pi / 180
    theta *= np.pi / 180
    x = np.linspace(-1, 1, 10000)
    y = np.sqrt(1 - x ** 2)
    borders = [None, None]
    (borders[0],) = ax.plot(x, y, "k-", alpha=1, lw=1)
    (borders[1],) = ax.plot(x, -y, "k-", alpha=1, lw=1)
    lats = get_ortho_latitude_lines(inc=inc, obl=obl)
    latlines = [None for n in lats]
    for n, l in enumerate(lats):
        (latlines[n],) = ax.plot(l[0], l[1], "k-", lw=0.5, alpha=0.5, zorder=99)
    lons = get_ortho_longitude_lines(inc=inc, obl=obl, theta=theta)
    lonlines = [None for n in lons]
    for n, l in enumerate(lons):
        (lonlines[n],) = ax.plot(l[0], l[1], "k-", lw=0.5, alpha=0.5, zorder=99)


# Params
ydeg = 30
res = 300
inc = 60
obl = 23.5
theta = 60
xs = -1
ys = 1
zs = 0.5
xo = 0.4
yo = 0.5
ro = 0.5

# Rotated params
obl_int = obl + np.arctan(ys / xs) * 180 / np.pi
xs_int = 0
ys_int = np.sqrt(ys ** 2 + xs ** 2)
obl_greens = obl + np.arctan(xo / yo) * 180 / np.pi
t = np.arctan(xo / yo)
xs_greens = xs * np.cos(t) - ys * np.sin(t)
ys_greens = xs * np.sin(t) + ys * np.cos(t)
bo = np.sqrt(xo ** 2 + yo ** 2)
t = np.arctan(ys / xs)
xo_int = xo * np.cos(t) - yo * np.sin(t)
yo_int = xo * np.sin(t) + yo * np.cos(t)

# Load the Earth map
mape = starry.Map(ydeg)
mape.load("earth", sigma=0.05)
mapf = starry.Map(ydeg, reflected=True)

# Render it
img = np.zeros((5, res, res))
flt = np.ones((5, res, res))
img[0] = mape.render(res=res)
mape.inc = inc
mape.obl = obl
img[1] = mape.render(theta=theta, res=res)
img[2] = mape.render(theta=theta, res=res)
flt[2] = mapf.render(xs=xs, ys=ys, zs=zs, res=res)
flt[2] /= np.nanmax(flt[2])
mape.obl = obl_int
img[3] = mape.render(theta=theta, res=res)
flt[3] = mapf.render(xs=xs_int, ys=ys_int, zs=zs, res=res)
flt[3] /= np.nanmax(flt[3])
mape.obl = obl_greens
img[4] = mape.render(theta=theta, res=res)
flt[4] = mapf.render(xs=xs_greens, ys=ys_greens, zs=zs, res=res)
flt[4] /= np.nanmax(flt[4])


# Setup
fig = plt.figure(figsize=(14, 5))
ax = [
    plt.subplot2grid((3, 5), (1, 4)),
    plt.subplot2grid((3, 5), (1, 3)),
    plt.subplot2grid((3, 5), (1, 2)),
    plt.subplot2grid((3, 5), (0, 1)),
    plt.subplot2grid((3, 5), (0, 0)),
    plt.subplot2grid((3, 5), (2, 1)),
    plt.subplot2grid((3, 5), (2, 0)),
]

cmape = LinearSegmentedColormap(
    "cmape",
    segmentdata={
        "red": [[0.0, 0.0, 0.122], [1.0, 1.0, 1.0]],
        "green": [[0.0, 0.0, 0.467], [1.0, 1.0, 1.0]],
        "blue": [[0.0, 0.0, 0.706], [1.0, 1.0, 1.0]],
    },
    N=256,
)

cmapf = LinearSegmentedColormap.from_list("cmapf", ["k", "k"], 256)
cmapf._init()
alphas = np.linspace(1.0, 0.0, cmapf.N + 3)
cmapf._lut[:, -1] = alphas

img_kw = dict(
    origin="lower",
    extent=(-1, 1, -1, 1),
    vmin=1e-8,
    cmap=cmape,
    interpolation="none",
    zorder=-2,
)

flt_kw = dict(
    origin="lower",
    extent=(-1, 1, -1, 1),
    vmin=0,
    vmax=1,
    cmap=cmapf,
    interpolation="none",
    zorder=-1,
)

# Show the maps
ax[0].imshow(img[0], **img_kw)
ax[0].imshow(flt[0], **flt_kw)
plot_grid(ax[0], 90, 0, 0)
ax[1].imshow(img[1], **img_kw)
ax[1].imshow(flt[1], **flt_kw)
plot_grid(ax[1], inc, obl, theta)
ax[2].imshow(img[2], **img_kw)
ax[2].imshow(flt[2], **flt_kw)
plot_grid(ax[2], inc, obl, theta)
ax[3].imshow(img[3], **img_kw)
ax[3].imshow(flt[3], **flt_kw)
plot_grid(ax[3], inc, obl_int, theta)
ax[4].imshow(img[3], **img_kw)
ax[4].imshow(flt[3], **flt_kw)
plot_grid(ax[4], inc, obl_int, theta)
ax[5].imshow(img[4], **img_kw)
ax[5].imshow(flt[4], **flt_kw)
plot_grid(ax[5], inc, obl_greens, theta)
ax[6].imshow(img[4], **img_kw)
ax[6].imshow(flt[4], **flt_kw)
plot_grid(ax[6], inc, obl_greens, theta)

# Show the occultor
ax[2].add_artist(
    plt.Circle(
        (xo, yo),
        ro,
        fc=(0.5, 0.5, 0.5, 1),
        ec="none",
        clip_on=False,
        zorder=100,
        alpha=0.5,
    )
)
ax[2].add_artist(
    plt.Circle((xo, yo), ro, fc="none", ec="k", lw=1, clip_on=False, zorder=100,)
)

ax[3].add_artist(
    plt.Circle(
        (xo_int, yo_int),
        ro,
        fc="none",
        ec="k",
        lw=1,
        ls="--",
        clip_on=False,
        zorder=100,
    )
)
ax[4].add_artist(
    plt.Circle(
        (xo_int, yo_int),
        ro,
        fc="none",
        ec="k",
        lw=1,
        ls="--",
        clip_on=False,
        zorder=100,
    )
)
ax[5].add_artist(
    plt.Circle(
        (0, bo), ro, fc=(0.5, 0.5, 0.5, 1), ec="k", lw=1, clip_on=False, zorder=100
    )
)
ax[6].add_artist(
    plt.Circle(
        (0, bo), ro, fc=(0.5, 0.5, 0.5, 1), ec="k", lw=1, clip_on=False, zorder=100
    )
)

# Appearance
for axis in ax:
    axis.set_rasterization_zorder(0)
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis.axis("off")

# Annotations
fig.patches.append(
    FancyArrowPatch(
        fig.transFigure.inverted().transform(ax[0].transData.transform((-0.9, 0))),
        fig.transFigure.inverted().transform(ax[1].transData.transform((0.9, 0))),
        transform=fig.transFigure,
        fc="k",
        arrowstyle="-|>",
        alpha=1,
        mutation_scale=10.0,
    )
)

fig.patches.append(
    FancyArrowPatch(
        fig.transFigure.inverted().transform(ax[1].transData.transform((-0.9, 0))),
        fig.transFigure.inverted().transform(ax[2].transData.transform((0.9, 0))),
        transform=fig.transFigure,
        fc="k",
        arrowstyle="-|>",
        alpha=1,
        mutation_scale=10.0,
    )
)

fig.patches.append(
    FancyArrowPatch(
        fig.transFigure.inverted().transform(ax[2].transData.transform((-0.9, 0))),
        fig.transFigure.inverted().transform(ax[3].transData.transform((0.9, 0))),
        transform=fig.transFigure,
        fc="k",
        arrowstyle="-|>",
        connectionstyle="bar,angle=90,fraction=-0.2",
        alpha=1,
        mutation_scale=10.0,
    )
)

fig.patches.append(
    FancyArrowPatch(
        fig.transFigure.inverted().transform(ax[3].transData.transform((-0.9, 0))),
        fig.transFigure.inverted().transform(ax[4].transData.transform((0.9, 0))),
        transform=fig.transFigure,
        fc="k",
        arrowstyle="-|>",
        alpha=1,
        mutation_scale=10.0,
    )
)

fig.patches.append(
    FancyArrowPatch(
        fig.transFigure.inverted().transform(ax[2].transData.transform((-0.9, 0))),
        fig.transFigure.inverted().transform(ax[5].transData.transform((0.9, 0))),
        transform=fig.transFigure,
        fc="k",
        arrowstyle="-|>",
        connectionstyle="bar,angle=90,fraction=0.2",
        alpha=1,
        mutation_scale=10.0,
    )
)

fig.patches.append(
    FancyArrowPatch(
        fig.transFigure.inverted().transform(ax[5].transData.transform((-0.9, 0))),
        fig.transFigure.inverted().transform(ax[6].transData.transform((0.9, 0))),
        transform=fig.transFigure,
        fc="k",
        arrowstyle="-|>",
        alpha=1,
        mutation_scale=10.0,
    )
)

ax[0].text(
    0.5,
    -0.2,
    r"$\mathcal{F}_0$",
    transform=ax[0].transAxes,
    size=18,
    ha="center",
    va="center",
)

ax[1].text(
    0.5,
    -0.2,
    r"$\mathcal{F}_\mathrm{sky}$",
    transform=ax[1].transAxes,
    size=18,
    ha="center",
    va="center",
)

ax[2].text(
    0.5,
    -0.2,
    r"$\mathcal{F}_\mathrm{sky}$",
    transform=ax[2].transAxes,
    size=18,
    ha="center",
    va="center",
)

ax[3].text(
    0.5,
    -0.2,
    r"$\mathcal{F}_\mathrm{int}$",
    transform=ax[3].transAxes,
    size=18,
    ha="center",
    va="center",
)

ax[4].text(
    0.5,
    -0.2,
    r"$\mathcal{F}_\mathrm{int}$",
    transform=ax[4].transAxes,
    size=18,
    ha="center",
    va="center",
)

ax[5].text(
    0.5,
    -0.2,
    r"$\mathcal{F}_\mathrm{greens}$",
    transform=ax[5].transAxes,
    size=18,
    ha="center",
    va="center",
)

ax[6].text(
    0.5,
    -0.2,
    r"$\mathcal{F}_\mathrm{greens}$",
    transform=ax[6].transAxes,
    size=18,
    ha="center",
    va="center",
)

ax[0].text(
    0.5,
    1.2,
    r"$\tilde{\mathbf{y}}$",
    transform=ax[0].transAxes,
    size=14,
    ha="center",
    va="center",
)

ax[1].text(
    0.5,
    1.2,
    r"$\tilde{\mathbf{y}}$",
    transform=ax[1].transAxes,
    size=14,
    ha="center",
    va="center",
)

ax[2].text(
    0.5,
    1.2,
    r"$\tilde{\mathbf{y}}$",
    transform=ax[2].transAxes,
    size=14,
    ha="center",
    va="center",
)

ax[3].text(
    0.5,
    1.2,
    r"$\tilde{\mathbf{y}}$",
    transform=ax[3].transAxes,
    size=14,
    ha="center",
    va="center",
)

ax[4].text(
    0.5,
    1.2,
    r"$\tilde{\mathbf{p}}$",
    transform=ax[4].transAxes,
    size=14,
    ha="center",
    va="center",
)

ax[5].text(
    0.5,
    1.2,
    r"$\tilde{\mathbf{y}}$",
    transform=ax[5].transAxes,
    size=14,
    ha="center",
    va="center",
)

ax[6].text(
    0.5,
    1.2,
    r"$\tilde{\mathbf{g}}$",
    transform=ax[6].transAxes,
    size=14,
    ha="center",
    va="center",
)

ax[0].text(
    -0.47,
    0.6,
    r"$\mathbf{R}$",
    transform=ax[0].transAxes,
    size=18,
    weight="bold",
    ha="center",
    va="center",
)

ax[1].text(
    -0.47,
    0.6,
    r"$\mathbf{\Psi}$",
    transform=ax[1].transAxes,
    size=18,
    weight="bold",
    ha="center",
    va="center",
)

ax[2].text(
    -0.3,
    1.15,
    r"$\mathbf{R''}$",
    transform=ax[2].transAxes,
    size=18,
    weight="bold",
    ha="center",
    va="center",
)

ax[2].text(
    -0.3,
    -0.15,
    r"$\mathbf{R'}$",
    transform=ax[2].transAxes,
    size=18,
    weight="bold",
    ha="center",
    va="center",
)


ax[3].text(
    -0.47,
    0.6,
    r"$\mathbf{A_1}$",
    transform=ax[3].transAxes,
    size=18,
    weight="bold",
    ha="center",
    va="center",
)

ax[5].text(
    -0.47,
    0.6,
    r"$\mathbf{A}$",
    transform=ax[5].transAxes,
    size=18,
    weight="bold",
    ha="center",
    va="center",
)


# Save
fig.savefig("frames.pdf", bbox_inches="tight", dpi=300)
