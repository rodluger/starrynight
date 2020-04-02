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


def add_arrow(ax1, ax2, label="", kind="straight", eps=0):
    fig = ax1.figure

    if kind == "straight":
        fig.patches.append(
            FancyArrowPatch(
                fig.transFigure.inverted().transform(
                    ax1.transData.transform((-0.9, 0))
                ),
                fig.transFigure.inverted().transform(
                    ax2.transData.transform((0.9 - eps, 0))
                ),
                transform=fig.transFigure,
                fc="k",
                arrowstyle="-|>",
                alpha=1,
                mutation_scale=10.0,
            )
        )
        ax1.text(
            -0.3,
            0.65,
            label,
            transform=ax1.transAxes,
            size=15,
            weight="bold",
            ha="center",
            va="center",
        )
    else:
        if kind == "bar_up":
            ytext = 1.15
            sgn = -1
        else:
            ytext = -0.15
            sgn = 1
        fig.patches.append(
            FancyArrowPatch(
                fig.transFigure.inverted().transform(
                    ax1.transData.transform((-0.9, 0))
                ),
                fig.transFigure.inverted().transform(
                    ax2.transData.transform((0.9 - eps, 0))
                ),
                transform=fig.transFigure,
                fc="k",
                arrowstyle="-|>",
                connectionstyle="bar,angle=90,fraction={}".format(sgn * 0.2),
                alpha=1,
                mutation_scale=10.0,
            )
        )
        ax1.text(
            -0.18,
            ytext,
            label,
            transform=ax1.transAxes,
            size=15,
            weight="bold",
            ha="center",
            va="center",
        )


def add_basis(ax, label=""):
    ax.text(
        0.5, 1.2, label, transform=ax.transAxes, size=14, ha="center", va="center",
    )


def add_frame(ax, label=""):
    ax.text(
        0.5, -0.2, label, transform=ax.transAxes, size=14, ha="center", va="center",
    )


def add_occultor(ax, xo, yo, ro):
    ax.add_artist(
        plt.Circle(
            (xo, yo), ro, fc=(0.5, 0.5, 0.5, 1), ec="k", lw=1, clip_on=False, zorder=100
        )
    )


def add_terminator(ax, b, theta):
    x = np.linspace(-1, 1, 1000)
    y = b * np.sqrt(1 - x ** 2)
    x_t = x * np.cos(theta) - y * np.sin(theta)
    y_t = x * np.sin(theta) + y * np.cos(theta)
    ax.plot(x_t, y_t, "k--", lw=1)


# Params
ydeg = 30
res = 300
inc = 60
obl = -23.5
theta = 60
xs = -1
ys = 1
zs = 0.5
xo = 0.4
yo = 0.5
ro = 0.5

# Rotated params
b = -zs / np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
theta = np.arctan2(xo, yo) - np.arctan2(xs, ys)
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
earth = starry.Map(ydeg)
earth.load("earth", sigma=0.05)
illum = starry.Map(ydeg, reflected=True)

# Render all the images we'll need
img0 = earth.render(res=res)
earth.inc = inc
earth.obl = obl
imgsky = earth.render(theta=theta, res=res)
fsky = illum.render(xs=xs, ys=ys, zs=zs, res=res)
fsky /= np.nanmax(fsky)
earth.obl = obl_int
imgint = earth.render(theta=theta, res=res)
fint = illum.render(xs=xs_int, ys=ys_int, zs=zs, res=res)
fint /= np.nanmax(fint)
earth.obl = obl_greens
imggreens = earth.render(theta=theta, res=res)
fgreens = illum.render(xs=xs_greens, ys=ys_greens, zs=zs, res=res)
fgreens /= np.nanmax(fgreens)

# Figure setup
fig = plt.figure(figsize=(16, 5))
axes = []
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

# First two
ax = plt.subplot2grid((3, 7), (1, 6))
ax.imshow(img0, **img_kw)
plot_grid(ax, 90, 0, 0)
add_basis(ax, r"$\tilde{y}$")
add_frame(ax, r"$\mathcal{F}_0$")
axes.append(ax)

ax = plt.subplot2grid((3, 7), (1, 5))
ax.imshow(imgsky, **img_kw)
plot_grid(ax, inc, obl, theta)
add_basis(ax, r"$\tilde{y}$")
add_frame(ax, r"$\mathcal{F}_\mathrm{sky}$")
add_terminator(ax, b, -np.arctan(ys / xs) * 180 / np.pi)
axes.append(ax)

# Top row
ax = plt.subplot2grid((3, 7), (0, 4))
ax.imshow(imgint, **img_kw)
plot_grid(ax, inc, obl_int, theta)
add_basis(ax, r"$\tilde{y}$")
add_frame(ax, r"$\mathcal{F}_\mathrm{poly}$")
add_terminator(ax, b, 0)
axes.append(ax)

ax = plt.subplot2grid((3, 7), (0, 3))
ax.imshow(imgint, **img_kw)
plot_grid(ax, inc, obl_int, theta)
add_basis(ax, r"$\tilde{p}$")
add_frame(ax, r"$\mathcal{F}_\mathrm{poly}$")
add_terminator(ax, b, 0)
axes.append(ax)

ax = plt.subplot2grid((3, 7), (0, 2))
ax.imshow(imgint, **img_kw)
ax.imshow(fint, **flt_kw)
plot_grid(ax, inc, obl_int, theta)
add_basis(ax, r"$\tilde{p}$")
add_frame(ax, r"$\mathcal{F}_\mathrm{poly}$")
add_terminator(ax, b, 0)
axes.append(ax)

ax = plt.subplot2grid((3, 7), (0, 1))
ax.text(
    0.65, 0.5, r"$f$", transform=ax.transAxes, size=14, ha="center", va="center",
)
ax.imshow(imgint, alpha=0, **img_kw)
axes.append(ax)

# Bottom row
ax = plt.subplot2grid((3, 7), (2, 4))
ax.imshow(imggreens, **img_kw)
plot_grid(ax, inc, obl_greens, theta)
add_basis(ax, r"$\tilde{y}$")
add_frame(ax, r"$\mathcal{F}_\mathrm{greens}$")
add_terminator(ax, b, theta)
add_occultor(ax, 0, bo, ro)
axes.append(ax)

ax = plt.subplot2grid((3, 7), (2, 3))
ax.imshow(imggreens, **img_kw)
plot_grid(ax, inc, obl_greens, theta)
add_basis(ax, r"$\tilde{p}$")
add_frame(ax, r"$\mathcal{F}_\mathrm{greens}$")
add_terminator(ax, b, theta)
add_occultor(ax, 0, bo, ro)
axes.append(ax)

ax = plt.subplot2grid((3, 7), (2, 2))
ax.imshow(imggreens, **img_kw)
ax.imshow(fgreens, **flt_kw)
plot_grid(ax, inc, obl_greens, theta)
add_basis(ax, r"$\tilde{p}$")
add_frame(ax, r"$\mathcal{F}_\mathrm{greens}$")
add_terminator(ax, b, theta)
add_occultor(ax, 0, bo, ro)
axes.append(ax)

ax = plt.subplot2grid((3, 7), (2, 1))
ax.imshow(imggreens, **img_kw)
ax.imshow(fgreens, **flt_kw)
plot_grid(ax, inc, obl_greens, theta)
add_basis(ax, r"$\tilde{g}$")
add_frame(ax, r"$\mathcal{F}_\mathrm{greens}$")
add_terminator(ax, b, theta)
add_occultor(ax, 0, bo, ro)
axes.append(ax)

ax = plt.subplot2grid((3, 7), (2, 0))
ax.text(
    0.65, 0.5, r"$f$", transform=ax.transAxes, size=14, ha="center", va="center",
)
ax.imshow(imggreens, alpha=0, **img_kw)
axes.append(ax)

# Indicate the transformations between each image
add_arrow(axes[0], axes[1], r"$\mathbf{R}$")
add_arrow(axes[1], axes[2], r"$\mathbf{R}''$", kind="bar_up")
add_arrow(axes[2], axes[3], r"$\mathbf{A_1}$")
add_arrow(axes[3], axes[4], r"$\mathbf{I}$")
add_arrow(axes[4], axes[5], r"$\mathbf{r}^\top$", eps=0.2)
add_arrow(axes[1], axes[6], r"$\mathbf{R}'$", kind="bar_down")
add_arrow(axes[6], axes[7], r"$\mathbf{A_1}$")
add_arrow(axes[7], axes[8], r"$\mathbf{I}$")
add_arrow(axes[8], axes[9], r"$\mathbf{A_2}$")
add_arrow(axes[9], axes[10], r"$\mathbf{s}^\top$", eps=0.2)

# Appearance
for ax in axes:
    ax.set_rasterization_zorder(0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")


# Save
fig.savefig("frames.pdf", bbox_inches="tight", dpi=300)
