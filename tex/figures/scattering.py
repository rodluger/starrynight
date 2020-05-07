import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


plt.switch_backend("MacOSX")


class Arrow3D(FancyArrowPatch):
    """
    From https://stackoverflow.com/a/22867877.
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        self.shrinkA = 0
        self.shrinkB = 0

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def add_arrow(ax, u, v, label=None, offset=0.1, **kwargs):
    kwargs["mutation_scale"] = kwargs.get("mutation_scale", 10)
    kwargs["lw"] = kwargs.get("lw", 1)
    kwargs["arrowstyle"] = kwargs.get("arrowstyle", "-|>")
    kwargs["color"] = kwargs.get("color", "k")
    a = Arrow3D([u[0], v[0]], [u[1], v[1]], [u[2], v[2]], **kwargs)
    ax.add_artist(a)

    if label is not None:
        ax.text(
            v[0] * (1 + offset),
            v[1] * (1 + offset),
            v[2] * (1 + offset),
            label,
            transform=ax.transData,
            fontsize=10,
            va="center",
            ha="center",
            clip_on=False,
        )

    return a


def add_projection(ax, u, v, **kwargs):
    kwargs["lw"] = kwargs.get("lw", 0.75)
    kwargs["ls"] = kwargs.get("ls", "--")
    kwargs["color"] = kwargs.get("color", "k")
    kwargs["alpha"] = kwargs.get("alpha", 0.5)
    (l1,) = ax.plot([u[0], v[0]], [u[1], v[1]], [0, 0], **kwargs)
    (l2,) = ax.plot([v[0], v[0]], [v[1], v[1]], [0, v[2]], **kwargs)
    return l1, l2


def normalize(v):
    v = np.array(v, dtype=float)
    return v / np.sqrt(np.sum(v ** 2))


def show_xy_angle(ax, angle, label, r=0.25):
    x = np.linspace(0, 1, 1000)
    ax.plot(r * np.cos(x * angle), r * np.sin(x * angle), "k-", lw=0.75, alpha=0.5)
    ax.text(
        1.5 * r * np.cos(0.5 * angle),
        1.5 * r * np.sin(0.5 * angle),
        0,
        label,
        transform=ax.transData,
        fontsize=10,
        va="center",
        ha="center",
        clip_on=False,
        alpha=0.5,
    )


def show_z_angle(ax, u, label, r=0.5):
    x = np.linspace(0, 1, 1000)
    f = 1 - 0.5 * (x - 1) * x
    fh = 1 - 0.5 * (0.5 - 1) * 0.5
    ax.plot(
        r * (u[0] + x * (zhat[0] - u[0])) * f,
        r * (u[1] + x * (zhat[1] - u[1])) * f,
        r * (u[2] + x * (zhat[2] - u[2])) * f,
        "k-",
        lw=0.75,
        alpha=0.5,
    )
    ax.text(
        1.25 * r * (u[0] + 0.5 * (zhat[0] - u[0])) * fh,
        1.25 * r * (u[1] + 0.5 * (zhat[1] - u[1])) * fh,
        1.25 * r * (u[2] + 0.5 * (zhat[2] - u[2])) * fh,
        label,
        transform=ax.transData,
        fontsize=10,
        va="center",
        ha="center",
        clip_on=False,
        alpha=0.5,
    )


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=20, azim=20)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.axis("off")

# Draw axes
origin = [0, 0, 0]
xhat = [1, 0, 0]
yhat = [0, 1, 0]
zhat = [0, 0, 1]
add_arrow(ax, origin, xhat, label=r"$\widehat{x}$")
add_arrow(ax, origin, yhat, label=r"$\widehat{y}$")
add_arrow(ax, origin, zhat, label=r"$\widehat{z}$")

# Draw surface
x, y = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.5, 0.5, 5))
z = 0 * x
ax.plot_surface(x, y, z, color="C0", alpha=0.2)

# Draw source vector
s = normalize([0.1, -0.4, 0.4])
add_arrow(ax, s, origin)
add_arrow(ax, origin, s, alpha=0, label=r"$\mathbf{s}$")
add_projection(ax, origin, s)

# Draw observer vector
v = normalize([-0.8, 0.6, 0.4])
add_arrow(ax, origin, v, label=r"$\mathbf{v}$")
add_projection(ax, origin, v)

# Draw angles in the xy plane
phi_i = np.arctan2(s[1], s[0])
phi_r = np.arctan2(v[1], v[0])
show_xy_angle(ax, phi_i, r"$-\phi_i$", 0.25)
show_xy_angle(ax, phi_r, r"$\phi_r$", 0.35)

# Draw angles b/w normal vector and s, v
show_z_angle(ax, s, r"$\vartheta_i$", 0.5)
show_z_angle(ax, v, r"$\vartheta_r$", 0.4)

plt.draw()
bbox = fig.bbox_inches.from_bounds(2, 2.4, 2.6, 2.1)
fig.savefig("scattering.pdf", bbox_inches=bbox)
