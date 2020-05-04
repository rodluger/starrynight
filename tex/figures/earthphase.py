"""Earth phase curve example."""
import starry
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Disable lazy mode
starry.config.lazy = False


# Instantiate the Earth in reflected light
map = starry.Map(25, reflected=True)
map.load("earth", sigma=0.06)
map.obl = 23.5

# Earth colormap
cmap = LinearSegmentedColormap(
    "earth",
    segmentdata={
        "red": [[0.0, 0.0, 0.122], [1.0, 1.0, 1.0]],
        "green": [[0.0, 0.0, 0.467], [1.0, 1.0, 1.0]],
        "blue": [[0.0, 0.0, 0.706], [1.0, 1.0, 1.0]],
    },
    N=256,
)

# Set up the plot
nim = 12
npts = 5000
nptsnum = 10
res = 300
fig = plt.figure(figsize=(12, 6.5))
ax_im = [plt.subplot2grid((9, nim), (0, n), rowspan=2) for n in range(nim)]
ax_in = [plt.subplot2grid((9, nim), (2, n)) for n in range(nim)]
ax_lc = plt.subplot2grid((9, nim), (3, 0), colspan=nim, rowspan=6)

# Orbital stuff
Porb = 365.256
Prot = 1.0
time = np.linspace(0, Porb, npts)
theta = (360.0 * time / Prot) % 360.0
xs = np.sin(2 * np.pi * time / Porb)
ys = np.zeros_like(time)
zs = -np.cos(2 * np.pi * time / Porb)

# Compute and plot the flux *analytically*
F = map.flux(theta=theta, xs=xs, ys=ys, zs=zs)
maxF = np.max(F)
F /= maxF
ax_lc.plot(time, F, "C0-", label="analytic", lw=0.5)

# Plot the earth images & compute the numerical flux
t_num = np.zeros(nim)
F_num = np.zeros(nim)
for n in range(nim):
    i = int(np.linspace(0, npts - 1, nim + 2)[n + 1])

    # Show the image at zero rotational phase
    map.show(
        ax=ax_im[n],
        cmap=cmap,
        xs=xs[i],
        ys=ys[i],
        zs=zs[i],
        theta=0,
        res=res,
        grid=False,
    )

    # Outline
    x = np.linspace(-1, 1, 1000)
    y = np.sqrt(1 - x ** 2)
    f = 0.98
    ax_im[n].plot(f * x, f * y, "k-", lw=0.5, zorder=0)
    ax_im[n].plot(f * x, -f * y, "k-", lw=0.5, zorder=0)

    # Appearance
    ax_im[n].axis("off")
    ax_im[n].set_xlim(-1.05, 1.05)
    ax_im[n].set_ylim(-1.05, 1.05)
    ax_im[n].set_rasterization_zorder(0)

    # Inset view of the phase curve: analytic
    theta_in = np.linspace(0, 360, 1000)
    F = map.flux(theta=theta_in, xs=xs[i], ys=ys[i], zs=zs[i])
    maxF = np.max(F)
    F /= maxF
    ax_in[n].plot(theta_in, F, "C0-", lw=1)

    # Inset view of the phase curve: numerical
    theta_in = np.linspace(0, 360, 10)
    x, y, z = map.ops.compute_ortho_grid(res)
    img = map.render(theta=theta_in, xs=xs[i], ys=ys[i], zs=zs[i], res=res)
    F = np.nansum(img, axis=(1, 2)) * 4 / res ** 2
    F /= maxF
    ax_in[n].plot(theta_in, F, "C1o", ms=1)
    ax_in[n].set_ylim(-0.3, 1.1)
    ax_in[n].axis("off")

# Appearance
ax_im[-1].set_zorder(-100)
ax_lc.set_xlabel("time [days]", fontsize=16)
ax_lc.set_ylabel("normalized flux", fontsize=16)
for tick in ax_lc.get_xticklabels() + ax_lc.get_yticklabels():
    tick.set_fontsize(14)

# Save
fig.savefig("earthphase.pdf", bbox_inches="tight", dpi=500)
