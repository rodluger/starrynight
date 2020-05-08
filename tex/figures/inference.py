import matplotlib.pyplot as plt
import numpy as np
import starry

# Config
starry.config.lazy = False
np.random.seed(0)

# Setup figure
fig, ax = plt.subplots(3, 2, figsize=(13, 7))
fig.subplots_adjust(hspace=0.4)
ax[0, 0].set_visible(False)
ax[0, 1].set_visible(False)
ax[1, 0].set_ylabel("flux", fontsize=10)
ax[1, 0].set_title("thermal", fontsize=12, fontweight="bold")
ax[1, 1].set_title("reflected", fontsize=12, fontweight="bold")

for axis in ax[1, :]:
    axis.tick_params(labelsize=8)
    axis.set_ylim(-0.05, 1.05)

# Show the true map
ax_top = fig.add_subplot(3, 1, 1)
map = starry.Map(20)
map.load("earth", sigma=0.1)
map.show(ax=ax_top, projection="moll")

# Solve the least-squares problem for thermal & reflected phase curves
for i, reflected in enumerate([False, True]):

    # Instantiate a map of the Earth
    map = starry.Map(20, reflected=reflected)
    map.load("earth", sigma=0.1)
    obl = 23.5
    porb = 365.25
    prot = 1.0

    # Hack: these amplitudes give us unit maximum flux
    if reflected:
        map.amp = 3.3958
    else:
        map.amp = 0.7560

    # Light curve over one year
    time = np.linspace(0, porb, 1000)

    # Starry settings
    map.obl = obl
    kwargs = {}
    kwargs["theta"] = (time % prot) * 360
    if reflected:
        kwargs["xs"] = np.sin(2 * np.pi * time / porb)
        kwargs["ys"] = 0
        kwargs["zs"] = -np.cos(2 * np.pi * time / porb)

    # Generate a fake dataset
    flux0 = map.flux(**kwargs)
    ferr = 1e-6
    flux = flux0 + ferr * np.random.randn(len(flux0))

    # Clear the map coefficients
    map.reset(obl=obl)

    # Solve the linear problem
    map.set_data(flux, C=ferr ** 2)
    L = np.zeros(map.Ny)
    L[0] = 1e-1
    L[1:] = 1e-3
    map.set_prior(L=L)
    map.solve(**kwargs)

    # Plot the light curve and the solution
    ax[1, i].plot(time, flux, "k.", ms=2, alpha=0.5, label="data")
    ax[1, i].plot(time, map.flux(**kwargs), "C0-", lw=0.5, alpha=0.25, label="model")
    ax[1, i].legend(loc="best", fontsize=6)
    if reflected:
        map.show(ax=ax[2, i], projection="moll", illuminate=False)
    else:
        map.show(ax=ax[2, i], projection="moll")
    ax[1, i].set_xlabel("time [days]", fontsize=10)

fig.savefig("inference.pdf", bbox_inches="tight")
