import matplotlib.pyplot as plt
import numpy as np
import starry

# Config
starry.config.lazy = False
np.random.seed(0)

# Setup figure
fig, ax = plt.subplots(5, 2, figsize=(13, 14))
fig.subplots_adjust(hspace=0.4)
ax[0, 0].set_visible(False)
ax[0, 1].set_visible(False)
ax[1, 0].set_ylabel("flux [normalized]", fontsize=10)
ax[1, 0].set_title("thermal phase curve", fontsize=12, fontweight="bold")
ax[1, 1].set_title("reflected phase curve", fontsize=12, fontweight="bold")
ax[2, 0].set_ylabel("flux [normalized]", fontsize=10)
ax[3, 0].set_title("thermal occultation", fontsize=12, fontweight="bold")
ax[3, 1].set_title("reflected occultation", fontsize=12, fontweight="bold")
for axis in np.append(ax[1, :], ax[3, :]):
    axis.tick_params(labelsize=8)
    axis.set_ylim(-0.05, 1.65)

# Show the true map
ax_top = fig.add_subplot(5, 1, 1)
ax_top.set_title("input", fontsize=12, fontweight="bold")
map = starry.Map(20)
map.load("earth", sigma=0.1)
map.show(ax=ax_top, projection="moll")

# Solve the least-squares problem for thermal & reflected phase curves
for j, rmoon in enumerate([0, 0.25]):
    for i, reflected in enumerate([False, True]):

        # Instantiate a map of the Earth
        map = starry.Map(20, reflected=reflected)
        map.load("earth", sigma=0.1)
        obl = 23.5
        porb = 365.25
        prot = 1.0
        pmoon = 1.37
        amoon = 2.0
        imoon = 85 * np.pi / 180

        # Hack: these amplitudes give us unit median flux
        if reflected:
            map.amp = 1.0 / 0.123
        else:
            map.amp = 1.0

        # Light curve over 10 days wiith 7 occultations
        time = np.linspace(120.0, 130.0, 325)
        for n in range(7):
            time = np.append(time, (88.25 + n) * pmoon + np.linspace(-0.15, 0.15, 25))
        time = np.sort(time)

        # Starry settings
        map.obl = obl
        kwargs = {}
        kwargs["theta"] = ((time % prot) / prot) * 360
        phi = ((time % pmoon) / pmoon) * 2 * np.pi
        kwargs["xo"] = amoon * np.sin(imoon) * np.cos(phi)
        kwargs["yo"] = amoon * np.cos(imoon) * np.cos(phi)
        kwargs["zo"] = amoon * np.sin(phi)
        kwargs["ro"] = rmoon
        if reflected:
            kwargs["xs"] = np.sin(2 * np.pi * time / porb)
            kwargs["ys"] = 0
            kwargs["zs"] = -np.cos(2 * np.pi * time / porb)

        # Generate a fake dataset
        flux0 = map.flux(**kwargs)
        ferr = 1e-6
        flux = flux0 + ferr * np.random.randn(len(flux0))

        # Clear the coefficients
        map.reset(obl=obl)

        # Solve the linear problem
        mu = np.zeros(map.Ny)
        if reflected:
            mu[0] = 1.0 / 0.123
        else:
            mu[0] = 1.0
        map.set_data(flux, C=ferr ** 2)
        L = np.zeros(map.Ny)
        L[0] = 1e-1
        L[1:] = 1e-3
        map.set_prior(mu=mu, L=L)
        map.solve(**kwargs)

        # Plot the light curve and the solution
        ax[1 + 2 * j, i].plot(
            time, flux, "k.", ms=1, alpha=0.5, label="data", zorder=-2
        )
        model = map.flux(**kwargs)
        ax[1 + 2 * j, i].plot(
            time,
            map.flux(**kwargs),
            "C0-",
            lw=0.5,
            alpha=0.25,
            label="model",
            zorder=-1,
        )
        ax[1 + 2 * j, i].legend(loc="best", fontsize=6)
        if reflected:
            map.show(ax=ax[2 + 2 * j, i], projection="moll", illuminate=False)
        else:
            map.show(ax=ax[2 + 2 * j, i], projection="moll")
        ax[1 + 2 * j, i].set_xlabel("time [days]", fontsize=10)
        ax[1 + 2 * j, i].set_rasterization_zorder(0)

        # Mark the occultations
        if rmoon > 0:
            for n in range(7):
                ax[1 + 2 * j, i].axvspan(
                    (88.25 + n) * pmoon - 0.15,
                    (88.25 + n) * pmoon + 0.15,
                    facecolor="C1",
                    edgecolor="none",
                    alpha=0.1,
                )

fig.savefig("inference_occ.pdf", bbox_inches="tight", dpi=300)
