"""Kelt-9b secondary eclipse example."""
import starry
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u


# Config
starry.config.lazy = False


# Star
star_map = starry.Map(udeg=2, amp=1)
star_map[1:] = [0.5, 0.25]
star = starry.Primary(star_map, m=2.50, mass_unit=u.M_sun, r=2.36, length_unit=u.R_sun)

# Planet
kwargs = dict(
    m=0.0,
    mass_unit=u.earthMass,
    porb=1.481,
    r=21.1770214,
    length_unit=u.earthRad,
    inc=87.2,
    t0=-0.65625,
)

# Time arrays (secondary eclipse ingress / full phase curve)
t_ingress = np.linspace(0, 0.017, 1000)
t_egress = 218 / 60 / 24 + t_ingress
t_sec = np.append(t_ingress, t_egress)
t_phase = kwargs["t0"] + kwargs["porb"] * np.linspace(-0.5, 0.5, 1000)

# Uniform
e_map = starry.Map(ydeg=1, reflected=False)
e = starry.Secondary(e_map, **kwargs)
sys = starry.System(star, e)
flux_em = sys.flux(t=t_sec, total=False)[1]

# Noon approximation
e_map = starry.Map(udeg=1, reflected=False)
e_map[1] = 1
e = starry.Secondary(e_map, **kwargs)
sys = starry.System(star, e)
flux_ld = sys.flux(t=t_sec, total=False)[1]

# Reflected, point source
e_map = starry.Map(ydeg=1, reflected=True)
e_map.amp *= 0.2
e = starry.Secondary(e_map, **kwargs)
sys = starry.System(star, e)
flux_ref = sys.flux(t=t_sec, total=False)[1]
flux_phase = sys.flux(t=t_phase, total=False)[1]

# Reflected, extended source
e_map = starry.Map(ydeg=1, reflected=True, source_npts=300)
e_map.amp *= 0.2
e = starry.Secondary(e_map, **kwargs)
sys = starry.System(star, e)
flux_ref_fin = sys.flux(t=t_sec, total=False)[1]
flux_phase_fin = sys.flux(t=t_phase, total=False)[1]

# Plot phase curves
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
fig.subplots_adjust(hspace=0.01)
t = t_phase - kwargs["t0"]
phase = t / kwargs["porb"]
ax[0].plot(phase, flux_phase * 1e6, "C0-", label="point source")
ax[0].plot(phase, flux_phase_fin * 1e6, "C4-", label="extended source")
ax[0].set_xlabel("phase", fontsize=18)
ax[0].set_ylabel("flux [ppm]", fontsize=18)
ax[0].xaxis.labelpad = 10
ax[0].yaxis.labelpad = 10
ax[0].tick_params(axis="both", which="major", labelsize=13)
ax[0].legend(loc="best", fontsize=12)
ax[1].plot(t, flux_phase / np.max(flux_phase), "C0-", label="point source")
ax[1].plot(t, flux_phase_fin / np.max(flux_phase_fin), "C4-", label="extended source")
ax[1].set_xlabel("time [days]", fontsize=18)
ax[1].set_ylabel("flux [normalized]", fontsize=18)
ax[1].xaxis.labelpad = 10
ax[1].yaxis.labelpad = 10
ax[1].tick_params(axis="both", which="major", labelsize=13)
ax[1].legend(loc="best", fontsize=13)
fig.savefig("kelt9b.pdf", bbox_inches="tight")

# Plot eclipse ingress/egress
fig = plt.figure(figsize=(10, 4))
eps = 0.01
ax = [fig.add_axes([0, 0, 0.5 - eps, 1]), fig.add_axes([0.5 + eps, 0, 0.5 - eps, 1])]

# Plot ingress (time in minutes)
t = t_sec * 60 * 24
idx = t <= t_ingress[-1] * 60 * 24
tmid = 0.5 * t_egress[-1] * 60 * 24
ax[0].plot(t[idx] - tmid, flux_ref[idx] / np.max(flux_ref), "C0-", label="point source")
ax[0].plot(
    t[idx] - tmid,
    flux_ref_fin[idx] / np.max(flux_ref_fin),
    "C4-",
    label="extended source",
)
ax[0].plot(t[idx] - tmid, flux_em[idx] / np.max(flux_em), "C1-", label="uniform")
ax[0].plot(t[idx] - tmid, flux_ld[idx] / np.max(flux_ld), "C1--", label="cosine")

# Plot egress
idx = t >= t_egress[0] * 60 * 24
ax[1].plot(t[idx] - tmid, flux_ref[idx] / np.max(flux_ref), "C0-", label="point source")
ax[1].plot(
    t[idx] - tmid,
    flux_ref_fin[idx] / np.max(flux_ref_fin),
    "C4-",
    label="extended source",
)
ax[1].plot(t[idx] - tmid, flux_em[idx] / np.max(flux_em), "C1-", label="uniform")
ax[1].plot(t[idx] - tmid, flux_ld[idx] / np.max(flux_ld), "C1--", label="cosine")
ax[1].legend(loc="lower right", fontsize=13)

# Make broken axis
ax[0].set_xlim(t_ingress[0] * 60 * 24 - tmid - 1, t_ingress[-1] * 60 * 24 - tmid)
ax[1].set_xlim(t_egress[0] * 60 * 24 - tmid, t_egress[-1] * 60 * 24 - tmid + 1)
ax[0].spines["right"].set_visible(False)
ax[1].spines["left"].set_visible(False)
ax[1].set_yticks([])
d = 0.01
kwargs = dict(transform=ax[0].transAxes, color="k", clip_on=False)
ax[0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
ax[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax[1].transAxes)
ax[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax[1].plot((-d, +d), (-d, +d), **kwargs)

# Label
ax[0].set_ylabel("flux [normalized]", fontsize=17)
ax[0].set_xlabel("time [minutes]", fontsize=17)
ax[0].xaxis.set_label_coords(0.5, -0.1, transform=fig.transFigure)
ax[0].tick_params(axis="both", which="major", labelsize=14)
ax[1].tick_params(axis="both", which="major", labelsize=14)

# We're done
fig.savefig("kelt9b_eclipse.pdf", bbox_inches="tight")
