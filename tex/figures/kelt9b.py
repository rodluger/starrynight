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
t_ingress = np.linspace(0, 0.0174, 1000)
t_phase = kwargs["t0"] + kwargs["porb"] * np.linspace(-0.5, 0.5, 1000)

# Uniform
e_map = starry.Map(ydeg=1, reflected=False)
e = starry.Secondary(e_map, **kwargs)
sys = starry.System(star, e)
flux_em = sys.flux(t=t_ingress, total=False)[1]

# Noon approximation
e_map = starry.Map(udeg=1, reflected=False)
e_map[1] = 1
e = starry.Secondary(e_map, **kwargs)
sys = starry.System(star, e)
flux_ld = sys.flux(t=t_ingress, total=False)[1]

# Reflected, point source
e_map = starry.Map(ydeg=1, reflected=True)
e_map.amp *= 0.2
e = starry.Secondary(e_map, **kwargs)
sys = starry.System(star, e)
flux_ref = sys.flux(t=t_ingress, total=False)[1]
flux_phase = sys.flux(t=t_phase, total=False)[1]

# Reflected, extended source
e_map = starry.Map(ydeg=1, reflected=True, source_npts=300)
e_map.amp *= 0.2
e = starry.Secondary(e_map, **kwargs)
sys = starry.System(star, e)
flux_ref_fin = sys.flux(t=t_ingress, total=False)[1]
flux_phase_fin = sys.flux(t=t_phase, total=False)[1]

# Plot phase curves
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
fig.subplots_adjust(hspace=0.025)
t = t_phase - kwargs["t0"]
phase = t / kwargs["porb"]
ax[0].plot(t, flux_phase * 1e6, "C0-", label="point source")
ax[0].plot(t, flux_phase_fin * 1e6, "C1-", label="extended source")
ax[0].set_xlabel("time [days]", fontsize=16)
ax[0].set_ylabel("flux [ppm]", fontsize=16)
ax[0].xaxis.labelpad = 10
ax[0].yaxis.labelpad = 10
ax[0].tick_params(axis="both", which="major", labelsize=12)
ax[0].legend(loc="best", fontsize=12)

# Plot eclipse ingress
t = (t_ingress * u.day).to(u.minute)
ax[1].plot(t, flux_em / flux_em[0], "C4-", label="uniform")
ax[1].plot(t, flux_ld / flux_ld[0], "C4--", label="cosine")
ax[1].plot(t, flux_ref / flux_ref[0], "C0-", label="point source")
ax[1].plot(t, flux_ref_fin / flux_ref_fin[0], "C1-", label="extended source")
ax[1].set_xlabel("phase", fontsize=16)
ax[1].set_ylabel("flux [normalized]", fontsize=16)
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right")
ax[1].legend(loc="best", fontsize=12)
ax[1].xaxis.labelpad = 10
ax[1].yaxis.labelpad = 10
ax[1].tick_params(axis="both", which="major", labelsize=12)

fig.savefig("kelt9b.pdf", bbox_inches="tight")
