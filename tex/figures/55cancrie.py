"""55 cancrie e secondary eclipse example."""
import starry
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

# Config
starry.config.lazy = False

# Star
star_map = starry.Map(udeg=2, amp=1)
star_map[1:] = [0.5, 0.25]
star = starry.Primary(
    star_map, m=0.905, mass_unit=u.M_sun, r=0.943, length_unit=u.R_sun
)

# Planet
kwargs = dict(
    m=7.99,
    mass_unit=u.earthMass,
    porb=0.73654737,
    r=1.875,
    length_unit=u.earthRad,
    inc=83.59,
    t0=0.400473685,
)

# Time arrays (secondary eclipse ingress / full phase curve)
t_ingress = np.linspace(0, 0.002, 1000)
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
e_map = starry.Map(ydeg=1, reflected=True, amp=1)
e = starry.Secondary(e_map, **kwargs)
sys = starry.System(star, e)
flux_ref = sys.flux(t=t_ingress, total=False)[1]
flux_phase = sys.flux(t=t_phase, total=False)[1]

# Reflected, extended source
e_map = starry.Map(ydeg=1, reflected=True, amp=1, source_npts=300)
e = starry.Secondary(e_map, **kwargs)
sys = starry.System(star, e)
flux_ref_fin = sys.flux(t=t_ingress, total=False)[1]
flux_phase_fin = sys.flux(t=t_phase, total=False)[1]

# Plot phase curves
fig, ax = plt.subplots(1, figsize=(8, 5))
t = t_phase - kwargs["t0"]
ax.plot(t, flux_phase * 1e6, "C0-", label="point source")
ax.plot(t, flux_phase_fin * 1e6, "C1-", label="extended source")
ax.set_xlabel("time [days]", fontsize=16)
ax.set_ylabel("flux [ppm]", fontsize=16)
ax.legend(loc="best")
fig.savefig("55cancrie.pdf", bbox_inches="tight")

# Plot eclipse ingress
fig, ax = plt.subplots(1, figsize=(8, 5))
t = (t_ingress * u.day).to(u.minute)
ax.plot(t, flux_em / flux_em[0], "C2-", label="uniform")
ax.plot(t, flux_ld / flux_ld[0], "C3-", label="noon")
ax.plot(t, flux_ref / flux_ref[0], "C0-", label="point source")
ax.plot(t, flux_ref_fin / flux_ref_fin[0], "C1-", label="extended source")
ax.set_xlabel("time [minutes]", fontsize=16)
ax.set_ylabel("flux [relative]", fontsize=16)
ax.legend(loc="best")
fig.savefig("55cancrie_ingress.pdf", bbox_inches="tight")
