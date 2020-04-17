"""55 cancrie e secondary eclipse example."""
import starry
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

# Config
starry.config.lazy = False
# plt.switch_backend("MacOSX")

# Time array
t = np.linspace(0, 0.002, 1000)
t0 = 0.400473685  # ingress start

# Approximate relative luminosity of the planet
tstar = 5172
te = 2709
rprs = 0.0182
e_amp = rprs ** 2 * (te / tstar) ** 4

# Star
star_map = starry.Map(udeg=2, amp=1)
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
    t0=t0,
)

e_map = starry.Map(ydeg=1, reflected=False, amp=e_amp)
e = starry.Secondary(e_map, **kwargs)
flux_em = starry.System(star, e).flux(t=t)
flux_em -= flux_em[-1]
flux_em /= flux_em[0]

e_map = starry.Map(udeg=1, reflected=False, amp=e_amp)
e_map[1] = 1
e = starry.Secondary(e_map, **kwargs)
flux_ld = starry.System(star, e).flux(t=t)
flux_ld -= flux_ld[-1]
flux_ld /= flux_ld[0]

e_map = starry.Map(ydeg=1, reflected=True, amp=e_amp)
e = starry.Secondary(e_map, **kwargs)
flux_ref = starry.System(star, e).flux(t=t)
flux_ref -= flux_ref[-1]
flux_ref /= flux_ref[0]


e_map = starry.Map(ydeg=1, reflected=True, amp=e_amp)
e = starry.Secondary(e_map, **kwargs)
flux_ref = starry.System(star, e).flux(t=t)
flux_ref -= flux_ref[-1]
flux_ref /= flux_ref[0]


# Plot the fluxes
tmin = (t * u.day).to(u.minute)
plt.plot(tmin, flux_em)
plt.plot(tmin, flux_ld)
plt.plot(tmin, flux_ref)

# plt.show()
