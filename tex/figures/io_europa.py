"""Europa-Io occultation from the PHEMU campaign."""
import numpy as np
from matplotlib import pyplot as plt
from astropy.time import Time
import astropy.units as u
from astropy.timeseries import TimeSeries
from astroquery.jplhorizons import Horizons
import os
import starry


starry.config.lazy = False


def get_body_ephemeris(times, body_id="501", step="1m"):

    start = times.isot[0]

    # Because Horizons time range doesn't include the endpoint
    # we need to add some extra time
    if step[-1] == "m":
        padding = 2 * float(step[:-1]) / (60 * 24)
    elif step[-1] == "h":
        padding = 2 * float(step[:-1]) / 24
    elif step[-1] == "d":
        padding = 2 * float(step[:-1])
    else:
        raise ValueError(
            "Unrecognized JPL Horizons step size. Use '1m' or '1h' for example."
        )
    end = Time(times.mjd[-1] + padding, format="mjd").isot

    # Query JPL Horizons
    epochs = {"start": start, "stop": end, "step": step}
    obj = Horizons(id=body_id, epochs=epochs, id_type="id")
    eph = obj.ephemerides(extra_precision=True)
    times_jpl = Time(eph["datetime_jd"], format="jd")

    # Store all data in a TimeSeries object
    data = TimeSeries(time=times)
    data["RA"] = np.interp(times.mjd, times_jpl.mjd, eph["RA"]) * eph["RA"].unit
    data["DEC"] = np.interp(times.mjd, times_jpl.mjd, eph["DEC"]) * eph["DEC"].unit
    data["ang_width"] = (
        np.interp(times.mjd, times_jpl.mjd, eph["ang_width"]) * eph["ang_width"].unit
    )
    data["phase_angle"] = (
        np.interp(times.mjd, times_jpl.mjd, eph["alpha_true"]) * eph["alpha_true"].unit
    )
    eph = obj.ephemerides(extra_precision=True)

    # Boolean flags for occultations/eclipses
    occ_sunlight = eph["sat_vis"] == "O"
    umbra = eph["sat_vis"] == "u"
    occ_umbra = eph["sat_vis"] == "U"
    partial = eph["sat_vis"] == "p"
    occ_partial = eph["sat_vis"] == "P"
    occulted = np.any([occ_umbra, occ_sunlight], axis=0)

    data["ecl_par"] = np.array(
        np.interp(times.mjd, times_jpl.mjd, partial), dtype=bool,
    )
    data["ecl_tot"] = np.array(np.interp(times.mjd, times_jpl.mjd, umbra), dtype=bool,)
    data["occ_umbra"] = np.array(
        np.interp(times.mjd, times_jpl.mjd, occ_umbra), dtype=bool,
    )
    data["occ_sun"] = np.array(
        np.interp(times.mjd, times_jpl.mjd, occ_sunlight), dtype=bool,
    )

    # Helper functions for dealing with angles and discontinuities
    subtract_angles = lambda x, y: np.fmod((x - y) + np.pi * 3, 2 * np.pi) - np.pi

    def interpolate_angle(x, xp, yp):
        """
        Interpolate an angular quantity on domain [-pi, pi) and avoid 
        discountinuities.

        """
        cosy = np.interp(x, xp, np.cos(yp))
        siny = np.interp(x, xp, np.sin(yp))

        return np.arctan2(siny, cosy)

    # Inclination of the starry map = 90 - latitude of the central point of
    # the observed disc
    data["inc"] = interpolate_angle(
        times.mjd, times_jpl.mjd, np.pi / 2 * u.rad - eph["PDObsLat"].to(u.rad),
    ).to(u.deg)

    # Rotational phase of the starry map is the observer longitude
    data["theta"] = (
        interpolate_angle(
            times.mjd, times_jpl.mjd, eph["PDObsLon"].to(u.rad) - np.pi * u.rad,
        ).to(u.deg)
    ) + 180 * u.deg

    # Obliquity of the starry map is the CCW angle from the celestial
    # NP to the NP of the target body
    data["obl"] = interpolate_angle(
        times.mjd, times_jpl.mjd, eph["NPole_ang"].to(u.rad),
    ).to(u.deg)

    # Compute the location of the subsolar point relative to the central
    # point of the disc
    lon_subsolar = subtract_angles(
        np.array(eph["PDSunLon"].to(u.rad)), np.array(eph["PDObsLon"].to(u.rad)),
    )
    lon_subsolar = 2 * np.pi - lon_subsolar  # positive lon. is to the east

    lat_subsolar = subtract_angles(
        np.array(eph["PDSunLat"].to(u.rad)), np.array(eph["PDObsLat"].to(u.rad)),
    )

    # Location of the subsolar point in cartesian Starry coordinates
    xs = np.array(eph["r"]) * np.cos(lat_subsolar) * np.sin(lon_subsolar)
    ys = np.array(eph["r"]) * np.sin(lat_subsolar)
    zs = np.array(eph["r"]) * np.cos(lat_subsolar) * np.cos(lon_subsolar)

    data["xs"] = np.interp(times.mjd, times_jpl.mjd, xs) * u.AU
    data["ys"] = np.interp(times.mjd, times_jpl.mjd, ys) * u.AU
    data["zs"] = np.interp(times.mjd, times_jpl.mjd, zs) * u.AU

    return data


def get_phemu_data(file="data/G20091204_2o1_JHS_0.txt"):
    y, m, d = file[6:10], file[10:12], file[12:14]
    date_mjd = Time(f"{y}-{m}-{d}", format="isot", scale="utc").to_value("mjd")
    data = np.genfromtxt(file)
    times_mjd = date_mjd + data[:, 0] / (60 * 24)
    time, flux = np.vstack([times_mjd, data[:, 1]])
    return Time(time, format="mjd"), flux


def get_starry_args(time):
    # Ephemeris
    eph_io = get_body_ephemeris(time, step="1m")
    eph_europa = get_body_ephemeris(time, body_id="502", step="1m",)

    # Get occultation parameters
    obl = np.mean(eph_io["obl"].value)
    inc = np.mean(eph_io["inc"].value)
    theta = np.mean(eph_io["theta"].value)
    ro = np.mean((eph_europa["ang_width"] / eph_io["ang_width"]).value)
    rel_ra = (eph_europa["RA"] - eph_io["RA"]).to(u.arcsec) / (
        0.5 * eph_io["ang_width"].to(u.arcsec)
    )
    rel_dec = (eph_europa["DEC"] - eph_io["DEC"]).to(u.arcsec) / (
        0.5 * eph_io["ang_width"].to(u.arcsec)
    )
    xo = -rel_ra.value
    yo = rel_dec.value
    xs = np.mean(eph_io["xs"].value)
    ys = np.mean(eph_io["ys"].value)
    zs = np.mean(eph_io["zs"].value)

    return inc, obl, dict(theta=theta, xo=xo, yo=yo, ro=ro, xs=xs, ys=ys, zs=zs)


# Grab the PHEMU light curve
time, flux = get_phemu_data()

# Instantiate a starry map & get geometrical parameters
map = starry.Map(ydeg=25, reflected=True)
map.inc, map.obl, kwargs = get_starry_args(time)

# Load the Galileo SSI / Voyager composite
# https://astrogeology.usgs.gov/search/map/Io/
# Voyager-Galileo/Io_GalileoSSI-Voyager_Global_Mosaic_1km
map.load("data/io_mosaic.jpg")

# View it
map.show(projection="moll", colorbar=True, illuminate=False, file="io.pdf")

# Model
model = map.flux(**kwargs)
model /= model[0]

# Plot it
fig, ax = plt.subplots(1, figsize=(14, 5))
ax.plot(time.value - 55169, flux, "k.", alpha=0.75, ms=4, label="data")
ax.plot(time.value - 55169, model, label="model")
ax.legend()
ax.set_xlabel("time [MJD - 55169]")
ax.set_ylabel("normalized flux")
fig.savefig("io_europa.pdf", bbox_inches="tight")
