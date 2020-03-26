"""Starry speed tests."""
import starry
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import theano
import theano.tensor as tt
import theano.sparse as ts
from theano.ifelse import ifelse
from scipy.optimize import root_scalar
from tqdm import tqdm
import time
import warnings


# Config
plt.switch_backend("MacOSX")
starry.config.lazy = False
starry.config.quiet = True
# warnings.simplefilter("ignore")
HUGE = 1e30
np.random.seed(1234)


_y = tt.dvector()
_x = tt.dvector()
_xs = tt.dvector()
_ys = tt.dvector()
_zs = tt.dvector()
_xo = tt.dvector()
_yo = tt.dvector()
_ro = tt.dscalar()


class Compare(object):
    """Compare different ways of evaluating the flux."""

    def __init__(self, y):
        self.y = y

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = np.array(y)
        self.ydeg = int(np.sqrt(len(y)) - 1)
        self.map_ref = starry.Map(ydeg=self.ydeg, reflected=True)
        self.map_emi = starry.Map(ydeg=self.ydeg)
        if self.ydeg > 0:
            self.map_ref[1:, :] = y[1:]
            self.map_emi[1:, :] = y[1:]
        self._A1y = ts.dot(self.map_ref.ops.A1, tt.as_tensor_variable(y))

        # Reset
        self.intensity_occulted(
            0.0, 0.0, [0.0], [0.0], [0.0], [0.0], [0.0], 0.0, reset=True
        )
        self.intensity(0.0, 0.0, [0.0], [0.0], [0.0], reset=True)
        self.flux([0.0], [0.0], [0.0], [0.0], [0.0], 0.0, reset=True)
        self.dfluxdro([0.0], [0.0], [0.0], [0.0], [0.0], 0.0, reset=True)
        if self.ydeg > 0:
            self.flux_emitted([0.0], [0.0], [0.0], [0.0], [0.0], 0.0, reset=True)
            self.dfluxdro_emitted([0.0], [0.0], [0.0], [0.0], [0.0], 0.0, reset=True)

    def intensity_occulted(self, y, x, xs, ys, zs, xo, yo, ro, reset=False):

        if reset or not hasattr(self, "_intensity_occulted"):

            def theano_code(y, x, xs, ys, zs, xo, yo, ro):

                # Get the z coord
                z = tt.sqrt(1 - x ** 2 - y ** 2)

                # Compute the intensity
                pT = self.map_ref.ops.pT(x, y, z)

                # Weight the intensity by the illumination
                # Dot the polynomial into the basis
                intensity = tt.shape_padright(tt.dot(pT, self._A1y))

                # Weight the intensity by the illumination
                xyz = tt.concatenate(
                    (
                        tt.reshape(x, [1, -1]),
                        tt.reshape(y, [1, -1]),
                        tt.reshape(z, [1, -1]),
                    )
                )
                I = self.map_ref.ops.compute_illumination(xyz, xs, ys, zs)
                intensity = tt.switch(tt.isnan(intensity), intensity, intensity * I)[
                    0, 0
                ]

                # Check if the point is visible
                result = ifelse(
                    ((x - xo) ** 2 + (y - yo) ** 2 < ro ** 2)[0],
                    tt.as_tensor_variable(0.0).astype(tt.config.floatX),
                    ifelse(
                        (x ** 2 + y ** 2 > 1)[0],
                        tt.as_tensor_variable(0.0).astype(tt.config.floatX),
                        intensity,
                    ),
                )
                return result

            self._intensity_occulted = theano.function(
                [_y, _x, _xs, _ys, _zs, _xo, _yo, _ro],
                theano_code(_y, _x, _xs, _ys, _zs, _xo, _yo, _ro),
            )

        return self._intensity_occulted(
            np.array([y]), np.array([x]), xs, ys, zs, xo, yo, ro
        )

    def intensity(self, y, x, xs, ys, zs, reset=False):

        if reset or not hasattr(self, "_intensity"):

            def theano_code(y, x, xs, ys, zs):

                # Get the z coord (abs to prevent numerical error)
                z = tt.sqrt(tt.abs_(1 - x ** 2 - y ** 2))

                # Compute the intensity
                pT = self.map_ref.ops.pT(x, y, z)

                # Weight the intensity by the illumination
                # Dot the polynomial into the basis
                intensity = tt.shape_padright(tt.dot(pT, self._A1y))

                # Weight the intensity by the illumination
                xyz = tt.concatenate(
                    (
                        tt.reshape(x, [1, -1]),
                        tt.reshape(y, [1, -1]),
                        tt.reshape(z, [1, -1]),
                    )
                )
                I = self.map_ref.ops.compute_illumination(xyz, xs, ys, zs)
                intensity = intensity * I
                return intensity

            self._intensity = theano.function(
                [_y, _x, _xs, _ys, _zs], theano_code(_y, _x, _xs, _ys, _zs),
            )

        return self._intensity(np.array([y]), np.array([x]), xs, ys, zs)

    def flux(self, xs, ys, zs, xo, yo, ro, reset=False):

        if reset or not hasattr(self, "_flux"):

            def theano_code(xs, ys, zs, xo, yo, ro):
                theta = tt.zeros_like(xo)
                zo = tt.ones_like(xo)
                inc = tt.as_tensor_variable(np.pi / 2).astype(tt.config.floatX)
                obl = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
                u = tt.as_tensor_variable([-1.0]).astype(tt.config.floatX)
                f = tt.as_tensor_variable([np.pi]).astype(tt.config.floatX)
                alpha = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
                y = tt.as_tensor_variable(self.y).astype(tt.config.floatX)
                return self.map_ref.ops.flux(
                    theta, xs, ys, zs, xo, yo, zo, ro, inc, obl, y, u, f, alpha
                )[0]

            self._flux = theano.function(
                [_xs, _ys, _zs, _xo, _yo, _ro],
                theano_code(_xs, _ys, _zs, _xo, _yo, _ro),
            )

        return self._flux(xs, ys, zs, xo, yo, ro)

    def dfluxdro(self, xs, ys, zs, xo, yo, ro, reset=False):

        if reset or not hasattr(self, "_dfluxdro"):

            def theano_code(xs, ys, zs, xo, yo, ro):
                theta = tt.zeros_like(xo)
                zo = tt.ones_like(xo)
                inc = tt.as_tensor_variable(np.pi / 2).astype(tt.config.floatX)
                obl = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
                u = tt.as_tensor_variable([-1.0]).astype(tt.config.floatX)
                f = tt.as_tensor_variable([np.pi]).astype(tt.config.floatX)
                alpha = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
                y = tt.as_tensor_variable(self.y).astype(tt.config.floatX)
                return theano.grad(
                    self.map_ref.ops.flux(
                        theta, xs, ys, zs, xo, yo, zo, ro, inc, obl, y, u, f, alpha
                    )[0],
                    ro,
                )

            self._dfluxdro = theano.function(
                [_xs, _ys, _zs, _xo, _yo, _ro],
                theano_code(_xs, _ys, _zs, _xo, _yo, _ro),
            )

        return self._dfluxdro(xs, ys, zs, xo, yo, ro)

    def flux_emitted(self, xs, ys, zs, xo, yo, ro, reset=False):

        if reset or not hasattr(self, "_flux_emitted"):

            def theano_code(xo, yo, ro):
                theta = tt.zeros_like(xo)
                zo = tt.ones_like(xo)
                inc = tt.as_tensor_variable(np.pi / 2).astype(tt.config.floatX)
                obl = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
                u = tt.as_tensor_variable([-1.0]).astype(tt.config.floatX)
                f = tt.as_tensor_variable([np.pi]).astype(tt.config.floatX)
                alpha = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
                y = tt.as_tensor_variable(self.y).astype(tt.config.floatX)
                return self.map_emi.ops.flux(
                    theta, xo, yo, zo, ro, inc, obl, y, u, f, alpha
                )[0]

            self._flux_emitted = theano.function(
                [_xo, _yo, _ro], theano_code(_xo, _yo, _ro),
            )

        return self._flux_emitted(xo, yo, ro)

    def dfluxdro_emitted(self, xs, ys, zs, xo, yo, ro, reset=False):

        if reset or not hasattr(self, "_dfluxdro_emitted"):

            def theano_code(xo, yo, ro):
                theta = tt.zeros_like(xo)
                zo = tt.ones_like(xo)
                inc = tt.as_tensor_variable(np.pi / 2).astype(tt.config.floatX)
                obl = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
                u = tt.as_tensor_variable([-1.0]).astype(tt.config.floatX)
                f = tt.as_tensor_variable([np.pi]).astype(tt.config.floatX)
                alpha = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
                y = tt.as_tensor_variable(self.y).astype(tt.config.floatX)
                return theano.grad(
                    self.map_emi.ops.flux(
                        theta, xo, yo, zo, ro, inc, obl, y, u, f, alpha
                    )[0],
                    ro,
                )

            self._dfluxdro_emitted = theano.function(
                [_xo, _yo, _ro], theano_code(_xo, _yo, _ro),
            )

        return self._dfluxdro_emitted(xo, yo, ro)

    def flux_brute(self, xs, ys, zs, xo, yo, ro, res=999, **kwargs):
        # Compute the flux by brute force grid integration
        img = self.map_ref.render(xs=xs, ys=ys, zs=zs, res=res)
        x, y, z = self.map_ref.ops.compute_ortho_grid(res)
        idx = x ** 2 + (y - yo) ** 2 > ro ** 2
        return np.nansum(img.flat[idx]) * 4 / res ** 2

    def flux_quad(
        self,
        xs,
        ys,
        zs,
        xo,
        yo,
        ro,
        epsabs=1e-6,
        epsrel=1e-6,
        boundaries=None,
        **kwargs
    ):
        # Compute the double integral numerically
        if boundaries is not None:
            val = 0
            err = 0
            for (xa, xb, y1, y2) in boundaries:
                val_new, err_new = dblquad(
                    self.intensity,
                    xa,
                    xb,
                    y1,
                    y2,
                    epsabs=epsabs,
                    epsrel=epsrel,
                    args=(np.atleast_1d(xs), np.atleast_1d(ys), np.atleast_1d(zs)),
                )
                val += val_new
                err += err_new
        else:
            xa, xb, y1, y2 = (
                -1,
                1,
                lambda x: -np.sqrt(1 - x ** 2),
                lambda x: np.sqrt(1 - x ** 2),
            )
            val, err = dblquad(
                self.intensity_occulted,
                xa,
                xb,
                y1,
                y2,
                epsabs=epsabs,
                epsrel=epsrel,
                args=(
                    np.atleast_1d(xs),
                    np.atleast_1d(ys),
                    np.atleast_1d(zs),
                    np.atleast_1d(xo),
                    np.atleast_1d(yo),
                    ro,
                ),
            )

        return val

    def display(
        self,
        xs,
        ys,
        zs,
        xo,
        yo,
        ro,
        ax=None,
        res=999,
        cmap="gray_r",
        occultor_color="k",
        boundaries=[],
    ):
        """Render an imshow view of the map."""
        # Render
        img = self.map_ref.render(xs=xs, ys=ys, zs=zs, res=res)

        # Mask the occultor
        x, y, z = self.map_ref.ops.compute_ortho_grid(res)
        idx = x ** 2 + (y - yo) ** 2 <= ro ** 2
        img.flat[idx] = HUGE

        # Mask the night side
        img[img == 0.0] = -HUGE

        # Figure out imshow limits
        cmap = plt.get_cmap(cmap)
        cmap.set_over(occultor_color)
        cmap.set_under("k")
        vmin = np.nanmin(img[np.abs(img) < HUGE])
        vmax = np.nanmax(img[np.abs(img) < HUGE])

        # Display
        if ax is None:
            fig, ax = plt.subplots(1)
            ax.axis("off")
        ax.imshow(
            img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=(-1, 1, -1, 1),
        )
        ax.add_artist(plt.Circle((xo, yo), ro, edgecolor="w", color=occultor_color))

        # Integration boundaries
        for bound in boundaries:
            xa, xb, y1, y2 = bound
            x = np.linspace(xa, xb, 1000)
            y1 = y1(x)
            y2 = y2(x)
            ax.plot(x, y1, "r-", lw=3)
            ax.plot(x, y2, "r-", lw=3)
            ax.plot(x[-1], y1[-1], "ro")
            ax.plot(x[-1], y2[-1], "ro")
            ax.plot([x[-1], x[-1]], [y1[-1], y2[-1]], "r--", lw=1)

        return ax

    def _run(self, func, nt=1, *args, **kwargs):
        t = time.time()
        val = func(*args, **kwargs)
        return (time.time() - t) / nt, val

    def compare(
        self, xs, ys, zs, xo, yo, ro, res=999, nt=1000, boundaries=None,
    ):
        """Compare different integration schemes."""

        # Initialize
        yfull = np.array(self.y)
        teval = np.zeros((7, ydeg + 1))
        value = np.zeros((7, ydeg + 1))

        # Params for the comparison
        kwargs = dict(xs=xs, ys=ys, zs=zs, xo=xo, yo=yo, ro=ro)
        kwargs_vec = dict(kwargs)
        for key in ["xs", "ys", "zs", "xo", "yo"]:
            kwargs_vec[key] *= np.ones(nt)

        styles = [
            ("starry: reflected", "C0", "-"),
            ("starry: reflected (grad)", "C1", "-"),
            ("starry: emitted", "C2", "-"),
            ("starry: emitted (grad)", "C3", "-"),
            ("grid", "C4", "-"),
            ("dblquad (naive)", "C5", "-"),
            ("dblquad (segmented)", "C6", "-"),
        ]

        # Loop over all degrees
        for l in tqdm(range(ydeg + 1)):
            self.y = yfull[: (l + 1) ** 2]
            teval[0, l], value[0, l] = self._run(self.flux, nt, **kwargs_vec)
            teval[1, l], value[1, l] = self._run(self.dfluxdro, nt, **kwargs_vec)
            if l > 0:
                teval[2, l], value[2, l] = self._run(
                    self.flux_emitted, nt, **kwargs_vec
                )
                teval[3, l], value[3, l] = self._run(
                    self.dfluxdro_emitted, nt, **kwargs_vec
                )
            teval[4, l], value[4, l] = self._run(self.flux_brute, res=res, **kwargs)
            teval[5, l], value[5, l] = self._run(
                self.flux_quad, boundaries=None, epsabs=1e-3, epsrel=1e-3, **kwargs
            )

            # TODO
            teval[6, l], value[6, l] = np.nan, np.nan  # self._run(
            # self.flux_quad, boundaries=boundaries, **kwargs
            # )

        # Zero-degree emitted light maps are limb-darkened maps,
        # so let's just set them equal to the l = 1 result for
        # simplicity
        teval[2, 0], value[2, 0] = teval[2, 1], value[2, 1]
        teval[3, 0], value[3, 0] = teval[3, 1], value[3, 1]

        # Assume starry solution is error-free (TODO: verify)
        error = np.abs(value - value[0].reshape(1, -1))
        error[1] = error[0]
        error[2] = error[0]
        error[3] = error[0]

        # Marker size is proportional to log error
        def ms(error):
            return 18 + max(-16, np.log10(error))

        # Plot it
        fig = plt.figure(figsize=(7, 4))
        ax = plt.subplot2grid((2, 5), (0, 0), colspan=4, rowspan=2)
        axleg1 = plt.subplot2grid((2, 5), (0, 4))
        axleg2 = plt.subplot2grid((2, 5), (1, 4))
        axleg1.axis("off")
        axleg2.axis("off")
        ax.set_xlabel("Spherical harmonic degree", fontsize=14)
        ax.set_xticks(range(ydeg + 1))
        for tick in ax.get_xticklabels():
            tick.set_fontsize(12)
        ax.set_ylabel("Evaluation time [seconds]", fontsize=14)

        # Starry
        for n in range(len(teval)):
            for l in range(ydeg + 1):
                ax.plot(l, teval[n, l], "o", color=styles[n][1], ms=ms(error[n, l]))
            ax.plot(
                range(ydeg + 1),
                teval[n],
                color=styles[n][1],
                ls=styles[n][2],
                lw=1,
                alpha=0.25,
            )
            axleg1.plot(
                [0, 1], [0, 1], color=styles[n][1], ls=styles[n][2], label=styles[n][0]
            )
        ax.set_yscale("log")
        axleg1.set_xlim(2, 3)
        leg = axleg1.legend(loc="center", frameon=False)
        leg.set_title("method", prop={"weight": "bold"})
        for logerr in [-16, -12, -8, -4, 0]:
            axleg2.plot(
                [0, 1],
                [0, 1],
                "o",
                color="gray",
                ms=ms(10 ** logerr),
                label=r"$%3d$" % logerr,
            )
        axleg2.set_xlim(2, 3)
        leg = axleg2.legend(loc="center", labelspacing=1, frameon=False)
        leg.set_title("log error", prop={"weight": "bold"})

        plt.show()
        # DEBUG fig.savefig("speed.pdf", bbox_inches="tight")

        # Restore
        self.y = yfull


if __name__ == "__main__":

    # Geometric params
    xs = -1
    ys = 0.5
    zs = 0.4
    xo = 0.0
    yo = 0.7
    ro = 0.8

    # Compute the integration boundaries
    boundaries = []
    theta = -np.arctan2(xs, ys)
    b = -zs / np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    c = np.cos(theta)
    s = np.sin(theta)
    q2 = c ** 2 + b ** 2 * s ** 2

    # 1: body lower limb -> body upper limb
    xa = -(1.0 / (2.0 * yo)) * np.sqrt(4 * yo ** 2 - (1 + yo ** 2 - ro ** 2) ** 2)
    boundaries.append(
        (-1, xa, lambda x: -np.sqrt(1 - x ** 2), lambda x: np.sqrt(1 - x ** 2))
    )

    # 2: body lower limb -> occultor lower limb
    xb = -np.cos(theta)
    x = np.linspace(xa, xb, 1000)
    boundaries.append(
        (
            xa,
            xb,
            lambda x: -np.sqrt(1 - x ** 2),
            lambda x: yo - np.sqrt(ro ** 2 - x ** 2),
        )
    )

    # 3: terminator -> occultor lower limb
    def diff(x):
        p = (x * c + b * s * np.sqrt(q2 - x ** 2)) / q2
        y1 = p * s + b * np.sqrt(1 - p ** 2) * c
        y2 = yo - np.sqrt(ro ** 2 - x ** 2)
        return y1 - y2

    res = root_scalar(diff, bracket=[0.36, 0.38], method="brentq", xtol=1e-16)
    xc = res.root
    x = np.linspace(xb, xc, 1000)
    b = -zs / np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    c = np.cos(theta)
    s = np.sin(theta)
    q2 = c ** 2 + b ** 2 * s ** 2

    def y1(x):
        p = (x * c + b * s * np.sqrt(q2 - x ** 2)) / q2
        return p * s + b * np.sqrt(1 - p ** 2) * c

    boundaries.append((xb, xc, y1, lambda x: yo - np.sqrt(ro ** 2 - x ** 2)))

    # Let's go
    ydeg = 5
    cmp = Compare(y=np.ones((ydeg + 1) ** 2))
    cmp.display(xs, ys, zs, xo, yo, ro, boundaries=boundaries)
    cmp.compare(xs, ys, zs, xo, yo, ro)
