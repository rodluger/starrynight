"""Starry speed and stability tests."""
import starry
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import theano
import theano.tensor as tt
import theano.sparse as ts
from theano.ifelse import ifelse
from tqdm import tqdm
import time
import warnings


# Config
starry.config.lazy = False
starry.config.quiet = True
warnings.simplefilter("ignore")
HUGE = 1e30
np.random.seed(1234)


# Theano dummy variables
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
        self.intensity(0.0, 0.0, [0.0], [0.0], [0.0], [0.0], [0.0], 0.0, reset=True)
        self.flux([0.0], [0.0], [0.0], [0.0], [0.0], 0.0, reset=True)
        self.dfluxdro([0.0], [0.0], [0.0], [0.0], [0.0], 0.0, reset=True)
        if self.ydeg > 0:
            self.flux_emitted([0.0], [0.0], [0.0], [0.0], [0.0], 0.0, reset=True)
            self.dfluxdro_emitted([0.0], [0.0], [0.0], [0.0], [0.0], 0.0, reset=True)

    def intensity(self, y, x, xs, ys, zs, xo, yo, ro, reset=False):

        if reset or not hasattr(self, "_intensity"):

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

            self._intensity = theano.function(
                [_y, _x, _xs, _ys, _zs, _xo, _yo, _ro],
                theano_code(_y, _x, _xs, _ys, _zs, _xo, _yo, _ro),
            )

        return self._intensity(np.array([y]), np.array([x]), xs, ys, zs, xo, yo, ro)

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
                )

            self._flux = theano.function(
                [_xs, _ys, _zs, _xo, _yo, _ro],
                theano_code(_xs, _ys, _zs, _xo, _yo, _ro),
            )

        flux = self._flux(xs, ys, zs, xo, yo, ro)
        err = np.max(flux) - np.min(flux)
        return np.median(flux), err

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

        return self._dfluxdro(xs, ys, zs, xo, yo, ro), np.nan

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
                )

            self._flux_emitted = theano.function(
                [_xo, _yo, _ro], theano_code(_xo, _yo, _ro),
            )

        flux = self._flux_emitted(xo, yo, ro)
        err = np.max(flux) - np.min(flux)
        return np.median(flux), err

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

        return self._dfluxdro_emitted(xo, yo, ro), np.nan

    def flux_brute(self, xs, ys, zs, xo, yo, ro, res=999, **kwargs):
        # Compute the flux by brute force grid integration
        img = self.map_ref.render(xs=xs, ys=ys, zs=zs, res=res)
        x, y, z = self.map_ref.ops.compute_ortho_grid(res)
        idx = x ** 2 + (y - yo) ** 2 > ro ** 2
        return np.nansum(img.flat[idx]) * 4 / res ** 2, np.nan

    def flux_quad(self, xs, ys, zs, xo, yo, ro, epsabs=1e-3, epsrel=1e-3, **kwargs):
        # Compute the double integral numerically
        val, err = dblquad(
            self.intensity,
            -1,
            1,
            lambda x: -np.sqrt(1 - x ** 2),
            lambda x: np.sqrt(1 - x ** 2),
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
        return val, err

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
        cmap="plasma",
        occultor_color="k",
    ):
        """Render an imshow view of the map."""
        # Render
        img = self.map_ref.render(xs=xs, ys=ys, zs=zs, res=res)

        # Mask the occultor
        x, y, z = self.map_ref.ops.compute_ortho_grid(res)
        idx = (x - xo) ** 2 + (y - yo) ** 2 <= ro ** 2
        img.flat[idx] = HUGE

        # Mask the night side
        img[img == 0.0] = -HUGE

        # Figure out imshow limits
        cmap = plt.get_cmap(cmap)
        cmap.set_over(occultor_color)
        cmap.set_under("k")
        viz = img[np.abs(img) < HUGE]
        if len(viz):
            vmin = np.nanmin(viz)
            vmax = np.nanmax(viz)
        else:
            vmin = 0
            vmax = 1

        # Display
        if ax is None:
            fig, ax = plt.subplots(1)
            ax.axis("off")
        ax.imshow(
            img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=(-1, 1, -1, 1),
        )
        ax.add_artist(plt.Circle((xo, yo), ro, edgecolor="w", color=occultor_color))

        return ax

    def _run(self, func, nt=1, *args, **kwargs):
        t = time.time()
        val, err = func(*args, **kwargs)
        return (time.time() - t) / nt, val, err

    def compare(
        self, xs, ys, zs, xo, yo, ro, res=999, nt=1000, epsabs=1e-3, epsrel=1e-3,
    ):
        """Compare different integration schemes."""

        # Initialize
        yfull = np.array(self.y)
        teval = np.zeros((6, ydeg + 1)) * np.nan
        value = np.zeros((6, ydeg + 1)) * np.nan
        error = np.zeros((6, ydeg + 1)) * np.nan

        # Params for the comparison
        kwargs = dict(xs=xs, ys=ys, zs=zs, xo=xo, yo=yo, ro=ro)

        # Vectorized params (for the starry computation)
        # We perturb them close to the machine level, and compute
        # the "error" as the max-min difference in the flux over
        # the interval. This is essentially a probe of the condition number
        # of the starry algorithm.
        kwargs_vec = dict(kwargs)
        for key in ["xs", "ys", "zs", "xo", "yo"]:
            kwargs_vec[key] *= np.ones(nt) + 1e-15 * np.random.randn(nt)

        styles = [
            ("starry: reflected", "C0", "-"),
            ("starry: reflected (grad)", "C0", "--"),
            ("starry: emitted", "C1", "-"),
            ("starry: emitted (grad)", "C1", "--"),
            ("grid", "C4", "-"),
            ("dblquad", "C4", "--"),
        ]

        # Loop over all degrees
        for l in tqdm(range(ydeg + 1)):
            self.y = yfull[: (l + 1) ** 2]
            teval[0, l], value[0, l], error[0, l] = self._run(
                self.flux, nt, **kwargs_vec
            )
            teval[1, l], value[1, l], error[1, l] = self._run(
                self.dfluxdro, nt, **kwargs_vec
            )
            if l > 0:
                teval[2, l], value[2, l], error[2, l] = self._run(
                    self.flux_emitted, nt, **kwargs_vec
                )
                teval[3, l], value[3, l], error[3, l] = self._run(
                    self.dfluxdro_emitted, nt, **kwargs_vec
                )
            teval[4, l], value[4, l], error[4, l] = self._run(
                self.flux_brute, res=res, **kwargs
            )
            teval[5, l], value[5, l], error[5, l] = self._run(
                self.flux_quad, epsabs=epsabs, epsrel=epsrel, **kwargs
            )

        # Zero-degree emitted light maps are limb-darkened maps,
        # so let's just set them equal to the l = 1 result for
        # simplicity
        teval[2, 0], value[2, 0], error[2, 0] = teval[2, 1], value[2, 1], error[2, 1]
        teval[3, 0], value[3, 0], error[3, 0] = teval[3, 1], value[3, 1], error[3, 1]

        # Compute the error on the numerical solution
        # assuming the starry solution is the ground truth
        error[4] = np.abs(value[4] - value[0])
        error[5] = np.abs(value[5] - value[0])

        # Propagate the error to the starry-grad terms
        # (identical by construction)
        error[1] = error[0]
        error[3] = error[2]

        # Marker size is proportional to log error
        def ms(error):
            return 18 + min(0, max(-16, np.log10(error)))

        # Plot it
        fig = plt.figure(figsize=(10, 4))
        ax = plt.subplot2grid((2, 12), (0, 0), colspan=9, rowspan=2)
        axleg1 = plt.subplot2grid((2, 12), (0, 9), colspan=3)
        axleg2 = plt.subplot2grid((2, 12), (1, 9), colspan=3)
        axleg1.axis("off")
        axleg2.axis("off")
        ax.set_xlabel("Spherical harmonic degree", fontsize=14)
        ax.set_xticks(range(ydeg + 1))
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(12)
        ax.set_ylabel("Evaluation time [seconds]", fontsize=14)

        # Starry
        for n in range(len(teval)):
            for l in range(ydeg + 1):
                ax.plot(l, teval[n, l], "o", color=styles[n][1], ms=ms(error[n, l]))
                if n in [0, 3, 4, 5]:
                    logerr = np.round(max(-16, np.log10(error[n, l])))
                    ax.annotate(
                        "{:.0f}".format(logerr),
                        xy=(l, teval[n, l]),
                        xycoords="data",
                        xytext=(0, 10 if n < 4 else 0),
                        textcoords="offset points",
                        va="center",
                        ha="center",
                        fontsize=6,
                        color=("k" if n < 4 else "w"),
                    )
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
        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
        axleg1.set_xlim(2, 3)
        leg = axleg1.legend(loc="center", frameon=False, fontsize=8)
        leg.set_title("method", prop={"weight": "bold", "size": 10})
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
        leg = axleg2.legend(loc="center", labelspacing=1, frameon=False, fontsize=8)
        leg.set_title("log error", prop={"weight": "bold", "size": 10})
        fig.savefig("speed.pdf", bbox_inches="tight")
        plt.close()

        # Restore
        self.y = yfull


if __name__ == "__main__":

    # Geometric params
    xs = -0.5
    ys = 0.5
    zs = 0.5
    xo = -0.5
    yo = -0.1
    ro = 1.25

    # Let's go
    ydeg = 10
    y = np.ones((ydeg + 1) ** 2)
    cmp = Compare(y=y)
    cmp.compare(xs, ys, zs, xo, yo, ro)
