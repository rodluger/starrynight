"""Starry speed tests."""
import starry
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import theano
import theano.tensor as tt
import theano.sparse as ts
from theano.ifelse import ifelse
from starry._core.utils import autocompile
from tqdm import tqdm
import time
import warnings


# Config
plt.switch_backend("MacOSX")
starry.config.lazy = False
starry.config.quiet = True
warnings.simplefilter("ignore")
HUGE = 1e30
np.random.seed(1234)


class Compare(object):
    """Compare different ways of evaluating the flux."""

    def __init__(self, ydeg):
        self.ydeg = ydeg

    @property
    def ydeg(self):
        return self._ydeg

    @ydeg.setter
    def ydeg(self, ydeg):
        self._ydeg = ydeg
        self.map_ref = starry.Map(ydeg=ydeg, reflected=True)
        self.map_emi = starry.Map(ydeg=ydeg)
        if ydeg > 0:
            self.map_ref[1:, :] = 1
            self.map_emi[1:, :] = 1
        self._A1y = ts.dot(
            self.map_ref.ops.A1, tt.as_tensor_variable(np.ones((ydeg + 1) ** 2))
        )

        # Dry run to force compile
        self.flux()
        self.dfluxdro()
        self.flux_emitted()
        self.dfluxdro_emitted()

    @autocompile
    def _intensity_theano(self, y, x, xs, ys, zs, xo, yo, ro):

        # Get the z coord
        z = tt.sqrt(1 - x ** 2 - y ** 2)

        # Compute the intensity
        pT = self.map_ref.ops.pT(x, y, z)

        # Weight the intensity by the illumination
        # Dot the polynomial into the basis
        intensity = tt.shape_padright(tt.dot(pT, self._A1y))

        # Weight the intensity by the illumination
        xyz = tt.concatenate(
            (tt.reshape(x, [1, -1]), tt.reshape(y, [1, -1]), tt.reshape(z, [1, -1]),)
        )
        I = self.map_ref.ops.compute_illumination(xyz, xs, ys, zs)
        intensity = tt.switch(tt.isnan(intensity), intensity, intensity * I)[0, 0]

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

    def _intensity(self, y, x, xs, ys, zs, xo, yo, ro):
        # Wrapper to get the dims correct
        y = np.array([y])
        x = np.array([x])
        xs = np.array([xs])
        ys = np.array([ys])
        zs = np.array([zs])
        xo = np.array([xo])
        yo = np.array([yo])
        ro = np.array(ro)
        return self._intensity_theano(y, x, xs, ys, zs, xo, yo, ro)

    @autocompile
    def _flux_theano(self, xs, ys, zs, xo, yo, ro):
        theta = tt.zeros_like(xo)
        zo = tt.ones_like(xo)
        inc = tt.as_tensor_variable(np.pi / 2).astype(tt.config.floatX)
        obl = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
        u = tt.as_tensor_variable([-1.0]).astype(tt.config.floatX)
        f = tt.as_tensor_variable([np.pi]).astype(tt.config.floatX)
        alpha = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
        y = tt.as_tensor_variable(np.ones((self.ydeg + 1) ** 2)).astype(
            tt.config.floatX
        )
        return self.map_ref.ops.flux(
            theta, xs, ys, zs, xo, yo, zo, ro, inc, obl, y, u, f, alpha
        )[0]

    def flux(self, xs=0, ys=0, zs=1, xo=0, yo=0, ro=0, **kwargs):
        # Compute the flux analytically
        xs = np.atleast_1d(xs)
        ys = np.atleast_1d(ys)
        zs = np.atleast_1d(zs)
        xo = np.atleast_1d(xo)
        yo = np.atleast_1d(yo)
        ro = np.array(ro)
        return self._flux_theano(xs, ys, zs, xo, yo, ro)

    @autocompile
    def _dfluxdro_theano(self, xs, ys, zs, xo, yo, ro):
        theta = tt.zeros_like(xo)
        zo = tt.ones_like(xo)
        inc = tt.as_tensor_variable(np.pi / 2).astype(tt.config.floatX)
        obl = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
        u = tt.as_tensor_variable([-1.0]).astype(tt.config.floatX)
        f = tt.as_tensor_variable([np.pi]).astype(tt.config.floatX)
        alpha = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
        y = tt.as_tensor_variable(np.ones((self.ydeg + 1) ** 2)).astype(
            tt.config.floatX
        )
        return theano.grad(
            self.map_ref.ops.flux(
                theta, xs, ys, zs, xo, yo, zo, ro, inc, obl, y, u, f, alpha
            )[0],
            ro,
        )

    def dfluxdro(self, xs=0, ys=0, zs=1, xo=0, yo=0, ro=0, **kwargs):
        # Compute the flux analytically
        xs = np.atleast_1d(xs)
        ys = np.atleast_1d(ys)
        zs = np.atleast_1d(zs)
        xo = np.atleast_1d(xo)
        yo = np.atleast_1d(yo)
        ro = np.array(ro)
        return self._dfluxdro_theano(xs, ys, zs, xo, yo, ro)

    @autocompile
    def _flux_emitted_theano(self, xo, yo, ro):
        theta = tt.zeros_like(xo)
        zo = tt.ones_like(xo)
        inc = tt.as_tensor_variable(np.pi / 2).astype(tt.config.floatX)
        obl = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
        u = tt.as_tensor_variable([-1.0]).astype(tt.config.floatX)
        f = tt.as_tensor_variable([np.pi]).astype(tt.config.floatX)
        alpha = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
        y = tt.as_tensor_variable(np.ones((self.ydeg + 1) ** 2)).astype(
            tt.config.floatX
        )
        return self.map_emi.ops.flux(theta, xo, yo, zo, ro, inc, obl, y, u, f, alpha)[0]

    def flux_emitted(self, xo=0, yo=0, ro=0, **kwargs):
        # Compute the flux in emission (starry 1.0)
        xo = np.atleast_1d(xo)
        yo = np.atleast_1d(yo)
        ro = np.array(ro)
        return self._flux_emitted_theano(xo, yo, ro)

    @autocompile
    def _dfluxdro_emitted_theano(self, xo, yo, ro):
        theta = tt.zeros_like(xo)
        zo = tt.ones_like(xo)
        inc = tt.as_tensor_variable(np.pi / 2).astype(tt.config.floatX)
        obl = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
        u = tt.as_tensor_variable([-1.0]).astype(tt.config.floatX)
        f = tt.as_tensor_variable([np.pi]).astype(tt.config.floatX)
        alpha = tt.as_tensor_variable(0.0).astype(tt.config.floatX)
        y = tt.as_tensor_variable(np.ones((self.ydeg + 1) ** 2)).astype(
            tt.config.floatX
        )
        return theano.grad(
            self.map_emi.ops.flux(theta, xo, yo, zo, ro, inc, obl, y, u, f, alpha)[0],
            ro,
        )

    def dfluxdro_emitted(self, xo=0, yo=0, ro=0, **kwargs):
        # Compute the flux in emission (starry 1.0)
        xo = np.atleast_1d(xo)
        yo = np.atleast_1d(yo)
        ro = np.array(ro)
        return self._dfluxdro_emitted_theano(xo, yo, ro)

    def flux_quad(
        self, xs=0, ys=0, zs=1, xo=0, yo=0, ro=0, epsabs=1e-8, epsrel=1e-8, **kwargs
    ):
        # Compute the double integral numerically
        val, err = dblquad(
            self._intensity,
            -1,
            1,
            lambda x: -np.sqrt(1 - x ** 2),
            lambda x: np.sqrt(1 - x ** 2),
            epsabs=epsabs,
            epsrel=epsrel,
            args=(xs, ys, zs, xo, yo, ro),
        )
        return val, err

    def flux_brute(self, xs=0, ys=0, zs=1, xo=0, yo=0, ro=0, res=999, **kwargs):
        # Compute the flux by brute force grid integration
        img = self.map_ref.render(xs=xs, ys=ys, zs=zs, res=res)
        x, y, z = self.map_ref.ops.compute_ortho_grid(res)
        idx = x ** 2 + (y - yo) ** 2 > ro ** 2
        return np.nansum(img.flat[idx]) * 4 / res ** 2

    def display(
        self,
        ax=None,
        xs=0,
        ys=0,
        zs=1,
        xo=0,
        yo=0,
        ro=0,
        res=999,
        cmap="plasma",
        occultor_color="k",
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

        # Figure out limits
        cmap = plt.get_cmap(cmap)
        cmap.set_over(occultor_color)
        cmap.set_under("k")
        vmin = np.nanmin(img[np.abs(img) < HUGE])
        vmax = np.nanmax(img[np.abs(img) < HUGE])

        # Display and return
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.imshow(
            img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=(-1, 1, -1, 1),
        )
        return ax

    def _run(self, func, nt=1, **kwargs):
        t = time.time()
        val = func(**kwargs)
        return (time.time() - t) / nt, val

    def compare(self, res=999, nt=1000):
        """Compare different integration schemes."""

        # Initialize
        ydeg = self.ydeg
        teval = np.zeros((6, ydeg + 1))
        value = np.zeros((6, ydeg + 1))

        # Params for the comparison
        kwargs = dict(xs=-1, ys=0.5, zs=0.4, xo=0.0, yo=0.7, ro=0.8)
        kwargs_vec = dict(kwargs)
        for key in ["xs", "ys", "zs", "xo", "yo"]:
            kwargs_vec[key] *= np.ones(nt)

        styles = [
            ("starry: reflected", "C0", "-"),
            ("starry: reflected (grad)", "C1", "-"),
            ("starry: emitted", "C2", "-"),
            ("starry: emitted (grad)", "C3", "-"),
            ("grid", "C4", "-"),
            ("dblquad", "C5", "-"),
        ]

        # Loop over all degrees
        for l in tqdm(range(ydeg + 1)):
            self.ydeg = l
            teval[0, l], value[0, l] = self._run(self.flux, nt, **kwargs_vec)
            teval[1, l], value[1, l] = self._run(self.dfluxdro, nt, **kwargs_vec)
            teval[2, l], value[2, l] = self._run(self.flux_emitted, nt, **kwargs_vec)
            teval[3, l], value[3, l] = self._run(
                self.dfluxdro_emitted, nt, **kwargs_vec
            )
            teval[4, l], value[4, l] = self._run(self.flux_brute, **kwargs)
            teval[5, l], value[5, l] = (
                np.nan,
                np.nan,
            )  # TODO self._run(self.flux_quad, **kwargs)

        # TODO
        error = np.abs(value - value[0].reshape(1, -1))
        error[:4] = 0

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
        self.ydeg = ydeg


if __name__ == "__main__":
    cmp = Compare(ydeg=10)
    cmp.compare()
