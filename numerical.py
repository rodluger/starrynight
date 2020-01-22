"""
Singularities:

    - bo = 0
    - b0 = 0 and theta = 90 (only one root)
    - b0 <~ 0.1 and theta = 90 (root finding fails I think)

"""
import matplotlib.pyplot as plt
import numpy as np
import starry
from starry._c_ops import Ops
from starry._core.ops.rotation import dotROp
from starry._core.ops.integration import sTOp
from starry._core.ops.polybasis import pTOp
import theano
from scipy.integrate import quad
from sympy import binomial
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc
import warnings

warnings.simplefilter("ignore")

# Integration codes
FLUX_NONE = 0
FLUX_DAY_OCC = 1
FLUX_DAY_VIS = 2
FLUX_NIGHT_OCC = 3
FLUX_NIGHT_VIS = 4


class Numerical(object):
    def __init__(self, y1, b, theta, bo, ro, tol=1e-7, epsabs=1e-12, epsrel=1e-12):

        self.y = np.append([1.0], y1)
        self.ydeg = int(np.sqrt(len(self.y)) - 1)
        self.b = b
        self.theta = theta
        self.bo = bo
        self.ro = ro
        self.tol = tol
        self.epsabs = epsabs
        self.epsrel = epsrel
        self._do_setup = True

    def _setup(self):
        # Have we done this already?
        if not self._do_setup:
            return
        self._do_setup = False

        # Instantiate a starry map
        self.map = starry.Map(ydeg=self.ydeg + 1)
        self.map_refl = starry.Map(ydeg=self.ydeg, reflected=True)

        # Instantiate the ops
        ops = Ops(self.ydeg + 1, 0, 0, 0)

        # Basis transform from poly to Green's
        self.A2 = np.array(theano.sparse.dot(ops.A, ops.A1Inv).eval())

        # Basis transform from Ylms to poly and back
        N = (self.ydeg + 1) ** 2
        self.A1 = np.array(ops.A1.todense())[:N, :N]
        self.A1Inv = np.array(ops.A1Inv.todense())

        # Z-rotation matrix
        theta = theano.tensor.dscalar()
        self.Rz = theano.function(
            [theta],
            dotROp(ops.dotR)(
                np.eye((self.ydeg + 1) ** 2),
                np.array(0.0),
                np.array(0.0),
                np.array(1.0),
                theta,
            ),
        )

        # Solution vector
        b = theano.tensor.dscalar()
        r = theano.tensor.dscalar()
        self.sT = theano.function([b, r], sTOp(ops.sT, (self.ydeg + 1) ** 2)([b], r),)

        # Poly basis op
        x = theano.tensor.dvector()
        y = theano.tensor.dvector()
        z = theano.tensor.dvector()
        self.pT = theano.function([x, y, z], pTOp(ops.pT, self.ydeg + 1)(x, y, z),)

    def visualize(self, res=999):

        # Find angles of intersection
        phi, lam, xi, code = self.angles()

        # Equation of half-ellipse
        x = np.linspace(-1, 1, 1000)
        y = self.b * np.sqrt(1 - x ** 2)
        x_t = x * np.cos(self.theta) - y * np.sin(self.theta)
        y_t = x * np.sin(self.theta) + y * np.cos(self.theta)

        # Shaded regions
        p = np.linspace(-1, 1, res)
        xpt, ypt = np.meshgrid(p, p)
        cond1 = xpt ** 2 + (ypt - self.bo) ** 2 < self.ro ** 2  # inside occultor
        cond2 = xpt ** 2 + ypt ** 2 < 1  # inside occulted
        xr = xpt * np.cos(self.theta) + ypt * np.sin(self.theta)
        yr = -xpt * np.sin(self.theta) + ypt * np.cos(self.theta)
        cond3 = yr > self.b * np.sqrt(1 - xr ** 2)  # above terminator
        img_day_occ = np.zeros_like(xpt)
        img_day_occ[cond1 & cond2 & cond3] = 1
        img_night = np.zeros_like(xpt)
        img_night[~cond1 & cond2 & ~cond3] = 1

        # Dayside image
        p = np.linspace(-1, 1, res)
        xpt = xpt.flatten()
        ypt = ypt.flatten()
        zpt = np.sqrt(1 - xpt ** 2 - ypt ** 2)
        cond1 = xpt ** 2 + (ypt - self.bo) ** 2 > self.ro ** 2  # outside occultor
        cond2 = xpt ** 2 + ypt ** 2 < 1  # inside occulted
        xr = xpt * np.cos(self.theta) + ypt * np.sin(self.theta)
        yr = -xpt * np.sin(self.theta) + ypt * np.cos(self.theta)
        cond3 = yr > self.b * np.sqrt(1 - xr ** 2)  # above terminator
        image = self.pT(xpt, ypt, zpt).dot(self.I()).dot(self.A1).dot(self.y)
        # DEBUG image[~(cond1 & cond2 & cond3)] = np.nan
        image = image.reshape(res, res)

        # Plot
        fig, ax = plt.subplots(1, 3, figsize=(14, 5))
        fig.subplots_adjust(left=0.025, right=0.975, bottom=0.05, top=0.95)
        ax[0].set_title("T", color="r")
        ax[1].set_title("P", color="r")
        ax[2].set_title("Q", color="r")

        # Labels
        for i in range(len(phi)):
            ax[0].annotate(
                r"$\xi_{} = {:.1f}^\circ$".format(i + 1, xi[i] * 180 / np.pi),
                xy=(0, 0),
                xycoords="axes fraction",
                xytext=(5, 25 - i * 20),
                textcoords="offset points",
                fontsize=10,
                color="C0",
            )
            ax[1].annotate(
                r"$\phi_{} = {:.1f}^\circ$".format(i + 1, phi[i] * 180 / np.pi),
                xy=(0, 0),
                xycoords="axes fraction",
                xytext=(5, 25 - i * 20),
                textcoords="offset points",
                fontsize=10,
                color="C0",
            )
        for i in range(len(lam)):
            ax[2].annotate(
                r"$\lambda_{} = {:.1f}^\circ$".format(i + 1, lam[i] * 180 / np.pi),
                xy=(0, 0),
                xycoords="axes fraction",
                xytext=(5, 25 - i * 20),
                textcoords="offset points",
                fontsize=10,
                color="C0",
            )

        # Draw basic shapes
        for axis in ax:
            axis.axis("off")
            axis.add_artist(plt.Circle((0, self.bo), self.ro, fill=False))
            axis.add_artist(plt.Circle((0, 0), 1, fill=False))
            axis.plot(x_t, y_t, "k-", lw=1)
            axis.set_xlim(-1.25, 1.25)
            axis.set_ylim(-1.25, 1.25)
            axis.set_aspect(1)
            """
            axis.imshow(
                img_day_occ,
                origin="lower",
                extent=(-1, 1, -1, 1),
                alpha=0.25,
                cmap=LinearSegmentedColormap.from_list(
                    "cmap1", [(0, 0, 0, 0), "C1"], 2
                ),
            )
            """  # DEBUG
            axis.imshow(
                img_night,
                origin="lower",
                extent=(-1, 1, -1, 1),
                alpha=0.75,
                cmap=LinearSegmentedColormap.from_list("cmap1", [(0, 0, 0, 0), "k"], 2),
            )
            axis.imshow(
                image,
                origin="lower",
                extent=(-1, 1, -1, 1),
                cmap="Greys_r",
                vmin=0,
                alpha=0.75,
            )

        # Draw integration paths
        if len(phi):

            # T
            # This is the *actual* angle along the ellipse
            xi_p = np.arctan(np.abs(self.b) * np.tan(xi))
            xi_p[xi_p < 0] += np.pi
            if np.abs(self.b) < 1e-4:
                ax[0].plot(
                    [
                        np.cos(xi[0]) * np.cos(self.theta),
                        np.cos(xi[1]) * np.cos(self.theta),
                    ],
                    [
                        np.cos(xi[0]) * np.sin(self.theta),
                        np.cos(xi[1]) * np.sin(self.theta),
                    ],
                    color="r",
                    lw=2,
                    zorder=3,
                )
            else:
                if self.b < 0:
                    xi_p = xi_p[::-1]
                arc = Arc(
                    (0, 0),
                    2,
                    2 * np.abs(self.b),
                    self.theta * 180 / np.pi,
                    np.sign(self.b) * xi_p[1] * 180 / np.pi,
                    np.sign(self.b) * xi_p[0] * 180 / np.pi,
                    color="r",
                    lw=2,
                    zorder=3,
                )
                ax[0].add_patch(arc)

            # P
            arc = Arc(
                (0, self.bo),
                2 * self.ro,
                2 * self.ro,
                0,
                phi[0] * 180 / np.pi,
                phi[1] * 180 / np.pi,
                color="r",
                lw=2,
                zorder=3,
            )
            ax[1].add_patch(arc)

        if len(lam):

            # Q
            arc = Arc(
                (0, 0),
                2,
                2,
                0,
                lam[0] * 180 / np.pi,
                lam[1] * 180 / np.pi,
                color="r",
                lw=2,
                zorder=3,
            )
            ax[2].add_patch(arc)

        # Draw axes
        ax[0].plot(
            [-np.cos(self.theta), np.cos(self.theta)],
            [-np.sin(self.theta), np.sin(self.theta)],
            color="k",
            ls="--",
            lw=0.5,
        )
        ax[0].plot([0], [0], "C0o", ms=4, zorder=4)
        ax[1].plot(
            [-self.ro, self.ro], [self.bo, self.bo], color="k", ls="--", lw=0.5,
        )
        ax[1].plot([0], [self.bo], "C0o", ms=4, zorder=4)
        ax[2].plot(
            [-1, 1], [0, 0], color="k", ls="--", lw=0.5,
        )
        ax[2].plot(0, 0, "C0o", ms=4, zorder=4)

        # Draw points of intersection & angles
        sz = [0.25, 0.5]
        for i, xi_i in enumerate(xi):

            # -- T --

            # xi angle
            ax[0].plot(
                [0, np.cos(np.sign(self.b) * xi_i + self.theta)],
                [0, np.sin(np.sign(self.b) * xi_i + self.theta)],
                color="C0",
                lw=1,
            )

            # tangent line
            x0 = np.cos(xi_i) * np.cos(self.theta)
            y0 = np.cos(xi_i) * np.sin(self.theta)
            ax[0].plot(
                [x0, np.cos(np.sign(self.b) * xi_i + self.theta)],
                [y0, np.sin(np.sign(self.b) * xi_i + self.theta)],
                color="k",
                ls="--",
                lw=0.5,
            )

            # mark the polar angle
            ax[0].plot(
                [np.cos(np.sign(self.b) * xi_i + self.theta)],
                [np.sin(np.sign(self.b) * xi_i + self.theta)],
                "C0o",
                ms=4,
                zorder=4,
            )

            # draw and label the angle arc
            if np.sin(xi_i) != 0:
                angle = sorted([self.theta, np.sign(self.b) * xi_i + self.theta])
                arc = Arc(
                    (0, 0),
                    sz[i],
                    sz[i],
                    0,
                    angle[0] * 180 / np.pi,
                    angle[1] * 180 / np.pi,
                    color="C0",
                    lw=0.5,
                )
                ax[0].add_patch(arc)
                ax[0].annotate(
                    r"$\xi_{}$".format(i + 1),
                    xy=(
                        0.5 * sz[i] * np.cos(0.5 * np.sign(self.b) * xi_i + self.theta),
                        0.5 * sz[i] * np.sin(0.5 * np.sign(self.b) * xi_i + self.theta),
                    ),
                    xycoords="data",
                    xytext=(
                        7 * np.cos(0.5 * np.sign(self.b) * xi_i + self.theta),
                        7 * np.sin(0.5 * np.sign(self.b) * xi_i + self.theta),
                    ),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="C0",
                )

            # points of intersection?
            for phi_i in phi:
                x_phi = self.ro * np.cos(phi_i)
                y_phi = self.bo + self.ro * np.sin(phi_i)
                x_xi = np.cos(self.theta) * np.cos(xi_i) - self.b * np.sin(
                    self.theta
                ) * np.sin(xi_i)
                y_xi = np.sin(self.theta) * np.cos(xi_i) + self.b * np.cos(
                    self.theta
                ) * np.sin(xi_i)
                if np.abs(y_phi - y_xi) < self.tol and np.abs(x_phi - x_xi) < self.tol:
                    ax[0].plot(
                        [self.ro * np.cos(phi_i)],
                        [self.bo + self.ro * np.sin(phi_i)],
                        "C0o",
                        ms=4,
                        zorder=4,
                    )

        for i, phi_i in enumerate(phi):

            # -- P --

            # points of intersection
            ax[1].plot(
                [0, self.ro * np.cos(phi_i)],
                [self.bo, self.bo + self.ro * np.sin(phi_i)],
                color="C0",
                ls="-",
                lw=1,
            )
            ax[1].plot(
                [self.ro * np.cos(phi_i)],
                [self.bo + self.ro * np.sin(phi_i)],
                "C0o",
                ms=4,
                zorder=4,
            )

            # draw and label the angle arc
            angle = sorted([0, phi_i])
            arc = Arc(
                (0, self.bo),
                sz[i],
                sz[i],
                0,
                angle[0] * 180 / np.pi,
                angle[1] * 180 / np.pi,
                color="C0",
                lw=0.5,
            )
            ax[1].add_patch(arc)
            ax[1].annotate(
                r"${}\phi_{}$".format("-" if phi_i < 0 else "", i + 1),
                xy=(
                    0.5 * sz[i] * np.cos(0.5 * (phi_i % (2 * np.pi))),
                    self.bo + 0.5 * sz[i] * np.sin(0.5 * (phi_i % (2 * np.pi))),
                ),
                xycoords="data",
                xytext=(
                    7 * np.cos(0.5 * (phi_i % (2 * np.pi))),
                    7 * np.sin(0.5 * (phi_i % (2 * np.pi))),
                ),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                color="C0",
                zorder=4,
            )

        for i, lam_i in zip(range(len(lam)), lam):

            # -- Q --

            # points of intersection
            ax[2].plot(
                [0, np.cos(lam_i)], [0, np.sin(lam_i)], color="C0", ls="-", lw=1,
            )
            ax[2].plot(
                [np.cos(lam_i)], [np.sin(lam_i)], "C0o", ms=4, zorder=4,
            )

            # draw and label the angle arc
            angle = sorted([0, lam_i])
            arc = Arc(
                (0, 0),
                sz[i],
                sz[i],
                0,
                angle[0] * 180 / np.pi,
                angle[1] * 180 / np.pi,
                color="C0",
                lw=0.5,
            )
            ax[2].add_patch(arc)
            ax[2].annotate(
                r"${}\lambda_{}$".format("-" if lam_i < 0 else "", i + 1),
                xy=(
                    0.5 * sz[i] * np.cos(0.5 * lam_i),
                    0.5 * sz[i] * np.sin(0.5 * lam_i),
                ),
                xycoords="data",
                xytext=(7 * np.cos(0.5 * lam_i), 7 * np.sin(0.5 * lam_i),),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                color="C0",
                zorder=4,
            )

        plt.show()

    def I(self):
        # Illumination matrix
        y0 = np.sqrt(1 - self.b ** 2)
        x = -y0 * np.sin(self.theta)
        y = y0 * np.cos(self.theta)
        z = -self.b
        p = np.array([0, x, z, y]) * 1.5  # NOTE: 3 / 2 is the starry normalization!
        n1 = 0
        n2 = 0
        I = np.zeros(((self.ydeg + 2) ** 2, (self.ydeg + 1) ** 2))
        for l1 in range(self.ydeg + 1):
            for m1 in range(-l1, l1 + 1):
                if (l1 + m1) % 2 == 0:
                    odd1 = False
                else:
                    odd1 = True
                n2 = 0
                for l2 in range(2):
                    for m2 in range(-l2, l2 + 1):
                        l = l1 + l2
                        n = l * l + l + m1 + m2
                        if odd1 and ((l2 + m2) % 2 != 0):
                            I[n - 4 * l + 2, n1] += p[n2]
                            I[n - 2, n1] -= p[n2]
                            I[n + 2, n1] -= p[n2]
                        else:
                            I[n, n1] += p[n2]
                        n2 += 1
                n1 += 1
        return I

    def flux_brute(self, res=999):
        self._setup()
        p = np.linspace(-1, 1, res)
        xpt, ypt = np.meshgrid(p, p)
        xpt = xpt.flatten()
        ypt = ypt.flatten()
        zpt = np.sqrt(1 - xpt ** 2 - ypt ** 2)
        cond1 = xpt ** 2 + (ypt - self.bo) ** 2 > self.ro ** 2  # outside occultor
        cond2 = xpt ** 2 + ypt ** 2 < 1  # inside occulted
        xr = xpt * np.cos(self.theta) + ypt * np.sin(self.theta)
        yr = -xpt * np.sin(self.theta) + ypt * np.cos(self.theta)
        cond3 = yr > self.b * np.sqrt(1 - xr ** 2)  # above terminator
        image = self.pT(xpt, ypt, zpt).dot(self.I()).dot(self.A1).dot(self.y)
        flux = 4 * np.sum(image[cond1 & cond2 & cond3]) / (res ** 2)
        return flux

    def flux(self):
        self._setup()

        # Get integration limits
        phi, lam, xi, code = self.angles()

        # Compute primitive integrals
        P = np.zeros((self.ydeg + 2) ** 2)
        Q = np.zeros((self.ydeg + 2) ** 2)
        T = np.zeros((self.ydeg + 2) ** 2)
        n = 0
        for l in range(self.ydeg + 2):
            for m in range(-l, l + 1):
                if len(phi):
                    P[n] = self.P(l, m, phi[0], phi[1])
                if len(lam):
                    Q[n] = self.Q(l, m, lam[0], lam[1])
                if len(xi):
                    T[n] = self.T(l, m, xi[0], xi[1])
                n += 1

        # Compute the occulted flux
        f = (P + Q + T).dot(self.A2).dot(self.I()).dot(self.A1).dot(self.y)

        # Use it to compute the visible flux
        y0 = np.sqrt(1 - self.b ** 2)
        xs = -y0 * np.sin(self.theta)
        ys = y0 * np.cos(self.theta)
        zs = -self.b
        y_refl = self.A1Inv.dot(self.I()).dot(self.A1).dot(self.y)
        self.map[1:, :] = y_refl[1:]
        self.map_refl[1:, :] = self.y[1:]
        if code == FLUX_DAY_OCC:

            fd = self.map_refl.flux(xs=xs, ys=ys, zs=zs).eval()[0]
            return fd - f

        elif code == FLUX_NIGHT_OCC:

            # TODO: SUPER HACKY, FIX ME
            fs = self.map.flux(xo=0, yo=self.bo, ro=self.ro).eval()[0]
            self.map.reset()
            f0 = self.map.flux(xo=0, yo=self.bo, ro=self.ro).eval()[0]
            fs -= f0
            fn = -self.map_refl.flux(xs=-xs, ys=-ys, zs=-zs).eval()[0]
            return fs - (fn - f)

        elif code == FLUX_DAY_VIS:

            return f

        else:
            # TODO
            raise NotImplementedError("TODO!")

    def G(self, l, m):
        mu = l - m
        nu = l + m

        # NOTE: The abs prevents NaNs when the argument of the sqrt is
        # zero but floating point error causes it to be ~ -eps.
        z = lambda x, y: np.sqrt(np.abs(1 - x ** 2 - y ** 2))

        if nu % 2 == 0:
            G = [lambda x, y: 0, lambda x, y: x ** (0.5 * (mu + 2)) * y ** (0.5 * nu)]
        elif (l == 1) and (m == 0):
            G = [
                lambda x, y: (1 - z(x, y) ** 3) / (3 * (1 - z(x, y) ** 2)) * (-y),
                lambda x, y: (1 - z(x, y) ** 3) / (3 * (1 - z(x, y) ** 2)) * x,
            ]
        elif (mu == 1) and (l % 2 == 0):
            G = [lambda x, y: x ** (l - 2) * z(x, y) ** 3, lambda x, y: 0]
        elif (mu == 1) and (l % 2 != 0):
            G = [lambda x, y: x ** (l - 3) * y * z(x, y) ** 3, lambda x, y: 0]
        else:
            G = [
                lambda x, y: 0,
                lambda x, y: x ** (0.5 * (mu - 3))
                * y ** (0.5 * (nu - 1))
                * z(x, y) ** 3,
            ]
        return G

    def primitive(self, l, m, x, y, dx, dy, theta1, theta2):
        """A general primitive integral computed numerically."""
        G = self.G(l, m)
        func = lambda theta: G[0](x(theta), y(theta)) * dx(theta) + G[1](
            x(theta), y(theta)
        ) * dy(theta)
        res, _ = quad(func, theta1, theta2, epsabs=self.epsabs, epsrel=self.epsrel,)
        return res

    def Q(self, l, m, lam1, lam2):
        """Compute the Q integral numerically from its integral definition."""
        x = lambda lam: np.cos(lam)
        y = lambda lam: np.sin(lam)
        dx = lambda lam: -np.sin(lam)
        dy = lambda lam: np.cos(lam)
        return self.primitive(l, m, x, y, dx, dy, lam1, lam2)

    def T(self, l, m, xi1, xi2):
        """Compute the T integral numerically from its integral definition."""
        x = lambda xi: np.cos(self.theta) * np.cos(xi) - self.b * np.sin(
            self.theta
        ) * np.sin(xi)
        y = lambda xi: np.sin(self.theta) * np.cos(xi) + self.b * np.cos(
            self.theta
        ) * np.sin(xi)
        dx = lambda xi: -np.cos(self.theta) * np.sin(xi) - self.b * np.sin(
            self.theta
        ) * np.cos(xi)
        dy = lambda xi: -np.sin(self.theta) * np.sin(xi) + self.b * np.cos(
            self.theta
        ) * np.cos(xi)
        return self.primitive(l, m, x, y, dx, dy, xi1, xi2)

    def P(self, l, m, phi1, phi2):
        """Compute the P integral numerically from its integral definition."""
        x = lambda phi: self.ro * np.cos(phi)
        y = lambda phi: self.bo + self.ro * np.sin(phi)
        dx = lambda phi: -self.ro * np.sin(phi)
        dy = lambda phi: self.ro * np.cos(phi)
        return self.primitive(l, m, x, y, dx, dy, phi1, phi2)

    def on_dayside(self, x, y):
        """Return True if a point is on the dayside."""
        xr = x * np.cos(self.theta) + y * np.sin(self.theta)
        yr = -x * np.sin(self.theta) + y * np.cos(self.theta)
        term = 1 - xr ** 2
        if term < 0 and term > -self.tol:
            term = 0
        yt = self.b * np.sqrt(term)
        return yr >= yt

    def angles(self):

        # TODO: Use Sturm's theorem here?

        # We'll solve for occultor-terminator intersections
        # in the frame where the semi-major axis of the
        # terminator ellipse is aligned with the x axis
        xo = self.bo * np.sin(self.theta)
        yo = self.bo * np.cos(self.theta)

        # Special case: b = 0
        if np.abs(self.b) < self.tol:

            x = np.array([])
            term = np.sqrt(self.ro ** 2 - yo ** 2)
            if np.abs(xo + term) < 1:
                x = np.append(x, xo + term)
            if np.abs(xo - term) < 1:
                x = np.append(x, xo - term)

        # Need to solve a quartic
        else:

            A = (1 - self.b ** 2) ** 2
            B = -4 * xo * (1 - self.b ** 2)
            C = -2 * (
                self.b ** 4
                + self.ro ** 2
                - 3 * xo ** 2
                - yo ** 2
                - self.b ** 2 * (1 + self.ro ** 2 - xo ** 2 + yo ** 2)
            )
            D = -4 * xo * (self.b ** 2 - self.ro ** 2 + xo ** 2 + yo ** 2)
            E = (
                self.b ** 4
                - 2 * self.b ** 2 * (self.ro ** 2 - xo ** 2 + yo ** 2)
                + (self.ro ** 2 - xo ** 2 - yo ** 2) ** 2
            )

            # Get all real roots `x` that satisfy `sgn(y(x)) = sgn(b)`.
            x = np.roots([A, B, C, D, E])
            x = np.array([xi.real for xi in x if np.abs(xi.imag) < self.tol])
            x = np.array(
                [
                    xi
                    for xi in x
                    if np.abs(
                        (xi - xo) ** 2
                        + (self.b * np.sqrt(1 - xi ** 2) - yo) ** 2
                        - self.ro ** 2
                    )
                    < self.tol
                ]
            )

        # Get rid of any multiplicity
        x = np.array(list(set(x)))

        # Check that the number of roots is correct
        x_l = np.cos(self.theta)
        y_l = np.sin(self.theta)
        l1 = x_l ** 2 + (y_l - self.bo) ** 2 < self.ro
        l2 = x_l ** 2 + (-y_l - self.bo) ** 2 < self.ro
        if (l1 and not l2) or (l2 and not l1):
            if len(x) == 1:
                # All good
                pass
            else:
                # There should be one root!
                if len(x) == 0:
                    raise RuntimeError(
                        "Unable to find the root. Try decreasing the tolerance."
                    )
                elif len(x) == 2:
                    # We likely have a rogue root that was included
                    # because of the tolerance.
                    # Pick the one with the smallest error
                    x = np.array(
                        [
                            x[
                                np.argmin(
                                    np.abs(
                                        (x - xo) ** 2
                                        + (self.b * np.sqrt(1 - x ** 2) - yo) ** 2
                                        - self.ro ** 2
                                    )
                                )
                            ]
                        ]
                    )

        # P-Q
        if len(x) == 0:

            # Use the standard starry algorithm instead!
            return np.array([]), np.array([]), np.array([]), FLUX_NONE

        # P-Q-T
        if len(x) == 1:

            # CASE 1

            # PHI
            # ---

            # Angle of intersection with occultor
            phi_o = np.arcsin(
                (1 - self.ro ** 2 - self.bo ** 2) / (2 * self.bo * self.ro)
            )
            # There are always two points; always pick the one
            # that's on the dayside for definiteness
            if not self.on_dayside(
                self.ro * np.cos(phi_o), self.bo + self.ro * np.sin(phi_o)
            ):
                phi_o = np.pi - phi_o

            # Angle of intersection with the terminator
            phi_t = self.theta + np.arctan2(
                self.b * np.sqrt(1 - x[0] ** 2) - yo, x[0] - xo
            )

            # Now ensure phi *only* spans the dayside.
            phi = self.sort_phi(np.array([phi_o, phi_t]), dayside=True)

            # LAMBDA
            # ------

            # Angle of intersection with occultor
            lam_o = np.arcsin((1 - self.ro ** 2 + self.bo ** 2) / (2 * self.bo))
            # There are always two points; always pick the one
            # that's on the dayside for definiteness
            if not self.on_dayside(np.cos(lam_o), np.sin(lam_o)):
                lam_o = np.pi - lam_o

            # Angle of intersection with the terminator
            lam_t = self.theta
            # There are always two points; always pick the one
            # that's inside the occultor
            if np.cos(lam_t) ** 2 + (np.sin(lam_t) - self.bo) ** 2 > self.ro ** 2:
                lam_t = np.pi + self.theta

            # Now ensure lam *only* spans the inside of the occultor.
            lam = self.sort_lam(np.array([lam_o, lam_t]))

            # XI
            # --

            # Angle of intersection with occultor
            xi_o = np.arctan2(np.sqrt(1 - x[0] ** 2), x[0])

            # Angle of intersection with the limb
            if (1 - xo) ** 2 + yo ** 2 < self.ro ** 2:
                xi_l = 0
            else:
                xi_l = np.pi

            # Now ensure xi *only* spans the inside of the occultor.
            xi = self.sort_xi(np.array([xi_l, xi_o]))

            # In all cases, we're computing the dayside occulted flux
            code = FLUX_DAY_OCC

        # P-T
        elif len(x) == 2:

            # Angles are easy
            lam = np.array([])
            phi = np.sort(
                (self.theta + np.arctan2(self.b * np.sqrt(1 - x ** 2) - yo, x - xo))
                % (2 * np.pi)
            )
            xi = np.sort(np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi))

            # Cases
            if self.bo <= 1 - self.ro:

                # No intersections with the limb

                # CASE 2

                phi = self.sort_phi(phi, dayside=True)
                xi = self.sort_xi(xi)
                code = FLUX_DAY_OCC

            else:

                # The occultor intersects the limb, so we need to
                # integrate along the simplest path.

                # Intersections with the limb (this is the same as `lam`)
                psi1 = np.arcsin((1 - self.ro ** 2 + self.bo ** 2) / (2 * self.bo))
                psi = np.sort(np.array([psi1, np.pi - psi1]) % (2 * np.pi))

                if phi[0] < psi[0] < psi[1] < phi[1]:

                    x = self.ro * np.cos(phi[1] + self.tol)
                    y = self.bo + self.ro * np.sin(phi[1] + self.tol)
                    if self.on_dayside(x, y):

                        # CASE 2
                        phi = self.sort_phi(phi, dayside=True)
                        xi = self.sort_xi(xi)
                        code = FLUX_DAY_OCC

                    else:

                        # CASE 3
                        phi = self.sort_phi(phi, dayside=False)
                        xi = self.sort_xi(xi)[::-1]
                        code = FLUX_NIGHT_OCC

                elif psi[0] < phi[0] < phi[1] < psi[1]:

                    x = self.ro * np.cos(phi[0] + self.tol)
                    y = self.bo + self.ro * np.sin(phi[0] + self.tol)
                    if self.on_dayside(x, y):

                        # CASE 4
                        phi = self.sort_phi(phi, dayside=True)
                        xi = self.sort_xi(xi)
                        code = FLUX_DAY_VIS

                        # DEBUG: Is this always needed?
                        phi = [phi[1], phi[0]]
                        xi = [xi[1], xi[0] - 2 * np.pi]

                    else:

                        raise NotImplementedError("TODO!")

                elif phi[0] < phi[1] < psi[0] < psi[1]:

                    x = self.ro * np.cos(phi[0] + self.tol)
                    y = self.bo + self.ro * np.sin(phi[0] + self.tol)
                    if self.on_dayside(x, y):

                        raise NotImplementedError("TODO!")

                    else:

                        raise NotImplementedError("TODO!")

                else:

                    breakpoint()

                    raise NotImplementedError("Unexpected branch.")

        # There's a pathological case with 4 roots we need to code up
        else:

            # TODO: Code this special case up
            raise NotImplementedError("TODO!")

        return phi, lam, xi, code

    def sort_phi(self, phi, dayside=True):
        # Sort a pair of `phi` angles according to the order
        # of the integration limits.
        phi1, phi2 = phi
        phi = np.array([phi1, phi2]) % (2 * np.pi)
        if phi[1] < phi[0]:
            phi[1] += 2 * np.pi
        x = self.ro * np.cos(phi[0] + self.tol)
        y = self.bo + self.ro * np.sin(phi[0] + self.tol)
        if dayside != self.on_dayside(x, y):
            phi = np.array([phi2, phi1]) % (2 * np.pi)
        if phi[1] < phi[0]:
            phi[1] += 2 * np.pi
        return phi

    def sort_xi(self, xi):
        # Sort a pair of `xi` angles according to the order
        # of the integration limits.
        xi1, xi2 = xi
        xi = np.array([xi1, xi2]) % (2 * np.pi)
        if xi[0] < xi[1]:
            xi[0] += 2 * np.pi
        x = np.cos(self.theta) * np.cos(xi[1] + self.tol) - self.b * np.sin(
            self.theta
        ) * np.sin(xi[1] + self.tol)
        y = np.sin(self.theta) * np.cos(xi[1] + self.tol) + self.b * np.cos(
            self.theta
        ) * np.sin(xi[1] + self.tol)
        if x ** 2 + (y - self.bo) ** 2 > self.ro ** 2:
            xi = np.array([xi2, xi1]) % (2 * np.pi)
        if xi[0] < xi[1]:
            xi[0] += 2 * np.pi
        return xi

    def sort_lam(self, lam):
        # Sort a pair of `lam` angles according to the order
        # of the integration limits.
        lam1, lam2 = lam
        lam = np.array([lam1, lam2]) % (2 * np.pi)
        if lam[1] < lam[0]:
            lam[1] += 2 * np.pi
        x = np.cos(lam[0] + self.tol)
        y = np.sin(lam[0] + self.tol)
        if x ** 2 + (y - self.bo) ** 2 > self.ro ** 2:
            lam = np.array([lam2, lam1]) % (2 * np.pi)
        if lam[1] < lam[0]:
            lam[1] += 2 * np.pi
        return lam


# CASE 1
args = [
    # b, theta, bo, ro
    [0.4, np.pi / 3, 0.5, 0.7],
    [-0.4, np.pi / 3, 0.5, 0.7],
    [0.4, 2 * np.pi - np.pi / 3, 0.5, 0.7],
    [-0.4, 2 * np.pi - np.pi / 3, 0.5, 0.7],
    [0.4, np.pi / 2, 0.5, 0.7],
    [0.4, np.pi / 2, 1.0, 0.2],
    [0.00001, np.pi / 2, 0.5, 0.7],
    [0, np.pi / 2, 0.5, 0.7],
    [0.4, -np.pi / 2, 0.5, 0.7],
    [-0.4, np.pi / 2, 0.5, 0.7],
]


case2 = [
    [0.4, np.pi / 6, 0.3, 0.3],
]

case3 = [
    [0.4, np.pi / 6, 0.6, 0.5],
]

case4 = [
    [-0.95, 0.0, 2.0, 2.5],
]

# BROKEN
case4 = [
    [-0.1, np.pi / 6, 0.6, 0.75],
]

for arg in case4:

    #
    b, theta, bo, ro = arg
    x = np.linspace(-1, 1, 1000)
    y = b * np.sqrt(1 - x ** 2)
    x_t = x * np.cos(theta) - y * np.sin(theta)
    y_t = x * np.sin(theta) + y * np.cos(theta)
    fig, axis = plt.subplots(1)
    axis.axis("off")
    axis.add_artist(plt.Circle((0, bo), ro, fill=False))
    axis.add_artist(plt.Circle((0, 0), 1, fill=False))
    axis.plot(x_t, y_t, "k-", lw=1)
    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-1.25, 1.25)
    axis.set_aspect(1)
    plt.show()

    N = Numerical([0, 0, 0], *arg)
    print("{:5.3f} / {:5.3f}".format(N.flux(), N.flux_brute()))
    N.visualize()

