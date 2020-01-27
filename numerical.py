"""

Ranges:

    - 0 < bo < inf
    - 0 < ro < inf
    - 0 < theta < 2pi
    - -1 < b < 1

Singularities:

    - bo = 0
    - bo = 0 and theta = 90 (only one root)
    - bo <~ 0.1 and theta = 90 (root finding fails I think)
    - check all edge cases

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
import os
from tqdm import tqdm

warnings.simplefilter("ignore")
starry.config.quiet = True


# Integration codes
FLUX_ZERO = 0
FLUX_DAY_OCC = 1
FLUX_DAY_VIS = 2
FLUX_NIGHT_OCC = 3
FLUX_NIGHT_VIS = 4
FLUX_SIMPLE_OCC = 5
FLUX_SIMPLE_REFL = 6
FLUX_SIMPLE_OCC_REFL = 7
FLUX_TRIP_DAY_OCC = 8
FLUX_TRIP_NIGHT_OCC = 9
FLUX_QUAD_DAY_VIS = 10
FLUX_QUAD_NIGHT_VIS = 11


class Numerical(object):
    def __init__(self, y, b, theta, bo, ro, tol=1e-7, epsabs=1e-12, epsrel=1e-12):

        self.y = np.array(y)
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

    def visualize(self, res=4999, name=None):

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
        img_night_occ = np.zeros_like(xpt)
        img_night_occ[cond1 & cond2 & ~cond3] = 1
        img_night = np.zeros_like(xpt)
        img_night[~cond1 & cond2 & ~cond3] = 1

        # Plot
        if len(lam):
            fig, ax = plt.subplots(1, 3, figsize=(14, 5))
            fig.subplots_adjust(left=0.025, right=0.975, bottom=0.05, top=0.825)
            ax[0].set_title("T", color="r")
            ax[1].set_title("P", color="r")
            ax[2].set_title("Q", color="r")
        else:
            fig, ax = plt.subplots(1, 2, figsize=(9, 5))
            fig.subplots_adjust(left=0.025, right=0.975, bottom=0.05, top=0.825)
            ax[0].set_title("T", color="r")
            ax[1].set_title("P", color="r")

        # Solution
        flux = self.flux()
        flux_brute = self.flux_brute(res=res)
        if np.abs(flux - flux_brute) > 0.002:
            color = "r"
        else:
            color = "k"
        fig.suptitle("{:5.3f} / {:5.3f}".format(flux, flux_brute), color=color)

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
            axis.imshow(
                img_day_occ,
                origin="lower",
                extent=(-1, 1, -1, 1),
                alpha=0.25,
                cmap=LinearSegmentedColormap.from_list("cmap1", [(0, 0, 0, 0), "k"], 2),
            )
            axis.imshow(
                img_night_occ,
                origin="lower",
                extent=(-1, 1, -1, 1),
                alpha=0.5,
                cmap=LinearSegmentedColormap.from_list("cmap1", [(0, 0, 0, 0), "k"], 2),
            )
            axis.imshow(
                img_night,
                origin="lower",
                extent=(-1, 1, -1, 1),
                alpha=0.75,
                cmap=LinearSegmentedColormap.from_list("cmap1", [(0, 0, 0, 0), "k"], 2),
            )

        # Draw integration paths
        if len(phi):
            for k in range(0, len(phi) // 2 + 1, 2):

                # T
                # This is the *actual* angle along the ellipse
                xi_p = np.arctan(np.abs(self.b) * np.tan(xi))
                xi_p[xi_p < 0] += np.pi
                if np.abs(self.b) < 1e-4:
                    ax[0].plot(
                        [
                            np.cos(xi[k]) * np.cos(self.theta),
                            np.cos(xi[k + 1]) * np.cos(self.theta),
                        ],
                        [
                            np.cos(xi[k]) * np.sin(self.theta),
                            np.cos(xi[k + 1]) * np.sin(self.theta),
                        ],
                        color="r",
                        lw=2,
                        zorder=3,
                    )
                else:
                    if xi_p[k] > xi_p[k + 1]:
                        # TODO: CHECK ME
                        xi_p[[k, k + 1]] = xi_p[[k + 1, k]]
                    arc = Arc(
                        (0, 0),
                        2,
                        2 * np.abs(self.b),
                        self.theta * 180 / np.pi,
                        np.sign(self.b) * xi_p[k + 1] * 180 / np.pi,
                        np.sign(self.b) * xi_p[k] * 180 / np.pi,
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
                    phi[k] * 180 / np.pi,
                    phi[k + 1] * 180 / np.pi,
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
        if len(lam):
            ax[2].plot(
                [-1, 1], [0, 0], color="k", ls="--", lw=0.5,
            )
            ax[2].plot(0, 0, "C0o", ms=4, zorder=4)

        # Draw points of intersection & angles
        sz = [0.25, 0.5, 0.75, 1.0]
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

        if name is not None:
            if not os.path.exists("tmp"):
                os.mkdir("tmp")
            fig.savefig("tmp/{}.pdf".format(name))
        else:
            plt.show()
        plt.close()

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

    def flux_brute(self, res=4999):
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

    def fs(self):
        y_refl = self.A1Inv.dot(self.I()).dot(self.A1).dot(self.y)
        A = self.map.design_matrix(xo=0, yo=self.bo, ro=self.ro).eval()[0]
        return A.dot(y_refl)

    def fd(self):
        y0 = np.sqrt(1 - self.b ** 2)
        xs = -y0 * np.sin(self.theta)
        ys = y0 * np.cos(self.theta)
        zs = -self.b
        A = self.map_refl.design_matrix(xs=xs, ys=ys, zs=zs).eval()[0]
        return A.dot(self.y)

    def fn(self):
        y0 = np.sqrt(1 - self.b ** 2)
        xs = -y0 * np.sin(self.theta)
        ys = y0 * np.cos(self.theta)
        zs = -self.b
        A = -self.map_refl.design_matrix(xs=-xs, ys=-ys, zs=-zs).eval()[0]
        return A.dot(self.y)

    def f(self, phi, lam, xi):
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
        A = (P + Q + T).dot(self.A2).dot(self.I()).dot(self.A1)
        return A.dot(self.y)

    def flux(self):
        # Setup
        self._setup()

        # Get integration code & limits
        phi, lam, xi, code = self.angles()

        # All branches
        if code == FLUX_ZERO:
            return 0.0
        elif code == FLUX_SIMPLE_OCC:
            return self.fs()
        elif code == FLUX_SIMPLE_REFL:
            return self.fd()
        elif code == FLUX_SIMPLE_OCC_REFL:
            return self.fs() - self.fn()
        elif code == FLUX_DAY_OCC:
            return self.fd() - self.f(phi, lam, xi)
        elif code == FLUX_NIGHT_OCC:
            return self.fs() - (self.fn() - self.f(phi, lam, xi))
        elif code == FLUX_DAY_VIS:
            return self.f(phi, lam, xi)
        elif code == FLUX_NIGHT_VIS:
            return self.fs() - self.f(phi, lam, xi)
        elif code == FLUX_TRIP_DAY_OCC:
            return self.fd() - (
                self.f(phi[:2], [], xi[:2]) + self.f(phi[2:], lam, xi[2:])
            )
        elif code == FLUX_TRIP_NIGHT_OCC:
            return self.fs() - (
                self.fn() - (self.f(phi[:2], [], xi[:2]) + self.f(phi[2:], lam, xi[2:]))
            )
        elif code == FLUX_QUAD_DAY_VIS:
            return self.f(phi[:2], [], xi[:2]) + self.f(phi[2:], [], xi[:2])
        elif code == FLUX_QUAD_NIGHT_VIS:
            return self.fs() - self.f(phi[:2], [], xi[:2]) - self.f(phi[2:], [], xi[:2])
        else:
            raise NotImplementedError("Unexpected branch.")

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
        if x ** 2 + y ** 2 > 1:
            breakpoint()
            raise ValueError("Point not on the unit disk.")
        xr = x * np.cos(self.theta) + y * np.sin(self.theta)
        yr = -x * np.sin(self.theta) + y * np.cos(self.theta)
        term = 1 - xr ** 2
        yt = self.b * np.sqrt(term)
        return yr >= yt

    def angles(self):

        # Trivial cases
        if self.bo <= self.ro - 1:

            # Complete occultation
            return np.array([]), np.array([]), np.array([]), FLUX_ZERO

        elif self.bo >= 1 + self.ro:

            # No occultation
            return np.array([]), np.array([]), np.array([]), FLUX_SIMPLE_REFL

        # TODO: Use Sturm's theorem here to save time?

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
        l1 = x_l ** 2 + (y_l - self.bo) ** 2 < self.ro ** 2
        l2 = x_l ** 2 + (-y_l - self.bo) ** 2 < self.ro ** 2
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

            # Trivial: use the standard starry algorithm

            if np.abs(1 - self.ro) <= self.bo <= 1 + self.ro:

                # The occultor intersects the limb at this point
                lam = np.arcsin((1 - self.ro ** 2 + self.bo ** 2) / (2 * self.bo))
                x = (1 - self.tol) * np.cos(lam)
                y = (1 - self.tol) * np.sin(lam)

                if self.on_dayside(x, y):

                    # This point is guaranteed to be on the night side
                    # We're going to check if it's under the occultor or not
                    x = (1 - self.tol) * np.cos(self.theta + 3 * np.pi / 2)
                    y = (1 - self.tol) * np.sin(self.theta + 3 * np.pi / 2)

                    if x ** 2 + (y - self.bo) ** 2 <= self.ro ** 2:

                        # The occultor is blocking some daylight
                        # and all of the night side
                        code = FLUX_SIMPLE_OCC

                    else:

                        # The occultor is only blocking daylight
                        code = FLUX_SIMPLE_OCC_REFL

                else:

                    # This point is guaranteed to be on the day side
                    # We're going to check if it's under the occultor or not
                    x = (1 - self.tol) * np.cos(self.theta + np.pi / 2)
                    y = (1 - self.tol) * np.sin(self.theta + np.pi / 2)

                    if x ** 2 + (y - self.bo) ** 2 <= self.ro ** 2:

                        # The occultor is blocking some night side
                        # and all of the day side
                        code = FLUX_ZERO

                    else:

                        # The occultor is only blocking the night side
                        code = FLUX_SIMPLE_REFL
            else:

                # The occultor does not intersect the limb or the terminator
                if self.on_dayside(0, self.bo):

                    # The occultor is only blocking daylight
                    code = FLUX_SIMPLE_OCC_REFL

                else:

                    # The occultor is only blocking the night side
                    code = FLUX_SIMPLE_REFL

            return np.array([]), np.array([]), np.array([]), code

        # P-Q-T
        if len(x) == 1:

            # PHI
            # ---

            # Angle of intersection with occultor
            phi_o = np.arcsin(
                (1 - self.ro ** 2 - self.bo ** 2) / (2 * self.bo * self.ro)
            )
            # There are always two points; always pick the one
            # that's on the dayside for definiteness
            if not self.on_dayside(
                (1 - self.tol) * self.ro * np.cos(phi_o),
                (1 - self.tol) * (self.bo + self.ro * np.sin(phi_o)),
            ):
                phi_o = np.pi - phi_o

            # Angle of intersection with the terminator
            phi_t = self.theta + np.arctan2(
                self.b * np.sqrt(1 - x[0] ** 2) - yo, x[0] - xo
            )

            # Now ensure phi *only* spans the dayside.
            phi = self.sort_phi(np.array([phi_o, phi_t]))

            # LAMBDA
            # ------

            # Angle of intersection with occultor
            lam_o = np.arcsin((1 - self.ro ** 2 + self.bo ** 2) / (2 * self.bo))
            # There are always two points; always pick the one
            # that's on the dayside for definiteness
            if not self.on_dayside(
                (1 - self.tol) * np.cos(lam_o), (1 - self.tol) * np.sin(lam_o)
            ):
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

                # No intersections with the limb (easy)
                phi = self.sort_phi(phi)
                xi = self.sort_xi(xi)
                code = FLUX_DAY_OCC

            else:

                # The occultor intersects the limb, so we need to
                # integrate along the simplest path.

                # 1. Rotate the points of intersection into a frame where the
                # semi-major axis of the terminator ellipse lies along the x axis
                # We're going to choose xi[0] to be the rightmost point in
                # this frame, so that the integration is counter-clockwise along
                # the terminator to xi[1].
                x = np.cos(self.theta) * np.cos(xi) - self.b * np.sin(
                    self.theta
                ) * np.sin(xi)
                y = np.sin(self.theta) * np.cos(xi) + self.b * np.cos(
                    self.theta
                ) * np.sin(xi)
                xr = x * np.cos(self.theta) + y * np.sin(self.theta)
                if xr[1] > xr[0]:
                    xi = xi[::-1]

                # 2. Now we need the point corresponding to xi[1] to be the same as the
                # point corresponding to phi[0] in order for the path to be continuous
                x_xi1 = np.cos(self.theta) * np.cos(xi[1]) - self.b * np.sin(
                    self.theta
                ) * np.sin(xi[1])
                y_xi1 = np.sin(self.theta) * np.cos(xi[1]) + self.b * np.cos(
                    self.theta
                ) * np.sin(xi[1])
                x_phi = self.ro * np.cos(phi)
                y_phi = self.bo + self.ro * np.sin(phi)
                if np.argmin((x_xi1 - x_phi) ** 2 + (y_xi1 - y_phi) ** 2) == 1:
                    phi = phi[::-1]

                # 3. Compare the *curvature* of the two sides of the
                # integration area. The curvatures are similar (i.e., same sign)
                # when cos(theta) < 0, in which case we must integrate *clockwise* along P.
                if np.cos(self.theta) < 0:
                    # Integrate *clockwise* along P
                    if phi[0] < phi[1]:
                        phi[0] += 2 * np.pi
                else:
                    # Integrate *counter-clockwise* along P
                    if phi[1] < phi[0]:
                        phi[1] += 2 * np.pi

                # 4. Determine the integration code. Let's identify the midpoint
                # along each integration path and average their (x, y)
                # coordinates to determine what kind of region we are
                # bounding.
                xi_mean = np.mean(xi)
                x_xi = np.cos(self.theta) * np.cos(xi_mean) - self.b * np.sin(
                    self.theta
                ) * np.sin(xi_mean)
                y_xi = np.sin(self.theta) * np.cos(xi_mean) + self.b * np.cos(
                    self.theta
                ) * np.sin(xi_mean)
                phi_mean = np.mean(phi)
                x_phi = self.ro * np.cos(phi_mean)
                y_phi = self.bo + self.ro * np.sin(phi_mean)
                x = 0.5 * (x_xi + x_phi)
                y = 0.5 * (y_xi + y_phi)
                if self.on_dayside(x, y):
                    if x ** 2 + (y - self.bo) ** 2 < self.ro ** 2:
                        # Dayside under occultor
                        code = FLUX_DAY_OCC
                        # We need to reverse the integration path, since
                        # the terminator is *under* the arc along the limb
                        # and we should instead start at the *leftmost* xi
                        # value.
                        phi = phi[::-1]
                        xi = xi[::-1]
                    else:
                        # Dayside visible
                        code = FLUX_DAY_VIS
                        if self.b < 0:
                            phi = phi[::-1]
                            xi = xi[::-1]
                else:
                    if x ** 2 + (y - self.bo) ** 2 < self.ro ** 2:
                        # Nightside under occultor
                        code = FLUX_NIGHT_OCC
                    else:
                        # Nightside visible
                        code = FLUX_NIGHT_VIS

        # There's a pathological case with 3 roots
        elif len(x) == 3:

            # TODO: Clean these up a bit

            if self.b > 0:

                if (-1 - xo) ** 2 + yo ** 2 < self.ro ** 2:

                    x = np.sort(x)
                    x = np.array([x[2], x[1], x[0]])

                    phi = np.append(
                        self.theta
                        + np.arctan2(self.b * np.sqrt(1 - x ** 2) - yo, x - xo),
                        np.arcsin(
                            (1 - self.ro ** 2 - self.bo ** 2) / (2 * self.bo * self.ro)
                        ),
                    ) % (2 * np.pi)
                    for n in range(3):
                        while phi[n + 1] < phi[n]:
                            phi[n + 1] += 2 * np.pi

                    xi = np.append(
                        np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi), np.pi
                    )
                    xi = np.array([xi[1], xi[0], xi[3], xi[2]])

                    lam = np.array(
                        [
                            np.arcsin(
                                (1 - self.ro ** 2 + self.bo ** 2) / (2 * self.bo)
                            ),
                            np.pi + self.theta,
                        ]
                    ) % (2 * np.pi)
                    if lam[1] < lam[0]:
                        lam[1] += 2 * np.pi

                else:

                    x = np.sort(x)
                    x = np.array([x[1], x[0], x[2]])

                    phi = np.append(
                        self.theta
                        + np.arctan2(self.b * np.sqrt(1 - x ** 2) - yo, x - xo),
                        np.pi
                        - np.arcsin(
                            (1 - self.ro ** 2 - self.bo ** 2) / (2 * self.bo * self.ro)
                        ),
                    ) % (2 * np.pi)
                    phi[[2, 3]] = phi[[3, 2]]
                    for n in range(3):
                        while phi[n + 1] < phi[n]:
                            phi[n + 1] += 2 * np.pi

                    xi = np.append(
                        np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi), 0.0
                    )
                    xi = np.array([xi[1], xi[0], xi[2], xi[3]])

                    lam = np.array(
                        [
                            self.theta,
                            np.pi
                            - np.arcsin(
                                (1 - self.ro ** 2 + self.bo ** 2) / (2 * self.bo)
                            ),
                        ]
                    ) % (2 * np.pi)
                    if lam[1] < lam[0]:
                        lam[1] += 2 * np.pi

                code = FLUX_TRIP_DAY_OCC

            else:

                if (-1 - xo) ** 2 + yo ** 2 < self.ro ** 2:

                    x = np.sort(x)
                    x = np.array([x[1], x[2], x[0]])

                    phi = np.append(
                        self.theta
                        + np.arctan2(self.b * np.sqrt(1 - x ** 2) - yo, x - xo),
                        np.pi
                        - np.arcsin(
                            (1 - self.ro ** 2 - self.bo ** 2) / (2 * self.bo * self.ro)
                        ),
                    ) % (2 * np.pi)
                    phi[[2, 3]] = phi[[3, 2]]
                    for n in range(3):
                        while phi[n + 1] < phi[n]:
                            phi[n + 1] += 2 * np.pi

                    xi = np.append(
                        np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi), np.pi
                    )
                    xi = np.array([xi[1], xi[0], xi[2], xi[3]])

                    lam = np.array(
                        [
                            np.pi + self.theta,
                            np.pi
                            - np.arcsin(
                                (1 - self.ro ** 2 + self.bo ** 2) / (2 * self.bo)
                            ),
                        ]
                    ) % (2 * np.pi)
                    if lam[1] < lam[0]:
                        lam[1] += 2 * np.pi

                else:

                    print("FOO")

                    x = np.sort(x)

                    phi = np.append(
                        self.theta
                        + np.arctan2(self.b * np.sqrt(1 - x ** 2) - yo, x - xo),
                        np.arcsin(
                            (1 - self.ro ** 2 - self.bo ** 2) / (2 * self.bo * self.ro)
                        ),
                    ) % (2 * np.pi)
                    for n in range(3):
                        while phi[n + 1] < phi[n]:
                            phi[n + 1] += 2 * np.pi

                    xi = np.append(
                        np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi), 0.0
                    )
                    xi = np.array([xi[1], xi[0], xi[3], xi[2]])

                    lam = np.array(
                        [
                            np.arcsin(
                                (1 - self.ro ** 2 + self.bo ** 2) / (2 * self.bo)
                            ),
                            self.theta,
                        ]
                    ) % (2 * np.pi)
                    if lam[1] < lam[0]:
                        lam[1] += 2 * np.pi

                code = FLUX_TRIP_NIGHT_OCC

        # And a pathological case with 4 roots
        elif len(x) == 4:

            lam = np.array([])
            phi = np.sort(
                (self.theta + np.arctan2(self.b * np.sqrt(1 - x ** 2) - yo, x - xo))
                % (2 * np.pi)
            )
            phi = phi[1], phi[0], phi[3], phi[2]
            xi = np.sort(np.arctan2(np.sqrt(1 - x ** 2), x) % (2 * np.pi))

            if self.b > 0:
                code = FLUX_QUAD_NIGHT_VIS
            else:
                xi = xi[1], xi[0], xi[3], xi[2]
                code = FLUX_QUAD_DAY_VIS

        else:

            raise NotImplementedError("Unexpected branch.")

        return phi, lam, xi, code

    def sort_phi(self, phi):
        # Sort a pair of `phi` angles according to the order
        # of the integration limits.
        phi1, phi2 = phi
        phi = np.array([phi1, phi2]) % (2 * np.pi)
        if phi[1] < phi[0]:
            phi[1] += 2 * np.pi
        x = self.ro * np.cos(phi[0] + self.tol)
        y = self.bo + self.ro * np.sin(phi[0] + self.tol)
        if (x ** 2 + y ** 2 > 1) or not self.on_dayside(x, y):
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


def mc_search(seed=0, total=1000, epsrel=0.01, res=999):
    N = Numerical([1, 1, 1, 1], 0, 0, 0, 0)
    np.random.seed(seed)
    for i in tqdm(range(total)):
        N.bo = 0
        N.ro = 2
        while (N.bo <= N.ro - 1) or (N.bo >= 1 + N.ro):
            if np.random.random() > 0.5:
                N.ro = np.random.random() * 10
                N.bo = np.random.random() * 20
            else:
                N.ro = np.random.random()
                N.bo = np.random.random() * 2
        N.theta = np.random.random() * 2 * np.pi
        N.b = 1 - 2 * np.random.random()

        flux = N.flux()
        flux_brute = N.flux_brute(res=res)
        if np.abs(flux - flux_brute) > epsrel:
            N.visualize(name="{:04d}".format(i), res=res)


# b, theta, bo, ro

SIMPLE = [
    [0.5, 0.1, 1.2, 0.1],
    [0.5, 0.1, 0.1, 1.2],
    [0.5, 0.1, 0.8, 0.1],
    [0.5, 0.1, 0.9, 0.2],
    [0.5, np.pi + 0.1, 0.8, 0.1],
    [0.5, np.pi + 0.1, 0.9, 0.2],
    [0.5, 0.1, 0.5, 1.25],
    [0.5, np.pi + 0.1, 0.5, 1.25],
]

PQT = [
    # b > 0
    [0.4, np.pi / 3, 0.5, 0.7],
    [0.4, 2 * np.pi - np.pi / 3, 0.5, 0.7],
    [0.4, np.pi / 2, 0.5, 0.7],
    [0.4, np.pi / 2, 1.0, 0.2],
    [0.00001, np.pi / 2, 0.5, 0.7],
    [0, np.pi / 2, 0.5, 0.7],
    [0.4, -np.pi / 2, 0.5, 0.7],
    # b < 0
    [-0.4, np.pi / 3, 0.5, 0.7],
    [-0.4, 2 * np.pi - np.pi / 3, 0.5, 0.7],
    [-0.4, np.pi / 2, 0.5, 0.7],
]

PT = [
    # b > 0
    [0.4, np.pi / 6, 0.3, 0.3],
    [0.4, np.pi + np.pi / 6, 0.1, 0.6],
    [0.4, np.pi + np.pi / 3, 0.1, 0.6],
    [0.4, np.pi / 6, 0.6, 0.5],
    [0.4, -np.pi / 6, 0.6, 0.5],
    [0.4, 0.1, 2.2, 2.0],
    [0.4, -0.1, 2.2, 2.0],
    [0.4, np.pi + np.pi / 6, 0.3, 0.8],
    [0.75, np.pi + 0.1, 4.5, 5.0],
    # b < 0
    [-0.95, 0.0, 2.0, 2.5],
    [-0.1, np.pi / 6, 0.6, 0.75],
    [-0.5, np.pi, 0.8, 0.5],
    [-0.1, 0.0, 0.5, 1.0],
]

TRIP = [
    [0.5488316824842527, 4.03591586925189, 0.34988513192814663, 0.7753986686719786,],
    [
        0.5488316824842527,
        2 * np.pi - 4.03591586925189,
        0.34988513192814663,
        0.7753986686719786,
    ],
    [
        -0.5488316824842527,
        4.03591586925189 - np.pi,
        0.34988513192814663,
        0.7753986686719786,
    ],
    [
        -0.5488316824842527,
        2 * np.pi - (4.03591586925189 - np.pi),
        0.34988513192814663,
        0.7753986686719786,
    ],
]

QUAD = [
    [0.5, np.pi, 0.99, 1.5],
    [-0.5, 0.0, 0.99, 1.5],
]

# TODO
EDGE = [[0.5, np.pi, 1.0, 1.5]]


for args in TRIP:
    N = Numerical([1, 0, 0, 0], *args)
    print("{:5.3f} / {:5.3f}".format(N.flux(), N.flux_brute()))
