"""


Singularities:

    - bo = 0
    - b0 = 0 and theta = 90 (only one root)
    - b0 <~ 0.1 and theta = 90 (root finding fails I think)

"""
import matplotlib.pyplot as plt
import numpy as np
from starry._c_ops import Ops
from starry._core.ops.rotation import dotROp
from starry._core.ops.integration import sTOp
from starry._core.ops.polybasis import pTOp
import theano
from scipy.integrate import quad
from sympy import binomial
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc


class Numerical(object):
    def __init__(self, y, b, theta, bo, ro, tol=1e-8, epsabs=1e-12, epsrel=1e-12):

        self.y = y
        self.ydeg = int(np.sqrt(len(y)) - 1)
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

        # Instantiate
        ops = Ops(self.ydeg, 0, 0, 0)

        # Basis transform from Ylm to Green's
        self.A = np.array(ops.A.todense())

        # Basis transform from Ylms to poly
        self.A1 = np.array(ops.A1.todense())

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
        self.pT = theano.function([x, y, z], pTOp(ops.pT, self.ydeg)(x, y, z),)

    def visualize(self, res=999):

        # Find angles of intersection
        phi, lam, xi = self.angles()

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
                cmap=LinearSegmentedColormap.from_list(
                    "cmap1", [(0, 0, 0, 0), "C1"], 2
                ),
            )
            axis.imshow(
                img_night,
                origin="lower",
                extent=(-1, 1, -1, 1),
                alpha=0.25,
                cmap=LinearSegmentedColormap.from_list("cmap1", [(0, 0, 0, 0), "k"], 2),
            )

        # Draw integration paths
        if len(phi):

            # T
            # This is the *actual* angle along the ellipse
            xi_p = np.arctan(b * np.tan(xi))
            # TODO: Check this criterion
            xi_p[xi_p < 0] += np.pi
            arc = Arc(
                (0, 0),
                2,
                2 * b,
                self.theta * 180 / np.pi,
                xi_p[1] * 180 / np.pi,
                xi_p[0] * 180 / np.pi,
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

        # Draw points of intersection & angles
        sz = [0.25, 0.5]
        for i, phi_i, xi_i, lam_i in zip(range(len(phi)), phi, xi, lam):

            # -- T --

            # xi angle
            ax[0].plot(
                [0, np.cos(xi_i + self.theta)],
                [0, np.sin(xi_i + self.theta)],
                color="C0",
                lw=1,
            )

            # tangent line
            if b > 0:
                x0 = np.cos(xi_i) * np.cos(self.theta)
                y0 = np.cos(xi_i) * np.sin(self.theta)
            else:
                x0 = self.ro * np.cos(phi_i)
                y0 = self.bo + self.ro * np.sin(phi_i)
            ax[0].plot(
                [x0, np.cos(xi_i + self.theta)],
                [y0, np.sin(xi_i + self.theta)],
                color="k",
                ls="--",
                lw=0.5,
            )

            # mark the polar angle
            ax[0].plot(
                [np.cos(xi_i + self.theta)],
                [np.sin(xi_i + self.theta)],
                "C0o",
                ms=4,
                zorder=4,
            )

            # draw and label the angle arc
            if np.sin(xi_i) != 0:
                angle = sorted([self.theta, xi_i + self.theta])
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
                        0.5 * sz[i] * np.cos(0.5 * xi_i + self.theta),
                        0.5 * sz[i] * np.sin(0.5 * xi_i + self.theta),
                    ),
                    xycoords="data",
                    xytext=(
                        7 * np.cos(0.5 * xi_i + self.theta),
                        7 * np.sin(0.5 * xi_i + self.theta),
                    ),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="C0",
                )

                # points of intersection
                ax[0].plot(
                    [self.ro * np.cos(phi[1 - i])],
                    [self.bo + self.ro * np.sin(phi[1 - i])],
                    "C0o",
                    ms=4,
                    zorder=4,
                )

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
                    0.5 * sz[i] * np.cos(0.5 * phi_i),
                    self.bo + 0.5 * sz[i] * np.sin(0.5 * phi_i),
                ),
                xycoords="data",
                xytext=(7 * np.cos(0.5 * phi_i), 7 * np.sin(0.5 * phi_i),),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                color="C0",
                zorder=4,
            )

            # -- Q --

            # points of intersection
            ax[2].plot(
                [0, np.cos(lam_i)], [0, np.sin(lam_i)], color="C0", ls="-", lw=1,
            )
            ax[2].plot(
                [np.cos(lam_i)], [np.sin(lam_i)], "C0o", ms=4, zorder=4,
            )
            ax[2].plot(
                [0, 0], "C0o", ms=4, zorder=4,
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

    def flux_brute(self, res=999):
        self._setup()
        p = np.linspace(-1, 1, res)
        xpt, ypt = np.meshgrid(p, p)
        xpt = xpt.flatten()
        ypt = ypt.flatten()
        zpt = np.sqrt(1 - xpt ** 2 - ypt ** 2)
        cond1 = xpt ** 2 + (ypt - self.bo) ** 2 < self.ro ** 2  # inside occultor
        cond2 = xpt ** 2 + ypt ** 2 < 1  # inside occulted
        xr = xpt * np.cos(self.theta) + ypt * np.sin(self.theta)
        yr = -xpt * np.sin(self.theta) + ypt * np.cos(self.theta)
        cond3 = yr > self.b * np.sqrt(1 - xr ** 2)  # above terminator
        image = self.pT(xpt, ypt, zpt).dot(self.A1).dot(self.y)
        flux = 4 * np.sum(image[cond1 & cond2 & cond3]) / (res ** 2)
        return flux

    def flux(self):

        self._setup()

        phi, lam, xi = self.angles()
        T = np.zeros((self.ydeg + 1) ** 2)
        P = np.zeros((self.ydeg + 1) ** 2)
        Q = np.zeros((self.ydeg + 1) ** 2)
        n = 0
        for l in range(self.ydeg + 1):
            for m in range(-l, l + 1):
                if len(phi):
                    P[n] = self.P(l, m, phi[0], phi[1])
                if len(xi):
                    T[n] = self.T(l, m, xi[0], xi[1])
                if len(lam):
                    Q[n] = self.Q(l, m, lam[0], lam[1])
                n += 1

        return (P + T + Q).dot(self.A).dot(y)

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

    def angles(self):

        # TODO: Use Sturm's theorem here

        # We'll solve for occultor-terminator intersections
        # in the frame where the semi-major axis of the
        # terminator ellipse is aligned with the x axis
        xo = self.bo * np.sin(self.theta)
        yo = self.bo * np.cos(self.theta)

        # Special case: b = 0
        if np.abs(self.b) < self.tol:

            x = np.array([])
            term = np.sqrt(self.ro ** 2 - yo ** 2)
            x = xo + term
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

        # No intersections with the terminator
        if len(x) == 0:

            raise RuntimeError("Occultor does not intersect the terminator.")

        # P-Q-T
        if len(x) == 1:

            # phi
            phi = np.array(
                [
                    np.pi
                    - np.arcsin(
                        (1 - self.ro ** 2 - self.bo ** 2) / (2 * self.bo * self.ro)
                    ),
                    theta + np.arctan2(self.b * np.sqrt(1 - x[0] ** 2) - yo, x[0] - xo),
                ]
            )

            # xi
            if (1 - xo) ** 2 + yo ** 2 < ro ** 2:
                x_xi = np.append(x, 1.0)
            elif (-1 - xo) ** 2 + yo ** 2 < ro ** 2:
                x_xi = np.append(x, -1.0)
            xi = np.arctan2(np.sqrt(1 - x_xi ** 2), x_xi)

            # lambda
            lam = np.array(
                [
                    self.theta,
                    np.pi
                    - np.arcsin((1 - self.ro ** 2 + self.bo ** 2) / (2 * self.bo)),
                ]
            )

            # TODO: What is the criterion for determining the order of these angles?

        # P-T
        elif len(x) == 2:

            # Sort from right to left
            x = x[np.argsort(x)[::-1]]
            phi = theta + np.arctan2(self.b * np.sqrt(1 - x ** 2) - yo, x - xo)
            xi = np.arctan2(np.sqrt(1 - x ** 2), x)
            lam = np.array([])

            # Ensure we're integrating counter-clockwise
            if phi[1] < phi[0]:
                phi[1] += 2 * np.pi

        # There's a pathological case with 4 roots we need to code up
        else:

            # TODO: Code this special case up
            raise NotImplementedError("TODO!")

        return phi, lam, xi


# DEBUG
b = 0.4
theta = np.pi / 3
bo = 0.5
ro = 0.7
y = [1, 1, 1, 1, 1, 1, 1, 1, 1]
N = Numerical(y, b, theta, bo, ro)
print(N.flux())
print(N.flux_brute())
N.visualize()
