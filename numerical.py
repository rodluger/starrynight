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
    def __init__(self, ydeg, epsabs=1e-12, epsrel=1e-12):

        # Instantiate
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.ydeg = ydeg
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

    def visualize(self, b, xo, yo, ro, tol=1e-8, res=999):

        # Impact parameter and occultor angle
        bo = np.sqrt(xo ** 2 + yo ** 2)
        omega = np.pi / 2 - np.arctan2(yo, xo)

        # Find angles of intersection
        phi, lam, xi = self.angles(b, xo, yo, ro, tol=tol)

        # Equation of ellipse
        x_t = np.linspace(-1, 1, 1000)
        y_t = b * np.sqrt(1 - x_t ** 2)

        # Shaded regions
        p = np.linspace(-1, 1, res)
        xpt, ypt = np.meshgrid(p, p)
        cond1 = (xpt - xo) ** 2 + (ypt - yo) ** 2 < ro ** 2  # inside occultor
        cond2 = xpt ** 2 + ypt ** 2 < 1  # inside occulted
        cond3 = ypt > b * np.sqrt(1 - xpt ** 2)  # above terminator
        img_day_occ = np.zeros_like(xpt)
        img_day_occ[cond1 & cond2 & cond3] = 1
        img_night = np.zeros_like(xpt)
        img_night[~cond1 & cond2 & ~cond3] = 1

        # Plot
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Draw shapes
        for axis in ax:
            axis.add_artist(plt.Circle((xo, yo), ro, fill=False))
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
            x = sorted(xo + ro * np.cos(phi))
            cond = (x_t > x[0]) & (x_t < x[1])
            ax[0].plot(x_t[cond], y_t[cond], "r-", lw=2)
            arc = Arc(
                (xo, yo),
                2 * ro,
                2 * ro,
                0,
                phi[0] * 180 / np.pi,
                phi[1] * 180 / np.pi,
                color="r",
                lw=2,
                zorder=3,
            )
            ax[1].add_patch(arc)

        # Draw points of intersection & angles
        sz = [0.4, 0.7]
        ax[0].plot([0, 1], [0, 0], "k-", alpha=0.5, lw=1)
        ax[1].plot([xo, xo + ro], [yo, yo], "k-", alpha=0.5, lw=1)

        for i, phi_i, xi_i in zip(range(len(phi)), phi, xi):
            ax[0].plot(
                [0, xo + ro * np.cos(phi_i)],
                [0, yo + ro * np.sin(phi_i)],
                color="C0",
                ls="--",
                lw=1,
            )
            ax[0].plot([0, np.cos(xi_i)], [0, np.sin(xi_i)], color="C0", lw=1)
            arc = Arc(
                (0, 0), sz[i], sz[i], 0, 0, xi_i * 180 / np.pi, color="C0", lw=0.5
            )
            ax[0].add_patch(arc)
            ax[0].annotate(
                r"$\xi_{}$".format(i + 1),
                xy=(0.5 * sz[i] * np.cos(0.5 * xi_i), 0.5 * sz[i] * np.sin(0.5 * xi_i)),
                xycoords="data",
                xytext=(7 * np.cos(0.5 * xi_i), 7 * np.sin(0.5 * xi_i)),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                color="C0",
            )
            ax[0].plot(
                [xo + ro * np.cos(phi_i)],
                [yo + ro * np.sin(phi_i)],
                "C0o",
                ms=4,
                zorder=4,
            )
            ax[0].plot([np.cos(xi_i)], [np.sin(xi_i)], "C0o", ms=4)
            ax[0].annotate(
                r"$x_{}$".format(i + 1),
                xy=(xo + ro * np.cos(phi_i), yo + ro * np.sin(phi_i)),
                xycoords="data",
                xytext=(7 * np.sign(np.cos(phi_i)), 7),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                color="C0",
            )
            ax[1].plot(
                [xo, xo + ro * np.cos(phi_i)],
                [yo, yo + ro * np.sin(phi_i)],
                color="C0",
                ls="-",
                lw=1,
            )
            ax[1].plot(
                [xo + ro * np.cos(phi_i)],
                [yo + ro * np.sin(phi_i)],
                "C0o",
                ms=4,
                zorder=4,
            )
            arc = Arc(
                (xo, yo), sz[i], sz[i], 0, 0, phi_i * 180 / np.pi, color="C0", lw=0.5,
            )
            ax[1].add_patch(arc)
            ax[1].annotate(
                r"$\phi_{}$".format(i + 1),
                xy=(
                    xo + 0.5 * sz[i] * np.cos(0.5 * phi_i),
                    yo + 0.5 * sz[i] * np.sin(0.5 * phi_i),
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

        plt.show()

    def flux_brute(self, y, b, xo, yo, ro, res=999):
        p = np.linspace(-1, 1, res)
        xpt, ypt = np.meshgrid(p, p)
        xpt = xpt.flatten()
        ypt = ypt.flatten()
        zpt = np.sqrt(1 - xpt ** 2 - ypt ** 2)
        cond1 = (xpt - xo) ** 2 + (ypt - yo) ** 2 < ro ** 2  # inside occultor
        cond2 = xpt ** 2 + ypt ** 2 < 1  # inside occulted
        cond3 = ypt > b * np.sqrt(1 - xpt ** 2)  # above terminator
        image = self.pT(xpt, ypt, zpt).dot(self.A1).dot(y)
        flux = 4 * np.sum(image[cond1 & cond2 & cond3]) / (res ** 2)
        return flux

    def flux(self, y, b, xo, yo, ro):

        phi, lam, xi = self.angles(b, xo, yo, ro)

        T = np.zeros((self.ydeg + 1) ** 2)
        P = np.zeros((self.ydeg + 1) ** 2)
        n = 0
        for l in range(self.ydeg + 1):
            for m in range(-l, l + 1):
                if len(phi):
                    P[n] = self.P(l, m, xo, yo, ro, phi[0], phi[1])
                if len(xi):
                    T[n] = self.T(l, m, b, xi[0], xi[1])
                n += 1

        return (P - T).dot(self.A).dot(y)

    def G(self, l, m):
        mu = l - m
        nu = l + m
        z = lambda x, y: np.sqrt(1 - x ** 2 - y ** 2)
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

    def Q(self, l, m, lam1, lam2):
        """Compute the Q integral numerically from its integral definition."""
        return self.T(l, m, 1, lam1, lam2)

    def T(self, l, m, b, xi1, xi2):
        """Compute the T integral numerically from its integral definition."""
        G = self.G(l, m)
        func = lambda xi: G[1](np.cos(xi), b * np.sin(xi)) * b * np.cos(xi) - G[0](
            np.cos(xi), b * np.sin(xi)
        ) * np.sin(xi)
        res, err = quad(func, xi1, xi2, epsabs=self.epsabs, epsrel=self.epsrel,)
        return res

    def P(self, l, m, xo, yo, ro, phi1, phi2):
        """Compute the S integral numerically from its integral definition."""
        G = self.G(l, m)
        func = (
            lambda phi: (
                G[1](xo + ro * np.cos(phi), yo + ro * np.sin(phi)) * np.cos(phi)
                - G[0](xo + ro * np.cos(phi), yo + ro * np.sin(phi)) * np.sin(phi)
            )
            * ro
        )
        res, err = quad(func, phi1, phi2, epsabs=self.epsabs, epsrel=self.epsrel,)
        return res

    def angles(self, b, xo, yo, ro, tol=1e-8):

        # Occultor-occulted (OO) intersections
        bo = np.sqrt(xo ** 2 + yo ** 2)
        sinlam = (1 - ro ** 2 + bo ** 2) / (2 * bo)
        y0 = sinlam
        x0 = np.sqrt(1 - sinlam ** 2)
        c = yo / bo
        s = -xo / bo
        x1 = x0 * c - y0 * s
        y1 = x0 * s + y0 * c
        x2 = -x0 * c - y0 * s
        y2 = -x0 * s + y0 * c
        oo_roots = [x1, x2]

        # TODO: Compute OO angles
        lam = np.array([])

        # Special case: b = 0 (TODO: tolerance)
        if b == 0:

            ot_roots = []
            term = np.sqrt(ro ** 2 - yo ** 2)
            x = xo + term
            if np.abs(x) < 1:
                ot_roots.append(x)
            x = xo - term
            if np.abs(x) < 1:
                ot_roots.append(x)

        else:

            # Occultor-terminator (OT) intersections
            A = (1 - b ** 2) ** 2
            B = -4 * xo * (1 - b ** 2)
            C = -2 * (
                b ** 4
                + ro ** 2
                - 3 * xo ** 2
                - yo ** 2
                - b ** 2 * (1 + ro ** 2 - xo ** 2 + yo ** 2)
            )
            D = -4 * xo * (b ** 2 - ro ** 2 + xo ** 2 + yo ** 2)
            E = (
                b ** 4
                - 2 * b ** 2 * (ro ** 2 - xo ** 2 + yo ** 2)
                + (ro ** 2 - xo ** 2 - yo ** 2) ** 2
            )

            # Get all real roots `x` that satisfy `sgn(y(x)) = sgn(b)`.
            ot_roots = np.roots([A, B, C, D, E])
            ot_roots = [x.real for x in ot_roots if np.abs(x.imag) < tol]
            ot_roots = [
                x
                for x in ot_roots
                if np.abs((x - xo) ** 2 + (b * np.sqrt(1 - x ** 2) - yo) ** 2 - ro ** 2)
                < tol
            ]

        # If there's only one root, add the endpoint
        if len(ot_roots) == 1:
            ot_roots += [np.sign(xo)]

        # There's a pathological case with 4 roots we need to code up
        if len(ot_roots) > 2:
            raise NotImplementedError("TODO!")

        # Compute OT angles
        if len(ot_roots):

            # Compute phi
            phi = np.array(
                [np.arctan2(b * np.sqrt(1 - x ** 2) - yo, x - xo) for x in ot_roots]
            )

            # Compute xi (TODO)
            xi = np.array([np.arctan2(np.sqrt(1 - x ** 2), x) for x in ot_roots])

        else:

            phi = np.array([])
            xi = np.array([])

        return phi, lam, xi


# DEBUG
xo = 0.3
yo = 0.1
ro = 0.5
b = 0.2
y = [0, 1, 0, 0]
N = Numerical(1)
print(N.flux(y, b, xo, yo, ro))
print(N.flux_brute(y, b, xo, yo, ro))
N.visualize(b, xo, yo, ro)
