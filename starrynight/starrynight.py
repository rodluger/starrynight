"""
TODO: Singularities

    - bo = 0
    - bo = 0 and theta = 90 (only one root)
    - bo <~ 0.1 and theta = 90 (root finding fails I think)
    - check all edge cases

"""
from .utils import *
from .geometry import get_angles
import numpy as np
import starry
from starry._c_ops import Ops
from starry._core.ops.rotation import dotROp
from starry._core.ops.integration import sTOp
from starry._core.ops.polybasis import pTOp
from scipy.integrate import quad
from scipy.special import binom
import theano
import warnings

warnings.simplefilter("ignore")
starry.config.quiet = True


__all__ = ["Brute", "Numerical", "Analytic"]


class StarryNight(object):
    def __init__(
        self,
        y=[1, 0, 0, 0],
        b=0.5,
        theta=0.0,
        bo=0.5,
        ro=0.1,
        tol=1e-7,
        epsabs=1e-12,
        epsrel=1e-12,
        res=4999,
    ):
        # Load kwargs
        self.y = np.array(y)
        self.ydeg = int(np.sqrt(len(self.y)) - 1)
        self.b = b
        self.theta = theta
        self.bo = bo
        self.ro = ro
        self.tol = tol
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.res = res

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

        # Design matrices
        xo = theano.tensor.dscalar()
        yo = theano.tensor.dscalar()
        ro = theano.tensor.dscalar()
        map = starry.Map(ydeg=self.ydeg + 1)
        self.design_matrix = theano.function(
            [xo, yo, ro], map.design_matrix(xo=xo, yo=yo, ro=ro),
        )

        xs = theano.tensor.dscalar()
        ys = theano.tensor.dscalar()
        zs = theano.tensor.dscalar()
        map_refl = starry.Map(ydeg=self.ydeg, reflected=True)
        self.design_matrix_refl = theano.function(
            [xs, ys, zs], map_refl.design_matrix(xs=xs, ys=ys, zs=zs),
        )

    def illum(self):
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

    def fs(self):
        y_refl = self.A1Inv.dot(self.illum()).dot(self.A1).dot(self.y)
        A = self.design_matrix(0.0, self.bo, self.ro)[0]
        return A.dot(y_refl)

    def fd(self):
        y0 = np.sqrt(1 - self.b ** 2)
        xs = -y0 * np.sin(self.theta)
        ys = y0 * np.cos(self.theta)
        zs = -self.b
        A = self.design_matrix_refl(xs, ys, zs)[0]
        return A.dot(self.y)

    def fn(self):
        y0 = np.sqrt(1 - self.b ** 2)
        xs = -y0 * np.sin(self.theta)
        ys = y0 * np.cos(self.theta)
        zs = -self.b
        A = -self.design_matrix_refl(-xs, -ys, -zs)[0]
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
        A = (P + Q + T).dot(self.A2).dot(self.illum()).dot(self.A1)
        return A.dot(self.y)

    def P(self, *args):
        raise NotImplementedError("This method must be subclassed.")

    def Q(self, *args):
        raise NotImplementedError("This method must be subclassed.")

    def T(self, *args):
        raise NotImplementedError("This method must be subclassed.")

    def flux(self):

        # Get integration code & limits
        phi, lam, xi, code = get_angles(
            self.b, self.theta, self.bo, self.ro, tol=self.tol
        )

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


class Brute(StarryNight):
    """Compute the flux using brute force grid integration."""

    def flux(self):
        p = np.linspace(-1, 1, self.res)
        xpt, ypt = np.meshgrid(p, p)
        xpt = xpt.flatten()
        ypt = ypt.flatten()
        zpt = np.sqrt(1 - xpt ** 2 - ypt ** 2)
        cond1 = xpt ** 2 + (ypt - self.bo) ** 2 > self.ro ** 2  # outside occultor
        cond2 = xpt ** 2 + ypt ** 2 < 1  # inside occulted
        xr = xpt * np.cos(self.theta) + ypt * np.sin(self.theta)
        yr = -xpt * np.sin(self.theta) + ypt * np.cos(self.theta)
        cond3 = yr > self.b * np.sqrt(1 - xr ** 2)  # above terminator
        image = self.pT(xpt, ypt, zpt).dot(self.illum()).dot(self.A1).dot(self.y)
        flux = 4 * np.sum(image[cond1 & cond2 & cond3]) / (self.res ** 2)
        return flux


class Numerical(StarryNight):
    """Compute the flux using Green's theorem and numerically solving the primitive integrals."""

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


class Analytic(StarryNight):
    """Compute the flux analytically."""

    def I(self, v, kappa1, kappa2):
        """Return the integral I."""
        # TODO: Compute in terms of elliptic integrals
        func = lambda x: np.sin(x) ** (2 * v)
        res, err = quad(
            func, 0.5 * kappa1, 0.5 * kappa2, epsabs=self.epsabs, epsrel=self.epsrel,
        )
        return res

    def J(self, v, kappa1, kappa2, k):
        """Return the integral J."""
        # TODO: Compute in terms of elliptic integrals
        func = lambda x: np.sin(x) ** (2 * v) * (1 - k ** (-2) * np.sin(x) ** 2) ** 1.5
        res, err = quad(
            func, 0.5 * kappa1, 0.5 * kappa2, epsabs=self.epsabs, epsrel=self.epsrel,
        )
        return res

    def V(self, i, u, v, delta):
        """Compute the Vieta coefficient A_{i, u, v}."""
        j1 = max(0, u - i)
        j2 = min(u + v - i, u)
        return sum(
            [
                float(binom(u, j))
                * float(binom(v, u + v - i - j))
                * (-1) ** (u + j)
                * delta ** (u + v - i - j)
                for j in range(j1, j2 + 1)
            ]
        )

    def K(self, u, v, kappa1, kappa2, delta):
        """Return the integral K, evaluated as a sum over I."""
        return sum(
            [
                self.V(i, u, v, delta) * self.I(i + u, kappa1, kappa2)
                for i in range(u + v + 1)
            ]
        )

    def L(self, u, v, t, kappa1, kappa2, delta, k):
        """Return the integral L, evaluated as a sum over J."""
        return k ** 3 * sum(
            [
                self.V(i, u, v, delta) * self.J(i + u + t, kappa1, kappa2, k)
                for i in range(u + v + 1)
            ]
        )

    def P(self, l, m, phi1, phi2):
        """Compute the P integral."""
        mu = l - m
        nu = l + m
        kappa1 = phi1 + np.pi / 2
        kappa2 = phi2 + np.pi / 2
        delta = (self.bo - self.ro) / (2 * self.ro)
        k = np.sqrt(
            (1 - self.ro ** 2 - self.bo ** 2 + 2 * self.bo * self.ro)
            / (4 * self.bo * self.ro)
        )
        if (mu / 2) % 2 == 0:
            return (
                2
                * (2 * self.ro) ** (l + 2)
                * self.K((mu + 4) // 4, nu // 2, kappa1, kappa2, delta)
            )
        elif (mu == 1) and (l % 2 == 0):
            return (
                (2 * self.ro) ** (l - 1)
                * (4 * self.bo * self.ro) ** (3.0 / 2.0)
                * (
                    self.L((l - 2) // 2, 0, 0, kappa1, kappa2, delta, k)
                    - 2 * self.L((l - 2) // 2, 0, 1, kappa1, kappa2, delta, k)
                )
            )
        elif (mu == 1) and (l != 1) and (l % 2 != 0):
            return (
                (2 * self.ro) ** (l - 1)
                * (4 * self.bo * self.ro) ** (3.0 / 2.0)
                * (
                    self.L((l - 3) // 2, 1, 0, kappa1, kappa2, delta, k)
                    - 2 * self.L((l - 3) // 2, 1, 1, kappa1, kappa2, delta, k)
                )
            )
        elif ((mu - 1) % 2) == 0 and ((mu - 1) // 2 % 2 == 0) and (l != 1):
            return (
                2
                * (2 * self.ro) ** (l - 1)
                * (4 * self.bo * self.ro) ** (3.0 / 2.0)
                * self.L((mu - 1) // 4, (nu - 1) // 2, 0, kappa1, kappa2, delta, k)
            )
        elif (mu == 1) and (l == 1):
            raise ValueError("This case is treated separately.")

        elif nu % 2 == 0:

            # TODO: Solve this analytically
            def K(u, v, kappa1, kappa2, delta):
                func = (
                    lambda phi: -np.sin(phi) ** (2 * u)
                    * (1 - np.sin(phi) ** 2) ** u
                    * (delta + np.sin(phi) ** 2) ** v
                )
                foo, _ = quad(
                    func,
                    0.5 * kappa1,
                    0.5 * kappa2,
                    epsabs=self.epsabs,
                    epsrel=self.epsrel,
                )
                return foo

            res = (
                2
                * (2 * self.ro) ** (l + 2)
                * K((mu + 4) / 4, nu // 2, kappa1, kappa2, delta)
            )

            return res

        else:

            # TODO: Solve this analytically
            def L0(u, v, kappa1, kappa2, delta, k):
                func = (
                    lambda phi: -(k ** 3)
                    * np.sin(phi) ** (2 * u)
                    * (1 - np.sin(phi) ** 2) ** u
                    * (delta + np.sin(phi) ** 2) ** v
                    * (1 - np.sin(phi) ** 2 / k ** 2) ** 1.5
                )
                foo, _ = quad(
                    func,
                    0.5 * kappa1,
                    0.5 * kappa2,
                    epsabs=self.epsabs,
                    epsrel=self.epsrel,
                )
                return foo

            res = (
                2
                * (2 * self.ro) ** (l - 1)
                * (4 * self.bo * self.ro) ** 1.5
                * L0((mu - 1) / 4, (nu - 1) // 2, kappa1, kappa2, delta, k)
            )

            return res

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

    def Pnum(self, l, m, phi1, phi2):
        """Compute the P integral numerically from its integral definition."""
        x = lambda phi: self.ro * np.cos(phi)
        y = lambda phi: self.bo + self.ro * np.sin(phi)
        dx = lambda phi: -self.ro * np.sin(phi)
        dy = lambda phi: self.ro * np.cos(phi)
        return self.primitive(l, m, x, y, dx, dy, phi1, phi2)

