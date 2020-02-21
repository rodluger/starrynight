from starrynight import StarryNight
from starrynight.geometry import get_angles
import numpy as np
from starry._core.ops.polybasis import pTOp
from scipy.integrate import quad
import theano


__all__ = ["Brute", "Numerical"]


class Brute(StarryNight):
    """Compute the flux using brute force grid integration."""

    def __init__(self, *args, res=4999, **kwargs):

        # Initialize
        super().__init__(*args, **kwargs)

        # Extra args
        self.res = res

        # Poly basis op
        x = theano.tensor.dvector()
        y = theano.tensor.dvector()
        z = theano.tensor.dvector()
        self.pT = theano.function([x, y, z], pTOp(self.ops.pT, self.ydeg + 1)(x, y, z),)

    def precompute(self, b, theta, bo, ro):
        # Ingest
        self.ingest(b, theta, bo, ro)

        # Illumination matrix
        self.IA1 = self.illum().dot(self.A1)

    def design_matrix(self, b, theta, bo, ro):

        # Pre-compute expensive stuff
        self.precompute(b, theta, bo, ro)

        # Grid up
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
        X = self.pT(xpt, ypt, zpt).dot(self.IA1)
        return 4 * np.sum(X[cond1 & cond2 & cond3], axis=0) / (self.res ** 2)


class Numerical(StarryNight):
    """Compute the flux using Green's theorem and numerically solving the primitive integrals."""

    def __init__(self, *args, epsabs=1e-12, epsrel=1e-12, **kwargs):
        self.epsabs = epsabs
        self.epsrel = epsrel
        super().__init__(*args, **kwargs)

    def precompute(self, b, theta, bo, ro):
        # Ingest
        self.ingest(b, theta, bo, ro)

        # Get integration code & limits
        self.kappa, self.lam, self.xi, self.code = get_angles(
            self.b, self.theta, self.costheta, self.sintheta, self.bo, self.ro
        )
        self.phi = self.kappa - np.pi / 2

        # Illumination matrix
        self.IA1 = self.illum().dot(self.A1)

        # Compute the three primitive integrals
        self.P = np.zeros((self.ydeg + 2) ** 2)
        self.Q = np.zeros((self.ydeg + 2) ** 2)
        self.T = np.zeros((self.ydeg + 2) ** 2)
        n = 0
        for l in range(self.ydeg + 2):
            for m in range(-l, l + 1):
                self.P[n] = self.Plm(l, m)
                self.Q[n] = self.Qlm(l, m)
                self.T[n] = self.Tlm(l, m)
                n += 1

    def Glm(self, l, m):
        mu = l - m
        nu = l + m

        # NOTE: The abs prevents NaNs when the argument of the sqrt is
        # zero but floating point error causes it to be ~ -eps.
        z = lambda x, y: np.maximum(1e-12, np.sqrt(np.abs(1 - x ** 2 - y ** 2)))

        if nu % 2 == 0:
            G = [lambda x, y: 0, lambda x, y: x ** (0.5 * (mu + 2)) * y ** (0.5 * nu)]
        elif (l == 1) and (m == 0):

            def G0(x, y):
                z_ = z(x, y)
                if z_ > 1 - 1e-8:
                    return -0.5 * y
                else:
                    return (1 - z_ ** 3) / (3 * (1 - z_ ** 2)) * (-y)

            def G1(x, y):
                z_ = z(x, y)
                if z_ > 1 - 1e-8:
                    return 0.5 * x
                else:
                    return (1 - z_ ** 3) / (3 * (1 - z_ ** 2)) * x

            G = [G0, G1]

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
        G = self.Glm(l, m)
        func = lambda theta: G[0](x(theta), y(theta)) * dx(theta) + G[1](
            x(theta), y(theta)
        ) * dy(theta)
        res, _ = quad(func, theta1, theta2, epsabs=self.epsabs, epsrel=self.epsrel,)
        return res

    def Qlm(self, l, m):
        """Compute the Q integral numerically from its integral definition."""
        res = 0
        for lam1, lam2 in self.lam.reshape(-1, 2):
            x = lambda lam: np.cos(lam)
            y = lambda lam: np.sin(lam)
            dx = lambda lam: -np.sin(lam)
            dy = lambda lam: np.cos(lam)
            res += self.primitive(l, m, x, y, dx, dy, lam1, lam2)
        return res

    def Tlm(self, l, m):
        """Compute the T integral numerically from its integral definition."""
        res = 0
        for xi1, xi2 in self.xi.reshape(-1, 2):
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
            res += self.primitive(l, m, x, y, dx, dy, xi1, xi2)
        return res

    def Plm(self, l, m):
        """Compute the P integral numerically from its integral definition."""
        res = 0
        for phi1, phi2 in self.phi.reshape(-1, 2):
            x = lambda phi: self.ro * np.cos(phi)
            y = lambda phi: self.bo + self.ro * np.sin(phi)
            dx = lambda phi: -self.ro * np.sin(phi)
            dy = lambda phi: self.ro * np.cos(phi)
            res += self.primitive(l, m, x, y, dx, dy, phi1, phi2)
        return res
