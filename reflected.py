import matplotlib.pyplot as plt
import numpy as np
from starry._c_ops import Ops
from starry._core.ops.rotation import dotROp
from starry._core.ops.integration import sTOp
from starry._core.ops.polybasis import pTOp
import theano


epsabs = 1e-12
epsrel = 1e-12


class Reflected(object):
    def __init__(self, ydeg):
        # Instantiate
        self.ydeg = ydeg
        ops = Ops(self.ydeg, 0, 0, 0)

        # Basis transform from Ylm to Green's
        self.A = np.array(ops.A.todense())

        # Basiss transform from Ylms to poly
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

    def visualize(self, xo, yo, ro, b):
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.add_artist(plt.Circle((0, 0), 1, fill=False))
        ax.add_artist(plt.Circle((xo, yo), ro, fill=False))
        x = np.linspace(-1, 1, 1000)
        ax.plot(x, b * np.sqrt(1 - x ** 2))
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect(1)
        plt.show()

    def flux_thermal(self, y, xo, yo, ro):
        bo = np.sqrt(xo ** 2 + yo ** 2)
        tz = np.pi / 2 - np.arctan2(yo, xo)
        return self.sT(bo, ro).dot(self.A).dot(self.Rz(tz)).dot(y)[0]

    def I(self, v, kappa, rho=0):
        """Return the integral I, evaluated numerically."""
        func = lambda x: np.sin(x) ** (2 * v)
        res, err = quad(
            func,
            -0.5 * kappa + rho,
            0.5 * kappa,
            epsabs=self.epsabs,
            epsrel=self.epsrel,
        )
        return res

    def J(self, v, kappa, k, rho=0):
        """Return the integral J, evaluated numerically."""
        func = lambda x: np.sin(x) ** (2 * v) * (1 - k ** (-2) * np.sin(x) ** 2) ** 1.5
        res, err = quad(
            func,
            -0.5 * kappa + rho,
            0.5 * kappa,
            epsabs=self.epsabs,
            epsrel=self.epsrel,
        )
        return res

    def V(self, i, u, v, delta):
        """Compute the Vieta coefficient A_{i, u, v}."""
        j1 = max(0, u - i)
        j2 = min(u + v - i, u)
        return sum(
            [
                float(binomial(u, j))
                * float(binomial(v, u + v - i - j))
                * (-1) ** (u + j)
                * delta ** (u + v - i - j)
                for j in range(j1, j2 + 1)
            ]
        )

    def K(self, u, v, kappa, delta, rho=0):
        """Return the integral K, evaluated as a sum over I."""
        return sum(
            [
                self.V(i, u, v, delta) * self.I(i + u, kappa, rho)
                for i in range(u + v + 1)
            ]
        )

    def L(self, u, v, t, kappa, delta, k, rho=0):
        """Return the integral L, evaluated as a sum over J."""
        return k ** 3 * sum(
            [
                self.V(i, u, v, delta) * self.J(i + u + t, kappa, k, rho)
                for i in range(u + v + 1)
            ]
        )

    def P(self, l, m, b, r, rho=0):
        """Compute the P integral numerically (TODO)."""
        mu = l - m
        nu = l + m
        if (np.abs(1 - r) < b) and (b < 1 + r):
            phi = np.arcsin((1 - r ** 2 - b ** 2) / (2 * b * r))
        else:
            phi = np.pi / 2
        kappa = phi + np.pi / 2
        delta = (b - r) / (2 * r)
        k = np.sqrt((1 - r ** 2 - b ** 2 + 2 * b * r) / (4 * b * r))
        if (mu / 2) % 2 == 0:
            return (
                2
                * (2 * r) ** (l + 2)
                * self.K((mu + 4) // 4, nu // 2, kappa, delta, rho)
            )
        elif (mu == 1) and (l % 2 == 0):
            return (
                (2 * r) ** (l - 1)
                * (4 * b * r) ** (3.0 / 2.0)
                * (
                    self.L((l - 2) // 2, 0, 0, kappa, delta, k, rho)
                    - 2 * self.L((l - 2) // 2, 0, 1, kappa, delta, k, rho)
                )
            )
        elif (mu == 1) and (l != 1) and (l % 2 != 0):
            return (
                (2 * r) ** (l - 1)
                * (4 * b * r) ** (3.0 / 2.0)
                * (
                    self.L((l - 3) // 2, 1, 0, kappa, delta, k, rho)
                    - 2 * self.L((l - 3) // 2, 1, 1, kappa, delta, k, rho)
                )
            )
        elif ((mu - 1) % 2) == 0 and ((mu - 1) // 2 % 2 == 0) and (l != 1):
            return (
                2
                * (2 * r) ** (l - 1)
                * (4 * b * r) ** (3.0 / 2.0)
                * self.L((mu - 1) // 4, (nu - 1) // 2, 0, kappa, delta, k, rho)
            )
        elif (mu == 1) and (l == 1):
            raise ValueError("This case is treated separately.")
        else:
            return 0
