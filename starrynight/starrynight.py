"""
TODO: Singularities

    - bo = 0
    - bo = 0 and theta = 90 (only one root)
    - bo <~ 0.1 and theta = 90 (root finding fails I think)
    - check all edge cases

"""
from .utils import *
from .geometry import get_angles
from .primitive import compute_W, compute_U, compute_I, compute_J
from .vieta import Vieta
from .linear import pal
from .special import E, F
import numpy as np
import starry
from starry._c_ops import Ops
from starry._core.ops.rotation import dotROp
from scipy.integrate import quad
from scipy.special import binom
import theano


__all__ = ["StarryNight"]


class StarryNight(object):
    def __init__(self, ydeg, tol=1e-7):
        # Load kwargs
        self.ydeg = ydeg
        self.tol = tol

        # Instantiate the ops
        self.ops = Ops(self.ydeg + 1, 0, 0, 0)

        # Basis transform from poly to Green's
        self.A2 = np.array(theano.sparse.dot(self.ops.A, self.ops.A1Inv).eval())

        # Basis transform from Ylms to poly and back
        N = (self.ydeg + 1) ** 2
        self.A1 = np.array(self.ops.A1.todense())[:N, :N]
        self.A1Inv = np.array(self.ops.A1Inv.todense())

        # Z-rotation matrix
        theta = theano.tensor.dscalar()
        self.Rz = theano.function(
            [theta],
            dotROp(self.ops.dotR)(
                np.eye((self.ydeg + 1) ** 2),
                np.array(0.0),
                np.array(0.0),
                np.array(1.0),
                theta,
            ),
        )

        # Design matrix for emitted light (with occultation)
        xo = theano.tensor.dscalar()
        yo = theano.tensor.dscalar()
        ro = theano.tensor.dscalar()
        map = starry.Map(ydeg=self.ydeg + 1)
        self.Xe = theano.function(
            [xo, yo, ro], map.design_matrix(xo=xo, yo=yo, ro=ro)[0]
        )

        # Design matrix for reflected light (no occultation)
        xs = theano.tensor.dscalar()
        ys = theano.tensor.dscalar()
        zs = theano.tensor.dscalar()
        map_refl = starry.Map(ydeg=self.ydeg, reflected=True)
        self.Xr = theano.function(
            [xs, ys, zs], map_refl.design_matrix(xs=xs, ys=ys, zs=zs)[0]
        )

    def illum(self):
        # Illumination matrix
        y0 = np.sqrt(1 - self.b ** 2)
        x = -y0 * np.sin(self.theta)
        y = y0 * np.cos(self.theta)
        z = -self.b
        # NOTE: 3 / 2 is the starry normalization for reflected light maps
        p = np.array([0, x, z, y]) * 1.5
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

    def Xs(self):
        return self.Xe(0.0, self.bo, self.ro).dot(self.A1Inv.dot(self.IA1))

    def Xd(self):
        y0 = np.sqrt(1 - self.b ** 2)
        xs = -y0 * np.sin(self.theta)
        ys = y0 * np.cos(self.theta)
        zs = -self.b
        return self.Xr(xs, ys, zs)

    def Xn(self):
        y0 = np.sqrt(1 - self.b ** 2)
        xs = -y0 * np.sin(self.theta)
        ys = y0 * np.cos(self.theta)
        zs = -self.b
        return -self.Xr(-xs, -ys, -zs)

    def X(self):
        P = np.zeros((self.ydeg + 2) ** 2)
        Q = np.zeros((self.ydeg + 2) ** 2)
        T = np.zeros((self.ydeg + 2) ** 2)
        n = 0
        for l in range(self.ydeg + 2):
            for m in range(-l, l + 1):
                P[n] = self.P(l, m)
                Q[n] = self.Q(l, m)
                T[n] = self.T(l, m)
                n += 1
        return (P + Q + T).dot(self.A2).dot(self.IA1)

    def precompute(self, b, theta, bo, ro):
        # Ingest
        self.b = b
        self.theta = theta
        self.bo = bo
        self.ro = ro

        # Get integration code & limits
        self.phi, self.lam, self.xi, self.code = get_angles(
            self.b, self.theta, self.bo, self.ro, tol=self.tol
        )

        # Basic variables
        self.delta = (self.bo - self.ro) / (2 * self.ro)
        self.k2 = (1 - self.ro ** 2 - self.bo ** 2 + 2 * self.bo * self.ro) / (
            4 * self.bo * self.ro
        )
        self.k = np.sqrt(self.k2)
        self.km2 = 1.0 / self.k2
        self.kappa = self.phi + np.pi / 2
        self.fourbr15 = (4 * self.bo * self.ro) ** 1.5
        self.k3fourbr15 = self.k ** 3 * self.fourbr15
        self.tworo = np.empty(self.ydeg + 4)
        self.tworo[0] = 1.0
        for i in range(1, self.ydeg + 4):
            self.tworo[i] = self.tworo[i - 1] * 2 * self.ro

        # Illumination matrix
        self.IA1 = self.illum().dot(self.A1)

        # Pre-compute the primitive integrals
        x = 0.5 * self.kappa
        s1 = np.sin(x)
        s2 = s1 ** 2
        c1 = np.cos(x)
        q2 = 1 - np.minimum(1.0, s2 / self.k2)
        q3 = q2 ** 1.5
        self.dE = pairdiff(E(x, self.km2))
        self.dF = pairdiff(F(x, self.km2))
        self.U = compute_U(2 * self.ydeg + 5, s1)
        self.I = compute_I(self.ydeg + 3, self.kappa, s1, c1)
        self.J = compute_J(
            self.ydeg + 1,
            self.k2,
            self.km2,
            self.kappa,
            s1,
            s2,
            c1,
            q2,
            self.dE,
            self.dF,
        )
        self.W = compute_W(self.ydeg, s2, q2, q3)

    def design_matrix(self, b, theta, bo, ro):

        # Pre-compute expensive stuff
        self.precompute(b, theta, bo, ro)

        # All branches
        if self.code == FLUX_ZERO:
            return np.zeros((self.ydeg + 1) ** 2)
        elif self.code == FLUX_SIMPLE_OCC:
            return self.Xs()
        elif self.code == FLUX_SIMPLE_REFL:
            return self.Xd()
        elif self.code == FLUX_SIMPLE_OCC_REFL:
            return self.Xs() - self.Xn()
        elif self.code == FLUX_DAY_OCC:
            return self.Xd() - self.X()
        elif self.code == FLUX_NIGHT_OCC:
            return self.Xs() - (self.Xn() - self.X())
        elif self.code == FLUX_DAY_VIS:
            return self.X()
        elif self.code == FLUX_NIGHT_VIS:
            return self.Xs() - self.X()
        elif self.code == FLUX_TRIP_DAY_OCC:
            return self.Xd() - self.X()
        elif self.code == FLUX_TRIP_NIGHT_OCC:
            return self.Xs() - (self.Xn() - self.X())
        elif self.code == FLUX_QUAD_DAY_VIS:
            return self.X()
        elif self.code == FLUX_QUAD_NIGHT_VIS:
            return self.Xs() - self.X()
        else:
            raise NotImplementedError("Unexpected branch.")

    def flux(self, y, b, theta, bo, ro):
        return self.design_matrix(b, theta, bo, ro).dot(y)

    def K(self, u, v):
        """Return the integral K, evaluated as a sum over I."""
        return sum(
            [Vieta(i, u, v, self.delta) * self.I[i + u] for i in range(u + v + 1)]
        )

    def L(self, u, v, t):
        """Return the integral L, evaluated as a sum over J."""
        return self.k ** 3 * sum(
            [Vieta(i, u, v, self.delta) * self.J[i + u + t] for i in range(u + v + 1)]
        )

    def P(self, l, m):
        """Compute the P integral."""
        mu = l - m
        nu = l + m

        if (mu / 2) % 2 == 0:

            # Same as in starry
            return 2 * self.tworo[l + 2] * self.K((mu + 4) // 4, nu // 2)

        elif (mu == 1) and (l % 2 == 0):

            # Same as in starry
            return (
                self.tworo[l - 1]
                * self.fourbr15
                * (self.L((l - 2) // 2, 0, 0) - 2 * self.L((l - 2) // 2, 0, 1))
            )

        elif (mu == 1) and (l != 1) and (l % 2 != 0):

            # Same as in starry
            return (
                self.tworo[l - 1]
                * self.fourbr15
                * (self.L((l - 3) // 2, 1, 0) - 2 * self.L((l - 3) // 2, 1, 1))
            )

        elif ((mu - 1) % 2) == 0 and ((mu - 1) // 2 % 2 == 0) and (l != 1):

            # Same as in starry
            return (
                2
                * self.tworo[l - 1]
                * self.fourbr15
                * self.L((mu - 1) // 4, (nu - 1) // 2, 0)
            )

        elif (mu == 1) and (l == 1):

            # Same as in starry, but using expression from Pal (2012)
            # Note there's a difference of pi/2 between Pal's `phi` and ours
            s2 = 0.0
            for i in range(0, len(self.kappa), 2):
                s2 += pal(
                    self.bo, self.ro, self.kappa[i] - np.pi, self.kappa[i + 1] - np.pi
                )
            return s2

        else:
            """
            A note about these cases. In the original starry code, these integrals
            are always zero because the integrand is antisymmetric about the
            midpoint. Now, however, the integration limits are different, so 
            there's no cancellation in general.

            The cases below are just the first and fourth cases in equation (D25) 
            of the starry paper. We can re-write them as the first and fourth cases 
            in (D32) and (D35), respectively, but note that we pick up a factor
            of `sgn(cos(phi))`, since the power of the cosine term in the integrand
            is odd.
            
            The other thing to note is that `u` in the call to `K(u, v)` is now
            a half-integer, so our Vieta trick (D36, D37) doesn't work out of the box.
            """

            if nu % 2 == 0:

                res = 0
                u = int((mu + 4.0) // 4)
                v = int(nu / 2)
                for i in range(u + v + 1):
                    res += Vieta(i, u, v, self.delta) * self.U[2 * (u + i) + 1]
                return 2 * self.tworo[l + 2] * res

            else:

                res = 0
                u = (mu - 1) // 4
                v = (nu - 1) // 2
                for i in range(u + v + 1):
                    res += Vieta(i, u, v, self.delta) * self.W[i + u]
                return self.tworo[l - 1] * self.k3fourbr15 * res

    def Q(self, *args):
        raise NotImplementedError("TODO.")

    def T(self, *args):
        raise NotImplementedError("TODO.")
