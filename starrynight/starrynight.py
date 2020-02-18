from .utils import *
from .geometry import get_angles
from .primitive import compute_P, compute_T, compute_Q
import numpy as np
import starry
from starry._c_ops import Ops
from starry._core.ops.rotation import dotROp
from scipy.integrate import quad
from scipy.special import binom
import theano


__all__ = ["StarryNight"]


class StarryNight(object):
    def __init__(self, ydeg):
        # Load kwargs
        self.ydeg = ydeg

        # Instantiate the ops
        self.ops = Ops(self.ydeg + 1, 0, 0, 0)

        # Basis transform from poly to Green's
        self.A2 = np.array(theano.sparse.dot(self.ops.A, self.ops.A1Inv).eval())

        # Basis transform from Ylms to poly and back
        N = (self.ydeg + 1) ** 2
        self.A1 = np.array(self.ops.A1.todense())[:N, :N]
        self.A1Inv = np.array(self.ops.A1Inv.todense())

        # Z-rotation matrix (for degree ydeg + 1)
        theta = theano.tensor.dscalar()
        self.Rz = theano.function(
            [theta],
            dotROp(self.ops.dotR)(
                np.eye((self.ydeg + 2) ** 2),
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
        return (self.P + self.Q + self.T).dot(self.A2).dot(self.IA1)

    def ingest(self, b, theta, bo, ro):
        self.b = b
        self.theta = theta % (2 * np.pi)
        self.costheta = np.cos(self.theta)
        self.sintheta = np.sin(self.theta)
        self.bo = bo
        self.ro = ro

    def precompute(self, b, theta, bo, ro):
        # Ingest
        self.ingest(b, theta, bo, ro)

        # Get integration code & limits
        self.kappa, self.lam, self.xi, self.code = get_angles(
            self.b, self.theta, self.costheta, self.sintheta, self.bo, self.ro,
        )

        # Illumination matrix
        self.IA1 = self.illum().dot(self.A1)

        # Compute the three primitive integrals if necessary
        if self.code not in [
            FLUX_ZERO,
            FLUX_SIMPLE_OCC,
            FLUX_SIMPLE_REFL,
            FLUX_SIMPLE_OCC_REFL,
        ]:
            self.P = compute_P(self.ydeg + 1, self.bo, self.ro, self.kappa)
            self.Q = compute_Q(self.ydeg + 1, self.lam)
            self.T = compute_T(self.ydeg + 1, self.b, self.theta, self.xi)
        else:
            self.P = None
            self.Q = None
            self.T = None

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

