from starrynight import Numerical, Analytic
from starrynight.geometry import get_angles
import numpy as np
import pytest

ydeg = 3
b, theta, bo, ro = [0.4, np.pi / 3, 0.5, 0.7]


A = Analytic(y=np.zeros((ydeg + 1) ** 2), tol=1e-7)
A.b, A.theta, A.bo, A.ro = b, theta, bo, ro
phi, _, _, _ = get_angles(A.b, A.theta, A.bo, A.ro)
phi1, phi2 = phi
for l in range(ydeg + 2):
    for m in range(-l, l + 1):
        if l == 1 and m == 0:
            P1 = np.nan
        else:
            P1 = A.P(l, m, phi1, phi2)
        P2 = A.Pnum(l, m, phi1, phi2)
        print("{:2d},{:2d}: {:7.4f} / {:7.4f}".format(l, m, P1, P2))
