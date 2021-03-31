import numpy as np

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

# Maximum number of iterations when computing `el2` and `rj`
STARRY_EL2_MAX_ITER = 100
STARRY_CRJ_MAX_ITER = 100

# Tolerances in `rj`
STARRY_CRJ_LO_LIM = 2e-26
STARRY_CRJ_HI_LIM = 3e24
STARRY_CRJ_TOL = 2e-2

# Maximum number of iterations & tolerance when computing 2F1
STARRY_2F1_MAXITER = 200
STARRY_2F1_TOL = 1e-15

# Square root of the desired precision in `el2`
STARRY_EL2_CA = 1e-8

# Replace `inf` with this value in argument to `el2`
STARRY_HUGE_TAN = 1e15

# If |sin(theta)| or |cos(theta)| is less than this, set = 0
STARRY_T_TOL = 1e-12

# Low, medium, and high tolerance for root polishing
STARRY_ROOT_TOL_LOW = 1e-2
STARRY_ROOT_TOL_MED = 1e-10
STARRY_ROOT_TOL_HIGH = 1e-15

# Tolerance for duplicate roots
STARRY_ROOT_TOL_DUP = 1e-8

# Maximum number of root polishing iterations
STARRY_ROOT_MAX_ITER = 50

# If |b| is less than this value, set = 0
STARRY_B_ZERO_TOL = 1e-8

# Tolerance for various functions that calculate phi, xi, and lam
STARRY_ANGLE_TOL = 1e-13

# Hacks. Determining the integration paths close to the singular
# points of the occultation is quite hard, and the solution can
# often oscillate between two regimes. These tolerances prevent us
# from entering those regimes, at the cost of precision loss near
# these singular points.
STARRY_COMPLETE_OCC_TOL = 1e-8
STARRY_NO_OCC_TOL = 1e-8
STARRY_GRAZING_TOL = 1e-8

# Tolerance for the Pal (2012) solver, which is very unstable
STARRY_PAL_BO_EQUALS_RO_TOL = 1e-3
STARRY_PAL_BO_EQUALS_RO_MINUS_ONE_TOL = 1e-3
STARRY_PAL_BO_EQUALS_ONE_MINUS_RO_TOL = 1e-3

# Nudge k^2 away from 1 when it gets this close
STARRY_K2_ONE_TOL = 1e-12


def parity(i):
    return -1 if (i % 2) == 0 else 1


def pairdiff(x):
    """Return the sum over pairwise differences of an array.

    This is used to evaluate a (series of) definite integral(s) given
    the antiderivatives at each of the integration limits.
    """
    if len(x) > 1:
        if len(x) % 2 == 0:
            return sum(-np.array(x)[::2] + np.array(x)[1::2])
        else:
            raise ValueError("Array length must be even.")
    elif len(x) == 0:
        return 0.0
    else:
        return x
