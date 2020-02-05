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


def pairdiff(x):
    return sum(-np.array(x)[::2] + np.array(x)[1::2])
