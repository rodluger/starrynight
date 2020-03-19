import numpy as np

# These are lists of [b, theta, bo, ro]
# corresponding to specific geometries
# we need to test.
CASE = [None for n in range(6)]

# Occultor does not touch the terminator
CASE[0] = [
    [0.5, 0.1, 1.2, 0.1],
    [0.5, 0.1, 0.1, 1.2],
    [0.5, 0.1, 0.8, 0.1],
    [0.5, 0.1, 0.9, 0.2],
    [0.5, np.pi + 0.1, 0.8, 0.1],
    [0.5, np.pi + 0.1, 0.9, 0.2],
    [0.5, 0.1, 0.5, 1.25],
    [0.5, np.pi + 0.1, 0.5, 1.25],
]

# Occultations involving all three primitive integrals
CASE[1] = [
    [0.4, np.pi / 3, 0.5, 0.7],
    [0.4, 2 * np.pi - np.pi / 3, 0.5, 0.7],
    [0.4, np.pi / 2, 0.5, 0.7],
    [0.4, np.pi / 2, 1.0, 0.2],
    [0.00001, np.pi / 2, 0.5, 0.7],
    [0, np.pi / 2, 0.5, 0.7],
    [0.4, -np.pi / 2, 0.5, 0.7],
    [-0.4, np.pi / 3, 0.5, 0.7],
    [-0.4, 2 * np.pi - np.pi / 3, 0.5, 0.7],
    [-0.4, np.pi / 2, 0.5, 0.7],
]

# Occultations involving only P and T
CASE[2] = [
    [0.4, np.pi / 6, 0.3, 0.3],
    [0.4, np.pi + np.pi / 6, 0.1, 0.6],
    [0.4, np.pi + np.pi / 3, 0.1, 0.6],
    [0.4, np.pi / 6, 0.6, 0.5],
    [0.4, -np.pi / 6, 0.6, 0.5],
    [0.4, 0.1, 2.2, 2.0],
    [0.4, -0.1, 2.2, 2.0],
    [0.4, np.pi + np.pi / 6, 0.3, 0.8],
    [0.75, np.pi + 0.1, 4.5, 5.0],
    [-0.95, 0.0, 2.0, 2.5],
    [-0.1, np.pi / 6, 0.6, 0.75],
    [-0.5, np.pi, 0.8, 0.5],
    [-0.1, 0.0, 0.5, 1.0],
]

# Occultations involving three points of intersection with the terminator
CASE[3] = [
    [0.5488316824842527, 4.03591586925189, 0.34988513192814663, 0.7753986686719786,],
    [
        0.5488316824842527,
        2 * np.pi - 4.03591586925189,
        0.34988513192814663,
        0.7753986686719786,
    ],
    [
        -0.5488316824842527,
        4.03591586925189 - np.pi,
        0.34988513192814663,
        0.7753986686719786,
    ],
    [
        -0.5488316824842527,
        2 * np.pi - (4.03591586925189 - np.pi),
        0.34988513192814663,
        0.7753986686719786,
    ],
]

# Occultations involving four points of intersection with the terminator
CASE[4] = [
    [0.5, np.pi, 0.99, 1.5],
    [-0.5, 0.0, 0.99, 1.5],
]

# Miscellaneous edge cases
CASE[5] = [
    [0.5, np.pi, 1.0, 1.5],
    [0.5, 2 * np.pi - np.pi / 4, 0.4, 0.4],
    [0.5, 2 * np.pi - np.pi / 4, 0.3, 0.3],
    [-0.25, 4 * np.pi / 3, 0.3, 0.3],
]

