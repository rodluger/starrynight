from mpmath import elliprf, elliprd, elliprj


def carlson_rf(x, y, z):
    # TODO: Code this up
    res = elliprf(x, y, z)
    return float(res.real)


def carlson_rd(x, y, z):
    # TODO: Code this up
    res = elliprd(x, y, z)
    return float(res.real)


def carlson_rj(x, y, z, p):
    # TODO: Code this up
    res = elliprj(x, y, z, p)
    return float(res.real)
