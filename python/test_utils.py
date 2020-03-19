import numpy as np


def compare(expected, got):
    ydeg = int(np.sqrt(len(expected)) - 1)
    errors = []
    n = 0
    for l in range(ydeg + 1):
        for m in range(-l, l + 1):
            if not np.allclose(expected[n], got[n]):
                errors.append(
                    "ERROR for (l, m) = ({:2d}, {:2d}): expected {:11.8f}, got {:11.8f}.".format(
                        l, m, expected[n], got[n]
                    )
                )
            n += 1

    assert not errors, "\n{}".format("\n".join(errors))
