from scipy.special import binom

__all__ = ["Vieta"]


def Vieta(i, u, v, delta):
    """Compute the Vieta coefficient A_{i, u, v}."""
    # TODO: Do recursively
    j1 = max(0, u - i)
    j2 = min(u + v - i, u)
    return sum(
        [
            float(binom(u, j))
            * float(binom(v, u + v - i - j))
            * (-1) ** (u + j)
            * delta ** (u + v - i - j)
            for j in range(j1, j2 + 1)
        ]
    )
