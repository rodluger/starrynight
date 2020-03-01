/**
\file iellip.h
\brief Incomplete elliptic integral computation.

Elliptic integrals computed following

    Bulirsch 1965, Numerische Mathematik, 7, 78
    Bulirsch 1965, Numerische Mathematik, 7, 353


*/

#ifndef _STARRY_IELLIP_H_
#define _STARRY_IELLIP_H_

#include "constants.h"
#include "utils.h"
#include <cmath>

namespace starry {
namespace iellip {

using std::abs;
using utils::mach_eps;
using utils::pi;

/**
Vectorized implementation of the `el2` function from
Bulirsch (1965). In this case, `x` is a *vector* of integration
limits. The halting condition does not depend on the value of `x`,
so it's much faster to evaluate all values of `x` at once!

*/

template <typename V, typename T> V el2(const V& x_, T kc, T a, T b) {

    if (kc == 0)
        throw std::runtime_error("Elliptic integral el2 did not converge because k = 1.");

    // We declare these params as vectors, 
    // but operate on them as arrays (because Eigen...)
    V c_, d_, p_, y_, f_, l_, g_, q_;
    f_ = x_ * 0;
    l_ = x_ * 0;
    auto x = x_.array();
    auto c = c_.array();
    auto d = d_.array();
    auto p = p_.array();
    auto y = y_.array();
    auto f = f_.array();
    auto l = l_.array();
    auto g = g_.array();
    auto q = q_.array();

    // Scalars
    T z, i, m, e, gp;
    int n;
    
    // Initial conditions
    c = x * x;
    d = c + 1.0;
    p = sqrt((1.0 + kc * kc * c) / d);
    d = x / d;
    c = d / (2 * p);
    z = a - b;
    i = a;
    a = (b + a) / 2;
    y = abs(1.0 / x);
    m = 1.0;

    // Iterate until convergence
    for (n = 0; i < STARRY_EL2_MAX_ITER; ++n) {

        b = i * kc + b;
        e = m * kc;
        g = e / p;
        d = f * g + d;
        f = c;
        i = a;
        p = g + p;
        c = (d / p + c) / 2;
        gp = m;
        m = kc + m;
        a = (b / m + a) / 2;
        y = -e / y + y;
        y = (y == 0).select(sqrt(e) * c * b, y);

        if (abs(gp - kc) > STARRY_EL2_CA * gp) {

            kc = sqrt(e) * 2;
            l = l * 2;
            l = (y  < 0).select(1.0 + l, l);

        } else {

            break;

        }

    }

    // Check for convergence
    if (n == STARRY_EL2_MAX_ITER - 1)
        throw std::runtime_error("Elliptic integral el2 did not converge.");

    l = (y  < 0).select(1.0 + l, l);
    q = (atan(m / y) + M_PI * l) * a / m;
    q = (x < 0).select(-q, q);
    return (q + c * z).matrix();

}


/**
Incomplete elliptic integral of the first kind

*/
template <typename V, typename T> V F(const V& tanphi, T k2) {

    T kc2 = 1 - k2;
    return el2(tanphi, sqrt(kc2), T(1.0), T(1.0));
  
}


/**
Incomplete elliptic integral of the second kind

*/
template <typename V, typename T> V E(const V& tanphi, T k2) {

    T kc2 = 1 - k2;
    return el2(tanphi, sqrt(kc2), T(1.0), kc2);
  
}


} // namespace iellip
} // namespace starry

#endif
