
/**
\file starrynight.h
\brief Solver for occultations of bodies with a night side (i.e., in reflected light).

*/

#ifndef _STARRY_NIGHT_H_
#define _STARRY_NIGHT_H_

#include "constants.h"
#include "utils.h"
#include "geometry.h"
#include "primitive.h"

namespace starry {
namespace night {

using namespace utils;
using namespace geometry;
using namespace primitive;

/**
    Compute the (unweighted) solution vector s_0^T. The actual solution vector
    s is computed from this as

        s^T = s_0^T . A2 . I . A2^-1

    where A2 is the change of basis matrix from Green's polynomials to 
    monomials, and I is the illumination matrix.

*/
template <typename S>
inline Vector<S> s0T(const int ydeg, const S& b, const S& theta, const S& bo, const S& ro, int& code) {

    // Total number of terms in `s0^T`
    int N = (ydeg + 2) * (ydeg + 2);

    // Get the angles of intersection between the
    // occultor and the occulted
    Vector<S> kappa, lam, xi;
    S costheta = cos(theta);
    S sintheta = sin(theta);
    code = get_angles(b, theta, costheta, sintheta, bo, ro, kappa, lam, xi);
    
    // Compute the solution vector
    Vector<S> result;

    // Does the occultor touch the terminator?
    if ((code != FLUX_ZERO) && (code != FLUX_SIMPLE_OCC) && (code != FLUX_SIMPLE_REFL) && (code != FLUX_SIMPLE_OCC_REFL)) {

        // Compute the primitive integrals
        Vector<S> PIntegral = P(ydeg + 1, bo, ro, kappa);
        Vector<S> QIntegral = Q(ydeg + 1, lam);
        Vector<S> TIntegral = T(ydeg + 1, b, theta, xi);
        result = PIntegral + QIntegral + TIntegral;

    } else {

        // We're done here: we'll use the standard starry algorithm instead!
        result.setZero(N);

    }

    return result;

}

/**

    Illumination matrix.

    TODO: We can backprop through this pretty easily.

*/
template <typename T>
inline Matrix<T> I(const int ydeg, const T& b, const T& theta) {

    int N2 = (ydeg + 2) * (ydeg + 2);
    int N1 = (ydeg + 1) * (ydeg + 1);
    Matrix<T> result(N2, N1);
    result.setZero();

    // NOTE: 3 / 2 is the starry normalization for reflected light maps
    T y0 = sqrt(1 - b * b);
    T x = -y0 * sin(theta);
    T y = y0 * cos(theta);
    T z = -b;
    Vector<T> p(4);
    p << 0, 1.5 * x, 1.5 * z, 1.5 * y;
    
    // Populate the matrix
    int n1 = 0;
    int n2 = 0;
    int l, n;
    bool odd1;
    for (int l1 = 0; l1 < ydeg + 1; ++l1) {
        for (int m1 = -l1; m1 < l1 + 1; ++m1) {
            if (is_even(l1 + m1)) odd1 = false;
            else odd1 = true;
            n2 = 0;
            for (int l2 = 0; l2 < 2; ++l2) {
                for (int m2 = -l2; m2 < l2 + 1; ++m2) {
                    l = l1 + l2;
                    n = l * l + l + m1 + m2;
                    if (odd1 && (!is_even(l2 + m2))) {
                        result(n - 4 * l + 2, n1) += p(n2);
                        result(n - 2, n1) -= p(n2);
                        result(n + 2, n1) -= p(n2);
                    } else {
                        result(n, n1) += p(n2);
                    }
                    n2 += 1;
                }
            }
            n1 += 1;
        }
    }
    return result;

}

} // namespace night
} // namespace starry

#endif