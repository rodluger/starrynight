
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

} // namespace night
} // namespace starry

#endif