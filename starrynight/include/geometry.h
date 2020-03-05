/**
\file geometry.h
\brief Circle-ellipse intersection stuff.

*/

#ifndef _STARRY_GEOMETRY_H_
#define _STARRY_GEOMETRY_H_

#include "constants.h"
#include "utils.h"

namespace starry {
namespace geometry {

using namespace utils;

/**
    Return True if a point is on the dayside.
*/
template <typename T>
inline bool on_dayside(const T& b, const T& theta, const T& costheta, const T& sintheta, const T& x, const T& y) {
    if (x * x + y * y > 1)
        throw std::runtime_error("Point not on the unit disk.");
    T xr = x * costheta + y * sintheta;
    T yr = -x * sintheta + y * costheta;
    T term = 1 - xr * xr;
    T yt = b * sqrt(term);
    return bool(yr >= yt);
}

/**
    Sort a pair of `phi` angles.
    
    The basic rule here: the direction phi1 -> phi2 must 
    always span the dayside.
*/
template <typename T>
inline Vector<T> sort_phi(const T& b, const T& theta, const T& costheta, const T& sintheta, const T& bo, const T& ro, const Vector<T>& phi_) {

    // First ensure the range is correct
    T phi1 = angle_mod(phi_(0), T(2.0 * pi<T>()));
    T phi2 = angle_mod(phi_(1), T(2.0 * pi<T>()));
    Vector<T> phi(2);
    phi << phi1, phi2;
    if (phi(1) < phi(0))
        phi << phi(0), phi(1) + 2 * pi<T>();

    // Now take the midpoint and check that it's on-planet and on the
    // dayside. If not, we swap the integration limits.
    T phim = phi.mean();
    T x = ro * cos(phim);
    T y = bo + ro * sin(phim);
    if ((x * x + y * y > 1) || !on_dayside(b, theta, costheta, sintheta, x, y))
        phi << angle_mod(phi2, T(2.0 * pi<T>())), angle_mod(phi1, T(2.0 * pi<T>()));
    if (phi(1) < phi(0))
        phi << phi(0), phi(1) + 2 * pi<T>();
    return phi;

}

/**
    Sort a pair of `xi` angles.
    
    The basic rule here: the direction xi2 --> xi1 must
    always span the inside of the occultor. (Note that
    the limits of the `T` integral are xi2 --> xi1, since
    we integrate *clockwise* along the arc.) Since xi
    is limited to the range [0, pi], enforcing this is
    actually trivial: we just need to make sure they are
    arranged in decreasing order.
*/
template <typename T>
inline Vector<T> sort_xi(const T& b, const T& theta, const T& costheta, const T& sintheta, const T& bo, const T& ro, const Vector<T>& xi_) {

    T xi1 = angle_mod(xi_(0), T(2.0 * pi<T>()));
    T xi2 = angle_mod(xi_(1), T(2.0 * pi<T>()));
    Vector<T> xi(2);
    xi << xi1, xi2;
    if (xi(0) < xi(1))
        xi << xi(1), xi(0) + 2 * pi<T>();
    return xi;

}


/**
    Sort a pair of `lam` angles.
    
    The basic rule here: the direction lam1 --> lam2
    must always span the inside of the occultor and the
    dayside.
*/
template <typename T>
inline Vector<T> sort_lam(const T& b, const T& theta, const T& costheta, const T& sintheta, const T& bo, const T& ro, const Vector<T>& lam_) {

    // First ensure the range is correct
    T lam1 = angle_mod(lam_(0), T(2.0 * pi<T>()));
    T lam2 = angle_mod(lam_(1), T(2.0 * pi<T>()));
    Vector<T> lam(2);
    lam << lam1, lam2;
    if (lam(1) < lam(0))
        lam << lam(0), lam(1) + 2 * pi<T>();

    // Now take the midpoint and ensure it is inside
    // the occultor *and* on the dayside. If not, swap
    // the integration limits.
    T lamm = lam.mean();
    T x = cos(lamm);
    T y = sin(lamm);
    if ((x * x + (y - bo) * (y - bo) > ro * ro) || !on_dayside(b, theta, costheta, sintheta, (1 - STARRY_ANGLE_TOL) * x, (1 - STARRY_ANGLE_TOL) * y))
        lam << angle_mod(lam2, T(2.0 * pi<T>())), angle_mod(lam1, T(2.0 * pi<T>()));
    if (lam(1) < lam(0))
        lam << lam(0), lam(1) + 2 * pi<T>();
    return lam;

}

/**
    Compute the points of intersection between the occultor and the terminator.

*/
template <typename T>
inline Vector<T> get_roots(const T& b_, const T& theta_, const T& costheta_, const T& sintheta_, const T& bo_, const T& ro_) {

    // Get the *values*
    using Scalar = typename T::Scalar;
    using Complex = std::complex<Scalar>;
    Scalar b = b_.value();
    Scalar theta = theta_.value();
    Scalar costheta = costheta_.value();
    Scalar sintheta = sintheta_.value();
    Scalar bo = bo_.value();
    Scalar ro = ro_.value();

    // Roots and derivs
    int nroots = 0;
    Vector<Scalar> x(4), dxdb(4), dxdtheta(4), dxdbo(4), dxdro(4);

    // We'll solve for occultor-terminator intersections
    // in the frame where the semi-major axis of the
    // terminator ellipse is aligned with the x axis
    Scalar xo = bo * sintheta;
    Scalar yo = bo * costheta;

    // Special case: b = 0
    if (abs(b) < STARRY_B_ZERO_TOL) {

        // Roots
        Scalar term = sqrt(ro * ro - yo * yo);
        if (abs(xo + term) < 1)
            x(nroots++) = xo + term;
        if ((abs(xo - term) < 1) && term != 0)
            x(nroots++) = xo - term;

        // Derivatives
        int s = yo < 0 ? 1 : -1;
        for (int n = 0; n < nroots; ++n) {
            dxdb(n) = s * sqrt((1 - x(n) * x(n)) * (ro * ro - (x(n) - xo) * (x(n) - xo))) / (x(n) - xo);
            dxdtheta(n) = bo * (costheta - s * sqrt(ro * ro - (x(n) - xo) * (x(n) - xo)) / (x(n) - xo) * sintheta);
            dxdbo(n) = sintheta + s * sqrt(ro * ro - (x(n) - xo) * (x(n) - xo)) / (x(n) - xo) * costheta;
            dxdro(n) = ro / (x(n) - xo);
        }

    // Need to solve a quartic
    } else {

        // Get the roots (eigenvalue problem)
        // TODO: Speed up these computations
        Scalar A = (1 - b * b) * (1 - b * b);
        Scalar B = -4 * xo * (1 - b * b);
        Scalar C = -2 * (
            b * b * b * b
            + ro * ro
            - 3 * xo * xo
            - yo * yo
            - b * b * (1 + ro * ro - xo * xo + yo * yo)
        );
        Scalar D = -4 * xo * (b * b - ro * ro + xo * xo + yo * yo);
        Scalar E = (
            b * b * b * b
            - 2 * b * b * (ro * ro - xo * xo + yo * yo)
            + (ro * ro - xo * xo - yo * yo) * (ro * ro - xo * xo - yo * yo)
        );

        // TODO: actually compute the roots!
        Vector<Complex> roots(4); //np.roots([A, B, C, D, E]) + 0j

        // Polish the roots using Newton's method on the *original*
        // function, which is more stable than the quartic expression.
        Complex absfp, absfm, absf, minf, f, df, minx;
        Scalar p, q, v, w, t;
        int s;
        for (int n = 0; n < 4; ++n) {

            /*
            We're looking for the intersection of the function
            
                 y1 = b * sqrt(1 - x^2)
            
            and the function
            
                 y2 = yo +/- sqrt(ro^2 - (x - xo^2))
            
            Let's figure out which of the two cases (+/-) this
            root is a solution to. We're then going to polish
            the root by minimizing the function
            
                 f = y1 - y2
            */

            A = sqrt(1 - roots(n) * roots(n));
            B = sqrt(ro * ro - (roots(n) - xo) * (roots(n) - xo));
            absfp = abs(b * A - yo + B);
            absfm = abs(b * A - yo - B);

            if (absfp < absfm) {
                s = 1;
                absf = absfp;
            } else {
                s = -1;
                absf = absfm;
            }

            /*
            Some roots may instead correspond to
            
                 y = -b * sqrt(1 - x^2)
            
            which is the wrong half of the terminator ellipse.
            Let's only move forward if |f| is decently small.
            */

            if (absf < STARRY_ROOT_TOL_LOW) {

                // Apply Newton's method to polish the root
                minf = INFINITY;
                for (int k = 0; k < STARRY_ROOT_MAX_ITER; ++k) {
                    A = sqrt(1 - roots(n) * roots(n));
                    B = sqrt(ro * ro - (roots(n) - xo) * (roots(n) - xo));
                    f = b * A + s * B - yo;
                    absf = abs(f);
                    if (absf < minf) {
                        minf = absf;
                        minx = roots(n);
                        if (minf <= STARRY_ROOT_TOL_HIGH)
                            break;
                    }
                    df = -(b * roots(n) / A + s * (roots(n) - xo) / B);
                    roots(n) -= f / df;
                }

                // Only keep the root if the solver actually converged
                if (minf < STARRY_ROOT_TOL_MED) {

                    // Only keep the root if it's real
                    if ((minx.imag < STARRY_ROOT_TOL_HIGH) && (abs(minx.real) <= 1)) {

                        // Discard the (tiny) imaginary part
                        minx = minx.real;

                        // Check that we haven't included this root already
                        bool good = true;
                        for (int n = 0; n < nroots; ++n) {
                            if (abs(x(n) - minx) < STARRY_ROOT_TOL_DUP) {
                                good = false;
                                break;
                            }
                        }
                        if (good) {

                            // Store the root
                            x(nroots) = minx;

                            // Now compute its derivatives
                            q = sqrt(ro * ro - (minx - xo) * (minx - xo));
                            p = sqrt(1 - minx * minx);
                            v = (minx - xo) / q;
                            w = b / p;
                            t = 1.0 / (-w * minx - s * v);
                            dxdb(nroots) = -t * p;
                            dxdtheta(nroots) = -t * bo * (sintheta + s * v * costheta);
                            dxdbo(nroots) = t * (costheta - s * v * sintheta);
                            dxdro(nroots) = -t * s * ro / q;

                            // Move on
                            ++nroots;

                        }
                    }
                }
            }
        }
    }

    // Check if the extrema of the terminator ellipse are occulted
    bool e1 = costheta * costheta + (sintheta - bo) * (sintheta - bo) < ro * ro + STARRY_ROOT_TOL_HIGH;
    bool e2 = costheta * costheta + (sintheta + bo) * (sintheta + bo) < ro * ro + STARRY_ROOT_TOL_HIGH;

    // One is occulted, the other is not.
    // Usually we should have a single root, but
    // pathological cases with 3 roots (and maybe 4?)
    // are also possible.
    if ((e1 && (!e1)) || (e2 && (!e1))) {
        if ((nroots == 0) || (nroots == 2)) {
            // TODO: Fix this case if it ever shows up
            throw std::runtime_error("ERROR: Solver did not find the correct number of roots.");
        }
    }

    // There is one root but none of the extrema are occulted.
    // This likely corresponds to a grazing occultation of the
    // dayside or nightside.
    if (nroots == 1) {
        if ((!e1) && (!e2)) { 
            // TODO: Check this more rigorously? For now
            // we just delete the root.
            nroots = 0;
        }
    }
    
    // TODO: Figure out the return ADScalar value
    Vector<T> result;
    return result;

}   

} // namespace geometry
} // namespace starry

#endif

 