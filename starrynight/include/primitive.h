/**
\file primitive.h
\brief Primitiv integrals for reflected light occultations.

*/

#ifndef _STARRY_PRIMITIVE_H_
#define _STARRY_PRIMITIVE_H_

#include "constants.h"
#include "utils.h"
#include "special.h"

namespace starry {
namespace primitive {

using namespace utils;
using namespace special;

/**

    Given s1 = sin(0.5 * kappa), compute the integral of

        cos(x) sin^v(x)

    from 0.5 * kappa1 to 0.5 * kappa2 recursively and return an array 
    containing the values of this function from v = 0 to v = vmax.

*/
template <typename T> 
inline Vector<T> U(const int vmax, Vector<T>& s1) {
    Vector<T> result(vmax + 1);
    result(0) = pairdiff(s1);
    Vector<T> term(s1.size());
    term.array() = s1.array() * s1.array();
    for (int v = 1; v < vmax + 1; ++v){
        result(v) = pairdiff(term) / (v + 1);
        term.array() *= s1.array();
    }
    return result;
}

/**
    Compute the helper integral I by upward recursion.

*/
template <typename T> 
inline Vector<T> I(const int nmax, Vector<T>& kappa, Vector<T>& s1, Vector<T>& c1) {
    Vector<T> result(nmax + 1);
    result(0) = 0.5 * pairdiff(kappa);
    Vector<T> s2(s1.size()), term(s1.size());
    s2.array() = s1.array() * s1.array();
    term.array() = s1.array() * s2.array();
    for (int v = 1; v < nmax + 1; ++v){
        result(v) = (1.0 / (2 * v)) * ((2 * v - 1) * result(v - 1) - pairdiff(term));
        term.array() *= s2.array();
    }
    return result;
}


/**
    Compute the expression

        s^(2n + 2) (3 / (n + 1) * 2F1(-1/2, n + 1, n + 2, 1 - q^2) + 2q^3) / (2n + 5)

    evaluated at n = [0 .. nmax], where

        s = sin(1/2 kappa)
        q = (1 - s^2 / k^2)^1/2

    by either upward recursion (stable for |1 - q^2| > 1/2) or downward 
    recursion (always stable).

*/
template <typename T> 
inline Vector<T> W_indef(const int nmax, const T& s2, const T& q2, const T& q3) {
    Vector<T> result(nmax + 1);

    if (abs(1 - q2) < 0.5) {

        // Setup
        T invs2 = 1 / s2;
        T z = (1 - q2) * invs2;
        T s2nmax = pow(s2, nmax);
        T x = q2 * q3 * s2nmax;

        // Upper boundary condition
        result(nmax) = (
            s2
            * s2nmax
            * (3 / (nmax + 1) * hyp2f1(-0.5, nmax + 1, nmax + 2, 1 - q2) + 2 * q3)
            / (2 * nmax + 5)
        );

        // Recurse down
        T f, A, B;
        for (int b = nmax - 1; b > -1; --b) {
            f = 1 / (b + 1);
            A = z * (1 + 2.5 * f);
            B = x * f;
            result(b) = A * result(b + 1) + B;
            x *= invs2;
        }

    } else {

        // Setup
        T z = s2 / (1 - q2);
        T x = -2 * q3 * (z - s2) * s2;

        // Lower boundary condition
        result(0) = (2 / 5) * (z * (1 - q3) + s2 * q3);

        // Recurse up
        T f, A, B;
        for (int b = 1; b < nmax; ++b) {
            f = 1 / (2 * b + 5);
            A = z * (2 * b) * f;
            B = x * f;
            result(b) = A * result(b - 1) + B;
            x *= s2;
        }

    }

    return result;
}

/**
    Compute the definite helper integral W from W_indef.

*/
template <typename T> 
inline Vector<T> W(const int nmax, const Vector<T>& s2, const Vector<T>& q2, const Vector<T>& q3) {
    size_t K = s2.size();
    Vector<T> result(K);
    result(K).setZero();
    for (size_t i = 0; i < K; i += 2) {
        result += W_indef(nmax, s2(i + 1), q2(i + 1), q3(i + 1)) - W_indef(nmax, s2(i), q2(i), q3(i));
    }
}

/**
    Compute the helper integral J.

    Returns the array J[0 .. nmax], computed recursively using
    a tridiagonal solver and a lower boundary condition
    (analytic in terms of elliptic integrals) and an upper
    boundary condition (computed numerically).

*/
template <typename T> 
inline Vector<T> J(const int nmax, const T& k2, const T& km2, const Vector<T>& kappa, 
                   const Vector<T>& s1, const Vector<T>& s2, const Vector<T>& c1, 
                   const Vector<T>& q2, const T& F, const T& E) {

    // Boundary conditions
    size_t K = kappa.size();
    Vector<T> z(K);
    z.array() = s1.array() * c1.array() * sqrt(q2.array());
    T resid = km2 * pairdiff(z);
    T f0 = (1.0 / 3.0) * (2 * (2 - km2) * E + (km2 - 1) * F + resid);
    T fN = J_numerical(nmax, k2, kappa);

    // Set up the tridiagonal problem
    Vector<T> a(nmax - 1), b(nmax - 1), c(nmax - 1);
    Vector<T> term(K);
    term.array() = k2 * z.array() * q2.array() * q2.array();
    T amp;
    int i = 0;
    for (int v = 2; v < nmax + 1; ++v) {
        amp = 1.0 / (2 * v + 3);
        a(i) = -2 * (v + (v - 1) * k2 + 1) * amp;
        b(i) = (2 * v - 3) * k2 * amp;
        c(i) = pairdiff(term) * amp;
        term.array() *= s2.array();
        ++i;
    }

    // Add the boundary conditions
    c(0) -= b(0) * f0;
    c(nmax - 2) -= fN;

    // Construct the tridiagonal matrix
    Matrix<T> A(nmax - 1, nmax - 1);
    A.setZero();
    A.diagonal(0) = a;
    A.diagonal(-1) = b.segment(1, nmax - 2);
    A.diagonal(1).setOnes();

    // Solve
    Vector<T> soln = A.lu().solve(c);

    // Append lower and upper boundary conditions
    Vector<T> result(nmax + 1);
    result(0) = f0;
    result(nmax) = fN;
    result.segment(1, nmax - 1) = soln;
    
    // We're done
    return result;

}


} // namespace primitive
} // namespace starry

#endif

 