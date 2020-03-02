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
using namespace utils;


template <class T> inline Vector<T> F(const Vector<T>& tanphi, const T& k2);
template <class T> inline Vector<T> E(const Vector<T>& tanphi, const T& k2);



/**
Vectorized implementation of the `el2` function from
Bulirsch (1965). In this case, `x` is a *vector* of integration
limits. The halting condition does not depend on the value of `x`,
so it's much faster to evaluate all values of `x` at once!

*/
template <typename T> 
Vector<T> el2(const Vector<T>& x_, const T& kc_, const T& a_, const T& b_) {

    // Make copies
    T kc = kc_;
    T a = a_;
    T b = b_;

    if (kc == 0)
        throw std::runtime_error("Elliptic integral el2 did not converge because k = 1.");

    // We declare these params as vectors, 
    // but operate on them as arrays (because Eigen...)
    Vector<T> c_, d_, p_, y_, f_, l_, g_, q_;
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
    q = (atan(m / y) + pi<T>() * l) * a / m;
    q = (x < 0).select(-q, q);
    return (q + c * z).matrix();

}

/**
Scalar implementation of the Carlson elliptic integral RJ.

Based on
    
    Bille Carlson,
    Computing Elliptic Integrals by Duplication,
    Numerische Mathematik,
    Volume 33, 1979, pages 1-16.

    Bille Carlson, Elaine Notis,
    Algorithm 577, Algorithms for Incomplete Elliptic Integrals,
    ACM Transactions on Mathematical Software,
    Volume 7, Number 3, pages 398-403, September 1981
    
    https://people.sc.fsu.edu/~jburkardt/f77_src/toms577/toms577.f

*/
template <typename T> 
T rj(const T& x_, const T& y_, const T& z_, const T& p_) {
    
    // Constants
    const T C1 = 3.0 / 14.0;
    const T C2 = 1.0 / 3.0;
    const T C3 = 3.0 / 22.0;
    const T C4 = 3.0 / 26.0;

    // Make copies
    T x = x_;
    T y = y_;
    T z = z_;
    T p = p_;

    // Limit checks
    if (x < STARRY_CRJ_LO_LIM)
        x = STARRY_CRJ_LO_LIM;
    else if (x > STARRY_CRJ_HI_LIM)
        x = STARRY_CRJ_HI_LIM;

    if (y < STARRY_CRJ_LO_LIM)
        y = STARRY_CRJ_LO_LIM;
    else if (y > STARRY_CRJ_HI_LIM)
        y = STARRY_CRJ_HI_LIM;

    if (z < STARRY_CRJ_LO_LIM)
        z = STARRY_CRJ_LO_LIM;
    else if (z > STARRY_CRJ_HI_LIM)
        z = STARRY_CRJ_HI_LIM;

    if (p < STARRY_CRJ_LO_LIM)
        p = STARRY_CRJ_LO_LIM;
    else if (p > STARRY_CRJ_HI_LIM)
        p = STARRY_CRJ_HI_LIM;

    T xn = x;
    T yn = y;
    T zn = z;
    T pn = p;
    T sigma = 0.0;
    T power4 = 1.0;

    T mu, invmu;
    T xndev, yndev, zndev, pndev;
    T eps;
    T ea, eb, ec, e2, e3, s1, s2, s3, value;
    T xnroot, ynroot, znroot;
    T lam, alpha, beta;

    for (int k = 0; k < STARRY_CRJ_MAX_ITER; ++k) {

        mu = 0.2 * (xn + yn + zn + pn + pn);
        invmu = 1.0 / mu;
        xndev = (mu - xn) * invmu;
        yndev = (mu - yn) * invmu;
        zndev = (mu - zn) * invmu;
        pndev = (mu - pn) * invmu;
        eps = max(max(max(abs(xndev), abs(yndev)), abs(zndev)), abs(pndev));

        if (eps < STARRY_CRJ_TOL) {

            ea = xndev * (yndev + zndev) + yndev * zndev;
            eb = xndev * yndev * zndev;
            ec = pndev * pndev;
            e2 = ea - 3.0 * ec;
            e3 = eb + 2.0 * pndev * (ea - ec);
            s1 = 1.0 + e2 * (-C1 + 0.75 * C3 * e2 - 1.5 * C4 * e3);
            s2 = eb * (0.5 * C2 + pndev * (-C3 - C3 + pndev * C4));
            s3 = pndev * ea * (C2 - pndev * C3) - C2 * pndev * ec;
            value = 3.0 * sigma + power4 * (s1 + s2 + s3) / (mu * sqrt(mu));
            return value;

        }

        xnroot = sqrt(xn);
        ynroot = sqrt(yn);
        znroot = sqrt(zn);
        lam = xnroot * (ynroot + znroot) + ynroot * znroot;
        alpha = pn * (xnroot + ynroot + znroot) + xnroot * ynroot * znroot;
        alpha = alpha * alpha;
        beta = pn * (pn + lam) * (pn + lam);
        if (alpha < beta)
            sigma += power4 * acos(sqrt(alpha / beta)) / sqrt(beta - alpha);
        else if (alpha > beta)
            sigma += power4 * acosh(sqrt(alpha / beta)) / sqrt(alpha - beta);
        else
            sigma = sigma + power4 / sqrt(beta);

        power4 *= 0.25;
        xn = 0.25 * (xn + lam);
        yn = 0.25 * (yn + lam);
        zn = 0.25 * (zn + lam);
        pn = 0.25 * (pn + lam);

    }
    
    // Bad...
    throw std::runtime_error("Elliptic integral rj did not converge.");

}

/**
Incomplete elliptic integral of the first kind.

*/
template <class T> 
inline Vector<T> F(const Vector<T>& tanphi, const T& k2) {
  T kc2 = 1 - k2;
  return el2(tanphi, sqrt(kc2), T(1.0), T(1.0));
}

/**
Incomplete elliptic integral of the first kind (with gradient).

*/
template <class T, int N>
inline Vector<ADScalar<T, N>> F(const Vector<ADScalar<T, N>>& tanphi, const ADScalar<T, N>& k2, const Vector<T>& F_value, const Vector<T>& E_value) {
  
  // Grab values
  size_t K = tanphi.size();
  T k2_value = k2.value();
  Vector<T> tanphi_value(K);
  for (size_t k = 0; k < K; ++k)
    tanphi_value(k) = tanphi(k).value();
  
  // Compute derivatives analytically
  Vector<T> p2(K), q2(K), t2(K), dFdtanphi(K), dFdk2(K);
  T kc2 = 1 - k2_value;
  t2.array() = tanphi_value.array() * tanphi_value.array();
  p2.array() = 1.0 / (1.0 + t2.array());
  q2.array() = p2.array() * t2.array();
  dFdtanphi.array() = p2.array() * pow(1.0 - k2_value * q2.array(), -0.5);
  dFdk2.array() = 0.5 * (E_value.array() / (k2_value * kc2) - F_value.array() / k2_value - tanphi_value.array() * dFdtanphi.array() / kc2);

  // Populate the autodiff vector and return
  Vector<ADScalar<T, N>> result(K);
  for (size_t k = 0; k < K; ++k) {
    result(k).value() = F_value(k);
    result(k).derivatives() = dFdtanphi(k) * tanphi(k).derivatives() + dFdk2(k) * k2.derivatives();
  }
  return result;

}

/**
Incomplete elliptic integral of the second kind.

*/
template <typename T> 
inline Vector<T> E(const Vector<T>& tanphi, const T& k2) {
    T kc2 = 1 - k2;
    return el2(tanphi, sqrt(kc2), T(1.0), kc2);
}

/**
Incomplete elliptic integral of the second kind (with gradient).

*/
template <class T, int N>
inline Vector<ADScalar<T, N>> E(const Vector<ADScalar<T, N>>& tanphi, const ADScalar<T, N>& k2, const Vector<T>& F_value, const Vector<T>& E_value) {

  // Grab values
  size_t K = tanphi.size();
  T k2_value = k2.value();
  Vector<T> tanphi_value(K);
  for (size_t k = 0; k < K; ++k)
    tanphi_value(k) = tanphi(k).value();
  
  // Compute derivatives analytically
  Vector<T> p2(K), q2(K), t2(K), dEdtanphi(K), dEdk2(K);
  t2.array() = tanphi_value.array() * tanphi_value.array();
  p2.array() = 1.0 / (1.0 + t2.array());
  q2.array() = p2.array() * t2.array();
  dEdtanphi.array() = p2.array() * pow(1.0 - k2_value * q2.array(), 0.5);
  dEdk2.array() = 0.5 * (E_value.array() - F_value.array()) / k2_value;

  // Populate the autodiff vector and return
  Vector<ADScalar<T, N>> result(K);
  for (size_t k = 0; k < K; ++k) {
    result(k).value() = E_value(k);
    result(k).derivatives() = dEdtanphi(k) * tanphi(k).derivatives() + dEdk2(k) * k2.derivatives();
  }
  return result;

}

/**
Modified incomplete elliptic integral of the third kind.

For kappa in [-pi, pi], this integral is proportional to the Carlson elliptic integral RJ:

  PI' = -2 sin^3(phi) * RJ(cos^2 phi, 1 - k^2 sin^2 phi, 1, 1 - n sin^2 phi)

where

  phi = kappa / 2
  n = -4 b r / (r - b)^2

It can also be written in terms of the Legendre forms:

  PI' = 6 / n * (F(phi, k^2) - PI(phi, n, k^2))

This integral is only used in the expression for computing the linear limb darkening
term (2) in the primitive integral P, based on the expressions in Pal (2012).

*/
template <typename T> 
inline Vector<T> PIprime(const Vector<T>& kappa, const T& k2, const Vector<T>& p) {
    size_t K = kappa.size();
    T phi, cp, cx, sx, w;
    Vector<T> result(K);
    for (size_t k = 0; k < K; ++k) {

        // Normalize phi to the range [0, 2pi]
        phi = fmod(kappa(k) - pi<T>(), 2 * pi<T>());
        if (kappa(k) < pi<T>()) phi += 2 * pi<T>();

        cp = cos(phi);
        cx = cos(0.5 * phi);
        sx = sin(0.5 * phi);
        w = 1.0 - cx * cx / k2;
        result(k) = (1.0 + cp) * cx * rj(w, sx * sx, 1.0, p(k));
    }
    return result;
}

} // namespace iellip
} // namespace starry

#endif
