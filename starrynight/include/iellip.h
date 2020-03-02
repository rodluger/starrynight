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


} // namespace iellip
} // namespace starry

#endif
