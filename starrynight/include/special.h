/**
\file special.h
\brief Special functions.

*/

#ifndef _STARRY_SPECIAL_H_
#define _STARRY_SPECIAL_H_

#include "constants.h"
#include "utils.h"
#include "quad.h"
#include <cmath>

namespace starry {
namespace special {

using std::abs;
using namespace utils;
using namespace quad;

/**

  Return the sum over pairwise differences of an array.

  This is used to evaluate a (series of) definite integral(s) given
  the antiderivatives at each of the integration limits.

*/
template <typename T> 
T pairdiff(const Vector<T>& array) {
    size_t K = array.size();
    if (K > 1) {
        if (K % 2 == 0) {
            int sgn = -1;
            T result = 0.0;
            for (size_t i = 0; i < K; ++i) {
              result += sgn * array(i);
              sgn *= -1;
            }
            return result;
        } else {
            throw std::runtime_error("Array length must be even in call to `pairdiff`.");
        }
    } else if (K == 1) {
        throw std::runtime_error("Array length must be even in call to `pairdiff`.");
    } else {
        // Empty array
        return 0.0; 
    }
}

template <typename T> 
T P2_integrand(const T& bo, const T& ro, const T& phi) {
  T c = cos(phi);
  T z = 1 - ro * ro - bo * bo - 2 * bo * ro * c;
  if (z < 1e-12) z = 1e-12;
  if (z > 1 - 1e-12) z = 1 - 1e-12;
  return (1.0 - z * sqrt(z)) / (1.0 - z) * (ro + bo * c) * ro / 3.0;
}

template <typename T> 
T dP2dbo_integrand(const T& bo, const T& ro, const T& phi) {
  T c = cos(phi);
  T z = 1 - ro * ro - bo * bo - 2 * bo * ro * c;
  if (z < 1e-12) z = 1e-12;
  if (z > 1 - 1e-12) z = 1 - 1e-12;
  T P = (1.0 - z * sqrt(z)) / (1.0 - z) * (ro + bo * c) * ro / 3.0;
  T q = 3.0 * sqrt(z) / (1.0 - z * sqrt(z)) - 2.0 / (1.0 - z);
  return P * ((bo + ro * c) * q + 1.0 / (bo + ro / c));
}

template <typename T> 
T dP2dro_integrand(const T& bo, const T& ro, const T& phi) {
  T c = cos(phi);
  T z = 1 - ro * ro - bo * bo - 2 * bo * ro * c;
  if (z < 1e-12) z = 1e-12;
  if (z > 1 - 1e-12) z = 1 - 1e-12;
  T P = (1.0 - z * sqrt(z)) / (1.0 - z) * (ro + bo * c) * ro / 3.0;
  T q = 3.0 * sqrt(z) / (1.0 - z * sqrt(z)) - 2.0 / (1.0 - z);
  return P * ((ro + bo * c) * q + 1.0 / ro + 1.0 / (ro + bo * c));
}

/**

*/
template <typename T> 
T P2_numerical(const T& bo, const T& ro, const Vector<T>& kappa) {

    using Scalar = typename T::Scalar;
    size_t K = kappa.size();

    std::function<Scalar(Scalar)> f = [bo, ro](Scalar phi) { return P2_integrand(bo.value(), ro.value(), phi); };
    std::function<Scalar(Scalar)> dfdbo = [bo, ro](Scalar phi) { return dP2dbo_integrand(bo.value(), ro.value(), phi); };
    std::function<Scalar(Scalar)> dfdro = [bo, ro](Scalar phi) { return dP2dro_integrand(bo.value(), ro.value(), phi); };
    
    // Compute the function value
    T res = 0.0;
    for (size_t i = 0; i < K; i += 2)
      res.value() += QUAD.integrate(kappa(i).value() - pi<Scalar>(), kappa(i + 1).value() - pi<Scalar>(), f);

    // Compute the derivatives.
    // Deriv wrt kappa is easy; need to integrate for the other two
    for (size_t i = 0; i < K; i += 2) {
      res.derivatives() += f(kappa(i + 1).value() - pi<Scalar>()) * kappa(i + 1).derivatives(); 
      res.derivatives() -= f(kappa(i).value() - pi<Scalar>()) * kappa(i).derivatives();
      res.derivatives() += bo.derivatives() * QUAD.integrate(kappa(i).value() - pi<Scalar>(), kappa(i + 1).value() - pi<Scalar>(), dfdbo);
      res.derivatives() += ro.derivatives() * QUAD.integrate(kappa(i).value() - pi<Scalar>(), kappa(i + 1).value() - pi<Scalar>(), dfdro);
    }

    return res;

}

/**
  Returns the difference of a pair (or pairs) of Pal integrals for the
  P2 (linear) term. Specifically, returns the sum of

        P2(bo, ro, kappa[i + 1]) - P2(bo, ro, kappa[i])

  for i = 0, 2, 4, ...
*/
template <typename T> 
T P2(const T& bo, const T& ro, const T& k2, const Vector<T>& kappa, const Vector<T>& s1, 
     const Vector<T>& s2, const Vector<T>& c1, const T& F, const T& E, 
     const T& PIp) {

    // Useful variables
    size_t K = kappa.size();
    T r2 = ro * ro;
    T b2 = bo * bo;
    T br = bo * ro;
    T bpr = bo + ro;
    T bmr = bo - ro;
    T d2 = r2 + b2 - 2 * br;
    T term = 0.5 / sqrt(br * k2);
    T p0 = 4.0 - 7.0 * r2 - b2;
    Vector<T> q2(K);
    q2.array() = r2 + b2 - 2 * br * (1 - 2 * s2.array());

    // Special cases
    if (bo == 0.0) {

        // Analytic limit
        if (ro < 1.0)
            return (1 - (1 - r2) * sqrt(1 - r2)) * pairdiff(kappa) / 3.0;
        else
            return pairdiff(kappa) / 3.0;

    } else if (abs(bo - ro) < STARRY_PAL_BO_EQUALS_RO_TOL) {

        // Solve numerically
        return P2_numerical(bo, ro, kappa);

    } else if (abs(bo - (ro - 1)) < STARRY_PAL_BO_EQUALS_RO_MINUS_ONE_TOL) {

        // Solve numerically
        return P2_numerical(bo, ro, kappa);

    } else if (abs(bo - (1 - ro)) < STARRY_PAL_BO_EQUALS_ONE_MINUS_RO_TOL) {

        // Solve numerically
        return P2_numerical(bo, ro, kappa);

    }

    // Constant term
    T A = 0.0;
    int sgn = -1;
    for (size_t i = 0; i < K; ++i) {
      A -= sgn * atan2(-bmr * c1(i), bpr * s1(i));
      if (kappa(i) > 3 * pi<T>()) {
        if (bmr > 0)
          A -= sgn * 2 * pi<T>();
        else if (bmr < 0)
          A += sgn * 2 * pi<T>();
      }
      A += 0.5 * sgn * kappa(i);
      if (q2(i) < 1.0)
        A -= 2 * (2.0 / 3.0) * br * sgn * (s1(i) * c1(i) * sqrt(1 - q2(i)));
      sgn *= -1;
    }

    // Carlson RD term
    T C = -2 * (2.0 / 3.0) * br * p0 * term * k2;

    // Carlson RF term
    T fac = -bpr / bmr;
    T B = -(
        (
            ((1.0 + 2.0 * r2 * r2 - 4.0 * r2) + (2.0 / 3.0) * br * (p0 + 5 * br) + fac)
            * term
        )
        + C
    );

    // Carlson PIprime term
    T D = -(2.0 / 3.0) * fac / d2 * term * br;

    return (A + B * F + C * E + D * PIp) / 3.0;

}

} // namespace special
} // namespace starry

#endif
