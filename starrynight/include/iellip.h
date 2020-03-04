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


template <class T>
class IncompleteEllipticIntegrals {
  
  // Autodiff wrt {b, theta, bo, ro}
  using A = ADScalar<T, 4>;

  protected:

    // Inputs
    A bo;
    A ro;
    Vector<A> kappa;
    size_t K;

    // Helper vars
    A k2;
    A k2inv;
    A k;
    A kinv;
    A kc2;
    A kc;
    A kc2inv;
    A kcinv;
    A p0;
    Vector<A> p;

    Vector<A> phi;
    Vector<A> coskap;
    Vector<A> cosphi;
    Vector<A> sinphi;
    Vector<A> w;

    // Complete elliptic integrals
    // Note that we don't actually need the derivs of PIp0; see below for details
    A F0;
    A E0;
    T PIp0;

    // Vectorized output
    Vector<A> Fv;
    Vector<A> Ev;


    /**

    */
    inline void compute_el2(const Vector<A>& tanphi_, const A& m_) {
      
      // Get the values
      Vector<T> tanphi(K);
      for (size_t i = 0; i < K; ++i) tanphi(i) = tanphi_(i).value();
      T m = m_.value();
      T mc = 1 - m;

      // Compute the elliptic integrals
      Fv = el2(tanphi, sqrt(1 - m), 1.0, 1.0);
      Ev = el2(tanphi, sqrt(1 - m), 1.0, 1 - m);

      // Compute their derivatives
      T p2, q2, t2, ddtanphi, ddm;
      for (size_t i = 0; i < K; ++i) {
        t2 = tanphi(i) * tanphi(i);
        p2 = 1.0 / (1.0 + t2);
        q2 = p2 * t2;
        ddtanphi = p2 / sqrt(1.0 - m * q2);
        ddm = 0.5 * (Ev(i).value() / (m * mc) - Fv(i).value() / m - tanphi(i) * ddtanphi / mc);
        Fv(i).derivatives() = ddtanphi * tanphi_(i).derivatives() + ddm * m_.derivatives();
        ddtanphi = p2 * sqrt(1.0 - m * q2);
        ddm = 0.5 * (Ev(i).value() - Fv(i).value()) / m;
        Ev(i).derivatives() = ddtanphi * tanphi_(i).derivatives() + ddm * m_.derivatives();
      }

    }
    
    /**
      Compute the incomplete elliptic integrals of the first and second kinds.

    */
    inline void compute_FE() {

      F = 0.0;
      E = 0.0;

      if (k2 < 1) {

          // Analytic continuation from (17.4.15-16) in Abramowitz & Stegun
          // A better format is here: https://dlmf.nist.gov/19.7#ii

          // Helper variables
          Vector<A> arg(K), arg2(K), tanphi(K);
          arg.array() = kinv * sin(0.5 * kappa.array());
          arg2.array() = 1.0 - arg.array() * arg.array();
          arg2.array() = (arg2.array() < 1e-12).select(1e-12, arg2.array());
          tanphi.array() = arg.array() * pow(arg2.array(), -0.5);

          // Compute the incomplete elliptic integrals
          compute_el2(tanphi, k2);
          Fv.array() *= k;
          Ev.array() = kinv * (Ev.array() - (1 - k2) * kinv * Fv.array());

          // Compute the *definite* integrals
          // Add offsets to account for the limited domain of `el2`
          int sgn = -1;
          for (size_t i = 0; i < K; ++i) {
              if (kappa(i) > 3 * pi<T>()) {
                  F += sgn * (4 * F0 + Fv(i));
                  E += sgn * (4 * E0 + Ev(i));
              } else if (kappa(i) > pi<T>()) {
                  F += sgn * (2 * F0 - Fv(i));
                  E += sgn * (2 * E0 - Ev(i));
              } else {
                  F += sgn * Fv(i);
                  E += sgn * Ev(i);
              }
              sgn *= -1;
          }

        } else {

          // Helper variables
          Vector<A> tanphi(K);
          tanphi.array() = tan(0.5 * kappa.array());

          // Compute the incomplete elliptic integrals
          compute_el2(tanphi, k2inv);

          // Compute the *definite* integrals
          // Add offsets to account for the limited domain of `el2`
          int sgn = -1;
          for (size_t i = 0; i < K; ++i) {
              if (kappa(i) > 3 * pi<T>()) {
                  F += sgn * (4 * F0 + Fv(i));
                  E += sgn * (4 * E0 + Ev(i));
              } else if (kappa(i) > pi<T>()) {
                  F += sgn * (2 * F0 + Fv(i));
                  E += sgn * (2 * E0 + Ev(i));
              } else {
                  F += sgn * Fv(i);
                  E += sgn * Ev(i);
              }
              sgn *= -1;
          }

        }
    }

    /**

      Modified incomplete elliptic integral of the third kind.

      This integral is proportional to the Carlson elliptic integral RJ:

          PI' = -2 sin^3(phi) * RJ(cos^2 phi, 1 - k^2 sin^2 phi, 1, 1 - n sin^2 phi)

      where

          phi = kappa / 2
          n = -4 b r / (r - b)^2

      It can also be written in terms of the Legendre forms:

          PI' = 6 / n * (F(phi, k^2) - PI(phi, n, k^2))

      This integral is only used in the expression for computing the linear limb darkening
      term (2) in the primitive integral P, based on the expressions in Pal (2012).

    */
    inline void compute_PIp() {
        
        // Stability hack
        if (fabs(bo.value() - ro.value()) < STARRY_PAL_BO_EQUALS_RO_TOL) {
          PIp = 0.0;
          return;
        }

        // Helper variables
        T val;
        T dvaldn, dvaldk2, dvaldkappa;
        T dPIdn, dPIdk2, dPIdkappa, dFdk2, dFdkappa;

        A n = -4 * bo * ro / ((ro - bo) * (ro - bo));


        // Compute the integrals
        int sgn = -1;
        PIp = 0.0;
        for (size_t i = 0; i < K; ++i) {
            
            if (w(i).value() >= 0) {

              // Compute the integral, valid for -pi < kappa < pi
              val = (1.0 - coskap(i).value()) * cosphi(i).value() * rj(w(i).value(), sinphi(i).value() * sinphi(i).value(), 1.0, p(i).value());

              // Add offsets to account for the limited domain of `rj`
              if (kappa(i) > 3 * pi<T>()) {
                  val += 2 * PIp0;
              } else if (kappa(i) > pi<T>()) {
                  val += PIp0;
              }

              // Derivatives. We compute these from the derivatives of the equivalent quantity
              //
              //    6 / n * (F(kappa / 2, k^2) - PI(kappa / 2, n, k^2))
              //
              // since that's easier. Note that this means we don't need the derivatives of PIp0.

              dPIdn = 0.0; // TODO
              dPIdk2 = 0.0;
              dPIdkappa = 0.0;
              dFdk2 = 0.0;
              dFdkappa = 0.0;

              dvaldn = -(val + 6 * dPIdn) / n.value();
              dvaldk2 = 6 / n.value() * (dPIdk2 - dFdk2);
              dvaldkappa = 6 / n.value() * (dPIdkappa - dFdkappa);

              // The integral value
              PIp.value() += sgn * val;

              PIp.derivatives() += sgn * (dvaldn * n.derivatives() + dvaldk2 * k2.derivatives() + dvaldkappa * kappa(i).derivatives());

            } else {
              
              // The integral is complex!
              throw std::runtime_error("Elliptic integral PI' evaluated to a complex number.");

            }
            
            sgn *= -1;

        }

    }

  public:

    // Outputs
    A F;
    A E;
    A PIp;


    //! Constructor
    explicit IncompleteEllipticIntegrals(const A& bo, const A& ro, const Vector<A>& kappa) :
        bo(bo), ro(ro), kappa(kappa), K(kappa.size()),
        p(K), phi(K), coskap(K), cosphi(K), sinphi(K), w(K)
    {

        // Helper vars
        phi.array() = 0.5 * (kappa.array() - pi<T>());
        for (size_t i = 0; i < K; ++i) {
          while (phi(i) < 0) phi(i) += pi<T>();
          while (phi(i) > pi<T>()) phi(i) -= pi<T>();
        }
        coskap.array() = cos(kappa.array());
        cosphi.array() = cos(phi.array());
        sinphi.array() = sin(phi.array());
        k2 = (1 - ro * ro - bo * bo + 2 * bo * ro) / (4 * bo * ro);
        k2inv = 1.0 / k2;
        k = sqrt(k2);
        kinv = 1.0 / k;
        kc2 = 1 - k2;
        kc = sqrt(kc2);
        kc2inv = 1 - k2inv;
        kcinv = sqrt(kc2inv);
        p0 = (ro * ro + bo * bo + 2 * ro * bo) / (ro * ro + bo * bo - 2 * ro * bo);
        p.array() = (ro * ro + bo * bo - 2 * ro * bo * coskap.array()) / (ro * ro + bo * bo - 2 * ro * bo);
        w.array() = 1.0 - cosphi.array() * cosphi.array() / k2;

        // TODO: Nudge k2 away from 1

        // Complete elliptic integrals
        if (k2 < 1) {
          
          // Values
          F0.value() = k.value() * CEL(k2.value(), 1.0, 1.0, 1.0);
          E0.value() = kinv.value() * (CEL(k2.value(), 1.0, 1.0, 1.0 - k2.value()) - (1.0 - k2.value()) * kinv.value() * F0.value());
          if (bo != ro) {
            PIp0 = -4 * k2.value() * k.value() * rj(0.0, 1 - k2.value(), 1.0, 1.0 / ((ro - bo) * (ro - bo)).value());
          } else {
            PIp0 = 0.0;
          }
          
          // Derivatives
          F0.derivatives() = 0.5 / k2.value() * (E0.value() / (1 - k2.value()) - F0.value()) * k2.derivatives();
          E0.derivatives() = 0.5 / k2.value() * (E0.value() - F0.value()) * k2.derivatives();

        } else {

          // Values
          F0.value() = CEL(k2inv.value(), 1.0, 1.0, 1.0);
          E0.value() = CEL(k2inv.value(), 1.0, 1.0, 1.0 - k2inv.value());
          if ((bo != 0) && (bo != ro)) {
              PIp0 = -12.0 / (1 - p0.value()) * (CEL(k2inv.value(), p0.value(), 1.0, 1.0) - F0.value());
          } else {
              PIp0 = 0.0;
          }

          // Derivatives
          F0.derivatives() = 0.5 / k2inv.value() * (E0.value() / (1 - k2inv.value()) - F0.value()) * k2inv.derivatives();
          E0.derivatives() = 0.5 / k2inv.value() * (E0.value() - F0.value()) * k2inv.derivatives();

        }

        // Compute
        compute_FE();
        compute_PIp();

    }

};


} // namespace iellip
} // namespace starry

#endif
