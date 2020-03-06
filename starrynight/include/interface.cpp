/**
\file interface.cpp
\brief Defines the entry point for the C++ API.

*/

// DEBUG MODE
#define STARRY_DEBUG 1

// Includes
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <iostream>
#include "utils.h"
#include "ellip.h"

// For testing
#ifdef STARRY_DEBUG
# include "iellip.h"
# include "special.h"
# include "quad.h"
# include "primitive.h"
# include "geometry.h"
# include "starrynight.h"
  using namespace starry::iellip;
  using namespace starry::special;
  using namespace starry::quad;
  using namespace starry::primitive;
  using namespace starry::geometry;
#endif

namespace py = pybind11;

// Register the Python module
PYBIND11_MODULE(_c_ops, m) {

// For testing
#ifdef STARRY_DEBUG
# include "testing.h"
#endif

using namespace starry::utils;
using namespace starry::night;

m.def("s0T", [](const int& ydeg, const double& b_, const double& theta_, const double& bo_, const double& ro_, const Vector<double>& bs0T) {
    
    // Total number of terms in `s0^T`
    int N = (ydeg + 2) * (ydeg + 2);

    // Seed the derivatives
    ADScalar<double, 4> b, theta, bo, ro;
    b.value() = b_;
    b.derivatives() = Vector<double>::Unit(4, 0);
    theta.value() = theta_;
    theta.derivatives() = Vector<double>::Unit(4, 1);
    bo.value() = bo_;
    bo.derivatives() = Vector<double>::Unit(4, 2);
    ro.value() = ro_;
    ro.derivatives() = Vector<double>::Unit(4, 3);

    // Compute!
    int code;
    Vector<ADScalar<double, 4>> result = s0T(ydeg, b, theta, bo, ro, code);

    // Process the derivatives
    Vector<double> result_value(N);
    double bb = 0.0, btheta = 0.0, bbo = 0.0, bro = 0.0;
    for (int n = 0; n < N; ++n) {
        result_value(n) = result(n).value();
        bb += result(n).derivatives()(0) * bs0T(n);
        btheta += result(n).derivatives()(1) * bs0T(n);
        bbo += result(n).derivatives()(2) * bs0T(n);
        bro += result(n).derivatives()(3) * bs0T(n);
    }

    // Return the vector, integration code, and all derivs
    return py::make_tuple(result_value, code, bb, btheta, bbo, bro);

});

}