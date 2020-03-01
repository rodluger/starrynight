/**
\file interface.cpp
\brief Defines the entry point for the C++ API.

*/

// Includes
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "utils.h"
#include "iellip.h"
namespace py = pybind11;

// Register the Python module
PYBIND11_MODULE(_c_ops, m) {
  
  // Import some useful stuff
  using namespace starry::utils;
  using namespace starry::iellip;

  // Incomplete elliptic integral of the first kind
  m.def("F", [](const Vector<double> &tanphi, const double &k2) {
    return F(tanphi, k2);
  });

  // Incomplete elliptic integral of the second kind
  m.def("E", [](const Vector<double> &tanphi, const double &k2) {
    return E(tanphi, k2);
  });

}