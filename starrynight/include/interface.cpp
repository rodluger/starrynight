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
#include <iostream>
#include "utils.h"

# ifdef STARRY_DEBUG
#   include "ellip.h"
#   include "iellip.h"
# endif

namespace py = pybind11;

// Register the Python module
PYBIND11_MODULE(_c_ops, m) {
  
  // Import some useful stuff
  using namespace starry::utils;
  using namespace starry::iellip;

# ifdef STARRY_DEBUG
#   include "testing.h"
# endif

}