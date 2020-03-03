// All three integrals
m.def("ellip", [](const double& bo_, const double& ro_, const Vector<double>& kappa_) {
    
    // For testing purposes, require two elements in kappa
    if (kappa_.size() != 2)
        throw std::runtime_error("Parameter kappa must be a two-element vector.");

    // Seed the derivatives
    ADScalar<double, 4> bo, ro;
    Vector<ADScalar<double, 4>> kappa(2);
    bo.value() = bo_;
    bo.derivatives() = Vector<double>::Unit(4, 0);
    ro.value() = ro_;
    ro.derivatives() = Vector<double>::Unit(4, 1);
    kappa(0).value() = kappa_(0);
    kappa(0).derivatives() = Vector<double>::Unit(4, 2);
    kappa(1).value() = kappa_(1);
    kappa(1).derivatives() = Vector<double>::Unit(4, 3);

    // Compute the integrals
    auto integrals = IncompleteEllipticIntegrals<double>(bo, ro, kappa);
    auto F = py::make_tuple(integrals.F.value(), integrals.F.derivatives());
    auto E = py::make_tuple(integrals.E.value(), integrals.E.derivatives());
    auto PIp = py::make_tuple(integrals.PIp.value(), integrals.PIp.derivatives());
    
    return py::make_tuple(F, E, PIp);

});