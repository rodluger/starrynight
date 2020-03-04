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

// P2 term
m.def("P2", [](const double& bo_, const double& ro_, const Vector<double>& kappa_) {
    
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

    // Compute
    auto integrals = IncompleteEllipticIntegrals<double>(bo, ro, kappa);
    ADScalar<double, 4> k2 = (1 - ro * ro - bo * bo + 2 * bo * ro) / (4 * bo * ro);
    Vector<ADScalar<double, 4>> s1(2), s2(2), c1(2);
    s1.array() = sin(0.5 * kappa.array());
    s2.array() = s1.array() * s1.array();
    c1.array() = cos(0.5 * kappa.array());
    auto result = P2(bo, ro, k2, kappa, s1, s2, c1, integrals.F, integrals.E, integrals.PIp);
    
    return py::make_tuple(result.value(), result.derivatives());

});

// P2 term (numerical)
m.def("P2_numerical", [](const double& bo_, const double& ro_, const Vector<double>& kappa_) {
    
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

    // Compute
    auto result = P2_numerical(bo, ro, kappa);
    
    return py::make_tuple(result.value(), result.derivatives());

});

// Gaussian quadrature
m.def("quad", [](const std::function<double(double)> &f, const double& a, const double& b) {
    return QUAD.integrate(a, b, f);
});

// P2 term (numerical)
m.def("J_numerical", [](const int N, const double& bo_, const double& ro_, const Vector<double>& kappa_) {
    
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

    // Compute
    ADScalar<double, 4> k2 = (1 - ro * ro - bo * bo + 2 * bo * ro) / (4 * bo * ro);
    auto result = J_numerical(N, k2, kappa);
    
    return py::make_tuple(result.value(), result.derivatives());

});