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

// J term (numerical)
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

// Gauss 2F1
m.def("hyp2f1", [](const double& a, const double& b, const double& c, const double& z_) {
    
    // Seed the derivatives
    ADScalar<double, 1> z;
    z.value() = z_;
    z.derivatives() = Vector<double>::Unit(1, 0);

    ADScalar<double, 1> result = hyp2f1(a, b, c, z);
    return py::make_tuple(result.value(), result.derivatives());
});

// The full J vector
m.def("J", [](const int nmax, const double& bo_, const double& ro_, const Vector<double>& kappa_) {
    
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

    // Pre-compute some stuff
    auto integrals = IncompleteEllipticIntegrals<double>(bo, ro, kappa);
    ADScalar<double, 4> k2 = (1 - ro * ro - bo * bo + 2 * bo * ro) / (4 * bo * ro);
    ADScalar<double, 4> km2 = 1 / k2;
    Vector<ADScalar<double, 4>> s1(2), s2(2), c1(2), q2(2);
    s1.array() = sin(0.5 * kappa.array());
    s2.array() = s1.array() * s1.array();
    c1.array() = cos(0.5 * kappa.array());
    q2.array() = 1.0 - s2.array() / k2;
    if (q2(0) < 0) q2(0) = 0;
    if (q2(1) < 0) q2(1) = 0;

    // Compute J
    Vector<ADScalar<double, 4>> result = J(nmax, k2, km2, kappa, s1, s2, c1, q2, integrals.F, integrals.E);
    
    // Return only the value
    Vector<double> result_value(nmax + 1);
    for (int i = 0; i < nmax + 1; ++i) {
        result_value(i) = result(i).value();
    }
    return result_value;

});

// The H vector
m.def("H", [](const int uvmax, const Vector<double>& xi_) {

    // For testing purposes, require two elements in xi
    if (xi_.size() != 2)
        throw std::runtime_error("Parameter xi must be a two-element vector.");

    // Seed the derivatives
    Vector<ADScalar<double, 2>> xi(2);
    xi(0).value() = xi_(0);
    xi(0).derivatives() = Vector<double>::Unit(2, 0);
    xi(1).value() = xi_(1);
    xi(1).derivatives() = Vector<double>::Unit(2, 1);

    // Compute H
    Matrix<ADScalar<double, 2>> result = H(uvmax, xi);

    // Return the value and derivs
    Matrix<double> result_value(uvmax + 1, uvmax + 1);
    Matrix<double> result_deriv0(uvmax + 1, uvmax + 1);
    Matrix<double> result_deriv1(uvmax + 1, uvmax + 1);
    result_value.setZero();
    result_deriv0.setZero();
    result_deriv1.setZero();
    for (int u = 0; u < uvmax + 1; ++u) {
        for (int v = 0; v < uvmax + 1 - u; ++v) {
            result_value(u, v) = result(u, v).value();
            result_deriv0(u, v) = result(u, v).derivatives()(0);
            result_deriv1(u, v) = result(u, v).derivatives()(1);
        }
    }
    return py::make_tuple(result_value, result_deriv0, result_deriv1);

});

// The T integral
m.def("T", [](const int ydeg, const double& b_, const double &theta_, const Vector<double>& xi_) {

    // For testing purposes, require two elements in xi
    if (xi_.size() != 2)
        throw std::runtime_error("Parameter xi must be a two-element vector.");

    // Seed the derivatives
    ADScalar<double, 4> b, theta;
    b.value() = b_;
    b.derivatives() = Vector<double>::Unit(4, 0);
    theta.value() = theta_;
    theta.derivatives() = Vector<double>::Unit(4, 1);
    Vector<ADScalar<double, 4>> xi(2);
    xi(0).value() = xi_(0);
    xi(0).derivatives() = Vector<double>::Unit(4, 2);
    xi(1).value() = xi_(1);
    xi(1).derivatives() = Vector<double>::Unit(4, 3);

    // Compute T
    Vector<ADScalar<double, 4>> result = T(ydeg, b, theta, xi);
    
    // Return the value + derivs
    Vector<double> result_value((ydeg + 1) * (ydeg + 1));
    Vector<double> result_deriv0((ydeg + 1) * (ydeg + 1));
    Vector<double> result_deriv1((ydeg + 1) * (ydeg + 1));
    Vector<double> result_deriv2((ydeg + 1) * (ydeg + 1));
    Vector<double> result_deriv3((ydeg + 1) * (ydeg + 1));
    for (int i = 0; i < (ydeg + 1) * (ydeg + 1); ++i) {
        result_value(i) = result(i).value();
        result_deriv0(i) = result(i).derivatives()(0);
        result_deriv1(i) = result(i).derivatives()(1);
        result_deriv2(i) = result(i).derivatives()(2);
        result_deriv3(i) = result(i).derivatives()(3);
    }

    return py::make_tuple(result_value, result_deriv0, result_deriv1, result_deriv2, result_deriv3);

});

// The Q integral
m.def("Q", [](const int ydeg,const Vector<double>& lam_) {

    // For testing purposes, require two elements in lam
    if (lam_.size() != 2)
        throw std::runtime_error("Parameter lam must be a two-element vector.");

    // Seed the derivatives
    Vector<ADScalar<double, 2>> lam(2);
    lam(0).value() = lam_(0);
    lam(0).derivatives() = Vector<double>::Unit(2, 0);
    lam(1).value() = lam_(1);
    lam(1).derivatives() = Vector<double>::Unit(2, 1);

    // Compute Q
    Vector<ADScalar<double, 2>> result = Q(ydeg, lam);

    // Return the value + derivs
    Vector<double> result_value((ydeg + 1) * (ydeg + 1));
    Vector<double> result_deriv0((ydeg + 1) * (ydeg + 1));
    Vector<double> result_deriv1((ydeg + 1) * (ydeg + 1));
    for (int i = 0; i < (ydeg + 1) * (ydeg + 1); ++i) {
        result_value(i) = result(i).value();
        result_deriv0(i) = result(i).derivatives()(0);
        result_deriv1(i) = result(i).derivatives()(1);
    }
    
    return py::make_tuple(result_value, result_deriv0, result_deriv1);
});

// The P integral
m.def("P", [](const int ydeg, const double& bo_, const double &ro_, const Vector<double>& kappa_) {

    // For testing purposes, require two elements in kappa
    if (kappa_.size() != 2)
        throw std::runtime_error("Parameter kappa must be a two-element vector.");

    // Seed the derivatives
    ADScalar<double, 4> bo, ro;
    bo.value() = bo_;
    bo.derivatives() = Vector<double>::Unit(4, 0);
    ro.value() = ro_;
    ro.derivatives() = Vector<double>::Unit(4, 1);
    Vector<ADScalar<double, 4>> kappa(2);
    kappa(0).value() = kappa_(0);
    kappa(0).derivatives() = Vector<double>::Unit(4, 2);
    kappa(1).value() = kappa_(1);
    kappa(1).derivatives() = Vector<double>::Unit(4, 3);

    // Compute P
    Vector<ADScalar<double, 4>> result = P(ydeg, bo, ro, kappa);

    // Return the value + derivs
    Vector<double> result_value((ydeg + 1) * (ydeg + 1));
    Vector<double> result_deriv0((ydeg + 1) * (ydeg + 1));
    Vector<double> result_deriv1((ydeg + 1) * (ydeg + 1));
    Vector<double> result_deriv2((ydeg + 1) * (ydeg + 1));
    Vector<double> result_deriv3((ydeg + 1) * (ydeg + 1));
    for (int i = 0; i < (ydeg + 1) * (ydeg + 1); ++i) {
        result_value(i) = result(i).value();
        result_deriv0(i) = result(i).derivatives()(0);
        result_deriv1(i) = result(i).derivatives()(1);
        result_deriv2(i) = result(i).derivatives()(2);
        result_deriv3(i) = result(i).derivatives()(3);
    }

    return py::make_tuple(result_value, result_deriv0, result_deriv1, result_deriv2, result_deriv3);
});

// Generic polynomial root solver
m.def("eigen_roots", [](const std::vector<double>& coeffs) {
    std::vector<std::complex<double>> roots = eigen_roots(coeffs);
    return roots;
});

// The root solver for the occultation problem
m.def("get_roots", [](const double& b_, const double& theta_, const double& bo_, const double &ro_) {

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

    // Compute
    ADScalar<double, 4> sintheta = sin(theta), costheta = cos(theta);
    Vector<ADScalar<double, 4>> result = get_roots(b, theta, costheta, sintheta, bo, ro);
    int nroots = result.size();

    // Return the value + derivs
    Vector<double> result_value(nroots);
    Vector<double> result_deriv0(nroots);
    Vector<double> result_deriv1(nroots);
    Vector<double> result_deriv2(nroots);
    Vector<double> result_deriv3(nroots);
    for (int i = 0; i < nroots; ++i) {
        result_value(i) = result(i).value();
        result_deriv0(i) = result(i).derivatives()(0);
        result_deriv1(i) = result(i).derivatives()(1);
        result_deriv2(i) = result(i).derivatives()(2);
        result_deriv3(i) = result(i).derivatives()(3);
    }

    return py::make_tuple(result_value, result_deriv0, result_deriv1, result_deriv2, result_deriv3);
});

// The root solver for the occultation problem
m.def("get_angles", [](const double& b_, const double& theta_, const double& bo_, const double &ro_) {


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

    // Compute
    ADScalar<double, 4> sintheta = sin(theta), costheta = cos(theta);
    Vector<ADScalar<double, 4>> kappa, lam, xi;
    int code = get_angles(b, theta, costheta, sintheta, bo, ro, kappa, lam, xi);

    // Return only the *values*
    Vector<double> kappa_value(kappa.size());
    for (int i = 0; i < kappa.size(); ++i) kappa_value(i) = kappa(i).value();
    Vector<double> lam_value(lam.size());
    for (int i = 0; i < lam.size(); ++i) lam_value(i) = lam(i).value();
    Vector<double> xi_value(xi.size());
    for (int i = 0; i < xi.size(); ++i) xi_value(i) = xi(i).value();

    return py::make_tuple(kappa_value, lam_value, xi_value, code);

});

