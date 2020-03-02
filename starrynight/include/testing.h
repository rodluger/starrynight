// Incomplete elliptic integral of the first kind
m.def("F", [](const Vector<double> &tanphi, const double &k2) {
    return F(tanphi, k2);
});

// Incomplete elliptic integral of the first kind (derivative)
m.def("dFdtanphi", [](const Vector<double> &tanphi, const double &k2) {

    size_t K = tanphi.size();

    // Seed tanphi
    Vector<ADScalar<double, 2>> tanphi_ad(K);
    for (size_t k = 0; k < K; ++k) {
        tanphi_ad(k).value() = tanphi(k);
        tanphi_ad(k).derivatives() = Vector<double>::Unit(2, 0);
    }

    // Seed k2
    ADScalar<double, 2> k2_ad;
    k2_ad.value() = k2;
    k2_ad.derivatives() = Vector<double>::Unit(2, 1);

    // Compute
    Vector<double> F_value = F(tanphi, k2);
    Vector<double> E_value = E(tanphi, k2);
    Vector<ADScalar<double, 2>> result_ad = F(tanphi_ad, k2_ad, F_value, E_value);

    // Extract the deriv
    Vector<double> result(K);
    for (size_t k = 0; k < K; ++k) {
        result(k) = result_ad(k).derivatives()(0);
    }
    return result;

});

// Incomplete elliptic integral of the first kind (derivative)
m.def("dFdk2", [](const Vector<double> &tanphi, const double &k2) {

    size_t K = tanphi.size();

    // Seed tanphi
    Vector<ADScalar<double, 2>> tanphi_ad(K);
    for (size_t k = 0; k < K; ++k) {
        tanphi_ad(k).value() = tanphi(k);
        tanphi_ad(k).derivatives() = Vector<double>::Unit(2, 0);
    }

    // Seed k2
    ADScalar<double, 2> k2_ad;
    k2_ad.value() = k2;
    k2_ad.derivatives() = Vector<double>::Unit(2, 1);

    // Compute
    Vector<double> F_value = F(tanphi, k2);
    Vector<double> E_value = E(tanphi, k2);
    Vector<ADScalar<double, 2>> result_ad = F(tanphi_ad, k2_ad, F_value, E_value);

    // Extract the deriv
    Vector<double> result(K);
    for (size_t k = 0; k < K; ++k) {
        result(k) = result_ad(k).derivatives()(1);
    }
    return result;

});

// Incomplete elliptic integral of the second kind
m.def("E", [](const Vector<double> &tanphi, const double &k2) {
    return E(tanphi, k2);
});

// Incomplete elliptic integral of the second kind (derivative)
m.def("dEdtanphi", [](const Vector<double> &tanphi, const double &k2) {

    size_t K = tanphi.size();

    // Seed tanphi
    Vector<ADScalar<double, 2>> tanphi_ad(K);
    for (size_t k = 0; k < K; ++k) {
        tanphi_ad(k).value() = tanphi(k);
        tanphi_ad(k).derivatives() = Vector<double>::Unit(2, 0);
    }

    // Seed k2
    ADScalar<double, 2> k2_ad;
    k2_ad.value() = k2;
    k2_ad.derivatives() = Vector<double>::Unit(2, 1);

    // Compute
    Vector<double> F_value = F(tanphi, k2);
    Vector<double> E_value = E(tanphi, k2);
    Vector<ADScalar<double, 2>> result_ad = E(tanphi_ad, k2_ad, F_value, E_value);

    // Extract the deriv
    Vector<double> result(K);
    for (size_t k = 0; k < K; ++k) {
        result(k) = result_ad(k).derivatives()(0);
    }
    return result;

});

// Incomplete elliptic integral of the second kind (derivative)
m.def("dEdk2", [](const Vector<double> &tanphi, const double &k2) {

    size_t K = tanphi.size();

    // Seed tanphi
    Vector<ADScalar<double, 2>> tanphi_ad(K);
    for (size_t k = 0; k < K; ++k) {
        tanphi_ad(k).value() = tanphi(k);
        tanphi_ad(k).derivatives() = Vector<double>::Unit(2, 0);
    }

    // Seed k2
    ADScalar<double, 2> k2_ad;
    k2_ad.value() = k2;
    k2_ad.derivatives() = Vector<double>::Unit(2, 1);

    // Compute
    Vector<double> F_value = F(tanphi, k2);
    Vector<double> E_value = E(tanphi, k2);
    Vector<ADScalar<double, 2>> result_ad = E(tanphi_ad, k2_ad, F_value, E_value);

    // Extract the deriv
    Vector<double> result(K);
    for (size_t k = 0; k < K; ++k) {
        result(k) = result_ad(k).derivatives()(1);
    }
    return result;

});

// Modified incomplete elliptic integral of the third kind
m.def("dPIprimedkappa", [](const double& kappa_, const double& k2_, const double& p_) {
    
    ADScalar<double, 3> k2, PIprime0;
    Vector<ADScalar<double, 3>> kappa(1), p(1);
    kappa(0).value() = kappa_;
    kappa(0).derivatives() = Vector<double>::Unit(3, 0);
    k2.value() = k2_;
    k2.derivatives() = Vector<double>::Unit(3, 1);
    p(0).value() = p_;
    p(0).derivatives() = Vector<double>::Unit(3, 2);

    // DEBUG
    PIprime0.value() = 0.0;
    PIprime0.derivatives().setZero();

    auto result = PIprime(kappa, k2, p, PIprime0);
    return result.derivatives()(0);

});

// All three integrals
m.def("ellip", [](const double& bo, const double& ro, const Vector<double> &kappa) {
    
    double k2 = (1 - ro * ro - bo * bo + 2 * bo * ro) / (4 * bo * ro);
    if (k2 > 1) k2 = 1.0 / k2;
    double p0 = (ro * ro + bo * bo + 2 * ro * bo) / (ro * ro + bo * bo - 2 * ro * bo);


    double F0 = CEL(k2, 1.0, 1.0, 1.0);
    double E0 = CEL(k2, 1.0, 1.0, 1.0 - k2);
    double PIprime0 = CEL(k2, p0, 1.0, 1.0);

    return ellip(bo, ro, kappa, F0, E0, PIprime0);
});