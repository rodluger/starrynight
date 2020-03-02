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