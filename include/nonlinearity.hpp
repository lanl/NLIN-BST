#pragma once

#include "config.hpp"

struct DerivPoint {
    double phi = 0.0;
    double P = 0.0;
    double phix = 0.0;
    double phith = 0.0;
    double phiz = 0.0;
    double r = 1.0;
    double f = 1.0;
};

double derivative_quadratic_Q(QKind kind, const DerivPoint& d);
double nonlinear_source_N(const Params& p, const DerivPoint& d);
