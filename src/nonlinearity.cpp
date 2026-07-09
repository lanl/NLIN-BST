#include "nonlinearity.hpp"

#include <algorithm>
#include <cmath>

// In tortoise coordinates,
//   ds^2 = -f dt^2 + f dx^2 + r^2 dOmega_2^2 + dz^2,
// so
//   grad phi . grad phi = -P^2/f + phi_x^2/f + phi_theta^2/r^2 + phi_z^2.
// The floor on f prevents derivative stress tests from blowing up solely due to
// evaluating too close to the horizon-side sponge.
double derivative_quadratic_Q(QKind kind, const DerivPoint& d) {
    const double ff = std::max(d.f, 1e-10);
    switch (kind) {
        case QKind::None:
            return 0.0;
        case QKind::Metric:
            return -d.P * d.P / ff + d.phix * d.phix / ff
                 + d.phith * d.phith / (d.r * d.r) + d.phiz * d.phiz;
        case QKind::Spatial:
            return d.phix * d.phix / ff + d.phith * d.phith / (d.r * d.r) + d.phiz * d.phiz;
        case QKind::NullX:
            return (-d.P * d.P + d.phix * d.phix) / ff;
        case QKind::BadTime:
            return d.P * d.P / ff;
    }
    return 0.0;
}

double nonlinear_source_N(const Params& p, const DerivPoint& d) {
    const double phi2 = d.phi * d.phi;
    const double cubic = p.mu * phi2 * d.phi;
    const double derivative = (p.lambda == 0.0) ? 0.0 : p.lambda * derivative_quadratic_Q(p.q_kind, d);
    const double numerator = cubic + derivative;

    if (p.alpha2_phi_box == 0.0) return numerator;

    double denom = 1.0 - p.alpha2_phi_box * d.phi;
    if (std::abs(denom) < 1e-8) {
        denom = (denom >= 0.0) ? 1e-8 : -1e-8;
    }
    return numerator / denom;
}
