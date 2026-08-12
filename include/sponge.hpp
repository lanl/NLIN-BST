#pragma once

#include "config.hpp"

// Shared sponge profile.  Solver::sponge_sigma keeps its existing name and
// delegates to this helper so the evolution and energy ledger use exactly the
// same damping coefficient.
inline double sponge_sigma_value(const Params& p, double x) {
    double s = 0.0;
    if (p.sponge_inner_width > 0.0) {
        const double a = (p.x_min + p.sponge_inner_width - x)
                       / p.sponge_inner_width;
        if (a > 0.0) s += p.sponge_inner_strength * a * a;
    }
    if (p.sponge_outer_width > 0.0) {
        const double a = (x - (p.x_max - p.sponge_outer_width))
                       / p.sponge_outer_width;
        if (a > 0.0) s += p.sponge_outer_strength * a * a;
    }
    return s;
}
