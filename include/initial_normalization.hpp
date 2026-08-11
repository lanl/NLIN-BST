#pragma once

#include "config.hpp"
#include "decomp.hpp"
#include "field.hpp"
#include "grid.hpp"

// The energy is split into pieces with simple amplitude scaling:
//   E(s phi, s P) = s^2 quadratic + mu s^4 quartic_base
// for the pure cubic model.
struct InitialEnergyParts {
    double quadratic = 0.0;
    double quartic_base = 0.0;

    double full_cubic(double mu) const {
        return quadratic + mu * quartic_base;
    }
};

// Uses the same variables, weights, and finite-difference operators as the
// current NLIN-BST diagnostics implementation.  Consequently the normalization
// is consistent with the energy reported by this code version.
InitialEnergyParts compute_initial_energy_parts(const Params& p,
                                                const Grid& grid,
                                                const Decomp& decomp,
                                                const State& state);

// InitialData::apply constructs either the legacy amp-scaled field
// (id_normalize=none) or an amp=1 reference shape (normalization enabled).
// This function then applies the requested global energy normalization without
// changing the existing InitialData interface or class/function names.
void normalize_and_scale_initial_state(const Params& p,
                                       const Grid& grid,
                                       const Decomp& decomp,
                                       State& state);
