#include "initial_normalization.hpp"

#include "operators.hpp"
#include "utils.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

InitialEnergyParts compute_initial_energy_parts(const Params& p,
                                                const Grid& grid,
                                                const Decomp& decomp,
                                                const State& state) {
    double local[2] = {0.0, 0.0};
    double global[2] = {0.0, 0.0};

    for (int il = 0; il < decomp.local_nx; ++il) {
        const int gi = decomp.i_start + il;
        const int ii = il + state.Q.ng;
        const double r = grid.r[gi];
        const double f = grid.f[gi];
        const double wx = grid.dx * ((gi == 0 || gi == p.nx - 1) ? 0.5 : 1.0);

        for (int j = 0; j < p.nth; ++j) {
            double wtheta = 2.0 * PI_CONST * std::sin(grid.theta[j]) * grid.dtheta;
            if (j == 0 || j == p.nth - 1) wtheta *= 0.5;

            for (int k = 0; k < p.nz; ++k) {
                const double weight = wx * wtheta * grid.dz;
                const double phi = state.Q(ii, j, k);
                const double P = state.P(ii, j, k);
                const double phix = dx1_4(state.Q, ii, j, k, grid.inv_dx);
                const double phith = dtheta1_mixed(state.Q, ii, j, k, grid.inv_dtheta);
                const double phiz = dz1_4(state.Q, ii, j, k, grid.inv_dz);

                local[0] += weight * (
                    0.5 * r * r * (P * P + phix * phix)
                    + 0.5 * f * phith * phith
                    + 0.5 * f * r * r * phiz * phiz);

                const double phi2 = phi * phi;
                local[1] += weight * 0.25 * f * r * r * phi2 * phi2;
            }
        }
    }

    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, decomp.comm);

    InitialEnergyParts result;
    result.quadratic = global[0];
    result.quartic_base = global[1];
    return result;
}

namespace {

double normalization_factor(const Params& p, const InitialEnergyParts& raw) {
    if (p.id_normalize == IDNormalizeKind::None) return 1.0;

    if (!(raw.quadratic > 0.0) || !std::isfinite(raw.quadratic)) {
        throw std::runtime_error("cannot energy-normalize zero or invalid initial data");
    }

    if (p.id_normalize == IDNormalizeKind::Quadratic) {
        return std::sqrt(p.id_target_energy / raw.quadratic);
    }

    // Solve for s>0 such that
    //   s^2 E2 + mu s^4 E4 = E_target.
    // This is the exact amplitude scaling for the pure defocusing cubic energy.
    if (p.mu < 0.0) {
        throw std::runtime_error(
            "full_cubic initial normalization requires mu >= 0; "
            "use --id-normalize quadratic for focusing or derivative-only tests");
    }

    const double a = p.mu * raw.quartic_base;
    const double b = raw.quadratic;
    if (a <= 1.0e-30) {
        return std::sqrt(p.id_target_energy / b);
    }

    const double discriminant = b * b + 4.0 * a * p.id_target_energy;
    const double y = (-b + std::sqrt(discriminant)) / (2.0 * a); // y=s^2
    if (!(y > 0.0) || !std::isfinite(y)) {
        throw std::runtime_error(
            "failed to determine full_cubic initial normalization factor");
    }
    return std::sqrt(y);
}

void scale_state(State& state, double scale) {
    for (double& value : state.Q.a) value *= scale;
    for (double& value : state.P.a) value *= scale;
}

} // namespace

void normalize_and_scale_initial_state(const Params& p,
                                       const Grid& grid,
                                       const Decomp& decomp,
                                       State& state) {
    const InitialEnergyParts raw =
        compute_initial_energy_parts(p, grid, decomp, state);

    double normalization_scale = 1.0;
    double final_scale = 1.0;

    if (p.id_normalize != IDNormalizeKind::None) {
        normalization_scale = normalization_factor(p, raw);
        // With normalization enabled, InitialData::apply intentionally builds an
        // amp=1 reference shape.  amp is the physical epsilon applied afterward.
        final_scale = p.amp * normalization_scale;
        scale_state(state, final_scale);
        exchange_state_ghosts(state, decomp);
    }

    const InitialEnergyParts final_parts =
        compute_initial_energy_parts(p, grid, decomp, state);

    if (decomp.rank == 0) {
        std::ofstream out(p.outdir + "/initial_data_normalization.txt");
        out << std::setprecision(17);
        out << "id_balance = " << to_string(p.id_balance) << "\n";
        out << "id_normalize = " << to_string(p.id_normalize) << "\n";
        out << "id_target_energy = " << p.id_target_energy << "\n";
        out << "amp = " << p.amp << "\n";
        out << "raw_E2 = " << raw.quadratic << "\n";
        out << "raw_E4_base = " << raw.quartic_base << "\n";
        out << "raw_full_cubic_reference = " << raw.full_cubic(p.mu) << "\n";
        out << "normalization_scale = " << normalization_scale << "\n";
        out << "final_scale_applied_after_InitialData = " << final_scale << "\n";
        out << "final_E2 = " << final_parts.quadratic << "\n";
        out << "final_E4_base = " << final_parts.quartic_base << "\n";
        out << "final_full_cubic_reference = "
            << final_parts.full_cubic(p.mu) << "\n";

        if (p.lambda != 0.0 || p.alpha2_phi_box != 0.0) {
            out << "note = derivative/alpha2 models do not in general conserve "
                   "the cubic reference energy\n";
        }

        if (p.verbose) {
            std::cout << std::setprecision(12)
                      << "Initial-data normalization:\n"
                      << "  id_balance=" << to_string(p.id_balance)
                      << " id_normalize=" << to_string(p.id_normalize) << "\n"
                      << "  raw E2=" << raw.quadratic
                      << " raw E4(base)=" << raw.quartic_base << "\n"
                      << "  normalization scale=" << normalization_scale
                      << " final scale=" << final_scale << "\n"
                      << "  final E2=" << final_parts.quadratic
                      << " final cubic-reference E="
                      << final_parts.full_cubic(p.mu) << "\n";
        }
    }
}
