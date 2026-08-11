#include "energy_balance.hpp"

#include "nonlinearity.hpp"
#include "operators.hpp"
#include "sponge.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

EnergyRateLocal rk4_weighted(double dt,
                             const EnergyRateLocal& r1,
                             const EnergyRateLocal& r2,
                             const EnergyRateLocal& r3,
                             const EnergyRateLocal& r4) {
    const double c = dt / 6.0;
    EnergyRateLocal result;
    result.inner_flux_out = c * (r1.inner_flux_out + 2.0 * r2.inner_flux_out
                               + 2.0 * r3.inner_flux_out + r4.inner_flux_out);
    result.outer_flux_out = c * (r1.outer_flux_out + 2.0 * r2.outer_flux_out
                               + 2.0 * r3.outer_flux_out + r4.outer_flux_out);
    result.inner_flux_positive = c * (
        r1.inner_flux_positive + 2.0 * r2.inner_flux_positive
      + 2.0 * r3.inner_flux_positive + r4.inner_flux_positive);
    result.outer_flux_positive = c * (
        r1.outer_flux_positive + 2.0 * r2.outer_flux_positive
      + 2.0 * r3.outer_flux_positive + r4.outer_flux_positive);
    result.sponge_loss = c * (r1.sponge_loss + 2.0 * r2.sponge_loss
                            + 2.0 * r3.sponge_loss + r4.sponge_loss);
    result.bulk_power = c * (r1.bulk_power + 2.0 * r2.bulk_power
                           + 2.0 * r3.bulk_power + r4.bulk_power);
    return result;
}

void add_in_place(EnergyRateLocal& dst, const EnergyRateLocal& src) {
    dst.inner_flux_out += src.inner_flux_out;
    dst.outer_flux_out += src.outer_flux_out;
    dst.inner_flux_positive += src.inner_flux_positive;
    dst.outer_flux_positive += src.outer_flux_positive;
    dst.sponge_loss += src.sponge_loss;
    dst.bulk_power += src.bulk_power;
}

} // namespace

EnergyBalance::EnergyBalance(const Params& p,
                             const Grid& grid,
                             const Decomp& decomp)
    : p_(p), grid_(grid), decomp_(decomp) {
    inner_probe_i_ = grid_.nearest_x_index(p_.horizon_probe_x);
    outer_probe_i_ = grid_.nearest_x_index(p_.outer_probe_x);

    if (inner_probe_i_ >= outer_probe_i_) {
        throw std::runtime_error(
            "energy control volume is empty: horizon probe must be left of outer probe");
    }

    if (p_.energy_balance && decomp_.rank == 0) {
        file_.open(p_.outdir + "/energy_balance.csv");
        file_ << "step,time,control_energy,initial_control_energy,"
              << "inner_flux_signed,outer_flux_signed,"
              << "inner_flux_positive,outer_flux_positive,"
              << "cum_inner_flux_signed,cum_outer_flux_signed,"
              << "cum_inner_flux_positive,cum_outer_flux_positive,"
              << "sponge_dissipation_rate,cum_sponge_dissipation,"
              << "bulk_power_rate,cum_bulk_power,"
              << "accounted_energy_signed,accounted_energy_one_way,"
              << "energy_balance_residual,relative_energy_balance_residual,"
              << "conservation_law_applicable,"
              << "inner_probe_x_actual,outer_probe_x_actual,"
              << "inner_probe_r_actual,outer_probe_r_actual\n";

        if (p_.verbose) {
            std::cout << std::setprecision(12)
                      << "Energy control volume:\n"
                      << "  inner probe i=" << inner_probe_i_
                      << " x=" << grid_.x[inner_probe_i_]
                      << " r=" << grid_.r[inner_probe_i_] << "\n"
                      << "  outer probe i=" << outer_probe_i_
                      << " x=" << grid_.x[outer_probe_i_]
                      << " r=" << grid_.r[outer_probe_i_] << "\n";

            const double outer_sponge_start =
                p_.x_max - p_.sponge_outer_width;
            if (p_.sponge_outer_strength > 0.0 &&
                grid_.x[outer_probe_i_] > outer_sponge_start) {
                std::cout
                    << "  note: outer probe lies inside the outer sponge; "
                    << "the ledger explicitly includes sponge loss between the probes.\n";
            }
            if (p_.ko_eps > 0.0) {
                std::cout
                    << "  note: KO dissipation is not represented by a separate analytic "
                    << "term; it appears in the balance residual. Use --ko-eps 0 for a "
                    << "clean continuum energy check.\n";
            }
            if ((p_.lambda != 0.0 && p_.q_kind != QKind::None) ||
                p_.alpha2_phi_box != 0.0) {
                std::cout
                    << "  note: derivative/alpha2 nonlinearities do not generally conserve "
                    << "the cubic reference energy. The ledger records their bulk work, "
                    << "but conservation_law_applicable will be zero.\n";
            }
        }
    }
}

void EnergyBalance::accumulate_rk4_local(double dt,
                                          const EnergyRateLocal& r1,
                                          const EnergyRateLocal& r2,
                                          const EnergyRateLocal& r3,
                                          const EnergyRateLocal& r4) {
    if (!p_.energy_balance || dt <= 0.0) return;
    add_in_place(cumulative_local_, rk4_weighted(dt, r1, r2, r3, r4));
}

double EnergyBalance::sponge_sigma(double x) const {
    return sponge_sigma_value(p_, x);
}

double EnergyBalance::compute_control_energy_local(const State& state) const {
    double local_energy = 0.0;

    for (int il = 0; il < decomp_.local_nx; ++il) {
        const int gi = decomp_.i_start + il;
        if (gi < inner_probe_i_ || gi > outer_probe_i_) continue;

        const int ii = il + state.Q.ng;
        const double r = grid_.r[gi];
        const double f = grid_.f[gi];
        double wx = grid_.dx;
        if (gi == inner_probe_i_ || gi == outer_probe_i_) wx *= 0.5;

        for (int j = 0; j < p_.nth; ++j) {
            double wtheta =
                2.0 * PI_CONST * std::sin(grid_.theta[j]) * grid_.dtheta;
            if (j == 0 || j == p_.nth - 1) wtheta *= 0.5;

            for (int k = 0; k < p_.nz; ++k) {
                const double phi = state.Q(ii, j, k);
                const double P = state.P(ii, j, k);
                const double phix =
                    dx1_4(state.Q, ii, j, k, grid_.inv_dx);
                const double phith =
                    dtheta1_mixed(state.Q, ii, j, k, grid_.inv_dtheta);
                const double phiz =
                    dz1_4(state.Q, ii, j, k, grid_.inv_dz);

                const double phi2 = phi * phi;
                const double density =
                    0.5 * r * r * (P * P + phix * phix)
                    + 0.5 * f * phith * phith
                    + 0.5 * f * r * r * phiz * phiz
                    + 0.25 * p_.mu * f * r * r * phi2 * phi2;

                local_energy += wx * wtheta * grid_.dz * density;
            }
        }
    }

    return local_energy;
}

EnergyRateLocal EnergyBalance::compute_rates_local(const State& state) const {
    EnergyRateLocal local;

    for (int il = 0; il < decomp_.local_nx; ++il) {
        const int gi = decomp_.i_start + il;
        if (gi < inner_probe_i_ || gi > outer_probe_i_) continue;

        const int ii = il + state.Q.ng;
        const double r = grid_.r[gi];
        const double f = grid_.f[gi];
        const double sigma = sponge_sigma(grid_.x[gi]);
        double wx = grid_.dx;
        if (gi == inner_probe_i_ || gi == outer_probe_i_) wx *= 0.5;

        for (int j = 0; j < p_.nth; ++j) {
            double wtheta =
                2.0 * PI_CONST * std::sin(grid_.theta[j]) * grid_.dtheta;
            if (j == 0 || j == p_.nth - 1) wtheta *= 0.5;

            for (int k = 0; k < p_.nz; ++k) {
                const double phi = state.Q(ii, j, k);
                const double P = state.P(ii, j, k);
                const double phix =
                    dx1_4(state.Q, ii, j, k, grid_.inv_dx);
                const double surface_weight = wtheta * grid_.dz;

                if (gi == inner_probe_i_) {
                    // An ingoing near-horizon wave has P*Phi_x > 0.
                    local.inner_flux_out += surface_weight * r * r * P * phix;
                }
                if (gi == outer_probe_i_) {
                    // An outgoing large-x wave has P*Phi_x < 0.
                    local.outer_flux_out -= surface_weight * r * r * P * phix;
                }

                const double volume_weight = wx * surface_weight;
                local.sponge_loss +=
                    volume_weight * sigma * r * r * P * P;

                if ((p_.lambda != 0.0 && p_.q_kind != QKind::None) ||
                    p_.alpha2_phi_box != 0.0) {
                    DerivPoint dp;
                    dp.phi = phi;
                    dp.P = P;
                    dp.phix = phix;
                    dp.phith =
                        dtheta1_mixed(state.Q, ii, j, k, grid_.inv_dtheta);
                    dp.phiz =
                        dz1_4(state.Q, ii, j, k, grid_.inv_dz);
                    dp.r = r;
                    dp.f = f;

                    const double N = nonlinear_source_N(p_, dp);
                    const double cubic = p_.mu * phi * phi * phi;
                    // Signed power added to the cubic reference-energy functional.
                    local.bulk_power += volume_weight
                        * (-f * r * r * P * (N - cubic));
                }
            }
        }
    }

    local.inner_flux_positive = std::max(local.inner_flux_out, 0.0);
    local.outer_flux_positive = std::max(local.outer_flux_out, 0.0);
    return local;
}

void EnergyBalance::write(int step, double time, const State& state) {
    if (!p_.energy_balance) return;

    const double local_energy = compute_control_energy_local(state);
    const EnergyRateLocal instantaneous_local = compute_rates_local(state);

    double control_energy = 0.0;
    MPI_Allreduce(&local_energy, &control_energy, 1, MPI_DOUBLE,
                  MPI_SUM, decomp_.comm);

    const double instantaneous_array[6] = {
        instantaneous_local.inner_flux_out,
        instantaneous_local.outer_flux_out,
        instantaneous_local.inner_flux_positive,
        instantaneous_local.outer_flux_positive,
        instantaneous_local.sponge_loss,
        instantaneous_local.bulk_power
    };
    double instantaneous_global[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(instantaneous_array, instantaneous_global, 6, MPI_DOUBLE,
                  MPI_SUM, decomp_.comm);

    const double cumulative_array[6] = {
        cumulative_local_.inner_flux_out,
        cumulative_local_.outer_flux_out,
        cumulative_local_.inner_flux_positive,
        cumulative_local_.outer_flux_positive,
        cumulative_local_.sponge_loss,
        cumulative_local_.bulk_power
    };
    double cumulative_global[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(cumulative_array, cumulative_global, 6, MPI_DOUBLE,
                  MPI_SUM, decomp_.comm);

    if (!have_initial_energy_) {
        initial_control_energy_ = control_energy;
        have_initial_energy_ = true;
    }

    // From dE_cv/dt = -F_inner - F_outer - D_sponge + P_bulk,
    // the signed accounted energy below should stay equal to E_cv(0).
    const double accounted_energy_signed =
        control_energy
        + cumulative_global[0]
        + cumulative_global[1]
        + cumulative_global[4]
        - cumulative_global[5];

    // One-way version used only as an intuitive partition when backflow is
    // negligible.  The signed ledger above is the authoritative identity.
    const double accounted_energy_one_way =
        control_energy
        + cumulative_global[2]
        + cumulative_global[3]
        + cumulative_global[4]
        - cumulative_global[5];

    const double residual =
        accounted_energy_signed - initial_control_energy_;
    const double relative_residual =
        (std::abs(initial_control_energy_) > 0.0)
        ? residual / std::abs(initial_control_energy_)
        : 0.0;

    const int conservation_law_applicable =
        ((p_.lambda == 0.0 || p_.q_kind == QKind::None) &&
         p_.alpha2_phi_box == 0.0 &&
         p_.ko_eps == 0.0) ? 1 : 0;

    if (decomp_.rank == 0) {
        file_ << std::setprecision(17)
              << step << "," << time << ","
              << control_energy << ","
              << initial_control_energy_ << ","
              << instantaneous_global[0] << ","
              << instantaneous_global[1] << ","
              << instantaneous_global[2] << ","
              << instantaneous_global[3] << ","
              << cumulative_global[0] << ","
              << cumulative_global[1] << ","
              << cumulative_global[2] << ","
              << cumulative_global[3] << ","
              << instantaneous_global[4] << ","
              << cumulative_global[4] << ","
              << instantaneous_global[5] << ","
              << cumulative_global[5] << ","
              << accounted_energy_signed << ","
              << accounted_energy_one_way << ","
              << residual << ","
              << relative_residual << ","
              << conservation_law_applicable << ","
              << grid_.x[inner_probe_i_] << ","
              << grid_.x[outer_probe_i_] << ","
              << grid_.r[inner_probe_i_] << ","
              << grid_.r[outer_probe_i_] << "\n";
        file_.flush();
    }
}
