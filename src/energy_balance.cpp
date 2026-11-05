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
        components_file_.open(p_.outdir + "/energy_components.csv");
        components_file_
            << "step,time,control_energy,energy_kinetic,energy_radial,"
            << "energy_angular,energy_kk,energy_cubic,energy_partition_residual\n";

        fluxes_file_.open(p_.outdir + "/energy_fluxes.csv");
        fluxes_file_
            << "step,time,inner_flux_signed,outer_flux_signed,"
            << "sponge_dissipation_rate,bulk_power_rate,"
            << "cum_inner_flux_signed,cum_outer_flux_signed,"
            << "cum_sponge_dissipation,cum_bulk_power\n";

        budget_file_.open(p_.outdir + "/energy_budget.csv");
        budget_file_
            << "step,time,initial_control_energy,control_energy,"
            << "cum_inner_flux_signed,cum_outer_flux_signed,"
            << "cum_sponge_dissipation,cum_bulk_power,"
            << "accounted_energy_signed,energy_balance_residual,"
            << "relative_energy_balance_residual,conservation_law_applicable\n";

        checks_file_.open(p_.outdir + "/energy_checks.csv");
        checks_file_
            << "step,time,residual_without_horizon,residual_without_outer,"
            << "residual_without_sponge,residual_without_bulk,"
            << "horizon_energy_only_estimate,horizon_inferred_from_balance,"
            << "horizon_inference_difference,instantaneous_balance_rhs,"
            << "control_energy_rate_finite_difference,rate_balance_residual\n";

        // Probe locations are constant in time; record once, not in every CSV row.
        std::ofstream probe_info(p_.outdir + "/energy_probe_locations.txt");
        probe_info << std::setprecision(17)
                   << "inner_probe_x_actual=" << grid_.x[inner_probe_i_] << "\n"
                   << "outer_probe_x_actual=" << grid_.x[outer_probe_i_] << "\n"
                   << "inner_probe_r_actual=" << grid_.r[inner_probe_i_] << "\n"
                   << "outer_probe_r_actual=" << grid_.r[outer_probe_i_] << "\n";

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

// Quadratic kinetic/radial/angular/KK energies plus the cubic potential.
// All use the same radial control volume and weights as the pre-existing code.
void EnergyBalance::compute_control_components_local(const State& state,
                                                     double components[5]) const {
    for (int c = 0; c < 5; ++c) components[c] = 0.0;
    for (int il = 0; il < decomp_.local_nx; ++il) {
        const int gi = decomp_.i_start + il;
        if (gi < inner_probe_i_ || gi > outer_probe_i_) continue;
        const int ii = il + state.Q.ng;
        const double r = grid_.r[gi];
        const double f = grid_.f[gi];
        double wx = grid_.dx;
        if (gi == inner_probe_i_ || gi == outer_probe_i_) wx *= 0.5;
        for (int j = 0; j < p_.nth; ++j) {
            double wtheta = 2.0 * PI_CONST * std::sin(grid_.theta[j]) * grid_.dtheta;
            if (j == 0 || j == p_.nth - 1) wtheta *= 0.5;
            for (int k = 0; k < p_.nz; ++k) {
                const double phi = state.Q(ii, j, k);
                const double P = state.P(ii, j, k);
                const double phix = dx1_4(state.Q, ii, j, k, grid_.inv_dx);
                const double phith = dtheta1_mixed(state.Q, ii, j, k, grid_.inv_dtheta);
                const double phiz = dz1_4(state.Q, ii, j, k, grid_.inv_dz);
                const double v = wx * wtheta * grid_.dz;
                components[0] += v * 0.5 * r * r * P * P;
                components[1] += v * 0.5 * r * r * phix * phix;
                components[2] += v * 0.5 * f * phith * phith;
                components[3] += v * 0.5 * f * r * r * phiz * phiz;
                const double phi2 = phi * phi;
                components[4] += v * 0.25 * p_.mu * f * r * r * phi2 * phi2;
            }
        }
    }
}

double EnergyBalance::compute_control_energy_local(const State& state) const {
    double components[5];
    compute_control_components_local(state, components);
    double total = 0.0;
    for (int c = 0; c < 5; ++c) total += components[c];
    return total;
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

    double local_components[5];
    compute_control_components_local(state, local_components);
    double global_components[5] = {};
    MPI_Allreduce(local_components, global_components, 5, MPI_DOUBLE,
                  MPI_SUM, decomp_.comm);
    double control_energy = 0.0;
    for (double v : global_components) control_energy += v;
    const EnergyRateLocal instantaneous_local = compute_rates_local(state);


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

    const double H = cumulative_global[0];
    const double O = cumulative_global[1];
    const double D = cumulative_global[4];
    const double B = cumulative_global[5];
    const double energy_departed = initial_control_energy_ - control_energy;
    const double accounted_no_horizon = accounted_energy_signed - H;
    const double accounted_no_outer = accounted_energy_signed - O;
    const double accounted_no_sponge = accounted_energy_signed - D;
    const double accounted_no_bulk = accounted_energy_signed + B;
    const double horizon_energy_only = energy_departed;
    const double horizon_inferred = energy_departed - O - D + B;
    const double horizon_inference_diff = H - horizon_inferred;
    const double instantaneous_rhs = -instantaneous_global[0]
        - instantaneous_global[1] - instantaneous_global[4]
        + instantaneous_global[5];
    // The finite-difference derivative is only diagnostic; it does not
    // participate in the RK4-stage cumulative ledger.
    double finite_difference_dEdt = 0.0;
    if (have_previous_output_ && time > previous_time_)
        finite_difference_dEdt = (control_energy - previous_energy_)
                                  / (time - previous_time_);
    const double rate_balance_residual = have_previous_output_
        ? finite_difference_dEdt - instantaneous_rhs : 0.0;
    previous_energy_ = control_energy;
    previous_time_ = time;
    have_previous_output_ = true;
    if (decomp_.rank == 0) {
        const double energy_partition_residual =
            control_energy - (global_components[0] + global_components[1]
                              + global_components[2] + global_components[3]
                              + global_components[4]);

        components_file_ << std::setprecision(17)
            << step << "," << time << "," << control_energy << ","
            << global_components[0] << "," << global_components[1] << ","
            << global_components[2] << "," << global_components[3] << ","
            << global_components[4] << "," << energy_partition_residual << "\n";

        fluxes_file_ << std::setprecision(17)
            << step << "," << time << ","
            << instantaneous_global[0] << "," << instantaneous_global[1] << ","
            << instantaneous_global[4] << "," << instantaneous_global[5] << ","
            << cumulative_global[0] << "," << cumulative_global[1] << ","
            << cumulative_global[4] << "," << cumulative_global[5] << "\n";

        budget_file_ << std::setprecision(17)
            << step << "," << time << ","
            << initial_control_energy_ << "," << control_energy << ","
            << H << "," << O << "," << D << "," << B << ","
            << accounted_energy_signed << "," << residual << ","
            << relative_residual << "," << conservation_law_applicable << "\n";

        checks_file_ << std::setprecision(17)
            << step << "," << time << ","
            << (accounted_no_horizon - initial_control_energy_) << ","
            << (accounted_no_outer - initial_control_energy_) << ","
            << (accounted_no_sponge - initial_control_energy_) << ","
            << (accounted_no_bulk - initial_control_energy_) << ","
            << horizon_energy_only << "," << horizon_inferred << ","
            << horizon_inference_diff << ","
            << instantaneous_rhs << "," << finite_difference_dEdt << ","
            << rate_balance_residual << "\n";

        components_file_.flush();
        fluxes_file_.flush();
        budget_file_.flush();
        checks_file_.flush();
    }
}
