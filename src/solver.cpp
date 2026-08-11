#include "solver.hpp"
#include "nonlinearity.hpp"
#include "operators.hpp"
#include "sponge.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

Solver::Solver(const Params& p, const Grid& grid, const Decomp& decomp)
    : p_(p),
      grid_(grid),
      decomp_(decomp),
      diagnostics_(p, grid, decomp),
      energy_balance_(p, grid, decomp),
      k1_(decomp.local_nx, p.nth, p.nz, p.nghost),
      k2_(decomp.local_nx, p.nth, p.nz, p.nghost),
      k3_(decomp.local_nx, p.nth, p.nz, p.nghost),
      k4_(decomp.local_nx, p.nth, p.nz, p.nghost),
      tmp_(decomp.local_nx, p.nth, p.nz, p.nghost) {
    need_cubic_ = (p_.mu != 0.0);
    need_derivative_source_ = (p_.lambda != 0.0 && p_.q_kind != QKind::None);
    need_phix_source_ = need_derivative_source_ &&
        (p_.q_kind == QKind::Metric || p_.q_kind == QKind::Spatial || p_.q_kind == QKind::NullX);
    need_theta_z_source_ = need_derivative_source_ &&
        (p_.q_kind == QKind::Metric || p_.q_kind == QKind::Spatial);
    precompute_local_coefficients();
}

void Solver::precompute_local_coefficients() {
    const int n = decomp_.local_nx;
    f_loc_.assign(n, 0.0);
    r_loc_.assign(n, 0.0);
    f_over_r2_loc_.assign(n, 0.0);
    sigma_loc_.assign(n, 0.0);
    lx_cm2_.assign(n, 0.0);
    lx_cm1_.assign(n, 0.0);
    lx_c0_.assign(n, 0.0);
    lx_cp1_.assign(n, 0.0);
    lx_cp2_.assign(n, 0.0);
    control_x_weight_loc_.assign(n, 0.0);

    theta_energy_weight_.assign(p_.nth, 0.0);
    for (int j = 0; j < p_.nth; ++j) {
        theta_energy_weight_[j] =
            2.0 * PI_CONST * std::sin(grid_.theta[j]) * grid_.dtheta;
        if (j == 0 || j == p_.nth - 1) theta_energy_weight_[j] *= 0.5;
    }

    inner_probe_i_ = energy_balance_.inner_probe_i();
    outer_probe_i_ = energy_balance_.outer_probe_i();

    const double inv12dx = grid_.inv_dx / 12.0;
    const double inv12dx2 = grid_.inv_dx2 / 12.0;
    for (int il = 0; il < n; ++il) {
        const int gi = decomp_.i_start + il;
        const double r = grid_.r[gi];
        const double f = grid_.f[gi];
        const double a = 2.0 * f / r;

        r_loc_[il] = r;
        f_loc_[il] = f;
        f_over_r2_loc_[il] = f / (r * r);
        sigma_loc_[il] = sponge_sigma(grid_.x[gi]);
        lx_cm2_[il] = -inv12dx2 + a * inv12dx;
        lx_cm1_[il] = 16.0 * inv12dx2 - 8.0 * a * inv12dx;
        lx_c0_[il] = -30.0 * inv12dx2;
        lx_cp1_[il] = 16.0 * inv12dx2 + 8.0 * a * inv12dx;
        lx_cp2_[il] = -inv12dx2 - a * inv12dx;

        if (gi >= inner_probe_i_ && gi <= outer_probe_i_) {
            double wx = grid_.dx;
            if (gi == inner_probe_i_ || gi == outer_probe_i_) wx *= 0.5;
            control_x_weight_loc_[il] = wx;
        }
    }
}

double Solver::sponge_sigma(double x) const {
    return sponge_sigma_value(p_, x);
}

void Solver::rhs(State& y, State& dy) {
    exchange_state_ghosts(y, decomp_);
    const int ng = y.Q.ng;

    double local_inner_flux = 0.0;
    double local_outer_flux = 0.0;
    double local_sponge_loss = 0.0;
    double local_bulk_power = 0.0;

#ifdef NLIN_WAVE_USE_OPENMP
#pragma omp parallel for collapse(2) schedule(static) \
    reduction(+:local_inner_flux,local_outer_flux,local_sponge_loss,local_bulk_power)
#endif
    for (int il = 0; il < decomp_.local_nx; ++il) {
        for (int j = 0; j < p_.nth; ++j) {
            const int ii = il + ng;
            const int gi = decomp_.i_start + il;
            const double r = r_loc_[il];
            const double f = f_loc_[il];
            const double f_over_r2 = f_over_r2_loc_[il];
            const double sig = sigma_loc_[il];
            const double cm2 = lx_cm2_[il];
            const double cm1 = lx_cm1_[il];
            const double c0 = lx_c0_[il];
            const double cp1 = lx_cp1_[il];
            const double cp2 = lx_cp2_[il];
            const bool in_control_volume = control_x_weight_loc_[il] > 0.0;
            const bool on_flux_probe =
                (gi == inner_probe_i_ || gi == outer_probe_i_);
            const double surface_weight =
                theta_energy_weight_[j] * grid_.dz;

            for (int k = 0; k < p_.nz; ++k) {
                const double phi = y.Q(ii, j, k);
                const double P = y.P(ii, j, k);

                const double Lx = radial_lx_4(y.Q, ii, j, k, cm2, cm1, c0, cp1, cp2);
                const double Ltheta = f_over_r2 * theta_laplacian_axisym(y.Q, ii, j, k, grid_);
                const double Lz = f * dz2_4(y.Q, ii, j, k, grid_.inv_dz2);

                DerivPoint dp;
                dp.phi = phi;
                dp.P = P;
                dp.r = r;
                dp.f = f;

                double phix_for_balance = 0.0;
                const bool need_phix_here =
                    need_phix_source_ || (p_.energy_balance && on_flux_probe);
                if (need_phix_here) {
                    phix_for_balance =
                        dx1_4(y.Q, ii, j, k, grid_.inv_dx);
                    dp.phix = phix_for_balance;
                }
                if (need_theta_z_source_) {
                    dp.phith =
                        dtheta1_mixed(y.Q, ii, j, k, grid_.inv_dtheta);
                    dp.phiz = dz1_4(y.Q, ii, j, k, grid_.inv_dz);
                }

                double N = 0.0;
                if (need_cubic_ || need_derivative_source_) {
                    N = nonlinear_source_N(p_, dp);
                }

                double rhs_Q = P;
                double rhs_P = Lx + Ltheta + Lz - f * N - sig * P;
                if (p_.ko_eps > 0.0) {
                    const double koQ = ko6_x(y.Q, ii, j, k, grid_.inv_dx)
                                     + ko6_z(y.Q, ii, j, k, grid_.inv_dz)
                                     + ko6_theta(y.Q, ii, j, k, grid_.inv_dtheta);
                    const double koP = ko6_x(y.P, ii, j, k, grid_.inv_dx)
                                     + ko6_z(y.P, ii, j, k, grid_.inv_dz)
                                     + ko6_theta(y.P, ii, j, k, grid_.inv_dtheta);
                    rhs_Q += p_.ko_eps * koQ;
                    rhs_P += p_.ko_eps * koP;
                }
                dy.Q(ii, j, k) = rhs_Q;
                dy.P(ii, j, k) = rhs_P;

                if (p_.energy_balance) {
                    if (gi == inner_probe_i_) {
                        local_inner_flux +=
                            surface_weight * r * r * P * phix_for_balance;
                    }
                    if (gi == outer_probe_i_) {
                        local_outer_flux -=
                            surface_weight * r * r * P * phix_for_balance;
                    }

                    if (in_control_volume) {
                        const double volume_weight =
                            control_x_weight_loc_[il] * surface_weight;
                        local_sponge_loss +=
                            volume_weight * sig * r * r * P * P;

                        if (p_.lambda != 0.0 || p_.alpha2_phi_box != 0.0) {
                            // If a derivative source is active, all derivatives
                            // required by nonlinear_source_N were already set.
                            // For alpha2 with a cubic-only numerator, no extra
                            // spatial derivatives are needed.
                            const double cubic = p_.mu * phi * phi * phi;
                            local_bulk_power += volume_weight
                                * (-f * r * r * P * (N - cubic));
                        }
                    }
                }
            }
        }
    }

    last_energy_rates_.inner_flux_out = local_inner_flux;
    last_energy_rates_.outer_flux_out = local_outer_flux;
    last_energy_rates_.inner_flux_positive = std::max(local_inner_flux, 0.0);
    last_energy_rates_.outer_flux_positive = std::max(local_outer_flux, 0.0);
    last_energy_rates_.sponge_loss = local_sponge_loss;
    last_energy_rates_.bulk_power = local_bulk_power;
}

void Solver::rk4_step(State& y, double dt) {
    rhs(y, k1_);
    const EnergyRateLocal e1 = last_energy_rates_;

    tmp_.combine_physical(1.0, y, 0.5 * dt, k1_);
    rhs(tmp_, k2_);
    const EnergyRateLocal e2 = last_energy_rates_;

    tmp_.combine_physical(1.0, y, 0.5 * dt, k2_);
    rhs(tmp_, k3_);
    const EnergyRateLocal e3 = last_energy_rates_;

    tmp_.combine_physical(1.0, y, dt, k3_);
    rhs(tmp_, k4_);
    const EnergyRateLocal e4 = last_energy_rates_;

    fused_rk4_update_physical(y, k1_, k2_, k3_, k4_, dt);
    energy_balance_.accumulate_rk4_local(dt, e1, e2, e3, e4);
}

void Solver::run(State& state) {
    const int nsteps = static_cast<int>(std::ceil(p_.t_final / p_.dt));
    double t = 0.0;
    double last_diag_t = 0.0;

    exchange_state_ghosts(state, decomp_);
    diagnostics_.write(0, t, state, 0.0);
    energy_balance_.write(0, t, state);
    diagnostics_.write_fields(0, t, state);

    int diag_count = 0;
    for (int step = 1; step <= nsteps; ++step) {
        const double dt = std::min(p_.dt, p_.t_final - t);
        if (dt <= 0.0) break;

        rk4_step(state, dt);
        t += dt;
        const bool need_output =
            (step % p_.output_every == 0) || (step == nsteps);
        if (need_output) {
            exchange_state_ghosts(state, decomp_);
            const double diag_dt = t - last_diag_t;
            diagnostics_.write(step, t, state, diag_dt);
            energy_balance_.write(step, t, state);
            last_diag_t = t;
            ++diag_count;

            if (decomp_.rank == 0 && p_.verbose) {
                std::cout << "step " << step << "/" << nsteps
                          << " t=" << t << std::endl;
            }
            if (p_.field_every > 0 &&
                diag_count % p_.field_every == 0) {
                diagnostics_.write_fields(step, t, state);
            }
        }
    }
}
