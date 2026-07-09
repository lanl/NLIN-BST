#include "diagnostics.hpp"
#include "operators.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

Diagnostics::Diagnostics(const Params& p, const Grid& grid, const Decomp& decomp)
    : p_(p), grid_(grid), decomp_(decomp) {
    if (decomp_.rank == 0) {
        std::filesystem::create_directories(p_.outdir);
        std::filesystem::create_directories(p_.outdir + "/fields");
    }
    MPI_Barrier(decomp_.comm);

    Y_.assign(p_.lmax + 1, std::vector<double>(p_.nth, 0.0));
    for (int ell = 0; ell <= p_.lmax; ++ell) {
        for (int j = 0; j < p_.nth; ++j) {
            Y_[ell][j] = Y_l0(ell, grid_.theta[j]);
        }
    }

    C_.assign(p_.nmax + 1, std::vector<double>(p_.nz, 0.0));
    S_.assign(p_.nmax + 1, std::vector<double>(p_.nz, 0.0));
    for (int n = 0; n <= p_.nmax; ++n) {
        const double kappa = kappa_n(p_.Lz, n);
        for (int k = 0; k < p_.nz; ++k) {
            C_[n][k] = (n == 0) ? 1.0 : std::cos(kappa * grid_.z[k]);
            S_[n][k] = (n == 0) ? 0.0 : std::sin(kappa * grid_.z[k]);
        }
    }

    theta_weight_.assign(p_.nth, 0.0);
    for (int j = 0; j < p_.nth; ++j) {
        theta_weight_[j] = 2.0 * PI_CONST * std::sin(grid_.theta[j]) * grid_.dtheta;
        if (j == 0 || j == p_.nth - 1) theta_weight_[j] *= 0.5;
    }

    build_angular_gram_inverse();

    modes_.reserve(static_cast<std::size_t>((p_.lmax + 1) * (p_.nmax + 1)));
    for (int ell = 0; ell <= p_.lmax; ++ell) {
        for (int n = 0; n <= p_.nmax; ++n) {
            ModeInfo m;
            m.ell = ell;
            m.n = n;
            m.kappa = kappa_n(p_.Lz, n);
            if (ell > 0) {
                m.q_trap = m.kappa * m.kappa * p_.rh * p_.rh / static_cast<double>(ell * (ell + 1));
            } else {
                m.q_trap = std::numeric_limits<double>::infinity();
            }

            double rmin = 0.0;
            double xmin = 0.0;
            const bool ok = find_stable_well(p_.rh, p_.Lz, ell, n,
                                             p_.rh * (1.0 + 1e-8), grid_.r.back(),
                                             rmin, xmin);
            m.trapped = ok;
            if (ok) {
                m.r_min = rmin;
                m.x_min = xmin;
                m.V_min = scalar_potential_V(rmin, p_.rh, p_.Lz, ell, n);
            } else {
                m.r_min = std::max(p_.r_center, p_.rh * (1.0 + 1e-6));
                m.x_min = grid_.x_from_r(m.r_min);
                m.V_min = scalar_potential_V(m.r_min, p_.rh, p_.Lz, ell, n);
            }
            modes_.push_back(m);
        }
    }

    open_files();
    write_mode_info();
    write_grid_files();
}

Diagnostics::~Diagnostics() = default;

void Diagnostics::build_angular_gram_inverse() {
    const int L = p_.lmax + 1;
    std::vector<std::vector<double>> G(L, std::vector<double>(L, 0.0));

    for (int a = 0; a < L; ++a) {
        for (int b = 0; b < L; ++b) {
            double sum = 0.0;
            for (int j = 0; j < p_.nth; ++j) {
                sum += theta_weight_[j] * Y_[a][j] * Y_[b][j];
            }
            G[a][b] = sum;
        }
    }

    angular_gram_inv_.assign(L, std::vector<double>(L, 0.0));
    for (int i = 0; i < L; ++i) angular_gram_inv_[i][i] = 1.0;

    for (int col = 0; col < L; ++col) {
        int pivot = col;
        double best = std::abs(G[col][col]);
        for (int row = col + 1; row < L; ++row) {
            const double val = std::abs(G[row][col]);
            if (val > best) {
                best = val;
                pivot = row;
            }
        }
        if (best < 1e-14) {
            throw std::runtime_error("angular Gram matrix nearly singular; increase nth or reduce lmax");
        }
        if (pivot != col) {
            std::swap(G[pivot], G[col]);
            std::swap(angular_gram_inv_[pivot], angular_gram_inv_[col]);
        }

        const double diag = G[col][col];
        for (int j = 0; j < L; ++j) {
            G[col][j] /= diag;
            angular_gram_inv_[col][j] /= diag;
        }
        for (int row = 0; row < L; ++row) {
            if (row == col) continue;
            const double factor = G[row][col];
            if (factor == 0.0) continue;
            for (int j = 0; j < L; ++j) {
                G[row][j] -= factor * G[col][j];
                angular_gram_inv_[row][j] -= factor * angular_gram_inv_[col][j];
            }
        }
    }

    if (decomp_.rank == 0 && p_.verbose) {
        std::cerr << "[Diagnostics] Built inverse angular Gram matrix for lmax="
                  << p_.lmax << ", nth=" << p_.nth << "\n";
    }
}

void Diagnostics::open_files() {
    if (decomp_.rank != 0) return;

    diag_file_.open(p_.outdir + "/diagnostics.csv");
    diag_file_ << "step,time,total_l2,trap_l2,total_energy,trap_energy,"
               << "max_abs_phi,max_abs_P,Hs_trap,ell_centroid,n_centroid,high_fraction\n";

    modal_file_.open(p_.outdir + "/modal_trap_spectrum.csv");
    modal_file_ << "step,time";
    for (const auto& m : modes_) modal_file_ << ",Etrap_l" << m.ell << "_n" << m.n;
    modal_file_ << "\n";

    modal_well_file_.open(p_.outdir + "/modal_well_coefficients.csv");
    modal_well_file_ << "step,time";
    for (const auto& m : modes_) {
        modal_well_file_ << ",PhiAbs_l" << m.ell << "_n" << m.n
                         << ",PhiCos_l" << m.ell << "_n" << m.n
                         << ",PhiSin_l" << m.ell << "_n" << m.n
                         << ",PAbs_l" << m.ell << "_n" << m.n
                         << ",PCos_l" << m.ell << "_n" << m.n
                         << ",PSin_l" << m.ell << "_n" << m.n;
    }
    modal_well_file_ << "\n";

    horizon_modal_file_.open(p_.outdir + "/horizon_spectrum.csv");
    horizon_modal_file_ << "step,time";
    for (const auto& m : modes_) horizon_modal_file_ << ",Hshell_l" << m.ell << "_n" << m.n;
    horizon_modal_file_ << "\n";

    flux_file_.open(p_.outdir + "/fluxes.csv");
    flux_file_ << "step,time,inner_flux_signed,outer_flux_signed,"
               << "inner_flux_positive,outer_flux_positive,cum_inner_flux,cum_outer_flux\n";
}

void Diagnostics::write_mode_info() const {
    if (decomp_.rank != 0) return;
    std::ofstream f(p_.outdir + "/mode_info.csv");
    f << "ell,n,kappa,q_trap,trapped,r_min,x_min,V_min\n";
    f << std::setprecision(17);
    for (const auto& m : modes_) {
        f << m.ell << "," << m.n << "," << m.kappa << "," << m.q_trap << ","
          << (m.trapped ? 1 : 0) << "," << m.r_min << "," << m.x_min << "," << m.V_min << "\n";
    }
}

void Diagnostics::write_grid_files() const {
    if (decomp_.rank != 0) return;

    {
        std::ofstream f(p_.outdir + "/grid_x.csv");
        f << "i,x,r,f\n" << std::setprecision(17);
        for (int i = 0; i < p_.nx; ++i) {
            f << i << "," << grid_.x[i] << "," << grid_.r[i] << "," << grid_.f[i] << "\n";
        }
    }
    {
        std::ofstream f(p_.outdir + "/grid_theta.csv");
        f << "j,theta,sin_theta,cot_theta\n" << std::setprecision(17);
        for (int j = 0; j < p_.nth; ++j) {
            f << j << "," << grid_.theta[j] << "," << grid_.sin_theta[j] << "," << grid_.cot_theta[j] << "\n";
        }
    }
    {
        std::ofstream f(p_.outdir + "/grid_z.csv");
        f << "k,z\n" << std::setprecision(17);
        for (int k = 0; k < p_.nz; ++k) {
            f << k << "," << grid_.z[k] << "\n";
        }
    }
    {
        std::ofstream f(p_.outdir + "/fields/README.txt");
        f << "Full binary field files are raw little-endian double arrays.\n";
        f << "Layout for field_vars=phi: field[i,j,k] with k fastest, then j, then i.\n";
        f << "Layout for field_vars=phiP: field[i,j,k,var] with var=0 phi, var=1 P.\n";
        f << "Shape is stored in each fields_XXXXXXXX.txt metadata file.\n";
    }
}

void Diagnostics::compute_energy_and_flux(const State& state,
                                          double& total_energy,
                                          double& trap_energy,
                                          double& total_l2,
                                          double& trap_l2,
                                          double& max_phi,
                                          double& max_P,
                                          double& inner_signed_flux,
                                          double& outer_signed_flux) const {
    double local_sum[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double global_sum[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double local_max[2] = {0.0, 0.0};
    double global_max[2] = {0.0, 0.0};

    const int i_h = grid_.nearest_x_index(p_.horizon_probe_x);
    const int i_o = grid_.nearest_x_index(p_.outer_probe_x);
    const double trap_x_center = grid_.x_from_r(std::max(p_.r_center, p_.rh * (1.0 + 1e-6)));

    for (int il = 0; il < decomp_.local_nx; ++il) {
        const int gi = decomp_.i_start + il;
        const int ii = il + state.Q.ng;
        const double r = grid_.r[gi];
        const double f = std::max(grid_.f[gi], 1e-12);
        const bool in_trap = std::abs(grid_.x[gi] - trap_x_center) <= p_.trap_dx_half_width;
        const double wx = grid_.dx * ((gi == 0 || gi == p_.nx - 1) ? 0.5 : 1.0);

        for (int j = 0; j < p_.nth; ++j) {
            const double wth = theta_weight_[j];
            for (int k = 0; k < p_.nz; ++k) {
                const double phi = state.Q(ii, j, k);
                const double P = state.P(ii, j, k);
                const double phix = dx1_4(state.Q, ii, j, k, grid_.inv_dx);
                const double phith = dtheta1_mixed(state.Q, ii, j, k, grid_.inv_dtheta);
                const double phiz = dz1_4(state.Q, ii, j, k, grid_.inv_dz);

                const double dV = wth * grid_.dz * wx;
                const double phi2 = phi * phi;
                const double energy_density =
                    0.5 * r * r * (P * P + phix * phix)
                  + 0.5 * f * phith * phith
                  + 0.5 * f * r * r * phiz * phiz
                  + 0.25 * p_.mu * f * r * r * phi2 * phi2;
                const double eint = dV * energy_density;
                const double l2int = dV * f * r * r * phi2;

                local_sum[0] += eint;
                local_sum[2] += l2int;
                if (in_trap) {
                    local_sum[1] += eint;
                    local_sum[3] += l2int;
                }

                local_max[0] = std::max(local_max[0], std::abs(phi));
                local_max[1] = std::max(local_max[1], std::abs(P));

                if (gi == i_h || gi == i_o) {
                    const double flux_density = wth * grid_.dz * r * r * P * phix;
                    if (gi == i_h) local_sum[4] += flux_density;
                    if (gi == i_o) local_sum[5] += flux_density;
                }
            }
        }
    }

    MPI_Allreduce(local_sum, global_sum, 8, MPI_DOUBLE, MPI_SUM, decomp_.comm);
    MPI_Allreduce(local_max, global_max, 2, MPI_DOUBLE, MPI_MAX, decomp_.comm);

    total_energy = global_sum[0];
    trap_energy = global_sum[1];
    total_l2 = global_sum[2];
    trap_l2 = global_sum[3];
    inner_signed_flux = global_sum[4];
    outer_signed_flux = global_sum[5];
    max_phi = global_max[0];
    max_P = global_max[1];
}

void Diagnostics::project_all_ell_components(const Field3D& u,
                                             int ii,
                                             int n,
                                             int derivative_kind,
                                             std::vector<double>& cos_coeff,
                                             std::vector<double>& sin_coeff) const {
    const int L = p_.lmax + 1;
    cos_coeff.assign(L, 0.0);
    sin_coeff.assign(L, 0.0);

    std::vector<double> raw_cos(L, 0.0);
    std::vector<double> raw_sin(L, 0.0);
    const double z_pref = (n == 0) ? (1.0 / p_.Lz) : (2.0 / p_.Lz);
    const bool do_sin = p_.modal_include_sine && n > 0;

    for (int ell = 0; ell < L; ++ell) {
        double sc = 0.0;
        double ss = 0.0;
        for (int j = 0; j < p_.nth; ++j) {
            const double wy = theta_weight_[j] * Y_[ell][j];
            for (int k = 0; k < p_.nz; ++k) {
                double val = 0.0;
                if (derivative_kind == 0) {
                    val = u(ii, j, k);
                } else {
                    val = dx1_4(u, ii, j, k, grid_.inv_dx);
                }
                const double wz = wy * grid_.dz;
                sc += val * wz * C_[n][k];
                if (do_sin) ss += val * wz * S_[n][k];
            }
        }
        raw_cos[ell] = z_pref * sc;
        raw_sin[ell] = z_pref * ss;
    }

    for (int ell = 0; ell < L; ++ell) {
        double cc = 0.0;
        double ss = 0.0;
        for (int ellp = 0; ellp < L; ++ellp) {
            cc += angular_gram_inv_[ell][ellp] * raw_cos[ellp];
            ss += angular_gram_inv_[ell][ellp] * raw_sin[ellp];
        }
        cos_coeff[ell] = cc;
        sin_coeff[ell] = ss;
    }
}

std::vector<double> Diagnostics::compute_modal_spectrum(const State& state, bool horizon_shell) const {
    std::vector<double> local(modes_.size(), 0.0);
    std::vector<double> global(modes_.size(), 0.0);

    const int hgi = grid_.nearest_x_index(p_.horizon_probe_x);
    int il_begin = 0;
    int il_end = decomp_.local_nx;
    if (horizon_shell) {
        if (!decomp_.owns_global_i(hgi)) {
            MPI_Allreduce(local.data(), global.data(), static_cast<int>(local.size()), MPI_DOUBLE, MPI_SUM, decomp_.comm);
            return global;
        }
        il_begin = decomp_.local_from_global(hgi);
        il_end = il_begin + 1;
    }

    std::vector<double> q_cos, q_sin, p_cos, p_sin, qx_cos, qx_sin;

    for (int il = il_begin; il < il_end; ++il) {
        const int gi = decomp_.i_start + il;
        const int ii = il + state.Q.ng;
        const double r = grid_.r[gi];
        const double f = grid_.f[gi];
        const double wx = grid_.dx * ((gi == 0 || gi == p_.nx - 1) ? 0.5 : 1.0);

        for (int n = 0; n <= p_.nmax; ++n) {
            project_all_ell_components(state.Q, ii, n, 0, q_cos, q_sin);
            project_all_ell_components(state.P, ii, n, 0, p_cos, p_sin);
            project_all_ell_components(state.Q, ii, n, 1, qx_cos, qx_sin);

            for (int ell = 0; ell <= p_.lmax; ++ell) {
                const int mindex = mode_index(ell, n);
                const ModeInfo& m = modes_[mindex];

                bool include = false;
                if (horizon_shell) {
                    include = true;
                } else if (m.trapped) {
                    include = std::abs(grid_.x[gi] - m.x_min) <= p_.trap_dx_half_width;
                }
                if (!include) continue;

                auto component_energy = [&](double phi_ln, double P_ln, double phix_ln) {
                    const double chi = r * phi_ln;
                    const double chit = r * P_ln;
                    const double chix = r * phix_ln + f * phi_ln;
                    const double V = scalar_potential_V(r, p_.rh, p_.Lz, ell, n);
                    return 0.5 * (chit * chit + chix * chix + V * chi * chi);
                };

                double e = component_energy(q_cos[ell], p_cos[ell], qx_cos[ell]);
                if (p_.modal_include_sine && n > 0) {
                    e += component_energy(q_sin[ell], p_sin[ell], qx_sin[ell]);
                }
                if (!horizon_shell) e *= wx;
                local[mindex] += e;
            }
        }
    }

    MPI_Allreduce(local.data(), global.data(), static_cast<int>(local.size()), MPI_DOUBLE, MPI_SUM, decomp_.comm);
    return global;
}

std::vector<ModeCoefficients> Diagnostics::compute_modal_well_coefficients(const State& state) const {
    std::vector<double> local(4 * modes_.size(), 0.0);
    std::vector<double> global(4 * modes_.size(), 0.0);
    std::vector<double> q_cos, q_sin, p_cos, p_sin;

    for (const auto& m : modes_) {
        if (!m.trapped) continue;
        const int gi = grid_.nearest_x_index(m.x_min);
        if (!decomp_.owns_global_i(gi)) continue;

        const int ii = decomp_.local_from_global(gi) + state.Q.ng;
        project_all_ell_components(state.Q, ii, m.n, 0, q_cos, q_sin);
        project_all_ell_components(state.P, ii, m.n, 0, p_cos, p_sin);

        const int idx = mode_index(m.ell, m.n);
        local[4 * idx + 0] = q_cos[m.ell];
        local[4 * idx + 1] = (p_.modal_include_sine && m.n > 0) ? q_sin[m.ell] : 0.0;
        local[4 * idx + 2] = p_cos[m.ell];
        local[4 * idx + 3] = (p_.modal_include_sine && m.n > 0) ? p_sin[m.ell] : 0.0;
    }

    MPI_Allreduce(local.data(), global.data(), static_cast<int>(global.size()), MPI_DOUBLE, MPI_SUM, decomp_.comm);

    std::vector<ModeCoefficients> coeffs(modes_.size());
    for (std::size_t i = 0; i < modes_.size(); ++i) {
        coeffs[i].phi_cos = global[4 * i + 0];
        coeffs[i].phi_sin = global[4 * i + 1];
        coeffs[i].P_cos = global[4 * i + 2];
        coeffs[i].P_sin = global[4 * i + 3];
    }
    return coeffs;
}

void Diagnostics::write(int step, double time, const State& state, double dt_since_last) {
    double total_E = 0.0;
    double trap_E = 0.0;
    double total_l2 = 0.0;
    double trap_l2 = 0.0;
    double max_phi = 0.0;
    double max_P = 0.0;
    double inner_signed = 0.0;
    double outer_signed = 0.0;

    compute_energy_and_flux(state, total_E, trap_E, total_l2, trap_l2,
                            max_phi, max_P, inner_signed, outer_signed);

    const double inner_pos = std::max(inner_signed, 0.0);
    const double outer_pos = std::max(-outer_signed, 0.0);
    if (have_prev_flux_ && dt_since_last > 0.0) {
        cum_inner_flux_ += 0.5 * (prev_inner_pos_flux_ + inner_pos) * dt_since_last;
        cum_outer_flux_ += 0.5 * (prev_outer_pos_flux_ + outer_pos) * dt_since_last;
    }
    prev_inner_pos_flux_ = inner_pos;
    prev_outer_pos_flux_ = outer_pos;
    have_prev_flux_ = true;

    const std::vector<double> spec = compute_modal_spectrum(state, false);
    const std::vector<double> hspec = compute_modal_spectrum(state, true);
    const std::vector<ModeCoefficients> coeffs = compute_modal_well_coefficients(state);

    double Hs = 0.0;
    double denom = 0.0;
    double ellnum = 0.0;
    double nnum = 0.0;
    double high = 0.0;
    for (std::size_t i = 0; i < modes_.size(); ++i) {
        const ModeInfo& m = modes_[i];
        if (!m.trapped) continue;
        const double weight = 1.0 + static_cast<double>(m.ell * (m.ell + 1))
                            + m.kappa * m.kappa * p_.rh * p_.rh;
        Hs += std::pow(weight, p_.sobolev_s) * spec[i];
        denom += spec[i];
        ellnum += static_cast<double>(m.ell) * spec[i];
        nnum += static_cast<double>(m.n) * spec[i];
        if (m.ell >= p_.total_high_ell_cut || m.n >= p_.total_high_n_cut) high += spec[i];
    }

    const double ell_cent = (denom > 0.0) ? ellnum / denom : 0.0;
    const double n_cent = (denom > 0.0) ? nnum / denom : 0.0;
    const double high_frac = (denom > 0.0) ? high / denom : 0.0;

    if (decomp_.rank == 0) {
        diag_file_ << std::setprecision(17)
                   << step << "," << time << "," << total_l2 << "," << trap_l2
                   << "," << total_E << "," << trap_E << "," << max_phi << "," << max_P
                   << "," << Hs << "," << ell_cent << "," << n_cent << "," << high_frac << "\n";

        flux_file_ << std::setprecision(17)
                   << step << "," << time << "," << inner_signed << "," << outer_signed
                   << "," << inner_pos << "," << outer_pos
                   << "," << cum_inner_flux_ << "," << cum_outer_flux_ << "\n";

        modal_file_ << std::setprecision(17) << step << "," << time;
        for (double v : spec) modal_file_ << "," << v;
        modal_file_ << "\n";

        modal_well_file_ << std::setprecision(17) << step << "," << time;
        for (const auto& c : coeffs) {
            const double phi_abs = std::sqrt(c.phi_cos * c.phi_cos + c.phi_sin * c.phi_sin);
            const double P_abs = std::sqrt(c.P_cos * c.P_cos + c.P_sin * c.P_sin);
            modal_well_file_ << "," << phi_abs << "," << c.phi_cos << "," << c.phi_sin
                             << "," << P_abs << "," << c.P_cos << "," << c.P_sin;
        }
        modal_well_file_ << "\n";

        horizon_modal_file_ << std::setprecision(17) << step << "," << time;
        for (double v : hspec) horizon_modal_file_ << "," << v;
        horizon_modal_file_ << "\n";

        diag_file_.flush();
        flux_file_.flush();
        modal_file_.flush();
        modal_well_file_.flush();
        horizon_modal_file_.flush();
    }
}

void Diagnostics::write_fields(int step, double time, const State& state) {
    switch (p_.field_mode) {
        case FieldOutputMode::None:
            return;
        case FieldOutputMode::Slices:
            write_field_slices(step, time, state);
            return;
        case FieldOutputMode::FullCSV:
            write_full_csv(step, time, state);
            return;
        case FieldOutputMode::FullBinary:
            write_full_binary(step, time, state);
            return;
        case FieldOutputMode::SlicesFullBinary:
            write_field_slices(step, time, state);
            write_full_binary(step, time, state);
            return;
    }
}

void Diagnostics::write_field_slices(int step, double time, const State& state) {
    const int j_eq = p_.nth / 2;
    const int k0 = 0;
    const std::string base = p_.outdir + "/fields/slices_" + fmt_step(step);
    const std::string f_eq = base + "_theta_equator.csv";
    const std::string f_z0 = base + "_z0.csv";

    for (int rnk = 0; rnk < decomp_.size; ++rnk) {
        MPI_Barrier(decomp_.comm);
        if (decomp_.rank != rnk) continue;

        {
            std::ofstream out;
            if (rnk == 0) {
                out.open(f_eq);
                out << "time,x_index,x,r,z_index,z,phi,P\n";
            } else {
                out.open(f_eq, std::ios::app);
            }
            out << std::setprecision(17);
            for (int il = 0; il < decomp_.local_nx; ++il) {
                const int gi = decomp_.i_start + il;
                const int ii = il + state.Q.ng;
                for (int k = 0; k < p_.nz; ++k) {
                    out << time << "," << gi << "," << grid_.x[gi] << "," << grid_.r[gi]
                        << "," << k << "," << grid_.z[k]
                        << "," << state.Q(ii, j_eq, k) << "," << state.P(ii, j_eq, k) << "\n";
                }
            }
        }

        {
            std::ofstream out;
            if (rnk == 0) {
                out.open(f_z0);
                out << "time,x_index,x,r,theta_index,theta,phi,P\n";
            } else {
                out.open(f_z0, std::ios::app);
            }
            out << std::setprecision(17);
            for (int il = 0; il < decomp_.local_nx; ++il) {
                const int gi = decomp_.i_start + il;
                const int ii = il + state.Q.ng;
                for (int j = 0; j < p_.nth; ++j) {
                    out << time << "," << gi << "," << grid_.x[gi] << "," << grid_.r[gi]
                        << "," << j << "," << grid_.theta[j]
                        << "," << state.Q(ii, j, k0) << "," << state.P(ii, j, k0) << "\n";
                }
            }
        }
    }
    MPI_Barrier(decomp_.comm);
}

void Diagnostics::write_full_csv(int step, double time, const State& state) {
    const std::string path = p_.outdir + "/fields/full_" + fmt_step(step) + ".csv";
    for (int rnk = 0; rnk < decomp_.size; ++rnk) {
        MPI_Barrier(decomp_.comm);
        if (decomp_.rank != rnk) continue;

        std::ofstream out;
        if (rnk == 0) {
            out.open(path);
            if (p_.field_vars == FieldOutputVars::Phi) {
                out << "time,i,x,r,j,theta,k,z,phi\n";
            } else {
                out << "time,i,x,r,j,theta,k,z,phi,P\n";
            }
        } else {
            out.open(path, std::ios::app);
        }
        out << std::setprecision(17);

        for (int il = 0; il < decomp_.local_nx; ++il) {
            const int gi = decomp_.i_start + il;
            const int ii = il + state.Q.ng;
            for (int j = 0; j < p_.nth; ++j) {
                for (int k = 0; k < p_.nz; ++k) {
                    out << time << "," << gi << "," << grid_.x[gi] << "," << grid_.r[gi]
                        << "," << j << "," << grid_.theta[j]
                        << "," << k << "," << grid_.z[k]
                        << "," << state.Q(ii, j, k);
                    if (p_.field_vars == FieldOutputVars::PhiP) {
                        out << "," << state.P(ii, j, k);
                    }
                    out << "\n";
                }
            }
        }
    }
    MPI_Barrier(decomp_.comm);
}

void Diagnostics::write_full_binary(int step, double time, const State& state) {
    const int nvars = (p_.field_vars == FieldOutputVars::PhiP) ? 2 : 1;
    const std::size_t local_points = static_cast<std::size_t>(decomp_.local_nx) * p_.nth * p_.nz;
    std::vector<double> buffer(local_points * static_cast<std::size_t>(nvars));

    std::size_t q = 0;
    for (int il = 0; il < decomp_.local_nx; ++il) {
        const int ii = il + state.Q.ng;
        for (int j = 0; j < p_.nth; ++j) {
            for (int k = 0; k < p_.nz; ++k) {
                if (nvars == 1) {
                    buffer[q++] = state.Q(ii, j, k);
                } else {
                    buffer[q++] = state.Q(ii, j, k);
                    buffer[q++] = state.P(ii, j, k);
                }
            }
        }
    }

    const std::string basename = "full_" + fmt_step(step) + ".bin";
    const std::string path = p_.outdir + "/fields/" + basename;

    MPI_File fh;
    MPI_File_open(decomp_.comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    const MPI_Offset total_bytes = static_cast<MPI_Offset>(sizeof(double))
                                 * static_cast<MPI_Offset>(p_.nx)
                                 * static_cast<MPI_Offset>(p_.nth)
                                 * static_cast<MPI_Offset>(p_.nz)
                                 * static_cast<MPI_Offset>(nvars);
    MPI_File_set_size(fh, total_bytes);

    MPI_Offset offset = static_cast<MPI_Offset>(sizeof(double))
                      * static_cast<MPI_Offset>(decomp_.i_start)
                      * static_cast<MPI_Offset>(p_.nth)
                      * static_cast<MPI_Offset>(p_.nz)
                      * static_cast<MPI_Offset>(nvars);

    std::size_t remaining = buffer.size();
    double* ptr = buffer.data();
    while (remaining > 0) {
        const int chunk = static_cast<int>(std::min<std::size_t>(remaining, 1000000000ULL));
        MPI_File_write_at(fh, offset, ptr, chunk, MPI_DOUBLE, MPI_STATUS_IGNORE);
        offset += static_cast<MPI_Offset>(chunk) * static_cast<MPI_Offset>(sizeof(double));
        ptr += chunk;
        remaining -= static_cast<std::size_t>(chunk);
    }

    MPI_File_close(&fh);

    if (decomp_.rank == 0) {
        const std::string meta = p_.outdir + "/fields/full_" + fmt_step(step) + ".txt";
        std::ofstream f(meta);
        f << std::setprecision(17);
        f << "file = " << basename << "\n";
        f << "step = " << step << "\n";
        f << "time = " << time << "\n";
        f << "format = raw_double_little_endian\n";
        f << "layout = i_j_k_var_with_k_fastest\n";
        f << "nx = " << p_.nx << "\n";
        f << "nth = " << p_.nth << "\n";
        f << "nz = " << p_.nz << "\n";
        f << "nvars = " << nvars << "\n";
        f << "vars = " << to_string(p_.field_vars) << "\n";
        f << "shape = " << p_.nx << " " << p_.nth << " " << p_.nz << " " << nvars << "\n";
        f << "index_formula = (((i*nth + j)*nz + k)*nvars + var)\n";
        f << "var0 = phi\n";
        if (nvars == 2) f << "var1 = P\n";
    }
}
