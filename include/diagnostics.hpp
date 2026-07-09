#pragma once

#include "config.hpp"
#include "decomp.hpp"
#include "field.hpp"
#include "grid.hpp"

#include <fstream>
#include <string>
#include <vector>

struct ModeInfo {
    int ell = 0;
    int n = 0;
    bool trapped = false;
    double kappa = 0.0;
    double q_trap = 0.0;
    double r_min = 0.0;
    double x_min = 0.0;
    double V_min = 0.0;
};

struct ModeCoefficients {
    double phi_cos = 0.0;
    double phi_sin = 0.0;
    double P_cos = 0.0;
    double P_sin = 0.0;
};

class Diagnostics {
public:
    Diagnostics(const Params& p, const Grid& grid, const Decomp& decomp);
    ~Diagnostics();

    void write(int step, double time, const State& state, double dt_since_last);
    void write_fields(int step, double time, const State& state);

private:
    const Params& p_;
    const Grid& grid_;
    const Decomp& decomp_;

    std::vector<ModeInfo> modes_;
    std::vector<std::vector<double>> Y_;
    std::vector<std::vector<double>> C_;
    std::vector<std::vector<double>> S_;
    std::vector<double> theta_weight_;
    std::vector<std::vector<double>> angular_gram_inv_;

    double cum_inner_flux_ = 0.0;
    double cum_outer_flux_ = 0.0;
    double prev_inner_pos_flux_ = 0.0;
    double prev_outer_pos_flux_ = 0.0;
    bool have_prev_flux_ = false;

    std::ofstream diag_file_;
    std::ofstream modal_file_;
    std::ofstream modal_well_file_;
    std::ofstream horizon_modal_file_;
    std::ofstream flux_file_;

    int mode_index(int ell, int n) const { return ell * (p_.nmax + 1) + n; }

    void open_files();
    void write_mode_info() const;
    void write_grid_files() const;
    void build_angular_gram_inverse();

    void compute_energy_and_flux(const State& state,
                                 double& total_energy,
                                 double& trap_energy,
                                 double& total_l2,
                                 double& trap_l2,
                                 double& max_phi,
                                 double& max_P,
                                 double& inner_signed_flux,
                                 double& outer_signed_flux) const;

    void project_all_ell_components(const Field3D& u,
                                    int ii,
                                    int n,
                                    int derivative_kind,
                                    std::vector<double>& cos_coeff,
                                    std::vector<double>& sin_coeff) const;

    std::vector<double> compute_modal_spectrum(const State& state, bool horizon_shell) const;
    std::vector<ModeCoefficients> compute_modal_well_coefficients(const State& state) const;

    void write_field_slices(int step, double time, const State& state);
    void write_full_csv(int step, double time, const State& state);
    void write_full_binary(int step, double time, const State& state);
};
