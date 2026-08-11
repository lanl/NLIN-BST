#pragma once

#include "config.hpp"
#include "decomp.hpp"
#include "diagnostics.hpp"
#include "energy_balance.hpp"
#include "field.hpp"
#include "grid.hpp"
#include <vector>

class Solver {
public:
    Solver(const Params& p, const Grid& grid, const Decomp& decomp);
    void run(State& state);

private:
    const Params& p_;
    const Grid& grid_;
    const Decomp& decomp_;
    Diagnostics diagnostics_;
    EnergyBalance energy_balance_;

    State k1_;
    State k2_;
    State k3_;
    State k4_;
    State tmp_;
    std::vector<double> f_loc_;
    std::vector<double> r_loc_;
    std::vector<double> f_over_r2_loc_;
    std::vector<double> sigma_loc_;
    std::vector<double> lx_cm2_;
    std::vector<double> lx_cm1_;
    std::vector<double> lx_c0_;
    std::vector<double> lx_cp1_;
    std::vector<double> lx_cp2_;

    // Additive energy-ledger data.  Existing solver variable names and public
    // declarations are unchanged.
    std::vector<double> theta_energy_weight_;
    std::vector<double> control_x_weight_loc_;
    int inner_probe_i_ = 0;
    int outer_probe_i_ = 0;
    EnergyRateLocal last_energy_rates_;

    bool need_cubic_ = false;
    bool need_derivative_source_ = false;
    bool need_phix_source_ = false;
    bool need_theta_z_source_ = false;
    void precompute_local_coefficients();
    double sponge_sigma(double x) const;
    void rhs(State& y, State& dy);
    void rk4_step(State& y, double dt);
};
