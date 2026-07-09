#pragma once

#include "config.hpp"
#include "decomp.hpp"
#include "diagnostics.hpp"
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

    bool need_cubic_ = false;
    bool need_derivative_source_ = false;
    bool need_phix_source_ = false;
    bool need_theta_z_source_ = false;

    void precompute_local_coefficients();
    double sponge_sigma(double x) const;
    void rhs(State& y, State& dy);
    void rk4_step(State& y, double dt);
};
