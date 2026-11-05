#pragma once

#include "config.hpp"
#include "decomp.hpp"
#include "field.hpp"
#include "grid.hpp"

#include <fstream>

// Local (per-MPI-rank) rates entering the control-volume energy identity.
//
// Sign convention:
//   inner_flux_out > 0 : energy leaves the exterior control volume toward
//                        decreasing x (the horizon side),
//   outer_flux_out > 0 : energy leaves toward increasing x,
//   sponge_loss    > 0 : energy is removed by -sigma P,
//   bulk_power     > 0 : the non-cubic source injects energy into the cubic
//                        reference-energy functional.
//
// The *_positive fields are one-way, visualization-oriented rates.  The
// signed rates are the authoritative quantities for the balance law.
struct EnergyRateLocal {
    double inner_flux_out = 0.0;
    double outer_flux_out = 0.0;
    double inner_flux_positive = 0.0;
    double outer_flux_positive = 0.0;
    double sponge_loss = 0.0;
    double bulk_power = 0.0;
};

// Energy bookkeeping for the fixed-background scalar evolution.
//
// Existing Diagnostics names, signatures, diagnostics.csv and fluxes.csv are
// left untouched.  This class writes four focused CSV files for the radial control volume
// bounded by horizon_probe_x and outer_probe_x.
class EnergyBalance {
public:
    EnergyBalance(const Params& p, const Grid& grid, const Decomp& decomp);

    bool enabled() const { return p_.energy_balance; }

    // Integrate local stage rates with the same RK4 weights used for the state.
    // No MPI collective is performed here; local cumulative integrals are
    // reduced only when an output row is written.
    void accumulate_rk4_local(double dt,
                              const EnergyRateLocal& r1,
                              const EnergyRateLocal& r2,
                              const EnergyRateLocal& r3,
                              const EnergyRateLocal& r4);

    // Write the current control-volume energy, instantaneous rates, cumulative
    // losses/fluxes, and corrected energy ledger.
    void write(int step, double time, const State& state);

    int inner_probe_i() const { return inner_probe_i_; }
    int outer_probe_i() const { return outer_probe_i_; }

private:
    const Params& p_;
    const Grid& grid_;
    const Decomp& decomp_;

    int inner_probe_i_ = 0;
    int outer_probe_i_ = 0;

    EnergyRateLocal cumulative_local_;
    double initial_control_energy_ = 0.0;
    bool have_initial_energy_ = false;
    bool have_previous_output_ = false;
    double previous_energy_ = 0.0;
    double previous_time_ = 0.0;

    std::ofstream components_file_;
    std::ofstream fluxes_file_;
    std::ofstream budget_file_;
    std::ofstream checks_file_;

    double sponge_sigma(double x) const;
    double compute_control_energy_local(const State& state) const;
    void compute_control_components_local(const State& state, double components[5]) const;
    EnergyRateLocal compute_rates_local(const State& state) const;
};
