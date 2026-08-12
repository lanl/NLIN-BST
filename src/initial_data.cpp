#include "initial_data.hpp"
#include "initial_normalization.hpp"
#include "utils.hpp"

#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

double radial_profile_value_kind(RadialProfileKind kind,
                                 double width,
                                 double r,
                                 double center) {
    const double rho = (r - center) / width;
    switch (kind) {
        case RadialProfileKind::Bump8:
            return bump8(rho);
        case RadialProfileKind::Gaussian:
            return gaussian_profile(rho);
        case RadialProfileKind::CInf:
            return cinf_bump(rho);
    }
    return 0.0;
}

double radial_profile_value(const Params& p, double r, double center) {
    return radial_profile_value_kind(p.radial_profile, p.width, r, center);
}

// Analytic dB/dr for the radial profiles used by the current repository.
// Since x=r_* and dr/dx=f, B_x=f B_r.
double radial_profile_dr_kind(RadialProfileKind kind,
                              double width,
                              double r,
                              double center) {
    const double rho = (r - center) / width;
    switch (kind) {
        case RadialProfileKind::Bump8: {
            const double a = 1.0 - rho * rho;
            if (a <= 0.0) return 0.0;
            return (-16.0 * rho / width) * std::pow(a, 7);
        }
        case RadialProfileKind::Gaussian: {
            const double B = std::exp(-rho * rho);
            return (-2.0 * rho / width) * B;
        }
        case RadialProfileKind::CInf: {
            const double a = 1.0 - rho * rho;
            if (a <= 0.0) return 0.0;
            const double B = std::exp(-1.0 / a);
            return (-2.0 * rho / (width * a * a)) * B;
        }
    }
    return 0.0;
}

double mode_cos(const Params& p, int n, double z) {
    return std::cos(kappa_n(p.Lz, n) * z);
}

// When global energy normalization is disabled, retain the repository's legacy
// convention in which --amp multiplies the field directly.  When normalization
// is enabled, first construct an amp=1 reference shape, normalize that shape,
// and apply --amp only afterward in normalize_and_scale_initial_state().
double initial_shape_amplitude(const Params& p) {
    return (p.id_normalize == IDNormalizeKind::None) ? p.amp : 1.0;
}

// Quadratic energy of the unit-amplitude separated seed
//
//   Phi = B(r) Y_{ell 0}(theta) cos(kappa_n z),  P=0.
//
// With normalized Y_{ell 0},
//
// E2 = Z_n/2 int dx [ r^2 B_x^2
//                    + f {ell(ell+1)+kappa_n^2 r^2} B^2 ],
//
// where Z_0=Lz and Z_n=Lz/2 for n>0.  This gives a black-string
// generalization of the paper's approximate 1/ell amplitude weighting.
double unit_mode_quadratic_energy(const Params& p,
                                  const Grid& grid,
                                  int ell,
                                  int n,
                                  double center,
                                  RadialProfileKind profile) {
    if (ell < 0 || n < 0) {
        throw std::runtime_error("negative ell or n in initial seed mode");
    }

    const double kappa = kappa_n(p.Lz, n);
    const double z_norm = (n == 0) ? p.Lz : 0.5 * p.Lz;
    const double angular = static_cast<double>(ell * (ell + 1));
    double integral = 0.0;

    for (int gi = 0; gi < p.nx; ++gi) {
        const double r = grid.r[gi];
        const double f = grid.f[gi];
        const double B = radial_profile_value_kind(profile, p.width, r, center);
        const double Br = radial_profile_dr_kind(profile, p.width, r, center);
        const double Bx = f * Br;
        const double wx = grid.dx * ((gi == 0 || gi == p.nx - 1) ? 0.5 : 1.0);

        const double density = 0.5 * z_norm *
            (r * r * Bx * Bx
             + f * (angular + kappa * kappa * r * r) * B * B);
        integral += wx * density;
    }
    return integral;
}

double mode_balance_factor(const Params& p,
                           const Grid& grid,
                           int ell,
                           int n,
                           double center,
                           RadialProfileKind profile) {
    switch (p.id_balance) {
        case IDBalanceKind::None:
            return 1.0;
        case IDBalanceKind::PaperL:
            if (ell <= 0) {
                throw std::runtime_error(
                    "--id-balance scale_l is undefined for ell=0; "
                    "use mode_energy or none");
            }
            return 1.0 / static_cast<double>(ell);
        case IDBalanceKind::ModeEnergy: {
            const double e2 = unit_mode_quadratic_energy(
                p, grid, ell, n, center, profile);
            if (!(e2 > 0.0) || !std::isfinite(e2)) {
                throw std::runtime_error(
                    "non-positive unit-mode quadratic energy in initial data");
            }
            return 1.0 / std::sqrt(e2);
        }
    }
    throw std::runtime_error("unknown initial-data balance kind");
}

using SeedMode = std::tuple<int, int, double, double, RadialProfileKind>;

void report_seed_modes(const Params& p,
                       const Grid& grid,
                       const Decomp& d,
                       const std::vector<SeedMode>& modes) {
    if (d.rank != 0) return;

    std::ofstream out(p.outdir + "/initial_mode_weights.txt");
    out << std::setprecision(17);
    out << "# ell n relative_coefficient balance_factor unit_E2 "
           "effective_coefficient center radial_profile\n";

    for (const auto& item : modes) {
        const int ell = std::get<0>(item);
        const int n = std::get<1>(item);
        const double relative = std::get<2>(item);
        const double center = std::get<3>(item);
        const RadialProfileKind profile = std::get<4>(item);
        const double unit_e2 = unit_mode_quadratic_energy(
            p, grid, ell, n, center, profile);
        const double balance = mode_balance_factor(
            p, grid, ell, n, center, profile);

        out << ell << " " << n << " " << relative << " " << balance << " "
            << unit_e2 << " " << relative * balance << " " << center << " "
            << to_string(profile) << "\n";
    }
}

void fill_time_symmetric(const Params& p,
                         const Grid& grid,
                         const Decomp& d,
                         State& s,
                         const std::function<double(double, double, double)>& phi_fun) {
    const int ng = s.Q.ng;
    for (int il = 0; il < d.local_nx; ++il) {
        const int gi = d.i_start + il;
        const int ii = il + ng;
        const double r = grid.r[gi];
        for (int j = 0; j < p.nth; ++j) {
            const double th = grid.theta[j];
            for (int k = 0; k < p.nz; ++k) {
                const double z = grid.z[k];
                s.Q(ii, j, k) = phi_fun(r, th, z);
                s.P(ii, j, k) = 0.0;
            }
        }
    }
    exchange_state_ghosts(s, d);
}

void finish_initial_data(const Params& p,
                         const Grid& grid,
                         const Decomp& d,
                         State& s) {
    normalize_and_scale_initial_state(p, grid, d, s);
}

class Eq44InitialData final : public InitialData {
public:
    std::string name() const override { return "eq44"; }

    void apply(const Params& p, const Grid& grid, const Decomp& d, State& s) const override {
        const double balance = mode_balance_factor(
            p, grid, p.ell0, p.n0, p.r_center, p.radial_profile);
        const double amplitude = initial_shape_amplitude(p);
        report_seed_modes(p, grid, d,
                          {{p.ell0, p.n0, 1.0, p.r_center, p.radial_profile}});

        fill_time_symmetric(p, grid, d, s, [&](double r, double th, double z) {
            return amplitude * balance * radial_profile_value(p, r, p.r_center)
                 * Y_l0(p.ell0, th) * mode_cos(p, p.n0, z);
        });
        finish_initial_data(p, grid, d, s);
    }
};

class TwoModeInitialData final : public InitialData {
public:
    std::string name() const override { return "two_mode"; }

    void apply(const Params& p, const Grid& grid, const Decomp& d, State& s) const override {
        const double balance1 = mode_balance_factor(
            p, grid, 2, 1, p.r_center, p.radial_profile);
        const double balance2 = mode_balance_factor(
            p, grid, 4, 2, p.r_center, p.radial_profile);
        const double amplitude = initial_shape_amplitude(p);

        report_seed_modes(p, grid, d,
                          {{2, 1, 1.0, p.r_center, p.radial_profile},
                           {4, 2, p.two_mode_a, p.r_center, p.radial_profile}});

        fill_time_symmetric(p, grid, d, s, [&](double r, double th, double z) {
            const double B = radial_profile_value(p, r, p.r_center);
            const double term1 = balance1 * Y_l0(2, th) * mode_cos(p, 1, z);
            const double term2 = p.two_mode_a * balance2
                               * Y_l0(4, th) * mode_cos(p, 2, z);
            return amplitude * B * (term1 + term2);
        });
        finish_initial_data(p, grid, d, s);
    }
};

class RandomTrappedInitialData final : public InitialData {
public:
    std::string name() const override { return "random_trapped"; }

    void apply(const Params& p, const Grid& grid, const Decomp& d, State& s) const override {
        struct Mode {
            int ell;
            int n;
            double amp;
            double phase;
            double center;
            double balance;
        };

        std::vector<Mode> modes = {
            {2, 1, 1.0, 0.13 + p.random_phase_seed, p.r_center, 1.0},
            {3, 1, 0.8, 1.71 + p.random_phase_seed, p.r_center, 1.0},
            {4, 1, 0.6, 2.37 + p.random_phase_seed, p.r_center, 1.0},
            {4, 2, 0.5, 0.91 + p.random_phase_seed, p.r_center, 1.0},
            {5, 2, 0.4, 2.84 + p.random_phase_seed, p.r_center, 1.0}
        };

        if (p.random_mode_centered) {
            for (auto& m : modes) {
                double rmin = 0.0;
                double xmin = 0.0;
                const bool ok = find_stable_well(
                    p.rh, p.Lz, m.ell, m.n,
                    p.rh * (1.0 + 1e-8), grid.r.back(), rmin, xmin);
                (void)xmin;
                if (ok) m.center = rmin;
            }
        }

        double coefficient_norm = 0.0;
        for (const auto& m : modes) coefficient_norm += m.amp * m.amp;
        coefficient_norm = std::sqrt(coefficient_norm);

        std::vector<SeedMode> report;
        for (auto& m : modes) {
            m.balance = mode_balance_factor(
                p, grid, m.ell, m.n, m.center, p.radial_profile);
            report.emplace_back(m.ell, m.n, m.amp / coefficient_norm,
                                m.center, p.radial_profile);
        }
        report_seed_modes(p, grid, d, report);
        const double amplitude = initial_shape_amplitude(p);

        fill_time_symmetric(p, grid, d, s, [&](double r, double th, double z) {
            double sum = 0.0;
            for (const auto& m : modes) {
                const double B = radial_profile_value(p, r, m.center);
                sum += (m.amp / coefficient_norm) * m.balance * B
                     * Y_l0(m.ell, th)
                     * std::cos(kappa_n(p.Lz, m.n) * z + m.phase);
            }
            return amplitude * sum;
        });
        finish_initial_data(p, grid, d, s);
    }
};

class ZIndependentControl final : public InitialData {
public:
    std::string name() const override { return "zindep"; }

    void apply(const Params& p, const Grid& grid, const Decomp& d, State& s) const override {
        const double balance = mode_balance_factor(
            p, grid, p.ell0, 0, p.r_center, p.radial_profile);
        const double amplitude = initial_shape_amplitude(p);
        report_seed_modes(p, grid, d,
                          {{p.ell0, 0, 1.0, p.r_center, p.radial_profile}});

        fill_time_symmetric(p, grid, d, s, [&](double r, double th, double) {
            return amplitude * balance * radial_profile_value(p, r, p.r_center)
                 * Y_l0(p.ell0, th);
        });
        finish_initial_data(p, grid, d, s);
    }
};

class S2SymmetricControl final : public InitialData {
public:
    std::string name() const override { return "s2"; }

    void apply(const Params& p, const Grid& grid, const Decomp& d, State& s) const override {
        const double balance = mode_balance_factor(
            p, grid, 0, p.n0, p.r_center, p.radial_profile);
        const double amplitude = initial_shape_amplitude(p);
        report_seed_modes(p, grid, d,
                          {{0, p.n0, 1.0, p.r_center, p.radial_profile}});

        fill_time_symmetric(p, grid, d, s, [&](double r, double, double z) {
            return amplitude * balance * radial_profile_value(p, r, p.r_center)
                 * Y_l0(0, 0.0) * mode_cos(p, p.n0, z);
        });
        finish_initial_data(p, grid, d, s);
    }
};

class OffWellControl final : public InitialData {
public:
    std::string name() const override { return "offwell"; }

    void apply(const Params& p, const Grid& grid, const Decomp& d, State& s) const override {
        const double balance = mode_balance_factor(
            p, grid, p.ell0, p.n0, p.offwell_r_center, p.radial_profile);
        const double amplitude = initial_shape_amplitude(p);
        report_seed_modes(p, grid, d,
                          {{p.ell0, p.n0, 1.0, p.offwell_r_center,
                            p.radial_profile}});

        fill_time_symmetric(p, grid, d, s, [&](double r, double th, double z) {
            return amplitude * balance
                 * radial_profile_value(p, r, p.offwell_r_center)
                 * Y_l0(p.ell0, th) * mode_cos(p, p.n0, z);
        });
        finish_initial_data(p, grid, d, s);
    }
};

class GaussianInitialData final : public InitialData {
public:
    std::string name() const override { return "gaussian"; }

    void apply(const Params& p, const Grid& grid, const Decomp& d, State& s) const override {
        const RadialProfileKind profile = RadialProfileKind::Gaussian;
        const double balance = mode_balance_factor(
            p, grid, p.ell0, p.n0, p.r_center, profile);
        const double amplitude = initial_shape_amplitude(p);
        report_seed_modes(p, grid, d,
                          {{p.ell0, p.n0, 1.0, p.r_center, profile}});

        fill_time_symmetric(p, grid, d, s, [&](double r, double th, double z) {
            return amplitude * balance
                 * radial_profile_value_kind(profile, p.width, r, p.r_center)
                 * Y_l0(p.ell0, th) * mode_cos(p, p.n0, z);
        });
        finish_initial_data(p, grid, d, s);
    }
};

} // namespace

std::unique_ptr<InitialData> make_initial_data(InitKind kind) {
    switch (kind) {
        case InitKind::Eq44:
            return std::make_unique<Eq44InitialData>();
        case InitKind::TwoMode:
            return std::make_unique<TwoModeInitialData>();
        case InitKind::RandomTrapped:
            return std::make_unique<RandomTrappedInitialData>();
        case InitKind::ZIndependent:
            return std::make_unique<ZIndependentControl>();
        case InitKind::S2Symmetric:
            return std::make_unique<S2SymmetricControl>();
        case InitKind::OffWell:
            return std::make_unique<OffWellControl>();
        case InitKind::Gaussian:
            return std::make_unique<GaussianInitialData>();
    }
    throw std::runtime_error("unknown initial data kind");
}
