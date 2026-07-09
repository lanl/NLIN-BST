#include "initial_data.hpp"
#include "utils.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

double radial_profile_value(const Params& p, double r, double center) {
    const double rho = (r - center) / p.width;
    switch (p.radial_profile) {
        case RadialProfileKind::Bump8:
            return bump8(rho);
        case RadialProfileKind::Gaussian:
            return gaussian_profile(rho);
        case RadialProfileKind::CInf:
            return cinf_bump(rho);
    }
    return 0.0;
}

double mode_cos(const Params& p, int n, double z) {
    return std::cos(kappa_n(p.Lz, n) * z);
}

void fill_time_symmetric(const Params& p, const Grid& grid, const Decomp& d, State& s,
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

class Eq44InitialData final : public InitialData {
public:
    std::string name() const override { return "eq44"; }

    void apply(const Params& p, const Grid& grid, const Decomp& d, State& s) const override {
        fill_time_symmetric(p, grid, d, s, [&](double r, double th, double z) {
            return p.amp * radial_profile_value(p, r, p.r_center)
                 * Y_l0(p.ell0, th) * mode_cos(p, p.n0, z);
        });
    }
};

class TwoModeInitialData final : public InitialData {
public:
    std::string name() const override { return "two_mode"; }

    void apply(const Params& p, const Grid& grid, const Decomp& d, State& s) const override {
        fill_time_symmetric(p, grid, d, s, [&](double r, double th, double z) {
            const double B = radial_profile_value(p, r, p.r_center);
            const double term1 = Y_l0(2, th) * mode_cos(p, 1, z);
            const double term2 = p.two_mode_a * Y_l0(4, th) * mode_cos(p, 2, z);
            return p.amp * B * (term1 + term2);
        });
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
        };

        std::vector<Mode> modes = {
            {2, 1, 1.0, 0.13 + p.random_phase_seed, p.r_center},
            {3, 1, 0.8, 1.71 + p.random_phase_seed, p.r_center},
            {4, 1, 0.6, 2.37 + p.random_phase_seed, p.r_center},
            {4, 2, 0.5, 0.91 + p.random_phase_seed, p.r_center},
            {5, 2, 0.4, 2.84 + p.random_phase_seed, p.r_center}
        };

        if (p.random_mode_centered) {
            for (auto& m : modes) {
                double rmin = 0.0;
                double xmin = 0.0;
                const bool ok = find_stable_well(p.rh, p.Lz, m.ell, m.n,
                                                 p.rh * (1.0 + 1e-8), grid.r.back(),
                                                 rmin, xmin);
                (void)xmin;
                if (ok) m.center = rmin;
            }
        }

        double norm = 0.0;
        for (const auto& m : modes) norm += m.amp * m.amp;
        norm = std::sqrt(norm);

        fill_time_symmetric(p, grid, d, s, [&](double r, double th, double z) {
            double sum = 0.0;
            for (const auto& m : modes) {
                const double B = radial_profile_value(p, r, m.center);
                sum += (m.amp / norm) * B * Y_l0(m.ell, th)
                     * std::cos(kappa_n(p.Lz, m.n) * z + m.phase);
            }
            return p.amp * sum;
        });
    }
};

class ZIndependentControl final : public InitialData {
public:
    std::string name() const override { return "zindep"; }

    void apply(const Params& p, const Grid& grid, const Decomp& d, State& s) const override {
        fill_time_symmetric(p, grid, d, s, [&](double r, double th, double) {
            return p.amp * radial_profile_value(p, r, p.r_center) * Y_l0(p.ell0, th);
        });
    }
};

class S2SymmetricControl final : public InitialData {
public:
    std::string name() const override { return "s2"; }

    void apply(const Params& p, const Grid& grid, const Decomp& d, State& s) const override {
        fill_time_symmetric(p, grid, d, s, [&](double r, double, double z) {
            return p.amp * radial_profile_value(p, r, p.r_center)
                 * Y_l0(0, 0.0) * mode_cos(p, p.n0, z);
        });
    }
};

class OffWellControl final : public InitialData {
public:
    std::string name() const override { return "offwell"; }

    void apply(const Params& p, const Grid& grid, const Decomp& d, State& s) const override {
        fill_time_symmetric(p, grid, d, s, [&](double r, double th, double z) {
            return p.amp * radial_profile_value(p, r, p.offwell_r_center)
                 * Y_l0(p.ell0, th) * mode_cos(p, p.n0, z);
        });
    }
};

class GaussianInitialData final : public InitialData {
public:
    std::string name() const override { return "gaussian"; }

    void apply(const Params& p, const Grid& grid, const Decomp& d, State& s) const override {
        fill_time_symmetric(p, grid, d, s, [&](double r, double th, double z) {
            const double rho = (r - p.r_center) / p.width;
            return p.amp * std::exp(-rho * rho) * Y_l0(p.ell0, th) * mode_cos(p, p.n0, z);
        });
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
