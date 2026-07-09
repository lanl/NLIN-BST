#include "grid.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

Grid::Grid(const Params& params) : p(params) {
    dx = (p.x_max - p.x_min) / static_cast<double>(p.nx - 1);
    dtheta = PI_CONST / static_cast<double>(p.nth - 1);
    dz = p.Lz / static_cast<double>(p.nz);
    inv_dx = 1.0 / dx;
    inv_dx2 = inv_dx * inv_dx;
    inv_dtheta = 1.0 / dtheta;
    inv_dtheta2 = inv_dtheta * inv_dtheta;
    inv_dz = 1.0 / dz;
    inv_dz2 = inv_dz * inv_dz;

    x.resize(p.nx);
    r.resize(p.nx);
    f.resize(p.nx);
    for (int i = 0; i < p.nx; ++i) {
        x[i] = p.x_min + static_cast<double>(i) * dx;
        r[i] = r_from_x(x[i]);
        f[i] = f_of_r(r[i]);
    }

    theta.resize(p.nth);
    sin_theta.resize(p.nth);
    cot_theta.resize(p.nth);
    for (int j = 0; j < p.nth; ++j) {
        theta[j] = static_cast<double>(j) * dtheta;
        sin_theta[j] = std::sin(theta[j]);
        if (j == 0 || j == p.nth - 1) {
            cot_theta[j] = 0.0;
        } else {
            cot_theta[j] = std::cos(theta[j]) / sin_theta[j];
        }
    }

    z.resize(p.nz);
    for (int k = 0; k < p.nz; ++k) {
        z[k] = static_cast<double>(k) * dz;
    }
}

double Grid::x_from_r(double rv) const {
    return rv + p.rh * std::log(rv / p.rh - 1.0);
}

double Grid::r_from_x(double xv) const {
    // Let y = r/rh - 1 and a = x/rh - 1.  Then y + log(y) = a.
    // Write y = exp(s), giving exp(s) + s = a.
    const double a = xv / p.rh - 1.0;
    double s = 0.0;
    if (a < -2.0) {
        s = a;
    } else if (a < 2.0) {
        s = 0.0;
    } else {
        s = std::log(a);
    }

    for (int it = 0; it < 50; ++it) {
        const double es = std::exp(s);
        const double F = es + s - a;
        const double dF = es + 1.0;
        const double ds = F / dF;
        s -= ds;
        if (std::abs(ds) < 1e-14) break;
    }

    return p.rh * (1.0 + std::exp(s));
}

int Grid::nearest_x_index(double xv) const {
    int idx = static_cast<int>(std::llround((xv - p.x_min) / dx));
    return std::max(0, std::min(p.nx - 1, idx));
}

double kappa_n(double Lz, int n) {
    return 2.0 * PI_CONST * static_cast<double>(n) / Lz;
}

double scalar_potential_V(double r, double rh, double Lz, int ell, int n) {
    const double f = 1.0 - rh / r;
    const double ell_term = static_cast<double>(ell * (ell + 1));
    const double kap = kappa_n(Lz, n);
    return f * (ell_term / (r * r) + kap * kap + rh / (r * r * r));
}

double scalar_potential_dVdr(double r, double rh, double Lz, int ell, int n) {
    const double A = static_cast<double>(ell * (ell + 1));
    const double B = kappa_n(Lz, n) * kappa_n(Lz, n);
    const double f = 1.0 - rh / r;
    const double h = A / (r * r) + B + rh / (r * r * r);
    const double fp = rh / (r * r);
    const double hp = -2.0 * A / (r * r * r) - 3.0 * rh / (r * r * r * r);
    return fp * h + f * hp;
}

namespace {

double bisection_root_dV(double a, double b, double rh, double Lz, int ell, int n) {
    double fa = scalar_potential_dVdr(a, rh, Lz, ell, n);
    double fb = scalar_potential_dVdr(b, rh, Lz, ell, n);
    if (fa == 0.0) return a;
    if (fb == 0.0) return b;

    for (int it = 0; it < 100; ++it) {
        const double m = 0.5 * (a + b);
        const double fm = scalar_potential_dVdr(m, rh, Lz, ell, n);
        if (std::abs(fm) < 1e-14 || std::abs(b - a) < 1e-13 * std::max(1.0, m)) {
            return m;
        }
        if ((fa < 0.0 && fm > 0.0) || (fa > 0.0 && fm < 0.0)) {
            b = m;
            fb = fm;
        } else {
            a = m;
            fa = fm;
        }
    }
    (void)fb;
    return 0.5 * (a + b);
}

double second_derivative_V_numeric(double r, double rh, double Lz, int ell, int n) {
    const double h = 1e-5 * std::max(1.0, r);
    return (scalar_potential_V(r + h, rh, Lz, ell, n)
          - 2.0 * scalar_potential_V(r, rh, Lz, ell, n)
          + scalar_potential_V(r - h, rh, Lz, ell, n)) / (h * h);
}

} // namespace

bool find_stable_well(double rh, double Lz, int ell, int n,
                      double r_lo, double r_hi,
                      double& r_min, double& x_min) {
    if (ell <= 0 || n <= 0) return false;

    const double ell_term = static_cast<double>(ell * (ell + 1));
    const double kap = kappa_n(Lz, n);
    const double q = kap * kap * rh * rh / ell_term;
    if (q >= 1.0 / 3.0) return false;

    r_lo = std::max(r_lo, rh * (1.0 + 1e-10));
    if (!(r_hi > r_lo)) return false;

    // Use logarithmic scanning because the two extrema can be separated by a
    // wide radial interval for small kappa.
    const int N = 4096;
    double best_r = -1.0;
    double best_V = std::numeric_limits<double>::infinity();

    double prev_r = r_lo;
    double prev_d = scalar_potential_dVdr(prev_r, rh, Lz, ell, n);

    for (int i = 1; i <= N; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(N);
        const double cur_r = r_lo * std::pow(r_hi / r_lo, t);
        const double cur_d = scalar_potential_dVdr(cur_r, rh, Lz, ell, n);

        const bool sign_change = (prev_d == 0.0) || (cur_d == 0.0) ||
                                 ((prev_d < 0.0 && cur_d > 0.0) || (prev_d > 0.0 && cur_d < 0.0));
        if (sign_change) {
            const double root = bisection_root_dV(prev_r, cur_r, rh, Lz, ell, n);
            const double d2 = second_derivative_V_numeric(root, rh, Lz, ell, n);
            if (d2 > 0.0) {
                const double V = scalar_potential_V(root, rh, Lz, ell, n);
                if (V < best_V) {
                    best_V = V;
                    best_r = root;
                }
            }
        }
        prev_r = cur_r;
        prev_d = cur_d;
    }

    if (best_r <= rh) return false;
    r_min = best_r;
    x_min = best_r + rh * std::log(best_r / rh - 1.0);
    return true;
}
