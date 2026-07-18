#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

// Composite closed fourth-order Newton-Cotes weights on a uniform grid that
// includes both endpoints.  Composite Simpson 1/3 is used when the number of
// intervals is even.  Otherwise Simpson 1/3 is applied to the leading even
// number of intervals and Simpson 3/8 to the final three intervals.
inline std::vector<double> newton_cotes4_weights(int npoints, double h) {
    if (npoints < 4) {
        throw std::runtime_error("newton_cotes4_weights requires at least four points");
    }
    if (!(h > 0.0)) {
        throw std::runtime_error("newton_cotes4_weights requires positive spacing");
    }

    const int intervals = npoints - 1;
    std::vector<double> w(static_cast<std::size_t>(npoints), 0.0);

    int simpson_intervals = intervals;
    bool use_three_eighths = false;
    if (intervals % 2 != 0) {
        simpson_intervals = intervals - 3;
        use_three_eighths = true;
    }

    if (simpson_intervals > 0) {
        for (int i = 0; i <= simpson_intervals; ++i) {
            double coefficient = 0.0;
            if (i == 0 || i == simpson_intervals) coefficient = 1.0;
            else coefficient = (i % 2 == 0) ? 2.0 : 4.0;
            w[static_cast<std::size_t>(i)] += (h / 3.0) * coefficient;
        }
    }

    if (use_three_eighths) {
        const int i0 = intervals - 3;
        w[static_cast<std::size_t>(i0)] += 3.0 * h / 8.0;
        w[static_cast<std::size_t>(i0 + 1)] += 9.0 * h / 8.0;
        w[static_cast<std::size_t>(i0 + 2)] += 9.0 * h / 8.0;
        w[static_cast<std::size_t>(i0 + 3)] += 3.0 * h / 8.0;
    }

    return w;
}

inline std::vector<double> closed_newton_cotes4_weights(int npoints, double h) {
    return newton_cotes4_weights(npoints, h);
}

// Periodic trapezoidal weights.  For smooth periodic functions this rule is at
// least fourth order and, for resolved Fourier modes, much more accurate.
inline std::vector<double> periodic_weights(int npoints, double h) {
    if (npoints < 1 || !(h > 0.0)) {
        throw std::runtime_error("invalid periodic quadrature grid");
    }
    return std::vector<double>(static_cast<std::size_t>(npoints), h);
}

// Compact C^4 shell window.  The old hard indicator introduced a non-smooth
// integrand and could destroy the formal quadrature order of trap diagnostics.
// Here rho=(x-center)/half_width and W=(1-rho^2)^5 inside the shell.
inline double compact_c4_window(double x, double center, double half_width) {
    if (!(half_width > 0.0)) return 0.0;
    const double rho = (x - center) / half_width;
    if (std::abs(rho) >= 1.0) return 0.0;
    const double a = 1.0 - rho * rho;
    return a * a * a * a * a;
}

// Five-point degree-four Lagrange interpolation on a uniform grid.  xi is the
// target coordinate measured in grid spacings from the first stencil point.
inline std::array<double, 5> lagrange5_value_weights(double xi) {
    std::array<double, 5> w{};
    for (int m = 0; m < 5; ++m) {
        double value = 1.0;
        for (int q = 0; q < 5; ++q) {
            if (q == m) continue;
            value *= (xi - static_cast<double>(q))
                     / static_cast<double>(m - q);
        }
        w[static_cast<std::size_t>(m)] = value;
    }
    return w;
}

inline std::array<double, 5> lagrange5_derivative_weights(double xi, double h) {
    std::array<double, 5> w{};
    for (int m = 0; m < 5; ++m) {
        double derivative = 0.0;
        for (int s = 0; s < 5; ++s) {
            if (s == m) continue;
            double term = 1.0 / static_cast<double>(m - s);
            for (int q = 0; q < 5; ++q) {
                if (q == m || q == s) continue;
                term *= (xi - static_cast<double>(q))
                        / static_cast<double>(m - q);
            }
            derivative += term;
        }
        w[static_cast<std::size_t>(m)] = derivative / h;
    }
    return w;
}

inline int lagrange5_stencil_start(double x,
                                   double x_min,
                                   double h,
                                   int npoints) {
    const double coordinate = (x - x_min) / h;
    const int nearest = static_cast<int>(std::llround(coordinate));
    return std::max(0, std::min(npoints - 5, nearest - 2));
}

// Fourth-order integration of an equally spaced time history.  This is retained
// for post-processing and startup checks; the production accumulated horizon
// and outer fluxes are integrated directly with the four RK stages.
inline double integrate_time_history_nc4(const std::vector<double>& times,
                                         const std::vector<double>& values) {
    if (times.size() != values.size() || times.empty()) return 0.0;
    if (times.size() == 1) return 0.0;

    const double h = times[1] - times[0];
    bool uniform = true;
    for (std::size_t i = 2; i < times.size(); ++i) {
        const double hi = times[i] - times[i - 1];
        if (std::abs(hi - h) > 1.0e-10 * std::max(1.0, std::abs(h))) {
            uniform = false;
            break;
        }
    }

    if (!uniform) {
        double integral = 0.0;
        for (std::size_t i = 1; i < times.size(); ++i) {
            integral += 0.5 * (times[i] - times[i - 1])
                        * (values[i] + values[i - 1]);
        }
        return integral;
    }

    if (times.size() == 2) return 0.5 * h * (values[0] + values[1]);
    if (times.size() == 3) {
        return (h / 3.0) * (values[0] + 4.0 * values[1] + values[2]);
    }

    const auto weights = newton_cotes4_weights(
        static_cast<int>(times.size()), h);
    double integral = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        integral += weights[i] * values[i];
    }
    return integral;
}
