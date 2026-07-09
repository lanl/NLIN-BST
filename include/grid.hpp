#pragma once

#include "config.hpp"
#include <vector>

// Uniform grid in tortoise coordinate x=r_*.  The physical radius r(x) is
// obtained by numerically inverting x = r + rh log(r/rh - 1).
struct Grid {
    Params p;
    double dx = 0.0;
    double dtheta = 0.0;
    double dz = 0.0;
    double inv_dx = 0.0;
    double inv_dx2 = 0.0;
    double inv_dtheta = 0.0;
    double inv_dtheta2 = 0.0;
    double inv_dz = 0.0;
    double inv_dz2 = 0.0;

    std::vector<double> x;
    std::vector<double> r;
    std::vector<double> f;
    std::vector<double> theta;
    std::vector<double> sin_theta;
    std::vector<double> cot_theta;
    std::vector<double> z;

    explicit Grid(const Params& params);

    double r_from_x(double xv) const;
    double x_from_r(double rv) const;
    double f_of_r(double rv) const { return 1.0 - p.rh / rv; }
    int nearest_x_index(double xv) const;
};

double kappa_n(double Lz, int n);
double scalar_potential_V(double r, double rh, double Lz, int ell, int n);
double scalar_potential_dVdr(double r, double rh, double Lz, int ell, int n);

// Locate the stable local minimum of V_{ell,n}(r), when it exists.  The search
// assumes r_lo > rh and r_hi is the outer edge of the numerical domain.
bool find_stable_well(double rh, double Lz, int ell, int n,
                      double r_lo, double r_hi,
                      double& r_min, double& x_min);
