#pragma once

#include "field.hpp"
#include "grid.hpp"

inline int z_wrap(int k, int nz) {
    if (k < 0) return k + nz;
    if (k >= nz) return k - nz;
    return k;
}

inline double dx1_4(const Field3D& u, int i, int j, int k, double inv_dx) {
    return (-u(i + 2, j, k) + 8.0 * u(i + 1, j, k)
            - 8.0 * u(i - 1, j, k) + u(i - 2, j, k)) * (inv_dx / 12.0);
}

inline double dx2_4(const Field3D& u, int i, int j, int k, double inv_dx2) {
    return (-u(i + 2, j, k) + 16.0 * u(i + 1, j, k) - 30.0 * u(i, j, k)
            + 16.0 * u(i - 1, j, k) - u(i - 2, j, k)) * (inv_dx2 / 12.0);
}

inline double radial_lx_4(const Field3D& u, int i, int j, int k,
                          double cm2, double cm1, double c0, double cp1, double cp2) {
    return cm2 * u(i - 2, j, k) + cm1 * u(i - 1, j, k) + c0 * u(i, j, k)
         + cp1 * u(i + 1, j, k) + cp2 * u(i + 2, j, k);
}

inline double dz1_4(const Field3D& u, int i, int j, int k, double inv_dz) {
    const int nz = u.nz;
    if (k >= 2 && k + 2 < nz) {
        return (-u(i, j, k + 2) + 8.0 * u(i, j, k + 1)
                - 8.0 * u(i, j, k - 1) + u(i, j, k - 2)) * (inv_dz / 12.0);
    }
    return (-u(i, j, z_wrap(k + 2, nz)) + 8.0 * u(i, j, z_wrap(k + 1, nz))
            - 8.0 * u(i, j, z_wrap(k - 1, nz)) + u(i, j, z_wrap(k - 2, nz))) * (inv_dz / 12.0);
}

inline double dz2_4(const Field3D& u, int i, int j, int k, double inv_dz2) {
    const int nz = u.nz;
    if (k >= 2 && k + 2 < nz) {
        return (-u(i, j, k + 2) + 16.0 * u(i, j, k + 1) - 30.0 * u(i, j, k)
                + 16.0 * u(i, j, k - 1) - u(i, j, k - 2)) * (inv_dz2 / 12.0);
    }
    return (-u(i, j, z_wrap(k + 2, nz)) + 16.0 * u(i, j, z_wrap(k + 1, nz)) - 30.0 * u(i, j, k)
            + 16.0 * u(i, j, z_wrap(k - 1, nz)) - u(i, j, z_wrap(k - 2, nz))) * (inv_dz2 / 12.0);
}

inline double dtheta1_mixed(const Field3D& u, int i, int j, int k, double inv_dth) {
    const int N = u.nth;
    if (j == 0 || j == N - 1) return 0.0;
    if (j >= 2 && j <= N - 3) {
        return (-u(i, j + 2, k) + 8.0 * u(i, j + 1, k)
                - 8.0 * u(i, j - 1, k) + u(i, j - 2, k)) * (inv_dth / 12.0);
    }
    if (j == 1) return (u(i, 2, k) - u(i, 0, k)) * (0.5 * inv_dth);
    return (u(i, N - 1, k) - u(i, N - 3, k)) * (0.5 * inv_dth);
}

inline double dtheta2_mixed(const Field3D& u, int i, int j, int k, double inv_dth2) {
    const int N = u.nth;
    if (j >= 2 && j <= N - 3) {
        return (-u(i, j + 2, k) + 16.0 * u(i, j + 1, k) - 30.0 * u(i, j, k)
                + 16.0 * u(i, j - 1, k) - u(i, j - 2, k)) * (inv_dth2 / 12.0);
    }
    if (j == 0) return 2.0 * (u(i, 1, k) - u(i, 0, k)) * inv_dth2;
    if (j == N - 1) return 2.0 * (u(i, N - 2, k) - u(i, N - 1, k)) * inv_dth2;
    if (j == 1) return (u(i, 2, k) - 2.0 * u(i, 1, k) + u(i, 0, k)) * inv_dth2;
    return (u(i, N - 1, k) - 2.0 * u(i, N - 2, k) + u(i, N - 3, k)) * inv_dth2;
}

inline double theta_laplacian_axisym(const Field3D& u, int i, int j, int k, const Grid& grid) {
    const int N = u.nth;
    if (j == 0) return 4.0 * (u(i, 1, k) - u(i, 0, k)) * grid.inv_dtheta2;
    if (j == N - 1) return 4.0 * (u(i, N - 2, k) - u(i, N - 1, k)) * grid.inv_dtheta2;
    return dtheta2_mixed(u, i, j, k, grid.inv_dtheta2)
         + grid.cot_theta[j] * dtheta1_mixed(u, i, j, k, grid.inv_dtheta);
}

inline double ko6_x(const Field3D& u, int i, int j, int k, double inv_dx) {
    const double d6 = u(i - 3, j, k) - 6.0 * u(i - 2, j, k) + 15.0 * u(i - 1, j, k)
                    - 20.0 * u(i, j, k) + 15.0 * u(i + 1, j, k) - 6.0 * u(i + 2, j, k)
                    + u(i + 3, j, k);
    return d6 * (inv_dx / 64.0);
}

inline double ko6_z(const Field3D& u, int i, int j, int k, double inv_dz) {
    const int nz = u.nz;
    double d6 = 0.0;
    if (k >= 3 && k + 3 < nz) {
        d6 = u(i, j, k - 3) - 6.0 * u(i, j, k - 2) + 15.0 * u(i, j, k - 1)
           - 20.0 * u(i, j, k) + 15.0 * u(i, j, k + 1) - 6.0 * u(i, j, k + 2)
           + u(i, j, k + 3);
    } else {
        d6 = u(i, j, z_wrap(k - 3, nz)) - 6.0 * u(i, j, z_wrap(k - 2, nz))
           + 15.0 * u(i, j, z_wrap(k - 1, nz)) - 20.0 * u(i, j, k)
           + 15.0 * u(i, j, z_wrap(k + 1, nz)) - 6.0 * u(i, j, z_wrap(k + 2, nz))
           + u(i, j, z_wrap(k + 3, nz));
    }
    return d6 * (inv_dz / 64.0);
}

inline double ko6_theta(const Field3D& u, int i, int j, int k, double inv_dth) {
    const int N = u.nth;
    if (j < 3 || j > N - 4) return 0.0;
    const double d6 = u(i, j - 3, k) - 6.0 * u(i, j - 2, k) + 15.0 * u(i, j - 1, k)
                    - 20.0 * u(i, j, k) + 15.0 * u(i, j + 1, k) - 6.0 * u(i, j + 2, k)
                    + u(i, j + 3, k);
    return d6 * (inv_dth / 64.0);
}
