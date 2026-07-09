#include "field.hpp"

#include <algorithm>
#include <mpi.h>
#include <stdexcept>

void Field3D::set_zero() {
    std::fill(a.begin(), a.end(), 0.0);
}

void Field3D::set_zero_physical() {
    const int first = ng;
    const int last = ng + nx_loc;
    for (int i = first; i < last; ++i) {
        for (int j = 0; j < nth; ++j) {
            for (int k = 0; k < nz; ++k) {
                (*this)(i, j, k) = 0.0;
            }
        }
    }
}

void Field3D::combine_physical(double a0, const Field3D& x0, double a1, const Field3D& x1) {
    const int first = ng;
    const int last = ng + nx_loc;
    for (int i = first; i < last; ++i) {
        for (int j = 0; j < nth; ++j) {
            for (int k = 0; k < nz; ++k) {
                (*this)(i, j, k) = a0 * x0(i, j, k) + a1 * x1(i, j, k);
            }
        }
    }
}

void State::set_zero() {
    Q.set_zero();
    P.set_zero();
}

void State::set_zero_physical() {
    Q.set_zero_physical();
    P.set_zero_physical();
}

void State::combine_physical(double a0, const State& x0, double a1, const State& x1) {
    Q.combine_physical(a0, x0.Q, a1, x1.Q);
    P.combine_physical(a0, x0.P, a1, x1.P);
}

void fused_rk4_update_physical(State& y,
                               const State& k1,
                               const State& k2,
                               const State& k3,
                               const State& k4,
                               double dt) {
    const double c1 = dt / 6.0;
    const double c2 = dt / 3.0;
    const int ng = y.Q.ng;
    const int first = ng;
    const int last = ng + y.Q.nx_loc;
    const int nth = y.Q.nth;
    const int nz = y.Q.nz;

    for (int i = first; i < last; ++i) {
        for (int j = 0; j < nth; ++j) {
            for (int k = 0; k < nz; ++k) {
                y.Q(i, j, k) += c1 * k1.Q(i, j, k) + c2 * k2.Q(i, j, k)
                              + c2 * k3.Q(i, j, k) + c1 * k4.Q(i, j, k);
                y.P(i, j, k) += c1 * k1.P(i, j, k) + c2 * k2.P(i, j, k)
                              + c2 * k3.P(i, j, k) + c1 * k4.P(i, j, k);
            }
        }
    }
}

void fill_physical_x_boundaries(Field3D& u, const Decomp& d) {
    const int ng = u.ng;
    const int slice = u.slice_size();

    if (d.left == MPI_PROC_NULL) {
        for (int g = 0; g < ng; ++g) {
            const int dest = g;
            const int src = ng;
            std::copy_n(&u.a[static_cast<std::size_t>(src) * slice], slice,
                        &u.a[static_cast<std::size_t>(dest) * slice]);
        }
    }

    if (d.right == MPI_PROC_NULL) {
        for (int g = 0; g < ng; ++g) {
            const int dest = ng + u.nx_loc + g;
            const int src = ng + u.nx_loc - 1;
            std::copy_n(&u.a[static_cast<std::size_t>(src) * slice], slice,
                        &u.a[static_cast<std::size_t>(dest) * slice]);
        }
    }
}

void exchange_ghosts(Field3D& u, const Decomp& d) {
    const int ng = u.ng;
    if (u.nx_loc < ng) {
        throw std::runtime_error("exchange_ghosts requires local_nx >= ng");
    }

    const int slice = u.slice_size();
    const int count = ng * slice;
    MPI_Status st;

    // Left physical edge -> left neighbor; right ghost <- right neighbor.
    MPI_Sendrecv(&u.a[static_cast<std::size_t>(ng) * slice], count, MPI_DOUBLE, d.left, 101,
                 &u.a[static_cast<std::size_t>(ng + u.nx_loc) * slice], count, MPI_DOUBLE, d.right, 101,
                 d.comm, &st);

    // Right physical edge -> right neighbor; left ghost <- left neighbor.
    MPI_Sendrecv(&u.a[static_cast<std::size_t>(ng + u.nx_loc - ng) * slice], count, MPI_DOUBLE, d.right, 202,
                 &u.a[0], count, MPI_DOUBLE, d.left, 202,
                 d.comm, &st);

    fill_physical_x_boundaries(u, d);
}

void exchange_state_ghosts(State& s, const Decomp& d) {
    exchange_ghosts(s.Q, d);
    exchange_ghosts(s.P, d);
}
