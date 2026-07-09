#pragma once

#include "decomp.hpp"
#include <cassert>
#include <cstddef>
#include <vector>

// 3D field on a local radial slab with x ghost cells.
// Indexing: i includes ghosts [0, local_nx+2*ng), j in theta, k in z.
class Field3D {
public:
    int ng = 3;
    int nx_loc = 0;
    int nth = 0;
    int nz = 0;
    std::vector<double> a;

    Field3D() = default;
    Field3D(int nx_local, int nth_, int nz_, int ng_ = 3) {
        resize(nx_local, nth_, nz_, ng_);
    }

    void resize(int nx_local, int nth_, int nz_, int ng_ = 3) {
        ng = ng_;
        nx_loc = nx_local;
        nth = nth_;
        nz = nz_;
        a.assign(static_cast<std::size_t>(nx_loc + 2 * ng) * nth * nz, 0.0);
    }

    int nx_tot() const { return nx_loc + 2 * ng; }
    int slice_size() const { return nth * nz; }

    std::size_t idx(int i, int j, int k) const {
        return (static_cast<std::size_t>(i) * nth + j) * nz + k;
    }

    double& operator()(int i, int j, int k) { return a[idx(i, j, k)]; }
    double operator()(int i, int j, int k) const { return a[idx(i, j, k)]; }

    void set_zero();
    void set_zero_physical();
    void combine_physical(double a0, const Field3D& x0, double a1, const Field3D& x1);
};

struct State {
    Field3D Q;
    Field3D P;

    State() = default;
    State(int nx_local, int nth, int nz, int ng = 3)
        : Q(nx_local, nth, nz, ng), P(nx_local, nth, nz, ng) {}

    void set_zero();
    void set_zero_physical();
    void combine_physical(double a0, const State& x0, double a1, const State& x1);
};

void fused_rk4_update_physical(State& y,
                               const State& k1,
                               const State& k2,
                               const State& k3,
                               const State& k4,
                               double dt);

void fill_physical_x_boundaries(Field3D& u, const Decomp& decomp);
void exchange_ghosts(Field3D& u, const Decomp& decomp);
void exchange_state_ghosts(State& s, const Decomp& decomp);
