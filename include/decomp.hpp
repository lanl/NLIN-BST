#pragma once

#include "config.hpp"
#include <mpi.h>

// One-dimensional MPI decomposition in tortoise coordinate x.  Each rank owns a
// contiguous interval of global x indices and all theta,z points.
struct Decomp {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    int nx_global = 0;
    int i_start = 0;
    int local_nx = 0;
    int left = MPI_PROC_NULL;
    int right = MPI_PROC_NULL;

    Decomp() = default;
    Decomp(int nx, MPI_Comm c);

    bool owns_global_i(int gi) const {
        return gi >= i_start && gi < i_start + local_nx;
    }

    int local_from_global(int gi) const {
        return gi - i_start;
    }

    void require_min_local_nx(int min_cells) const;
};
