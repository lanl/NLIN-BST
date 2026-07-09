#include "decomp.hpp"

#include <sstream>
#include <stdexcept>

Decomp::Decomp(int nx, MPI_Comm c) : comm(c), nx_global(nx) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int base = nx / size;
    const int rem = nx % size;
    local_nx = base + (rank < rem ? 1 : 0);
    i_start = rank * base + (rank < rem ? rank : rem);
    left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
}

void Decomp::require_min_local_nx(int min_cells) const {
    int local_ok = (local_nx >= min_cells) ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm);
    if (!global_ok) {
        std::ostringstream ss;
        ss << "too many MPI ranks for nx=" << nx_global
           << ": each rank must own at least " << min_cells
           << " x points for ghost exchange, but rank " << rank
           << " owns " << local_nx;
        throw std::runtime_error(ss.str());
    }
}
