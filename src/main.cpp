#include "config.hpp"
#include "decomp.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "initial_data.hpp"
#include "solver.hpp"
#include "utils.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    try {
        Params p = parse_args(argc, argv, comm);
        ensure_dir(p.outdir, comm);
        print_params(p, rank);

        Grid grid(p);
        Decomp decomp(p.nx, comm);
        decomp.require_min_local_nx(p.nghost);

        State state(decomp.local_nx, p.nth, p.nz, p.nghost);
        auto id = make_initial_data(p.init_kind);
        id->apply(p, grid, decomp, state);

        if (rank == 0) {
            std::string text = params_to_string(p);
            text += "init_class = " + id->name() + "\n";
            write_text_file_rank0(p.outdir + "/params.txt", text, rank);
        }

        Solver solver(p, grid, decomp);
        solver.run(state);
    } catch (const std::exception& e) {
        if (rank == 0) std::cerr << "ERROR: " << e.what() << std::endl;
        MPI_Abort(comm, 1);
    }

    MPI_Finalize();
    return 0;
}
