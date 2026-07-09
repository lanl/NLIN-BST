# Initial-data solver for the perturbed 5D black string

## Build

With MPI:

```bash
mpicxx -O3 -std=c++17 -o black_string_id_mpi black_string_id_mpi.cpp
```

With the included Makefile:

```bash
make CXX=mpicxx
```

For a serial build on a machine without MPI:

```bash
make serial
```

The source has a serial MPI shim, so `g++` builds are useful for debugging. For production runs, compile with an MPI compiler.

## Example runs

Small serial check:

```bash
./black_string_id_mpi --Nx 61 --Nz 24 --tol 1e-5 --out id_L1p4Lc.bin --csv
```

MPI run:

```bash
mpirun -np 8 ./black_string_id_mpi --Nx 200 --Nz 64 --L 20.02 --tol 1e-5 --out id_L1p4Lc.bin
```

Background validation:

```bash
./black_string_id_mpi --A 0 --Nx 81 --Nz 16 --tol 1e-10 --out background.bin
```

`--A 0` should converge to the unperturbed ingoing Eddington-Finkelstein black string in the solver's discrete equations.

## Reader example

`read_bsid_example.cpp` is a minimal C++ reader that shows how to import the binary file into another code:

```bash
make reader
./read_bsid_example id_L1p4Lc.bin
```
