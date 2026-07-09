#pragma once

#include <mpi.h>
#include <string>

constexpr double PI_CONST = 3.141592653589793238462643383279502884;

void ensure_dir(const std::string& path, MPI_Comm comm);
void write_text_file_rank0(const std::string& path, const std::string& text, int rank);
std::string fmt_step(int step);

// Real normalized axisymmetric spherical harmonic Y_{l0}(theta).
double legendre_P(int l, double x);
double Y_l0(int l, double theta);

// Radial profiles.
double bump8(double rho);
double cinf_bump(double rho);
double gaussian_profile(double rho);
