#include "utils.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

void ensure_dir(const std::string& path, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        std::filesystem::create_directories(path);
    }
    MPI_Barrier(comm);
}

void write_text_file_rank0(const std::string& path, const std::string& text, int rank) {
    if (rank != 0) return;
    std::ofstream f(path);
    f << text;
}

std::string fmt_step(int step) {
    std::ostringstream s;
    s << std::setw(8) << std::setfill('0') << step;
    return s.str();
}

double legendre_P(int l, double x) {
    if (l == 0) return 1.0;
    if (l == 1) return x;

    double p0 = 1.0;
    double p1 = x;
    for (int n = 2; n <= l; ++n) {
        const double p = ((2.0 * n - 1.0) * x * p1 - (n - 1.0) * p0) / static_cast<double>(n);
        p0 = p1;
        p1 = p;
    }
    return p1;
}

double Y_l0(int l, double theta) {
    const double c = std::cos(theta);
    return std::sqrt((2.0 * l + 1.0) / (4.0 * PI_CONST)) * legendre_P(l, c);
}

double bump8(double rho) {
    const double a = 1.0 - rho * rho;
    if (a <= 0.0) return 0.0;
    const double a2 = a * a;
    const double a4 = a2 * a2;
    return a4 * a4;
}

double cinf_bump(double rho) {
    const double a = 1.0 - rho * rho;
    if (a <= 0.0) return 0.0;
    return std::exp(-1.0 / a);
}

double gaussian_profile(double rho) {
    return std::exp(-rho * rho);
}
