// Minimal C++17 reader for files produced by black_string_id_mpi.cpp.
// Build: g++ -O2 -std=c++17 -o read_bsid_example read_bsid_example.cpp
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct BSIDData {
    int32_t Nx = 0, Nz = 0, q = 0;
    double M = 0, L = 0, A = 0, r0 = 0, sigma = 0, rmin = 0, xmax = 0;
    std::vector<double> x, r, z;
    std::vector<double> grr, grz, gzz, gtheta, gphi;
    std::vector<double> krr, krz, kzz, ktheta, kphi;
    std::vector<double> alpha, beta_r, beta_z;
    size_t idx(int i, int j) const { return static_cast<size_t>(i) * static_cast<size_t>(Nz) + static_cast<size_t>(j); }
};

template <class T>
void read_scalar(std::ifstream& in, T& x) {
    in.read(reinterpret_cast<char*>(&x), sizeof(T));
    if (!in) throw std::runtime_error("unexpected end of file");
}

void read_vec(std::ifstream& in, std::vector<double>& v, size_t n) {
    v.resize(n);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(double)));
    if (!in) throw std::runtime_error("unexpected end of file while reading vector");
}

BSIDData read_bsid(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("could not open " + path);
    char magic[16];
    in.read(magic, 16);
    if (std::string(magic, magic + 8) != "BSIDMPI1") throw std::runtime_error("not a BSIDMPI1 file");

    BSIDData d;
    read_scalar(in, d.Nx);
    read_scalar(in, d.Nz);
    read_scalar(in, d.q);
    read_scalar(in, d.M);
    read_scalar(in, d.L);
    read_scalar(in, d.A);
    read_scalar(in, d.r0);
    read_scalar(in, d.sigma);
    read_scalar(in, d.rmin);
    read_scalar(in, d.xmax);

    const size_t n1x = static_cast<size_t>(d.Nx);
    const size_t n1z = static_cast<size_t>(d.Nz);
    const size_t n2 = n1x * n1z;
    read_vec(in, d.x, n1x);
    read_vec(in, d.r, n1x);
    read_vec(in, d.z, n1z);
    read_vec(in, d.grr, n2);
    read_vec(in, d.grz, n2);
    read_vec(in, d.gzz, n2);
    read_vec(in, d.gtheta, n2);
    read_vec(in, d.gphi, n2);
    read_vec(in, d.krr, n2);
    read_vec(in, d.krz, n2);
    read_vec(in, d.kzz, n2);
    read_vec(in, d.ktheta, n2);
    read_vec(in, d.kphi, n2);
    read_vec(in, d.alpha, n2);
    read_vec(in, d.beta_r, n2);
    read_vec(in, d.beta_z, n2);
    return d;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: read_bsid_example file.bin\n";
        return 2;
    }
    try {
        BSIDData d = read_bsid(argv[1]);
        std::cout << "Read BSID file: Nx=" << d.Nx << " Nz=" << d.Nz << " M=" << d.M << " L=" << d.L << "\n";
        const int i = d.Nx / 2;
        const int j = d.Nz / 2;
        const size_t id = d.idx(i,j);
        std::cout << "Sample point i=" << i << " j=" << j << " x=" << d.x[i] << " r=" << d.r[i] << " z=" << d.z[j] << "\n";
        std::cout << "  grr=" << d.grr[id] << " gtheta=" << d.gtheta[id]
                  << " krr=" << d.krr[id] << " ktheta=" << d.ktheta[id]
                  << " alpha=" << d.alpha[id] << " beta^r=" << d.beta_r[id] << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
