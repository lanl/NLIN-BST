// Initial-data generator for the five-dimensional perturbed black string
// described in Choptuik et al., Phys. Rev. D 68, 044001 (2003),
//
// The implementation follows the Appendix-A fixed-point iteration:
//   1. Hamiltonian constraint: solve for g_rr line-by-line outward in x.
//   2. r-momentum constraint: solve for k_theta_theta inward in x.
//   3. z-momentum constraint: solve for k_rr forward in z.
// The code uses compactified coordinate x = r/(1+r), periodic z,
// angular metric variable g_theta_theta = gamma_Omega/r^2, and scaled
// extrinsic-curvature variables k_rr = r^2 K_rr/alpha and
// k_theta_theta = K_theta_theta/alpha.
//
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if __has_include(<mpi.h>)
#include <mpi.h>
#define BSID_HAVE_MPI 1
#else
#define BSID_HAVE_MPI 0
using MPI_Comm = int;
static constexpr MPI_Comm MPI_COMM_WORLD = 0;
static constexpr int MPI_DOUBLE = 0;
static constexpr int MPI_INT = 1;
static constexpr int MPI_SUM = 2;
static constexpr int MPI_MAX = 3;
inline int MPI_Init(int*, char***) { return 0; }
inline int MPI_Finalize() { return 0; }
inline int MPI_Comm_rank(MPI_Comm, int* rank) { *rank = 0; return 0; }
inline int MPI_Comm_size(MPI_Comm, int* size) { *size = 1; return 0; }
inline int MPI_Barrier(MPI_Comm) { return 0; }
inline double MPI_Wtime() { return 0.0; }
inline int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, int datatype, int op, MPI_Comm) {
    if (datatype == MPI_DOUBLE) {
        std::memcpy(recvbuf, sendbuf, static_cast<size_t>(count) * sizeof(double));
    } else if (datatype == MPI_INT) {
        std::memcpy(recvbuf, sendbuf, static_cast<size_t>(count) * sizeof(int));
    }
    (void)op;
    return 0;
}
#endif

namespace bsid {

constexpr double pi = 3.141592653589793238462643383279502884;

struct Params {
    int Nx = 161;
    int Nz = 64;
    double M = 1.0;
    // The paper uses L_c ~= 14.3 M and a representative nonlinear run L=1.4 L_c.
    double L = 1.4 * 14.3;
    double A = 0.1;
    int q = 1;
    double r0 = 2.5;
    double sigma = 0.5;
    double rmin = 1.0;
    double xmax = 0.995;
    int max_iter = 50;
    double tol = 1.0e-5;
    int newton_max = 20;
    double newton_tol = 1.0e-11;
    bool write_csv = false;
    bool quiet = false;
    std::string out = "black_string_id.bin";
};

struct Range { int begin = 0; int end = 0; };

Range split_range(int n, int rank, int size) {
    const int base = n / size;
    const int rem = n % size;
    const int begin = rank * base + std::min(rank, rem);
    const int count = base + (rank < rem ? 1 : 0);
    return {begin, begin + count};
}

void mpi_allreduce_sum(std::vector<double>& v) {
    std::vector<double> recv(v.size(), 0.0);
    MPI_Allreduce(v.data(), recv.data(), static_cast<int>(v.size()), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    v.swap(recv);
}

double mpi_allreduce_max(double x) {
    double y = 0.0;
    MPI_Allreduce(&x, &y, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return y;
}

double mpi_allreduce_sum_scalar(double x) {
    double y = 0.0;
    MPI_Allreduce(&x, &y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return y;
}

struct Background {
    static double alpha(double r, double M) {
        return std::sqrt(r / (r + 2.0 * M));
    }
    static double dlog_alpha_dr(double r, double M) {
        return M / (r * (r + 2.0 * M));
    }
    static double grr(double r, double M) {
        return 1.0 + 2.0 * M / r;
    }
    static double ktheta(double M) {
        return 2.0 * M;
    }
    static double krr(double r, double M) {
        // k_rr = r^2 K_rr/alpha for the ingoing Eddington-Finkelstein black string.
        return -2.0 * M * (r + M) / r;
    }
    static double beta_r_background(double r, double M) {
        return 2.0 * M / (r + 2.0 * M);
    }
};

struct Perturbation {
    double b = 1.0;    // g_theta_theta = gamma_Omega/r^2
    double br = 0.0;
    double brr = 0.0;
    double bz = 0.0;
    double bzz = 0.0;
};

Perturbation gtheta_and_derivs(double r, double z, const Params& p) {
    const double kk = 2.0 * pi * static_cast<double>(p.q) / p.L;
    const double y = r - p.r0;
    const double sig2 = p.sigma * p.sigma;
    const double e = std::exp(-(y * y) / sig2);
    const double s = std::sin(kk * z);
    const double c = std::cos(kk * z);
    const double amp = p.A * e;

    Perturbation out;
    out.b = 1.0 + amp * s;
    out.br = amp * s * (-2.0 * y / sig2);
    out.brr = amp * s * ((4.0 * y * y) / (sig2 * sig2) - 2.0 / sig2);
    out.bz = amp * kk * c;
    out.bzz = -amp * kk * kk * s;
    if (!(out.b > 0.0)) {
        std::ostringstream os;
        os << "g_theta_theta became non-positive at r=" << r << ", z=" << z
           << "; reduce --A or change perturbation parameters.";
        throw std::runtime_error(os.str());
    }
    return out;
}

struct PhiDerivs {
    double phi = 0.0;
    double phir = 0.0;
    double phirr = 0.0;
    double phiz = 0.0;
    double phizz = 0.0;
};

PhiDerivs phi_derivs(double r, const Perturbation& b) {
    const double u = std::sqrt(b.b);
    const double u3 = u * u * u;
    const double ur = b.br / (2.0 * u);
    const double urr = b.brr / (2.0 * u) - (b.br * b.br) / (4.0 * u3);
    const double uz = b.bz / (2.0 * u);
    const double uzz = b.bzz / (2.0 * u) - (b.bz * b.bz) / (4.0 * u3);

    PhiDerivs pd;
    pd.phi = r * u;
    pd.phir = u + r * ur;
    pd.phirr = 2.0 * ur + r * urr;
    pd.phiz = r * uz;
    pd.phizz = r * uzz;
    return pd;
}

struct ResidualNorms {
    double H_inf = 0.0;
    double Mr_inf = 0.0;
    double Mz_inf = 0.0;
    double H_l2 = 0.0;
    double Mr_l2 = 0.0;
    double Mz_l2 = 0.0;
    double max_all() const { return std::max(H_inf, std::max(Mr_inf, Mz_inf)); }
};

class Grid {
public:
    explicit Grid(Params p) : p_(std::move(p)) {
        if (p_.Nx < 3) throw std::runtime_error("Nx must be at least 3.");
        if (p_.Nz < 4) throw std::runtime_error("Nz must be at least 4.");
        if (!(p_.rmin > 0.0)) throw std::runtime_error("rmin must be positive.");
        if (!(p_.xmax > 0.0 && p_.xmax < 1.0)) throw std::runtime_error("xmax must satisfy 0 < xmax < 1.");
        if (!(p_.L > 0.0)) throw std::runtime_error("L must be positive.");
        if (!(p_.sigma > 0.0)) throw std::runtime_error("sigma must be positive.");
        xmin_ = p_.rmin / (1.0 + p_.rmin);
        if (!(p_.xmax > xmin_)) throw std::runtime_error("xmax must be larger than x(rmin).");
        dx_ = (p_.xmax - xmin_) / static_cast<double>(p_.Nx - 1);
        dz_ = p_.L / static_cast<double>(p_.Nz);
        x_.resize(p_.Nx);
        r_.resize(p_.Nx);
        z_.resize(p_.Nz);
        for (int i = 0; i < p_.Nx; ++i) {
            x_[i] = xmin_ + static_cast<double>(i) * dx_;
            r_[i] = x_[i] / (1.0 - x_[i]);
        }
        for (int j = 0; j < p_.Nz; ++j) z_[j] = static_cast<double>(j) * dz_;

        const size_t n = static_cast<size_t>(p_.Nx) * static_cast<size_t>(p_.Nz);
        grr_.assign(n, 1.0);
        ktheta_.assign(n, Background::ktheta(p_.M));
        krr_.assign(n, 0.0);
        initialize_background();
    }

    const Params& params() const { return p_; }
    int Nx() const { return p_.Nx; }
    int Nz() const { return p_.Nz; }
    double dx() const { return dx_; }
    double dz() const { return dz_; }
    double x(int i) const { return x_[i]; }
    double r(int i) const { return r_[i]; }
    double z(int j) const { return z_[j]; }
    size_t idx(int i, int j) const { return static_cast<size_t>(i) * static_cast<size_t>(p_.Nz) + static_cast<size_t>(j); }

    double& grr(int i, int j) { return grr_[idx(i,j)]; }
    double grr(int i, int j) const { return grr_[idx(i,j)]; }
    double& ktheta(int i, int j) { return ktheta_[idx(i,j)]; }
    double ktheta(int i, int j) const { return ktheta_[idx(i,j)]; }
    double& krr(int i, int j) { return krr_[idx(i,j)]; }
    double krr(int i, int j) const { return krr_[idx(i,j)]; }

    const std::vector<double>& grr_vec() const { return grr_; }
    const std::vector<double>& ktheta_vec() const { return ktheta_; }
    const std::vector<double>& krr_vec() const { return krr_; }
    std::vector<double>& grr_vec() { return grr_; }
    std::vector<double>& ktheta_vec() { return ktheta_; }
    std::vector<double>& krr_vec() { return krr_; }
    const std::vector<double>& x_vec() const { return x_; }
    const std::vector<double>& r_vec() const { return r_; }
    const std::vector<double>& z_vec() const { return z_; }

    void initialize_background() {
        for (int i = 0; i < p_.Nx; ++i) {
            const double rr = r_[i];
            for (int j = 0; j < p_.Nz; ++j) {
                grr(i,j) = Background::grr(rr, p_.M);
                ktheta(i,j) = Background::ktheta(p_.M);
                krr(i,j) = Background::krr(rr, p_.M);
            }
        }
    }

private:
    Params p_;
    double xmin_ = 0.0;
    double dx_ = 0.0;
    double dz_ = 0.0;
    std::vector<double> x_, r_, z_;
    std::vector<double> grr_, ktheta_, krr_;
};

std::vector<double> periodic_dz(const std::vector<double>& u, double dz) {
    const int n = static_cast<int>(u.size());
    std::vector<double> uz(n, 0.0);
    for (int j = 0; j < n; ++j) {
        const int jp = (j + 1) % n;
        const int jm = (j + n - 1) % n;
        uz[j] = (u[jp] - u[jm]) / (2.0 * dz);
    }
    return uz;
}

std::vector<double> periodic_dzz(const std::vector<double>& u, double dz) {
    const int n = static_cast<int>(u.size());
    std::vector<double> uzz(n, 0.0);
    for (int j = 0; j < n; ++j) {
        const int jp = (j + 1) % n;
        const int jm = (j + n - 1) % n;
        uzz[j] = (u[jp] - 2.0 * u[j] + u[jm]) / (dz * dz);
    }
    return uzz;
}

// a_r RHS from the Hamiltonian constraint H=0 for metric
// ds_4^2 = a(r,z) dr^2 + dz^2 + [r^2 b(r,z)] dOmega_2^2,
// K^r_r = alpha krr/(a r^2), K^theta_theta = alpha ktheta/(r^2 b), K^z_z = 0.
// The scalar curvature is evaluated as a warped product over the 2D base (r,z).
double hamiltonian_ar_rhs(double r, double z, double a, double az, double azz,
                          double krr, double ktheta, const Params& p) {
    const Perturbation b = gtheta_and_derivs(r, z, p);
    const PhiDerivs pd = phi_derivs(r, b);
    const double alpha = Background::alpha(r, p.M);

    const double R_base = -azz / a + (az * az) / (2.0 * a * a);
    const double grad_phi_sq = (pd.phir * pd.phir) / a + pd.phiz * pd.phiz;
    const double delta_no_ar = pd.phirr / a + pd.phizz + (az * pd.phiz) / (2.0 * a);
    const double extr = alpha * alpha * (
        4.0 * krr * ktheta / (a * b.b * std::pow(r, 4)) +
        2.0 * ktheta * ktheta / (b.b * b.b * std::pow(r, 4)));

    const double rest = R_base + 2.0 * (1.0 - grad_phi_sq) / (pd.phi * pd.phi)
                      - 4.0 * delta_no_ar / pd.phi + extr;
    const double coeff = 2.0 * pd.phir / (a * a * pd.phi);
    if (!(std::abs(coeff) > 1.0e-30)) {
        throw std::runtime_error("Hamiltonian a_r coefficient is too small.");
    }
    return -rest / coeff;
}

void hamiltonian_row_residual(const Grid& g, int i,
                              const std::vector<double>& a_prev,
                              const std::vector<double>& a_cur,
                              std::vector<double>& res) {
    const Params& p = g.params();
    const int Nz = g.Nz();
    res.assign(Nz, 0.0);
    std::vector<double> amid(Nz, 0.0);
    for (int j = 0; j < Nz; ++j) amid[j] = 0.5 * (a_prev[j] + a_cur[j]);
    const std::vector<double> az = periodic_dz(amid, g.dz());
    const std::vector<double> azz = periodic_dzz(amid, g.dz());

    const double xmid = 0.5 * (g.x(i-1) + g.x(i));
    const double rmid = xmid / (1.0 - xmid);
    const double drdx = 1.0 / ((1.0 - xmid) * (1.0 - xmid));

    for (int j = 0; j < Nz; ++j) {
        const double z = g.z(j);
        const double krr_mid = 0.5 * (g.krr(i-1,j) + g.krr(i,j));
        const double ktheta_mid = 0.5 * (g.ktheta(i-1,j) + g.ktheta(i,j));
        const double ar = hamiltonian_ar_rhs(rmid, z, amid[j], az[j], azz[j], krr_mid, ktheta_mid, p);
        const double ax_rhs = drdx * ar;
        res[j] = (a_cur[j] - a_prev[j]) / g.dx() - ax_rhs;
    }
}

// Standard tridiagonal solve with a[0] unused and c[n-1] unused.
std::vector<double> solve_tridiagonal(const std::vector<double>& a,
                                      const std::vector<double>& b,
                                      const std::vector<double>& c,
                                      const std::vector<double>& rhs) {
    const int n = static_cast<int>(b.size());
    if (n < 2) throw std::runtime_error("tridiagonal system too small");
    std::vector<double> cp(n, 0.0), dp(n, 0.0), x(n, 0.0);
    double denom = b[0];
    if (std::abs(denom) < 1.0e-300) throw std::runtime_error("singular tridiagonal pivot");
    cp[0] = c[0] / denom;
    dp[0] = rhs[0] / denom;
    for (int i = 1; i < n; ++i) {
        denom = b[i] - a[i] * cp[i-1];
        if (std::abs(denom) < 1.0e-300) throw std::runtime_error("singular tridiagonal pivot");
        cp[i] = (i == n-1) ? 0.0 : c[i] / denom;
        dp[i] = (rhs[i] - a[i] * dp[i-1]) / denom;
    }
    x[n-1] = dp[n-1];
    for (int i = n-2; i >= 0; --i) x[i] = dp[i] - cp[i] * x[i+1];
    return x;
}

std::vector<double> solve_cyclic_tridiagonal(const std::vector<double>& lower,
                                             const std::vector<double>& diag,
                                             const std::vector<double>& upper,
                                             const std::vector<double>& rhs) {
    const int n = static_cast<int>(diag.size());
    if (n <= 2) throw std::runtime_error("cyclic tridiagonal system must have n>2");

    // Matrix entries:
    //   A[j,j] = diag[j]
    //   A[j,j-1] = lower[j] with lower[0] the top-right corner
    //   A[j,j+1] = upper[j] with upper[n-1] the bottom-left corner
    const double alpha = lower[0];       // A[0,n-1]
    const double beta  = upper[n-1];     // A[n-1,0]
    const double gamma = -diag[0];
    if (std::abs(gamma) < 1.0e-300) throw std::runtime_error("bad cyclic-tridiagonal gamma");

    std::vector<double> a(n, 0.0), b(diag), c(n, 0.0), u(n, 0.0);
    for (int i = 1; i < n; ++i) a[i] = lower[i];
    for (int i = 0; i < n-1; ++i) c[i] = upper[i];

    b[0] = diag[0] - gamma;
    b[n-1] = diag[n-1] - alpha * beta / gamma;

    u[0] = gamma;
    u[n-1] = alpha;

    std::vector<double> x = solve_tridiagonal(a, b, c, rhs);
    std::vector<double> z = solve_tridiagonal(a, b, c, u);

    const double denom = 1.0 + z[0] + beta * z[n-1] / gamma;
    if (std::abs(denom) < 1.0e-300) throw std::runtime_error("singular cyclic-tridiagonal correction");
    const double fact = (x[0] + beta * x[n-1] / gamma) / denom;
    for (int i = 0; i < n; ++i) x[i] -= fact * z[i];
    return x;
}

double inf_norm(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

void solve_hamiltonian_row(Grid& g, int i, int rank, int size) {
    const Params& p = g.params();
    const int Nz = g.Nz();
    std::vector<double> a_prev(Nz), a_cur(Nz), res(Nz), rhs(Nz);
    for (int j = 0; j < Nz; ++j) {
        a_prev[j] = g.grr(i-1, j);
        a_cur[j] = g.grr(i, j);
    }

    const Range col_rng = split_range(Nz, rank, size);
    double prev_norm = std::numeric_limits<double>::infinity();

    for (int it = 0; it < p.newton_max; ++it) {
        hamiltonian_row_residual(g, i, a_prev, a_cur, res);
        const double rnorm = inf_norm(res);
        if (rnorm < p.newton_tol) break;
        if (it > 0 && rnorm > 1.0e10 * prev_norm) {
            throw std::runtime_error("Hamiltonian Newton residual exploded.");
        }
        prev_norm = rnorm;

        std::vector<double> lower(Nz, 0.0), diag(Nz, 0.0), upper(Nz, 0.0);
        // Finite-difference the local cyclic-tridiagonal Jacobian. Columns are shared across MPI ranks.
        for (int c = col_rng.begin; c < col_rng.end; ++c) {
            const double saved = a_cur[c];
            const double eps = std::sqrt(std::numeric_limits<double>::epsilon()) * std::max(1.0, std::abs(saved));
            a_cur[c] = saved + eps;
            std::vector<double> resp;
            hamiltonian_row_residual(g, i, a_prev, a_cur, resp);
            a_cur[c] = saved;

            const int jm = (c + Nz - 1) % Nz;
            const int jp = (c + 1) % Nz;
            diag[c] = (resp[c] - res[c]) / eps;
            // R_{c-1} depends on variable c as its + neighbor.
            upper[jm] = (resp[jm] - res[jm]) / eps;
            // R_{c+1} depends on variable c as its - neighbor.
            lower[jp] = (resp[jp] - res[jp]) / eps;
        }
        mpi_allreduce_sum(lower);
        mpi_allreduce_sum(diag);
        mpi_allreduce_sum(upper);

        for (int j = 0; j < Nz; ++j) rhs[j] = -res[j];
        std::vector<double> delta = solve_cyclic_tridiagonal(lower, diag, upper, rhs);

        // Damped Newton step to keep g_rr positive and reduce the residual.
        double lambda = 1.0;
        bool accepted = false;
        std::vector<double> trial(Nz, 0.0), rtrial;
        for (int ls = 0; ls < 16; ++ls) {
            bool positive = true;
            for (int j = 0; j < Nz; ++j) {
                trial[j] = a_cur[j] + lambda * delta[j];
                if (!(trial[j] > 1.0e-12) || !std::isfinite(trial[j])) positive = false;
            }
            if (positive) {
                hamiltonian_row_residual(g, i, a_prev, trial, rtrial);
                if (inf_norm(rtrial) <= (1.0 - 1.0e-4 * lambda) * rnorm || inf_norm(rtrial) < p.newton_tol) {
                    accepted = true;
                    break;
                }
            }
            lambda *= 0.5;
        }
        if (!accepted) {
            // Last resort: take a very small positivity-preserving step. This is useful near strong perturbations.
            lambda = 1.0e-3;
            bool positive = true;
            for (int j = 0; j < Nz; ++j) {
                trial[j] = a_cur[j] + lambda * delta[j];
                if (!(trial[j] > 1.0e-12) || !std::isfinite(trial[j])) positive = false;
            }
            if (!positive) throw std::runtime_error("Hamiltonian Newton line search failed.");
        }
        a_cur.swap(trial);

        double dnorm = 0.0;
        for (double d : delta) dnorm = std::max(dnorm, std::abs(lambda * d));
        if (dnorm < p.newton_tol * std::max(1.0, inf_norm(a_cur))) break;
    }

    // Store the solved row on every rank.
    for (int j = 0; j < Nz; ++j) g.grr(i,j) = a_cur[j];
}

void solve_hamiltonian(Grid& g, int rank, int size) {
    // Inner boundary: black-string value, as in Appendix A.
    for (int j = 0; j < g.Nz(); ++j) g.grr(0,j) = Background::grr(g.r(0), g.params().M);
    for (int i = 1; i < g.Nx(); ++i) solve_hamiltonian_row(g, i, rank, size);
}

void solve_r_momentum(Grid& g, int rank, int size) {
    const Params& p = g.params();
    const int Nx = g.Nx();
    const int Nz = g.Nz();
    const Range zrng = split_range(Nz, rank, size);
    std::vector<double> next(static_cast<size_t>(Nx) * static_cast<size_t>(Nz), 0.0);

    for (int j = zrng.begin; j < zrng.end; ++j) {
        next[g.idx(Nx-1,j)] = Background::ktheta(p.M);
        for (int i = Nx-2; i >= 0; --i) {
            const double r0 = g.r(i);
            const double r1 = g.r(i+1);
            const double rm = 0.5 * (r0 + r1);
            const double zm = g.z(j);
            const double dr = r1 - r0;
            const double a_mid = 0.5 * (g.grr(i,j) + g.grr(i+1,j));
            const double kr_mid = 0.5 * (g.krr(i,j) + g.krr(i+1,j));
            const Perturbation b = gtheta_and_derivs(rm, zm, p);
            const PhiDerivs pd = phi_derivs(rm, b);
            const double Gamma = pd.phir / pd.phi;
            const double P = Gamma * b.b * kr_mid / a_mid;
            const double Q = Gamma + Background::dlog_alpha_dr(rm, p.M) - 2.0 / rm - b.br / b.b;
            const double knew = next[g.idx(i+1,j)];
            const double denom = 1.0 - 0.5 * dr * Q;
            if (std::abs(denom) < 1.0e-300) throw std::runtime_error("r-momentum denominator is singular.");
            next[g.idx(i,j)] = ((1.0 + 0.5 * dr * Q) * knew - dr * P) / denom;
        }
    }
    mpi_allreduce_sum(next);
    g.ktheta_vec().swap(next);
}

void solve_z_momentum(Grid& g, int rank, int size) {
    const Params& p = g.params();
    const int Nx = g.Nx();
    const int Nz = g.Nz();
    const Range xrng = split_range(Nx, rank, size);
    std::vector<double> next(static_cast<size_t>(Nx) * static_cast<size_t>(Nz), 0.0);

    for (int i = xrng.begin; i < xrng.end; ++i) {
        const double rr = g.r(i);
        next[g.idx(i,0)] = Background::krr(rr, p.M);
        for (int j = 1; j < Nz; ++j) {
            const double zmid = 0.5 * (g.z(j-1) + g.z(j));
            const double a_mid = 0.5 * (g.grr(i,j-1) + g.grr(i,j));
            const double az_mid = (g.grr(i,j) - g.grr(i,j-1)) / g.dz();
            const double kt_mid = 0.5 * (g.ktheta(i,j-1) + g.ktheta(i,j));
            const double ktz_mid = (g.ktheta(i,j) - g.ktheta(i,j-1)) / g.dz();
            const Perturbation b = gtheta_and_derivs(rr, zmid, p);
            const double P = az_mid / (2.0 * a_mid);
            const double S = -a_mid * (2.0 * ktz_mid / b.b - kt_mid * b.bz / (b.b * b.b));
            const double kprev = next[g.idx(i,j-1)];
            const double denom = 1.0 - 0.5 * g.dz() * P;
            if (std::abs(denom) < 1.0e-300) throw std::runtime_error("z-momentum denominator is singular.");
            next[g.idx(i,j)] = ((1.0 + 0.5 * g.dz() * P) * kprev + g.dz() * S) / denom;
        }
    }
    mpi_allreduce_sum(next);
    g.krr_vec().swap(next);
}

double deriv_r_xgrid(const Grid& g, const std::vector<double>& u, int i, int j) {
    const int Nx = g.Nx();
    double ux = 0.0;
    double xloc = g.x(i);
    if (i == 0) {
        ux = (-3.0 * u[g.idx(0,j)] + 4.0 * u[g.idx(1,j)] - u[g.idx(2,j)]) / (2.0 * g.dx());
    } else if (i == Nx - 1) {
        ux = (3.0 * u[g.idx(Nx-1,j)] - 4.0 * u[g.idx(Nx-2,j)] + u[g.idx(Nx-3,j)]) / (2.0 * g.dx());
    } else {
        ux = (u[g.idx(i+1,j)] - u[g.idx(i-1,j)]) / (2.0 * g.dx());
    }
    const double drdx = 1.0 / ((1.0 - xloc) * (1.0 - xloc));
    return ux / drdx;
}

double deriv_z_periodic(const Grid& g, const std::vector<double>& u, int i, int j) {
    const int Nz = g.Nz();
    const int jp = (j + 1) % Nz;
    const int jm = (j + Nz - 1) % Nz;
    return (u[g.idx(i,jp)] - u[g.idx(i,jm)]) / (2.0 * g.dz());
}

double deriv_zz_periodic(const Grid& g, const std::vector<double>& u, int i, int j) {
    const int Nz = g.Nz();
    const int jp = (j + 1) % Nz;
    const int jm = (j + Nz - 1) % Nz;
    return (u[g.idx(i,jp)] - 2.0 * u[g.idx(i,j)] + u[g.idx(i,jm)]) / (g.dz() * g.dz());
}

ResidualNorms compute_residuals(const Grid& g, int rank, int size) {
    // These are the residuals of the same second-order finite-difference equations
    // used by the Appendix-A sweeps. They are the right convergence diagnostics for
    // this solver. A continuum pointwise residual can be formed separately by using
    // deriv_r_xgrid/deriv_z_periodic if desired.
    const Params& p = g.params();
    const int Nx = g.Nx();
    const int Nz = g.Nz();

    double H_inf_local = 0.0, Mr_inf_local = 0.0, Mz_inf_local = 0.0;
    double H_l2_local = 0.0, Mr_l2_local = 0.0, Mz_l2_local = 0.0;
    long long H_count_local = 0, Mr_count_local = 0, Mz_count_local = 0;

    // Hamiltonian residuals at radial half-steps, distributed over i.
    const Range hrng = split_range(Nx - 1, rank, size); // rows i=1..Nx-1 represented by local h+1
    for (int h = hrng.begin; h < hrng.end; ++h) {
        const int i = h + 1;
        std::vector<double> a_prev(Nz), a_cur(Nz), res;
        for (int j = 0; j < Nz; ++j) {
            a_prev[j] = g.grr(i-1,j);
            a_cur[j] = g.grr(i,j);
        }
        hamiltonian_row_residual(g, i, a_prev, a_cur, res);
        for (double v : res) {
            H_inf_local = std::max(H_inf_local, std::abs(v));
            H_l2_local += v * v;
            ++H_count_local;
        }
    }

    // r-momentum residuals at radial half-steps. Distributed over z columns.
    const Range zrng = split_range(Nz, rank, size);
    for (int j = zrng.begin; j < zrng.end; ++j) {
        for (int i = 0; i < Nx - 1; ++i) {
            const double r0 = g.r(i);
            const double r1 = g.r(i+1);
            const double rm = 0.5 * (r0 + r1);
            const double dr = r1 - r0;
            const double a_mid = 0.5 * (g.grr(i,j) + g.grr(i+1,j));
            const double kr_mid = 0.5 * (g.krr(i,j) + g.krr(i+1,j));
            const Perturbation b = gtheta_and_derivs(rm, g.z(j), p);
            const PhiDerivs pd = phi_derivs(rm, b);
            const double Gamma = pd.phir / pd.phi;
            const double P = Gamma * b.b * kr_mid / a_mid;
            const double Q = Gamma + Background::dlog_alpha_dr(rm, p.M) - 2.0 / rm - b.br / b.b;
            const double kt_mid = 0.5 * (g.ktheta(i,j) + g.ktheta(i+1,j));
            const double R = (g.ktheta(i+1,j) - g.ktheta(i,j)) / dr - (P - Q * kt_mid);
            Mr_inf_local = std::max(Mr_inf_local, std::abs(R));
            Mr_l2_local += R * R;
            ++Mr_count_local;
        }
    }

    // z-momentum residuals at z half-steps used by the forward integration.
    // The interval from z_{Nz-1} to z_0 is not part of the Appendix-A initial-value
    // integration; a separate periodic-wrap diagnostic can be added if required.
    const Range xrng = split_range(Nx, rank, size);
    for (int i = xrng.begin; i < xrng.end; ++i) {
        const double rr = g.r(i);
        for (int j = 1; j < Nz; ++j) {
            const double zmid = 0.5 * (g.z(j-1) + g.z(j));
            const double a_mid = 0.5 * (g.grr(i,j-1) + g.grr(i,j));
            const double az_mid = (g.grr(i,j) - g.grr(i,j-1)) / g.dz();
            const double kt_mid = 0.5 * (g.ktheta(i,j-1) + g.ktheta(i,j));
            const double ktz_mid = (g.ktheta(i,j) - g.ktheta(i,j-1)) / g.dz();
            const double kr_mid = 0.5 * (g.krr(i,j-1) + g.krr(i,j));
            const Perturbation b = gtheta_and_derivs(rr, zmid, p);
            const double P = az_mid / (2.0 * a_mid);
            const double S = -a_mid * (2.0 * ktz_mid / b.b - kt_mid * b.bz / (b.b * b.b));
            const double R = (g.krr(i,j) - g.krr(i,j-1)) / g.dz() - (P * kr_mid + S);
            Mz_inf_local = std::max(Mz_inf_local, std::abs(R));
            Mz_l2_local += R * R;
            ++Mz_count_local;
        }
    }

    ResidualNorms n;
    n.H_inf = mpi_allreduce_max(H_inf_local);
    n.Mr_inf = mpi_allreduce_max(Mr_inf_local);
    n.Mz_inf = mpi_allreduce_max(Mz_inf_local);
    const double H_count = mpi_allreduce_sum_scalar(static_cast<double>(H_count_local));
    const double Mr_count = mpi_allreduce_sum_scalar(static_cast<double>(Mr_count_local));
    const double Mz_count = mpi_allreduce_sum_scalar(static_cast<double>(Mz_count_local));
    n.H_l2 = std::sqrt(mpi_allreduce_sum_scalar(H_l2_local) / std::max(1.0, H_count));
    n.Mr_l2 = std::sqrt(mpi_allreduce_sum_scalar(Mr_l2_local) / std::max(1.0, Mr_count));
    n.Mz_l2 = std::sqrt(mpi_allreduce_sum_scalar(Mz_l2_local) / std::max(1.0, Mz_count));
    return n;
}

double max_change(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

void solve_initial_data(Grid& g, int rank, int size) {
    const Params& p = g.params();
    for (int iter = 0; iter < p.max_iter; ++iter) {
        const std::vector<double> old_grr = g.grr_vec();
        const std::vector<double> old_kt = g.ktheta_vec();
        const std::vector<double> old_kr = g.krr_vec();

        solve_hamiltonian(g, rank, size);
        solve_r_momentum(g, rank, size);
        solve_z_momentum(g, rank, size);

        const ResidualNorms rn = compute_residuals(g, rank, size);
        const double dc = std::max(max_change(old_grr, g.grr_vec()),
                         std::max(max_change(old_kt, g.ktheta_vec()),
                                  max_change(old_kr, g.krr_vec())));
        if (rank == 0 && !p.quiet) {
            std::cout << "iter " << std::setw(4) << iter + 1
                      << "  |H|_inf=" << std::scientific << std::setprecision(6) << rn.H_inf
                      << "  |Mr|_inf=" << rn.Mr_inf
                      << "  |Mz|_inf=" << rn.Mz_inf
                      << "  change=" << dc << std::defaultfloat << "\n";
        }
        if (rn.max_all() < p.tol) {
            if (rank == 0 && !p.quiet) {
                std::cout << "Converged: max constraint residual " << std::scientific << rn.max_all()
                          << " < tol " << p.tol << std::defaultfloat << "\n";
            }
            return;
        }
    }
    if (rank == 0) {
        const ResidualNorms rn = compute_residuals(g, rank, size);
        std::cerr << "Warning: reached max iterations. Final max residual = "
                  << std::scientific << rn.max_all() << std::defaultfloat << "\n";
    }
}

void write_metadata(const Grid& g, const std::string& base) {
    const Params& p = g.params();
    std::ofstream m(base + ".meta.txt");
    if (!m) throw std::runtime_error("could not write metadata file");
    m << "black_string_id_mpi metadata\n";
    m << "paper_conventions: x=r/(1+r), z periodic, Eq.(6) scaled k variables\n";
    m << "index_order: i-major with j-fastest; flat index = i*Nz + j\n";
    m << "z_grid: [0,L) with no duplicate endpoint\n";
    m << "fields_binary_order:\n";
    m << "  x[Nx], r[Nx], z[Nz]\n";
    m << "  grr, grz, gzz, gtheta_norm, gphi_norm\n";
    m << "  krr_scaled, krz, kzz, ktheta_scaled, kphi_scaled\n";
    m << "  alpha, beta_r_freeze, beta_z\n";
    m << "metric_meaning:\n";
    m << "  gamma_rr=grr, gamma_rz=grz, gamma_zz=gzz, gamma_Omega=r^2*gtheta_norm\n";
    m << "  gamma_phiphi=r^2*gphi_norm*sin(theta)^2; gphi_norm=gtheta_norm by spherical symmetry\n";
    m << "extrinsic_meaning:\n";
    m << "  K_rr=alpha*krr_scaled/r^2, K_theta_theta=alpha*ktheta_scaled, K_zz=K_rz=0\n";
    m << "  K_phi_phi=alpha*kphi_scaled*sin(theta)^2; kphi_scaled=ktheta_scaled\n";
    m << "gauge_output:\n";
    m << "  alpha=sqrt(r/(r+2M)); beta_z=0\n";
    m << "  beta_r_freeze=2*alpha*K_theta_theta/d_r(gamma_theta_theta)\n";
    m << "params:\n";
    m << "  Nx " << p.Nx << "\n";
    m << "  Nz " << p.Nz << "\n";
    m << "  M " << std::setprecision(17) << p.M << "\n";
    m << "  L " << p.L << "\n";
    m << "  A " << p.A << "\n";
    m << "  q " << p.q << "\n";
    m << "  r0 " << p.r0 << "\n";
    m << "  sigma " << p.sigma << "\n";
    m << "  rmin " << p.rmin << "\n";
    m << "  xmax " << p.xmax << "\n";
}

template <class T>
void write_scalar(std::ofstream& out, T x) {
    out.write(reinterpret_cast<const char*>(&x), sizeof(T));
}

void write_vector(std::ofstream& out, const std::vector<double>& v) {
    out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(double)));
}

void write_binary(const Grid& g, const std::string& path) {
    const Params& p = g.params();
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("could not open binary output file");
    const char magic[16] = {'B','S','I','D','M','P','I','1','\0','\0','\0','\0','\0','\0','\0','\0'};
    out.write(magic, 16);
    int32_t Nx = p.Nx, Nz = p.Nz, q = p.q;
    write_scalar(out, Nx);
    write_scalar(out, Nz);
    write_scalar(out, q);
    write_scalar(out, p.M);
    write_scalar(out, p.L);
    write_scalar(out, p.A);
    write_scalar(out, p.r0);
    write_scalar(out, p.sigma);
    write_scalar(out, p.rmin);
    write_scalar(out, p.xmax);

    write_vector(out, g.x_vec());
    write_vector(out, g.r_vec());
    write_vector(out, g.z_vec());

    const size_t n = static_cast<size_t>(p.Nx) * static_cast<size_t>(p.Nz);
    std::vector<double> grz(n, 0.0), gzz(n, 1.0), gtheta(n), gphi(n);
    std::vector<double> krz(n, 0.0), kzz(n, 0.0), kphi(n);
    std::vector<double> alpha(n), beta_r(n), beta_z(n, 0.0);

    for (int i = 0; i < p.Nx; ++i) {
        const double rr = g.r(i);
        for (int j = 0; j < p.Nz; ++j) {
            const size_t id = g.idx(i,j);
            const Perturbation b = gtheta_and_derivs(rr, g.z(j), p);
            gtheta[id] = b.b;
            gphi[id] = b.b;
            kphi[id] = g.ktheta(i,j);
            alpha[id] = Background::alpha(rr, p.M);
            const double denom = 2.0 * rr * b.b + rr * rr * b.br; // d_r gamma_theta_theta
            beta_r[id] = 2.0 * alpha[id] * alpha[id] * g.ktheta(i,j) / denom;
        }
    }

    write_vector(out, g.grr_vec());
    write_vector(out, grz);
    write_vector(out, gzz);
    write_vector(out, gtheta);
    write_vector(out, gphi);
    write_vector(out, g.krr_vec());
    write_vector(out, krz);
    write_vector(out, kzz);
    write_vector(out, g.ktheta_vec());
    write_vector(out, kphi);
    write_vector(out, alpha);
    write_vector(out, beta_r);
    write_vector(out, beta_z);

    if (!out) throw std::runtime_error("error while writing binary output");
    write_metadata(g, path);
}

void write_csv(const Grid& g, const std::string& path) {
    const Params& p = g.params();
    std::ofstream out(path);
    if (!out) throw std::runtime_error("could not open CSV output file");
    out << std::setprecision(17);
    out << "i,j,x,r,z,grr,grz,gzz,gtheta_norm,gphi_norm,krr_scaled,krz,kzz,ktheta_scaled,kphi_scaled,alpha,beta_r_freeze,beta_z,Krr_phys,Ktheta_phys\n";
    for (int i = 0; i < p.Nx; ++i) {
        const double rr = g.r(i);
        const double aa = Background::alpha(rr, p.M);
        for (int j = 0; j < p.Nz; ++j) {
            const Perturbation b = gtheta_and_derivs(rr, g.z(j), p);
            const double denom = 2.0 * rr * b.b + rr * rr * b.br;
            const double beta_r = 2.0 * aa * aa * g.ktheta(i,j) / denom;
            const double Krr = aa * g.krr(i,j) / (rr * rr);
            const double Ktheta = aa * g.ktheta(i,j);
            out << i << ',' << j << ',' << g.x(i) << ',' << rr << ',' << g.z(j) << ','
                << g.grr(i,j) << ",0,1," << b.b << ',' << b.b << ','
                << g.krr(i,j) << ",0,0," << g.ktheta(i,j) << ',' << g.ktheta(i,j) << ','
                << aa << ',' << beta_r << ",0," << Krr << ',' << Ktheta << '\n';
        }
    }
}

void usage(std::ostream& os) {
    os << "Usage: black_string_id_mpi [options]\n"
       << "Options:\n"
       << "  --Nx N              radial compactified grid points (default 161)\n"
       << "  --Nz N              periodic z grid points (default 64)\n"
       << "  --M M               black-string mass parameter (default 1)\n"
       << "  --L L               z-period length (default 1.4*14.3)\n"
       << "  --A A               perturbation amplitude in g_theta_theta (default 0.1)\n"
       << "  --q q               integer z wave number (default 1)\n"
       << "  --r0 r0             radial center of perturbation (default 2.5)\n"
       << "  --sigma s           radial Gaussian width delta_r (default 0.5)\n"
       << "  --rmin r            inner boundary radius, usually inside horizon (default 1)\n"
       << "  --xmax x            outer compactified coordinate, < 1 (default 0.995)\n"
       << "  --tol eps           target max constraint residual (default 1e-5)\n"
       << "  --max-it N          maximum outer fixed-point iterations (default 50)\n"
       << "  --newton-max N      max Newton iterations per Hamiltonian line (default 20)\n"
       << "  --newton-tol eps    Newton row tolerance (default 1e-11)\n"
       << "  --out file.bin      binary output path (default black_string_id.bin)\n"
       << "  --csv               also write file.bin.csv\n"
       << "  --quiet             suppress per-iteration log\n"
       << "  --help              show this help\n";
}

Params parse_args(int argc, char** argv) {
    Params p;
    auto need_value = [&](int& i) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value after ") + argv[i]);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(std::cout); std::exit(0); }
        else if (arg == "--Nx") p.Nx = std::stoi(need_value(i));
        else if (arg == "--Nz") p.Nz = std::stoi(need_value(i));
        else if (arg == "--M") p.M = std::stod(need_value(i));
        else if (arg == "--L") p.L = std::stod(need_value(i));
        else if (arg == "--A") p.A = std::stod(need_value(i));
        else if (arg == "--q") p.q = std::stoi(need_value(i));
        else if (arg == "--r0") p.r0 = std::stod(need_value(i));
        else if (arg == "--sigma" || arg == "--delta-r") p.sigma = std::stod(need_value(i));
        else if (arg == "--rmin") p.rmin = std::stod(need_value(i));
        else if (arg == "--xmax") p.xmax = std::stod(need_value(i));
        else if (arg == "--tol") p.tol = std::stod(need_value(i));
        else if (arg == "--max-it") p.max_iter = std::stoi(need_value(i));
        else if (arg == "--newton-max") p.newton_max = std::stoi(need_value(i));
        else if (arg == "--newton-tol") p.newton_tol = std::stod(need_value(i));
        else if (arg == "--out") p.out = need_value(i);
        else if (arg == "--csv") p.write_csv = true;
        else if (arg == "--quiet") p.quiet = true;
        else throw std::runtime_error("unknown option: " + arg);
    }
    return p;
}

} // namespace bsid

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        bsid::Params p = bsid::parse_args(argc, argv);
        bsid::Grid grid(p);

        if (rank == 0 && !p.quiet) {
            std::cout << "black_string_id_mpi: Nx=" << p.Nx << " Nz=" << p.Nz
                      << " ranks=" << size << " M=" << p.M << " L=" << p.L
                      << " A=" << p.A << " q=" << p.q << "\n";
#if BSID_HAVE_MPI
            std::cout << "MPI: enabled\n";
#else
            std::cout << "MPI: serial shim enabled because mpi.h was not found at compile time\n";
#endif
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();
        bsid::solve_initial_data(grid, rank, size);
        MPI_Barrier(MPI_COMM_WORLD);
        const double t1 = MPI_Wtime();

        const bsid::ResidualNorms rn = bsid::compute_residuals(grid, rank, size);
        if (rank == 0) {
            if (!p.quiet) {
                std::cout << "Final residuals:\n"
                          << "  |H|_inf  = " << std::scientific << rn.H_inf << "   |H|_l2  = " << rn.H_l2 << "\n"
                          << "  |Mr|_inf = " << rn.Mr_inf << "   |Mr|_l2 = " << rn.Mr_l2 << "\n"
                          << "  |Mz|_inf = " << rn.Mz_inf << "   |Mz|_l2 = " << rn.Mz_l2 << "\n"
                          << std::defaultfloat;
                std::cout << "Elapsed seconds: " << (t1 - t0) << "\n";
            }
            bsid::write_binary(grid, p.out);
            if (p.write_csv) bsid::write_csv(grid, p.out + ".csv");
            if (!p.quiet) {
                std::cout << "Wrote " << p.out << " and " << p.out << ".meta.txt";
                if (p.write_csv) std::cout << " plus " << p.out << ".csv";
                std::cout << "\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " error: " << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
