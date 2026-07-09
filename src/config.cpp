#include "config.hpp"
#include "utils.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string get_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + argv[i]);
    }
    ++i;
    return std::string(argv[i]);
}

double get_double(int& i, int argc, char** argv) {
    return std::stod(get_arg(i, argc, argv));
}

int get_int(int& i, int argc, char** argv) {
    return std::stoi(get_arg(i, argc, argv));
}

QKind parse_q_kind(const std::string& s) {
    if (s == "none") return QKind::None;
    if (s == "metric") return QKind::Metric;
    if (s == "spatial") return QKind::Spatial;
    if (s == "null_x") return QKind::NullX;
    if (s == "badtime") return QKind::BadTime;
    throw std::runtime_error("unknown q-kind: " + s + " (allowed: none, metric, spatial, null_x, badtime)");
}

InitKind parse_init_kind(const std::string& s) {
    if (s == "eq44") return InitKind::Eq44;
    if (s == "two_mode") return InitKind::TwoMode;
    if (s == "random_trapped") return InitKind::RandomTrapped;
    if (s == "zindep") return InitKind::ZIndependent;
    if (s == "s2") return InitKind::S2Symmetric;
    if (s == "offwell") return InitKind::OffWell;
    if (s == "gaussian") return InitKind::Gaussian;
    throw std::runtime_error("unknown init: " + s + " (allowed: eq44, two_mode, random_trapped, zindep, s2, offwell, gaussian)");
}

RadialProfileKind parse_radial_profile(const std::string& s) {
    if (s == "bump8") return RadialProfileKind::Bump8;
    if (s == "gaussian") return RadialProfileKind::Gaussian;
    if (s == "cinf") return RadialProfileKind::CInf;
    throw std::runtime_error("unknown radial profile: " + s + " (allowed: bump8, gaussian, cinf)");
}

FieldOutputMode parse_field_mode(const std::string& s) {
    if (s == "none") return FieldOutputMode::None;
    if (s == "slices") return FieldOutputMode::Slices;
    if (s == "full_csv") return FieldOutputMode::FullCSV;
    if (s == "full_binary") return FieldOutputMode::FullBinary;
    if (s == "slices_full_binary") return FieldOutputMode::SlicesFullBinary;
    throw std::runtime_error("unknown field mode: " + s + " (allowed: none, slices, full_csv, full_binary, slices_full_binary)");
}

FieldOutputVars parse_field_vars(const std::string& s) {
    if (s == "phi") return FieldOutputVars::Phi;
    if (s == "phiP" || s == "phip" || s == "both") return FieldOutputVars::PhiP;
    throw std::runtime_error("unknown field vars: " + s + " (allowed: phi, phiP)");
}

void apply_field_output_arg(Params& p, const std::string& s) {
    if (s == "full" || s == "full_phi") {
        p.field_mode_name = "full_binary";
        p.field_vars_name = "phi";
    } else if (s == "full_state" || s == "full_phiP" || s == "full_phip") {
        p.field_mode_name = "full_binary";
        p.field_vars_name = "phiP";
    } else if (s == "slices_full_phi") {
        p.field_mode_name = "slices_full_binary";
        p.field_vars_name = "phi";
    } else if (s == "slices_full_state" || s == "slices_full_phiP" || s == "slices_full_phip") {
        p.field_mode_name = "slices_full_binary";
        p.field_vars_name = "phiP";
    } else {
        p.field_mode_name = s;
    }
}

void apply_full_output_format_arg(Params& p, const std::string& s) {
    if (s == "binary" || s == "bin" || s == "raw") {
        if (p.field_mode_name == "full_csv") p.field_mode_name = "full_binary";
    } else if (s == "csv" || s == "text") {
        if (p.field_mode_name == "full_binary") p.field_mode_name = "full_csv";
        else if (p.field_mode_name == "slices_full_binary") p.field_mode_name = "full_csv";
    } else {
        throw std::runtime_error("unknown full-output-format: " + s + " (allowed: binary, csv)");
    }
}

void finalize_and_validate(Params& p) {
    p.q_kind = parse_q_kind(p.q_kind_name);
    p.init_kind = parse_init_kind(p.init_name);
    p.radial_profile = parse_radial_profile(p.radial_profile_name);
    p.field_mode = parse_field_mode(p.field_mode_name);
    p.field_vars = parse_field_vars(p.field_vars_name);

    if (p.modal_z_basis == "cosine") {
        p.modal_include_sine = false;
    } else if (p.modal_z_basis == "cos_sin") {
        p.modal_include_sine = true;
    } else {
        throw std::runtime_error("unknown modal-z-basis: " + p.modal_z_basis + " (allowed: cosine, cos_sin)");
    }

    auto finite = [](double x) { return std::isfinite(x); };
    if (p.nx < 16 || p.nth < 8 || p.nz < 8) {
        throw std::runtime_error("grid too small: require nx>=16, nth>=8, nz>=8");
    }
    if (p.nghost < 3) {
        throw std::runtime_error("nghost must be at least 3 for fourth-order stencils and KO6");
    }
    if (!finite(p.rh) || p.rh <= 0.0) throw std::runtime_error("rh must be positive and finite");
    if (!finite(p.Lz) || p.Lz <= 0.0) throw std::runtime_error("Lz must be positive and finite");
    if (!finite(p.x_min) || !finite(p.x_max) || p.x_min >= p.x_max) {
        throw std::runtime_error("x_min and x_max must be finite with x_min < x_max");
    }
    if (!finite(p.dt) || p.dt <= 0.0) throw std::runtime_error("dt must be positive and finite");
    if (!finite(p.t_final) || p.t_final < 0.0) throw std::runtime_error("t_final must be nonnegative and finite");
    if (p.output_every <= 0) throw std::runtime_error("output_every must be positive");
    if (p.field_every < 0) throw std::runtime_error("field_every must be nonnegative");
    if (!finite(p.width) || p.width <= 0.0) throw std::runtime_error("width must be positive and finite");
    if (!finite(p.r_center) || p.r_center <= p.rh) throw std::runtime_error("r_center must be finite and greater than rh");
    if (!finite(p.offwell_r_center) || p.offwell_r_center <= p.rh) {
        throw std::runtime_error("offwell_r_center must be finite and greater than rh");
    }
    if (p.ell0 < 0 || p.n0 < 0) throw std::runtime_error("ell0 and n0 must be nonnegative");
    if (p.lmax < 0 || p.nmax < 0) throw std::runtime_error("lmax and nmax must be nonnegative");
    if (p.lmax + 1 > p.nth) {
        throw std::runtime_error("lmax is too large for nth; require lmax+1 <= nth for angular projection");
    }
    if (p.nmax >= p.nz / 2) {
        throw std::runtime_error("nmax must be below the z-grid Nyquist frequency; require nmax < nz/2");
    }
    if (p.n0 >= p.nz / 2) {
        throw std::runtime_error("n0 must be below the z-grid Nyquist frequency; require n0 < nz/2");
    }
    if (!finite(p.sponge_inner_width) || !finite(p.sponge_outer_width) ||
        !finite(p.sponge_inner_strength) || !finite(p.sponge_outer_strength)) {
        throw std::runtime_error("sponge parameters must be finite");
    }
    if (p.sponge_inner_width < 0.0 || p.sponge_outer_width < 0.0 ||
        p.sponge_inner_strength < 0.0 || p.sponge_outer_strength < 0.0) {
        throw std::runtime_error("sponge widths and strengths must be nonnegative");
    }
    if (!finite(p.ko_eps) || p.ko_eps < 0.0) throw std::runtime_error("ko_eps must be nonnegative and finite");
    if (!finite(p.trap_dx_half_width) || p.trap_dx_half_width <= 0.0) {
        throw std::runtime_error("trap_dx_half_width must be positive and finite");
    }
    if (!finite(p.horizon_probe_x) || !finite(p.outer_probe_x)) {
        throw std::runtime_error("probe locations must be finite");
    }
    if (p.horizon_probe_x < p.x_min || p.horizon_probe_x > p.x_max ||
        p.outer_probe_x < p.x_min || p.outer_probe_x > p.x_max) {
        throw std::runtime_error("probe locations must lie inside the x domain");
    }
    if (p.total_high_ell_cut < 0 || p.total_high_n_cut < 0) {
        throw std::runtime_error("high-mode cuts must be nonnegative");
    }
    if (p.field_every == 0) {
        p.field_mode_name = "none";
        p.field_mode = FieldOutputMode::None;
    }
}

} // namespace

const char* to_string(QKind v) {
    switch (v) {
        case QKind::None: return "none";
        case QKind::Metric: return "metric";
        case QKind::Spatial: return "spatial";
        case QKind::NullX: return "null_x";
        case QKind::BadTime: return "badtime";
    }
    return "unknown";
}

const char* to_string(InitKind v) {
    switch (v) {
        case InitKind::Eq44: return "eq44";
        case InitKind::TwoMode: return "two_mode";
        case InitKind::RandomTrapped: return "random_trapped";
        case InitKind::ZIndependent: return "zindep";
        case InitKind::S2Symmetric: return "s2";
        case InitKind::OffWell: return "offwell";
        case InitKind::Gaussian: return "gaussian";
    }
    return "unknown";
}

const char* to_string(RadialProfileKind v) {
    switch (v) {
        case RadialProfileKind::Bump8: return "bump8";
        case RadialProfileKind::Gaussian: return "gaussian";
        case RadialProfileKind::CInf: return "cinf";
    }
    return "unknown";
}

const char* to_string(FieldOutputMode v) {
    switch (v) {
        case FieldOutputMode::None: return "none";
        case FieldOutputMode::Slices: return "slices";
        case FieldOutputMode::FullCSV: return "full_csv";
        case FieldOutputMode::FullBinary: return "full_binary";
        case FieldOutputMode::SlicesFullBinary: return "slices_full_binary";
    }
    return "unknown";
}

const char* to_string(FieldOutputVars v) {
    switch (v) {
        case FieldOutputVars::Phi: return "phi";
        case FieldOutputVars::PhiP: return "phiP";
    }
    return "unknown";
}

void print_usage(int rank) {
    if (rank != 0) return;
    std::cout << R"USAGE(
nlin_wave_bs: nonlinear scalar wave on a fixed 5D Schwarzschild black string

Core examples:
  mpirun -np 4 ./nlin_wave_bs --outdir run_linear --mu 0 --lambda 0 --q-kind none
  mpirun -np 4 ./nlin_wave_bs --outdir run_eq44  --mu 1 --lambda 0 --init eq44
  mpirun -np 4 ./nlin_wave_bs --outdir run_full  --mu 1 --init eq44 --field-mode full_binary --field-vars phi --field-every 1

Geometry/grid/time:
  --rh R --Lz L --x-min X --x-max X
  --nx N --nth N --nz N --dt DT --t-final T
  --output-every N

Model:
  --mu MU --lambda LAM --alpha2 A
  --q-kind KIND       none, metric, spatial, null_x, badtime

Initial data:
  --init KIND         eq44, two_mode, random_trapped, zindep, s2, offwell, gaussian
  --radial-profile P  bump8, gaussian, cinf
  --amp EPS --ell0 L --n0 N --r-center R --width W
  --two-mode-a A --offwell-r-center R
  --random-mode-centered / --random-single-center

Diagnostics:
  --lmax L --nmax N --sobolev-s S
  --trap-dx-half-width X
  --high-ell-cut L --high-n-cut N
  --horizon-probe-x X --outer-probe-x X
  --modal-z-basis cosine|cos_sin

Field output:
  --field-mode none|slices|full_csv|full_binary|slices_full_binary
  --field-output none|slices|full_phi|full_state|slices_full_phi|slices_full_state
  --full-output-format binary|csv
  --field-vars phi|phiP
  --field-every N      output full/slice fields every N diagnostics; 0 disables
  --no-fields          shortcut for --field-mode none --field-every 0
  --full-phi           shortcut for --field-mode full_binary --field-vars phi
  --full-phiP          shortcut for --field-mode full_binary --field-vars phiP

Other:
  --sponge-inner-width W --sponge-outer-width W
  --sponge-inner-strength S --sponge-outer-strength S
  --ko-eps E
  --outdir DIR --quiet --help
)USAGE";
}

Params parse_args(int argc, char** argv, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    Params p;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--help" || a == "-h") {
            print_usage(rank);
            MPI_Abort(comm, 0);
        } else if (a == "--outdir") p.outdir = get_arg(i, argc, argv);
        else if (a == "--rh") p.rh = get_double(i, argc, argv);
        else if (a == "--Lz") p.Lz = get_double(i, argc, argv);
        else if (a == "--x-min") p.x_min = get_double(i, argc, argv);
        else if (a == "--x-max") p.x_max = get_double(i, argc, argv);
        else if (a == "--nx") p.nx = get_int(i, argc, argv);
        else if (a == "--nth") p.nth = get_int(i, argc, argv);
        else if (a == "--nz") p.nz = get_int(i, argc, argv);
        else if (a == "--nghost") p.nghost = get_int(i, argc, argv);
        else if (a == "--dt") p.dt = get_double(i, argc, argv);
        else if (a == "--t-final") p.t_final = get_double(i, argc, argv);
        else if (a == "--output-every") p.output_every = get_int(i, argc, argv);
        else if (a == "--field-every") p.field_every = get_int(i, argc, argv);
        else if (a == "--mu") p.mu = get_double(i, argc, argv);
        else if (a == "--lambda" || a == "--lam") p.lambda = get_double(i, argc, argv);
        else if (a == "--alpha2") p.alpha2_phi_box = get_double(i, argc, argv);
        else if (a == "--q-kind") p.q_kind_name = get_arg(i, argc, argv);
        else if (a == "--init") p.init_name = get_arg(i, argc, argv);
        else if (a == "--radial-profile") p.radial_profile_name = get_arg(i, argc, argv);
        else if (a == "--amp") p.amp = get_double(i, argc, argv);
        else if (a == "--ell0") p.ell0 = get_int(i, argc, argv);
        else if (a == "--n0") p.n0 = get_int(i, argc, argv);
        else if (a == "--r-center") p.r_center = get_double(i, argc, argv);
        else if (a == "--width") p.width = get_double(i, argc, argv);
        else if (a == "--two-mode-a") p.two_mode_a = get_double(i, argc, argv);
        else if (a == "--offwell-r-center") p.offwell_r_center = get_double(i, argc, argv);
        else if (a == "--random-phase-seed") p.random_phase_seed = get_double(i, argc, argv);
        else if (a == "--random-mode-centered") p.random_mode_centered = true;
        else if (a == "--random-single-center") p.random_mode_centered = false;
        else if (a == "--sponge-inner-width") p.sponge_inner_width = get_double(i, argc, argv);
        else if (a == "--sponge-outer-width") p.sponge_outer_width = get_double(i, argc, argv);
        else if (a == "--sponge-inner-strength") p.sponge_inner_strength = get_double(i, argc, argv);
        else if (a == "--sponge-outer-strength") p.sponge_outer_strength = get_double(i, argc, argv);
        else if (a == "--ko-eps") p.ko_eps = get_double(i, argc, argv);
        else if (a == "--lmax") p.lmax = get_int(i, argc, argv);
        else if (a == "--nmax") p.nmax = get_int(i, argc, argv);
        else if (a == "--sobolev-s") p.sobolev_s = get_int(i, argc, argv);
        else if (a == "--trap-dx-half-width") p.trap_dx_half_width = get_double(i, argc, argv);
        else if (a == "--high-ell-cut") p.total_high_ell_cut = get_int(i, argc, argv);
        else if (a == "--high-n-cut") p.total_high_n_cut = get_int(i, argc, argv);
        else if (a == "--horizon-probe-x") p.horizon_probe_x = get_double(i, argc, argv);
        else if (a == "--outer-probe-x") p.outer_probe_x = get_double(i, argc, argv);
        else if (a == "--modal-z-basis") p.modal_z_basis = get_arg(i, argc, argv);
        else if (a == "--cosine-only") p.modal_z_basis = "cosine";
        else if (a == "--field-mode" || a == "--fields" || a == "--field-output") apply_field_output_arg(p, get_arg(i, argc, argv));
        else if (a == "--field-vars") p.field_vars_name = get_arg(i, argc, argv);
        else if (a == "--full-output-format" || a == "--full-field-format") apply_full_output_format_arg(p, get_arg(i, argc, argv));
        else if (a == "--no-fields") { p.field_mode_name = "none"; p.field_every = 0; }
        else if (a == "--full-phi") { p.field_mode_name = "full_binary"; p.field_vars_name = "phi"; if (p.field_every == 0) p.field_every = 1; }
        else if (a == "--full-phiP") { p.field_mode_name = "full_binary"; p.field_vars_name = "phiP"; if (p.field_every == 0) p.field_every = 1; }
        else if (a == "--quiet") p.verbose = false;
        else throw std::runtime_error("unknown option: " + a);
    }

    finalize_and_validate(p);
    return p;
}

std::string params_to_string(const Params& p) {
    std::ostringstream ss;
    ss << std::setprecision(17);
    ss << "nlin_wave_bs parameters\n";
    ss << "equation = Box_g phi = mu phi^3 + lambda Q + alpha2 phi Box_g phi\n";
    ss << "rh = " << p.rh << "\n";
    ss << "Lz = " << p.Lz << "\n";
    ss << "x_min = " << p.x_min << "\n";
    ss << "x_max = " << p.x_max << "\n";
    ss << "nx = " << p.nx << "\n";
    ss << "nth = " << p.nth << "\n";
    ss << "nz = " << p.nz << "\n";
    ss << "nghost = " << p.nghost << "\n";
    ss << "dt = " << p.dt << "\n";
    ss << "t_final = " << p.t_final << "\n";
    ss << "output_every = " << p.output_every << "\n";
    ss << "field_every = " << p.field_every << "\n";
    ss << "mu = " << p.mu << "\n";
    ss << "lambda = " << p.lambda << "\n";
    ss << "alpha2_phi_box = " << p.alpha2_phi_box << "\n";
    ss << "q_kind = " << to_string(p.q_kind) << "\n";
    ss << "init = " << to_string(p.init_kind) << "\n";
    ss << "radial_profile = " << to_string(p.radial_profile) << "\n";
    ss << "amp = " << p.amp << "\n";
    ss << "ell0 = " << p.ell0 << "\n";
    ss << "n0 = " << p.n0 << "\n";
    ss << "r_center = " << p.r_center << "\n";
    ss << "width = " << p.width << "\n";
    ss << "two_mode_a = " << p.two_mode_a << "\n";
    ss << "offwell_r_center = " << p.offwell_r_center << "\n";
    ss << "random_phase_seed = " << p.random_phase_seed << "\n";
    ss << "random_mode_centered = " << (p.random_mode_centered ? 1 : 0) << "\n";
    ss << "sponge_inner_width = " << p.sponge_inner_width << "\n";
    ss << "sponge_outer_width = " << p.sponge_outer_width << "\n";
    ss << "sponge_inner_strength = " << p.sponge_inner_strength << "\n";
    ss << "sponge_outer_strength = " << p.sponge_outer_strength << "\n";
    ss << "ko_eps = " << p.ko_eps << "\n";
    ss << "lmax = " << p.lmax << "\n";
    ss << "nmax = " << p.nmax << "\n";
    ss << "sobolev_s = " << p.sobolev_s << "\n";
    ss << "trap_dx_half_width = " << p.trap_dx_half_width << "\n";
    ss << "high_ell_cut = " << p.total_high_ell_cut << "\n";
    ss << "high_n_cut = " << p.total_high_n_cut << "\n";
    ss << "horizon_probe_x = " << p.horizon_probe_x << "\n";
    ss << "outer_probe_x = " << p.outer_probe_x << "\n";
    ss << "modal_z_basis = " << p.modal_z_basis << "\n";
    ss << "field_mode = " << to_string(p.field_mode) << "\n";
    ss << "field_vars = " << to_string(p.field_vars) << "\n";
    return ss.str();
}

void print_params(const Params& p, int rank) {
    if (rank != 0 || !p.verbose) return;
    std::cout << std::setprecision(12)
              << "Parameters:\n"
              << "  outdir=" << p.outdir << "\n"
              << "  rh=" << p.rh << " Lz=" << p.Lz << " x=[" << p.x_min << "," << p.x_max << "]\n"
              << "  grid nx,nth,nz=" << p.nx << "," << p.nth << "," << p.nz
              << " nghost=" << p.nghost << " dt=" << p.dt << " tfinal=" << p.t_final << "\n"
              << "  model mu=" << p.mu << " lambda=" << p.lambda << " alpha2=" << p.alpha2_phi_box
              << " q_kind=" << to_string(p.q_kind) << "\n"
              << "  init=" << to_string(p.init_kind) << " profile=" << to_string(p.radial_profile)
              << " amp=" << p.amp << " ell0=" << p.ell0 << " n0=" << p.n0
              << " r_center=" << p.r_center << " width=" << p.width << "\n"
              << "  diagnostics lmax,nmax=" << p.lmax << "," << p.nmax
              << " modal_z_basis=" << p.modal_z_basis << "\n"
              << "  fields mode=" << to_string(p.field_mode) << " vars=" << to_string(p.field_vars)
              << " every=" << p.field_every << " diagnostics\n";
}
