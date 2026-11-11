#pragma once
#include <mpi.h>
#include <string>

// Runtime choices are parsed once into enums.  The original string is kept only
// for parameter files and human-readable output.
enum class QKind {
    None,
    Metric,
    Spatial,
    NullX,
    BadTime
};

enum class InitKind {
    Eq44,
    TwoMode,
    RandomTrapped,
    ZIndependent,
    S2Symmetric,
    OffWell,
    Gaussian
};

enum class RadialProfileKind {
    Bump8,
    Gaussian,
    CInf
};

enum class FieldOutputMode {
    None,
    Slices,
    FullCSV,
    FullBinary,
    SlicesFullBinary
};

enum class FieldOutputVars {
    Phi,
    PhiP
};

// Optional balancing of individual seed modes before the global initial-data
// normalization is applied.
enum class IDBalanceKind {
    None,
    PaperL,
    ModeEnergy
};

// Optional normalization of the complete initial-data shape.
enum class IDNormalizeKind {
    None,
    Quadratic,
    FullCubic
};

struct Params {
    // Geometry and domain.
    double rh = 1.0;
    double Lz = 6.0;
    double x_min = -20.0;
    double x_max = 40.0;
    int nx = 384;
    int nth = 64;
    int nz = 64;
    int nghost = 3;

    // Time integration.
    double dt = 0.005;
    double t_final = 50.0;
    int output_every = 100;
    // Field output is tied to diagnostic outputs: field_every=1 writes at every
    // diagnostic, field_every=20 writes at every twentieth diagnostic, and 0
    // disables all field dumps regardless of field_mode.
    int field_every = 20;

    // Nonlinear model.  The evolved equation is
    //   Box_g phi = mu phi^3 + lambda Q + alpha2_phi_box phi Box_g phi,
    // implemented as
    //   Box_g phi = (mu phi^3 + lambda Q)/(1 - alpha2_phi_box phi).
    // For the defocusing cubic model in the note, use mu=1, lambda=0.
    double mu = 0.0;
    double lambda = 0.0;
    double alpha2_phi_box = 0.0;
    std::string q_kind_name = "none";
    QKind q_kind = QKind::None;

    // Initial data.
    std::string init_name = "eq44";
    InitKind init_kind = InitKind::Eq44;
    std::string radial_profile_name = "bump8";
    RadialProfileKind radial_profile = RadialProfileKind::Bump8;
    double amp = 0.1;
    int ell0 = 2;
    int n0 = 1;
    double r_center = 9.45;
    double width = 1.0;
    int two_mode_ell1 = 2;
    int two_mode_n1 = 1;
    int two_mode_ell2 = 4;
    int two_mode_n2 = 2;
    double two_mode_a = 0.5;
    double offwell_r_center = 5.0;
    double random_phase_seed = 1.2345;
    bool random_mode_centered = true;

    // Initial-data comparison controls.
    //
    // id_balance=scale_l reproduces the approximate 1/ell weighting used in
    // Appendix A of arXiv:2411.17445.  id_balance=mode_energy is the preferred
    // black-string generalization: each seed mode is multiplied by the inverse
    // square root of its unit-amplitude quadratic energy, including both the
    // angular and Kaluza-Klein contributions.
    std::string id_balance_name = "none";
    IDBalanceKind id_balance = IDBalanceKind::None;

    // id_normalize fixes the energy of the complete reference shape at amp=1.
    // The user amplitude amp is then applied after that normalization.
    std::string id_normalize_name = "none";
    IDNormalizeKind id_normalize = IDNormalizeKind::None;
    double id_target_energy = 1.0;

    // Sponge layers and artificial dissipation.
    double sponge_inner_width = 4.0;
    double sponge_outer_width = 8.0;
    double sponge_inner_strength = 0.20;
    double sponge_outer_strength = 0.20;
    double ko_eps = 0.02;

    // Modal diagnostics.
    int lmax = 8;
    int nmax = 8;
    int sobolev_s = 2;
    double trap_dx_half_width = 2.0;
    int total_high_ell_cut = 5;
    int total_high_n_cut = 4;
    double horizon_probe_x = -12.0;
    double outer_probe_x = 34.0;
    std::string modal_z_basis = "cos_sin"; // cosine or cos_sin
    bool modal_include_sine = true;

    // A separate control-volume energy ledger is written to
    // energy_balance.csv.  This keeps the existing diagnostics.csv convention
    // unchanged while providing signed horizon/outer flux accounting.
    bool energy_balance = true;

    // Output.
    std::string outdir = "run_nlin_wave";
    std::string field_mode_name = "slices"; // none, slices, full_csv, full_binary, slices_full_binary
    FieldOutputMode field_mode = FieldOutputMode::Slices;
    std::string field_vars_name = "phi"; // phi or phiP
    FieldOutputVars field_vars = FieldOutputVars::Phi;
    bool verbose = true;
};

void print_usage(int rank);
Params parse_args(int argc, char** argv, MPI_Comm comm);
void print_params(const Params& p, int rank);
std::string params_to_string(const Params& p);

const char* to_string(QKind v);
const char* to_string(InitKind v);
const char* to_string(RadialProfileKind v);
const char* to_string(FieldOutputMode v);
const char* to_string(FieldOutputVars v);
const char* to_string(IDBalanceKind v);
const char* to_string(IDNormalizeKind v);
