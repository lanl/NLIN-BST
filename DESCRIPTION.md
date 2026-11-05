## Output overview

All four files are controlled together by the existing `--energy-balance` / `--no-energy-balance` flag, and written at the existing `--output-every` diagnostic cadence. 
They always use the **same `step` and `time`**, so they can be joined on `(step,time)` if necessary. 
Use `--no-fields` to suppress large field snapshots while retaining energy diagnostics.

| Filename | Columns (including step,time) | Primary use |
|---|---:|---|
| `energy_components.csv` | 9 | Kinetic, radial, angular, KK, cubic energy stored inside the probes |
| `energy_fluxes.csv` | 10 | Signed instantaneous fluxes and RK4-integrated cumulative losses/work |
| `energy_budget.csv` | 12 | The conservation identity and relative residual |
| `energy_checks.csv` | 12 | Effect of omitting a budget term; inferred horizon loss; rate checks |
| `energy_probe_locations.txt` | metadata | Actual snapped probe positions, written only once |


### `energy_components.csv`

`step,time,control_energy,energy_kinetic,energy_radial,energy_angular,energy_kk,energy_cubic,energy_partition_residual`

For `P=Phi_t`, `f=1-r_h/r`, and tortoise radius `x`, all integrals are over the **same control volume** `x_H <= x <= x_O`, with measure `$$dx d \Omega dz$$`:

- `energy_kinetic`: $$\(\frac{1}{2} \int r^2P^2\)$$.
- `energy_radial`: $$\(\frac12\int r^2\Phi_x^2\)$$.
- `energy_angular`: $$\(\frac12\int f\Phi_\theta^2\)$$.
- `energy_kk`: $$\(\frac12\int fr^2\Phi_z^2\)$$.
- `energy_cubic`: $$\(\frac{\mu}{4}\int fr^2\Phi^4\)$$, zero for `--mu 0`.
- `control_energy`: their sum, denoted $$\(E_{cv}(t)\)$$.
- `energy_partition_residual`: direct sum minus five components; near roundoff by construction, **not** a conservation test.

Use this file for a stacked plot of how energy **remaining in the control volume** is partitioned among gradients, kinetic motion, and nonlinear potential. It does **not** tell you where escaped energy went.

### `energy_fluxes.csv` 

`step,time,inner_flux_signed,outer_flux_signed,sponge_dissipation_rate,bulk_power_rate,cum_inner_flux_signed,cum_outer_flux_signed,cum_sponge_dissipation,cum_bulk_power`

Instantaneous rates:

- `inner_flux_signed = F_H`: flux **out of the control volume** through the inner (horizon-side) probe. Positive means energy leaves toward smaller `x`.
- `outer_flux_signed = F_O`: flux out through the outer probe. Positive means energy leaves toward larger `x`.
- `sponge_dissipation_rate = D_\sigma`: nonnegative energy sink due to `-sigma P` **inside** the probe interval.
- `bulk_power_rate = P_{bulk}`: signed work of derivative/alpha2 terms relative to the cubic reference-energy functional; positive adds energy.

Corresponding `cum_*` columns are the signed time integrals $$\(H=\int F_H dt\)$$, $$\(O=\int F_O dt\)$$, $$\(D=\int D_\sigma dt\)$$, $$\(B=\int P_{bulk}dt\)$$. The cumulative values are integrated at **all RK4 stages** across time steps, **not** just from  sparse CSV output rows.

Signed fluxes may become negative (backflow). For a conservation law do **not** clip them to nonnegative numbers. These are fluxes at finite **worldtube probes**, not the mathematical horizon or infinity.

### `energy_budget.csv` 

`step,time,initial_control_energy,control_energy,cum_inner_flux_signed,cum_outer_flux_signed,cum_sponge_dissipation,cum_bulk_power,accounted_energy_signed,energy_balance_residual,relative_energy_balance_residual,conservation_law_applicable`

Denote the values from the four cumulative columns by `H,O,D,B`, and `E0 = initial_control_energy`. The primary ledger is

$$
\[
A(t)=E_{cv}(t)+H(t)+O(t)+D(t)-B(t),
\qquad R(t)=A(t)-E_0.
\]
$$

`accounted_energy_signed = A`; `energy_balance_residual = R`; `relative_energy_balance_residual = R/|E0|` for nonzero `E0` (zero otherwise).

The `conservation_law_applicable` flag is 1 for the standard linear/pure-cubic model without KO and with no effective derivative/alpha2 source; zero otherwise. It is **not** a quality indicator. For applicable continuum solutions the signed  residual should diminish under refinement, subject to finite-domain boundary and quadrature error.

A useful **fraction plot** can be formed from this file without storing redundant columns:

$$
\[
E_{cv}/E_0,\quad H/E_0,\quad O/E_0,\quad D/E_0,\quad -B/E_0.
\]
$$        

Their sum is $$\(1+R/E_0\)$$. Use signed values; not all fractions are necessarily positive.

### `energy_checks.csv` 

`step,time,residual_without_horizon,residual_without_outer,residual_without_sponge,residual_without_bulk,horizon_energy_only_estimate,horizon_inferred_from_balance,horizon_inference_difference,instantaneous_balance_rhs,                       control_energy_rate_finite_difference,rate_balance_residual`

Let `R` be the full residual, `H,O,D,B` the cumulative terms. Then:

- `residual_without_horizon = R-H`
- `residual_without_outer = R-O`
- `residual_without_sponge = R-D`
- `residual_without_bulk = R+B`

These **ablate a term only in the diagnostic identity**, not in the evolution. Example: if the full balance is accurate, `residual_without_sponge` should approximately equal `-D`.

- `horizon_energy_only_estimate = E0-E_cv(t)` is a *naive* estimate that confounds horizon, outward loss, damping and source work.
- `horizon_inferred_from_balance = E0-E_cv(t)-O-D+B` is the corrected inference.
- `horizon_inference_difference = H - horizon_inferred_from_balance = R`; an algebraic restatement of the conservation residual, **not** independent confirmation of horizon flux.
- `instantaneous_balance_rhs = -F_H-F_O-D_sigma+P_bulk`.
- `control_energy_rate_finite_difference` uses consecutive diagnostic outputs.
- `rate_balance_residual` is their difference. This last quantity depends on output spacing, especially for oscillatory waves. It is not a substitute for the cumulative RK4 budget residual.

### `energy_probe_locations.txt`

Records `inner_probe_x_actual`, `outer_probe_x_actual`, `inner_probe_r_actual`, `outer_probe_r_actual`. The code uses the *nearest grid points* to the requested probe positions. Keeping these values in a one-time text file avoids repeating   four constants in every time series.


## Field output and control switches 

These settings control **fields**, not the energy ledger:

| Option | What it writes | Cost / use |
|---|---|---|
| `--no-fields` | No field snapshots; CSV diagnostics remain. | Cheapest for long energy-only runs. |
| `--field-mode none` | No fields. | Same concept; `--field-every 0` also disables fields. |
| `--field-mode slices` | Two CSV planes per snapshot: `fields/slices_STEP_theta_equator.csv` (x-z plane, theta near equator), `fields/slices_STEP_z0.csv` (x-theta plane, z=0), both include `phi,P`. | Visualization. |
| `--field-mode full_csv` | `fields/full_STEP.csv`, rows with `x,r,theta,z,phi` and optionally `P`. | Readable but extremely large. |
| `--field-mode full_binary` | MPI-written full 3D binary snapshots. | Recommended for actual field-difference convergence. |
| `--field-mode slices_full_binary` | Both the two slices and the binary full field. | Useful small pilot runs. |
| `--field-vars phi` | `Phi` only in full output. | ~8 bytes/grid point/snapshot. |
| `--field-vars phiP` | `Phi` and `P` in full output. | ~16 bytes/grid point/snapshot; recommended for convergence. |
| `--output-every N` | Main diagnostics every `N` evolution steps. | Interval in physical time = `N*dt`. |
| `--field-every M` | Field snapshot at every `M`th diagnostic output. | Interval = `N*M*dt` (for positive M). |
| `--energy-balance` | Turn on `energy_balance.csv` in the compatible branch. | Independent of field snapshots. |
| `--no-energy-balance` | Turn off `energy_balance.csv`. | Does **not** disable basic `diagnostics.csv`. |


### Other principal CSV outputs 

| File | Use / interpretation |
|---|---|
| `diagnostics.csv` | `total_l2`, `trap_l2`, `total_energy`, `trap_energy`, maxima, Sobolev/trapped statistics, mode centroids and high-mode fraction. `total_energy` is **whole-grid** energy, not control-volume energy. |
| `fluxes.csv` | Instantaneous flux and cumulative **positive-clipped** flux estimates at sparse diagnostic times; use `energy_balance.csv` signed RK4 cumulative terms for conservation instead. |
| `modal_trap_spectrum.csv` | Energy per tracked `(ell,n)` integrated in each mode's *own trapping region*; do not naively sum as a global partition. |
| `modal_well_coefficients.csv` | Cos/sin components of projected `Phi,P` at modal well. |
| `horizon_spectrum.csv` | Modal shell diagnostic near the inner surface; not a full horizon energy flux decomposition. |
| `mode_info.csv` | `(ell,n)`, wavenumber, trapping flags and well position/potential. |
| `grid_x.csv`, `grid_theta.csv`, `grid_z.csv` | Coordinates, geometry, and grid metadata. |
| `initial_mode_weights.txt`, `initial_data_normalization.txt` | Present in the previously normalized-ID branch; document balancing and reference energy normalization. |
| `modal_total_spectrum.csv` | **Only if** your branch independently includes the full-domain modal-spectrum patch; not added by this two-file energy refactor. |

