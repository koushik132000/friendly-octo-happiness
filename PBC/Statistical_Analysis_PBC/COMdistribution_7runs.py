#==============================================================================
# Computes the z-direction center of mass (COM) distribution of polymer chains
# for full 3D PBC (Periodic Boundary Conditions) system.
#
# Statistical treatment — two-level error analysis:
#
#   Level 1 — WITHIN each run (1000 files, TIME-CORRELATED):
#             Snapshots are not independent in time. Blocking analysis
#             gives the correlation-corrected error on each run's mean.
#
#   Level 2 — ACROSS 7 runs (INDEPENDENT):
#             Each run has a different random seed. The 7 run means are
#             independent → inter-run SEM is exact:
#                 inter_sem = std(run_means, ddof=1) / sqrt(NRUNS)
#
#   Combined final error:
#             final_err = sqrt( inter_sem^2
#                             + (mean_blk_err / sqrt(n_valid))^2 )
#
# PBC unwrapping:
#   All three coordinates (x, y, z) are unwrapped using the minimum
#   image convention before computing the chain COM. This is essential
#   for 3D PBC — a chain can span ANY boundary.
#   After unwrapping, z_com is re-wrapped into [0, Lz) for binning.
#
# Input  : echains.{run}.{fn}.csv     — wrapped bead coordinates (x, y, z)
# Output : com_dis_final_avg.csv      — final averaged COM distribution
#==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# USER SETTINGS
# ==============================================================================

DATA_DIR  = Path.cwd()

Lx        = 7.0
Ly        = 7.0
Lz        = 20.0

N_BINS    = 20
BIN_WIDTH = Lz / N_BINS    # = 1.0

nc        = 36              # number of chains per run
bc        = 20              # beads per chain

NRUNS       = 7
FN_MAX      = 1000
EQUIL_START = 25

# ==============================================================================
# LEVEL-1: BLOCKING ANALYSIS WITHIN A SINGLE RUN
#
# Why blocking?
#   Consecutive snapshots are correlated — naive std/sqrt(N) underestimates
#   the true error. Blocking groups snapshots into blocks of increasing size.
#   Once block size > correlation length, block means are independent and the
#   error estimate plateaus. That plateau = true error on this run's mean.
# ==============================================================================

def blocking_error(time_series):
    """
    Correlation-corrected error via blocking analysis.

    Parameters
    ----------
    time_series : ndarray, shape (n_snapshots, N_BINS)

    Returns
    -------
    plateau_err : ndarray, shape (N_BINS,)
        True error on this run's mean, corrected for autocorrelation.
    errors : ndarray, shape (n_block_sizes, N_BINS)
        Error at every block size — for diagnostic plots.
    bsizes : list of int
        Block sizes used.
    """
    n      = len(time_series)
    errors = []
    bsizes = []
    blk    = 1

    while n // blk >= 10:             # need ≥10 blocks to estimate std
        n_blocks    = n // blk
        trimmed     = time_series[:n_blocks * blk]
        block_means = trimmed.reshape(n_blocks, blk, -1).mean(axis=1)
        err         = block_means.std(axis=0, ddof=1) / np.sqrt(n_blocks)
        errors.append(err)
        bsizes.append(blk)
        blk *= 2

    errors = np.array(errors)         # (n_block_sizes, N_BINS)

    # ------------------------------------------------------------------
    # Plateau detection via relative gradient convergence.
    # Find where error stops growing (relative change < 5%) and average
    # from that point. Always average at least 2 rows for stability.
    #
    # BUG FIX 5 — original used min(plateau_start, len(errors)-1)
    # which could give a single-row mean. Changed to len(errors)-2
    # to always average at least 2 block sizes at the plateau.
    # ------------------------------------------------------------------
    if len(errors) >= 4:
        denom      = np.maximum(errors[:-1], 1e-12)
        rel_change = np.abs(np.diff(errors, axis=0)) / denom

        converged_from = []
        for z in range(errors.shape[1]):
            large = np.where(rel_change[:, z] > 0.05)[0]
            converged_from.append(large[-1] + 1 if len(large) > 0 else 0)

        # BUG FIX 5: cap at len-2 so plateau average always has ≥2 points
        plateau_start = min(max(converged_from), len(errors) - 2)
        plateau_err   = errors[plateau_start:].mean(axis=0)
    else:
        plateau_err = errors[-1]

    return plateau_err, errors, bsizes


# ==============================================================================
# 3D MINIMUM IMAGE UNWRAPPING
#
# For 3D PBC a chain can span the boundary in x, y, OR z.
# Naive mean(z_beads) on wrapped coordinates gives a wrong COM whenever
# the chain crosses the z boundary. Unwrapping reconstructs the true
# contiguous chain geometry before computing the COM.
#
# The minimum image convention:
#   For each consecutive bead pair, take the shortest displacement
#   (which must be < half the box length for a bonded lattice chain).
# ==============================================================================

def unwrap_chain_3d(x_beads, y_beads, z_beads):
    """
    Unwrap all three coordinates using minimum image convention.

    Parameters
    ----------
    x_beads, y_beads, z_beads : ndarray, shape (bc,)
        Wrapped bead coordinates for one chain.

    Returns
    -------
    x_uw, y_uw, z_uw : ndarray, shape (bc,)
        Unwrapped coordinates. Use z_uw.mean() for the true z-COM.
    """
    n_beads = len(x_beads)

    x_uw = np.empty(n_beads)
    y_uw = np.empty(n_beads)
    z_uw = np.empty(n_beads)

    x_uw[0] = x_beads[0]
    y_uw[0] = y_beads[0]
    z_uw[0] = z_beads[0]

    for i in range(1, n_beads):

        # x — minimum image
        dx = x_beads[i] - x_beads[i - 1]
        if   dx >  0.5 * Lx: dx -= Lx
        elif dx < -0.5 * Lx: dx += Lx
        x_uw[i] = x_uw[i - 1] + dx

        # y — minimum image
        dy = y_beads[i] - y_beads[i - 1]
        if   dy >  0.5 * Ly: dy -= Ly
        elif dy < -0.5 * Ly: dy += Ly
        y_uw[i] = y_uw[i - 1] + dy

        # z — minimum image (critical for 3D PBC)
        # Without this, a chain straddling z=0/Lz gets a wrong COM.
        dz = z_beads[i] - z_beads[i - 1]
        if   dz >  0.5 * Lz: dz -= Lz
        elif dz < -0.5 * Lz: dz += Lz
        z_uw[i] = z_uw[i - 1] + dz

    return x_uw, y_uw, z_uw


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

run_means    = []   # one mean profile per run  → shape (NRUNS, N_BINS)
run_blk_errs = []   # one blocking error per run → shape (NRUNS, N_BINS)
last_eq_prof = None

for run in range(NRUNS):

    print("\n" + "=" * 60)
    print(f"PROCESSING RUN {run}")
    print("=" * 60)

    snap_profiles = []

    for fn in range(1, FN_MAX + 1):

        infname = DATA_DIR / f"echains.{run}.{fn}.csv"
        if not infname.exists():
            continue

        pos = pd.read_csv(infname, dtype={"x": float, "y": float, "z": float})

        # +0.5 centers each bead in its lattice cell before unwrapping
        xs = pos['x'].values + 0.5
        ys = pos['y'].values + 0.5
        zs = pos['z'].values + 0.5

        hits = np.zeros(N_BINS, dtype=int)

        for c in range(nc):

            x_beads = xs[c * bc:(c + 1) * bc]
            y_beads = ys[c * bc:(c + 1) * bc]
            z_beads = zs[c * bc:(c + 1) * bc]

            # Unwrap all three directions — essential for 3D PBC
            _, _, z_uw = unwrap_chain_3d(x_beads, y_beads, z_beads)

            # z-COM from unwrapped coordinates
            z_com = z_uw.mean()

            # Re-wrap z_com into [0, Lz) for binning
            z_com_wrapped = z_com % Lz

            # BUG FIX 1+2 — use floor + clip instead of int() alone.
            # int() truncates toward zero (wrong for negative values).
            # floor is always correct. clip handles the rare float edge
            # where z_com_wrapped rounds to exactly Lz giving bin=N_BINS.
            bin_index = np.clip(
                int(np.floor(z_com_wrapped / BIN_WIDTH)),
                0, N_BINS - 1
            )
            hits[bin_index] += 1

        total_hits = hits.sum()
        if total_hits != nc:
            print(f"  WARNING run {run} snapshot {fn}: "
                  f"{total_hits} hits instead of {nc} — skipping")
            continue

        # Normalize so uniform distribution → rho = 1.0 everywhere
        rho_com = hits / (total_hits / N_BINS)

        # Mirror averaging — exploits midplane symmetry of 3D PBC box
        mirror_avg = 0.5 * (rho_com + rho_com[::-1])
        snap_profiles.append(mirror_avg)

        if fn % 100 == 0:
            print(f"  Run {run} | snapshot {fn}")

    # --------------------------------------------------------------------------
    # Guards
    # --------------------------------------------------------------------------
    if len(snap_profiles) == 0:
        print(f"  ERROR run {run}: no valid snapshots — skipping run.")
        continue

    snap_profiles = np.array(snap_profiles)   # (n_snapshots, N_BINS)

    if len(snap_profiles) <= EQUIL_START:
        print(f"  ERROR run {run}: insufficient snapshots after burn-in — skipping.")
        continue

    eq_profiles = snap_profiles[EQUIL_START:]
    n_eq        = len(eq_profiles)

    # --------------------------------------------------------------------------
    # Level-1: blocking analysis on this run's correlated snapshots
    # --------------------------------------------------------------------------
    plateau_err, errors, bsizes = blocking_error(eq_profiles)

    run_means.append(eq_profiles.mean(axis=0))
    run_blk_errs.append(plateau_err)
    last_eq_prof = eq_profiles

    print(f"  Run {run} | {n_eq} equilibrated snapshots")
    print(f"           | blocking error at midplane: {plateau_err[N_BINS//2]:.5f}")
    print(f"           | plateau block size ~ "
          f"{bsizes[min(len(bsizes)//2, len(bsizes)-1)]}")

# ==============================================================================
# FINAL STATISTICS — TWO-LEVEL COMBINATION
#
# run_means    : (n_valid, N_BINS) — independent run means
# run_blk_errs : (n_valid, N_BINS) — within-run blocking errors
#
# Level-2 inter-run SEM (runs are independent → exact):
#   inter_sem = std(run_means, ddof=1) / sqrt(n_valid)
#
# Level-1 mean blocking error (averaged across runs for stability):
#   mean_blk = mean(run_blk_errs, axis=0)
#
# Combined final error:
#   final_err = sqrt( inter_sem^2 + (mean_blk / sqrt(n_valid))^2 )
#
#   The second term scales by 1/n_valid because the blocking error
#   estimates uncertainty on ONE run's mean; its contribution to the
#   GRAND mean (average of n_valid independent runs) is reduced by
#   1/sqrt(n_valid).
# ==============================================================================

if len(run_means) == 0:
    raise RuntimeError("No valid runs were processed. Check DATA_DIR and file names.")

n_valid      = len(run_means)
run_means    = np.array(run_means)       # (n_valid, N_BINS)
run_blk_errs = np.array(run_blk_errs)   # (n_valid, N_BINS)

final_mean   = run_means.mean(axis=0)
inter_sem    = run_means.std(axis=0, ddof=1) / np.sqrt(n_valid)
mean_blk_err = run_blk_errs.mean(axis=0)

# BUG FIX 4 — combined two-level error (was missing in original)
final_err    = np.sqrt(inter_sem**2 + (mean_blk_err / np.sqrt(n_valid))**2)

# Physical z-bin centres: 0.5, 1.5, ..., 19.5
z_centers = (np.arange(N_BINS) + 0.5) * BIN_WIDTH

# ==============================================================================
# PRINT SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("TWO-LEVEL ERROR ANALYSIS SUMMARY  (3D PBC)")
print("=" * 70)
print(f"  Valid runs       : {n_valid} / {NRUNS}")
print(f"\n{'z':>6}  {'rho':>8}  {'inter_sem':>10}  {'blk_err':>10}  {'final_err':>10}")
print("-" * 52)
for i in range(N_BINS):
    print(f"  {z_centers[i]:>4.1f}  {final_mean[i]:>8.5f}  "
          f"{inter_sem[i]:>10.5f}  {mean_blk_err[i]:>10.5f}  {final_err[i]:>10.5f}")

print(f"\n  Inter-run SEM range     : [{inter_sem.min():.5f},  {inter_sem.max():.5f}]")
print(f"  Mean blocking err range : [{mean_blk_err.min():.5f},  {mean_blk_err.max():.5f}]")
print(f"  Final combined err range: [{final_err.min():.5f},  {final_err.max():.5f}]")

# ==============================================================================
# SAVE OUTPUT
# ==============================================================================

odf = pd.DataFrame({
    "z"           : z_centers,
    "rho"         : final_mean,
    "err"         : final_err,        # combined — USE THIS for error bars
    "inter_sem"   : inter_sem,        # Level-2 only
    "blk_err"     : mean_blk_err,     # Level-1 only (mean across runs)
})
odf.to_csv("com_dis_final_avg.csv", index=False)
print(f"\nOutput written to com_dis_final_avg.csv")
print(f"Columns: z, rho, err (combined), inter_sem (Level-2), blk_err (Level-1)")

# ==============================================================================
# PLOT — main result with combined error bars
# ==============================================================================

fig, ax = plt.subplots(figsize=(7, 4))

ax.errorbar(odf['z'], odf['rho'], yerr=odf['err'],
            fmt='o-', capsize=4, label='COM density ± combined error')
ax.errorbar(odf['z'], odf['rho'], yerr=odf['inter_sem'],
            fmt='s--', capsize=2, alpha=0.4, label='± inter-run SEM only')

ax.axvline(x=Lz / 2, color='black', linestyle=':', linewidth=0.8, label='midplane')
ax.axhline(y=1.0,    color='red',   linestyle='--',               label='uniform')

ax.set_xlabel("z (lattice units)")
ax.set_ylabel("Normalized COM density")
ax.set_title("COM Distribution along z (3D PBC) — two-level errors")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("com_distribution.png", dpi=150)
plt.show()

# ==============================================================================
# BLOCKING DIAGNOSTIC PLOT
# Shows how error grows with block size and where plateau is (last run)
# ==============================================================================

_, errors_diag, bsizes_diag = blocking_error(last_eq_prof)
z_idx = N_BINS // 2

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.semilogx(bsizes_diag, errors_diag[:, z_idx], 'o-', label='blocking error')
ax2.axhline(y=run_blk_errs[-1][z_idx], color='red',
            linestyle='--', label='plateau (last run)')
ax2.set_xlabel("Block size")
ax2.set_ylabel("Error estimate")
ax2.set_title(f"Blocking diagnostic — z layer {z_idx} (run {n_valid - 1})")
ax2.grid(alpha=0.3)
ax2.legend()
fig2.tight_layout()
fig2.savefig("blocking_diagnostic.png", dpi=150)
plt.show()
