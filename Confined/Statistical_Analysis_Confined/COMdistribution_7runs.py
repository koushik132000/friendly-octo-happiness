#==============================================================================
# Computes the z-direction center of mass (COM) distribution of polymer chains
# for Zwall confined system.
#
# Protocol:
# 1) Each run reads echains coordinate files, applies +0.5 translation,
#    and computes the z-COM for each chain per snapshot.
# 2) Each run bins the z-COMs into z-layers, normalizes, and applies
#    mirror averaging per snapshot, then averages over all snapshots.
# 3) All run averages are collected, final mean and standard error are
#    computed across independent runs, and written to one output CSV file.
#
# Input  : echains.{run}.{fn}.csv     — wrapped bead coordinates
# Output : com_dis_final_avg.csv      — final averaged COM distribution
#==============================================================================

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

# ====================================================================
# USER SETTINGS
# ====================================================================

DATA_DIR = Path.cwd()

Lz = 20.0
B  = 20
dz = Lz / B
nc = 36
bc = 20

NRUNS       = 7
FN_MAX      = 1000
EQUIL_START = 25

# ====================================================================
# MAIN LOOP OVER INDEPENDENT RUNS
# For each run and each snapshot this section:
# 1) Reads echains bead coordinates and applies +0.5 translation to z
# 2) Computes z-COM for each chain by summing bead z-coordinates
# 3) Bins z-COMs into B=20 z-layers and counts hits per layer
# 4) Normalizes hit counts by average hits per bin
# 5) Applies mirror averaging: mean[l] = 0.5*(rho[l] + rho[19-l])
# 6) Collects all snapshots and computes run-wise mean profile
# ====================================================================

run_means = []  # one mean profile per independent run

for run in range(NRUNS):

    print("\n" + "="*60)
    print(f"PROCESSING RUN {run}")
    print("="*60)

    # Collect mirror-averaged profiles from all snapshots for this run
    snap_profiles = []

    for fn in range(1, FN_MAX + 1):

        # Read echains coordinate file for this run and snapshot
        infname = DATA_DIR / f"echains.{run}.{fn}.csv"

        if not infname.exists():
            continue

        pos = pd.read_csv(infname, dtype={"x": float, "y": float, "z": float})

        # Translate z coordinates by +0.5 to center bead in lattice cell
        zs = pos['z'].values + 0.5

        # Initialize hit counter for B bins
        hits = np.zeros(B, dtype=int)

        # Compute z-COM for each chain and bin into the appropriate layer
        for c in range(nc):
            sz    = zs[c * bc : (c + 1) * bc].sum()   # vectorized, no inner loop
            z_com = sz / bc
            bin_index = int(z_com // dz)
            if 0 <= bin_index < B:
                hits[bin_index] += 1

        # Verify total hits equals number of chains
        total_hits = hits.sum()
        if total_hits != nc:
            print(f"  WARNING run {run} snapshot {fn}: {total_hits} hits, expected {nc}")
            continue

        # Normalize hit counts by average hits per bin
        avg_hits = hits.mean()
        rho_com  = hits / avg_hits

        # Mirror averaging to exploit box symmetry about midplane
        mirror_avg = 0.5 * (rho_com + rho_com[::-1])
        snap_profiles.append(mirror_avg)

        if fn % 100 == 0:
            print(f"  Run {run} | snapshot {fn}")

    # ------------------------------------------------------------------
    # Discard burn-in snapshots and compute run-wise mean profile
    # ------------------------------------------------------------------
    snap_profiles = np.array(snap_profiles)          # shape: (n_snapshots, 20)
    eq_profiles   = snap_profiles[EQUIL_START:]      # discard burn-in
    run_means.append(np.mean(eq_profiles, axis=0))   # one mean per run

# ==============================================================================
# FINAL STATISTICS ACROSS INDEPENDENT RUNS
# std across 7 independent run means gives the correct error bar
# ==============================================================================

run_means = np.array(run_means)    # shape: (NRUNS, 20)

rho_mean = np.mean(run_means, axis=0)
rho_std  = np.std(run_means,  axis=0, ddof=1)
rho_err  = rho_std / np.sqrt(NRUNS)

# z-bin centers: 0.5, 1.5, ..., 19.5
z_centers = np.arange(B) + 0.5

print("\n" + "="*70)
print("FINAL EQUILIBRATED COM DISTRIBUTION")
print("="*70)

odf = pd.DataFrame({
    "z"   : z_centers,
    "rho" : rho_mean,
    "err" : rho_err,
})
odf.to_csv("com_dis_final_avg1.csv", index=False)
print(f"\nOutput written to com_dis_final_avg.csv")
print(f"Used {NRUNS} independent runs, each with {FN_MAX - EQUIL_START} equilibrated snapshots")

# ==============================================================================
# PLOT
# ==============================================================================

odf.plot(x='z', y='rho', yerr='err', capsize=4)

plt.axvline(x=10,  color='black', linestyle=':',  linewidth=0.5)
plt.axhline(y=1.0, color='r',     linestyle='--')

plt.xlabel("z layer")
plt.ylabel("Normalized COM density")
plt.title("COM Distribution along z")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
