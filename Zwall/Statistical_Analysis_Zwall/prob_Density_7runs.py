"""
Script determines conditional polymer segment density profiles P(z | z_COM = target).
It reads wrapped bead positions from trajectory files in serial (without MPI).
Selects chains whose z-Center of Mass (z-COM) falls within each target z-layer,
counts how many beads are in each z-layer, and computes the average density profile
along with the standard error across all processes/ranks.
"""

# ====================================================================
# IMPORT LIBRARIES
# ====================================================================

import numpy as np
import pandas as pd
from pathlib import Path
import os

# ====================================================================
# USER SETTINGS
# ====================================================================

# Directory containing all CSV trajectory files
DATA_DIR = Path.cwd()

# ------------------------------------------------------------
# Lattice and Binning Settings
# ------------------------------------------------------------
LZ = 20      # Box height
B  = 20      # Number of z-bins/layers (same as LZ)
dz = 1.0     # Bin/layer width

# ------------------------------------------------------------
# Polymer Architecture
# ------------------------------------------------------------
NC = 36      # Number of polymer chains per trajectory file
BC = 20      # Number of beads per chain

# ------------------------------------------------------------
# Rank/File Settings
# ------------------------------------------------------------
NPROC = 7
# Number of process ranks/file indices used in the data generation
# (files are numbered 0 to NPROC-1)

# ------------------------------------------------------------
# Snapshot Settings
# ------------------------------------------------------------
FN_MAX = 1000
# Maximum snapshot index

EQUIL_START = 0
# Burn-in cutoff
# All snapshots before this are discarded from equilibrium statistics.

# ====================================================================
# FIND SNAPSHOT FILE
# ====================================================================

def find_snapshot(fn, rank):
    """
    Locate the CSV file for a given:
    snapshot index (fn)
    process rank (rank)

    Example:
    fn   = 125
    rank = 3

    Searches for:
    echains.3.125.csv
    """
    filename = f"echains.{rank}.{fn}.csv"
    path = DATA_DIR / filename

    if path.exists():
        return path

    return None

# ====================================================================
# COMPUTE CONDITIONAL DENSITY
# ====================================================================

def analyze():
    
    # ------------------------------------------------------------
    # Target z-COM layers (1.5, 2.5, ..., 9.5)
    # ------------------------------------------------------------
    z_com_targets = [tar + 0.5 for tar in range(0, 10)]

    print("="*70)
    print("STARTING CONDITIONAL BEAD DENSITY CALCULATION (SERIAL MODE)")
    print("="*70)

    # Loop over all target z-COM layers
    for target in z_com_targets:

        print("\n" + "-"*60)
        print(f"PROCESSING TARGET z-COM LAYER = {target:.1f}")
        print("-"*60)

        # Array for storing final average profile of each process/rank
        rank_means = []

        # ============================================================
        # LOOP OVER RANKS (Serial processing of rank data)
        # ============================================================
        for rank in range(NPROC):
            
            # Profiles for each snapshot of this rank
            snapshot_profiles = []

            # Loop over snapshots
            for fn in range(1, FN_MAX + 1):
                
                file = find_snapshot(fn, rank)
                if file is None:
                    continue

                # Read CSV trajectory file
                df = pd.read_csv(file, dtype={"x": float, "y": float, "z": float})

                # Verify file integrity
                expected_rows = NC * BC
                if len(df) != expected_rows:
                    print(
                        f"  WARNING: {file.name} "
                        f"contains {len(df)} rows "
                        f"(expected {expected_rows})"
                    )
                    continue

                # Translate z coordinates by +0.5 to center bead in lattice cell
                zs = df['z'].values + 0.5

                bead_counts = np.zeros(B)
                chain_count = 0

                # Loop over all chains
                for c in range(NC):
                    # Extract z coordinates for this chain
                    z_beads = zs[c * BC : (c + 1) * BC]

                    # Compute z-COM for this chain
                    z_com = np.mean(z_beads)

                    # Select chain only if z-COM is within 0.5 of the target layer
                    if abs(z_com - target) >= dz / 2:
                        continue

                    # Count beads in each z-layer for this selected chain
                    for zval in z_beads:
                        bin_index = int(zval // dz)
                        bin_index = min(max(bin_index, 0), B - 1)
                        bead_counts[bin_index] += 1

                    chain_count += 1

                # Skip snapshot if no chains were selected for this target
                if chain_count == 0:
                    continue

                # Normalize bead counts by number of selected chains and beads per chain
                bead_density = bead_counts / (chain_count * BC)
                snapshot_profiles.append(bead_density)

            # Skip rank if no snapshots had selected chains
            if len(snapshot_profiles) == 0:
                print(f"  No chains found for rank {rank} at z-COM = {target:.1f}")
                continue

            # Average over all snapshots after burn-in for this rank
            snapshot_profiles = np.array(snapshot_profiles)
            
            if len(snapshot_profiles) > EQUIL_START:
                eq_profiles = snapshot_profiles[EQUIL_START:]
                rank_avg    = np.mean(eq_profiles, axis=0)
                rank_means.append(rank_avg)
            else:
                print(f"  WARNING: Rank {rank} has fewer snapshots than EQUIL_START")

        # Skip target if no data found across any ranks
        if len(rank_means) == 0:
            print(f"  No data found across all ranks for target z-COM = {target:.1f}")
            continue

        # ============================================================
        # STATISTICS ACROSS RANKS
        # ============================================================
        rank_means = np.array(rank_means)   # shape: (num_valid_ranks, 20)
        rho_mean   = np.mean(rank_means, axis=0)
        
        if len(rank_means) > 1:
            rho_std = np.std(rank_means, axis=0, ddof=1)
            rho_err = rho_std / np.sqrt(len(rank_means))
        else:
            rho_err = np.zeros(B)  # Standard error is 0 if only 1 rank is valid

        # z-bin centers
        z_centers = np.arange(B) + 0.5

        # ============================================================
        # WRITE OUTPUT CSV FOR THIS TARGET z-COM
        # ============================================================
        outname = f"prob_density_zcom_{target:.1f}.csv"
        with open(outname, "w") as f:
            f.write("z,mean_density,std_error\n")
            for i in range(B):
                f.write(f"{z_centers[i]:.2f},{rho_mean[i]:.6f},{rho_err[i]:.6f}\n")

        print(f"  --> Saved: {outname} (averaged over {len(rank_means)} ranks)")

    print("\n" + "="*70)
    print("ALL CALCULATIONS COMPLETE")
    print("="*70)

# ====================================================================
# PROGRAM ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    analyze()
