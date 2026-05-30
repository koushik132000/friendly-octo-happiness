"""
======================================================================
 POLYMER CHAIN ANALYSIS FOR MPI MONTE CARLO SIMULATIONS
======================================================================

This script analyzes polymer configurations generated from:

    mpirun -np 7 ./a.out

where each MPI rank produces files like:

    echains.0.1.csv
    echains.1.1.csv
    ...
    echains.6.1.csv

----------------------------------------------------------------------
 WHAT THIS CODE COMPUTES
----------------------------------------------------------------------

For every polymer chain:

    1. End-to-end distance squared
            Ree²

    2. Radius of gyration squared
            Rg²

    3. Directional gyration components
            Rgx², Rgy², Rgz²

For equilibrated production trajectories:

    • RMS polymer size
    • Statistical uncertainty (SEM)
    • Time evolution of chain dimensions

----------------------------------------------------------------------
 IMPORTANT PHYSICAL IDEA
----------------------------------------------------------------------

Each MPI rank is treated as an INDEPENDENT Monte Carlo simulation.

This is statistically much better than combining all chains together
as independent samples because configurations within a trajectory
are time-correlated.

Therefore:

    • averages are first computed INSIDE each rank
    • final statistics are computed ACROSS ranks

This gives more reliable error bars.

----------------------------------------------------------------------
 AUTHOR NOTES
----------------------------------------------------------------------

The code is intentionally written with:
    • extremely explicit variable names
    • educational comments
    • physically meaningful structure
    • minimal hidden logic

so future-you can understand it after six months :)

======================================================================
"""

# ====================================================================
# IMPORT LIBRARIES
# ====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ====================================================================
# USER SETTINGS
# ====================================================================

# Directory containing all CSV trajectory files
DATA_DIR = Path.cwd()

# ------------------------------------------------------------
# Polymer architecture
# ------------------------------------------------------------

NC = 36
# Number of polymer chains PER MPI rank

BC = 20
# Number of beads per chain

# ------------------------------------------------------------
# MPI simulation settings
# ------------------------------------------------------------

NPROC = 7
# Number of MPI ranks used in:
#     mpirun -np 7 ./a.out

# ------------------------------------------------------------
# Snapshot settings
# ------------------------------------------------------------

FN_MAX = 1000
# Maximum snapshot index

EQUIL_START = 50
# Burn-in cutoff
#
# All snapshots before this are discarded
# from equilibrium statistics.

# ------------------------------------------------------------
# Terminal output control
# ------------------------------------------------------------

PRINT_EVERY = 100
# Print progress every N snapshots
#
# Set to:
#     0  -> silent mode
#     1  -> print every snapshot


# ====================================================================
# FIND SNAPSHOT FILE
# ====================================================================

def find_snapshot(fn, rank):
    """
    Locate the CSV file for a given:

        snapshot index
        MPI rank

    Example:
        fn   = 125
        rank = 3

    searches for:

        echains.3.125.csv
    """

    filename = f"unwrapc.{rank}.{fn}.csv"

    path = DATA_DIR / filename

    if path.exists():
        return path

    return None


# ====================================================================
# COMPUTE SINGLE-CHAIN METRICS
# ====================================================================

def compute_chain_metrics(df):
    """
    Compute polymer structural observables
    for all chains in ONE snapshot file.

    Returns:

        Ree²
        Rg²
        Rgx²
        Rgy²
        Rgz²

    for every chain.
    """

    # ------------------------------------------------------------
    # Allocate arrays
    # ------------------------------------------------------------

    Ree2 = np.empty(NC)

    Rg2  = np.empty(NC)

    Rg2x = np.empty(NC)
    Rg2y = np.empty(NC)
    Rg2z = np.empty(NC)

    # ============================================================
    # LOOP OVER POLYMER CHAINS
    # ============================================================

    for c in range(NC):

        # --------------------------------------------------------
        # Extract bead coordinates for chain c
        # --------------------------------------------------------

        pts = df.iloc[
            c*BC:(c+1)*BC
        ][["x", "y", "z"]].values.astype(float)

        # ========================================================
        # END-TO-END DISTANCE
        # ========================================================
        #
        # Ree² = |r_N - r_1|²
        #
        # Measures overall chain extension.
        #
        # ========================================================

        dr = pts[-1] - pts[0]

        Ree2[c] = np.dot(dr, dr)

        # ========================================================
        # RADIUS OF GYRATION
        # ========================================================
        #
        #              1
        # Rg² = --------------- Σ |r_i - r_cm|²
        #           N beads
        #
        # Measures average spatial spread
        # of the polymer around its center of mass.
        #
        # ========================================================

        # Center of mass

        r_cm = pts.mean(axis=0)

        # Displacement of every bead
        # from center of mass

        disp = pts - r_cm

        # Squared displacements

        sq = disp**2

        # Total Rg²

        Rg2[c] = np.mean(np.sum(sq, axis=1))

        # --------------------------------------------------------
        # Directional contributions
        # --------------------------------------------------------

        comp = np.mean(sq, axis=0)

        Rg2x[c] = comp[0]
        Rg2y[c] = comp[1]
        Rg2z[c] = comp[2]

    return Ree2, Rg2, Rg2x, Rg2y, Rg2z


# ====================================================================
# COMPUTE RMS VALUE + SEM
# ====================================================================

def rms_and_error(data):
    """
    Convert squared quantity into RMS quantity.

    Example:

        input:
            Rg² samples

        output:
            Rg ± SEM

    Uses standard error propagation:

        d(sqrt(x)) = dx / (2 sqrt(x))
    """

    # Mean squared value

    mean2 = np.mean(data)

    # RMS quantity

    rms = np.sqrt(mean2)

    # Standard deviation

    std2 = np.std(data, ddof=1)

    # Standard error of mean

    sem2 = std2 / np.sqrt(len(data))

    # Propagated RMS uncertainty

    err_rms = sem2 / (2*rms)

    return rms, err_rms


# ====================================================================
# MAIN ANALYSIS ROUTINE
# ====================================================================

def analyze():

    # ============================================================
    # ARRAYS FOR FINAL RANK-WISE STATISTICS
    # ============================================================

    rank_Rg = []

    rank_Rgx = []
    rank_Rgy = []
    rank_Rgz = []

    rank_Ree = []

    # ============================================================
    # ARRAYS FOR PLOTTING
    # ============================================================

    steps = []

    Rg_vs_time = []

    Rgx_vs_time = []
    Rgy_vs_time = []
    Rgz_vs_time = []

    # ============================================================
    # LOOP OVER MPI RANKS
    # ============================================================

    for rank in range(NPROC):

        print("\n" + "="*60)

        print(f"PROCESSING MPI RANK {rank}")

        print("="*60)

        # --------------------------------------------------------
        # Time-series arrays for THIS rank
        # --------------------------------------------------------

        rank_time_Rg = []

        rank_time_Rgx = []
        rank_time_Rgy = []
        rank_time_Rgz = []

        rank_time_Ree = []

        # ========================================================
        # LOOP OVER SNAPSHOTS
        # ========================================================

        for fn in range(FN_MAX + 1):

            # ----------------------------------------------------
            # Find corresponding trajectory file
            # ----------------------------------------------------

            file = find_snapshot(fn, rank)

            if file is None:
                continue

            # ----------------------------------------------------
            # Read CSV trajectory
            # ----------------------------------------------------

            df = pd.read_csv(file)

            expected_rows = NC * BC

            # ----------------------------------------------------
            # Verify file integrity
            # ----------------------------------------------------

            if len(df) != expected_rows:

                print(
                    f"WARNING: {file.name} "
                    f"contains {len(df)} rows "
                    f"(expected {expected_rows})"
                )

                continue

            # ----------------------------------------------------
            # Compute chain observables
            # ----------------------------------------------------

            Ree2, Rg2, Rg2x, Rg2y, Rg2z = \
                compute_chain_metrics(df)

            # ====================================================
            # CONVERT TO RMS QUANTITIES
            # ====================================================

            Ree_rms, _ = rms_and_error(Ree2)

            Rg_rms, _ = rms_and_error(Rg2)

            Rgx_rms, _ = rms_and_error(Rg2x)

            Rgy_rms, _ = rms_and_error(Rg2y)

            Rgz_rms, _ = rms_and_error(Rg2z)

            # ----------------------------------------------------
            # Store time evolution
            # ----------------------------------------------------

            rank_time_Ree.append(Ree_rms)

            rank_time_Rg.append(Rg_rms)

            rank_time_Rgx.append(Rgx_rms)

            rank_time_Rgy.append(Rgy_rms)

            rank_time_Rgz.append(Rgz_rms)

            # ----------------------------------------------------
            # Global plotting arrays
            #
            # Only use rank 0 for visualization
            # to avoid overcrowded plots.
            # ----------------------------------------------------

            if rank == 0:

                steps.append(fn)

                Rg_vs_time.append(Rg_rms)

                Rgx_vs_time.append(Rgx_rms)

                Rgy_vs_time.append(Rgy_rms)

                Rgz_vs_time.append(Rgz_rms)

            # ----------------------------------------------------
            # Progress printing
            # ----------------------------------------------------

            if PRINT_EVERY > 0 and fn % PRINT_EVERY == 0:

                print(
                    f"Rank {rank:2d} | "
                    f"Snapshot {fn:4d} | "
                    f"Rg = {Rg_rms:.4f}"
                )

        # ========================================================
        # DISCARD NON-EQUILIBRATED REGION
        # ========================================================

        eq_Ree = np.array(rank_time_Ree[EQUIL_START:])

        eq_Rg = np.array(rank_time_Rg[EQUIL_START:])

        eq_Rgx = np.array(rank_time_Rgx[EQUIL_START:])

        eq_Rgy = np.array(rank_time_Rgy[EQUIL_START:])

        eq_Rgz = np.array(rank_time_Rgz[EQUIL_START:])

        # --------------------------------------------------------
        # Compute equilibrium mean for THIS rank
        # --------------------------------------------------------

        rank_Ree.append(np.mean(eq_Ree))

        rank_Rg.append(np.mean(eq_Rg))

        rank_Rgx.append(np.mean(eq_Rgx))

        rank_Rgy.append(np.mean(eq_Rgy))

        rank_Rgz.append(np.mean(eq_Rgz))

    # ============================================================
    # FINAL STATISTICS ACROSS MPI RANKS
    # ============================================================

    rank_Ree = np.array(rank_Ree)

    rank_Rg = np.array(rank_Rg)

    rank_Rgx = np.array(rank_Rgx)
    rank_Rgy = np.array(rank_Rgy)
    rank_Rgz = np.array(rank_Rgz)

    # ------------------------------------------------------------
    # Final averages
    # ------------------------------------------------------------

    final_Ree = np.mean(rank_Ree)

    final_Rg = np.mean(rank_Rg)

    final_Rgx = np.mean(rank_Rgx)

    final_Rgy = np.mean(rank_Rgy)

    final_Rgz = np.mean(rank_Rgz)

    # ------------------------------------------------------------
    # Standard error across independent MPI trajectories
    # ------------------------------------------------------------

    err_Ree = np.std(rank_Ree, ddof=1) / np.sqrt(len(rank_Ree))

    err_Rg = np.std(rank_Rg, ddof=1) / np.sqrt(len(rank_Rg))

    err_Rgx = np.std(rank_Rgx, ddof=1) / np.sqrt(len(rank_Rgx))

    err_Rgy = np.std(rank_Rgy, ddof=1) / np.sqrt(len(rank_Rgy))

    err_Rgz = np.std(rank_Rgz, ddof=1) / np.sqrt(len(rank_Rgz))

    # ============================================================
    # PRINT FINAL RESULTS
    # ============================================================

    print("\n" + "="*70)

    print("FINAL EQUILIBRATED POLYMER STATISTICS")

    print("="*70)

    print()

    print(
        f"Ree_rms  = "
        f"{final_Ree:.6f} ± {err_Ree:.6f}"
    )

    print(
        f"Rg_rms   = "
        f"{final_Rg:.6f} ± {err_Rg:.6f}"
    )

    print()

    print("Directional anisotropy:")

    print()

    print(
        f"Rgx_rms  = "
        f"{final_Rgx:.6f} ± {err_Rgx:.6f}"
    )

    print(
        f"Rgy_rms  = "
        f"{final_Rgy:.6f} ± {err_Rgy:.6f}"
    )

    print(
        f"Rgz_rms  = "
        f"{final_Rgz:.6f} ± {err_Rgz:.6f}"
    )

    print()

    # ============================================================
    # PLOT TIME EVOLUTION
    # ============================================================

    plt.figure(figsize=(9,6))

    plt.plot(
        steps,
        Rg_vs_time,
        linewidth=2,
        label="Rg"
    )

    plt.plot(
        steps,
        Rgx_vs_time,
        "--",
        label="Rgx"
    )

    plt.plot(
        steps,
        Rgy_vs_time,
        "--",
        label="Rgy"
    )

    plt.plot(
        steps,
        Rgz_vs_time,
        "--",
        label="Rgz"
    )

    # ------------------------------------------------------------
    # Visual marker for equilibration region
    # ------------------------------------------------------------

    plt.axvline(
        EQUIL_START,
        linestyle=":",
        linewidth=2,
        label="equilibration cutoff"
    )

    plt.xlabel("Snapshot")

    plt.ylabel("Polymer RMS Size")

    plt.title("Polymer Radius of Gyration Evolution")

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ====================================================================
# PROGRAM ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    analyze()
