"""
Script determines polymer segment density in a specific z layer.
It reads position files for all beads of a polymer.
Uses z position of the beads to determine number of beads in a z layer.
Segment density determined from the number of beads in a layer and xy area.
"""
#====================================================================
# IMPORT LIBRARIES
#====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ====================================================================
# USER SETTINGS
# ====================================================================

# Directory containing all CSV trajectory files
DATA_DIR = Path.cwd()
# ------------------------------------------------------------
# Lattice settings
# ------------------------------------------------------------
#XYZ dimensions of the simulation box
LX=7
LY=7
LZ=20
factor=LX*LY

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

EQUIL_START = 0
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

	echains.3.0125.csv
	"""

	filename = f"echains.{rank}.{fn}.csv"

	path = DATA_DIR / filename

	if path.exists():
		return path

	return None

# ====================================================================
# COMPUTE SEGMENT DENSITY IN A Z LAYER
# ====================================================================

def comp_seg_dens_z(df):
	"""
	Compute system segment density in ONE snapshot file.

	Returns:
	Segment density for every z layer
	"""
	ndz=(
		df["z"]
		.value_counts()
		.reindex(range(0,LZ),fill_value=0)
		.tolist()
	)
	sdz1=np.array(ndz)/factor
	sdz2=np.array(ndz[::-1])/factor
	sdz=(sdz1+sdz2)/2
	return sdz

def analyze():

    # ============================================================
    # ARRAYS FOR FINAL RANK-WISE STATISTICS
    # ============================================================

    rank_sdz = []

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
        rank_time_sdz = []
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

            expected_rows = NC*BC

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
            # Compute segment density
            # ----------------------------------------------------

            sdz = comp_seg_dens_z(df)
        
	    # ----------------------------------------------------
	    # Store time evolution
            # ----------------------------------------------------
            rank_time_sdz.append(sdz)

	# ========================================================
        # DISCARD NON-EQUILIBRATED REGION
        # ========================================================
        eq_sdz = np.array(rank_time_sdz[EQUIL_START:])
	
        # --------------------------------------------------------
        # Compute equilibrium mean for THIS rank
        # -------------------------------------------------------- 
        rank_sdz.append(np.mean(eq_sdz,axis=0))
  
    # ============================================================
    # FINAL STATISTICS ACROSS MPI RANKS
    # ============================================================
    sdz_avg=np.mean(rank_sdz,axis=0)
    sdz_err=np.std(rank_sdz,axis=0,ddof=1)/np.sqrt(len(rank_sdz))
    # ============================================================
    # PRINT FINAL RESULTS
    # ============================================================

    print("\n" + "="*70)

    print("FINAL EQUILIBRATED POLYMER STATISTICS")

    print("="*70)

    print()

    index=np.arange(0.5,20.5,1)
    odf=pd.DataFrame({
	"z":index,
        "rho":sdz_avg,
        "err":sdz_err
    })
    odf.to_csv('RhoExVol.csv',index=False)
    # ============================================================
    # PLOT SEGMENT DENSITY
    # ============================================================

    odf.plot(x='z',y='rho',yerr="err",capsize=4)
  
    plt.ylim(0.3,0.9)
    
    plt.axhline(y=0.735, color='r', linestyle='--') 
    
    plt.xlabel("z layer")

    plt.ylabel("segment density")

    plt.title("Layer Segment Density")

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.show()

# ====================================================================
# PROGRAM ENTRY POINT
# ====================================================================

if __name__ == "__main__":

	analyze()
