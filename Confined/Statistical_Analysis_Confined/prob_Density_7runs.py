#===============================================================================
# Computes the conditional bead density profile for polymer chains
# whose z-center of mass (z-COM) falls in a specific z-layer.
#
# For each target z-COM layer:
# 1) Read echains bead coordinates for each snapshot
# 2) Compute z-COM for each chain
# 3) Select chains whose z-COM falls within the target layer
# 4) For selected chains, count how many beads are in each z-layer
# 5) Average over all snapshots and ranks
# 6) Write one output CSV per target z-COM layer
#
# Input  : echains.{rank}.{fn}.csv  — wrapped bead coordinates
# Output : prob_density_zcom_{target}.csv  — one file per target z-COM
#================================================================================

import numpy as np
import pandas as pd
import os
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cwd = os.getcwd()
print("Rank", rank, "cwd:", cwd)

ipname = 'echains'

Lz  = 20.0   # box height
B   = 20     # number of z-layers
dz  = 1.0    # layer width
nc  = 36     # chains per rank
bc  = 20     # beads per chain

FN_MAX      = 1000
EQUIL_START = 25

#z_com_targets = [0.5, 1.5, 2.5, 5.5, 8.5]
z_com_targets=[]
for tar in range(0,10):
    z_com_targets.append(tar+0.5)

for target in z_com_targets:

    print(f"\nProcessing target z-COM = {target}")

    #Collect snapshot-wise mean bead density for each rank
    rank_means = []

    for r in range(size):

        #Collect bead density profiles from all snapshots for this rank
        snapshot_profiles = []

        for i in range(1, FN_MAX + 1):

            infname = ipname + '.' + str(r) + '.' + str(i) + '.csv'

            if not os.path.exists(infname):
                continue

            pos = pd.read_csv(infname, dtype={"x": float, "y": float, "z": float})

            #Translate z coordinates by +0.5 to center bead in lattice cell
            zs = pos['z'].values + 0.5

            #Initialize bead count array for this snapshot
            bead_counts = np.zeros(B)
            chain_count = 0

            #Loop over all chains
            for c in range(nc):

                #Extract z coordinates for this chain
                z_beads = zs[c * bc : (c + 1) * bc]

                #Compute z-COM for this chain
                z_com = np.mean(z_beads)

                #Select chain only if z-COM is within 0.5 of the target layer
                if abs(z_com - target) >= dz / 2:
                    continue

                #Count beads in each z-layer for this selected chain
                for zval in z_beads:
                    bin_index = int(zval // dz)
                    bin_index = min(max(bin_index, 0), B - 1)
                    bead_counts[bin_index] += 1

                chain_count += 1

            #Skip snapshot if no chains were selected for this target
            if chain_count == 0:
                continue

            #Normalize bead counts by number of selected chains and beads per chain
            bead_density = bead_counts / (chain_count * bc)
            snapshot_profiles.append(bead_density)

        #Skip rank if no snapshots had selected chains
        if len(snapshot_profiles) == 0:
            print(f"  No chains found for rank {r} at z-COM = {target}")
            continue

        #Average over all snapshots after burn-in for this rank
        snapshot_profiles = np.array(snapshot_profiles)   # shape: (nsnaps, 20)
        eq_profiles       = snapshot_profiles[EQUIL_START:]
        rank_avg          = np.mean(eq_profiles, axis=0)
        rank_means.append(rank_avg)

    #Skip target if no data found
    if len(rank_means) == 0:
        print(f"No data found for z-COM = {target}")
        continue

    #Final mean and standard error across ranks
    rank_means = np.array(rank_means)   # shape: (num_ranks, 20)
    rho_mean   = np.mean(rank_means, axis=0)
    rho_std    = np.std(rank_means, axis=0, ddof=1)
    rho_err    = rho_std / np.sqrt(len(rank_means))

    #z-bin centers
    z_centers = np.arange(B) + 0.5

    #Write output CSV for this target z-COM
    outname = f"prob_density_zcom_{target:.1f}.csv"
    with open(outname, "w") as f:
        f.write("z,mean_density,std_error\n")
        for i in range(B):
            f.write(f"{z_centers[i]:.2f},{rho_mean[i]:.6f},{rho_err[i]:.6f}\n")

    print(f"  Saved: {outname}")

print("\nDone.")
