import numpy as np
import pandas as pd
import os
from mpi4py import MPI

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cwd  = os.getcwd()
path = cwd

#==============================================================================
# BOX DIMENSIONS — required for PBC unwrapping in all three directions
#==============================================================================
Lx = 20.0
Ly = 20.0
Lz = 20.0
L  = np.array([Lx, Ly, Lz])

# COM bins
#z_com_list = [0.5,1.5,2.5,5.5,8.5]
z_com_list = []
for tar in range(0,10):
    z_com_list.append(tar+0.5)

dz         = 1.0

TotalBeads = 20
nc         = 36       # number of chains per snapshot
nfiles     = 1000

Nblocks    = 20
block_size = nfiles // Nblocks

print(f"Rank {rank} running")

#==============================================================================
# UNWRAP FUNCTION — minimum image convention for a single chain
# Applies PBC unwrapping in all three directions (x, y, z).
# coords : numpy array of shape (TotalBeads, 3), wrapped coordinates
# returns: numpy array of shape (TotalBeads, 3), unwrapped coordinates
#==============================================================================
def unwrap_chain(coords, L):
    unwrapped = coords.copy().astype(float)
    for i in range(1, len(unwrapped)):
        delta = unwrapped[i] - unwrapped[i - 1]
        delta -= L * np.round(delta / L)
        unwrapped[i] = unwrapped[i - 1] + delta
    return unwrapped

# LOOP OVER EACH z_com TARGET BIN
for target in z_com_list:

    print(f"\nRank {rank}: Processing zCOM = {target}")

    block_means = []

    # BLOCK LOOP
    for b in range(Nblocks):

        block_sum   = np.zeros(TotalBeads)
        block_count = 0

        jstart = b * block_size + 1
        jend   = jstart + block_size

        # FILE LOOP
        for j in range(jstart, jend):

            #------------------------------------------------------------------
            # Read echains file — single input, no intermediate files needed
            # Format: echains.{rank}.{fn}.csv with columns x, y, z
            #------------------------------------------------------------------
            fname = os.path.join(path, f"echains.{rank}.{j}.csv")

            if not os.path.exists(fname):
                continue

            pos = pd.read_csv(fname, dtype={"x": float, "y": float, "z": float})
            coords_all = pos[["x", "y", "z"]].to_numpy(dtype=float)

            # LOOP OVER ALL CHAINS IN THIS SNAPSHOT
            for c in range(nc):

                start = TotalBeads * c
                end   = start + TotalBeads

                coords_wrapped = coords_all[start:end]

                #--------------------------------------------------------------
                # UNWRAP in all three directions using minimum image convention
                #--------------------------------------------------------------
                coords_unwrapped = unwrap_chain(coords_wrapped, L)

                #--------------------------------------------------------------
                # Compute z-COM from unwrapped coordinates
                # Re-wrap into [0, Lz) for COM binning
                #--------------------------------------------------------------
                z_unwrapped = coords_unwrapped[:, 2]
                z_com       = np.mean(z_unwrapped) % Lz

                # Skip chain if its z-COM does not fall in target bin
                if abs(z_com - target) >= dz / 2:
                    continue

                #--------------------------------------------------------------
                # Re-wrap z beads into [0, Lz) for density binning
                #--------------------------------------------------------------
                z_rewrapped = z_unwrapped % Lz

                hits = np.zeros(TotalBeads)
                for zval in z_rewrapped:
                    bin_index = int(np.floor(zval))
                    bin_index = min(max(bin_index, 0), TotalBeads - 1)
                    hits[bin_index] += 1

                # Normalize by chain length to get per-bead density
                hits /= TotalBeads

                block_sum   += hits
                block_count += 1

        if block_count > 0:
            block_means.append(block_sum / block_count)

    block_means = np.array(block_means)

    # MPI GATHER — collect block means from all ranks onto rank 0
    all_blocks = comm.gather(block_means, root=0)

    # FINAL STATISTICS
    if rank == 0:

        valid_blocks = []
        for blk in all_blocks:
            if len(blk) > 0:
                valid_blocks.append(blk)

        if len(valid_blocks) == 0:
            print(f"No samples for zCOM={target}")
            continue

        valid_blocks = np.vstack(valid_blocks)

        mean   = np.mean(valid_blocks, axis=0)
        std    = np.std(valid_blocks, axis=0, ddof=1)
        stderr = std / np.sqrt(valid_blocks.shape[0])

        # SAVE OUTPUT — one CSV per z_com target bin
        df = pd.DataFrame({
            "z"      : np.arange(TotalBeads)+0.5,
            "mean_density" : mean,
            "std_error": stderr
        })

        OUTPUT_FILE = os.path.join(path, f"prob_density_zcom_{target:.1f}.csv")
        df.to_csv(OUTPUT_FILE, index=False, float_format="%.4f")
        print("Saved:", OUTPUT_FILE)
