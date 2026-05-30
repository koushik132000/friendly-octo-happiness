import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================
# USER SETTINGS
# ==============================================================

DATA_DIR = Path.cwd()

NC = 36          # chains per rank
BC = 20          # beads per chain
NPROC = 7        # MPI ranks
FN_MAX = 1000    # max snapshot
EQUIL_START = 25 # burn-in cutoff

PRINT_EVERY = 100

FILE_PATTERNS = [
    "echains.{rank}.{fn}.csv",
    "mchains{fn}.csv",
]

# ==============================================================
# FIND SNAPSHOT FILE
# ==============================================================

def find_snapshot(fn, rank):

    for pat in FILE_PATTERNS:

        try:
            p = DATA_DIR / pat.format(fn=fn, rank=rank)
        except (KeyError, IndexError):
            continue

        if p.exists():
            return p

    return None

# ==============================================================
# COMPUTE CHAIN METRICS
# ==============================================================

def compute_chain_metrics(df):

    Ree2 = np.empty(NC)

    Rg2  = np.empty(NC)

    Rg2x = np.empty(NC)
    Rg2y = np.empty(NC)
    Rg2z = np.empty(NC)

    # ----------------------------------------------------------

    for c in range(NC):

        pts = df.iloc[
            c*BC:(c+1)*BC
        ][["x","y","z"]].values.astype(float) + 0.5

        # ------------------------------------------------------
        # Ree²
        # ------------------------------------------------------

        dr = pts[-1] - pts[0]

        Ree2[c] = np.dot(dr, dr)

        # ------------------------------------------------------
        # Rg²
        # ------------------------------------------------------

        r_cm = pts.mean(axis=0)

        disp = pts - r_cm

        sq = disp**2

        Rg2[c] = np.mean(np.sum(sq, axis=1))

        comp = np.mean(sq, axis=0)

        Rg2x[c] = comp[0]
        Rg2y[c] = comp[1]
        Rg2z[c] = comp[2]

    return Ree2, Rg2, Rg2x, Rg2y, Rg2z

# ==============================================================
# RMS + propagated error
# ==============================================================

def rms_and_error(data):

    mean2 = np.mean(data)

    rms = np.sqrt(mean2)

    std2 = np.std(data, ddof=1)

    sem2 = std2 / np.sqrt(len(data))

    err_rms = sem2 / (2*rms)

    return rms, err_rms

# ==============================================================
# MAIN ANALYSIS
# ==============================================================

def analyze():

    # ----------------------------------------------------------
    # store rank-wise equilibrium means
    # ----------------------------------------------------------

    rank_Rg = []

    rank_Rgx = []
    rank_Rgy = []
    rank_Rgz = []

    # ----------------------------------------------------------
    # for plotting
    # ----------------------------------------------------------

    steps = []

    Rg_vs_time = []

    Rgx_vs_time = []
    Rgy_vs_time = []
    Rgz_vs_time = []

    # ==========================================================
    # LOOP OVER RANKS
    # ==============================================================

    for rank in range(NPROC):

        print(f"\nProcessing rank {rank}")

        # ------------------------------------------------------
        # time series for THIS rank
        # ------------------------------------------------------

        rank_time_Rg = []

        rank_time_Rgx = []
        rank_time_Rgy = []
        rank_time_Rgz = []

        # ------------------------------------------------------
        # LOOP OVER SNAPSHOTS
        # ------------------------------------------------------

        for fn in range(1, FN_MAX + 1):

            file = find_snapshot(fn, rank)

            if file is None:
                continue

            df = pd.read_csv(file)

            expected_rows = NC * BC

            if len(df) != expected_rows:

                print(
                    f"WARNING: {file.name} "
                    f"has wrong size"
                )

                continue

            Ree2, Rg2, Rg2x, Rg2y, Rg2z = \
                compute_chain_metrics(df)

            # --------------------------------------------------
            # CHAIN ENSEMBLE RMS
            # --------------------------------------------------

            Rg_rms, _ = rms_and_error(Rg2)

            Rgx_rms, _ = rms_and_error(Rg2x)

            Rgy_rms, _ = rms_and_error(Rg2y)

            Rgz_rms, _ = rms_and_error(Rg2z)

            # --------------------------------------------------
            # STORE TIME SERIES
            # --------------------------------------------------

            rank_time_Rg.append(Rg_rms)

            rank_time_Rgx.append(Rgx_rms)

            rank_time_Rgy.append(Rgy_rms)

            rank_time_Rgz.append(Rgz_rms)

            # --------------------------------------------------
            # global plotting arrays
            # only from rank 0
            # --------------------------------------------------

            if rank == 0:

                steps.append(fn)

                Rg_vs_time.append(Rg_rms)

                Rgx_vs_time.append(Rgx_rms)

                Rgy_vs_time.append(Rgy_rms)

                Rgz_vs_time.append(Rgz_rms)

            # --------------------------------------------------

            if PRINT_EVERY > 0 and fn % PRINT_EVERY == 0:

                print(
                    f"Rank {rank} "
                    f"snapshot {fn} "
                    f"Rg = {Rg_rms:.4f}"
                )

        # ======================================================
        # EQUILIBRATED RANK MEAN
        # ======================================================

        eq_Rg = np.array(rank_time_Rg[EQUIL_START:])

        eq_Rgx = np.array(rank_time_Rgx[EQUIL_START:])

        eq_Rgy = np.array(rank_time_Rgy[EQUIL_START:])

        eq_Rgz = np.array(rank_time_Rgz[EQUIL_START:])

        rank_Rg.append(np.mean(eq_Rg))

        rank_Rgx.append(np.mean(eq_Rgx))

        rank_Rgy.append(np.mean(eq_Rgy))

        rank_Rgz.append(np.mean(eq_Rgz))

    # ==========================================================
    # FINAL STATISTICS ACROSS INDEPENDENT RANKS
    # ==============================================================

    rank_Rg = np.array(rank_Rg)

    rank_Rgx = np.array(rank_Rgx)
    rank_Rgy = np.array(rank_Rgy)
    rank_Rgz = np.array(rank_Rgz)

    # ----------------------------------------------------------

    final_Rg = np.mean(rank_Rg)

    err_Rg = np.std(rank_Rg, ddof=1) / np.sqrt(len(rank_Rg))

    # ----------------------------------------------------------

    final_Rgx = np.mean(rank_Rgx)

    err_Rgx = np.std(rank_Rgx, ddof=1) / np.sqrt(len(rank_Rgx))

    # ----------------------------------------------------------

    final_Rgy = np.mean(rank_Rgy)

    err_Rgy = np.std(rank_Rgy, ddof=1) / np.sqrt(len(rank_Rgy))

    # ----------------------------------------------------------

    final_Rgz = np.mean(rank_Rgz)

    err_Rgz = np.std(rank_Rgz, ddof=1) / np.sqrt(len(rank_Rgz))

    # ==========================================================
    # PRINT RESULTS
    # ==============================================================

    print("\n" + "="*60)

    print("FINAL EQUILIBRATED RESULTS")

    print("="*60)

    print()

    print(
        f"Rg_rms  = "
        f"{final_Rg:.6f} ± {err_Rg:.6f}"
    )

    print(
        f"Rgx_rms = "
        f"{final_Rgx:.6f} ± {err_Rgx:.6f}"
    )

    print(
        f"Rgy_rms = "
        f"{final_Rgy:.6f} ± {err_Rgy:.6f}"
    )

    print(
        f"Rgz_rms = "
        f"{final_Rgz:.6f} ± {err_Rgz:.6f}"
    )

    print()

    # ==========================================================
    # PLOT
    # ==============================================================

    plt.figure(figsize=(8,6))

    plt.plot(
        steps,
        Rg_vs_time,
        label="Rg",
        linewidth=2
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

    plt.axvline(
        EQUIL_START,
        linestyle=":",
        label="equilibration"
    )

    plt.xlabel("Snapshot")

    plt.ylabel("RMS size")

    plt.title("Radius of Gyration vs Time")

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.show()

# ==============================================================
# RUN
# ==============================================================

if __name__ == "__main__":

    analyze()
