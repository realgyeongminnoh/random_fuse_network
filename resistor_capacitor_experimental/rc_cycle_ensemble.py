#!/usr/bin/env python3
"""
RC charge↔discharge ensemble runner (experimental variant).

- Minimal I/O:
  * For each width: saves `num_cycle.npy` (int32, shape=(seeds,)) in the width folder.
  * For each seed: saves `volts_ext.npy` (float64) in width/SEED/.
- Default output root: data/
- Parallelization: per-width, seeds distributed to up to --cpus workers.
"""
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Allow "src" imports when running from repo root
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from src.array import Array
from src.matrix_charge import Matrix as Matrix_charge
from src.matrix_discharge import Matrix as Matrix_discharge
from src.equation import Equation
from src.failure import Failure


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RFN RC charge↔discharge ensemble (experimental)")
    ap.add_argument("--length", type=int, required=True, help="L (>=3)")
    ap.add_argument("--val-cap", type=float, required=True, help="capacitance value")
    ap.add_argument("--time-step", type=float, required=True, help="time step")
    ap.add_argument("--cond-ext", type=float, required=True, help="external conductance for discharge")
    ap.add_argument("--widths", type=float, nargs="+", required=True, help="list of width values")
    ap.add_argument("--seeds", type=int, required=True, help="number of seeds (0..seeds-1)")
    ap.add_argument("--cpus", type=int, default=1, help="max workers")
    ap.add_argument("--out", type=str, default="data", help="output root directory (default: data)")
    # profiles are heavy; keep them off by default
    ap.add_argument("--save-volts-profile", action="store_true", help="save heavy voltage profiles (off by default)")
    return ap.parse_args()


def simulate_one(seed: int, length: int, width: float, val_cap: float, time_step: float,
                 cond_ext: float, save_volts_profile: bool, out_width: Path) -> int:
    """
    Run one seed and save only volts_ext (per seed). Return num_cycles.
    """
    # --- setup ---
    array = Array(length=length, mode_analysis=False)
    matrix_ch = Matrix_charge(matrix_init=None, array=array, val_cap=val_cap, time_step=time_step)
    matrix_dch = Matrix_discharge(matrix_init=None, array=array, cond_ext=cond_ext, val_cap=val_cap, time_step=time_step)
    equation = Equation(array=array, matrix_ch=matrix_ch, matrix_dch=matrix_dch, save_volts_profile=save_volts_profile)
    failure = Failure(array=array, matrix_ch=matrix_ch, matrix_dch=matrix_dch, equation=equation,
                      width=width, seed=seed, save_volts_profile=save_volts_profile)

    solve_r = equation.solve_r_amd

    # --- cycle runners ---
    def run_charge_cycle_first(duration_cycle: int) -> bool:
        solve = equation.solve_true_init_ch
        break_edge = failure.break_edge_init_ch
        solve(); break_edge()  # t=0

        solve = equation.solve_ch
        break_edge = failure.break_edge_ch
        for _ in range(duration_cycle - 1):
            if not solve_r():
                return False
            solve(); break_edge()
        return True

    def run_charge_cycle_from_dch(duration_cycle: int) -> bool:
        if not solve_r(): return False

        solve = equation.solve_init_ch
        break_edge = failure.break_edge_ch
        solve(); break_edge()  # t=0 after D→C

        solve = equation.solve_ch
        break_edge = failure.break_edge_ch
        for _ in range(duration_cycle - 1):
            if not solve_r():
                return False
            solve(); break_edge()
        return True

    def run_discharge_cycle(duration_cycle: int) -> bool:
        if not solve_r(): return False

        solve = equation.solve_init_dch
        break_edge = failure.break_edge_dch
        solve(); break_edge()  # t=0 for C→D

        solve = equation.solve_dch
        break_edge = failure.break_edge_dch
        for _ in range(duration_cycle - 1):
            if not solve_r():
                return False
            solve(); break_edge()
        return True

    duration_charge = array.length // 2
    duration_discharge = array.length // 2

    ok = run_charge_cycle_first(duration_charge)
    num_cycles = 0
    while ok:
        ok = run_discharge_cycle(duration_discharge) and run_charge_cycle_from_dch(duration_charge)
        num_cycles += 1

    # --- save per-seed volts_ext ---
    seed_dir = out_width / f"{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    np.save(seed_dir / "volts_ext.npy", np.asarray(failure.volts_ext, dtype=np.float64))

    return int(num_cycles)


def run_width(width: float, args: argparse.Namespace) -> None:
    # width directory: out_root/{valcap}_{tstep}/{length}/{width}/
    out_root = Path(args.out)
    cap_time_dir = out_root / f"{float(args.val_cap)}_{float(args.time_step)}"
    out_width = cap_time_dir / f"{float(width):.2f}"
    out_width.mkdir(parents=True, exist_ok=True)

    n = int(args.seeds)
    num_cycles = np.zeros(n, dtype=np.int32)

    max_workers = min(args.cpus, n) if args.cpus > 0 else 1
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                simulate_one,
                seed, args.length, width,
                args.val_cap, args.time_step, args.cond_ext,
                bool(args.save_volts_profile), out_width
            ): seed for seed in range(n)
        }
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                num_cycles[s] = fut.result()
            except Exception as e:
                # If a seed fails, reflect it with -1
                num_cycles[s] = -1

    # Save aggregated num_cycle per width
    np.save(out_width / "num_cycle.npy", num_cycles)


def main():
    args = parse_args()

    # loop widths sequentially; seeds run in parallel
    for w in args.widths:
        run_width(float(w), args)


if __name__ == "__main__":
    main()
