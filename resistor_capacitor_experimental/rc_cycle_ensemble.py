import warnings
from scipy.sparse.linalg import MatrixRankWarning

# rc_cycle_ensemble.py
import argparse
import numpy as np
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from src.array import Array
from src.matrix_charge import Matrix as Matrix_charge
from src.matrix_discharge import Matrix as Matrix_discharge
from src.equation import Equation
from src.failure import Failure

# -------------------- helpers --------------------
def validate_args(args: argparse.Namespace) -> None:
    if not (args.length >= 3): raise ValueError("length >= 3")
    if not (0 <= args.cond_ext): raise ValueError("cond_ext >= 0")
    if not (args.time_step > 0): raise ValueError("time_step > 0")
    if not (args.cpu >= 1): raise ValueError("cpu >= 1")

def get_output_dir(base: Path, val_cap: float, time_step: float,
                   length: int, width: float, seed: int) -> Path:
    out = base / f"{val_cap}_{time_step}" / str(length) / str(width) / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    return out

def count_broken_by_cycle(failure: Failure, duration_charge: int, duration_discharge: int):
    t = np.asarray(failure.idxs_time_edge_broken, dtype=np.int64)
    if t.size == 0:
        return 0, 0
    period = duration_charge + duration_discharge
    r = t % period
    is_ch = r < duration_charge
    return int(is_ch.sum()), int((~is_ch).sum())

def run_charge_cycle_first(equation: Equation, failure: Failure, duration_cycle: int, solve_r) -> bool:
    solve = equation.solve_true_init_ch
    break_edge = failure.break_edge_init_ch
    solve(); break_edge()
    solve = equation.solve_ch
    break_edge = failure.break_edge_ch
    for _ in range(duration_cycle - 1):
        if not solve_r(): return False
        solve(); break_edge()
    return True

def run_charge_cycle_from_dch(equation: Equation, failure: Failure, duration_cycle: int, solve_r) -> bool:
    if not solve_r():                            # ← guard before init
        return False
    solve = equation.solve_init_ch
    break_edge = failure.break_edge_ch
    solve(); break_edge()                        # t=0 after D→C (caps nonzero)

    solve = equation.solve_ch
    break_edge = failure.break_edge_ch
    for _ in range(duration_cycle - 1):
        if not solve_r(): return False
        solve(); break_edge()
    return True

def run_discharge_cycle(equation: Equation, failure: Failure, duration_cycle: int, solve_r) -> bool:
    if not solve_r():                            # ← guard before init
        return False
    solve = equation.solve_init_dch
    break_edge = failure.break_edge_dch
    solve(); break_edge()                        # t=0 for C→D

    solve = equation.solve_dch
    break_edge = failure.break_edge_dch
    for _ in range(duration_cycle - 1):
        if not solve_r(): return False
        solve(); break_edge()
    return True

# -------------------- globals filled in main() and used by workers --------------------
array_global = None
matrix_init_ch_global = None
matrix_init_dch_global = None
duration_charge_global = None
duration_discharge_global = None
val_cap_global = None
time_step_global = None
cond_ext_global = None
save_volts_profile_global = None
output_base_global = None

SEED_STRIDE = 1_000_000
MAX_ATTEMPTS = 25  # bump if you want

def run_one_attempt(width: float, seed: int):
    # build per-attempt objects using shared array/matrices
    matrix_ch = Matrix_charge(matrix_init=matrix_init_ch_global, array=None)
    matrix_dch = Matrix_discharge(matrix_init=matrix_init_dch_global, array=None)
    equation = Equation(array=array_global, matrix_ch=matrix_ch, matrix_dch=matrix_dch)
    failure = Failure(array=array_global, matrix_ch=matrix_ch, matrix_dch=matrix_dch,
                      equation=equation, width=float(width), seed=int(seed),
                      save_volts_profile=save_volts_profile_global)

    solve_r = equation.solve_r_amd

    if run_charge_cycle_first(equation, failure, duration_charge_global, solve_r):
        while run_discharge_cycle(equation, failure, duration_discharge_global, solve_r) and \
              run_charge_cycle_from_dch(equation, failure, duration_charge_global, solve_r):
            pass

    num_broken_total = int(len(failure.idxs_edge_broken))
    num_ch, num_dch = count_broken_by_cycle(failure, duration_charge_global, duration_discharge_global)
    cycles_endured = int(failure.counter_time_step // (duration_charge_global + duration_discharge_global))
    volts_ext = np.asarray(failure.volts_ext, dtype=np.float64)

    return {
        "width": float(width),
        "seed": int(seed),  # actual seed used in this successful attempt
        "volts_ext": volts_ext,
        "num_broken_total": num_broken_total,
        "num_broken_charge": num_ch,
        "num_broken_discharge": num_dch,
        "cycles_endured": cycles_endured,
    }

def run_one_with_retry(width: float, base_seed: int):
    # promote singular KKT to exception so we can retry
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        attempt = 0
        while attempt < MAX_ATTEMPTS:
            seed = int(base_seed + attempt * SEED_STRIDE)
            try:
                rec = run_one_attempt(width, seed)
                # save under the actual seed used
                outdir = get_output_dir(output_base_global, val_cap_global, time_step_global,
                                        array_global.length, width, seed)
                np.savez_compressed(outdir / "summary.npz", **rec)
                return
            except MatrixRankWarning:
                attempt += 1
                continue
        # if we get here, we failed too many times — skip this task silently
        return

def parallel_job(task):
    width, base_seed = task
    run_one_with_retry(width, base_seed)

# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser(description="RC cycle ensemble (shared Array & init matrices)")
    parser.add_argument("--length", type=int, default=30)
    parser.add_argument("--val_cap", type=float, default=0.01)
    parser.add_argument("--time_step", type=float, default=0.01)
    parser.add_argument("--cond_ext", type=float, default=100.0)
    parser.add_argument("--widths", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--cpu", type=int, default=15)
    parser.add_argument("--save_volts_profile", action="store_true")
    parser.add_argument("--output_base", type=Path, default=Path(__file__).resolve().parent / "data_rc_cycle")
    args = parser.parse_args()

    validate_args(args)

    if mp.get_start_method() != "fork":
        raise EnvironmentError("linux/fork start method recommended for zero-copy sharing")

    # shared across workers (copy-on-write via fork)
    global array_global, matrix_init_ch_global, matrix_init_dch_global
    global duration_charge_global, duration_discharge_global
    global val_cap_global, time_step_global, cond_ext_global, save_volts_profile_global, output_base_global

    array_global = Array(length=args.length, mode_analysis=False)
    matrix_init_ch_global  = Matrix_charge(matrix_init=None, array=array_global,
                                           val_cap=args.val_cap, time_step=args.time_step)
    matrix_init_dch_global = Matrix_discharge(matrix_init=None, array=array_global,
                                              cond_ext=args.cond_ext, val_cap=args.val_cap, time_step=args.time_step)

    duration_charge_global    = args.length // 2
    duration_discharge_global = args.length // 2
    val_cap_global = args.val_cap
    time_step_global = args.time_step
    cond_ext_global = args.cond_ext
    save_volts_profile_global = args.save_volts_profile
    output_base_global = args.output_base
    output_base_global.mkdir(parents=True, exist_ok=True)

    # task list: 4 widths × N seeds
    tasks = [(w, s) for w in args.widths for s in range(args.seeds)]

    max_workers = min(args.cpu, len(tasks))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for _ in ex.map(parallel_job, tasks):
            pass

if __name__ == "__main__":
    main()
