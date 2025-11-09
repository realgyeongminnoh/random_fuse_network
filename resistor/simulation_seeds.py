import os
import argparse
import numpy as np
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from src.array import Array
from src.matrix import Matrix
from src.equation import Equation
from src.failure import Failure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random Fuse Network - Resistor: multiple-seed Simulation")
    parser.add_argument("--length", "--l", type=int, required=True, help="Number of Vertical/Horizonatal Bonds per Column/Row >= 10 (recommended)")
    parser.add_argument("--width", "--w", type=float, required=True, help="Width of Random Uniform Distribution of Threshold Voltage Drops ∈ [0, 2]")
    parser.add_argument("--seed_min", "--smin", type=int, required=True, help="Smallest Seed Number of Random Uniform Distribution of Threshold Voltage Drops")
    parser.add_argument("--seed_max", "--smax", type=int, required=True, help="Largest Seed Number of Random Uniform Distribution of Threshold Voltage Drops")
    parser.add_argument("--save", action="store_true", help="[warning: time & memory] Collect and Save volts_edge for All t")
    parser.add_argument("--cpu", type=int, required=False, default=15, help="Force a Fixed Number of CPU Processors; default = 15")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (args.length >= 3):
        raise ValueError("length >= 3")
    if not (0 <= args.width <= 2):
        raise ValueError("width ∈ [0, 2]")
    if not (0 <= args.seed_min):
        raise ValueError("seed_min >= 0")
    if not (args.seed_min <= args.seed_max):
        raise ValueError("seed_min <= seed_max")
    if not (1 <= args.cpu):
        raise ValueError(f"1 <= cpu")


def get_output_dir_common(args: argparse.Namespace) -> Path:
    output_dir_common = Path(__file__).resolve().parent / "data" / str(args.length) / str(args.width)
    output_dir_common.mkdir(parents=True, exist_ok=True)

    for seed in range(args.seed_min, args.seed_max + 1):
        (output_dir_common / str(seed)).mkdir(parents=True, exist_ok=True)
    return output_dir_common


def save_result(output_dir: Path, failure: Failure) -> None:
    np.save(output_dir / "idxs_edge_broken.npy", np.array(failure.idxs_edge_broken, dtype=np.int32))
    np.save(output_dir / "idxs_edge_unalive.npy", np.array(failure.idxs_edge_unalive, dtype=np.int32))
    np.save(output_dir / "volts_ext.npy", np.array(failure.volts_ext, dtype=np.float64))
    if hasattr(failure, "volts_edge_profile"):
        np.save(output_dir / "volts_edge_profile.npy", np.array(failure.volts_edge_profile, dtype=np.float64))


def parallel_job(seed: int) -> None:
    # initialization - seed-specific
    matrix = Matrix(matrix_init=matrix_init_global, array=None)
    equation = Equation(array=array_global, matrix=matrix)
    failure = Failure(array=array_global, matrix=matrix, equation=equation, width=width_global, seed=seed, save_volts_profile=save_volts_profile_global)

    # breakdown simulation
    # 1st bond breaking [t=0]
    solve = equation.solve_init
    break_edge = failure.break_edge_init
    solve()
    break_edge()

    # 2nd ~ (length)th bond breaking [t=1 ~ t=(length-1)]
    check_graph = equation.check_graph
    solve = equation.solve_mmd
    break_edge = failure.break_edge
    for _ in range(length_global - 1):
        check_graph()
        solve()
        break_edge()

    # (length+1)th ~ (macroscopic failure) bond breaking [t=(length) ~ (total number of broken bonds-1)]
    solve = equation.solve_amd
    while check_graph():
        solve()
        break_edge()

    save_result(output_dir_common_global / str(seed), failure)


def main(length: int, width: float, seed_min: int, seed_max: int, save_volts_profile: bool, cpu: int) -> None:
    global length_global, width_global, save_volts_profile_global
    length_global, width_global, save_volts_profile_global = length, width, save_volts_profile

    # initialization - shared across seeds
    global array_global, matrix_init_global
    array_global = Array(length=length, mode_analysis=False)
    matrix_init_global = Matrix(matrix_init=None, array=array_global)
    
    # parallelization
    max_workers = (seed_max - seed_min + 1) if (seed_max - seed_min + 1) < cpu else cpu
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(parallel_job, range(seed_min, seed_max + 1))


if __name__=="__main__":
    if mp.get_start_method() != "fork":
        raise EnvironmentError("linux recommended for multiprocessing")
    
    args = parse_args()
    validate_args(args)
    output_dir_common_global = get_output_dir_common(args)

    main(args.length, args.width, args.seed_min, args.seed_max, args.save, args.cpu)
    raise SystemExit(0)