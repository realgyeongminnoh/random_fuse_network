import argparse
import numpy as np
from pathlib import Path

from src.array import Array
from src.matrix import Matrix
from src.equation import Equation
from src.failure import Failure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random Fuse Network - Resistor: Single-seed Simulation")
    parser.add_argument("--length", "--l", type=int, required=True, help="Number of Vertical/Horizonatal Bonds per Column/Row >= 10 (recommended)")
    parser.add_argument("--width", "--w", type=float, required=True, help="Width of Random Uniform Distribution of Threshold Voltage Drops ∈ [0, 2]")
    parser.add_argument("--seed", "--s", type=int, required=True, help="Seed Number of Random Uniform Distribution of Threshold Voltage Drops")
    parser.add_argument("--save", action="store_true", help="Collect and Save volts_edge for All t")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (args.length >= 3):
        raise ValueError("length >= 3")
    if not (0 <= args.width <= 2):
        raise ValueError("width ∈ [0, 2]")
    if not (0 <= args.seed):
        raise ValueError("seed >= 0")


def get_output_dir(args: argparse.Namespace) -> Path | None:
    output_dir = Path(__file__).resolve().parent / "data" / "0.0" / str(args.length) / str(args.width) / str(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if (output_dir / "idxs_edge_broken.npy").exists() and (output_dir / "idxs_edge_unalive").exists() and (output_dir / "volts_ext.npy").exists():
        if args.save:
            if (output_dir / "volts_edge_profile.npy").exists():
                return None
            else:
                return output_dir
        return None
    return output_dir


def save_result(output_dir: Path, failure: Failure) -> None:
    np.save(output_dir / "idxs_edge_broken.npy", np.array(failure.idxs_edge_broken, dtype=np.int32))
    np.save(output_dir / "idxs_edge_unalive.npy", np.array(failure.idxs_edge_unalive, dtype=np.int32))
    np.save(output_dir / "volts_ext.npy", np.array(failure.volts_ext, dtype=np.float64))
    if hasattr(failure, "volts_edge_profile"):
        np.save(output_dir / "volts_edge_profile.npy", np.array(failure.volts_edge_profile, dtype=np.float64))


def main(length: int, width: float, seed: int, save_volts_profile: bool) -> None:
    # initialization
    array = Array(length=length, mode_analysis=False)
    matrix = Matrix(matrix_init=None, array=array)
    equation = Equation(array=array, matrix=matrix)
    failure = Failure(array=array, matrix=matrix, equation=equation, width=width, seed=seed, save_volts_profile=save_volts_profile)

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
    for _ in range(array.length - 1):
        check_graph()
        solve()
        break_edge()

    # (length+1)th ~ (macroscopic failure) bond breaking [t=(length) ~ (total number of broken bonds-1)]
    solve = equation.solve_amd
    while check_graph():
        solve()
        break_edge()

    save_result(output_dir_global, failure)


if __name__=="__main__":
    args = parse_args()
    validate_args(args)
    output_dir_global = get_output_dir(args)
    if output_dir_global is None: raise SystemExit("[simulation_seed.py] the result already exists")

    main(args.length, args.width, args.seed, args.save)
    raise SystemExit(0)