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
    parser.add_argument("--save", action="store_true", help="Collect and Save volts_edge, volts_cap, and volts_cond for All t")
    parser.add_argument("--cap", type=float, required=False, default=1.0, help="Set Capacitance of All Capacitors; Default = 1.0")
    parser.add_argument("--time", type=float, required=False, default=0.01, help="Set Time Step; Default = 0.01")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (args.length >= 3):
        raise ValueError("length >= 3")
    if not (0 <= args.width <= 2):
        raise ValueError("width ∈ [0, 2]")
    if not (0 <= args.seed):
        raise ValueError("seed >= 0")
    if not (0 <= args.cap):
        raise ValueError("cap >= 0")
    if not (0 < args.time):
        raise ValueError("time > 0")


def get_output_dir(args: argparse.Namespace) -> Path | None:
    output_dir = Path(__file__).resolve().parent / "data" / str(args.length) / str(args.width) / f"{args.cap}_{args.time}" / str(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if (output_dir / "idxs_edge_broken.npy").exists() and (output_dir / "volts_ext.npy").exists():
        if args.save:
            if (output_dir / "volts_edge_profile.npy").exists() and (output_dir / "volts_cap_profile.npy") and (output_dir / "volts_cond_profile.npy"):
                return None
            else:
                return output_dir
        return None
    return output_dir


def save_result(output_dir: Path, failure: Failure) -> None:
    np.save(output_dir / "idxs_edge_broken.npy", np.array(failure.idxs_edge_broken, dtype=np.int32))
    np.save(output_dir / "volts_ext.npy", np.array(failure.volts_ext, dtype=np.float64))
    if hasattr(failure, "volts_edge_profile"):
        np.save(output_dir / "volts_edge_profile.npy", np.array(failure.volts_edge_profile, dtype=np.float64))
        np.save(output_dir / "volts_cap_profile.npy", np.array(failure.volts_cap_profile, dtype=np.float64))
        np.save(output_dir / "volts_cond_profile.npy", np.array(failure.volts_cond_profile, dtype=np.float64))


def main(length: int, width: float, seed: int, save_volts_profile: bool, val_cap: float, time_step: float) -> None:
    # initialization
    array = Array(length=length)
    matrix = Matrix(matrix_init=None, array=array, val_cap=val_cap, time_step=time_step)
    equation = Equation(array=array, matrix=matrix, save_volts_profile=save_volts_profile)
    failure = Failure(array=array, matrix=matrix, equation=equation, width=width, seed=seed, save_volts_profile=save_volts_profile)
    
    # breakdown simulation
    # 1st bond breaking [t=0]
    solve = equation.solve_init
    break_edge = failure.break_edge_init
    solve()
    break_edge()

    # 2nd ~ (length)th bond breaking [t=1 ~ t=(length-1)]
    solve = equation.solve
    solve_r = equation.solve_r_mmd
    break_edge = failure.break_edge
    for _ in range(array.length - 1):
        solve_r()
        solve()
        break_edge()

    # (length+1)th ~ (macroscopic failure) bond breaking [t=(length) ~ (total number of broken bonds-1)]
    solve_r = equation.solve_r_amd
    while solve_r():
        solve()
        break_edge()

    save_result(output_dir_global, failure)


if __name__=="__main__":
    args = parse_args()
    validate_args(args)
    output_dir_global = get_output_dir(args)
    if output_dir_global is None: raise SystemExit("[simulation_seed.py] the result already exists")

    main(args.length, args.width, args.seed, args.save, args.cap, args.time)
    raise SystemExit(0)