import os
import argparse
import numpy as np

from src import *


def ParseArgs():
    parser = argparse.ArgumentParser(description="Random Fuse Network - Resistor: Individual Simulation")
    parser.add_argument("--length", "--l", type=int, required=True, help="Number of Vertical/Horizonatal Bonds per column >= 10")
    parser.add_argument("--width", "--w", type=float, required=True, help="Width of Random Uniform Distribution of Threshold Voltage Drops ∈ [0, 2]")
    parser.add_argument("--seed", "--s", type=int, required=True, help="Seed Number ∈ [0, 4294967295]")
    parser.add_argument("--saveEdgeVolts", "--v", action="store_true", help="[Maximal Saving] Collect and Save EdgeVolts")
    parser.add_argument("--saveOnlyAvalancheSize", "--a", action="store_true", help="[Minimal Saving] Save Only Avalanche Size")
    return parser.parse_args()


def ValidateArgs(args):
    if not (args.length >= 10):
        raise ValueError("length >= 10")
    if not (0 <= args.width <= 2):
        raise ValueError("width ∈ [0, 2]")
    if not (0 <= args.seed <= 4294967295):
        raise ValueError("seed ∈ [0, 4294967295]")
    if args.saveEdgeVolts and args.saveOnlyAvalancheSize:
        raise ValueError("choose only one saving option: minimal (--a) / maximal (--v)")


def SetPath(length, width, seed, saveEdgeVolts):
    filePath = (
        f"{os.path.abspath(__file__).split('simulation_individual.py')[0]}data/"
        f"{length}/"
        f"{width}/"
        f"{seed}/"
    )
    os.makedirs(filePath, exist_ok=True)

    if (os.path.exists(f"{filePath}idxBrokenEdges.npy") and os.path.exists(f"{filePath}extVolts.npy")):
        if saveEdgeVolts:
            if os.path.exists(f"{filePath}dataRawEdgeVolts.npy"):
                return False
            else:
                return filePath
        return False
    return filePath


def SaveResult(filePath, failure, saveEdgeVolts):
    np.save(f"{filePath}idxBrokenEdges.npy", np.array(failure.idxBrokenEdges, dtype=np.int32))
    if failure.extVolts[-1] > 3.4028235e38:
        np.save(f"{filePath}extVolts.npy", np.array(failure.extVolts, dtype=np.float64))
        if saveEdgeVolts:
            np.save(f"{filePath}dataRawEdgeVolts.npy", np.array(failure.dataRawEdgeVolts, dtype=np.float64))
    elif failure.extVolts[-1] > 6.5504e4:
        np.save(f"{filePath}extVolts.npy", np.array(failure.extVolts, dtype=np.float32))
        if saveEdgeVolts:
            np.save(f"{filePath}dataRawEdgeVolts.npy", np.array(failure.dataRawEdgeVolts, dtype=np.float32))
    else:
        np.save(f"{filePath}extVolts.npy", np.array(failure.extVolts, dtype=np.float16))
        if saveEdgeVolts:
            np.save(f"{filePath}dataRawEdgeVolts.npy", np.array(failure.dataRawEdgeVolts, dtype=np.float16))


def Main(length, width, seed, saveEdgeVolts, saveOnlyAvalancheSize):
    # saving initialization
    if not saveOnlyAvalancheSize:
        filePath = SetPath(length, width, seed, saveEdgeVolts)
        if not filePath:
            return

    # instantiation
    edgeList = EdgeList(length=length)
    matrix = Matrix(edgeList, matCondInit=None)
    randList = RandList(edgeList, width=width, seed=seed)
    equation = Equation(edgeList, matrix)
    failure = Failure(edgeList, matrix, randList, equation)

    # breakdown simulation
    # 1st bond breaking [t=0]
    Compute = equation.ComputeMmd
    IterationAlgorithm = failure.IterAlgoYesEdgeVoltsInit if saveEdgeVolts else failure.IterAlgoNoEdgeVoltsInit
    Compute()
    IterationAlgorithm()

    # 2nd ~ (length)th bond breaking [t=1 ~ t=(length - 1)]
    IterationAlgorithm = failure.IterAlgoYesEdgeVolts if saveEdgeVolts else failure.IterAlgoNoEdgeVolts
    for _ in range(length - 1):
        Compute()
        IterationAlgorithm()

    # (length+1)th ~ (macroscopic failure) bond breaking [t=(length) ~ t=(total number of broken bonds-1)]
    Compute = equation.ComputeAmd
    while Compute():
        IterationAlgorithm()
    
    # saving execution
    if saveOnlyAvalancheSize:
        print(length, width, seed, len(failure.idxBrokenEdges))
        return
    SaveResult(filePath, failure, saveEdgeVolts)


if __name__ == "__main__":
    args = ParseArgs()
    ValidateArgs(args)
    Main(args.length, args.width, args.seed, args.saveEdgeVolts, args.saveOnlyAvalancheSize)