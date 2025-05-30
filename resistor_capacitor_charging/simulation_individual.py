import os
import argparse
import numpy as np

from src import *


def ParseArgs():
    parser = argparse.ArgumentParser(description="Random Fuse Network - Resistor-Capacitor (charging): Individual Simulation")
    parser.add_argument("--length", "--l", type=int, required=True, help="Number of Vertical/Horizonatal Bonds per Column/Row >= 3")
    parser.add_argument("--width", "--w", type=float, required=True, help="Width of Random Uniform Distribution of Threshold Voltage Drops ∈ [0, 2]")
    parser.add_argument("--seed", "--s", type=int, required=True, help="Seed Number ∈ [0, 4294967295]")
    parser.add_argument("--saveEdgeVolts", "--v", action="store_true", help="[Maximal Saving] Collect and Save EdgeVolts")
    parser.add_argument("--valCap", "--c", type=float, required=False, default=1.0, help="Capacitance value for all capacitors > 0 (default = 1)")
    return parser.parse_args()


def ValidateArgs(args):
    if not (args.length >= 3):
        raise ValueError("length >= 3")
    if not (0 <= args.width <= 2):
        raise ValueError("width ∈ [0, 2]")
    if not (0 <= args.seed <= 4294967295):
        raise ValueError("seed ∈ [0, 4294967295]")
    if not (args.valCap > 0):
        raise ValueError("valCap > 0")
    

def SetPath(length, width, seed, saveEdgeVolts, valCap):
    filePath = (
        f"{os.path.abspath(__file__).split('simulation_individual.py')[0]}data/"
        f"{length}/"
        f"{width}/"
        f"{seed}/"
        f"{valCap}/"
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
    np.save(f"{filePath}idxBrokenEdges.npy", failure.idxBrokenEdges)

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


def Main(length, width, seed, saveEdgeVolts, valCap):
    # saving initialization
    filePath = SetPath(length, width, seed, saveEdgeVolts, valCap)
    if not filePath:
        return

    # instantiation
    edgeList = EdgeList(length=length)
    matrix = Matrix(edgeList, valCap=valCap, matCondInit=None, matDivCapInit=None, matDivCombInit=None)
    randList = RandList(edgeList, width=width, seed=seed)
    equation = Equation(edgeList, matrix)
    failure = Failure(edgeList, matrix, randList, equation)

    # breakdown simulation
    # 1st bond breaking [t=0]
    Compute = equation.ComputeInit
    IterationAlgorithm = failure.IterAlgoYesEdgeVoltsInit if saveEdgeVolts else failure.IterAlgoNoEdgeVoltsInit
    Compute()
    IterationAlgorithm()

    # 2nd ~ (length)th bond breaking [t=1 ~ t=(length - 1)]
    Compute = equation.Compute
    IterationAlgorithm = failure.IterAlgoYesEdgeVolts if saveEdgeVolts else failure.IterAlgoNoEdgeVolts
    ComputeMmdRfnResistor = equation.ComputeMmdRfnResistorYesEdgeVolts if saveEdgeVolts else equation.ComputeMmdRfnResistorNoEdgeVolts
    for _ in range(length - 1):
        ComputeMmdRfnResistor()
        Compute()
        IterationAlgorithm()

    # (length+1)th ~ (macroscopic failure) bond breaking [t=(length) ~ t=(total number of broken bonds-1)]
    ComputeAmdRfnResistor = equation.ComputeAmdRfnResistorYesEdgeVolts if saveEdgeVolts else equation.ComputeAmdRfnResistorNoEdgeVolts
    while ComputeAmdRfnResistor():
        Compute()
        IterationAlgorithm()

    # saving execution
    SaveResult(filePath, failure, saveEdgeVolts)


if __name__ == "__main__":
    args = ParseArgs()
    ValidateArgs(args)
    Main(args.length, args.width, args.seed, args.saveEdgeVolts, args.valCap)