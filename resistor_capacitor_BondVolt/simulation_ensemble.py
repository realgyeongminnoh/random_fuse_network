import os
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from src import *


def ParseArgs():
    parser = argparse.ArgumentParser(description="Random Fuse Network - Resistor-Capacitor (using bond voltage for 'edgeVolt'): Ensemble Simulation")
    parser.add_argument("--length", "--l", type=int, required=True, help="Number of Vertical/Horizonatal Bonds per Column/Row >= 3")
    parser.add_argument("--width", "--w", type=float, required=True, help="Width of Random Uniform Distribution of Threshold Voltage Drops ∈ [0, 2]")
    parser.add_argument("--seedMin", "--smin", type=int, required=True, help="Smallest Seed Number >= 0: seeds = [seedMin, seedMin+1, ..., seedMax-1, seedMax]")
    parser.add_argument("--seedMax", "--smax", type=int, required=True, help="Greatest Seed Number <= 4294967295: seeds = [seedMin, seedMin+1, ..., seedMax-1, seedMax]")    
    parser.add_argument("--saveEdgeVolts", "--v", action="store_true", help="[Maximal Saving] Collect and Save edgeVolts")
    parser.add_argument("--saveOnlyExtVolts", "--e", action="store_true", help="[Minimal Saving] Save Only extVolts")
    parser.add_argument("--forceCpu", "--c", type=int, required=False, help="Force a Fixed Number of CPU Processors")
    return parser.parse_args()


def ValidateArgs(args):
    if not (args.length >= 3):
        raise ValueError("length >= 3")
    if not (0 <= args.width <= 2):
        raise ValueError("width ∈ [0, 2]")
    if not (args.seedMin >= 0): 
        raise ValueError("seedMin >= 0")
    if not (args.seedMax <= 4294967295):
        raise ValueError("seedMax <= 4294967295")
    if not (args.seedMin <= args.seedMax):
        raise ValueError("seedMin <= seedMax")
    if args.saveEdgeVolts and args.saveOnlyExtVolts:
        raise ValueError("choose only one saving option: minimal (--a) / maximal (--v)")
    if (args.forceCpu is not None) and not (1 <= args.forceCpu <= os.cpu_count()):
        raise ValueError("designated processor number must not exceed the number of processors in your computer")
    

def SetSharedPath(length, width, seedMin, seedMax):
    sharedFilePath = (
        f"{os.path.abspath(__file__).split('simulation_ensemble.py')[0]}data/"
        f"{length}/"
        f"{width}/"
    )
    os.makedirs(sharedFilePath, exist_ok=True)

    for seed in range(seedMin, seedMax + 1):
        filePath = sharedFilePath + f"{seed}/"
        os.makedirs(filePath, exist_ok=True)

    return sharedFilePath


def SaveResult(filePath, failure):
    if not saveOnlyExtVoltsGlobal:
        np.save(f"{filePath}idxBrokenEdges.npy", failure.idxBrokenEdges)

    if failure.extVolts[-1] < 6.5505e4:
        np.save(f"{filePath}extVolts.npy", np.array(failure.extVolts, dtype=np.float16))
        if saveEdgeVoltsGlobal:
            np.save(f"{filePath}dataRawEdgeVolts.npy", np.array(failure.dataRawEdgeVolts, dtype=np.float16))
    elif failure.extVolts[-1] < 3.4028236e38:
        np.save(f"{filePath}extVolts.npy", np.array(failure.extVolts, dtype=np.float32))
        if saveEdgeVoltsGlobal:
            np.save(f"{filePath}dataRawEdgeVolts.npy", np.array(failure.dataRawEdgeVolts, dtype=np.float32))
    else:
        np.save(f"{filePath}extVolts.npy", np.array(failure.extVolts, dtype=np.float64))
        if saveEdgeVoltsGlobal:
            np.save(f"{filePath}dataRawEdgeVolts.npy", np.array(failure.dataRawEdgeVolts, dtype=np.float64))


def ParallelJob(seed):
    # instantiation - seed-specific
    matrix = Matrix(edgeListGlobal, matCondInit=matCondInitGlobal.copy(), matDivCapInit=matDivCapInitGlobal.copy(), matDivCombInit=matDivCombInitGlobal.copy())
    randList = RandList(edgeListGlobal, width=widthGlobal, seed=seed)
    equation = Equation(edgeListGlobal, matrix)
    failure = Failure(edgeListGlobal, matrix, randList, equation)

    # breakdown simulation
    # 1st bond breaking [t=0]
    Compute = equation.ComputeInit
    IterationAlgorithm = failure.IterAlgoYesEdgeVoltsInit if saveEdgeVoltsGlobal else failure.IterAlgoNoEdgeVoltsInit
    Compute()
    IterationAlgorithm()

    # 2nd ~ (length)th bond breaking [t=1 ~ t=(length - 1)]
    Compute = equation.Compute
    IterationAlgorithm = failure.IterAlgoYesEdgeVolts if saveEdgeVoltsGlobal else failure.IterAlgoNoEdgeVolts
    ComputeMmdRfnResistor = equation.ComputeMmdRfnResistorYesEdgeVolts if saveEdgeVoltsGlobal else equation.ComputeMmdRfnResistorNoEdgeVolts
    for _ in range(lengthGlobal - 1):
        ComputeMmdRfnResistor()
        Compute()
        IterationAlgorithm()

    # (length+1)th ~ (macroscopic failure) bond breaking [t=(length) ~ t=(total number of broken bonds-1)]
    ComputeAmdRfnResistor = equation.ComputeAmdRfnResistorYesEdgeVolts if saveEdgeVoltsGlobal else equation.ComputeAmdRfnResistorNoEdgeVolts
    while ComputeAmdRfnResistor():
        Compute()
        IterationAlgorithm()

    # saving execution
    SaveResult(sharedFilePathGlobal + f"{seed}/", failure)


def Main(length, width, seedMin, seedMax, saveEdgeVolts, saveOnlyExtVolts, forceCpu):
    # globalization & saving initialization - shared across seeds
    global lengthGlobal, widthGlobal
    global saveEdgeVoltsGlobal, saveOnlyExtVoltsGlobal, sharedFilePathGlobal
    lengthGlobal, widthGlobal = length, width
    saveEdgeVoltsGlobal, saveOnlyExtVoltsGlobal = saveEdgeVolts, saveOnlyExtVolts
    sharedFilePathGlobal = SetSharedPath(length, width, seedMin, seedMax)

    # instantiation - shared across seeds
    global edgeListGlobal, matrixGlobal, matCondInitGlobal, matDivCapInitGlobal, matDivCombInitGlobal
    edgeListGlobal = EdgeList(length=length)
    matrixGlobal = Matrix(edgeListGlobal, matCondInit=None, matDivCapInit=None, matDivCombInit=None)
    matCondInitGlobal, matDivCapInitGlobal, matDivCombInitGlobal = matrixGlobal.matCond, matrixGlobal.matDivCap, matrixGlobal.matDivComb

    # parallelization
    maxWorkers = (seedMax - seedMin + 1) if (seedMax - seedMin + 1) <= os.cpu_count() else os.cpu_count()
    maxWorkers = forceCpu if forceCpu else maxWorkers
    with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
        executor.map(ParallelJob, range(seedMin, seedMax + 1))


if __name__ == "__main__":
    args = ParseArgs()
    ValidateArgs(args)
    Main(args.length, args.width, args.seedMin, args.seedMax, args.saveEdgeVolts, args.saveOnlyExtVolts, args.forceCpu)