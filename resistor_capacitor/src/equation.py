import numpy as np
from scipy.sparse.linalg import spsolve

from src.edge_list import EdgeList
from src.matrix import Matrix


class Equation:
    def __init__(self, edgeList: EdgeList, matrix: Matrix):
        self.edgeList = edgeList
        self.matrix = matrix
        self.extVolt = 1
        self.vectorRhs = np.zeros(self.edgeList.sizeMatCond, dtype=np.float64)
        self.nodeVolts = np.empty(self.edgeList.sizeMatDivComb)


    def ComputeInit(self):
        self.vectorRhs[-1] = self.extVolt
        nodeVoltsIC = spsolve(self.matrix.matCond, self.vectorRhs, permc_spec="MMD_AT_PLUS_A")

        length, lengthDoubled = self.edgeList.length, 2 * self.edgeList.length
        nodeVolts = self.nodeVolts

        nodeVolts[-1] = nodeVoltsIC[-1]
        nodeVolts[:self.edgeList.numNodeMid] = nodeVoltsIC[1:-1]
        nodeVolts[self.edgeList.numNodeMid:self.edgeList.numNodeMid + length] = self.extVolt
        for idxNode1, idxNode2 in self.edgeList.edgesDivCap[length:]:
            nodeVolts[idxNode2 - lengthDoubled] = nodeVolts[idxNode1 - length]


    def Compute(self):
        self.nodeVolts = self.matrix.matDivCap @ self.nodeVolts
        self.nodeVolts[-1] = 0
        self.nodeVolts = spsolve(self.matrix.matDivComb, self.nodeVolts, permc_spec="MMD_AT_PLUS_A")


    def TerminationCondition(self):
        self.vectorRhs[-1] = self.extVolt
        return np.abs(spsolve(self.matrix.matCond, self.vectorRhs, permc_spec="COLAMD")[-1]) > 1e-5