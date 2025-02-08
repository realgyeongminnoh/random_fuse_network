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
        self.nodeVolts = None

    
    def ComputeMmd(self):
        self.vectorRhs[-1] = self.extVolt
        self.nodeVolts = spsolve(self.matrix.matCond, self.vectorRhs, permc_spec="MMD_AT_PLUS_A")
        

    def ComputeAmd(self):
        self.vectorRhs[-1] = self.extVolt
        self.nodeVolts = spsolve(self.matrix.matCond, self.vectorRhs, permc_spec="COLAMD")
        return np.abs(self.nodeVolts[-1]) > 1e-5