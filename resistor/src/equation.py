import warnings
import numpy as np
from scipy.sparse.linalg import spsolve

from src.edge_list import EdgeList
from src.matrix import Matrix


warnings.filterwarnings("error")

class Equation:
    def __init__(self, edgeList: EdgeList, matrix: Matrix):
        self.edgeList = edgeList
        self.matrix = matrix
        self.extVolt = 1
        self.vectorRhs = np.zeros(self.edgeList.sizeMatCond, dtype=np.float64)
        self.nodeVolts = None
        self.failure = None

    
    def GetFailure(self, failure):
        self.failure = failure # lazy initialization for rare case: a broken leaf edge


    def _UndoUpdateMatCond(self, idxLeafEdge):
        idxNode1, idxNode2 = self.edgeList.edges[idxLeafEdge]
        idxNode1New, idxNode2New = idxNode1 - self.edgeList.length + 1, idxNode2 - self.edgeList.length + 1
        matCond = self.matrix.matCond

        if idxNode2New <= self.edgeList.numNodeMid:
            if idxNode1New > 0:
                matCond[idxNode1New, idxNode1New] += 1
                matCond[idxNode2New, idxNode2New] += 1
                matCond[idxNode1New, idxNode2New] -= 1
                matCond[idxNode2New, idxNode1New] -= 1
            else:
                matCond[0, 0] += 1
                matCond[idxNode2New, idxNode2New] += 1
                matCond[0, idxNode2New] -= 1
                matCond[idxNode2New, 0] -= 1
        else:
                matCond[idxNode1New, idxNode1New] += 1


    def ComputeMmdYesEdgeVolts(self):
        self.vectorRhs[-1] = self.extVolt
        try:
            self.nodeVolts = spsolve(self.matrix.matCond, self.vectorRhs, permc_spec="MMD_AT_PLUS_A")
        except: # singular matrix iff previously broken edge was leaf
            idxLeafEdge = self.failure.idxBrokenEdges.pop()
            self.failure.idxLeafEdges.append(idxLeafEdge)
            
            # undo update of previous step Iteration Algorithm of failure
            self._UndoUpdateMatCond(idxLeafEdge)
            self.extVolt = self.failure.extVolts[-2]
            unscalingFactor = self.extVolt / self.failure.extVolts.pop()
            if unscalingFactor != 1: 
                self.nodeVolts *= unscalingFactor
            self.failure.dataRawEdgeVolts.pop()

            # redo previous step Iteration Algorithm of failure with the updated idxLeafEdges
            self.failure.IterAlgoYesEdgeVolts()
            # recursel redo current step Compute of equation 
            self.ComputeMmdYesEdgeVolts()
           

    def ComputeMmdNoEdgeVolts(self):
        self.vectorRhs[-1] = self.extVolt
        try:
            self.nodeVolts = spsolve(self.matrix.matCond, self.vectorRhs, permc_spec="MMD_AT_PLUS_A")
        except:
            idxLeafEdge = self.failure.idxBrokenEdges.pop()
            self.failure.idxLeafEdges.append(idxLeafEdge)
            
            self._UndoUpdateMatCond(idxLeafEdge)
            self.extVolt = self.failure.extVolts[-2]
            unscalingFactor = self.extVolt / self.failure.extVolts.pop()
            if unscalingFactor != 1: 
                self.nodeVolts *= unscalingFactor

            self.failure.IterAlgoNoEdgeVolts()
            self.ComputeMmdNoEdgeVolts()


    def ComputeAmdYesEdgeVolts(self):
        self.vectorRhs[-1] = self.extVolt
        try:
            self.nodeVolts = spsolve(self.matrix.matCond, self.vectorRhs, permc_spec="COLAMD")
            return np.abs(self.nodeVolts[-1]) > 1e-5
        except:
            idxLeafEdge = self.failure.idxBrokenEdges.pop()
            self.failure.idxLeafEdges.append(idxLeafEdge)

            self._UndoUpdateMatCond(idxLeafEdge)
            self.extVolt = self.failure.extVolts[-2]
            unscalingFactor = self.extVolt / self.failure.extVolts.pop()
            if unscalingFactor != 1:
                self.nodeVolts *= unscalingFactor
            self.failure.dataRawEdgeVolts.pop()

            self.failure.IterAlgoYesEdgeVolts()
            return self.ComputeAmdYesEdgeVolts()


    def ComputeAmdNoEdgeVolts(self):
        self.vectorRhs[-1] = self.extVolt
        try:
            self.nodeVolts = spsolve(self.matrix.matCond, self.vectorRhs, permc_spec="COLAMD")
            return np.abs(self.nodeVolts[-1]) > 1e-5
        except:
            idxLeafEdge = self.failure.idxBrokenEdges.pop()
            self.failure.idxLeafEdges.append(idxLeafEdge)

            self._UndoUpdateMatCond(idxLeafEdge)
            self.extVolt = self.failure.extVolts[-2]
            unscalingFactor = self.extVolt / self.failure.extVolts.pop()
            if unscalingFactor != 1:
                self.nodeVolts *= unscalingFactor

            self.failure.IterAlgoNoEdgeVolts()
            return self.ComputeAmdNoEdgeVolts()