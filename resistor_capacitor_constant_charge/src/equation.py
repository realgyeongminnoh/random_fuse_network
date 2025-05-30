import warnings
import numpy as np
from scipy.sparse.linalg import splu, spsolve

from src.edge_list import EdgeList
from src.matrix import Matrix


warnings.filterwarnings("error")

class Equation:
    def __init__(self, edgeList: EdgeList, matrix: Matrix):
        self.edgeList = edgeList
        self.matrix = matrix
        self.extVolt = 1
        self.vectorRhs = np.zeros(self.edgeList.sizeMatCond, dtype=np.float64)
        self.nodeVolts = np.empty(self.edgeList.sizeMatDivComb) ###############################################################################
        self.failure = None


    def GetFailure(self, failure):
        self.failure = failure # lazy initialization for rare case: a broken leaf edge

    
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


    def _UndoUpdateMatDivCapComb(self, idxLeafEdge):
        numNodeDivMid = self.edgeList.numNodeDivMid
        idxNodeBotMinusOne = self.edgeList.numNode - self.edgeList.length - 1
        idxLeafEdgeDivCond = self.edgeList.idxMapFromEdgesToEdgesDivCond[idxLeafEdge]
        idxNode1, idxNode2 = self.edgeList.edgesDivCond[idxLeafEdgeDivCond]
        idxNode1New, idxNode2New = idxNode1 - self.edgeList.length, idxNode2 - 2 * self.edgeList.length
        matDivCap, matDivComb = self.matrix.matDivCap, self.matrix.matDivComb

        if idxNodeBotMinusOne < idxNode1:
            matDivCap[numNodeDivMid, idxNode2New] += 1
            matDivComb[numNodeDivMid, idxNode2New] += 1
            matDivComb[idxNode2New, idxNode2New] += 0.01
        else:
            matDivComb[idxNode1New, idxNode1New] += 0.01
            matDivComb[idxNode2New, idxNode2New] += 0.01
            matDivComb[idxNode1New, idxNode2New] -= 0.01
            matDivComb[idxNode2New, idxNode1New] -= 0.01


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


    def ComputeMmdRfnResistorYesEdgeVolts(self):
        try:
            splu(self.matrix.matCond, permc_spec="MMD_AT_PLUS_A")
        except: # singular matrix iff previously broken edge was leaf
            idxLeafEdge = self.failure.idxBrokenEdges.pop()
            self.failure.idxLeafEdges.append(idxLeafEdge)
            
            # undo update of previous step Iteration Algorithm of failure
            self._UndoUpdateMatDivCapComb(idxLeafEdge)
            self._UndoUpdateMatCond(idxLeafEdge)
            self.extVolt = self.failure.extVolts[-2]
            unscalingFactor = self.extVolt / self.failure.extVolts.pop()
            if unscalingFactor != 1:
                self.nodeVolts *= unscalingFactor ###############################################################################
            self.failure.dataRawEdgeVolts.pop()

            # redo previous step Iteration Algorithm of failure with the updated idxLeafEdges
            self.failure.IterAlgoYesEdgeVolts()
            # redo current step Compute of equation; recurse in case of another leaf edge being broken
            self.ComputeMmdRfnResistorYesEdgeVolts()


    def ComputeMmdRfnResistorNoEdgeVolts(self):
        try:
            splu(self.matrix.matCond, permc_spec="MMD_AT_PLUS_A")
        except:
            idxLeafEdge = self.failure.idxBrokenEdges.pop()
            self.failure.idxLeafEdges.append(idxLeafEdge)

            self._UndoUpdateMatDivCapComb(idxLeafEdge)
            self._UndoUpdateMatCond(idxLeafEdge)
            self.extVolt = self.failure.extVolts[-2]
            unscalingFactor = self.extVolt / self.failure.extVolts.pop()
            if unscalingFactor != 1:
                self.nodeVolts *= unscalingFactor ###############################################################################

            self.failure.IterAlgoNoEdgeVolts()
            self.ComputeMmdRfnResistorNoEdgeVolts()


    def ComputeAmdRfnResistorYesEdgeVolts(self):
        try:
            self.vectorRhs[-1] = self.extVolt
            extCurrRfnResistor = np.abs(spsolve(self.matrix.matCond, self.vectorRhs, permc_spec="COLAMD")[-1])
            return extCurrRfnResistor > 1e-5
        except:
            idxLeafEdge = self.failure.idxBrokenEdges.pop()
            self.failure.idxLeafEdges.append(idxLeafEdge)

            self._UndoUpdateMatDivCapComb(idxLeafEdge)
            self._UndoUpdateMatCond(idxLeafEdge)
            self.extVolt = self.failure.extVolts[-2]
            unscalingFactor = self.extVolt / self.failure.extVolts.pop()
            if unscalingFactor != 1:
                self.nodeVolts *= unscalingFactor ###############################################################################
            self.failure.dataRawEdgeVolts.pop()

            self.failure.IterAlgoYesEdgeVolts()
            return self.ComputeAmdRfnResistorYesEdgeVolts()


    def ComputeAmdRfnResistorNoEdgeVolts(self):
        try:
            self.vectorRhs[-1] = self.extVolt
            extCurrRfnResistor = np.abs(spsolve(self.matrix.matCond, self.vectorRhs, permc_spec="COLAMD")[-1])
            return extCurrRfnResistor > 1e-5
        except:
            idxLeafEdge = self.failure.idxBrokenEdges.pop()
            self.failure.idxLeafEdges.append(idxLeafEdge)

            self._UndoUpdateMatDivCapComb(idxLeafEdge)
            self._UndoUpdateMatCond(idxLeafEdge)
            self.extVolt = self.failure.extVolts[-2]
            unscalingFactor = self.extVolt / self.failure.extVolts.pop()
            if unscalingFactor != 1:
                self.nodeVolts *= unscalingFactor ###############################################################################

            self.failure.IterAlgoNoEdgeVolts()
            return self.ComputeAmdRfnResistorNoEdgeVolts()