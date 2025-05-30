import numpy as np
import warnings 

from src.edge_list import EdgeList
from src.matrix import Matrix
from src.rand_list import RandList
from src.equation import Equation


warnings.simplefilter("ignore", category=RuntimeWarning) # ignoring division by zero cases for scalingFactors computation (edgeVolt = 0; no other cases)

class Failure:
    def __init__(self, edgeList: EdgeList, matrix: Matrix, randList: RandList, equation: Equation):
        self.edgeList = edgeList
        self.matrix = matrix
        self.randList = randList
        self.equation = equation
        self.edgeVolts = np.empty(self.edgeList.numEdge)
        self.idxBrokenEdges = []
        self.idxLeafEdges = []
        self.extVolts = []
        self.dataRawEdgeVolts = []
        self.equation.GetFailure(self) # lazy initialization for rare case: a broken leaf edge

    
    def _GenerateEdgeVolts(self):
        length = self.edgeList.length
        idxNodeBotMinusOne = self.edgeList.numNode - length - 1
        extVolt, nodeVolts, edgeVolts = self.equation.extVolt, self.equation.nodeVolts, self.edgeVolts 

        for idxEdge, (idxNode1, idxNode2) in enumerate(self.edgeList.edges):
            if idxNode1 < length:
                edgeVolts[idxEdge] = extVolt - nodeVolts[idxNode2 - length]
            elif idxNode2 > idxNodeBotMinusOne:
                edgeVolts[idxEdge] = nodeVolts[idxNode1 - length]
            else:
                edgeVolts[idxEdge] = nodeVolts[idxNode1 - length] - nodeVolts[idxNode2 - length]

        np.abs(edgeVolts, out=edgeVolts)


    def _UpdateMatDivCapComb(self, idxBrokenEdge):
        numNodeDivMid = self.edgeList.numNodeDivMid
        idxNodeBotMinusOne = self.edgeList.numNode - self.edgeList.length - 1
        idxBrokenEdgeDivCond = self.edgeList.idxMapFromEdgesToEdgesDivCond[idxBrokenEdge]
        idxNode1, idxNode2 = self.edgeList.edgesDivCond[idxBrokenEdgeDivCond]
        idxNode1New, idxNode2New = idxNode1 - self.edgeList.length, idxNode2 - 2 * self.edgeList.length
        matDivCap, matDivComb = self.matrix.matDivCap, self.matrix.matDivComb

        if idxNodeBotMinusOne < idxNode1:
            matDivCap[numNodeDivMid, idxNode2New] -= 1
            matDivComb[numNodeDivMid, idxNode2New] -= 1
            matDivComb[idxNode2New, idxNode2New] -= 0.01
        else:
            matDivComb[idxNode1New, idxNode1New] -= 0.01
            matDivComb[idxNode2New, idxNode2New] -= 0.01
            matDivComb[idxNode1New, idxNode2New] += 0.01
            matDivComb[idxNode2New, idxNode1New] += 0.01


    def _UpdateMatCond(self, idxBrokenEdge):
        idxNode1, idxNode2 = self.edgeList.edges[idxBrokenEdge]
        idxNode1New, idxNode2New = idxNode1 - self.edgeList.length + 1, idxNode2 - self.edgeList.length + 1
        matCond = self.matrix.matCond

        if idxNode2New <= self.edgeList.numNodeMid:
            if idxNode1New > 0:
                matCond[idxNode1New, idxNode1New] -= 1
                matCond[idxNode2New, idxNode2New] -= 1
                matCond[idxNode1New, idxNode2New] += 1
                matCond[idxNode2New, idxNode1New] += 1
            else:
                matCond[0, 0] -= 1
                matCond[idxNode2New, idxNode2New] -= 1
                matCond[0, idxNode2New] += 1
                matCond[idxNode2New, 0] += 1
        else:
                matCond[idxNode1New, idxNode1New] -= 1


    def IterAlgoYesEdgeVoltsInit(self):
        self._GenerateEdgeVolts()
        idxBrokenEdge = int(np.argmin(self.randList.randThrVolts - self.edgeVolts))

        self._UpdateMatDivCapComb(idxBrokenEdge)
        self._UpdateMatCond(idxBrokenEdge)
        self.idxBrokenEdges.append(idxBrokenEdge)
        self.extVolts.append(0)
        self.dataRawEdgeVolts.append(self.edgeVolts.copy())

        # at the first bond breaking, extVolt (external potential) is always scaled to the value at which the maximally stressed bond is broken
        scalingFactor = self.randList.randThrVolts[idxBrokenEdge] / self.edgeVolts[idxBrokenEdge]
        self.equation.extVolt *= scalingFactor
        self.equation.nodeVolts *= scalingFactor
        self.extVolts[0] = float(self.equation.extVolt)


    def IterAlgoNoEdgeVoltsInit(self):
        self._GenerateEdgeVolts()
        idxBrokenEdge = int(np.argmin(self.randList.randThrVolts - self.edgeVolts))

        self._UpdateMatDivCapComb(idxBrokenEdge)
        self._UpdateMatCond(idxBrokenEdge)
        self.idxBrokenEdges.append(idxBrokenEdge)
        self.extVolts.append(0)

        scalingFactor = self.randList.randThrVolts[idxBrokenEdge] / self.edgeVolts[idxBrokenEdge]
        self.equation.extVolt *= scalingFactor
        self.equation.nodeVolts *= scalingFactor
        self.extVolts[0] = float(self.equation.extVolt)


    def IterAlgoYesEdgeVolts(self):
        self._GenerateEdgeVolts()
        stresses = self.randList.randThrVolts - self.edgeVolts
        stresses[self.idxBrokenEdges] = 1e10
        stresses[self.idxLeafEdges] = 1e10
        idxBrokenEdge = int(np.argmin(stresses)) # maximally stressed bond is ...

        if stresses[idxBrokenEdge] <= 0: # overstressed - no scaling of extVolt (external potential)
            self._UpdateMatDivCapComb(idxBrokenEdge)
            self._UpdateMatCond(idxBrokenEdge)
            self.idxBrokenEdges.append(idxBrokenEdge)
            self.extVolts.append(self.extVolts[-1])
            self.dataRawEdgeVolts.append(self.edgeVolts.copy())

        else: # understressed - finding new idxBrokenBond such that extVolt is raised (scaled up) by a minimum amount
            scalingFactors = self.randList.randThrVolts / self.edgeVolts
            scalingFactors[self.idxBrokenEdges] = 1e10
            scalingFactors[self.idxLeafEdges] = 1e10
            idxBrokenEdge = int(np.argmin(scalingFactors))
            scalingFactor = scalingFactors[idxBrokenEdge]

            self._UpdateMatDivCapComb(idxBrokenEdge)
            self._UpdateMatCond(idxBrokenEdge)
            self.idxBrokenEdges.append(idxBrokenEdge)
            self.extVolts.append(self.extVolts[-1])
            self.dataRawEdgeVolts.append(self.edgeVolts.copy())

            self.equation.extVolt *= scalingFactor
            self.equation.nodeVolts *= scalingFactor
            self.extVolts[-1] = float(self.equation.extVolt)


    def IterAlgoNoEdgeVolts(self):
        self._GenerateEdgeVolts()
        stresses = self.randList.randThrVolts - self.edgeVolts
        stresses[self.idxBrokenEdges] = 1e10
        stresses[self.idxLeafEdges] = 1e10
        idxBrokenEdge = int(np.argmin(stresses))

        if stresses[idxBrokenEdge] <= 0:
            self._UpdateMatDivCapComb(idxBrokenEdge)
            self._UpdateMatCond(idxBrokenEdge)
            self.idxBrokenEdges.append(idxBrokenEdge)
            self.extVolts.append(self.extVolts[-1])

        else:
            scalingFactors = self.randList.randThrVolts / self.edgeVolts
            scalingFactors[self.idxBrokenEdges] = 1e10
            scalingFactors[self.idxLeafEdges] = 1e10
            idxBrokenEdge = int(np.argmin(scalingFactors))
            scalingFactor = scalingFactors[idxBrokenEdge]

            self._UpdateMatDivCapComb(idxBrokenEdge)
            self._UpdateMatCond(idxBrokenEdge)
            self.idxBrokenEdges.append(idxBrokenEdge)
            self.extVolts.append(self.extVolts[-1])

            self.equation.extVolt *= scalingFactor
            self.equation.nodeVolts *= scalingFactor
            self.extVolts[-1] = float(self.equation.extVolt)