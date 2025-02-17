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
        self.idxBrokenEdgesDivCap = []
        self.idxLeafEdgesDivCap = []
        self.extVolts = []
        self.dataRawEdgeVolts = []
        self.equation.GetFailure(self) # lazy initialization for rare case: a broken leaf edge0
        self.idxBrokenEdges = None

    
    def _GenerateEdgeVolts(self):
        length = self.edgeList.length
        lengthMinusOne, lengthDoubled = length - 1, 2 * length
        extVolt, nodeVolts, edgeVolts = self.equation.extVolt, self.equation.nodeVolts, self.edgeVolts 

        for idxEdgeDivCap, (idxNode1, idxNode2) in enumerate(self.edgeList.edgesDivCap):
            if idxNode1 > lengthMinusOne:
                edgeVolts[idxEdgeDivCap] = nodeVolts[idxNode1 - length] - nodeVolts[idxNode2 - lengthDoubled]
            else:
                edgeVolts[idxEdgeDivCap] = extVolt - nodeVolts[idxNode2 - lengthDoubled]

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
        idxBrokenEdgeDivCap = int(np.argmin(self.randList.randThrVolts))
        idxBrokenEdge = self.edgeList.idxMapFromEdgesDivCapToEdges[idxBrokenEdgeDivCap]

        self._UpdateMatDivCapComb(idxBrokenEdge)
        self._UpdateMatCond(idxBrokenEdge)
        self.idxBrokenEdgesDivCap.append(idxBrokenEdgeDivCap)
        self.extVolts.append(0)
        self.dataRawEdgeVolts.append(np.zeros(self.edgeList.numEdge))


    def IterAlgoNoEdgeVoltsInit(self):
        idxBrokenEdgeDivCap = int(np.argmin(self.randList.randThrVolts))
        idxBrokenEdge = self.edgeList.idxMapFromEdgesDivCapToEdges[idxBrokenEdgeDivCap]

        self._UpdateMatDivCapComb(idxBrokenEdge)
        self._UpdateMatCond(idxBrokenEdge)
        self.idxBrokenEdgesDivCap.append(idxBrokenEdgeDivCap)
        self.extVolts.append(0)


    def IterAlgoYesEdgeVolts(self):
        self._GenerateEdgeVolts()
        scalingFactors = self.randList.randThrVolts / self.edgeVolts
        scalingFactors[self.idxBrokenEdgesDivCap] = 1e10
        scalingFactors[self.idxLeafEdgesDivCap] = 1e10
        idxBrokenEdgeDivCap = int(np.argmin(scalingFactors))
        edgeVoltBrokenEdgeDivCap = self.edgeVolts[idxBrokenEdgeDivCap]
        scalingFactor = scalingFactors[idxBrokenEdgeDivCap]
        idxBrokenEdge = self.edgeList.idxMapFromEdgesDivCapToEdges[idxBrokenEdgeDivCap]

        self._UpdateMatDivCapComb(idxBrokenEdge)
        self._UpdateMatCond(idxBrokenEdge)
        self.idxBrokenEdgesDivCap.append(idxBrokenEdgeDivCap)
        self.extVolts.append(self.extVolts[-1])
        self.dataRawEdgeVolts.append(self.edgeVolts.copy())

        if scalingFactor > 1:
            if edgeVoltBrokenEdgeDivCap > 1e-10:
                self.equation.extVolt *= scalingFactor
                self.equation.nodeVolts *= scalingFactor
                self.extVolts[-1] = float(self.equation.extVolt)
            else:
                print(self.edgeList.length, self.randList.width, self.randList.seed)


    def IterAlgoNoEdgeVolts(self):
        self._GenerateEdgeVolts()
        scalingFactors = self.randList.randThrVolts / self.edgeVolts
        scalingFactors[self.idxBrokenEdgesDivCap] = 1e10
        scalingFactors[self.idxLeafEdgesDivCap] = 1e10
        idxBrokenEdgeDivCap = int(np.argmin(scalingFactors))
        edgeVoltBrokenEdgeDivCap = self.edgeVolts[idxBrokenEdgeDivCap]
        scalingFactor = scalingFactors[idxBrokenEdgeDivCap]
        idxBrokenEdge = self.edgeList.idxMapFromEdgesDivCapToEdges[idxBrokenEdgeDivCap]

        self._UpdateMatDivCapComb(idxBrokenEdge)
        self._UpdateMatCond(idxBrokenEdge)
        self.idxBrokenEdgesDivCap.append(idxBrokenEdgeDivCap)
        self.extVolts.append(self.extVolts[-1])

        if scalingFactor > 1:
            if edgeVoltBrokenEdgeDivCap > 1e-10:
                self.equation.extVolt *= scalingFactor
                self.equation.nodeVolts *= scalingFactor
                self.extVolts[-1] = float(self.equation.extVolt)
            else:
                print(self.edgeList.length, self.randList.width, self.randList.seed)