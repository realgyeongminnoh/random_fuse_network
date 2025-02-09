import numpy as np
from scipy.sparse import csc_array

from src.edge_list import EdgeList
from src.matrix import Matrix
from src.rand_list import RandList
from src.equation import Equation


class Failure:
    def __init__(self, edgeList: EdgeList, matrix: Matrix, randList: RandList, equation: Equation):
        self.edgeList = edgeList
        self.matrix = matrix
        self.randList = randList
        self.equation = equation
        self.edgeVolts = np.empty(self.edgeList.numEdge)
        self.idxBrokenEdgesDivCap = []
        # self.idxLeafEdges = []
        self.extVolts = []
        self.dataRawEdgeVolts = []


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
        length = self.edgeList.length
        lengthDoubled = 2 * length
        numNode, numNodeDivMid = self.edgeList.numNode, self.edgeList.numNodeDivMid
        idxNodeBotMinusOne = numNode - length - 1
        idxBrokenEdgeDivCond = self.edgeList.idxMapFromEdgesToEdgesDivCond[idxBrokenEdge]
        idxNode1, idxNode2 = self.edgeList.edgesDivCond[idxBrokenEdgeDivCond]
        idxNode1New, idxNode2New = idxNode1 - length, idxNode2 - lengthDoubled

        if idxNodeBotMinusOne < idxNode1:
            self.matrix.matDivCap[numNodeDivMid, idxNode2New] -= 1
            self.matrix.matDivComb[numNodeDivMid, idxNode2New] -= 1
            self.matrix.matDivComb[idxNode2New, idxNode2New] -= 0.01
        else:
            self.matrix.matDivComb[idxNode1New, idxNode1New] -= 0.01
            self.matrix.matDivComb[idxNode2New, idxNode2New] -= 0.01
            self.matrix.matDivComb[idxNode1New, idxNode2New] += 0.01
            self.matrix.matDivComb[idxNode2New, idxNode1New] += 0.01


    def _UpdateMatCond(self, idxBrokenEdge):
        lengthMinusOne  = self.edgeList.length - 1
        idxNode1, idxNode2 = self.edgeList.edges[idxBrokenEdge]
        idxNode1New, idxNode2New = idxNode1 - lengthMinusOne, idxNode2 - lengthMinusOne

        if idxNode2New <= self.edgeList.numNodeMid:
            data = np.array([-1, -1, 1, 1], dtype=np.int16)
            if idxNode1New > 0:
                row = np.array([idxNode1New, idxNode2New, idxNode1New, idxNode2New], dtype=np.int32)
                col = np.array([idxNode1New, idxNode2New, idxNode2New, idxNode1New], dtype=np.int32)
            else:
                row = np.array([0, idxNode2New, 0, idxNode2New], dtype=np.int32)
                col = np.array([0, idxNode2New, idxNode2New, 0], dtype=np.int32)
        else:
            row = np.array([idxNode1New], dtype=np.int32)
            col = np.array([idxNode1New], dtype=np.int32)
            data = np.array([-1], dtype=np.int16)

        matCondDelta = csc_array((data, (row, col)), shape=(self.edgeList.sizeMatCond, self.edgeList.sizeMatCond))
        self.matrix.matCond += matCondDelta


    def IterAlgoYesEdgeVoltsInit(self):
        edgeStresses = -self.randList.randThrVolts

        self.idxBrokenEdgesDivCap.append(int(np.argmax(edgeStresses)))
        self.extVolts.append(0)
        self.dataRawEdgeVolts.append(np.zeros(self.edgeList.numEdge))

        idxBrokenEdge = self.edgeList.idxMapFromEdgesDivCapToEdges[np.argmax(edgeStresses)]
        self._UpdateMatDivCapComb(idxBrokenEdge)
        self._UpdateMatCond(idxBrokenEdge)


    def IterAlgoNoEdgeVoltsInit(self):
        pass


    def IterAlgoYesEdgeVolts(self):
        self._GenerateEdgeVolts()
        edgeStresses = self.edgeVolts - self.randList.randThrVolts
        edgeStresses[self.idxBrokenEdgesDivCap] = -1e10
        idxBrokenEdgeDivCap = int(np.argmax(edgeStresses))
        edgeVoltBrokenEdgeDivCap = self.edgeVolts[idxBrokenEdgeDivCap]

        self.idxBrokenEdgesDivCap.append(idxBrokenEdgeDivCap)
        self.extVolts.append(self.extVolts[-1])
        self.dataRawEdgeVolts.append(self.edgeVolts.copy())

        if edgeStresses[idxBrokenEdgeDivCap] < 0:
            if edgeVoltBrokenEdgeDivCap > 1e-10:
                scalingFactor = self.randList.randThrVolts[idxBrokenEdgeDivCap] / edgeVoltBrokenEdgeDivCap
                self.equation.extVolt *= scalingFactor
                self.equation.nodeVolts *= scalingFactor
                self.extVolts[-1] = float(self.equation.extVolt)

        idxBrokenEdge = self.edgeList.idxMapFromEdgesDivCapToEdges[idxBrokenEdgeDivCap]
        self._UpdateMatDivCapComb(idxBrokenEdge)
        self._UpdateMatCond(idxBrokenEdge)


    def IterAlgoNoEdgeVolts(self):
        pass