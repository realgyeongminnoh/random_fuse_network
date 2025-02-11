import numpy as np

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
        self.idxBrokenEdges = []
        self.idxLeafEdges = []
        self.extVolts = []
        self.dataRawEdgeVolts = []
        self.equation.GetFailure(self) # lazy initialization for rare case: a broken leaf edge


    def _GenerateEdgeVolts(self):
        edgeList = self.edgeList
        nodeVoltsModified = self.equation.nodeVolts[1:-1]
        edgeVolts = self.edgeVolts

        edgeVolts[:edgeList.length] = self.equation.extVolt - nodeVoltsModified[:edgeList.length]
        edgeVolts[edgeList.idxEdgeBot] = nodeVoltsModified[edgeList.idxNode1EdgeBot]
        edgeVolts[edgeList.idxEdgeMid] = nodeVoltsModified[edgeList.idxNode1EdgeMid] - nodeVoltsModified[edgeList.idxNode2EdgeMid]
        
        np.abs(edgeVolts, out=edgeVolts)

    
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
        idxBrokenEdge = int(np.argmax(self.edgeVolts - self.randList.randThrVolts))
        edgeVoltBrokenEdge = self.edgeVolts[idxBrokenEdge]

        self._UpdateMatCond(idxBrokenEdge)
        self.idxBrokenEdges.append(idxBrokenEdge)
        self.extVolts.append(0)
        self.dataRawEdgeVolts.append(self.edgeVolts.copy())

        if edgeVoltBrokenEdge > 1e-10:
            scalingFactor = self.randList.randThrVolts[idxBrokenEdge] / edgeVoltBrokenEdge
            self.equation.extVolt *= scalingFactor
            self.equation.nodeVolts *= scalingFactor
            self.extVolts[0] = float(self.equation.extVolt)


    def IterAlgoNoEdgeVoltsInit(self):
        self._GenerateEdgeVolts()
        idxBrokenEdge = int(np.argmax(self.edgeVolts - self.randList.randThrVolts))
        edgeVoltBrokenEdge = self.edgeVolts[idxBrokenEdge]

        self._UpdateMatCond(idxBrokenEdge)
        self.idxBrokenEdges.append(idxBrokenEdge)
        self.extVolts.append(0)

        if edgeVoltBrokenEdge > 1e-10:
            scalingFactor = self.randList.randThrVolts[idxBrokenEdge] / edgeVoltBrokenEdge
            self.equation.extVolt *= scalingFactor
            self.equation.nodeVolts *= scalingFactor
            self.extVolts[0] = float(self.equation.extVolt)


    def IterAlgoYesEdgeVolts(self):
        self._GenerateEdgeVolts()
        edgeStresses = self.edgeVolts - self.randList.randThrVolts
        edgeStresses[self.idxBrokenEdges] = -1e10
        edgeStresses[self.idxLeafEdges] = -1e10 
        idxBrokenEdge = int(np.argmax(edgeStresses))
        edgeVoltBrokenEdge = self.edgeVolts[idxBrokenEdge]

        self._UpdateMatCond(idxBrokenEdge)
        self.idxBrokenEdges.append(idxBrokenEdge)
        self.extVolts.append(self.extVolts[-1])
        self.dataRawEdgeVolts.append(self.edgeVolts.copy())

        if edgeStresses[idxBrokenEdge] < 0:
            if edgeVoltBrokenEdge > 1e-10:
                scalingFactor = self.randList.randThrVolts[idxBrokenEdge] / edgeVoltBrokenEdge
                self.equation.extVolt *= scalingFactor
                self.equation.nodeVolts *= scalingFactor
                self.extVolts[-1] = float(self.equation.extVolt)


    def IterAlgoNoEdgeVolts(self):
        self._GenerateEdgeVolts()
        edgeStresses = self.edgeVolts - self.randList.randThrVolts
        edgeStresses[self.idxBrokenEdges] = -1e10
        edgeStresses[self.idxLeafEdges] = -1e10 
        idxBrokenEdge = int(np.argmax(edgeStresses))
        edgeVoltBrokenEdge = self.edgeVolts[idxBrokenEdge]

        self._UpdateMatCond(idxBrokenEdge)
        self.idxBrokenEdges.append(idxBrokenEdge)
        self.extVolts.append(self.extVolts[-1])

        if edgeStresses[idxBrokenEdge] < 0:
            if edgeVoltBrokenEdge > 1e-10:
                scalingFactor = self.randList.randThrVolts[idxBrokenEdge] / edgeVoltBrokenEdge
                self.equation.extVolt *= scalingFactor
                self.equation.nodeVolts *= scalingFactor
                self.extVolts[-1] = float(self.equation.extVolt)