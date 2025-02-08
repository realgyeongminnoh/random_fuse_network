import numpy as np


class EdgeList:
    def __init__(self, length: int):
        self.length = length
        self.numNode = length ** 2 + length
        self.numNodeMid = self.numNode - 2 * length
        self.numEdge = 2 * length ** 2 - length
        self.sizeMatCond = self.numNodeMid + 2
        self.edges = []
        self.idxEdgeTop, self.idxEdgeBot, self.idxEdgeMid = None, None, None
        self.idxNode1EdgeBot, self.idxNode1EdgeMid, self.idxNode2EdgeMid = None, None, None
        self._GenerateEdges()
        self._GenerateIdxNodeEdgeSets()


    def _GenerateEdges(self):
        length = self.length
        lengthMinusOne, lengthSquared = length - 1, length ** 2
        edges = self.edges
        idxNode = 0
        
        for j in range(length + 1):
            for i in range(length):
                if i < lengthMinusOne:
                    if lengthMinusOne < idxNode < lengthSquared:
                        edges.append((idxNode, idxNode + 1))
                        if i == 0:
                            edges.append((idxNode, idxNode + lengthMinusOne))
                if j < length: 
                    edges.append((idxNode, idxNode + length))
                idxNode += 1


    def _GenerateIdxNodeEdgeSets(self):
        self.idxEdgeTop = np.arange(self.length, dtype=np.int32)
        self.idxEdgeBot = self.numEdge + np.append(np.arange((-2 * self.length + 2), 0, 2, dtype=np.int32), np.int32(-1))
        self.idxEdgeMid = np.delete(np.arange(self.numEdge), np.append(self.idxEdgeTop, self.idxEdgeBot))
        self.idxNode1EdgeBot = self.numNodeMid + self.idxEdgeTop - self.length
        edgesMinusLengthNumpy = np.array(self.edges, dtype=np.int32) - self.length
        self.idxNode1EdgeMid = edgesMinusLengthNumpy[self.idxEdgeMid, 0]
        self.idxNode2EdgeMid = edgesMinusLengthNumpy[self.idxEdgeMid, 1]