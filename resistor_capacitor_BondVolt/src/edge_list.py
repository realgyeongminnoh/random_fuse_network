import numpy as np

class EdgeList:
    def __init__(self, length: int):
        self.length = length
        self.numNode = length ** 2 + length
        self.numNodeMid = self.numNode - 2 * length
        self.numEdge = 2 * length ** 2 - length
        self.numNodeDiv = self.numNode + self.numEdge
        self.numNodeDivMid = self.numNodeDiv - 2 * length
        self.numEdgeDiv = 2 * self.numEdge
        self.sizeMatCond = self.numNodeMid + 2
        self.sizeMatDivComb = self.numNodeDivMid + 1
        self.edges, self.edgesDiv = [], []
        self.edgesDivCap, self.edgesDivCond = [], []
        self.idxMapFromEdgesToEdgesDivCond = None 
        self._GenerateEdges()
        self._GenerateEdgesDiv()
        self._GenerateEdgesDivCapCond()
        self._GenerateIdxMapForCond()


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


    def _GenerateEdgesDiv(self):
        edges, edgesDiv = self.edges, self.edgesDiv
        idxNode = self.numNode

        for idxNode1, idxNode2 in edges:
            edgesDiv.extend(((idxNode1, idxNode), (idxNode2, idxNode)))
            idxNode += 1

        edgesDiv.sort(key=lambda x: x)       


    def _GenerateEdgesDivCapCond(self):
        length, lengthMinusOne = self.length, self.length - 1
        numEdgeDivMinusLength = self.numEdgeDiv - length
        edgesDivCap, edgesDivCond = self.edgesDivCap, self.edgesDivCond
        counter = 0

        for idxEdgeDiv, edgeDiv in enumerate(self.edgesDiv):
            # edgeDivTop
            if idxEdgeDiv < length:
                edgesDivCap.append(edgeDiv)
            # edgeDivMid
            elif lengthMinusOne < idxEdgeDiv < numEdgeDivMinusLength:
                if edgeDiv[0] % length in (0, lengthMinusOne):
                    if counter in (2, 3):
                        edgesDivCap.append(edgeDiv)
                    else:
                        edgesDivCond.append(edgeDiv)
                    counter = (counter + 1) % 4
                else:
                    if counter in (1, 3): 
                        edgesDivCap.append(edgeDiv)
                    else:
                        edgesDivCond.append(edgeDiv)
                    counter = (counter + 1) % 4
            # edgeDivBot
            else:
                edgesDivCond.append(edgeDiv)


    def _GenerateIdxMapForCond(self):
        # edgesDivCond[idxMap[idxEdge]] -> edges[idxEdge]
        self.idxMapFromEdgesToEdgesDivCond = np.array(sorted(range(self.numEdge), key=lambda idxEdge: self.edgesDivCond[idxEdge][1]), dtype=np.int32)