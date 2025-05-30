import numpy as np
from scipy.sparse import csc_array

from src.edge_list import EdgeList


class Matrix:
    def __init__(self, edgeList: EdgeList, valCap: float = 1, matCondInit = None, matDivCapInit = None, matDivCombInit = None):
        self.edgeList = edgeList
        self.valCap = float(valCap)
        self.matCond, self.matDivCap, self.matDivComb = matCondInit, matDivCapInit, matDivCombInit
        if self.matCond is None:
            self._GenerateMatCond()
            self._GenerateMatDivCap()
            self._GenerateMatDivComb()


    def _GenerateMatCond(self):
        lengthMinusOne = self.edgeList.length - 1
        numNodeMid = self.edgeList.numNodeMid
        row, col, data = [], [], []

        row.extend((0, numNodeMid + 1))
        col.extend((numNodeMid + 1, 0))
        data.extend((1, 1))
        for idxNode1, idxNode2 in self.edgeList.edges:
            idxNode1New, idxNode2New = idxNode1 - lengthMinusOne, idxNode2 - lengthMinusOne
            if idxNode1New < 1:
                row.extend((0, idxNode2New, 0, idxNode2New))
                col.extend((0, idxNode2New, idxNode2New, 0))
                data.extend((1, 1, -1, -1))
            elif idxNode2New > numNodeMid:
                row.append(idxNode1New)
                col.append(idxNode1New)
                data.append(1)
            else:
                row.extend((idxNode1New, idxNode2New, idxNode1New, idxNode2New))
                col.extend((idxNode1New, idxNode2New, idxNode2New, idxNode1New))
                data.extend((1, 1, -1, -1))

        row, col, data = np.array(row, dtype=np.int32), np.array(col, dtype=np.int32), np.array(data, dtype=np.int32)
        self.matCond = csc_array((data, (row, col)), shape=(self.edgeList.sizeMatCond, self.edgeList.sizeMatCond))


    def _GenerateMatDivCap(self):
        length, lengthDoubled = self.edgeList.length, 2 * self.edgeList.length
        numNodeDivMid = self.edgeList.numNodeDivMid
        valCap = self.valCap
        row, col, data = [], [], []

        row.append(numNodeDivMid)
        col.append(numNodeDivMid)
        data.append(1)
        for idxNode1, idxNode2 in self.edgeList.edgesDivCond[-length:]:
            row.append(numNodeDivMid)
            col.append(idxNode2 - lengthDoubled)
            data.append(1)
        for idxNode1, idxNode2 in self.edgeList.edgesDivCap:
            idxNode1New, idxNode2New = idxNode1 - length, idxNode2 - lengthDoubled
            if idxNode1New < 0:
                row.append(idxNode2New)
                col.append(idxNode2New)
                data.append(valCap)
            else:
                row.extend((idxNode1New, idxNode2New, idxNode1New, idxNode2New))
                col.extend((idxNode1New, idxNode2New, idxNode2New, idxNode1New))
                data.extend((valCap, valCap, -valCap, -valCap))

        row, col, data = np.array(row, dtype=np.int32), np.array(col, dtype=np.int32), np.array(data, dtype=np.float64)
        self.matDivCap = csc_array((data, (row, col)), shape=(self.edgeList.sizeMatDivComb, self.edgeList.sizeMatDivComb))


    def _GenerateMatDivComb(self):
        length, lengthDoubled = self.edgeList.length, 2 * self.edgeList.length
        idxNodeBotMinusOne = self.edgeList.numNode - length - 1
        row, col, data = [], [], []

        for idxNode1, idxNode2 in self.edgeList.edgesDivCond:
            idxNode1New, idxNode2New = idxNode1 - length, idxNode2 - lengthDoubled
            if idxNodeBotMinusOne < idxNode1:
                row.append(idxNode2New)
                col.append(idxNode2New)
                data.append(1)
            else:
                row.extend((idxNode1New, idxNode2New, idxNode1New, idxNode2New))
                col.extend((idxNode1New, idxNode2New, idxNode2New, idxNode1New))
                data.extend((1, 1, -1, -1))
                
        row, col, data = np.array(row, dtype=np.int32), np.array(col, dtype=np.int32), np.array(data, dtype=np.float64)
        self.matDivComb = csc_array((data, (row, col)), shape=(self.edgeList.sizeMatDivComb, self.edgeList.sizeMatDivComb))
        self.matDivComb *= 0.01
        self.matDivComb += self.matDivCap