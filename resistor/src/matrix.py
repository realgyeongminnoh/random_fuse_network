import numpy as np
from scipy.sparse import csc_array

from src.edge_list import EdgeList


class Matrix:
    def __init__(self, edgeList: EdgeList, matCondInit = None):
        self.edgeList = edgeList
        self.matCond = matCondInit
        if self.matCond is None:
            self._GenerateMatCond()
        

    def _GenerateMatCond(self):
        length = self.edgeList.length
        idxEdgeTop = self.edgeList.idxEdgeTop
        idxNode1, idxNode2 = self.edgeList.idxNode1EdgeMid, self.edgeList.idxNode2EdgeMid
        sizeTop = 4 * length
        sizeMid = 4 * (self.edgeList.numEdge - 2 * length)
        rowTop, colTop = np.empty(sizeTop, dtype=np.int32), np.empty(sizeTop, dtype=np.int32)
        rowMid, colMid = np.empty(sizeMid, dtype=np.int32), np.empty(sizeMid, dtype=np.int32)

        rowTop[0::4], rowTop[1::4], rowTop[2::4], rowTop[3::4] = -1, idxEdgeTop, -1, idxEdgeTop 
        colTop[0::4], colTop[1::4], colTop[2::4], colTop[3::4] = -1, idxEdgeTop, idxEdgeTop, -1
        rowMid[0::4], rowMid[1::4], rowMid[2::4], rowMid[3::4] = idxNode1, idxNode2, idxNode1, idxNode2
        colMid[0::4], colMid[1::4], colMid[2::4], colMid[3::4] = idxNode1, idxNode2, idxNode2, idxNode1

        dataTop = np.tile([1, 1, -1, -1], length).astype(np.int16)
        dataMid = np.tile([1, 1, -1, -1], int(sizeMid / 4)).astype(np.int16)

        row = np.concatenate((rowTop, rowMid, self.edgeList.idxNode1EdgeBot)) + 1
        col = np.concatenate((colTop, colMid, self.edgeList.idxNode1EdgeBot)) + 1
        data = np.concatenate((dataTop, dataMid, np.ones(length, dtype=np.int16)))

        row = np.append(row, np.array([0, self.edgeList.numNodeMid + 1], dtype=np.int32))
        col = np.append(col, np.array([self.edgeList.numNodeMid + 1, 0], dtype=np.int32))
        data = np.append(data, np.array([1, 1], dtype=np.int16))

        self.matCond = csc_array((data, (row, col)), shape=(self.edgeList.sizeMatCond, self.edgeList.sizeMatCond))