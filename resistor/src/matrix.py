from __future__ import annotations
import numpy as np
from scipy.sparse import csc_array

from .array import Array


class Matrix:
    """
    [ vtop | v3, v4, ..., v8 | -I_ext ] (L=3)
    """
    def __init__(self, matrix_init: Matrix = None, array: Array = None):
        if matrix_init is None:
            self.size_cond: int = array.num_node_mid + 2
            self.cond: csc_array = self._initialize_cond(array)
        else:
            self.size_cond: int = matrix_init.size_cond
            self.cond: csc_array = matrix_init.cond.copy()

    def _initialize_cond(self, array: Array):
        length = array.length
        length_minus_one = length - 1
        num_node_mid = array.num_node_mid
        row, col, data = [], [], []

        row.extend((0, num_node_mid + 1))
        col.extend((num_node_mid + 1, 0))
        data.extend((1, 1))
        for idx_node1, idx_node2 in array.edges:
            idx_node1_new, idx_node2_new = idx_node1 - length_minus_one, idx_node2 - length_minus_one
            if idx_node1_new < 1:
                row.extend((0, idx_node2_new, 0, idx_node2_new))
                col.extend((0, idx_node2_new, idx_node2_new, 0))
                data.extend((1, 1, -1, -1))
            elif idx_node2_new > num_node_mid:
                row.append(idx_node1_new)
                col.append(idx_node1_new)
                data.append(1)
            else:
                row.extend((idx_node1_new, idx_node2_new, idx_node1_new, idx_node2_new))
                col.extend((idx_node1_new, idx_node2_new, idx_node2_new, idx_node1_new))
                data.extend((1, 1, -1, -1))

        row, col, data = np.array(row, dtype=np.int32), np.array(col, dtype=np.int32), np.array(data, dtype=np.float64)
        return csc_array((data, (row, col)), shape=(self.size_cond, self.size_cond))