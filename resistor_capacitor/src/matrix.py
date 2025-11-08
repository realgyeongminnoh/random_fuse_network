from __future__ import annotations
import numpy as np
from scipy.sparse import csc_array

from .array import Array


class Matrix:
    """
    [ v3, v4, ..., v8 | v12, v13, ..., v26 | -I_ext ] (L=3, for matrices representing the subdivided graph)
    """
    def __init__(self, matrix_init: Matrix = None, array: Array = None, val_cap: float = 1.0, time_step: float = 0.01):
        if matrix_init is None:
            self.val_cap: float = float(val_cap)
            self.time_step: float = float(time_step)
            self.size_cond: int = array.num_node_mid + 2
            self.size_div_comb: int = array.num_node_div_mid + 1
            self.cond: csc_array = self._initialize_cond(array)
            self.div_cap: csc_array = self._initialize_div_cap(array)
            self.div_comb: csc_array = self._initialize_div_comb(array)
        else:
            self.val_cap: float = matrix_init.val_cap
            self.time_step: float = matrix_init.time_step
            self.size_cond: int = matrix_init.size_cond
            self.size_div_comb: int = matrix_init.size_div_comb
            self.cond: csc_array = matrix_init.cond.copy()
            self.div_cap: csc_array = matrix_init.div_cap.copy()
            self.div_comb: csc_array = matrix_init.div_comb.copy()

    def _initialize_cond(self, array: Array):
        length_minus_one = array.length - 1
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

    def _initialize_div_cap(self, array: Array):
        length, length_double = array.length, array.length_double
        val_cap = self.val_cap
        row, col, data = [], [], []

        for idx_node1, idx_node2 in array.edges_div_cap:
            idx_node1_new, idx_node2_new = idx_node1 - length, idx_node2 - length_double
            if idx_node1_new < 0:
                row.append(idx_node2_new)
                col.append(idx_node2_new)
                data.append(val_cap) # [capacitor bond] 12, 13, 14 in 3x3 example
            else:
                row.extend((idx_node1_new, idx_node2_new, idx_node1_new, idx_node2_new))
                col.extend((idx_node1_new, idx_node2_new, idx_node2_new, idx_node1_new))
                data.extend((val_cap, val_cap, -val_cap, -val_cap)) # [capacitor bond] the rest of mid nodes in V of G_div

        row, col, data = np.array(row, dtype=np.int32), np.array(col, dtype=np.int32), np.array(data, dtype=np.float64)
        return csc_array((data, (row, col)), shape=(self.size_div_comb, self.size_div_comb))

    def _initialize_div_comb(self, array: Array):
        length, length_double = array.length, array.length_double
        idx_node_bot_first_minus_one = array.idx_node_bot_first_minus_one
        num_node_div_mid = array.num_node_div_mid
        row, col, data = [], [], []

        row.append(num_node_div_mid)
        col.append(num_node_div_mid)
        data.append(1) # -I_ext(t)'s coefficient
        for idx_node1, idx_node2 in array.edges_div_cond:
            idx_node1_new, idx_node2_new = idx_node1 - length, idx_node2 - length_double
            if idx_node_bot_first_minus_one < idx_node1:
                row.extend((idx_node2_new, num_node_div_mid))
                col.extend((idx_node2_new, idx_node2_new))
                data.extend((1, 1)) # [resistor bond] 23, 25, 26
            else:
                row.extend((idx_node1_new, idx_node2_new, idx_node1_new, idx_node2_new))
                col.extend((idx_node1_new, idx_node2_new, idx_node2_new, idx_node1_new))
                data.extend((1, 1, -1, -1)) # [resistor bond] the rest of mid nodes in V of G_div

        row, col, data = np.array(row, dtype=np.int32), np.array(col, dtype=np.int32), np.array(data, dtype=np.float64)
        div_cond = csc_array((data, (row, col)), shape=(self.size_div_comb, self.size_div_comb))
        return self.div_cap + self.time_step * div_cond # C + Δt * G = div_comb(ined)