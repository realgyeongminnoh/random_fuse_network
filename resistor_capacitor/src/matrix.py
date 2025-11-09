from __future__ import annotations
import numpy as np
from scipy.sparse import csc_array

from .array import Array


class Matrix:
    """
    [ v3, v4, ..., v8 | v12, v13, ..., v26 | -I_ext ] (L=3, for matrices representing the subdivided graph)
    """
    def __init__(self, matrix_init: Matrix = None, array: Array = None, val_cap: float = 1.0):
        if matrix_init is None:
            self.val_cap: float = float(val_cap)
            self.val_cap_reciprocal: float = 1.0 / self.val_cap
            self.size_div_comb: int = array.num_node_div_mid + 1
            self.div_cap: csc_array = self._initialize_div_cap(array)
            self.div_comb: csc_array = self._initialize_div_comb(array)
        else:
            self.val_cap: float = matrix_init.val_cap
            self.val_cap_reciprocal: float = matrix_init.val_cap_reciprocal
            self.size_div_comb: int = matrix_init.size_div_comb
            self.div_cap: csc_array = matrix_init.div_cap.copy()
            self.div_comb: csc_array = matrix_init.div_comb.copy()

    def _initialize_div_cap(self, array: Array):
        length, length_double = array.length, array.length_double
        row, col, data = [], [], []

        for idx_node1, idx_node2 in array.edges_div_cap:
            idx_node1_new, idx_node2_new = idx_node1 - length, idx_node2 - length_double
            if idx_node1_new < 0:
                row.append(idx_node2_new)
                col.append(idx_node2_new)
                data.append(1) # [capacitor bond] 12, 13, 14 in 3x3 example
            else:
                row.extend((idx_node1_new, idx_node2_new, idx_node1_new, idx_node2_new))
                col.extend((idx_node1_new, idx_node2_new, idx_node2_new, idx_node1_new))
                data.extend((1, 1, -1, -1)) # [capacitor bond] the rest of mid nodes in V of G_div

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
        return self.div_cap + self.val_cap_reciprocal * div_cond # C0 + (Δt/c) * G = div_comb(ined)