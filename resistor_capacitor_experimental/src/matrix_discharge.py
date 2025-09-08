from __future__ import annotations
import numpy as np
from scipy.sparse import csc_array, bmat

from .array import Array


class Matrix:
    """
    [ vtop | v3, v4, ..., v8 | v12, v13, ..., v26 ] (L=3, for matrices representing the subdivided graph)
    """
    def __init__(self, matrix_init: Matrix = None, array: Array = None, cond_ext: float = 1.0, val_cap: float = 1.0, time_step: float = 0.01):
        if matrix_init is None:
            self.cond_ext: float = float(cond_ext)
            self.val_cap: float = float(val_cap)
            self.time_step: float = float(time_step)
            self.size_div_comb: int = array.num_node_div_mid + 1
            self.size_div_block: int = self.size_div_comb + array.num_edge
            self.div_cap: csc_array = self._initialize_div_cap(array)
            self.div_comb: csc_array = None
            self.div_block: csc_array = None
            self._initialize_rest(array)
        else:
            self.cond_ext: float = matrix_init.cond_ext
            self.val_cap: float = matrix_init.val_cap
            self.time_step: float = matrix_init.time_step
            self.size_div_comb: int = matrix_init.size_div_comb
            self.size_div_block: int = matrix_init.size_div_block
            self.div_cap: csc_array = matrix_init.div_cap.copy()
            self.div_comb: csc_array = matrix_init.div_comb.copy()
            self.div_block: csc_array = matrix_init.div_block.copy()

    def _initialize_div_cap(self, array: Array):
        length = array.length
        length_minus_one = array.length - 1
        length_double_minus_one = array.length_double - 1
        val_cap = self.val_cap
        row, col, data = [], [], []

        for idx_node1, idx_node2 in array.edges_div_cap:
            idx_node1_new, idx_node2_new = idx_node1 - length_minus_one, idx_node2 - length_double_minus_one
            if idx_node1 < length: # vtop (v0 v1 v2 in L=3) is no longer known (vs. charging version) (no more dirichlet boundary for the top bus bar)
                row.extend((0, idx_node2_new, 0, idx_node2_new))
                col.extend((0, idx_node2_new, idx_node2_new, 0))
                data.extend((val_cap, val_cap, -val_cap, -val_cap))
            else:
                row.extend((idx_node1_new, idx_node2_new, idx_node1_new, idx_node2_new))
                col.extend((idx_node1_new, idx_node2_new, idx_node2_new, idx_node1_new))
                data.extend((val_cap, val_cap, -val_cap, -val_cap))

        row, col, data = np.array(row, dtype=np.int32), np.array(col, dtype=np.int32), np.array(data, dtype=np.float64)
        return csc_array((data, (row, col)), shape=(self.size_div_comb, self.size_div_comb))

    def _initialize_rest(self, array: Array):
        length = array.length
        length_minus_one = array.length - 1
        length_double_minus_one = array.length_double - 1
        idx_node_bot_first_minus_one = array.idx_node_bot_first_minus_one
        row, col, data = [], [], []

        for idx_cap, (idx_node1, idx_node2) in enumerate(array.edges_div_cap):
            idx_node1_new, idx_node2_new = idx_node1 - length_minus_one, idx_node2 - length_double_minus_one
            
            if idx_node1 < length: # corresponds to vtop
                row.extend((idx_cap, idx_cap))
                col.extend((0, idx_node2_new))
                data.extend((1, -1))
            else:
                row.extend((idx_cap, idx_cap))
                col.extend((idx_node1_new, idx_node2_new))
                data.extend((1, -1))

        row, col, data = np.array(row, dtype=np.int32), np.array(col, dtype=np.int32), np.array(data, dtype=np.float64)
        div_cap_inc = csc_array((data, (row, col)), shape=(array.num_edge, self.size_div_comb)) # div_cap_inc

        row, col, data = [], [], [] # div_cond, div_comb

        row.append(0) # vtop'th KCL equations
        col.append(0) # that corresponds to vtop's coefficient
        data.append(self.cond_ext) # it's from g_ext (vtop - vbot) = 0 (where vbot = 0)
        for idx_node1, idx_node2 in array.edges_div_cond:
            idx_node1_new, idx_node2_new = idx_node1 - length_minus_one, idx_node2 - length_double_minus_one
            if idx_node_bot_first_minus_one < idx_node1:
                row.append(idx_node2_new)
                col.append(idx_node2_new)
                data.append(1)
            else:
                row.extend((idx_node1_new, idx_node2_new, idx_node1_new, idx_node2_new))
                col.extend((idx_node1_new, idx_node2_new, idx_node2_new, idx_node1_new))
                data.extend((1, 1, -1, -1))

        row, col, data = np.array(row, dtype=np.int32), np.array(col, dtype=np.int32), np.array(data, dtype=np.float64)
        div_cond = csc_array((data, (row, col)), shape=(self.size_div_comb, self.size_div_comb))
        self.div_comb = self.div_cap + self.time_step * div_cond # C + Δt * G = div_comb(ined)

        self.div_block = bmat( # matrix used to solve for init cond with nonzero cap volt in the discharging model
            [[div_cond, div_cap_inc.T], [div_cap_inc, csc_array((array.num_edge, array.num_edge))]],
            format="csc",
        )