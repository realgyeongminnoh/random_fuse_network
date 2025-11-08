import warnings
import numpy as np
from scipy.sparse.linalg import spsolve, splu

from .array import Array
from .matrix import Matrix

warnings.filterwarnings("error") # non-dual-graph-based memory-efficient solution for "island" issue (2> disconnected components of primal graph) but i was overly protective and its never happening after KCL equations fix


class Equation:
    def __init__(self, array: Array, matrix: Matrix, save_volts_profile: bool = False):
        self.array: Array = array
        self.matrix: Matrix = matrix
        self.failure = None

        self.volt_ext: float = 1.0
        self.vec_rhs: np.ndarray[np.float64] = np.zeros(self.matrix.size_cond, dtype=np.float64)
        self.volts_node_div: np.ndarray[np.float64] = np.empty(self.matrix.size_div_comb, dtype=np.float64)
        self.volts_node_div_prev: np.ndarray[np.float64] | None = None

        if save_volts_profile:
            self._pop_volts_edge_profile_dynamic = self._pop_volts_edge_profile
            self._pop_volts_cap_profile_dynamic = self._pop_volts_cap_profile
            self._pop_volts_cond_profile_dynamic = self._pop_volts_cond_profile
        else:
            self._pop_volts_edge_profile_dynamic = self._no_op0
            self._pop_volts_cap_profile_dynamic = self._no_op0
            self._pop_volts_cond_profile_dynamic = self._no_op0

    def _lazy_init_failure(self, failure):
        self.failure = failure

    def _pop_volts_edge_profile(self):
        self.failure.volts_edge_profile.pop()

    def _pop_volts_cap_profile(self):
        self.failure.volts_cap_profile.pop()

    def _pop_volts_cond_profile(self):
        self.failure.volts_cond_profile.pop()

    @staticmethod
    def _no_op0(): pass

    def _undo_update_matrix(self, idx_edge_island):
        array = self.array
        idx_node1, idx_node2 = array.edges_div_cond[idx_edge_island]
        idx_node1_new, idx_node2_new = idx_node1 - array.length, idx_node2 - array.length_double
        div_comb = self.matrix.div_comb
        time_step = self.matrix.time_step

        if array.idx_node_bot_first_minus_one < idx_node1:
            div_comb[array.num_node_div_mid, idx_node2_new] += time_step
            div_comb[idx_node2_new, idx_node2_new] += time_step
        else:
            div_comb[idx_node1_new, idx_node1_new] += time_step
            div_comb[idx_node2_new, idx_node2_new] += time_step
            div_comb[idx_node1_new, idx_node2_new] -= time_step
            div_comb[idx_node2_new, idx_node1_new] -= time_step

        idx_node1, idx_node2 = array.edges[idx_edge_island]
        idx_node1_new, idx_node2_new = idx_node1 - array.length + 1, idx_node2 - array.length + 1
        cond = self.matrix.cond

        if idx_node2_new <= array.num_node_mid:
            if idx_node1_new > 0:
                cond[idx_node1_new, idx_node1_new] += 1.0
                cond[idx_node2_new, idx_node2_new] += 1.0
                cond[idx_node1_new, idx_node2_new] -= 1.0
                cond[idx_node2_new, idx_node1_new] -= 1.0
            else:
                cond[0, 0] += 1.0
                cond[idx_node2_new, idx_node2_new] += 1.0
                cond[0, idx_node2_new] -= 1.0
                cond[idx_node2_new, 0] -= 1.0
        else:
            cond[idx_node1_new, idx_node1_new] += 1.0

    def solve_init(self) -> None:
        length = self.array.length
        volts_node = np.concatenate(([1.0], np.linspace((length - 1) / length, 1 / length, length - 1).repeat(length), [-1.0]))

        length, length_double = self.array.length, self.array.length_double
        volts_node_div = self.volts_node_div

        volts_node_div[-1] = volts_node[-1]
        volts_node_div[:self.array.num_node_mid] = volts_node[1:-1]
        volts_node_div[self.array.num_node_mid:self.array.num_node_mid + length] = self.volt_ext
        for idx_node1, idx_node2 in self.array.edges_div_cap[length:]:
            volts_node_div[idx_node2 - length_double] = volts_node_div[idx_node1 - length]

    def solve_r_mmd(self) -> None:
        try:
            self.vec_rhs[-1] = self.volt_ext
            splu(self.matrix.cond, permc_spec="MMD_AT_PLUS_A")
        except:
            idx_edge_island = self.failure.idxs_edge_broken.pop()
            self.failure.idxs_edge_island.append(idx_edge_island)
            
            idx_node1, idx_node2 = self.array.edges[idx_edge_island]
            self.failure.degrees[idx_node1] += 1
            self.failure.degrees[idx_node2] += 1

            self._undo_update_matrix(idx_edge_island)
            self.failure.volts_ext.pop()
            self.volt_ext = self.failure.volts_ext[-1]
            if self.volts_node_div_prev is not None:
                self.volts_node_div[:] = self.volts_node_div_prev
                self.volts_node_div_prev = None
                
            self._pop_volts_edge_profile_dynamic()
            self._pop_volts_cap_profile_dynamic()
            self._pop_volts_cond_profile_dynamic()

            self.failure.break_edge()
            self.solve_r_mmd()
    
    def solve_r_amd(self) -> bool:
        try:
            self.vec_rhs[-1] = self.volt_ext
            return np.abs(spsolve(self.matrix.cond, self.vec_rhs, permc_spec="COLAMD")[-1]) > 1e-5
        except:
            idx_edge_island = self.failure.idxs_edge_broken.pop()
            self.failure.idxs_edge_island.append(idx_edge_island)
            
            idx_node1, idx_node2 = self.array.edges[idx_edge_island]
            self.failure.degrees[idx_node1] += 1
            self.failure.degrees[idx_node2] += 1

            self._undo_update_matrix(idx_edge_island)
            self.failure.volts_ext.pop()
            self.volt_ext = self.failure.volts_ext[-1]
            if self.volts_node_div_prev is not None:
                self.volts_node_div[:] = self.volts_node_div_prev
                self.volts_node_div_prev = None
                
            self._pop_volts_edge_profile_dynamic()
            self._pop_volts_cap_profile_dynamic()
            self._pop_volts_cond_profile_dynamic()

            self.failure.break_edge()
            return self.solve_r_amd()

    def solve(self) -> None:
        self.volts_node_div = self.matrix.div_cap @ self.volts_node_div
        self.volts_node_div[-1] = 0 # for numerical stability # external current of RC cannot be used as a heuristic termination condition
        self.volts_node_div = spsolve(self.matrix.div_comb, self.volts_node_div, permc_spec="MMD_AT_PLUS_A")