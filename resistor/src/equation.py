import numpy as np
from scipy.sparse.linalg import spsolve

from .array import Array
from .matrix import Matrix


class Equation:
    def __init__(self, array: Array, matrix: Matrix, save_volts_profile: bool = False):
        self.array: Array = array
        self.matrix: Matrix = matrix
        self.failure = None

        self.volt_ext: float = 1.0
        self.vec_rhs: np.ndarray[np.float64] = np.zeros(self.matrix.size_cond, dtype=np.float64)
        self.volts_node: np.ndarray[np.float64] = None

        if save_volts_profile:
            self._pop_volts_edge_profile_dynamic = self._pop_volts_edge_profile
        else:
            self._pop_volts_edge_profile_dynamic = self._no_op0

    def _lazy_init_failure(self, failure):
        self.failure = failure

    def _pop_volts_edge_profile(self):
        self.failure.volts_edge_profile.pop()

    @staticmethod
    def _no_op0(): pass

    def _undo_update_matrix(self, idx_edge_island):
        idx_node1, idx_node2 = self.array.edges[idx_edge_island]
        idx_node1_new, idx_node2_new = idx_node1 - self.array.length + 1, idx_node2 - self.array.length + 1
        cond = self.matrix.cond

        if idx_node2_new <= self.array.num_node_mid:
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
        self.volts_node = np.concatenate(([1.0], np.linspace((length - 1) / length, 1 / length, length - 1).repeat(length), [-1.0]))

    def solve_mmd(self) -> None:
        self.vec_rhs[-1] = self.volt_ext
        self.volts_node = spsolve(self.matrix.cond, self.vec_rhs, permc_spec="MMD_AT_PLUS_A")

    def solve_amd(self) -> None:
        self.vec_rhs[-1] = self.volt_ext
        self.volts_node = spsolve(self.matrix.cond, self.vec_rhs, permc_spec="COLAMD")

    def check_graph(self) -> bool:
        while True:
            idx_edge_broken = self.failure.idxs_edge_broken[-1]
            flag = self.failure._dsu_update(idx_edge_broken)
            if flag == 0: # no cycle
                return True
            if flag == 1: # termination cycle
                return False
            
            # island cycle
            idx_edge_island = self.failure.idxs_edge_broken.pop()
            self.failure.idxs_edge_island.append(idx_edge_island)

            idx_node1, idx_node2 = self.array.edges[idx_edge_island]
            self.failure.degrees[idx_node1] += 1
            self.failure.degrees[idx_node2] += 1

            self._undo_update_matrix(idx_edge_island)
            self.volt_ext = self.failure.volts_ext[-2]
            factor_unscaling = self.volt_ext / self.failure.volts_ext.pop()
            if factor_unscaling != 1:
                self.volts_node *= factor_unscaling

            self._pop_volts_edge_profile_dynamic()
            self.failure.break_edge()