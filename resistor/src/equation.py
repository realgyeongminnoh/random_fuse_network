# import warnings
import numpy as np
from scipy.sparse.linalg import spsolve

from src.array import Array
from src.matrix import Matrix

# warnings.filterwarnings("error") # non-dual-graph-based memory-efficient solution to a rare case of "leaf" edges being broken (only in resistor-only model) # degree-based solution implemented


class Equation:
    def __init__(self, array: Array, matrix: Matrix): # , save_volts_profile: bool = False):
        self.array: Array = array
        self.matrix: Matrix = matrix
        self.failure = None
        
        self.volt_ext: float = 1.0
        self.vec_rhs: np.ndarray[np.float64] = np.zeros(self.matrix.size_cond, dtype=np.float64)
        self.volts_node: np.ndarray[np.float64] = None

        # if save_volts_profile:
        #     self._pop_volts_edge_profile_dynamic = self._pop_volts_edge_profile
        # else:
        #     self._pop_volts_edge_profile_dynamic = self._no_op0

    def _lazy_init_failure(self, failure):
        self.failure = failure    

    def solve_mmd(self) -> None:
        self.vec_rhs[-1] = self.volt_ext
        self.volts_node = spsolve(self.matrix.cond, self.vec_rhs, permc_spec="MMD_AT_PLUS_A")

    def solve_amd(self) -> bool:
        self.vec_rhs[-1] = self.volt_ext
        self.volts_node = spsolve(self.matrix.cond, self.vec_rhs, permc_spec="COLAMD")
        return np.abs(self.volts_node[-1]) > 1e-5 # only works with L >= 10 # heuristic non-dual-graph-based solution for macroscopic failure identification

    # def _pop_volts_edge_profile(self):
    #     self.failure.volts_edge_profile.pop()

    # @staticmethod
    # def _no_op0(): pass

    # def _undo_update_matrix_cond(self, idx_edge_leaf):
    #     idx_node1, idx_node2 = self.array.edges[idx_edge_leaf]
    #     idx_node1_new, idx_node2_new = idx_node1 - self.array.length + 1, idx_node2 - self.array.length + 1
    #     cond = self.matrix.cond

    #     if idx_node2_new <= self.array.num_node_mid:
    #         if idx_node1_new > 0:
    #             cond[idx_node1_new, idx_node1_new] += 1
    #             cond[idx_node2_new, idx_node2_new] += 1
    #             cond[idx_node1_new, idx_node2_new] -= 1
    #             cond[idx_node2_new, idx_node1_new] -= 1
    #         else:
    #             cond[0, 0] += 1
    #             cond[idx_node2_new, idx_node2_new] += 1
    #             cond[0, idx_node2_new] -= 1
    #             cond[idx_node2_new, 0] -= 1
    #     else:
    #         cond[idx_node1_new, idx_node1_new] += 1

    # def solve_mmd(self) -> None:
    #     try:
    #         self.vec_rhs[-1] = self.volt_ext
    #         self.volts_node = spsolve(self.matrix.cond, self.vec_rhs, permc_spec="MMD_AT_PLUS_A")
    #     except: # singular matrix iff the immediately preceding broken edge was leaf
    #         idx_edge_leaf = self.failure.idxs_edge_broken.pop()
    #         self.failure.idxs_edge_leaf.append(idx_edge_leaf)

    #         # undo the update of mat_cond and other bookkeeping in the previous step
    #         self._undo_update_matrix_cond(idx_edge_leaf)
    #         self.volt_ext = self.failure.volts_ext[-2]
    #         factor_unscaling = self.volt_ext / self.failure.volts_ext.pop()
    #         if factor_unscaling != 1:
    #             self.volts_node *= factor_unscaling
    #         self._pop_volts_edge_profile_dynamic()

    #         # redo the previous step (break_edge) with the updated idxs_edge_leaf
    #         self.failure.break_edge()
    #         # redo the current step (solve_mmd); recurse in case of another leaf edge being broken
    #         self.solve_mmd()        
    
    # def solve_amd(self) -> bool:
    #     try:
    #         self.vec_rhs[-1] = self.volt_ext
    #         self.volts_node = spsolve(self.matrix.cond, self.vec_rhs, permc_spec="COLAMD")
    #         return np.abs(self.volts_node[-1]) > 1e-5 # only works with L >= 10 # heuristic non-dual-graph-based solution for macroscopic failure identification
    #     except:
    #         idx_edge_leaf = self.failure.idxs_edge_broken.pop()
    #         self.failure.idxs_edge_leaf.append(idx_edge_leaf)

    #         self._undo_update_matrix_cond(idx_edge_leaf)
    #         self.volt_ext = self.failure.volts_ext[-2]
    #         factor_unscaling = self.volt_ext / self.failure.volts_ext.pop()
    #         if factor_unscaling != 1:
    #             self.volts_node *= factor_unscaling
    #         self._pop_volts_edge_profile_dynamic()

    #         self.failure.break_edge()
    #         return self.solve_amd()