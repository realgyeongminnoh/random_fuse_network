import warnings
import numpy as np
from scipy.sparse.linalg import spsolve, splu, SuperLU

from .array import Array
from .matrix_charge import Matrix as Matrix_charge
from .matrix_discharge import Matrix as Matrix_discharge

warnings.filterwarnings("error") # non-dual-graph-based memory-efficient solution for "island" issue (2> disconnected components of primal graph) but i was overly protective and its never happening after KCL equations fix

class Equation:
    def __init__(self, array: Array, matrix_ch: Matrix_charge, matrix_dch: Matrix_discharge, save_volts_profile: bool = False):
        self.array: Array = array
        self.matrix_ch: Matrix_charge = matrix_ch
        self.matrix_dch: Matrix_discharge = matrix_dch
        self.failure = None

        self.volt_ext: float = 1.0
        self._curr_fpe_proof: float = 1 / (array.length ** 2 - array.length + 1) / 10
        self.div_comb_dch_splu: SuperLU | None = None
        self.vec_rhs: np.ndarray[np.float64] = np.zeros(self.matrix_ch.size_cond, dtype=np.float64)
        self.vec_rhs_div: np.ndarray[np.float64] = np.zeros(self.matrix_ch.size_div_block, dtype=np.float64)
        self.volts_node_div: np.ndarray[np.float64] = np.empty(self.matrix_ch.size_div_comb, dtype=np.float64)

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

    def solve_true_init_ch(self) -> None: # for zero volts_cap
        self.vec_rhs[-1] = self.volt_ext
        volts_node = spsolve(self.matrix_ch.cond, self.vec_rhs, permc_spec="MMD_AT_PLUS_A")
        
        length, length_double = self.array.length, self.array.length_double
        volts_node_div = self.volts_node_div

        volts_node_div[-1] = volts_node[-1]
        volts_node_div[:self.array.num_node_mid] = volts_node[1:-1]
        volts_node_div[self.array.num_node_mid:self.array.num_node_mid + length] = self.volt_ext
        for idx_node1, idx_node2 in self.array.edges_div_cap[length:]:
            volts_node_div[idx_node2 - length_double] = volts_node_div[idx_node1 - length]

    def solve_init_ch(self) -> None: # NOT TIME ADVANCE # for nonzero volts_cap mid simulation (discharge -> charge), also works for zero volts_cap ...
        self.vec_rhs_div[self.matrix_ch.size_div_comb:] = self.failure.volts_cap
        self.vec_rhs_div[self.matrix_ch.size_div_comb:self.matrix_ch.size_div_comb + self.array.length] -= self.volt_ext
        self.volts_node_div[:] = spsolve(self.matrix_ch.div_block, self.vec_rhs_div, permc_spec="MMD_ATA")[:self.matrix_ch.size_div_comb]

    def solve_init_dch(self) -> None: # NOT TIME ADVANCE # for nonzero volts_cap mid simulation (charge -> discharge) 
        self.vec_rhs_div[self.matrix_dch.size_div_comb:] = self.failure.volts_cap
        self.volts_node_div[:] = spsolve(self.matrix_dch.div_block, self.vec_rhs_div, permc_spec="MMD_ATA")[:self.matrix_dch.size_div_comb]
        self.failure._last_time_step_broken = True
          
    def solve_ch(self) -> None:
        self.volts_node_div = self.matrix_ch.div_cap @ self.volts_node_div
        self.volts_node_div[-1] = 0
        self.volts_node_div = spsolve(self.matrix_ch.div_comb, self.volts_node_div, permc_spec="MMD_AT_PLUS_A")

    def solve_dch(self) -> None:
        self.volts_node_div = self.matrix_dch.div_cap @ self.volts_node_div
        if self.failure._last_time_step_broken or self.div_comb_dch_splu is None:
            self.div_comb_dch_splu = splu(self.matrix_dch.div_comb, permc_spec="MMD_AT_PLUS_A")
        self.volts_node_div = self.div_comb_dch_splu.solve(self.volts_node_div)

    def _undo_update_matrix_ch(self, idx_edge_island):
        array = self.array
        idx_node1, idx_node2 = array.edges_div_cond[idx_edge_island]
        idx_node1_new, idx_node2_new = idx_node1 - array.length, idx_node2 - array.length_double
        div_comb = self.matrix_ch.div_comb
        div_block = self.matrix_ch.div_block
        time_step = self.matrix_ch.time_step

        if array.idx_node_bot_first_minus_one < idx_node1:
            div_comb[array.num_node_div_mid, idx_node2_new] += time_step
            div_comb[idx_node2_new, idx_node2_new] += time_step
            div_block[array.num_node_div_mid, idx_node2_new] += 1.0
            div_block[idx_node2_new, idx_node2_new] += 1.0
        else:
            div_comb[idx_node1_new, idx_node1_new] += time_step
            div_comb[idx_node2_new, idx_node2_new] += time_step
            div_comb[idx_node1_new, idx_node2_new] -= time_step
            div_comb[idx_node2_new, idx_node1_new] -= time_step
            div_block[idx_node1_new, idx_node1_new] += 1.0
            div_block[idx_node2_new, idx_node2_new] += 1.0
            div_block[idx_node1_new, idx_node2_new] -= 1.0
            div_block[idx_node2_new, idx_node1_new] -= 1.0

        idx_node1, idx_node2 = array.edges[idx_edge_island]
        idx_node1_new, idx_node2_new = idx_node1 - array.length + 1, idx_node2 - array.length + 1
        cond = self.matrix_ch.cond

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

    def _undo_update_matrix_dch(self, idx_edge_island):
        array = self.array
        idx_node1, idx_node2 = array.edges_div_cond[idx_edge_island]
        idx_node1_new, idx_node2_new = idx_node1 - array.length + 1, idx_node2 - array.length_double + 1
        div_comb = self.matrix_dch.div_comb
        div_block = self.matrix_dch.div_block
        time_step = self.matrix_dch.time_step

        if array.idx_node_bot_first_minus_one < idx_node1:
            div_comb[idx_node2_new, idx_node2_new] += time_step
            div_block[idx_node2_new, idx_node2_new] += 1.0
        else:
            div_comb[idx_node1_new, idx_node1_new] += time_step
            div_comb[idx_node2_new, idx_node2_new] += time_step
            div_comb[idx_node1_new, idx_node2_new] -= time_step
            div_comb[idx_node2_new, idx_node1_new] -= time_step
            div_block[idx_node1_new, idx_node1_new] += 1.0
            div_block[idx_node2_new, idx_node2_new] += 1.0
            div_block[idx_node1_new, idx_node2_new] -= 1.0
            div_block[idx_node2_new, idx_node1_new] -= 1.0

    def solve_r_amd(self) -> bool:
        try:
            self.vec_rhs[-1] = 1.0
            return np.abs(spsolve(self.matrix_ch.cond, self.vec_rhs, permc_spec="COLAMD")[-1]) > self._curr_fpe_proof # 1e-5
        except:
            idx_edge_island = self.failure.idxs_edge_broken.pop()
            self.failure.idxs_edge_island.append(idx_edge_island)

            idx_node1, idx_node2 = self.array.edges[idx_edge_island]
            self.failure.degrees[idx_node1] += 1
            self.failure.degrees[idx_node2] += 1

            self._undo_update_matrix_ch(idx_edge_island)
            self._undo_update_matrix_dch(idx_edge_island)
            self.failure.volts_ext.pop()
            self.volt_ext = self.failure.volts_ext[-1]
            if self.volts_node_div_prev is not None:
                self.volts_node_div[:] = self.volts_node_div_prev
                self.volts_node_div_prev = None

            self._pop_volts_edge_profile_dynamic()
            self._pop_volts_cap_profile_dynamic()
            self._pop_volts_cond_profile_dynamic()

            if self.failure._last_time_step_break_edge_type:
                self.failure.break_edge_ch()
            else:
                self.failure.break_edge_dch()
        return self.solve_r_amd()