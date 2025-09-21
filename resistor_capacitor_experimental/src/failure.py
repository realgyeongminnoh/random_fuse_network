import warnings
import numpy as np

from .array import Array
from .matrix_charge import Matrix as Matrix_charge
from .matrix_discharge import Matrix as Matrix_discharge
from .equation import Equation

warnings.simplefilter("ignore", category=RuntimeWarning) # division by zero (from broken / leaf edge execlusion) ignore because inf do not cause issue for argmin


class Failure:
    def __init__(
        self, array: Array, matrix_ch: Matrix_charge, matrix_dch: Matrix_discharge, equation: Equation,
        width: float, seed: int, save_volts_profile: bool = False, _test1: bool = True, _test2: bool = True,
    ):
        self.array: Array = array
        self.matrix_ch: Matrix_charge = matrix_ch
        self.matrix_dch: Matrix_discharge = matrix_dch
        self.equation: Equation = equation
        self.equation._lazy_init_failure(self)
        self.width = float(width)
        self.seed = int(seed)
        self.breaking_strengths: np.ndarray[np.float64] = self._generate_breaking_strengths() # 1. bond voltage for now (2. capacitor voltage can only lead to charge-nonconservation scaling mechanism; 3. resistor voltage = resistor current = capacitor current = bond current implications are unclear yet)

        self.degrees: np.ndarray[np.int32] = np.array([1] * array.length + [4] * array.num_node_mid + [1] * array.length, dtype=np.int32)
        self.volts_edge: np.ndarray[np.float64] = np.empty(self.array.num_edge, dtype=np.float64)
        self.volts_cap: np.ndarray[np.float64] = np.zeros(self.array.num_edge, dtype=np.float64)
        self.idxs_edge_broken: list[int] = []
        self.idxs_edge_leaf: list[int] = []
        self.idxs_edge_island: list[int] = []
        self.volts_ext: list[float] = []

        self.counter_time_step: int = 0
        self.idxs_time_edge_broken: list[int] = []
        self._last_time_step_broken: bool = True
        self._last_time_step_break_edge_type: bool = True # True: charge, False: discharge
        self._test1, self._test2 = _test1, _test2 #################################################
        if save_volts_profile:
            self.volts_edge_profile: list[np.ndarray[np.float64]] = []
            self.volts_edge_signed: np.ndarray[np.float64] = np.empty(self.array.num_edge, dtype=np.float64)
            self._compute_volts_edge_ch = self._compute_volts_edge_save_ch
            self._compute_volts_edge_dch = self._compute_volts_edge_save_dch
            self._append_volts_edge_profile_scale_dynamic = self._append_volts_edge_profile_scale
            self._append_volts_edge_profile_dynamic = self._append_volts_edge_profile
            
            self.volts_cap_profile: list[np.ndarray[np.float64]] = []
            self._compute_volts_cap_dynamic_ch = self._compute_volts_cap_ch
            self._compute_volts_cap_dynamic_dch = self._compute_volts_cap_dch
            self._append_volts_cap_profile_dynamic = self._append_volts_cap_profile
    
            self.volts_cond_profile: list[np.ndarray[np.float64]] = []
            self.volts_cond: np.ndarray[np.float64] = np.empty(self.array.num_edge, dtype=np.float64)
            self._compute_volts_cond_dynamic_ch = self._compute_volts_cond_ch
            self._compute_volts_cond_dynamic_dch = self._compute_volts_cond_dch
            self._append_volts_cond_profile_dynamic = self._append_volts_cond_profile

        else:
            self._compute_volts_edge_ch = self._compute_volts_edge_unsave_ch
            self._compute_volts_edge_dch = self._compute_volts_edge_unsave_dch
            self._append_volts_edge_profile_scale_dynamic = self._no_op1
            self._append_volts_edge_profile_dynamic = self._no_op0

            self._compute_volts_cap_dynamic_ch = self._no_op0
            self._compute_volts_cap_dynamic_dch = self._no_op0
            self._append_volts_cap_profile_dynamic = self._no_op0

            self._compute_volts_cond_dynamic_ch = self._no_op0
            self._compute_volts_cond_dynamic_dch = self._no_op0
            self._append_volts_cond_profile_dynamic = self._no_op0


    def _append_volts_edge_profile_scale(self, scaling_factor):
        self.volts_edge_profile.append(self.volts_edge_signed.copy().__imul__(scaling_factor))

    def _append_volts_edge_profile(self):
        self.volts_edge_profile.append(self.volts_edge_signed.copy())

    def _append_volts_cap_profile(self):
        self.volts_cap_profile.append(self.volts_cap.copy())

    def _append_volts_cond_profile(self):
        self.volts_cond_profile.append(self.volts_cond.copy())

    @staticmethod
    def _no_op0(): pass

    @staticmethod
    def _no_op1(_): pass

    def _generate_breaking_strengths(self):
        breaking_strengths_min_nom, breaking_strengths_max_nom = 1.0 - self.width / 2.0, 1.0 + self.width / 2.0
        np.random.seed(seed=self.seed)
        breaking_strengths = np.random.uniform(breaking_strengths_min_nom, breaking_strengths_max_nom, self.array.num_edge)
        return breaking_strengths
    
    def _compute_volts_edge_unsave_ch(self):
        array = self.array
        volts_edge = self.volts_edge

        volts_edge[:array.length] = self.equation.volt_ext - self.equation.volts_node_div[:array.length]
        volts_edge[array.idxs_edge_bot] = self.equation.volts_node_div[array.idxs_edge_bot_node1]
        volts_edge[array.idxs_edge_mid] = self.equation.volts_node_div[array.idxs_edge_mid_node1] - self.equation.volts_node_div[array.idxs_edge_mid_node2]
        np.abs(volts_edge, out=volts_edge) # unsigned voltage drop

    def _compute_volts_edge_save_ch(self):
        "save signed volts_edge (signed according to E (treat it as a directed graph); pbc edges require special handling if volts_edge was to be equated to volts_cap + (- volts_cond))"
        # for changing sign convention west -> east, north -> south (for visualization or etc. that do not rely on E (i -> j) order itself)
        # np.array(failure.volts_edge_profile)[:, array.idxs_edge_horizontal_no_pbc] *= -1.0
        array = self.array
        volts_edge = self.volts_edge

        volts_edge[:array.length] = self.equation.volt_ext - self.equation.volts_node_div[:array.length]
        volts_edge[array.idxs_edge_bot] = self.equation.volts_node_div[array.idxs_edge_bot_node1]
        volts_edge[array.idxs_edge_mid] = self.equation.volts_node_div[array.idxs_edge_mid_node1] - self.equation.volts_node_div[array.idxs_edge_mid_node2]
        self.volts_edge_signed[:] = volts_edge
        np.abs(volts_edge, out=volts_edge)

    def _compute_volts_edge_unsave_dch(self):
        array = self.array
        volts_edge = self.volts_edge

        volts_edge[:array.length] = self.equation.volts_node_div[0] - self.equation.volts_node_div[1:array.length + 1]
        volts_edge[array.idxs_edge_bot] = self.equation.volts_node_div[array.idxs_edge_bot_node1 + 1]
        volts_edge[array.idxs_edge_mid] = self.equation.volts_node_div[array.idxs_edge_mid_node1 + 1] - self.equation.volts_node_div[array.idxs_edge_mid_node2 + 1]
        np.abs(volts_edge, out=volts_edge) # unsigned voltage drop

    def _compute_volts_edge_save_dch(self):
        array = self.array
        volts_edge = self.volts_edge

        volts_edge[:array.length] = self.equation.volts_node_div[0] - self.equation.volts_node_div[1:array.length + 1]
        volts_edge[array.idxs_edge_bot] = self.equation.volts_node_div[array.idxs_edge_bot_node1 + 1]
        volts_edge[array.idxs_edge_mid] = self.equation.volts_node_div[array.idxs_edge_mid_node1 + 1] - self.equation.volts_node_div[array.idxs_edge_mid_node2 + 1]
        self.volts_edge_signed[:] = volts_edge
        np.abs(volts_edge, out=volts_edge) # unsigned voltage drop

    def _compute_volts_cap_ch(self):
        "compute failure.volts_cap based on equation.volts_node_div (SIGNED BASED ON E DIV CAP; ORDERED BASED ON E)"
        volts_cap = self.volts_cap
        volt_ext = self.equation.volt_ext
        volts_node_div = self.equation.volts_node_div
        length, length_double = self.array.length, self.array.length_double

        for idx_cap, (idx_node1, idx_node2) in enumerate(self.array.edges_div_cap):
            idx_node1_new, idx_node2_new = idx_node1 - length, idx_node2 - length_double
            if idx_node1 < length:
                volts_cap[idx_cap] = volt_ext - volts_node_div[idx_node2_new]
            else:
                volts_cap[idx_cap] = volts_node_div[idx_node1_new] - volts_node_div[idx_node2_new]

    def _compute_volts_cap_dch(self):
        volts_cap = self.volts_cap
        volts_node_div = self.equation.volts_node_div
        volt_top = volts_node_div[0]
        length = self.array.length
        length_minus_one, length_double_minus_one = self.array.length - 1, self.array.length_double - 1

        for idx_cap, (idx_node1, idx_node2) in enumerate(self.array.edges_div_cap):
            idx_node1_new, idx_node2_new = idx_node1 - length_minus_one, idx_node2 - length_double_minus_one
            if idx_node1 < length:
                volts_cap[idx_cap] = volt_top - volts_node_div[idx_node2_new]
            else:
                volts_cap[idx_cap] = volts_node_div[idx_node1_new] - volts_node_div[idx_node2_new]

    def _compute_volts_cond_ch(self):
        "compute failure.volts_cond based on equation.volts_node_div (SIGNED BASED ON E DIV COND; ORDERED BASED ON E)"
        volts_cond = self.volts_cond
        volts_node_div = self.equation.volts_node_div
        length, length_double = self.array.length, self.array.length_double
        idx_node_bot_first_minus_one = self.array.idx_node_bot_first_minus_one

        for idx_cond, (idx_node1, idx_node2) in enumerate(self.array.edges_div_cond):
            idx_node1_new, idx_node2_new = idx_node1 - length, idx_node2 - length_double
            if idx_node_bot_first_minus_one < idx_node1:
                volts_cond[idx_cond] = - volts_node_div[idx_node2_new]
            else:
                volts_cond[idx_cond] = volts_node_div[idx_node1_new] - volts_node_div[idx_node2_new]

    def _compute_volts_cond_dch(self):
        volts_cond = self.volts_cond
        volts_node_div = self.equation.volts_node_div
        length_minus_one, length_double_minus_one = self.array.length - 1, self.array.length_double - 1
        idx_node_bot_first_minus_one = self.array.idx_node_bot_first_minus_one

        for idx_cond, (idx_node1, idx_node2) in enumerate(self.array.edges_div_cond):
            idx_node1_new, idx_node2_new = idx_node1 - length_minus_one, idx_node2 - length_double_minus_one
            if idx_node_bot_first_minus_one < idx_node1:
                volts_cond[idx_cond] = - volts_node_div[idx_node2_new]
            else:
                volts_cond[idx_cond] = volts_node_div[idx_node1_new] - volts_node_div[idx_node2_new]

    def _charge_conservation_ch(self):
        "modify equation.volts_node_div based on failure.volts_cap computed pre-scaling; this is due to constant C and energy conservation law"
        length, length_double = self.array.length, self.array.length_double
        volt_ext = self.equation.volt_ext
        volts_node_div = self.equation.volts_node_div

        for (idx_node1, idx_node2), volt_cap in zip(self.array.edges_div_cap, self.volts_cap):
            if idx_node1 < length:
                volts_node_div[idx_node2 - length_double] = volt_ext - volt_cap
            else:
                volts_node_div[idx_node2 - length_double] = volts_node_div[idx_node1 - length] - volt_cap

    def _charge_conservation_dch(self):
        volts_node_div = self.equation.volts_node_div
        volt_top = volts_node_div[0]
        length = self.array.length
        length_minus_one, length_double_minus_one = self.array.length - 1, self.array.length_double - 1

        for (idx_node1, idx_node2), volt_cap in zip(self.array.edges_div_cap, self.volts_cap):
            if idx_node1 < length:
                volts_node_div[idx_node2 - length_double_minus_one] = volt_top - volt_cap
            else:
                volts_node_div[idx_node2 - length_double_minus_one] = volts_node_div[idx_node1 - length_minus_one] - volt_cap

    def _update_matrix_ch(self, idx_edge_broken):
        array = self.array
        idx_node1, idx_node2 = array.edges_div_cond[idx_edge_broken]
        idx_node1_new, idx_node2_new = idx_node1 - array.length, idx_node2 - array.length_double
        div_comb = self.matrix_ch.div_comb
        div_block = self.matrix_ch.div_block
        time_step = self.matrix_ch.time_step

        if array.idx_node_bot_first_minus_one < idx_node1:
            div_comb[array.num_node_div_mid, idx_node2_new] -= time_step
            div_comb[idx_node2_new, idx_node2_new] -= time_step
            div_block[array.num_node_div_mid, idx_node2_new] -= 1.0
            div_block[idx_node2_new, idx_node2_new] -= 1.0
        else:
            div_comb[idx_node1_new, idx_node1_new] -= time_step
            div_comb[idx_node2_new, idx_node2_new] -= time_step
            div_comb[idx_node1_new, idx_node2_new] += time_step
            div_comb[idx_node2_new, idx_node1_new] += time_step
            div_block[idx_node1_new, idx_node1_new] -= 1.0
            div_block[idx_node2_new, idx_node2_new] -= 1.0
            div_block[idx_node1_new, idx_node2_new] += 1.0
            div_block[idx_node2_new, idx_node1_new] += 1.0

        idx_node1, idx_node2 = array.edges[idx_edge_broken]
        idx_node1_new, idx_node2_new = idx_node1 - array.length + 1, idx_node2 - array.length + 1
        cond = self.matrix_ch.cond

        if idx_node2_new <= array.num_node_mid:
            if idx_node1_new > 0:
                cond[idx_node1_new, idx_node1_new] -= 1.0
                cond[idx_node2_new, idx_node2_new] -= 1.0
                cond[idx_node1_new, idx_node2_new] += 1.0
                cond[idx_node2_new, idx_node1_new] += 1.0
            else:
                cond[0, 0] -= 1.0
                cond[idx_node2_new, idx_node2_new] -= 1.0
                cond[0, idx_node2_new] += 1.0
                cond[idx_node2_new, 0] += 1.0
        else:
            cond[idx_node1_new, idx_node1_new] -= 1.0

    def _update_matrix_dch(self, idx_edge_broken):
        array = self.array
        idx_node1, idx_node2 = array.edges_div_cond[idx_edge_broken]
        idx_node1_new, idx_node2_new = idx_node1 - array.length + 1, idx_node2 - array.length_double + 1
        div_comb = self.matrix_dch.div_comb
        div_block = self.matrix_dch.div_block
        time_step = self.matrix_dch.time_step

        if array.idx_node_bot_first_minus_one < idx_node1:
            div_comb[idx_node2_new, idx_node2_new] -= time_step
            div_block[idx_node2_new, idx_node2_new] -= 1.0
        else:
            div_comb[idx_node1_new, idx_node1_new] -= time_step
            div_comb[idx_node2_new, idx_node2_new] -= time_step
            div_comb[idx_node1_new, idx_node2_new] += time_step
            div_comb[idx_node2_new, idx_node1_new] += time_step
            div_block[idx_node1_new, idx_node1_new] -= 1.0
            div_block[idx_node2_new, idx_node2_new] -= 1.0
            div_block[idx_node1_new, idx_node2_new] += 1.0
            div_block[idx_node2_new, idx_node1_new] += 1.0

    def _find_edge_broken_leaf_proof(self, quantities):
        degrees = self.degrees
        idx_edge_broken = int(np.argmin(quantities))
        idx_node1, idx_node2 = self.array.edges[idx_edge_broken]
        degrees[idx_node1] -= 1
        degrees[idx_node2] -= 1

        if idx_node1 < self.array.length:
            if degrees[idx_node2] == 0:
                degrees[idx_node1] += 1
                degrees[idx_node2] += 1
                quantities[idx_edge_broken] = np.inf
                self.idxs_edge_leaf.append(idx_edge_broken)
                return self._find_edge_broken_leaf_proof(quantities)
            return idx_edge_broken
        
        if idx_node2 > self.array.idx_node_bot_first_minus_one:
            if degrees[idx_node1] == 0:
                degrees[idx_node1] += 1
                degrees[idx_node2] += 1
                quantities[idx_edge_broken] = np.inf
                self.idxs_edge_leaf.append(idx_edge_broken)
                return self._find_edge_broken_leaf_proof(quantities)
            return idx_edge_broken
        
        if degrees[idx_node1] * degrees[idx_node2] == 0:
            degrees[idx_node1] += 1
            degrees[idx_node2] += 1
            quantities[idx_edge_broken] = np.inf
            self.idxs_edge_leaf.append(idx_edge_broken)
            return self._find_edge_broken_leaf_proof(quantities)
        return idx_edge_broken

    def break_edge_true_init_ch(self) -> None:
        self._compute_volts_edge_ch()
        stresses_edge_neg = self.breaking_strengths - self.volts_edge
        stresses_edge_neg[self.idxs_edge_broken] = np.inf
        stresses_edge_neg[self.idxs_edge_leaf] = np.inf
        stresses_edge_neg[self.idxs_edge_island] = np.inf
        idx_edge_broken = self._find_edge_broken_leaf_proof(stresses_edge_neg)
        factor_scaling = self.breaking_strengths[idx_edge_broken] / self.volts_edge[idx_edge_broken]

        self._update_matrix_ch(idx_edge_broken)
        self._update_matrix_dch(idx_edge_broken)
        self.idxs_edge_broken.append(idx_edge_broken)
        self.idxs_time_edge_broken.append(self.counter_time_step)

        self._compute_volts_cap_ch()
        self.equation.volt_ext *= factor_scaling
        self.equation.volts_node_div_prev = self.equation.volts_node_div.copy()
        self.equation.volts_node_div *= factor_scaling
        self._charge_conservation_ch()

        self._append_volts_edge_profile_scale_dynamic(factor_scaling)
        self._compute_volts_cap_dynamic_ch()
        self._append_volts_cap_profile_dynamic()
        self._compute_volts_cond_dynamic_ch()
        self._append_volts_cond_profile_dynamic()
        self.volts_ext.append(float(self.equation.volt_ext))

        self.counter_time_step += 1

    def break_edge_ch(self) -> None:
        self._compute_volts_edge_ch()
        stresses_edge_neg = self.breaking_strengths - self.volts_edge
        stresses_edge_neg[self.idxs_edge_broken] = np.inf
        stresses_edge_neg[self.idxs_edge_leaf] = np.inf
        stresses_edge_neg[self.idxs_edge_island] = np.inf
        idx_edge_broken = self._find_edge_broken_leaf_proof(stresses_edge_neg)

        if stresses_edge_neg[idx_edge_broken] <= 0:
            self._update_matrix_ch(idx_edge_broken)
            self._update_matrix_dch(idx_edge_broken)
            self.idxs_edge_broken.append(idx_edge_broken)
            self.idxs_time_edge_broken.append(self.counter_time_step)

            self.equation.volts_node_div_prev = None
            self._append_volts_edge_profile_dynamic()
            self._compute_volts_cap_ch() # must compute (not dynamic)

        else:
            idx_node1, idx_node2 = self.array.edges[idx_edge_broken]
            self.degrees[idx_node1] += 1
            self.degrees[idx_node2] += 1

            factors_scaling = self.breaking_strengths / self.volts_edge
            factors_scaling[self.idxs_edge_broken] = np.inf
            factors_scaling[self.idxs_edge_leaf] = np.inf
            factors_scaling[self.idxs_edge_island] = np.inf
            idx_edge_broken = self._find_edge_broken_leaf_proof(factors_scaling)
            factor_scaling = factors_scaling[idx_edge_broken]


            if self._test1: # new correct bookkeeeping
                self.idxs_edge_broken.append(idx_edge_broken)
                self.idxs_time_edge_broken.append(self.counter_time_step)

                self._compute_volts_cap_ch() # update volts_cap based on volts_node_div
                self.volts_cap_prev = self.volts_cap.copy() ##############################
                self.equation.volt_ext *= factor_scaling # update volts_ext
                self.equation.volts_node_div_prev = self.equation.volts_node_div.copy() # save pre-scaling volts_node_div incase of fall back from attempted breaking of idx_edge_island next time step
            
                self.equation.solve_init_ch() # compute post-scaling volts_node_div based on volts_cap and post-scaling volt_ext
                if self._test2: 
                    self._charge_conservation_ch() ######################################
                self._compute_volts_cap_ch() ##################################
                if np.max(np.abs(self.volts_cap - self.volts_cap_prev)) > 1e-14: print(np.max(self.volts_cap - self.volts_cap_prev)) ###########


                self._update_matrix_ch(idx_edge_broken)
                self._update_matrix_dch(idx_edge_broken)

                self._compute_volts_edge_ch()
                self._append_volts_edge_profile_dynamic()
            else:  # old , still correct (just saved profile is "incorrect", but actual simulation data are all correct because this only cause bookkeeping issue, no effects on the breakdown dynamics logic)
                self._update_matrix_ch(idx_edge_broken)
                self._update_matrix_dch(idx_edge_broken)
                self.idxs_edge_broken.append(idx_edge_broken)
                self.idxs_time_edge_broken.append(self.counter_time_step)

                self._compute_volts_cap_ch()
                self.equation.volt_ext *= factor_scaling
                self.equation.volts_node_div_prev = self.equation.volts_node_div.copy()
                self.equation.volts_node_div *= factor_scaling
                self._charge_conservation_ch()

                self._append_volts_edge_profile_scale_dynamic(factor_scaling)
                self._compute_volts_cap_ch() # must compute (not dynamic)

        self._append_volts_cap_profile_dynamic()
        self._compute_volts_cond_dynamic_ch()
        self._append_volts_cond_profile_dynamic()
        self.volts_ext.append(float(self.equation.volt_ext))

        self._last_time_step_broken = True
        self._last_time_step_break_edge_type = True
        self.counter_time_step += 1

    def break_edge_dch(self) -> None:
        self._compute_volts_edge_dch()
        stresses_edge_neg = self.breaking_strengths - self.volts_edge
        stresses_edge_neg[self.idxs_edge_broken] = np.inf
        stresses_edge_neg[self.idxs_edge_leaf] = np.inf
        stresses_edge_neg[self.idxs_edge_island] = np.inf
        idx_edge_broken = self._find_edge_broken_leaf_proof(stresses_edge_neg)

        if stresses_edge_neg[idx_edge_broken] <= 0:
            self._update_matrix_ch(idx_edge_broken)
            self._update_matrix_dch(idx_edge_broken)
            self.idxs_edge_broken.append(idx_edge_broken)
            self.idxs_time_edge_broken.append(self.counter_time_step)

            self.equation.volts_node_div_prev = None
            self._last_time_step_broken = True

        else:
            idx_node1, idx_node2 = self.array.edges[idx_edge_broken]
            self.degrees[idx_node1] += 1
            self.degrees[idx_node2] += 1

            self.equation.volts_node_div_prev = None
            self._last_time_step_broken = False

        self._append_volts_edge_profile_dynamic()
        self._compute_volts_cap_dch() # must compute (not dynamic)
        self._append_volts_cap_profile_dynamic()
        self._compute_volts_cond_dynamic_dch()
        self._append_volts_cond_profile_dynamic()
        self.volts_ext.append(float(self.equation.volts_node_div[0]))

        self._last_time_step_break_edge_type = False
        self.counter_time_step += 1