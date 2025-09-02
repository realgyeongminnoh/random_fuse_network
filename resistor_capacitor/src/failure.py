import warnings
import numpy as np

from src.array import Array
from src.matrix import Matrix
from src.equation import Equation

warnings.simplefilter("ignore", category=RuntimeWarning) # division by zero (from broken / leaf edge execlusion) ignore because inf do not cause issue for argmin


class Failure:
    def __init__(self, array: Array, matrix: Matrix, equation: Equation, width: float, seed: int, save_volts_profile: bool = False):
        self.array = array
        self.matrix = matrix
        self.equation = equation
        self.equation._lazy_init_failure(self)
        self.width = float(width)
        self.seed = int(seed)
        self.breaking_strengths: np.ndarray[np.float64] = self._generate_breaking_strengths() # 1. bond voltage for now (2. capacitor voltage can only lead to charge-nonconservation scaling mechanism; 3. resistor voltage = resistor current = capacitor current = bond current implications are unclear yet)

        self.degrees: np.ndarray[np.int32] = np.array([1] * array.length + [4] * array.num_node_mid + [1] * array.length, dtype=np.int32)
        self.volts_edge: np.ndarray[np.float64] = np.empty(self.array.num_edge, dtype=np.float64)
        self.volts_cap: np.ndarray[np.float64] = np.empty(self.array.num_edge, dtype=np.float64)
        self.idxs_edge_broken: list[int] = []
        self.idxs_edge_leaf: list[int] = []
        self.idxs_edge_island: list[int] = []
        self.volts_ext: list[float] = []

        if save_volts_profile:
            self.volts_edge_profile: list[np.ndarray[np.float64]] = []
            self.volts_edge_signed: np.ndarray[np.float64] = np.empty(self.array.num_edge, dtype=np.float64)
            self._compute_volts_edge = self._compute_volts_edge_save
            self._append_volts_edge_profile_scale_dynamic = self._append_volts_edge_profile_scale 
            self._append_volts_edge_profile_dynamic = self._append_volts_edge_profile

            self.volts_cap_profile: list[np.ndarray[np.float64]] = []
            self._compute_volts_cap_dynamic = self._compute_volts_cap
            self._append_volts_cap_profile_dynamic = self._append_volts_cap_profile

            self.volts_cond_profile: list[np.ndarray[np.float64]] = []
            self.volts_cond: np.ndarray[np.float64] = np.empty(self.array.num_edge, dtype=np.float64)
            self._compute_volts_cond_dynamic = self._compute_volts_cond
            self._append_volts_cond_profile_dynamic = self._append_volts_cond_profile
            
        else:
            self._compute_volts_edge = self._compute_volts_edge_unsave
            self._append_volts_edge_profile_scale_dynamic = self._no_op1
            self._append_volts_edge_profile_dynamic = self._no_op0

            self._compute_volts_cap_dynamic = self._no_op0
            self._append_volts_cap_profile_dynamic = self._no_op0

            self._compute_volts_cond_dynamic = self._no_op0
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

    def _compute_volts_edge_unsave(self):
        array = self.array
        volts_edge = self.volts_edge

        volts_edge[:array.length] = self.equation.volt_ext - self.equation.volts_node_div[:array.length]
        volts_edge[array.idxs_edge_bot] = self.equation.volts_node_div[array.idxs_edge_bot_node1]
        volts_edge[array.idxs_edge_mid] = self.equation.volts_node_div[array.idxs_edge_mid_node1] - self.equation.volts_node_div[array.idxs_edge_mid_node2]
        np.abs(volts_edge, out=volts_edge) # unsigned voltage drop

    def _compute_volts_edge_save(self):
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

    def _compute_volts_cap(self):
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

    def _compute_volts_cond(self):
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

    def _charge_conservation(self):
        "modify equation.volts_node_div based on failure.volts_cap computed pre-scaling; this is due to constant C and energy conservation law"
        length, length_double = self.array.length, self.array.length_double
        volt_ext = self.equation.volt_ext
        volts_node_div = self.equation.volts_node_div

        for (idx_node1, idx_node2), volt_cap in zip(self.array.edges_div_cap, self.volts_cap):
            if idx_node1 < length:
                volts_node_div[idx_node2 - length_double] = volt_ext - volt_cap
            else:
                volts_node_div[idx_node2 - length_double] = volts_node_div[idx_node1 - length] - volt_cap

    def _update_matrix(self, idx_edge_broken):
        array = self.array
        idx_node1, idx_node2 = array.edges_div_cond[idx_edge_broken]
        idx_node1_new, idx_node2_new = idx_node1 - array.length, idx_node2 - array.length_double
        div_comb = self.matrix.div_comb
        time_step = self.matrix.time_step

        if array.idx_node_bot_first_minus_one < idx_node1:
            div_comb[array.num_node_div_mid, idx_node2_new] -= time_step
            div_comb[idx_node2_new, idx_node2_new] -= time_step
        else:
            div_comb[idx_node1_new, idx_node1_new] -= time_step
            div_comb[idx_node2_new, idx_node2_new] -= time_step
            div_comb[idx_node1_new, idx_node2_new] += time_step
            div_comb[idx_node2_new, idx_node1_new] += time_step

        idx_node1, idx_node2 = array.edges[idx_edge_broken]
        idx_node1_new, idx_node2_new = idx_node1 - array.length + 1, idx_node2 - array.length + 1
        cond = self.matrix.cond

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
    
    def break_edge_init(self) -> None:
        self._compute_volts_edge()
        idx_edge_broken = int(np.argmin(self.breaking_strengths - self.volts_edge))
        idx_node1, idx_node2 = self.array.edges[idx_edge_broken]
        self.degrees[idx_node1] -= 1
        self.degrees[idx_node2] -= 1

        self._update_matrix(idx_edge_broken)
        self.idxs_edge_broken.append(idx_edge_broken)

        factor_scaling = self.breaking_strengths[idx_edge_broken] / self.volts_edge[idx_edge_broken]
        self.equation.volt_ext *= factor_scaling
        self.equation.volts_node_div *= factor_scaling

        self._append_volts_edge_profile_scale_dynamic(factor_scaling)
        self._compute_volts_cap_dynamic()
        self._append_volts_cap_profile_dynamic()
        self._compute_volts_cond_dynamic()
        self._append_volts_cond_profile_dynamic()
        self.volts_ext.append(float(self.equation.volt_ext))

    def break_edge(self) -> None:
        self._compute_volts_edge()
        stresses_edge_neg = self.breaking_strengths - self.volts_edge
        stresses_edge_neg[self.idxs_edge_broken] = np.inf
        stresses_edge_neg[self.idxs_edge_leaf] = np.inf
        stresses_edge_neg[self.idxs_edge_island] = np.inf
        idx_edge_broken = self._find_edge_broken_leaf_proof(stresses_edge_neg)

        if stresses_edge_neg[idx_edge_broken] <= 0:
            self._update_matrix(idx_edge_broken)
            self.idxs_edge_broken.append(idx_edge_broken)

            self.equation.volts_node_div_prev = None
            self._append_volts_edge_profile_dynamic()

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

            self._update_matrix(idx_edge_broken)
            self.idxs_edge_broken.append(idx_edge_broken)

            self._compute_volts_cap()
            self.equation.volt_ext *= factor_scaling
            self.equation.volts_node_div_prev = self.equation.volts_node_div.copy()
            self.equation.volts_node_div *= factor_scaling
            self._charge_conservation()
            self._append_volts_edge_profile_scale_dynamic(factor_scaling)

        self._compute_volts_cap_dynamic()
        self._append_volts_cap_profile_dynamic()
        self._compute_volts_cond_dynamic()
        self._append_volts_cond_profile_dynamic()
        self.volts_ext.append(float(self.equation.volt_ext))