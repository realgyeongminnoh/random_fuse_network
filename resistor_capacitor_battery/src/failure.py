import warnings
import numpy as np

from .array import Array
from .matrix import Matrix
from .equation import Equation

warnings.simplefilter("ignore", category=RuntimeWarning) # division by zero (from broken / leaf edge execlusion) ignore because inf do not cause issue for argmin


class Failure:
    def __init__(self, array: Array, matrix: Matrix, equation: Equation, width: float, seed: int, save_volts_profile: bool = False):
        self.array: Array = array
        self.matrix: Matrix = matrix
        self.equation: Equation = equation
        self.equation._lazy_init_failure(self)
        self.width = float(width)
        self.seed = int(seed)
        self.breaking_strengths: np.ndarray[np.float64] = self._generate_breaking_strengths()

        self.num_edge_per_layer: np.ndarray[np.int32] = np.array([array.length] * array.length) # per horizontal layer
        self.volts_edge: np.ndarray[np.float64] = np.empty(self.array.num_edge, dtype=np.float64)
        self.volts_cap: np.ndarray[np.float64] = np.empty(self.array.num_edge, dtype=np.float64)
        self.idxs_edge_broken: list[int] = []
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

    def _compute_volts_edge_init(self):
        self.volts_edge[:] = 1 / self.array.length
        if hasattr(self, "volts_edge_profile"):
            self.volts_edge_signed[:] = self.volts_edge

    def _compute_volts_edge_unsave(self):
        length = self.array.length
        volts_node_div = self.equation.volts_node_div
        volts_edge = self.volts_edge

        volts_edge[:length] = self.equation.volt_ext - volts_node_div[0]
        volts_edge[-length:] = volts_node_div[length - 2]
        volts_edge[length:-length] = np.repeat([
            volts_node_div[idx_node1] - volts_node_div[idx_node2]
            for idx_node1, idx_node2 
            in [(idx_node, idx_node + 1) for idx_node in range(length - 2)]
        ], length)

        np.abs(volts_edge, out=volts_edge)

    def _compute_volts_edge_save(self):
        length = self.array.length
        volts_node_div = self.equation.volts_node_div
        volts_edge = self.volts_edge

        volts_edge[:length] = self.equation.volt_ext - volts_node_div[0]
        volts_edge[-length:] = volts_node_div[length - 2]
        volts_edge[length:-length] = np.repeat([
            volts_node_div[idx_node1] - volts_node_div[idx_node2]
            for idx_node1, idx_node2 
            in [(idx_node, idx_node + 1) for idx_node in range(length - 2)]
        ], length)

        self.volts_edge_signed[:] = volts_edge
        np.abs(volts_edge, out=volts_edge)

    def _compute_volts_cap(self):
        volts_cap = self.volts_cap
        volt_ext = self.equation.volt_ext
        volts_node_div = self.equation.volts_node_div

        for idx_cap, (idx_node1, idx_node2) in enumerate(self.array.edges_div_cap):
            idx_node1_new, idx_node2_new = idx_node1 - 1, idx_node2 - 2
            if idx_node1_new < 0:
                volts_cap[idx_cap] = volt_ext - volts_node_div[idx_node2_new]
            else:
                volts_cap[idx_cap] = volts_node_div[idx_node1_new] - volts_node_div[idx_node2_new]

    def _compute_volts_cond(self):
        volts_cond = self.volts_cond
        length = self.array.length
        volts_node_div = self.equation.volts_node_div

        for idx_cond, (idx_node1, idx_node2) in enumerate(self.array.edges_div_cond):
            idx_node1_new, idx_node2_new = idx_node1 - 1, idx_node2 - 2
            if idx_node1 == length:
                volts_cond[idx_cond] = - volts_node_div[idx_node2_new]
            else:
                volts_cond[idx_cond] = volts_node_div[idx_node1_new] - volts_node_div[idx_node2_new]

    def _charge_conservation(self):
        volt_ext = self.equation.volt_ext
        volts_node_div = self.equation.volts_node_div

        for (idx_node1, idx_node2), volt_cap in zip(self.array.edges_div_cap, self.volts_cap):
            if idx_node1 == 0: # same as idx_node1_new < 0:
                volts_node_div[idx_node2 - 2] = volt_ext - volt_cap
            else:
                volts_node_div[idx_node2 - 2] = volts_node_div[idx_node1 - 1] - volt_cap

    def _update_matrix(self, idx_edge_broken):
        array, length = self.array, self.array.length
        idx_node1, idx_node2 = array.edges_div_cond[idx_edge_broken]
        idx_node1_new, idx_node2_new = idx_node1 - 1, idx_node2 - 2
        div_comb = self.matrix.div_comb
        val_cap_reciprocal = self.matrix.val_cap_reciprocal

        if idx_node1 == length:
            div_comb[array.num_node_div_mid, idx_node2_new] -= val_cap_reciprocal
            div_comb[idx_node2_new, idx_node2_new] -= val_cap_reciprocal
        else:
            div_comb[idx_node1_new, idx_node1_new] -= val_cap_reciprocal
            div_comb[idx_node2_new, idx_node2_new] -= val_cap_reciprocal
            div_comb[idx_node1_new, idx_node2_new] += val_cap_reciprocal
            div_comb[idx_node2_new, idx_node1_new] += val_cap_reciprocal
    
    def break_edge_init(self) -> None:
        self._compute_volts_edge_init()
        idx_edge_broken = int(np.argmin(self.breaking_strengths / self.volts_edge))
        
        self.num_edge_per_layer[idx_edge_broken // self.array.length] -= 1 # update num_edge_per_layer for equation.check_graph
        self._update_matrix(idx_edge_broken)
        self.idxs_edge_broken.append(idx_edge_broken)

        factor_scaling = self.breaking_strengths[idx_edge_broken] / self.volts_edge[idx_edge_broken]
        self.equation.volt_ext *= factor_scaling
        self.equation.volts_node_div *= factor_scaling

        self._append_volts_edge_profile_scale_dynamic(factor_scaling)
        if hasattr(self, "volts_edge_profile"):
            self.volts_cap[:] = np.zeros(self.array.num_edge, dtype=np.float64)
            self.volts_cond[:] = self.volts_edge_profile[0] * -1.0
        self._append_volts_cap_profile_dynamic()
        self._append_volts_cond_profile_dynamic()
        self.volts_ext.append(float(self.equation.volt_ext))

    def break_edge(self) -> None:
        self._compute_volts_edge()
        stresses_edge_neg = self.breaking_strengths - self.volts_edge
        stresses_edge_neg[self.idxs_edge_broken] = np.inf
        idx_edge_broken = int(np.argmin(stresses_edge_neg))

        if stresses_edge_neg[idx_edge_broken] <= 0:
            self.num_edge_per_layer[idx_edge_broken // self.array.length] -= 1 # update num_edge_per_layer for equation.check_graph
            self._update_matrix(idx_edge_broken)
            self.idxs_edge_broken.append(idx_edge_broken)

            self._append_volts_edge_profile_dynamic()

        else:
            factors_scaling = self.breaking_strengths / self.volts_edge
            factors_scaling[self.idxs_edge_broken] = np.inf
            idx_edge_broken = int(np.argmin(factors_scaling))
            factor_scaling = factors_scaling[idx_edge_broken]

            self.num_edge_per_layer[idx_edge_broken // self.array.length] -= 1 # update num_edge_per_layer for equation.check_graph
            self._update_matrix(idx_edge_broken)
            self.idxs_edge_broken.append(idx_edge_broken)

            self._compute_volts_cap()
            self.equation.volt_ext *= factor_scaling
            self.equation.volts_node_div *= factor_scaling
            self._charge_conservation()
            self._append_volts_edge_profile_scale_dynamic(factor_scaling)

        self._compute_volts_cap_dynamic()
        self._append_volts_cap_profile_dynamic()
        self._compute_volts_cond_dynamic()
        self._append_volts_cond_profile_dynamic()
        self.volts_ext.append(float(self.equation.volt_ext))       