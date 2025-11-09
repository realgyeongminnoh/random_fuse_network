import warnings
import numpy as np

from .array import Array
from .matrix import Matrix
from .equation import Equation

warnings.simplefilter("ignore", category=RuntimeWarning) # division by zero for finding scaling factors and some (i.e., horizontal) edges have 0 voltage drops # also in extremely rare cases with small L those are broken


class Failure:
    def __init__(self, array: Array, matrix: Matrix, equation: Equation, width: float, seed: int, save_volts_profile: bool = False):
        self.array: Array = array
        self.matrix: Matrix = matrix
        self.equation: Equation = equation
        self.width = float(width)
        self.seed = int(seed)
        self.breaking_strengths: np.ndarray[np.float64] = self._generate_breaking_strengths()

        self.volts_edge: np.ndarray[np.float64] = np.empty(self.array.num_edge, dtype=np.float64)
        self.idxs_edge_broken: list[int] = []
        self.idxs_edge_unalive: list[int] = []
        self.volts_ext: list[float] = []

        if save_volts_profile:
            self.volts_edge_profile: list[np.ndarray[np.float64]] = []
            self.volts_edge_signed: np.ndarray[np.float64] = np.empty(self.array.num_edge, dtype=np.float64)
            self._compute_volts_edge = self._compute_volts_edge_save
            self._append_volts_edge_profile_scale_dynamic = self._append_volts_edge_profile_scale
            self._append_volts_edge_profile_dynamic = self._append_volts_edge_profile
            
        else:
            self._compute_volts_edge = self._compute_volts_edge_unsave
            self._append_volts_edge_profile_scale_dynamic = self._no_op1
            self._append_volts_edge_profile_dynamic = self._no_op0

        self.equation._lazy_init_failure(self)

    def _append_volts_edge_profile_scale(self, scaling_factor):
        self.volts_edge_profile.append(self.volts_edge_signed.copy().__imul__(scaling_factor))

    def _append_volts_edge_profile(self):
        self.volts_edge_profile.append(self.volts_edge_signed.copy())

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
        self.volts_edge[:] = np.zeros(self.array.num_edge, dtype=np.float64)
        self.volts_edge[self.array.idxs_edge_vertical] = 1 / self.array.length
        if hasattr(self, "volts_edge_profile"):
            self.volts_edge_signed[:] = self.volts_edge

    def _compute_volts_edge_unsave(self):
        array = self.array
        volts_edge = self.volts_edge

        volts_edge[:array.length] = self.equation.volt_ext - self.equation.volts_node[1:array.length_plus_one]
        volts_edge[array.idxs_edge_bot] = self.equation.volts_node[array.idxs_edge_bot_node1]
        volts_edge[array.idxs_edge_mid] = self.equation.volts_node[array.idxs_edge_mid_node1] - self.equation.volts_node[array.idxs_edge_mid_node2]
        np.abs(volts_edge, out=volts_edge)

    def _compute_volts_edge_save(self):
        "save signed volts_edge (signed according to E (treat it as a directed graph); non-pbc horizontal edges require special handling)"
        # for changing sign convention west -> east, north -> south (for visualization or etc. that do not rely on E (i -> j) order itself)
        array = self.array
        volts_edge = self.volts_edge

        volts_edge[:array.length] = self.equation.volt_ext - self.equation.volts_node[1:array.length_plus_one]
        volts_edge[array.idxs_edge_bot] = self.equation.volts_node[array.idxs_edge_bot_node1]
        volts_edge[array.idxs_edge_mid] = self.equation.volts_node[array.idxs_edge_mid_node1] - self.equation.volts_node[array.idxs_edge_mid_node2]
        self.volts_edge_signed[:] = volts_edge
        np.abs(volts_edge, out=volts_edge)

    def _update_matrix(self, idx_edge_broken):
        idx_node1, idx_node2 = self.array.edges[idx_edge_broken]
        idx_node1_new, idx_node2_new = idx_node1 - self.array.length + 1, idx_node2 - self.array.length + 1
        cond = self.matrix.cond

        if idx_node2_new <= self.array.num_node_mid:
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

    def break_edge_init(self):
        self._compute_volts_edge_init()
        idx_edge_broken = int(np.argmin(self.breaking_strengths / self.volts_edge))

        self._update_matrix(idx_edge_broken)
        self.idxs_edge_broken.append(idx_edge_broken)

        factor_scaling = self.breaking_strengths[idx_edge_broken] / self.volts_edge[idx_edge_broken]
        self.equation.volt_ext *= factor_scaling
        self.equation.volts_node *= factor_scaling
        self._append_volts_edge_profile_scale_dynamic(factor_scaling)
        self.volts_ext.append(float(self.equation.volt_ext))

    def break_edge(self):
        self._compute_volts_edge()
        stresses_edge_neg = self.breaking_strengths - self.volts_edge
        stresses_edge_neg[self.idxs_edge_broken] = np.inf
        stresses_edge_neg[self.idxs_edge_unalive] = np.inf
        idx_edge_broken = int(np.argmin(stresses_edge_neg))

        if stresses_edge_neg[idx_edge_broken] <= 0:
            self._update_matrix(idx_edge_broken)
            self.idxs_edge_broken.append(idx_edge_broken)
            self._append_volts_edge_profile_dynamic()
            self.volts_ext.append(float(self.equation.volt_ext))
        
        else:
            factors_scaling = self.breaking_strengths / self.volts_edge
            factors_scaling[self.idxs_edge_broken] = np.inf
            factors_scaling[self.idxs_edge_unalive] = np.inf
            idx_edge_broken = int(np.argmin(factors_scaling))
            factor_scaling = factors_scaling[idx_edge_broken]

            self._update_matrix(idx_edge_broken)
            self.idxs_edge_broken.append(idx_edge_broken)

            self.equation.volt_ext *= factor_scaling
            self.equation.volts_node *= factor_scaling
            self._append_volts_edge_profile_scale_dynamic(factor_scaling)
            self.volts_ext.append(float(self.equation.volt_ext))