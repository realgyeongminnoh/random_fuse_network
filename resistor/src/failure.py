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
        self.equation._lazy_init_failure(self)
        self.width = float(width)
        self.seed = int(seed)
        self.breaking_strengths: np.ndarray[np.float64] = self._generate_breaking_strengths()

        self.degrees: np.ndarray[np.int32] = np.array([1] * array.length + [4] * array.num_node_mid + [1] * array.length, dtype=np.int32)
        self.volts_edge: np.ndarray[np.float64] = np.empty(self.array.num_edge, dtype=np.float64)
        self.idxs_edge_broken: list[int] = []
        self.idxs_edge_leaf: list[int] = []
        self.idxs_edge_island: list[int] = []
        self.volts_ext: list[float] = []

        self.parents: np.ndarray[np.int32] = self.array.parents.copy()
        self.sizes: np.ndarray[np.int32] = self.array.sizes.copy()
        self.parities: np.ndarray[np.uint8] = self.array.parities.copy()

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
        # np.array(failure.volts_edge_profile)[:, array.idxs_edge_horizontal_no_pbc] *= -1.0
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

    def _find_edge_broken_leaf_proof(self, quantities):
        degrees = self.degrees
        idx_edge_broken = int(np.argmin(quantities))
        idx_node1, idx_node2 = self.array.edges[idx_edge_broken]
        degrees[idx_node1] -= 1
        degrees[idx_node2] -= 1

        if idx_node1 < self.array.length: # top edge
            if degrees[idx_node2] == 0:
                degrees[idx_node1] += 1
                degrees[idx_node2] += 1
                quantities[idx_edge_broken] = np.inf
                self.idxs_edge_leaf.append(idx_edge_broken)
                return self._find_edge_broken_leaf_proof(quantities) # recurse with updated quantities which exclude the above idx_edge_broken (leaf)
            return idx_edge_broken
        
        if idx_node2 > self.array.idx_node_bot_first_minus_one: # bot edge
            if degrees[idx_node1] == 0:
                degrees[idx_node1] += 1
                degrees[idx_node2] += 1
                quantities[idx_edge_broken] = np.inf
                self.idxs_edge_leaf.append(idx_edge_broken)
                return self._find_edge_broken_leaf_proof(quantities)
            return idx_edge_broken
        
        # mid edge
        if degrees[idx_node1] * degrees[idx_node2] == 0:
            degrees[idx_node1] += 1
            degrees[idx_node2] += 1
            quantities[idx_edge_broken] = np.inf
            self.idxs_edge_leaf.append(idx_edge_broken)
            return self._find_edge_broken_leaf_proof(quantities)
        return idx_edge_broken

    def break_edge_init(self):
        self._compute_volts_edge_init()
        idx_edge_broken = int(np.argmin(self.breaking_strengths / self.volts_edge))
        idx_node1, idx_node2 = self.array.edges[idx_edge_broken]
        self.degrees[idx_node1] -= 1
        self.degrees[idx_node2] -= 1

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
        stresses_edge_neg[self.idxs_edge_leaf] = np.inf
        stresses_edge_neg[self.idxs_edge_island] = np.inf
        idx_edge_broken = self._find_edge_broken_leaf_proof(stresses_edge_neg)

        if stresses_edge_neg[idx_edge_broken] <= 0:
            self._update_matrix(idx_edge_broken)
            self.idxs_edge_broken.append(idx_edge_broken)
            self._append_volts_edge_profile_dynamic()
            self.volts_ext.append(float(self.equation.volt_ext))
        
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

            self.equation.volt_ext *= factor_scaling
            self.equation.volts_node *= factor_scaling
            self._append_volts_edge_profile_scale_dynamic(factor_scaling)
            self.volts_ext.append(float(self.equation.volt_ext))

    def _dsu_find(self, idx_face: int) -> tuple[int, np.uint8]:
        if self.parents[idx_face] == idx_face: # root
            return idx_face, 0
        root, parity = self._dsu_find(int(self.parents[idx_face])) # recursion scope until root is found

        # recursion unwind ( per face along the path(root(initial face) -> initial face) )
        self.parents[idx_face] = root
        self.parities[idx_face] ^= parity
        return root, self.parities[idx_face] # return to-root parity of initial face

    def _dsu_union(self, root_face1: int, root_face2: int, xor_triple: np.uint8) -> None:
        if self.sizes[root_face1] > self.sizes[root_face2]:
            root_face1, root_face2 = root_face2, root_face1 # size(root(f1)) <= size(root(f2))
            
        self.parents[root_face1] = root_face2 # parent(root(f1)) = root(f2)
        self.sizes[root_face2] += self.sizes[root_face1]
        self.parities[root_face1] = xor_triple

    def _dsu_update(self, idx_edge_broken: int) -> int:
        idx_face1, idx_face2 = self.array.idxs_edge_to_edges_dual[idx_edge_broken] # dual(e_b) = (f1, f2)
        parity_edge_broken = self.array.parities_edge_broken[idx_edge_broken]

        root_face1, parity_face1 = self._dsu_find(idx_face1)
        root_face2, parity_face2 = self._dsu_find(idx_face2)

        if root_face1 != root_face2: # root(f1) =/= root(f2)
            # parity(f1) ^ parity(root(f1) -> root(f2)) ^ parity(f2) [DSU TREE] == parity(edge_broken(f1->f2)) [DUAL GRAPH]
            self._dsu_union(root_face1, root_face2, parity_face1 ^ parity_face2 ^ parity_edge_broken)
            return 0 # no cycle

        # root(f1) = root(f2)
        if parity_face1 ^ parity_face2 ^ parity_edge_broken == 1:
            return 1 # termination cycle # odd #(seam crossing dual edges) in the cycle path
        return 2 # island cycle # even #(seam crossing dual edges) in the cycle path