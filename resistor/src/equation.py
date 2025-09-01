import numpy as np
from scipy.sparse.linalg import spsolve

from src.array import Array
from src.matrix import Matrix


class Equation:
    def __init__(self, array: Array, matrix: Matrix):
        self.array: Array = array
        self.matrix: Matrix = matrix
        self.failure = None
        
        self.volt_ext: float = 1.0
        self.vec_rhs: np.ndarray[np.float64] = np.zeros(self.matrix.size_cond, dtype=np.float64)
        self.volts_node: np.ndarray[np.float64] = None

    def _lazy_init_failure(self, failure):
        self.failure = failure    

    def solve_mmd(self) -> None:
        self.vec_rhs[-1] = self.volt_ext
        self.volts_node = spsolve(self.matrix.cond, self.vec_rhs, permc_spec="MMD_AT_PLUS_A")

    def solve_amd(self) -> bool:
        self.vec_rhs[-1] = self.volt_ext
        self.volts_node = spsolve(self.matrix.cond, self.vec_rhs, permc_spec="COLAMD")
        return np.abs(self.volts_node[-1]) > 1e-5 # only works with L >= 10 # heuristic non-dual-graph-based solution for macroscopic failure identification