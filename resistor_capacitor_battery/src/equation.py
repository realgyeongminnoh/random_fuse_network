import numpy as np
from scipy.sparse.linalg import spsolve

from .array import Array
from .matrix import Matrix


class Equation:
    # for this model with a simpler topology (compared to the original 2D lattice)
    # there's no leaf/island edge issue so things can become much simpler
    def __init__(self, array: Array, matrix: Matrix):
        self.array: Array = array
        self.matrix: Matrix = matrix
        self.failure = None

        self.volt_ext: float = 1.0
        self.volts_node_div: np.ndarray[np.float64] = np.empty(self.matrix.size_div_comb, dtype=np.float64)

    def _lazy_init_failure(self, failure):
        self.failure = failure
    
    def solve_init(self) -> None:
        """
        "hard-coded" initial nodal voltage computation
        """
        length = self.array.length
        volts_node_div = self.volts_node_div
        volts_node_div_unique = np.linspace(1, 1 / length, length)

        volts_node_div[-1] = -1.0
        volts_node_div[:length - 1] = volts_node_div_unique[1:]
        volts_node_div[length - 1:-1] = np.repeat(volts_node_div_unique, length)

    def check_graph(self) -> bool:
        """
        graph-based termination condition (network breakdown) check
        """
        return ~np.any(self.failure.num_edge_per_layer == 0)

    def solve(self) -> None:
        self.volts_node_div = self.matrix.div_cap @ self.volts_node_div
        self.volts_node_div[-1] = 0
        self.volts_node_div = spsolve(self.matrix.div_comb, self.volts_node_div, permc_spec="MMD_AT_PLUS_A")