import numpy as np


class Array:
    def __init__(self, length: int):
        length = int(length)
        self.length: int = length
        self.num_node: int = length ** 2 + length
        self.num_node_mid: int = self.num_node - 2 * length
        self.idx_node_bot_first_minus_one: int = self.num_node - self.length - 1
        self.num_edge: int = 2 * length ** 2 - length

        self.edges: list[tuple[int, int]] = self._generate_edges()
        self._generate_idxs()

    def _generate_edges(self):
        length = self.length
        length_minus_one = length - 1
        edges = []
        idx_node = 0

        for j in range(length + 1):
            for i in range(length):
                if i < length_minus_one:
                    if 0 < j < length:
                        edges.append((idx_node, idx_node + 1))
                        if i == 0:
                            edges.append((idx_node, idx_node + length_minus_one))
                if j < length:
                    edges.append((idx_node, idx_node + length))
                idx_node += 1

        return edges

    def _generate_idxs(self):
        self.idxs_edge_top: np.ndarray[np.int32] = np.arange(self.length, dtype=np.int32)
        self.idxs_edge_bot: np.ndarray[np.int32] = self.num_edge + np.append(np.arange((-2 * self.length + 2), 0, 2, dtype=np.int32), np.int32(-1))
        self.idxs_edge_mid: np.ndarray[np.int32] = np.delete(np.arange(self.num_edge, dtype=np.int32), np.append(self.idxs_edge_top, self.idxs_edge_bot))
        
        # node index inside the nodal voltage vector in the matrix equation
        edge_list_minus_length = np.array(self.edges, dtype=np.int32) - self.length
        self.idxs_edge_bot_node1: np.ndarray[np.int32] = (self.num_node_mid - self.length) + self.idxs_edge_top
        self.idxs_edge_mid_node1: np.ndarray[np.int32] = edge_list_minus_length[self.idxs_edge_mid, 0]
        self.idxs_edge_mid_node2: np.ndarray[np.int32] = edge_list_minus_length[self.idxs_edge_mid, 1]

        # # horizontal and vertical edges
        # length, length_minus_one = self.length, self.length - 1

        # idxs_edge_horizontal = []
        # for idx_edge in np.linspace(length, length * (2 * length_minus_one - 1), length_minus_one, dtype=np.int32).tolist():
        #     idxs_edge_horizontal += [idx_edge]
        #     idxs_edge_horizontal += (idx_edge + 2 * np.arange(0, length_minus_one) + 1).tolist()

        # self.idxs_edge_horizontal = np.array(idxs_edge_horizontal, dtype=np.int32)
        # self.idxs_edge_vertical = np.delete(np.arange(self.num_edge, dtype=np.int32), self.idxs_edge_horizontal)