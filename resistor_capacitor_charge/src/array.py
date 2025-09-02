import numpy as np


class Array:
    def __init__(self, length: int, mode_analysis: bool = False):
        self.mode_analysis: bool = mode_analysis
        length = int(length)
        self.length: int = length
        self.length_double: int = 2 * length
        self.num_node: int = length ** 2 + length
        self.num_node_mid: int = self.num_node - self.length_double
        self.idx_node_bot_first_minus_one: int = self.num_node - self.length - 1
        self.num_edge: int = 2 * length ** 2 - length
        
        self.num_node_div: int = self.num_node + self.num_edge
        self.num_node_div_mid: int = self.num_node_div - self.length_double
        self.num_edge_div: int = 2 * self.num_edge

        self.edges: list[tuple[int, int]] = self._generate_edges()
        self.edges_div_cap, self.edges_div_cond = self._generate_edges_div_cap_cond()
        self._generate_idxs() # REORDERS E DIV CAP AND COND TO MATCH E ORDER

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

    def _generate_edges_div_cap_cond(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        # edges_div
        edges_div = []
        idx_node = self.num_node

        for idx_node1, idx_node2 in self.edges:
            edges_div.extend(((idx_node1, idx_node), (idx_node2, idx_node)))
            idx_node += 1

        edges_div.sort(key=lambda x: x)
    
        # edges_div_cap, edges_div_cond
        length = self.length
        length_minus_one = length - 1
        num_edge_div_minus_length = self.num_edge_div - length
        edges_div_cap, edges_div_cond = [], []
        counter = 0

        for idx_edge_div, edge_div in enumerate(edges_div):
            # edge_div_top
            if idx_edge_div < length:
                edges_div_cap.append(edge_div)
            # edge_div_mid
            elif length_minus_one < idx_edge_div < num_edge_div_minus_length:
                if edge_div[0] % length in (0, length_minus_one):
                    if counter in (2, 3):
                        edges_div_cap.append(edge_div)
                    else:
                        edges_div_cond.append(edge_div)
                    counter = (counter + 1) % 4
                else:
                    if counter in (1, 3): 
                        edges_div_cap.append(edge_div)
                    else:
                        edges_div_cond.append(edge_div)
                    counter = (counter + 1) % 4
            # edge_div_bot
            else:
                edges_div_cond.append(edge_div)

        return edges_div_cap, edges_div_cond

    def _generate_idxs(self):
        self.idxs_edge_top: np.ndarray[np.int32] = np.arange(self.length, dtype=np.int32)
        self.idxs_edge_bot: np.ndarray[np.int32] = self.num_edge + np.append(np.arange((-2 * self.length + 2), 0, 2, dtype=np.int32), np.int32(-1))
        self.idxs_edge_mid: np.ndarray[np.int32] = np.delete(np.arange(self.num_edge, dtype=np.int32), np.append(self.idxs_edge_top, self.idxs_edge_bot))
        
        # node index inside the nodal voltage vector in the matrix equation
        edges_minus_length = np.array(self.edges, dtype=np.int32) - self.length
        self.idxs_edge_bot_node1: np.ndarray[np.int32] = (self.num_node_mid - self.length) + self.idxs_edge_top
        self.idxs_edge_mid_node1: np.ndarray[np.int32] = edges_minus_length[self.idxs_edge_mid, 0]
        self.idxs_edge_mid_node2: np.ndarray[np.int32] = edges_minus_length[self.idxs_edge_mid, 1]

        # index mapping for cap
        # edges_div_cap[idxs[idx_edge]] <-> edges[idx_edge]
        idxs_edge_to_edge_div_cap: np.ndarray[np.int32] = np.array(
            sorted(range(self.num_edge), key=lambda idx_edge: self.edges_div_cap[idx_edge][1]), dtype=np.int32)

        # index mapping for cond
        # edges_div_cond[idxs[idx_edge]] <-> edges[idx_edge]
        idxs_edge_to_edge_div_cond: np.ndarray[np.int32] = np.array(
            sorted(range(self.num_edge), key=lambda idx_edge: self.edges_div_cond[idx_edge][1]), dtype=np.int32)
        
        # REORDERED TO MATCH E ORDER
        self.edges_div_cap[:] = [self.edges_div_cap[idx_edge_div] for idx_edge_div in idxs_edge_to_edge_div_cap]
        self.edges_div_cond[:] = [self.edges_div_cond[idx_edge_div] for idx_edge_div in idxs_edge_to_edge_div_cond]

        if self.mode_analysis:
            self.idxs_edge_to_edge_div_cap = idxs_edge_to_edge_div_cap
            self.idxs_edge_to_edge_div_cond = idxs_edge_to_edge_div_cond

            # index lists of horizontal, vertical, pbc, horizontal except pbc edges
            length, length_minus_one = self.length, self.length - 1

            idxs_edge_horizontal = []
            for idx_edge in np.linspace(length, length * (2 * length_minus_one - 1), length_minus_one, dtype=np.int32).tolist():
                idxs_edge_horizontal += [idx_edge]
                idxs_edge_horizontal += (idx_edge + 2 * np.arange(0, length_minus_one) + 1).tolist()

            self.idxs_edge_horizontal = np.array(idxs_edge_horizontal, dtype=np.int32)
            self.idxs_edge_vertical = np.delete(np.arange(self.num_edge, dtype=np.int32), self.idxs_edge_horizontal)
            self.idxs_edge_pbc = (length + 1) + (2 * length) * np.arange(length - 1, dtype=np.int32)
            self.idxs_edge_horizontal_no_pbc = self.idxs_edge_horizontal[~np.isin(self.idxs_edge_horizontal, self.idxs_edge_pbc, assume_unique=True)]