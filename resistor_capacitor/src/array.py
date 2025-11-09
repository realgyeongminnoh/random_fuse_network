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
        self._initialize_check_graph()
        
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

        # others
        length, length_minus_one, length_double = self.length, self.length - 1, self.length * 2
        self.idxs_edge_vertical: np.ndarray[np.int32] = np.concatenate((
            np.arange(length, dtype=np.int32),
            np.repeat(np.arange(length + 2, (length + 2) + length_double * length_minus_one, length_double, dtype=np.int32), length) 
            + np.tile(np.array([*range(0, length_minus_one * 2, 2)] + [length_minus_one * 2 - 1], dtype=np.int32), length - 1)
        ))
        if self.mode_analysis:
            self.idxs_edge_horizontal: np.ndarray[np.int32] = np.delete(np.arange(self.num_edge, dtype=np.int32), self.idxs_edge_vertical)
            self.idxs_edge_pbc: np.ndarray[np.int32] = (length + 1) + (2 * length) * np.arange(length - 1, dtype=np.int32)
            self.idxs_edge_horizontal_no_pbc: np.ndarray[np.int32] = self.idxs_edge_horizontal[~np.isin(self.idxs_edge_horizontal, self.idxs_edge_pbc, assume_unique=True)]

            self.idxs_edge_to_edge_div_cap = idxs_edge_to_edge_div_cap
            self.idxs_edge_to_edge_div_cond = idxs_edge_to_edge_div_cond

            # mapping idx_edge to edge_dual ( (face1, face2) )
            self.idxs_edge_to_edges_dual: list[tuple[int, int]] = [None] * self.num_edge
            for idx_edge, edge in enumerate(self.edges):
                idx_node1, idx_node2 = edge # idx_node1 < idx_node2

                # vertical
                if idx_node2 == idx_node1 + length:
                    if idx_node1 % length == 0: # EDGE_DUAL AFFECTED BY PBC PERIODICITY
                        idx_face1 = idx_node1
                        idx_face2 = idx_node2 - 1
                    else:
                        idx_face1 = idx_node1 - 1
                        idx_face2 = idx_node1
                    
                # horizontal
                else:
                    if idx_node1 % length == 0 and idx_node2 % length == length_minus_one: # EDGE_DUAL AFFECTED BY PBC PERIODICITY
                        idx_face1 = idx_node1 - 1
                        idx_face2 = idx_node2
                    else:
                        idx_face1 = idx_node1 - length
                        idx_face2 = idx_node1

                self.idxs_edge_to_edges_dual[idx_edge] = (idx_face1, idx_face2)

    def _initialize_check_graph(self):
        # 1) endpoints as int32 (edges stored i<j already)
        self.u, self.v = np.array(self.edges, dtype=np.int32).T

        # 2) terminals (shared, never mutate)
        self.is_top_node = np.zeros(self.num_node, dtype=bool);  self.is_top_node[:self.length] = True
        self.is_bot_node = np.zeros(self.num_node, dtype=bool);  self.is_bot_node[self.num_node - self.length:] = True
        self.idxs_node_top = np.arange(self.length, dtype=np.int32)
        self.idxs_node_bot = np.arange(self.num_node - self.length, self.num_node, dtype=np.int32)

        # 3) CSR adjacency for the UNDIRECTED graph (half-edges) + edge index per half-edge
        deg = np.zeros(self.num_node, dtype=np.int32)
        for (a, b) in zip(self.u, self.v):
            deg[a] += 1; deg[b] += 1
        indptr = np.empty(self.num_node + 1, dtype=np.int32)
        indptr[0] = 0; np.cumsum(deg, out=indptr[1:])

        indices = np.empty(indptr[-1], dtype=np.int32)
        half_edge_idx = np.empty(indptr[-1], dtype=np.int32)  # edge id per half-edge
        # fill
        cur = indptr.copy()
        for e, (a, b) in enumerate(zip(self.u, self.v)):
            ia = cur[a]; indices[ia] = b; half_edge_idx[ia] = e; cur[a] += 1
            ib = cur[b]; indices[ib] = a; half_edge_idx[ib] = e; cur[b] += 1

        self.indptr = indptr
        self.indices = indices
        self.half_edge_idx = half_edge_idx