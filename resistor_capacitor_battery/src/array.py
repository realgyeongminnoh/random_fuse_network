import numpy as np 


class Array:
    def __init__(self, length: int):
        length = int(length)
        self.length: int = length

        self.num_node: int = length + 1
        self.num_node_mid: int = length - 1
        self.num_edge: int = length ** 2
        self.num_node_div: int = self.num_node + self.num_edge
        self.num_node_div_mid: int = self.num_node_div - 2
        self.num_edge_div: int = 2 * self.num_edge

        self.edges: list[tuple[int, int]] = self._generate_edges()
        self.edges_div_cap, self.edges_div_cond = self._generate_edges_div_cap_cond()
        
    def _generate_edges(self):
        length = self.length
        edges = []
        for idx_node in range(length):
            for _ in range(length):
                edges.append((idx_node, idx_node + 1))
        
        return edges
    
    def _generate_edges_div_cap_cond(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        edges_div_cap, edges_div_cond = [], []
        
        for (idx_node1, idx_node2), idx_node_div in zip(self.edges, range(self.num_node, self.num_node_div)):
            edges_div_cap.append((idx_node1, idx_node_div))
            edges_div_cond.append((idx_node2, idx_node_div))

        return edges_div_cap, edges_div_cond