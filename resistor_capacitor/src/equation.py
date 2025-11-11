import warnings
import numpy as np
from scipy.sparse.linalg import spsolve, splu

from .array import Array
from .matrix import Matrix


class Equation:
    def __init__(self, array: Array, matrix: Matrix):
        self.array: Array = array
        self.matrix: Matrix = matrix
        self.failure = None

        self.volt_ext: float = 1.0
        self.volts_node_div: np.ndarray[np.float64] = np.empty(self.matrix.size_div_comb, dtype=np.float64)
        self._initialize_check_graph()

    def _lazy_init_failure(self, failure):
        self.failure = failure

    def solve_init(self) -> None:
        length = self.array.length
        volts_node = np.concatenate(([1.0], np.linspace((length - 1) / length, 1 / length, length - 1).repeat(length), [-1.0]))

        length, length_double = self.array.length, self.array.length_double
        volts_node_div = self.volts_node_div

        volts_node_div[-1] = volts_node[-1]
        volts_node_div[:self.array.num_node_mid] = volts_node[1:-1]
        volts_node_div[self.array.num_node_mid:self.array.num_node_mid + length] = self.volt_ext
        for idx_node1, idx_node2 in self.array.edges_div_cap[length:]:
            volts_node_div[idx_node2 - length_double] = volts_node_div[idx_node1 - length]

    def solve(self) -> None:
        self.volts_node_div = self.matrix.div_cap @ self.volts_node_div
        self.volts_node_div[-1] = 0 # for numerical stability # external current of RC cannot be used as a heuristic termination condition
        self.volts_node_div = spsolve(self.matrix.div_comb, self.volts_node_div, permc_spec="MMD_AT_PLUS_A")

    def _initialize_check_graph(self):
        num_node = self.array.num_node
        num_edge = self.array.num_edge

        self._cg_disc = np.empty(num_node, np.int32)
        self._cg_low = np.empty(num_node, np.int32)
        self._cg_parent = np.empty(num_node, np.int32)
        self._cg_parent_edge = np.empty(num_node, np.int32)
        self._cg_tin = np.empty(num_node, np.int32)
        self._cg_tout = np.empty(num_node, np.int32)
        self._cg_tops_sub = np.empty(num_node, np.int32)
        self._cg_bots_sub = np.empty(num_node, np.int32)
        self._cg_in_st = np.empty(num_node, np.bool_)
        self._cg_dead = np.empty(num_edge, np.bool_)
        self._cg_intact = np.empty(num_edge, np.bool_)
        self._cg_edge_unal = np.empty(num_edge, np.bool_)

        # queues/stacks (ring buffers)
        self._cg_q = np.empty(num_node, np.int32)  # BFS queue
        self._cg_stack_node = np.empty(num_node, np.int32)  # DFS node stack
        self._cg_stack_ptr = np.empty(num_node, np.int32)  # DFS neighbor cursor

    def check_graph(self) -> bool:
        """
        Update failure.idxs_edge_unalive (indices of edges with theoretically 0 current during transient regime - free of round-off errors)
        return False iff TOPS and BOTTOMS are disconnected (termination).
        """
        A = self.array
        F = self.failure

        u, v = A.u, A.v
        indptr, indices, hedge = A.indptr, A.indices, A.half_edge_idx
        is_top, is_bot = A.is_top_node, A.is_bot_node

        # -------------------------------------------------
        # Dead/intact masks (broken ∪ previously-unalive)
        # -------------------------------------------------
        dead = self._cg_dead; dead.fill(False)
        if F.idxs_edge_broken:
            dead[np.asarray(F.idxs_edge_broken, np.int32)] = True
        if F.idxs_edge_unalive:
            dead[np.asarray(F.idxs_edge_unalive, np.int32)] = True
        intact = self._cg_intact; np.logical_not(dead, out=intact)

        # -------------------------------------------------
        # BFS: s–t component (seed with ALL top nodes)
        # -------------------------------------------------
        in_st = self._cg_in_st; in_st.fill(False)
        q = self._cg_q; head = 0; tail = 0

        # prefer precomputed top indices if present
        starts = A.idxs_node_top
        in_st[starts] = True
        q[:starts.size] = starts
        head = 0
        tail = starts.size
        while head < tail:
            x = q[head]; head += 1
            lo, hi = indptr[x], indptr[x+1]
            for i in range(lo, hi):
                eid = hedge[i]
                if not intact[eid]:
                    continue
                y = indices[i]
                if not in_st[y]:
                    in_st[y] = True
                    q[tail] = y; tail += 1

        # termination: no bottom reachable from tops
        if not in_st[A.idxs_node_bot].any():
            self.failure._compute_volts_cap() # LAST COMPUTE VOLTS CAP
            return False

        # -------------------------------------------------
        # Edges outside the s–t component are unalive now
        # -------------------------------------------------
        edge_unal = self._cg_edge_unal
        np.logical_and(intact, ~(in_st[u] & in_st[v]), out=edge_unal)

        # -------------------------------------------------
        # DFS low-links (iterative) over s–t component only
        # Peel only TRUE islands: child-subtree with NO tops AND NO bottoms
        # -------------------------------------------------
        disc = self._cg_disc; disc.fill(-1)
        low = self._cg_low
        parent = self._cg_parent; parent.fill(-1)
        parent_edge = self._cg_parent_edge; parent_edge.fill(-1)
        tin = self._cg_tin; tin.fill(-1)
        tout = self._cg_tout; tout.fill(-1)
        tops_sub = self._cg_tops_sub
        bots_sub = self._cg_bots_sub

        t = 0
        stack_n = self._cg_stack_node
        stack_p = self._cg_stack_ptr

        comp_nodes = np.flatnonzero(in_st)  # roots only inside the s–t component
        for root in comp_nodes:
            if disc[root] != -1:
                continue
            # discover root
            sp = 0
            stack_n[sp] = root; stack_p[sp] = indptr[root]; sp += 1
            disc[root] = low[root] = t; tin[root] = t; t += 1
            tops_sub[root] = 1 if is_top[root] else 0
            bots_sub[root] = 1 if is_bot[root] else 0

            while sp:
                u0 = stack_n[sp-1]
                lo, hi = indptr[u0], indptr[u0+1]
                cur = stack_p[sp-1]

                # advance to next usable neighbor (intact & in s–t component)
                while cur < hi:
                    eid = hedge[cur]
                    if intact[eid]:
                        w = indices[cur]
                        if in_st[w] and w != parent[u0]:
                            break
                    cur += 1

                if cur < hi:
                    # found a usable neighbor at cursor 'cur'
                    w = indices[cur]
                    stack_p[sp-1] = cur + 1  # advance cursor before descending/processing

                    if disc[w] == -1:
                        # tree edge: descend
                        parent[w] = u0
                        parent_edge[w] = hedge[cur]
                        disc[w] = low[w] = t; tin[w] = t; t += 1
                        tops_sub[w] = 1 if is_top[w] else 0
                        bots_sub[w] = 1 if is_bot[w] else 0
                        stack_n[sp] = w
                        stack_p[sp] = indptr[w]
                        sp += 1
                    else:
                        # back edge: update low-link
                        if disc[w] < low[u0]:
                            low[u0] = disc[w]
                else:
                    # no more neighbors for u0 → finish node
                    tout[u0] = t; t += 1
                    sp -= 1
                    p = parent[u0]
                    if p != -1:
                        if low[u0] < low[p]:
                            low[p] = low[u0]
                        tops_sub[p] += tops_sub[u0]
                        bots_sub[p] += bots_sub[u0]
                        # bridge p--u0 ?
                        if low[u0] > disc[p]:
                            # peel only TRUE islands (no tops AND no bottoms)
                            if (tops_sub[u0] == 0) and (bots_sub[u0] == 0):
                                e_bridge = parent_edge[u0]
                                if e_bridge != -1 and intact[e_bridge]:
                                    edge_unal[e_bridge] = True
                                in_sub = (tin >= tin[u0]) & (tout <= tout[u0])
                                sub_mask = intact & in_sub[u] & in_sub[v]
                                edge_unal[sub_mask] = True

        old = F.idxs_edge_unalive
        if old:
            edge_unal[np.asarray(old, np.int32)] = True

        F.idxs_edge_unalive = np.flatnonzero(edge_unal).tolist()
        return True