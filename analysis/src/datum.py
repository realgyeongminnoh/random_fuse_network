import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from resistor_capacitor.src.array import Array
from resistor_capacitor.src.failure import Failure


class Datum:
    "for single simulation (seed)"
    def __init__(self, failure: Failure):
        self.array: Array = failure.array
        self.failure: Failure = failure

        # model
        if not hasattr(failure.matrix, "val_cap"):
            self.model: str = "resistor"
        elif hasattr(failure.array, "idxs_edge_vertical"):
            self.model: str = "resistor_capacitor"
        else:
            self.model: str = "resistor_capacitor_battery"

    def check_profile(self) -> float:
        "must return a small value arising from round-off errors (floating point)"
        if self.model == "resistor":
            return 0.0
        
        volts_cap_profile, volts_cond_profile, volts_edge_profile = (
            np.array(self.failure.volts_cap_profile), np.array(self.failure.volts_cond_profile), np.array(self.failure.volts_edge_profile))

        # broken -> nan
        for idx_time, idx_edge_broken in enumerate(self.failure.idxs_edge_broken):
            volts_cap_profile[(idx_time + 1):, idx_edge_broken] = np.nan
            volts_cond_profile[(idx_time + 1):, idx_edge_broken] = np.nan
            volts_edge_profile[(idx_time + 1):, idx_edge_broken] = np.nan

        # horizontal non pbc edges sign-flipping
        if self.model == "resistor_capacitor":
            volts_edge_profile[:, Array(self.array.length, True).idxs_edge_horizontal_no_pbc] *= -1.0

        return float(np.nanmax(np.abs(volts_edge_profile - (volts_cap_profile - volts_cond_profile))))

    def check_connectivity(self) -> int:
        "must give 2; does not guarantee: [connectivity == 2] --> [crack from horizontal cut, not some island]; new check_graph guarantees 2"
        if self.model == "resistor_capacitor_battery":
            return 2
        
        length, edges = self.array.length, self.array.edges.copy()
        edges.extend([(idx_node, idx_node + 1) for idx_node in range(length - 1)] + [(0, length - 1)])
        edges.extend([(idx_node, idx_node + 1) for idx_node in range(length ** 2, length ** 2 + length - 1)] + [(length ** 2, length ** 2 + length - 1)])
        edges = np.array(edges, dtype=np.int32)

        graph = nx.Graph()
        graph.add_edges_from(edges)
        graph.remove_edges_from(edges[self.failure.idxs_edge_broken])
        return nx.number_connected_components(graph)

    def check_num_cycle(self) -> int:
        "must give 1l new check_graph also guarantees 2 as well as previous DSU + parity method"
        if self.model == "resistor_capacitor_battery":
            return 1
        
        graph_dual = nx.Graph()
        graph_dual.add_edges_from([self.array.idxs_edge_to_edges_dual[idx_edge_broken] for idx_edge_broken in self.failure.idxs_edge_broken])
        return len(nx.cycle_basis(graph_dual))

    def draw_profile(
        self, type_profile: str, signed: bool = False, 
        size_fig: float = 10.0, size_obj: float = 1.0, rainbow: bool = False
    ) -> None:
        
        # volts_profile
        if type_profile == "edge":
            volts_profile = np.array(self.failure.volts_edge_profile, dtype=np.float64).transpose() # [edge, time]
        elif type_profile == "stress": # stress per bond per time
            volts_profile = np.array(
                np.abs(self.failure.volts_edge_profile) - np.tile(self.failure.breaking_strengths, (len(self.failure.idxs_edge_broken), 1)), 
                dtype=np.float64).transpose() # [edge, time]
        elif type_profile == "cap":
            if self.model == "resistor": return
            volts_profile = np.array(self.failure.volts_cap_profile, dtype=np.float64).transpose() # [edge, time]
        elif type_profile == "cond":
            if self.model == "resistor": return
            volts_profile = np.array(self.failure.volts_cond_profile, dtype=np.float64).transpose() # [edge, time]
        if not signed and not (type_profile == "stress"): np.abs(volts_profile, out=volts_profile)
        num_edge, num_time = volts_profile.shape

        # idxs_edge_broken
        idxs_edge_broken: np.ndarray = np.array(self.failure.idxs_edge_broken, dtype=np.int32)
        num_broken_edge = len(idxs_edge_broken)

        # broken -> nan
        for idx_time, idx_edge_broken in enumerate(idxs_edge_broken):
            volts_profile[idx_edge_broken, (idx_time + 1):] = np.nan

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(size_fig, size_fig), dpi=72)

        if not rainbow:
            # plot
            ax.scatter(0, volts_profile[idxs_edge_broken[0], 0], c="red", s=size_obj, zorder=num_edge)
            for idx_edge, volts_per_edge in enumerate(volts_profile):
                if idx_edge in idxs_edge_broken:
                    ax.plot(range(num_time), volts_per_edge, c="red", lw=size_obj, zorder=num_edge)
                else:
                    ax.plot(range(num_time), volts_per_edge, c="black", lw=size_obj)
        else:
            # color
            colors_rainbow = plt.colormaps["jet_r"](np.linspace(0, 1, num_broken_edge))[:, :3]
            colors = np.zeros((num_edge, 3), dtype=np.float64)
            for idx_edge_broken, color_rainbow in zip(idxs_edge_broken, colors_rainbow):
                colors[idx_edge_broken] = color_rainbow
            
            # zorder
            zorders = np.zeros(num_edge, dtype=np.int32)
            zorders[idxs_edge_broken] = np.arange(num_broken_edge, 0, -1)

            # plot
            ax.scatter(0, volts_profile[idxs_edge_broken[0], 0], color=colors[idxs_edge_broken[0]], s=size_obj, zorder=zorders[[idxs_edge_broken[0]]])
            for (idx_edge, volts_per_edge), color, zorder in zip(enumerate(volts_profile), colors, zorders):
                if idx_edge in idxs_edge_broken:
                    ax.plot(range(num_time), volts_per_edge, c=color, lw=size_obj, zorder=zorder)
                else:
                    ax.plot(range(num_time), volts_per_edge, c=color, lw=size_obj, zorder=zorder)

        # plot details

        plt.tight_layout()
        plt.show()

    def initialize_graph(
        self, size_fig: float = 10.0, size_obj: float = 2.0, draw_dual: bool = False, draw_unalive: bool = False,
        save: bool = False, pad_inches: float = 0.7, transparent: bool = False
    ) -> None:
        self.size_fig, self.size_obj, self.draw_dual, self.draw_unalive, self.save, self.pad_inches, self.transparent = size_fig, size_obj, draw_dual, draw_unalive, save, pad_inches, transparent
        
        if self.draw_dual:
            self.array = Array(length=self.array.length, mode_analysis=True)
        length = self.array.length
        num_node = length ** 2 + length
        self.idxs_edge_pbc = (length + 1) + (2 * length) * np.arange(length - 1, dtype=np.int32)
        self.num_edge_broken = len(self.failure.idxs_edge_broken)

        if self.model != "resistor_capacitor_battery":
            # graph generation (PBC cylinder --> 2D rectangle)
            self.edges_pseudo = np.array(self.array.edges, dtype=np.int32)
            self.edges_pseudo[self.idxs_edge_pbc] = np.vstack([np.arange(-1, -length, -1, dtype=np.int32), self.edges_pseudo[self.idxs_edge_pbc][:, 1]]).T
            self.graph = nx.Graph()
            self.graph.add_nodes_from(range((-length + 1), num_node))
            self.graph.add_edges_from(self.edges_pseudo)

            # node position generation
            pos_pseudo = {idx_node: (idx_node % length, length - idx_node // length) for idx_node in range(num_node)}
            self.pos = {
                idx_node: tuple(pos_node.tolist())
                for (idx_node, pos_node) in zip(
                    range(-length + 1, 0),
                    np.vstack([np.full((length - 1), length), np.arange(1, length)]).T,
                )
            } | pos_pseudo

            # base color generation (in the same orders as node & edge list in draw.graph)
            self.colors_node = np.zeros((num_node + length - 1, 3), dtype=np.int8)
            self.colors_node[range(length - 1)] = np.ones(3, dtype=np.int8)
            self.colors_edge = np.zeros((self.array.num_edge, 3), dtype=np.int8)

            # busbar graph generation
            self.graph_busbar = nx.Graph()
            self.graph_busbar.add_edges_from([(0, (length - 1)), ((length ** 2), (length ** 2 + length - 1))])
            self.pos_busbar = {idx_node: self.pos[idx_node] for idx_node in list(self.graph_busbar.nodes)}

        else: # resistor_capacitor_battery
            # graph generation (already can be represented by 2D rectangle; just the supernode "resolved" into multiple pseudo nodes for plotting)
            edges_pseudo = [(idx_node, idx_node + length) for idx_node in range(length ** 2)]
            self.graph = nx.Graph()
            self.graph.add_nodes_from(range(num_node))
            self.graph.add_edges_from(edges_pseudo)

            # node position generation
            self.pos = {idx_node: (idx_node % length, length - idx_node // length) for idx_node in range(num_node)}

            # base color generation (in the same orders as node & edge list in draw.graph)
            self.colors_node = np.zeros((1, 3), dtype=np.int8)
            self.colors_edge = np.zeros((self.array.num_edge, 3), dtype=np.int8)

            # busbar graph generation
            self.graph_busbar = nx.Graph()
            self.graph_busbar.add_edges_from([(idx_node, idx_node + length -1) for idx_node in range(0, length ** 2 + length, length)])
            self.pos_busbar = {idx_node: self.pos[idx_node] for idx_node in list(self.graph_busbar.nodes)}

        # dual graph
        self.num_cycle: int = 0
        self.cycles_pseudo: list = []
        self.pos_dual_pseudo: dict = {}

        def _generate_pos_dual(self, length: int) -> None:
            pos_dual_x, pos_dual_y = np.meshgrid(np.arange(length) + 0.5, np.arange(length, 0, -1) - 0.5)
            pos_dual_x, pos_dual_y = pos_dual_x.reshape(-1), pos_dual_y.reshape(-1)
            pos_dual = {idx_face: (float(pos_face_x), float(pos_face_y)) for idx_face, (pos_face_x, pos_face_y) in enumerate(zip(pos_dual_x, pos_dual_y))}
            if self.model != "resistor_capacitor_battery":
                self.pos_dual_pseudo = {(-length + i): (length + 0.5, i + 0.5) for i in range(length)} | pos_dual
            else:
                self.pos_dual_pseudo = {(-length + i): (-0.5, i + 0.5) for i in range(length)} | pos_dual

        _generate_pos_dual(self, length)

        def _generate_cycles_pseudo(self) -> None:
            length, length_minus_one = self.array.length, self.array.length - 1

            if self.model != "resistor_capacitor_battery":
                graph_dual = nx.Graph()
                graph_dual.add_edges_from([self.array.idxs_edge_to_edges_dual[idx_edge_broken] for idx_edge_broken in self.failure.idxs_edge_broken])

                for cycle_in_face in nx.cycle_basis(graph_dual):
                    self.num_cycle += 1
                    cycle = [] # cycle in dual edge
                    for idx_face1, idx_face2 in zip(cycle_in_face[:-1], cycle_in_face[1:]):
                        if idx_face1 > idx_face2:
                            idx_face1, idx_face2 = idx_face2, idx_face1
                        cycle += [(idx_face1, idx_face2)]
                    cycle += [(cycle_in_face[0], cycle_in_face[-1]) if cycle_in_face[0] < cycle_in_face[-1] else (cycle_in_face[-1], cycle_in_face[0])]

                    self.cycles_pseudo.append([(-(idx_face2 + 1) // length, idx_face2) if (idx_face2 - idx_face1 == length_minus_one) else (idx_face1, idx_face2) 
                        for idx_face1, idx_face2 in cycle]) # cycle pseudo for drawing
            
            else: # resistor_capacitor_battery                
                num_edge_per_layer = np.array([length] * length)
                for idx_edge_broken in self.failure.idxs_edge_broken:
                    num_edge_per_layer[idx_edge_broken // length] -= 1

                for idx_layer_cycle in np.where(num_edge_per_layer == 0)[0]: # idxs_layer_cycle
                    self.num_cycle += 1
                    edges = np.array(self.array.edges)

                    edges_cycle = [(idx_node + (idx_layer_cycle * length), idx_node + length + (idx_layer_cycle * length)) for idx_node in range(length)]
                    idxs_edge_cycle = [int(np.where((edges[:,0] == u) & (edges[:,1] == v))[0][0]) for (u,v) in edges_cycle]
                    cycle = [self.array.idxs_edge_to_edges_dual[idx_edge_broken] for idx_edge_broken in idxs_edge_cycle]
                    self.cycles_pseudo.append([(-(idx_face2 + 1) // length, idx_face1) if (idx_face2 - idx_face1 == length_minus_one) else (idx_face1, idx_face2) 
                        for idx_face1, idx_face2 in cycle]) # cycle pseudo for drawing          
                    
        _generate_cycles_pseudo(self)

    def draw_graph(
        self, idxs_edge_broken: list | None = None, num_edge_broken_rainbow: int | None = 0, with_labels: bool = False,
    ) -> None:
        if idxs_edge_broken is None: idxs_edge_broken = self.failure.idxs_edge_broken
        if num_edge_broken_rainbow is None: num_edge_broken_rainbow = len(idxs_edge_broken)
        
        # edge coloring
        colors_edge = self.colors_edge.astype(np.int8)
        colors_edge[idxs_edge_broken] = np.ones(3, dtype=np.int8)
        if num_edge_broken_rainbow > 0:
            colors_edge = colors_edge.astype(np.float16)
            if num_edge_broken_rainbow == 1:
                colors_edge_broken = np.array([0, 0, 1], dtype=np.float16)
            else:
                colors_edge_broken = plt.colormaps["jet_r"](np.linspace(0.1, 0.9, num_edge_broken_rainbow))[:, :3]
            colors_edge[idxs_edge_broken[-num_edge_broken_rainbow:]] = colors_edge_broken
        
        if self.model != "resistor_capacitor_battery":
            if self.draw_unalive:
                colors_edge[self.failure.idxs_edge_unalive] = np.array([0, 1, 0], dtype=np.float16)
            colors_edge = np.vstack([colors_edge[self.idxs_edge_pbc][::-1], np.delete(colors_edge, self.idxs_edge_pbc, axis=0)])

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(self.size_fig, self.size_fig), dpi=72)

        nx.draw(
            self.graph, self.pos, ax=ax,
            node_shape="s", node_color=self.colors_node, edgecolors="none", node_size=self.size_obj ** 2,
            edge_color=colors_edge, width=self.size_obj,
            with_labels=with_labels, font_color="blue", font_size=self.size_fig * 2,
        )

        nx.draw(
            self.graph_busbar, self.pos_busbar, ax=ax,
            node_shape="s", node_color="black", edgecolors="none", node_size=4 * self.size_obj ** 2,
            edge_color="black", width=self.size_obj * 2,
        )

        if self.draw_dual:
            colors_cycle = plt.colormaps["jet_r"](np.linspace(0.1, 0.9, self.num_cycle))[:, :3]
            for cycle_pseudo, color_cycle in zip(self.cycles_pseudo, colors_cycle):
                graph_dual = nx.Graph()
                graph_dual.add_edges_from(cycle_pseudo)

                nx.draw(
                    graph_dual, pos=self.pos_dual_pseudo, ax=ax,
                    node_shape="s", node_color=[color_cycle], edgecolors="none", node_size=self.size_obj ** 2,
                    edge_color=[color_cycle], width=self.size_obj,
                    with_labels=with_labels, font_color="green", font_size=self.size_fig * 2,
                )

        # plot detail

        plt.tight_layout()
        plt.show()