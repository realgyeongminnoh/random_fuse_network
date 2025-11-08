import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import re, os, struct, math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from resistor_capacitor.src.array import Array
from resistor_capacitor.src.failure import Failure


class Data:
    "for multiple simulation (seed)"
    def __init__(self, model: str, val_cap: str | None = None):
        self.model = model
        self.val_cap = val_cap
        self.path_base = Path(__file__).resolve().parents[2]/ model / "data"
        if model != "resistor":
            self.path_base = self.path_base / val_cap

        self.config_to_seeds, self.config_to_num_seed = self._get_configs()
    
    def _get_configs(self):
        is_float = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$').match
        tmp = defaultdict(list)

        with os.scandir(self.path_base) as it_length:
            for e_length in it_length:
                if not (e_length.is_dir() and e_length.name.isdigit()):
                    continue
                length = int(e_length.name)

                with os.scandir(e_length.path) as it_width:
                    for e_width in it_width:
                        name_w = e_width.name
                        if not (e_width.is_dir() and is_float(name_w)):
                            continue
                        width = float(name_w)

                        seeds = []
                        with os.scandir(e_width.path) as it_seed:
                            for e_seed in it_seed:
                                if not (e_seed.is_dir() and e_seed.name.isdigit()):
                                    continue
                                seed = int(e_seed.name)

                                with os.scandir(e_seed.path) as it_files:
                                    if next(it_files, None) is not None:
                                        seeds.append(seed)

                        if seeds:
                            tmp[(length, width)] = np.array(sorted(seeds), dtype=np.int32)

        keys = sorted(tmp.keys(), key=lambda k: (k[0], k[1]))
        config_to_seeds: dict[tuple[int, float], np.ndarray[np.int32]] = {k: tmp[k] for k in keys}
        config_to_num_seed: dict[tuple[int, float], int] = {k: tmp[k].size for k in keys}
        return config_to_seeds, config_to_num_seed
    
    def check_volts_ext(self) -> list[tuple[int, float, int]]:
        path_base = self.path_base

        def is_empty_or_last_nan_inf(path_npy: Path) -> bool:
            unpack_h = struct.unpack
            unpack_d = struct.unpack
            isfinite = math.isfinite

            with open(path_npy, 'rb') as f:
                f.read(6)
                vmaj = f.read(1)[0]
                f.read(1)

                if vmaj == 1:
                    header_len = unpack_h('<H', f.read(2))[0]
                else:
                    header_len = unpack_h('<I', f.read(4))[0]

                data_offset = f.tell() + header_len

                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size <= data_offset:
                    return True

                f.seek(size - 8, os.SEEK_SET)
                v = unpack_d('<d', f.read(8))[0]
                return not isfinite(v)
            
        def check_one(task):
            length, width, seed = task
            path_npy = path_base / str(length) / str(width) / str(seed) / "volts_ext.npy"
            return (length, width, seed) if is_empty_or_last_nan_inf(path_npy) else None

        tasks = (
            (length, width, seed)
            for (length, width), seeds in self.config_to_seeds.items()
            for seed in seeds
        )

        results = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            for result in executor.map(check_one, tasks):
                if result:
                    results.append(result)
                    # print(*result)
        return results

    @staticmethod    
    def _connectivity_parallel_job(seed):
        graph = graph_init_global.copy()
        graph.remove_edges_from(edges[np.load(path_length_width_global/ str(seed) / "idxs_edge_broken.npy")])
        return (length_global, width_global, seed) if nx.number_connected_components(graph) != 2 else None
    
    def check_connectivity(self) -> list[tuple[int, float, int]]:
        "does not guarantee: [connectivity == 2] --> [crack from horizontal cut, not some island]"
        if self.model == "resistor_capacitor_battery": # rcb already passes this
            return []
        
        results = []
        for length in np.unique([l for (l, _) in list(self.config_to_num_seed)]).astype(np.int32):
            global edges, graph_init_global, length_global
            length_global = int(length)
            edges = Array(length).edges

            # top & bottom edges loop
            edges.extend([(idx_node, idx_node + 1) for idx_node in range(length - 1)] + [(0, length - 1)])
            edges.extend([(idx_node, idx_node + 1) for idx_node in range(length ** 2, length ** 2 + length - 1)] + [(length ** 2, length ** 2 + length - 1)])
            edges = np.array(edges, dtype=np.int32)

            graph_init_global = nx.Graph()
            graph_init_global.add_edges_from(edges)

            for width in [w for (l, w) in list(self.config_to_num_seed) if l == length]:
                global width_global, path_length_width_global
                width_global = float(width)
                path_length_width_global = self.path_base / str(length_global) / str(width_global)

                with ProcessPoolExecutor(max_workers=18) as executor:
                    for result in executor.map(Data._connectivity_parallel_job, self.config_to_seeds[(int(length), width)]): 
                        if result:
                            results.append(result)
                            # print(*result)
        return results

    @staticmethod
    def _num_cycle_parallel_job(seed):
        graph_dual = nx.Graph()
        graph_dual.add_edges_from(idxs_edge_to_edges_dual_global[np.load(path_length_width_global/ str(seed) / "idxs_edge_broken.npy")])
        return (length_global, width_global, seed) if len(nx.cycle_basis(graph_dual)) != 1 else None

    def check_num_cycle(self) -> list[tuple[int, float, int]]:
        "guarantees: [cycle == 1] --> [crack from horizontal cut, not some island]"
        if self.model == "resistor_capacitor_battery": # rcb already passes this
            return []
        
        results = []
        for length in np.unique([l for (l, _) in list(self.config_to_num_seed)]).astype(np.int32):
            global length_global, idxs_edge_to_edges_dual_global
            length_global = int(length)
            idxs_edge_to_edges_dual_global = np.array(Array(length).idxs_edge_to_edges_dual, dtype=np.int32)

            for width in [w for (l, w) in list(self.config_to_num_seed) if l == length]:
                global width_global, path_length_width_global
                width_global = float(width)
                path_length_width_global = self.path_base / str(length_global) / str(width_global)

                with ProcessPoolExecutor(max_workers=18) as executor:
                    for result in executor.map(Data._num_cycle_parallel_job, self.config_to_seeds[(int(length), width)]): 
                        if result:
                            results.append(result)
                            # print(*result)
        return results
