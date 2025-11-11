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
    def __init__(self, model: str):
        self.model = model
        
        self.path_base = Path(__file__).resolve().parents[2]/ model / "data"
        
        self.config_to_seeds, self.config_to_num_seed = self._get_configs()
        self.num_datum: int = sum(list(self.config_to_num_seed.values()))

        self.val_caps_unique: np.ndarray[np.float64] = np.unique([c for (c, _, _) in list(self.config_to_num_seed)]).astype(np.float64)
        self.lengths_unique: np.ndarray[np.int32] = np.unique([l for (_, l, _) in list(self.config_to_num_seed)]).astype(np.int32)
        self.widths_unique: np.ndarray[np.float64] = np.unique([w for (_, _, w) in list(self.config_to_num_seed)]).astype(np.float64)
        self.idxs_val_cap: np.ndarray[np.int32] = np.arange(self.val_caps_unique.shape[0], dtype=np.int32)
        self.idxs_length: np.ndarray[np.int32] = np.arange(self.lengths_unique.shape[0], dtype=np.int32)
        self.idxs_width: np.ndarray[np.int32] = np.arange(self.widths_unique.shape[0], dtype=np.int32)

        # dict[val_cap, length, width]
        self.dict_vb: dict[tuple[float, int, float], float] = None
        self.dict_nb: dict[tuple[float, int, float], float] = None
        self.dict_eb: dict[tuple[float, int, float], float] = None
        # total[idxs_val_cap, idxs_length, idxs_width]
        self.total_vb: np.ndarray[np.float64] = None
        self.total_nb: np.ndarray[np.float64] = None
        self.total_eb: np.ndarray[np.float64] = None 

    def _get_configs(self):
        is_float = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$').match
        tmp = defaultdict(list)

        with os.scandir(self.path_base) as it_cap:
            for e_cap in it_cap:
                if not (e_cap.is_dir() and is_float(e_cap.name)):
                    continue
                val_cap = float(e_cap.name)

                with os.scandir(e_cap.path) as it_length:
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

                                        # require at least one file in seed dir
                                        with os.scandir(e_seed.path) as it_files:
                                            if next(it_files, None) is not None:
                                                seeds.append(seed)

                                if seeds:
                                    tmp[(val_cap, length, width)] = np.array(sorted(seeds), dtype=np.int32)

        keys = sorted(tmp.keys(), key=lambda k: (k[0], k[1], k[2]))
        config_to_seeds: dict[tuple[float, int, float], np.ndarray] = {k: tmp[k] for k in keys}
        config_to_num_seed: dict[tuple[float, int, float], int] = {k: tmp[k].size for k in keys}
        return config_to_seeds, config_to_num_seed
        
    def check_volts_ext(self) -> list[tuple[float, int, float, int]]:
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
            val_cap, length, width, seed = task
            path_npy = path_base / str(val_cap) / str(length) / str(width) / str(seed) / "volts_ext.npy"
            return (val_cap, length, width, seed) if is_empty_or_last_nan_inf(path_npy) else None

        tasks = (
            (val_cap, length, width, seed)
            for (val_cap, length, width), seeds in self.config_to_seeds.items()
            for seed in seeds
        )

        results = []
        with ThreadPoolExecutor(max_workers=19) as executor:
            for result in executor.map(check_one, tasks):
                if result:
                    results.append(result)
                    # print(*result)
        return results

    @staticmethod    
    def _parallel_job_check_connectivity(seed):
        graph = graph_init_global.copy()
        graph.remove_edges_from(edges_global[np.load(path_val_cap_length_width_global/ str(seed) / "idxs_edge_broken.npy")])
        return (val_cap_global, length_global, width_global, seed) if nx.number_connected_components(graph) != 2 else None
    
    def check_connectivity(self) -> list[tuple[float, int, float, int]]:
        "depreciated (just use check num cycle); does not guarantee: [connectivity == 2] --> [crack from horizontal cut, not some island]; must be empty as new check_graph guarantees it"
        if self.model == "resistor_capacitor_battery": # rcb already passes this
            return []
        
        results = []
        for length in np.array(self.lengths_unique, dtype=np.int32):
            global edges_global, graph_init_global, length_global, val_cap_global
            length_global = int(length)
            edges_global = Array(length).edges

            # top & bottom edges loop
            edges_global.extend([(idx_node, idx_node + 1) for idx_node in range(length - 1)] + [(0, length - 1)])
            edges_global.extend([(idx_node, idx_node + 1) for idx_node in range(length ** 2, length ** 2 + length - 1)] + [(length ** 2, length ** 2 + length - 1)])
            edges_global = np.array(edges_global, dtype=np.int32)
            graph_init_global = nx.Graph()
            graph_init_global.add_edges_from(edges_global)

            for val_cap in np.unique([c for (c, l, _) in list(self.config_to_num_seed) if l == length]).tolist():
                global val_cap_global
                val_cap_global = float(val_cap)

                for width in [w for (c, l, w) in list(self.config_to_num_seed) if (l == length_global and c == val_cap_global)]:
                    global width_global, path_val_cap_length_width_global
                    width_global = float(width)
                    path_val_cap_length_width_global = self.path_base / str(val_cap_global) / str(length_global) / str(width_global)

                    with ProcessPoolExecutor(max_workers=19) as executor:
                        for result in executor.map(
                            Data._parallel_job_check_connectivity, self.config_to_seeds[(val_cap_global, length_global, width_global)]): 
                            if result:
                                results.append(result)
                                # print(*result)
        return results

    @staticmethod
    def _parallel_job_check_num_cycle(seed):
        graph_dual = nx.Graph()
        graph_dual.add_edges_from(idxs_edge_to_edges_dual_global[np.load(path_val_cap_length_width_global / str(seed) / "idxs_edge_broken.npy")])
        return (val_cap_global, length_global, width_global, seed) if len(nx.cycle_basis(graph_dual)) != 1 else None

    def check_num_cycle(self) -> list[tuple[float, int, float, int]]:
        "guarantees: [cycle == 1] --> [crack from horizontal cut, not some island]; must be empty as new check_graph guarantees it"
        if self.model == "resistor_capacitor_battery":
            return []

        results = []
        for length in self.lengths_unique.tolist():
            global length_global, idxs_edge_to_edges_dual_global
            length_global = int(length)
            idxs_edge_to_edges_dual_global = np.array(Array(length=length, mode_analysis=True).idxs_edge_to_edges_dual, dtype=np.int32)

            for val_cap in np.unique([c for (c, l, _) in list(self.config_to_num_seed) if l == length]).tolist():
                global val_cap_global
                val_cap_global = float(val_cap)

                for width in [w for (c, l, w) in list(self.config_to_num_seed) if (l == length_global and c == val_cap_global)]:
                    global width_global, path_val_cap_length_width_global
                    width_global = float(width)
                    path_val_cap_length_width_global = self.path_base / str(val_cap_global) / str(length_global) / str(width_global)

                    with ProcessPoolExecutor(max_workers=19) as executor:
                        for result in executor.map(
                            Data._parallel_job_check_num_cycle, self.config_to_seeds[(val_cap_global, length_global, width_global)]
                        ):
                            if result:
                                results.append(result)
                                # print(*result)
        return results

    @staticmethod
    def _parallel_job_compute_macro_r(seed):
        return (
            float(np.load(path_val_cap_length_width_global / str(seed) / "volts_ext.npy", mmap_mode="r")[-1]),
            int(np.load(path_val_cap_length_width_global / str(seed) / "idxs_edge_broken.npy", mmap_mode="r").shape[0]),
            0.0,
        )

    @staticmethod
    def _parallel_job_compute_macro_rc(seed):
        idxs_edge_broken = np.load(path_val_cap_length_width_global / str(seed) / "idxs_edge_broken.npy", mmap_mode="r")
        volts_cap_last = np.load(path_val_cap_length_width_global / str(seed) / "volts_cap_last.npy")
        volts_cap_last[idxs_edge_broken] = 0.0
        volts_cap_last[np.load(path_val_cap_length_width_global / str(seed) / "idxs_edge_unalive.npy", mmap_mode="r")] = 0.0
        return (
            float(np.load(path_val_cap_length_width_global / str(seed) / "volts_ext.npy", mmap_mode="r")[-1]),
            int(idxs_edge_broken.shape[0]),
            float(np.dot(volts_cap_last, volts_cap_last)),
        )

    @staticmethod
    def _parallel_job_compute_macro_rcb(seed):
        idxs_edge_broken = np.load(path_val_cap_length_width_global / str(seed) / "idxs_edge_broken.npy", mmap_mode="r")
        volts_cap_last = np.load(path_val_cap_length_width_global / str(seed) / "volts_cap_last.npy")
        volts_cap_last[idxs_edge_broken] = 0.0
        return (
            float(np.load(path_val_cap_length_width_global / str(seed) / "volts_ext.npy", mmap_mode="r")[-1]),
            int(idxs_edge_broken.shape[0]),
            float(np.dot(volts_cap_last, volts_cap_last)),
        )

    def compute_macro(self):
        """
        ASSUME SYMMETRY IN DATA STRUCTURE (val_cap, length, width)
        self.dict_vb: (final external voltage) / L - v _ \n
        self.dict_nb: (# broken edges) / L - 1 \n 
        self.dict_eb: c / 2 * Σ_("alive" capacitor) (v_cap ** 2)
        """
        if self.model == "resistor":
            _job = Data._parallel_job_compute_macro_r
        elif self.model == "resistor_capacitor":
            _job = Data._parallel_job_compute_macro_rc
        else:
            _job = Data._parallel_job_compute_macro_rcb

        dict_vb, dict_nb, dict_eb = {}, {}, {}
        for val_cap in self.val_caps_unique.tolist():
            for length in self.lengths_unique.tolist():
                for width in self.widths_unique.tolist():
                    global path_val_cap_length_width_global
                    path_val_cap_length_width_global = self.path_base / str(val_cap) / str(length) / str(width)
                    with ThreadPoolExecutor(max_workers=19) as executor:
                        results = np.array(list(executor.map(
                            _job, self.config_to_seeds[val_cap, length, width]
                        )))
                    dict_vb[val_cap, length, width] = float(results[:, 0].mean() / length - (1 - width / 2))
                    dict_nb[val_cap, length, width] = float(results[:, 1].mean() / length - 1)
                    dict_eb[val_cap, length, width] = float(results[:, 2].mean() * val_cap / 2) # may require normalization w.r.t. L or w

        self.dict_vb, self.dict_nb, self.dict_eb = dict_vb, dict_nb, dict_eb
        self.total_vb = np.array(list(self.dict_vb.values())).reshape(self.idxs_val_cap.shape[0], self.idxs_length.shape[0], self.idxs_width.shape[0])
        self.total_nb = np.array(list(self.dict_nb.values())).reshape(self.idxs_val_cap.shape[0], self.idxs_length.shape[0], self.idxs_width.shape[0])
        self.total_eb = np.array(list(self.dict_eb.values())).reshape(self.idxs_val_cap.shape[0], self.idxs_length.shape[0], self.idxs_width.shape[0])