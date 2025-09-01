import os, re, struct, math
from pathlib import Path
import numpy as np
import networkx as nx
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from src.array import Array
from src.failure import Failure


# from src.utils import check_validity_seed; check_validity_seed(array, failure)

# from src.utils import get_configs_seeds, check_volts_ext_seeds, check_connectivity_seeds, connectivity_parallel_job
# dict_length_width_to_seeds, dict_length_width_to_num_seed = get_configs_seeds()
# record_volts_ext = check_volts_ext_seeds(dict_length_width_to_seeds)
# record_connectivity = check_connectivity_seeds(dict_length_width_to_seeds, dict_length_width_to_num_seed)


def check_validity_seed(array: Array, failure: Failure):
    """
    [single seed] quick check for external volt and connectivity check
    """
    if np.isnan(failure.volts_ext).sum() + np.isinf(failure.volts_ext).sum() > 0:
        print("there is nan/inf values in external voltages")

    graph = nx.Graph()
    edges_copy = array.edges.copy()
    length = array.length

    # top & bottom edges loop
    edges_copy.extend([(idx_node, idx_node + 1) for idx_node in range(length - 1)] + [(0, length - 1)])
    edges_copy.extend([(idx_node, idx_node + 1) for idx_node in range(length ** 2, length ** 2 + length - 1)] + [(length ** 2, length ** 2 + length - 1)])

    graph.add_edges_from(edges_copy)
    graph.remove_edges_from(np.array(edges_copy)[failure.idxs_edge_broken])

    num_cc = nx.number_connected_components(graph)
    print("number of connected components: ", num_cc) if num_cc != 2 else None

    
def get_configs_seeds() -> tuple[dict, dict]:
    """
    [all seeds in data] get dicts of (length, width) -> seeds AND num_seed
    """
    path_data = Path("/disk/disk3/gyeongmin/project/random_fuse_network/resistor_capacitor/data")
    is_float = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$').match # gpt voodoo

    tmp = defaultdict(list)

    with os.scandir(path_data) as it_length:
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

                    path_params = os.path.join(e_width.path, "1.0_0.01")
                    if not os.path.isdir(path_params):
                        continue

                    seeds = []
                    with os.scandir(path_params) as it_seed:
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
    dict_length_width_to_seeds = {k: tmp[k] for k in keys}
    dict_length_width_to_num_seed = {k: tmp[k].size for k in keys}
    return dict_length_width_to_seeds, dict_length_width_to_num_seed


def check_volts_ext_seeds(dict_length_width_to_seeds) -> list[tuple[int, float, int]]:
    """
    [all seeds in data] rare case: 0 edge volt being broken because of being the most stressed bond (due to pure chance: not observed for MT19937 with L >= 10)
    """
    path_data = Path(__file__).resolve().parents[1] / "data"

    def last_is_nan_or_inf_tail(path_npy: Path) -> bool:
        with open(path_npy, 'rb', buffering=0) as f:
            f.seek(-8, os.SEEK_END)
            v = struct.unpack('<d', f.read(8))[0] # gpt voodoo
        return not math.isfinite(v)

    def check_one(tup):
        length, width, seed = tup
        path_npy = path_data / str(length) / str(width) / "1.0_0.01" /  str(seed) / "volts_ext.npy"
        return (length, width, seed) if last_is_nan_or_inf_tail(path_npy) else None

    tasks = ((length, width, int(seed))
            for (length, width), seeds in dict_length_width_to_seeds.items()
            for seed in seeds)
    
    record = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for res in ex.map(check_one, tasks, chunksize=1024):
            if res:
                record.append([res])
                print(*res)
    return record


def connectivity_parallel_job(seed):
    graph = graph_init.copy()
    idxs_edge_broken = np.load(path_length_width / str(seed) / "idxs_edge_broken.npy")
    graph.remove_edges_from(array.edges[idxs_edge_broken])
    num_cc = nx.number_connected_components(graph)
    return (length_global, width_global, seed, num_cc) if num_cc != 2 else None


def check_connectivity_seeds(dict_length_width_to_seeds) -> list[tuple[int, float, int, int]]:
    """
    [all seeds in data] identify simulations with primal graph connected components greater than 2 
    """
    lengths_unique = np.unique(np.array(list(dict_length_width_to_seeds))[:, 0]).astype(np.int32)
    path_data = Path(__file__).resolve().parents[1] / "data"
    record = []

    for length in lengths_unique:
        global array, graph_init, length_global
        array = Array(length)
        graph_init = nx.Graph()
        length_global = length

        # top & bottom edges loop
        array.edges.extend([(idx_node, idx_node + 1) for idx_node in range(length - 1)] + [(0, length - 1)])
        array.edges.extend([(idx_node, idx_node + 1) for idx_node in range(length ** 2, length ** 2 + length - 1)] + [(length ** 2, length ** 2 + length - 1)])

        graph_init.add_edges_from(array.edges)
        array.edges = np.array(array.edges, dtype=np.int32) # array edge modification

        for width in [w for (l, w) in list(dict_length_width_to_seeds) if l == length]:
            global path_length_width, width_global
            path_length_width = path_data / str(length) / str(width) / "1.0_0.01"
            width_global = width

            seeds = dict_length_width_to_seeds[(length, width)]
            with ProcessPoolExecutor(max_workers=20) as executor:
                for result in list(executor.map(connectivity_parallel_job, seeds)): 
                    if result:
                        record.append(result)
                        print(result)     
    return record