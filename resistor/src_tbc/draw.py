import gc
import os
from pathlib import Path
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from src.array import Array
from src.matrix import Matrix
from src.equation import Equation
from src.failure import Failure


class Draw:
    def __init__(self, array: Array, matrix: Matrix, equation: Equation, failure: Failure):
        self.array = array
        self.matrix = matrix
        self.equation = equation
        self.failure = failure
        self.dir_data = Path(__file__).resolve().parents[1] / "data" / f"{self.array.length}" / f"{self.failure.width}" / f"{self.failure.seed}"

    def _save(self, file_name, verbose: bool = True):
        if self.save and not self.exists:
            os.makedirs(self.dir_data, exist_ok=True)
            plt.savefig(self.dir_data / f"{file_name}.png", bbox_inches="tight", pad_inches=self.pad_inches, transparent=self.transparent)
            print(f"saved: '{file_name}.png'") if verbose else None

    def _skip(self, file_name):
        if self.save + self.show == 0:
            return True

        self.exists = False
        if self.save and os.path.exists(self.dir_data / f"{file_name}.png"):
            print(f"already exists: '{file_name}.png'")
            self.exists = True
        
        if not self.show and self.exists:
            return True

    def graph_initialize(
        self, figsize: tuple = (5, 5), dpi: int = 72 * 2, size_edge: float = 2, pad_inches: float = -0.2, transparent: bool = False,
        save: bool = False, show: bool = False,
    ):
        self.figsize, self.dpi, self.size_edge, self.pad_inches, self.transparent = figsize, dpi, size_edge, pad_inches, transparent
        self.save, self.show = save, show
        if self.save + self.show == 0:
            return None

        length = self.array.length
        num_node = self.array.num_node
        self.idxs_edge_pbc = (length + 1) + (2 * length) * np.arange(length - 1, dtype=np.int32)
        self.num_edge_broken = len(self.failure.idxs_edge_broken)

        # graph generation (PBC cylinder --> 2D rectangle)
        self.graph = nx.Graph()
        self.graph.add_nodes_from([*range((-length + 1), num_node)])
        edges_pseudo = np.array(self.array.edges, dtype=np.int32)
        edges_pseudo[self.idxs_edge_pbc] = np.vstack([np.arange(-1, -length, -1, dtype=np.int32), edges_pseudo[self.idxs_edge_pbc][:, 1]]).T
        self.graph.add_edges_from(edges_pseudo.tolist())
        self.edges_pseudo = edges_pseudo
        del edges_pseudo; gc.collect()

        # node position generation
        pos_pseudo = {idx_node: (idx_node % length, length - idx_node // length) for idx_node in range(num_node)}
        self.pos = {
            idx_node: tuple(pos_node.tolist())
            for (idx_node, pos_node) in 
            zip(
                [*range(-length + 1, 0)],
                np.vstack([np.full((length - 1), length), np.arange(1, length)]).T,
            )} | pos_pseudo
        del pos_pseudo; gc.collect()

        # base color generation (in the same orders as node & edge list in draw.graph)
        self.colors_node = np.zeros((num_node + length - 1, 3), dtype=np.int8)
        self.colors_node[[*range(2 * length - 1)] + [*range(num_node - 1, num_node + length - 1)]] = np.ones(3, dtype=np.int8)
        self.colors_edge = np.zeros((self.array.num_edge, 3), dtype=np.int8)

    def graph_specific(self, idxs_edge_broken: list, num_edge_broken_colored: int = 0, with_labels: bool = False):
        if self._skip(f"{len(idxs_edge_broken)}_{num_edge_broken_colored}"):
            return None
        
        # edge coloring
        colors_edge = self.colors_edge.astype(np.float16)
        colors_edge[idxs_edge_broken] = np.ones(3, dtype=np.int8)
        if num_edge_broken_colored > 0:
            if num_edge_broken_colored == 1:
                colors_edge_broken = np.array([0, 0, 1], dtype=np.float16)
            else:
                colors_edge_broken = plt.colormaps["jet_r"](np.linspace(0.1, 0.9, num_edge_broken_colored))[:, :3]
            colors_edge[idxs_edge_broken[-num_edge_broken_colored:]] = colors_edge_broken
        colors_edge = np.vstack([colors_edge[self.idxs_edge_pbc][::-1], np.delete(colors_edge, self.idxs_edge_pbc, axis=0)])

        # plotting
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        nx.draw(
            self.graph, pos=self.pos, ax=ax,
            node_shape="s", node_color=self.colors_node, edgecolors="none", edge_color=colors_edge, 
            node_size=self.size_edge ** 2, width=self.size_edge,
            with_labels=with_labels, font_color="blue", font_size=10,
        )
        plt.tight_layout()

        self._save(f"{len(idxs_edge_broken)}_{num_edge_broken_colored}", verbose=True)
        plt.show() if self.show else plt.close()
        self.exists = False

    def graph_breakdown_images(self):
        if not self.save:
            if self.show:
                save_temp = self.save
                self.save = False
                self.graph_specific(self.failure.idxs_edge_broken, 0)
                self.save = save_temp
            return None
        
        self.dir_data = self.dir_data / "breakdown"
        os.makedirs(self.dir_data, exist_ok=True)

        if all([os.path.exists(self.dir_data / f"{count}.png") for count in range(self.num_edge_broken + 1)]):
            print(f"already exists: '0.png' ~ '{self.num_edge_broken}.png'")
            self.dir_data = self.dir_data.parent
            if self.show:
                save_temp = self.save
                self.save = False
                self.graph_specific(self.failure.idxs_edge_broken, 0)
                self.save = save_temp
            return None

        length = self.array.length
        length_minus_one = length - 1
        size_edge = self.size_edge
        size_edge_squared = self.size_edge ** 2
        edges = self.array.edges
        pos = self.pos
        colors_node = self.colors_node
        idxs_edge_pbc = self.idxs_edge_pbc
        count = 0

        # initial network
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        nx.draw(
            self.graph, pos=pos, ax=ax,
            node_shape="s", node_color=colors_node, edgecolors="none", edge_color="black", 
            node_size=size_edge_squared, width=size_edge,
        )

        plt.tight_layout()
        self._save(count, False)
        count += 1

        # non-initial network
        for idx_edge_broken in self.failure.idxs_edge_broken:
            idx_node1, idx_node2 = edges[idx_edge_broken]
            if idx_edge_broken in idxs_edge_pbc:
                idx_node1 = -int(idx_node1 / length)

            (x1, y1), (x2, y2) = pos[idx_node1], pos[idx_node2]
            ax.scatter(x1, y1, facecolors=colors_node[idx_node1 + length_minus_one], edgecolors="none", marker="s", zorder=3, s=size_edge_squared)
            ax.scatter(x2, y2, facecolors=colors_node[idx_node2 + length_minus_one], edgecolors="none", marker="s", zorder=3, s=size_edge_squared)
            ax.plot([x1, x2], [y1, y2], color="white", lw=size_edge)

            plt.tight_layout()
            self._save(count, False)
            count += 1

        print(f"saved: '0.png' ~ '{self.num_edge_broken}.png'")
        self.dir_data = self.dir_data.parent

    def graph_breakdown_video(self, delete_images: bool = True, fps: int = 30, frames_per_graph: int = 5, frames_last_graph: int = 30):
        if not self.save:
            return None

        path_output = self.dir_data / "breakdown.mp4"
        if os.path.exists(path_output):
            print(f"already exists: 'breakdown.mp4'")
            return None

        dir_breakdown = self.dir_data / "breakdown"
        images = [dir_breakdown / f"{count}.png" for count in range(self.num_edge_broken + 1)]

        frame_sample = cv2.imread(str(images[0]))
        orig_height, orig_width = frame_sample.shape[:2]

        target_width, target_height = 1920, 1080
        offset_x = (target_width - orig_width) // 2
        offset_y = (target_height - orig_height) // 2

        if offset_x < 0 or offset_y < 0:
            raise ValueError(f"original size: {orig_width}x{orig_height}, target size: 1920x1080")

        video = cv2.VideoWriter(
            str(path_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps=fps,
            frameSize=(target_width, target_height)
        )

        def write_centered(image_path, repeat):
            frame = cv2.imread(str(image_path))
            canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
            canvas[offset_y:offset_y + orig_height, offset_x:offset_x + orig_width] = frame
            for _ in range(repeat):
                video.write(canvas)

        for image in images:
            write_centered(image, frames_per_graph)

        write_centered(images[-1], frames_last_graph - frames_per_graph)

        video.release()
        print("saved: 'breakdown.mp4'")

        if delete_images:
            for image in images:
                os.remove(image)
            os.rmdir(dir_breakdown)
            print(f"deleted: '0.png' ~ '{self.num_edge_broken}.png'")

    # def graph_breakdown_video(self, delete_images: bool = True, fps:int = 30, frames_per_graph: int = 5, frames_last_graph: int = 30):
    #     if not self.save:
    #         return None
        
    #     if os.path.exists(self.dir_data / "breakdown.mp4"):
    #         print(f"already exists: 'breakdown.mp4'")
    #         return None
        
    #     dir_breakdown = self.dir_data / "breakdown"
    #     images = [dir_breakdown / f"{count}.png" for count in range(self.num_edge_broken + 1)]
    #     vid_height, vid_width, _ = cv2.imread(images[0]).shape
    #     video = cv2.VideoWriter(self.dir_data / "breakdown.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps=fps, frameSize=(vid_width, vid_height))

    #     for image in images:
    #         for _ in range(frames_per_graph):
    #             video.write(cv2.imread(image))
    #     for _ in range(frames_last_graph - frames_per_graph):
    #         video.write(cv2.imread(image))

    #     video.release()
    #     print("saved: 'breakdown.mp4'")
    #     if delete_images:
    #         for image in images:
    #             os.remove(image)
    #         os.rmdir(dir_breakdown)
    #         print(f"deleted: '0.png' ~ '{self.num_edge_broken}.png'")


    def plot_initialize(
        self,
    ):
        pass