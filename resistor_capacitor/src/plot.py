import os
import gc
import cv2
import numpy as np
import networkx as nx
from scipy.sparse import coo_array
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.edge_list import EdgeList
from src.matrix import Matrix
from src.rand_list import RandList
from src.failure import Failure


class Plot:
    def __init__(self, edgeList: EdgeList, matrix: Matrix, randList: RandList, failure: Failure):
        self.edgeList = edgeList
        self.matrix = matrix
        self.randList = randList
        self.failure = failure
        self.save = None
        self.show = None
        self.pretty = None
        self.filePath1 = None
        self.filePath2 = None
        self.sizeFigGraph = None
        self.sizeNodeGraph = None
        self.sizeEdgeGraph = None
        self.idxEdgesPseudo = None
        self.graph = None
        self.pos = None
        self.colorWhiteInt = np.ones(3, dtype=np.int8)
        self.colorWhiteFloat = np.ones(3, dtype=np.float16)
        self.colorNodes = None
        self.colorEdges = None
        self.count = 0
        self.sizeFig = None
        self.sizeFont = None
        self.sizeLabel = None
        self.numBrokenEdges = 0
        self.addends = None
        self.multipliers = None
        self.idxHighlightedEdges = None
        self.xticks = None
        self._SetPath()
        if type(failure.idxBrokenEdges) == np.ndarray:
            failure.idxBrokenEdges = failure.idxBrokenEdges.tolist()
        if type(failure.extVolts) == np.ndarray:
            failure.extVolts = failure.extVolts.tolist()


    def _SetPath(self):
        self.filePath1 = f"{os.getcwd()}/data/{self.edgeList.length}/"
        self.filePath2 = self.filePath1 + f"{self.randList.width}/{self.randList.seed}/"
        os.makedirs(self.filePath1, exist_ok=True)
        os.makedirs(self.filePath2, exist_ok=True)


    def _Save(self, filePath, fileName, padInches, dpi: int = 300, transparent: bool = False, verbose: bool = True):
        if self.save:
            padInches = 0.1 if padInches is None else padInches
            plt.savefig(f"{filePath}{fileName}.png", bbox_inches="tight", pad_inches=padInches, dpi=dpi, transparent=transparent)
            if verbose:
                print(f"Saved: '{fileName}.png'")


    def _Skip(self, filePath, fileName):
        if os.path.exists(f"{filePath}{fileName}.png"):
            print(f"Already exists: '{fileName}.png'")
            return True
        
        if self.save + self.show == 0:
            return True
    

    def InitializeGraph(self, sizeFigGraph: tuple = (20, 20), sizeNodeGraph: float = 2, sizeEdgeGraph: float = 2, save: bool = False, show: bool = False):
        length, numNode = self.edgeList.length, self.edgeList.numNode
        self.save, self.show = save, show
        if self.save + self.show == 0:
            return None

        self.sizeFigGraph, self.sizeNodeGraph, self.sizeEdgeGraph = sizeFigGraph, sizeNodeGraph, sizeEdgeGraph
        self.idxEdgesPseudo = (length + 1) + (2 * length) * np.arange(length - 1)
        self.numBrokenEdges = len(self.failure.idxBrokenEdges)

        self.graph = nx.Graph()
        self.graph.add_nodes_from([*range((-length + 1), numNode)])
        edgesPseudo = np.array(self.edgeList.edges)
        edgesPseudo[self.idxEdgesPseudo] = np.vstack([np.arange(-1, -length, -1), edgesPseudo[self.idxEdgesPseudo][:, 1]]).T
        self.graph.add_edges_from(edgesPseudo.tolist())
        del edgesPseudo; gc.collect()

        pos = {idxNode: (idxNode % length, idxNode // length) for idxNode in range(numNode)}
        self.pos = {idxNode: tuple(posNode.tolist()) for (idxNode, posNode) in zip([*range(-length + 1, 0)], np.vstack([np.full((length - 1), length), np.arange(length - 1, 0, -1)]).T)} | pos
        del pos; gc.collect()

        self.colorNodes = np.zeros((numNode + length - 1, 3), dtype=np.int8)
        self.colorNodes[[*range(2 * length - 1)] + [*range(numNode - 1, numNode + length -1)]] = self.colorWhiteInt
        self.colorEdges = np.zeros((self.edgeList.numEdge, 3), dtype=np.int8)


    def PlotSequentialGraphs(self):
        if os.path.exists(f"{self.filePath2}Breakdown.mp4"):
            print(f"Already exists: 'Breakdown.mp4'")
            choice = input(f"Save/Show '0.png' ~ '{self.numBrokenEdges}.png' (y/ANY) ?: "); print()
            if choice != "y":
                return None

        if self.save + self.show == 0:
            return None

        if self.show:
            choice = input(f"Reconfirm - Show '0.png' ~ '{self.numBrokenEdges}.png' (y/ANY) ?: "); print()
            show = True if choice == "y" else False
        else:
            show = False

        fig, ax = plt.subplots(1, 1, figsize=self.sizeFigGraph)
        nx.draw(self.graph, pos=self.pos, ax=ax, node_size=self.sizeNodeGraph, width=self.sizeEdgeGraph, node_color=self.colorNodes, edge_color="#000000", with_labels=False)
        self._Save(self.filePath2, f"{self.count}", -0.65, verbose=False)
        plt.show(fig) if show else plt.close(fig)
        self.count += 1

        selfColorEdges = self.colorEdges
        selfColorWhiteInt = self.colorWhiteInt
        selfIdxEdgesPseudo = self.idxEdgesPseudo

        for idxBrokenEdge in self.failure.idxBrokenEdges:
            selfColorEdges[idxBrokenEdge] = selfColorWhiteInt
            colorEdges = np.vstack([selfColorEdges[selfIdxEdgesPseudo][::-1], np.delete(selfColorEdges, selfIdxEdgesPseudo, axis=0)])
            
            fig, ax = plt.subplots(1, 1, figsize=self.sizeFigGraph)
            nx.draw(self.graph, pos=self.pos, ax=ax, node_size=self.sizeNodeGraph, width=self.sizeEdgeGraph, node_color=self.colorNodes, edge_color=colorEdges, with_labels=False)
            self._Save(self.filePath2, f"{self.count}", -0.65, verbose=False)
            plt.show(fig) if show else plt.close(fig)
            self.count += 1
        self.count = 0

        if self.save:
            print(f"Saved: '0.png' ~ '{self.numBrokenEdges}.png'")


    def PlotSpecificGraph(self, idxBrokenEdges: list, numColorBrokenEdges: int = 0):
        if self._Skip(self.filePath2, f"{len(idxBrokenEdges)}"):
            return None
        
        colorEdges = np.zeros((self.edgeList.numEdge, 3), dtype=np.float16)
        colorEdges[idxBrokenEdges] = self.colorWhiteFloat
        if numColorBrokenEdges > 0:
            if numColorBrokenEdges == 1:
                colorBrokenEdges = np.array([0, 0, 1], dtype=np.float16)
            else:
                colorBrokenEdges = plt.colormaps["jet_r"](np.linspace(0.1, 0.9, numColorBrokenEdges))[:, :3]
            colorEdges[idxBrokenEdges[-numColorBrokenEdges:]] = colorBrokenEdges
        colorEdges = np.vstack([colorEdges[self.idxEdgesPseudo][::-1], np.delete(colorEdges, self.idxEdgesPseudo, axis=0)])
        
        fig, ax = plt.subplots(1, 1, figsize=self.sizeFigGraph)
        nx.draw(self.graph, pos=self.pos, ax=ax, node_size=self.sizeNodeGraph, width=self.sizeEdgeGraph, node_color=self.colorNodes, edge_color=colorEdges, with_labels=False)
        
        self._Save(self.filePath2, f"{len(idxBrokenEdges)}", -0.65)
        plt.show(fig) if self.show else plt.close(fig)


    def CreateVideoDeleteImages(self, fpsPerGraph: int = 5):
        if self.save:
            if os.path.exists(f"{self.filePath2}Breakdown.mp4"):
                print(f"Already exists: 'Breakdown.mp4'")
                return None

            images = [os.path.join(self.filePath2, f"{i}.png") for i in range(self.numBrokenEdges)]
            vidHeight, vidWidth, _ = cv2.imread(images[0]).shape
            video = cv2.VideoWriter(f"{self.filePath2}Breakdown.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (vidWidth, vidHeight))

            for image in images:
                img = cv2.imread(image)
                for _ in range(fpsPerGraph):
                    video.write(img)
                os.remove(image)
            for _ in range(30):
                video.write(img)

            video.release()
            cv2.destroyAllWindows()
            print("Saved: 'Breakdown.mp4'")
            print(f"Deleted: '0.png' ~ '{self.numBrokenEdges}.png'")


    def InitializePlot(self, ptpDivider: float = 4, xTicksStep: int = 10, xTickLastShow: bool = False, sizeFig: tuple = (10, 10), sizeFont: float = 20, sizeLabel: float = 15, save: bool = False, show: bool = False, pretty: bool = False):
        self.sizeFig, self.sizeFont, self.sizeLabel = sizeFig, sizeFont, sizeLabel
        self.save, self.show, self.pretty = save, show, pretty
        self.numBrokenEdges = len(self.failure.idxBrokenEdges)
        if self.save + self.show == 0:
            return None
    
        if self.numBrokenEdges != 0:
            self.addends = np.array(self.failure.extVolts) - np.array([self.failure.extVolts[0]] + self.failure.extVolts[:-1])
            thrAddend = np.min(self.addends) + np.ptp(self.addends) / ptpDivider

            extVoltsNpArray = np.array(self.failure.extVolts)
            idxZeroExtVolts = np.where(extVoltsNpArray == 0)[0]
            extVoltsDivider = np.array([self.failure.extVolts[0]] + self.failure.extVolts[:-1])
            if len(idxZeroExtVolts) != 0:
                extVoltsDivider[np.append(idxZeroExtVolts, (idxZeroExtVolts[-1] + 1))] = 1
                self.multipliers = extVoltsNpArray / extVoltsDivider
                self.multipliers[np.append(idxZeroExtVolts, (idxZeroExtVolts[-1] + 1))] = 1
            else:
                self.multipliers = extVoltsNpArray / extVoltsDivider
            thrMultiplier = np.min(self.multipliers) + np.ptp(self.multipliers) / ptpDivider

            self.idxHighlightedEdges = np.union1d(np.where(self.multipliers > thrMultiplier)[0], np.where(self.addends > thrAddend)[0]).tolist()
            if ptpDivider == 1:
                self.xticks = np.arange(self.numBrokenEdges+1, dtype=np.int32)[::xTicksStep]
                if xTickLastShow:
                    self.xticks = np.array([*set(np.append(self.xticks, self.numBrokenEdges))])
            else:
                self.xticks = self.idxHighlightedEdges + [self.numBrokenEdges] if 0 in self.idxHighlightedEdges else [0] + self.idxHighlightedEdges + [self.numBrokenEdges]


    def PlotRealizations(self):
        if self._Skip(self.filePath2, "Realizations"):
            return None

        fig, ax = plt.subplots(1, 1, figsize=self.sizeFig)
        hist, binEdges = np.histogram(self.randList.randThrVolts, density=False, bins=10, range=(self.randList.minThrVolt, self.randList.maxThrVolt))
        hist = hist / hist.sum() * 100
        ax.bar(binEdges[:-1], hist, width=np.diff(binEdges), align="edge", ec="black", color="#0000FF")

        if self.pretty:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        else:
            ax.set_title("Realizations of Threshold Voltage Drops", fontsize=self.sizeFont)
            ax.set_xlabel("Voltage Drop (V)", fontsize=self.sizeFont)
            ax.set_ylabel("Percentage of Edges (%)", fontsize=self.sizeFont)
            ax.tick_params(labelsize=self.sizeLabel)
            ax.set_xticks(np.arange(0, 2.1, 0.1))
        ax.set_xlim(0, 2)

        fig.tight_layout()
        self._Save(self.filePath2, "Realizations", None)
        plt.show(fig) if self.show else plt.close(fig)


    def PlotMatCond(self):
        if self._Skip(self.filePath1, "MatCond"):
            return None
        
        choice = input("Reconfirm - Save/Show 'MatCond.png' (y/ANY) ?: "); print()
        if choice != "y":
            return None

        matCondCOO = coo_array(self.matrix.matCond)
        cooData = np.column_stack((matCondCOO.row, matCondCOO.col, np.abs(matCondCOO.data)))
        cooZeros = cooData.copy()
        cooZeros[:, 2] = np.zeros(matCondCOO.nnz, dtype=np.int32)
        cooLines = np.array([(cooData[idxNnz], cooZeros[idxNnz]) for idxNnz in range(matCondCOO.nnz)])

        zmax = 4 if self.edgeList.length < 4 else self.edgeList.length
        colorLines = (matCondCOO.data + 1) / (zmax - 1)
        colorLines = plt.colormaps["viridis"](colorLines)[:, :3]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        for cooLine, colorLine in zip(cooLines, colorLines):
            ax.plot(*cooLine.T, color=colorLine)
        del matCondCOO, cooData, cooZeros, cooLines, colorLines; gc.collect()
        s = self.edgeList.sizeMatCond - 1
        for boundaryLine in np.array([[[0, 0, 0], [s, 0, 0]], [[s, 0, 0], [s, s, 0]], [[s, s, 0], [0, s, 0]], [[0, s, 0], [0, 0, 0]]], dtype=np.int32):
            ax.plot(*boundaryLine.T,  color="black")

        plt.colorbar(
            plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=-1, vmax=zmax)),
            cax=fig.add_axes([ax.get_position().x1 - 0.09, ax.get_position().y0 + 0.093, 0.01, ax.get_position().height - 0.44])
        )
        ax.set_xlim(0, s)
        ax.set_ylim(0, s)
        zmax = 4 if self.edgeList.length < 4 else self.edgeList.length
        ax.set_zlim(0, zmax * 5)
        ax.set_axis_off()
        ax.view_init(azim=0, elev=30)
        ax.set_proj_type("persp", focal_length=0.5)
        self._Save(self.filePath1, "MatCond", -0.5, transparent=self.pretty)
        plt.show(fig) if self.show else plt.close(fig)


    def PlotExtVolts(self, sizeScatter: float = 2):
        if self._Skip(self.filePath2, "ExtVolts"):
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=self.sizeFig)
        ax.scatter(0, self.failure.extVolts[0], s=(sizeScatter * 10), c="#0000FF")
        ax.scatter(np.arange(self.numBrokenEdges, dtype=np.int32), self.failure.extVolts, s=sizeScatter, c="#0000FF")
        ax.scatter(self.numBrokenEdges, self.failure.extVolts[-1], s=sizeScatter, c="#0000FF")

        ax.set_xlim(0, self.numBrokenEdges)
        ymin = 0 if ax.get_ylim()[0] < ((ax.get_yticks()[1] - ax.get_yticks()[0]) * 0.5) else ax.get_ylim()[0]
        ax.vlines(x=self.idxHighlightedEdges, ymin=ymin, ymax=np.array(self.failure.extVolts)[self.idxHighlightedEdges], colors="black", linewidth=0.5, linestyles="dotted", zorder=0)
        ax.set_ylim(ymin)
        if self.pretty:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        else:
            ax.set_title("External Voltage Drop against Time", fontsize=self.sizeFont)
            ax.set_xlabel("Time (s)", fontsize=self.sizeFont)
            ax.set_ylabel("External Voltage Drop (V)", fontsize=self.sizeFont)
            ax.tick_params(labelsize=self.sizeLabel)
            ax.set_xticks(self.xticks)

        fig.tight_layout()
        self._Save(self.filePath2, "ExtVolts", None)
        plt.show(fig) if self.show else plt.close(fig)


    def PlotAddendsMultipliers(self, sizeLineWidth: float = 1):
        if self._Skip(self.filePath2, "AddendsMultipliers"):
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.sizeFig)
        ax1.plot(np.append(self.addends, 0.0), label="Addends", color="#0000FF", linewidth=sizeLineWidth)
        ax2.plot(np.append(self.multipliers, 1.0), label="Multiplier", color="#FF0000", linewidth=sizeLineWidth)

        if self.pretty:
            ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        else:
            ax1.set_title("Addend and Multiplier of External Voltage Drop against Time", fontsize=self.sizeFont)
            ax2.set_xlabel("Time (s)", fontsize=self.sizeFont)
            ax1.set_ylabel("Addend", fontsize=self.sizeFont)
            ax2.set_ylabel("Multiplier", fontsize=self.sizeFont)
            ax1.set_xticks(self.xticks)
            ax1.xaxis.set_tick_params(labelbottom=False)
            ax2.set_xticks(self.xticks)
            ax2Yticks = ax2.get_yticks()
            if ax2Yticks[0] != 1:
                ax2Yticks = ax2Yticks[:-1]
                ax2Yticks[0] = 1
            ax2.set_yticks(ax2Yticks)
            ax1.tick_params(labelsize=self.sizeLabel)
            ax2.tick_params(labelsize=self.sizeLabel)

        ax1.set_xlim(0, self.numBrokenEdges)
        ax2.set_xlim(0, self.numBrokenEdges)
        ax1.set_ylim(0)
        ax2.set_ylim(1)
        ax1.vlines(x=self.idxHighlightedEdges, ymin=0, ymax=ax1.get_ylim()[1], colors="black", linewidth=0.5, linestyles="dotted", zorder=0)
        ax2.vlines(x=self.idxHighlightedEdges, ymin=1, ymax=ax2.get_ylim()[1], colors="black", linewidth=0.5, linestyles="dotted", zorder=0)

        fig.tight_layout()
        self._Save(self.filePath2, "AddendsMultipliers", None)
        plt.show(fig) if self.show else plt.close(fig)


    def PlotDataEdgeVolts(self, dataRawEdgeVoltsInput: np.ndarray, intervalOfInterest: tuple = None, sizeLineWidth: float = 0.25, rainbow: bool = False):
        if len(dataRawEdgeVoltsInput) == 0:
            print("Emtpy dataRawEdgeVoltsInput")
            return None

        if self._Skip(self.filePath2, "DataEdgeVolts"):
            return None
        
        intervalOfInterest = (0, self.numBrokenEdges) if intervalOfInterest is None else intervalOfInterest
        xmin, xmax = intervalOfInterest
        if (xmin < 0) or (xmax > self.numBrokenEdges) or (xmin >= xmax):
            raise ValueError("Invalid Interval Input")
        xmin, xmax = max(0, xmin - 1), min(xmax + 1, self.numBrokenEdges)
        xmaxForSlicing = self.numBrokenEdges if xmax >= self.numBrokenEdges else (xmax + 1)

        dataEdgeVolts = np.vstack([np.array(dataRawEdgeVoltsInput), np.zeros(self.edgeList.numEdge)])
        for idxTime, (multiplier, idxBrokenEdge) in enumerate(zip(self.multipliers, self.failure.idxBrokenEdges)):
            dataEdgeVolts[idxTime, :] *= multiplier
            if idxTime == 0:
                dataEdgeVolts[0, :] *= self.failure.extVolts[0]

            dataEdgeVolts[(idxTime + 1):, idxBrokenEdge] = None
        dataEdgeVolts = dataEdgeVolts.T

        zorderEdges = [1 if idxEdge in self.failure.idxBrokenEdges else 0 for idxEdge in range(self.edgeList.numEdge)]
        if rainbow: 
            tempConst = self.numBrokenEdges + 1
            for (idxTime, idxBrokenEdge) in enumerate(self.failure.idxBrokenEdges):
                zorderEdges[idxBrokenEdge] = tempConst - idxTime
            colorEdges = np.zeros((self.edgeList.numEdge, 3), dtype=np.float16)
            colorBrokenEdges = plt.colormaps["jet_r"](np.linspace(0, 1, xmaxForSlicing - xmin))[:, :3]
            for (idxBrokenEdge, colorBrokenEdge) in zip(self.failure.idxBrokenEdges[xmin:xmaxForSlicing], colorBrokenEdges):
                colorEdges[idxBrokenEdge] = colorBrokenEdge
            del colorBrokenEdges; gc.collect()
        else: 
            colorEdges = np.zeros((self.edgeList.numEdge, 3), dtype=np.int8)
            colorEdges[self.failure.idxBrokenEdges] = np.array([1, 0, 0], dtype=np.int8)

        fig, ax = plt.subplots(1, 1, figsize=self.sizeFig)
        for datumEdgeVolts, zorderEdge, colorEdge in zip(dataEdgeVolts, zorderEdges, colorEdges):
            ax.plot(datumEdgeVolts, zorder=zorderEdge, color=colorEdge, linewidth=sizeLineWidth)

        if self.pretty:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        else:
            ax.set_title("Voltage Drop per Edge against Time", fontsize=self.sizeFont)
            ax.set_xlabel("Time (s)", fontsize=self.sizeFont)
            ax.set_ylabel("Voltage Drop per Edge (V)", fontsize=self.sizeFont)
            ax.tick_params(labelsize=self.sizeLabel)
            ax.set_xticks(self.xticks if ((xmin, xmax) == intervalOfInterest) else list(set(np.append(np.array(self.xticks), np.array([xmin, xmax])).tolist())))
            ax.legend(handles=[
                    Line2D([0], [0], color='red', lw=1),
                    Line2D([0], [0], color='black', lw=1)
                ], loc="upper left", labels=['Broken Edges', 'Unbroken Edges'], fontsize=(self.sizeFont / 2),
            )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, np.nanmax(dataEdgeVolts[:, xmin:xmaxForSlicing]) * 1.1)

        fig.tight_layout()
        plt.tight_layout()
        self._Save(self.filePath2, "DataEdgeVolts", None)
        plt.show(fig) if self.show else plt.close(fig)