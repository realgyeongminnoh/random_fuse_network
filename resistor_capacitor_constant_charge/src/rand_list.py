import numpy as np

from src.edge_list import EdgeList


class RandList:
    def __init__(self, edgeList: EdgeList, width: float, seed: int):
        self.edgeList = edgeList
        self.width = float(width)
        self.seed = seed
        self.minThrVolt = 1.0 - width / 2
        self.maxThrVolt = 1.0 + width / 2
        self.randThrVolts = None
        self._GenerateRandThrVolts()

    
    def _GenerateRandThrVolts(self):
        np.random.seed(self.seed)
        self.randThrVolts = np.random.uniform(self.minThrVolt, self.maxThrVolt, self.edgeList.numEdge)