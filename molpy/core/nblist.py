# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-10-05
# version: 0.0.1

import freud
import numpy as np

class NeighborList:

    def __init__(self, query_args:dict):

        self.query_args = query_args
        
    def query(self, box, points):
        
        self.nblist = freud.AABBQuery(box, points).query(points, self.query_args).toNeighborList()

    def get_neighbor_index(self, i):

        mask = self.nblist.query_point_indices == i
        return self.nblist.point_indices[mask]

    def get_neighbor_positions(self, i):

        index = self.get_neighbor_index(i)
        return self.nblist.points[index]

    def get_neighbor_counts(self, i=None):
        if i is None:
            return len(self.get_neighbor_positions(i))
        return self.nblist.neighbor_counts
