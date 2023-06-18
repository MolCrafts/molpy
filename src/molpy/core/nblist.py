# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-10-05
# version: 0.0.1

import freud
import numpy as np

class NeighborList:

    def __init__(self, box, points, query_args:dict):

        self.query_args = query_args
        self.nblist = freud.AABBQuery(box, points).query(points, query_args)

    def get_pairs(self, padding=0, capacity_multiplier=1.2):

        nblist = self.nblist.toNeighborList()
        nblist = np.vstack((nblist[:, 0], nblist[:, 1])).T
        nblist = nblist.astype(np.int32)
        msk = (nblist[:, 0] - nblist[:, 1]) < 0
        nblist = nblist[msk]
        
        if padding:
            if padding < 0 or not isinstance(padding, int):
                raise ValueError(f'padding must be set as int')
            else:
                pd_arr = np.zeros((len(nblist)*capacity_multiplier, nblist.shape[1]), dtype=np.int32)
                nblist = np.concatenate((nblist, pd_arr), axis=0)

        return nblist