import numpy as np
from molpy.core.box import Box
from functools import partial

class _NeighborListBase:

    def build(self):
        """build neighborlist from scratch, always reset the cache
        """

    def update(self):
        """update neighborlist, 
        """
        ...

    def query(self):
        """query neighbors of given atoms"""
        ...

class _NeighborListCache:

    def __init__(self, xyz=None, box=None, cutoff=None):

        # config
        self._box = box
        self._cutoff = cutoff

        # cache
        self._xyz = xyz
        self._mapping = None
        self._diff = None
        self._distances = None

    @property
    def box(self):
        return self._box
    
    @property
    def cutoff(self):
        return self._cutoff
    
    @property
    def xyz(self):
        return self._xyz
    
    @property
    def mapping(self):
        return self._mapping
    
    @property
    def diff(self):
        return self._diff
    
    @property
    def distances(self):
        return self._distances

    def update_cache(self, xyz, mapping, diff, distances):
        ...

    def update_config(self, box, cutoff):
        ...

# class NblistSkin:

#     def __init__(self, nblist: _NeighborListBase, skin: float):
        
#         self._nblist = nblist
#         self._skin = skin

#     def build(self, xyz: np.ndarray, box: Box, cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

#         mapping, diff, distances = self._nblist.build(xyz, box, cutoff + self._skin)

#         skin_mask = np.logical_not(distances < cutoff)
#         self.skin_mapping = mapping[skin_mask]
#         self.skin_diff = diff[skin_mask]
#         self.skin_distances = distances[skin_mask]
#         return mapping[not skin_mask], diff[not skin_mask], distances[not skin_mask]
    
#     def update(self, xyz):
        
#         mapping, diff, distances = self.query(xyz, self._nblist._cache.cutoff+ self._skin)

#         self._nblist._cache.update_cache(xyz, mapping, diff, distances)

#         skin_mask = np.logical_not(distances < self._nblist._cache.cutoff)
#         self.skin_mapping = mapping[skin_mask]
#         self.skin_diff = diff[skin_mask]
#         self.skin_distances = distances[skin_mask]

#         return self._cache.mapping, self._cache.diff, self._cache.distances
    
#     def query(self, centers: np.ndarray, cutoff: float):
        
#         return self._nblist.query(centers, cutoff + self._skin)
        

class NaiveNbList(_NeighborListBase, _NeighborListCache):

    def __init__(self):
        
        self._cache = _NeighborListCache()

    def build(self, xyz: np.ndarray, box: Box, cutoff: float, exclude_ii=True, exclude_ji=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        assert cutoff > 0, "cutoff must be positive"

        self._cache.update_config(box, cutoff)
        
        # naive impl: O(n^2)
        dr = box.diff_self(xyz)  # dr.shape == (n, n, 3)
        r = np.linalg.norm(dr, axis=-1)
        mask = r < cutoff
        if exclude_ii:
            mask[np.diag_indices_from(mask)] = False
        if exclude_ji:
            mask[np.tril_indices_from(mask, k=-1)] = False
        mapping = np.argwhere(mask)
        diff = dr[mask]
        distances = r[mask]
        
        return mapping, diff, distances
    
    # def batch_build(self, xyz:np.ndarray, box: Box, cutoff: float, exclude_ii=True, exclude_ji=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    #     batched_build = partial(self.build, cutoff=cutoff, exclude_ii=exclude_ii, exclude_ji=exclude_ji)
    #     return np.vectorize(batched_build)(xyz, box)
    
    def query(self, centers: np.ndarray, cutoff: float):
        
        dr = self._cache.box.diff_all(centers, self._cache.xyz)
        r = np.linalg.norm(dr, axis=-1)
        mask = r < cutoff
        mapping = np.argwhere(mask)
        diff = dr[mask]
        distances = r[mask]
        
        return mapping, diff, distances

    def update(self, xyz, ):

        mapping, diff, distances = self.query(xyz, self._cache.cutoff)

        self._cache.update_cache(xyz, mapping, diff, distances)

        return self._cache.mapping, self._cache.diff, self._cache.distances

class CellListNbList(_NeighborListBase, _NeighborListCache):
    ...