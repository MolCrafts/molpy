# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

from typing import Literal
import numpy as np

class Box:
    
    def __init__(self, boundary_condition: Literal['ppp'], **kwargs) -> None:
        
        """[summary]
        """
        
        # ref:
        # https://docs.lammps.org/Howto_triclinic.html
        # https://hoomd-blue.readthedocs.io/en/stable/box.html
        
        self.boundary_condition = boundary_condition

        if 'xlo' in kwargs:
            self.defByEdge(**kwargs)
            
        elif 'a1' in kwargs:
            self.defByLatticeVectors(**kwargs)

    def defByEdge(self, xlo, xhi, ylo, yhi, zlo, zhi, xy=0, xz=0, yz=0):
        
        self.xlo = xlo
        self.ylo = ylo
        self.zlo = zlo
        self.xhi = xhi
        self.yhi = yhi
        self.zhi = zhi
        
        self.lx = self.xhi - self.xlo
        self.ly = self.yhi - self.ylo
        self.lz = self.zhi - self.zlo
        
        self.xy = xy
        self.xz = xz
        self.yz = yz    
        
        self.gamma = np.cosh(xy/np.sqrt(1+xy**2))
        self.beta = np.cosh(xz/np.sqrt(1+xz**2+yz**2))
        self.alpha = np.cosh((xy*xz+yz)/np.sqrt(1+xy**2)/np.sqrt(1+xz**2+yz**2))
        self._post_def_()
        
    def defByLatticeVectors(self, a1, a2, a3):
        self.lx = np.linalg.norm(a1)
        a2x = a1@a2/np.linalg(a1)
        self.ly = np.sqrt(a2@a2-a2x**2)
        self.xy = a2x/self.ly
        crossa1a2 = np.cross(a1, a2)
        self.lz = a3 * (crossa1a2/np.linalg.norm(crossa1a2))
        a3x = a1@a3/np.linalg.norm(a1)
        self.xz = a3x/self.lz
        self.yz = (a2@a3-a2x*a3x)/self.ly/self.lz
        
        self.xlo = 0
        self.xhi = self.lx
        self.ylo = 0
        self.yhi = self.ly
        self.zlo = 0
        self.zhi = self.lz
        self._post_def_()
        
    def _post_def_(self):
        
        # box matrix h
        self._cellpar = np.array([
            [self.lx, self.xy*self.ly, self.xz*self.lz],
            [0, self.ly, self.yz*self.lz],
            [0, 0, self.lz]
        ])

        self._boundary_condition = self.boundary_condition
        self._x_bc, self._y_bc, self._z_bc = self.boundary_condition
        
    @property
    def x_vec(self):
        return self._cellpar[:, 0]
    
    @property
    def y_vec(self):
        return self._cellpar[:, 1]
    
    @property
    def z_vec(self):
        return self._cellpar[:, 2]
    
    @property
    def origin(self):
        return np.array([self.xlo, self.ylo, self.zlo])
    
    def wrap(self, position):
        
        edge_length = np.array([self.lx, self.ly, self.lz])
        offset = np.floor_divide(position, edge_length)
        return position - offset*edge_length
    
    def displace(self, pos1, pos2):
        
        return self.wrap(pos1) - self.wrap(pos2)
    
class CellList:
    
    def __init__(self, box, rcutoff) -> None:
        self.box = box
        self.rcutoff = rcutoff
        
    def build(self):
        
        pass
        

    
#!/usr/bin/env python
import os
import sys
import numpy as np

CELL_LIST_BUFFER = 80
MAX_N_NEIGHBOURS = 600

class NBLS:
    '''
    The container object for neighbour list information
    '''
    
    def __init__(self, nbs, n_nbs, distances2, dr_vecs):
        self.nbs = nbs
        self.n_nbs = n_nbs
        self.distances2 = distances2
        self.dr_vecs = dr_vecs
        return

# @njit
def construct_nblist(positions, box, rc):
    '''
    Build neighbour list
    Inputs:
        positions:
            N * 3: the positions of each site
        box:
            3 * 3: the box dimension array, axes arranged in rows
        rc:
            float: the cutoff distance
    Outputs:
        nbs:
            N * MAX_N_NEIGHBOURS: the indices of neighbors for each site
            For each row (e.g., row i), only the first 0~n_nbs[i] elements are meaningful
        n_nbs:
            len(N) array: the number of neighbors for each site
        distances2:
            N * MAX_N_NEIGHBOURS: the r^2 between each neighbor and the central atom
            Again, only the first 0~n_nbs[i] elements in each row are meaningful
        dr_vecs:
            N * MAX_N_NEIGHBOURS * 3: the \vec{r} pointing from the central atom to the neighbors
    '''
    # Some constants

    # construct cells
    box_inv = np.linalg.inv(box)
    # len_abc_star = np.linalg.norm(box_inv, axis=0)
    len_abc_star = np.sqrt(np.sum(box_inv*box_inv, axis=0))
    h_box = 1 / len_abc_star # the "heights" of the box
    n_cells = np.floor(h_box/rc).astype(np.int32)
    n_totc = np.prod(n_cells)
    n_atoms = len(positions)
    density = n_atoms / np.prod(n_cells)
    n_max = int(np.floor(density)) + CELL_LIST_BUFFER
    cell = np.diag(1./n_cells).dot(box)
    cell_inv = np.linalg.inv(cell)

    upositions = positions.dot(cell_inv)
    indices = np.floor(upositions).astype(np.int32)
    # for safty, do pbc shift
    indices[:, 0] = indices[:, 0] % n_cells[0]
    indices[:, 1] = indices[:, 1] % n_cells[1]
    indices[:, 2] = indices[:, 2] % n_cells[2]
    cell_list = np.full((n_cells[0], n_cells[1], n_cells[2], n_max), -1)
    counts = np.full((n_cells[0], n_cells[1], n_cells[2]), 0)
    for i_atom in range(n_atoms):
        ia, ib, ic = indices[i_atom]
        cell_list[ia, ib, ic, counts[ia, ib, ic]] = i_atom
        counts[ia, ib, ic] += 1

    # cell adjacency
    na, nb, nc = n_cells
    cell_list = cell_list.reshape((n_totc, n_max))
    counts = counts.reshape(n_totc)
   
    rc2 = rc * rc
    nbs = np.empty((n_atoms, MAX_N_NEIGHBOURS), dtype=np.int32)
    n_nbs = np.zeros(n_atoms, dtype=np.int32)
    distances2 = np.empty((n_atoms, MAX_N_NEIGHBOURS))
    dr_vecs = np.empty((n_atoms, MAX_N_NEIGHBOURS, 3))
    icells_3d = np.empty((n_totc, 3))
    for ic in range(n_totc):
        icells_3d[ic] = np.array([ic // (nb*nc), (ic // nc) % nb, ic % nc], dtype=np.int32)
    for ic in range(n_totc):
        ic_3d = icells_3d[ic]
        if counts[ic] == 0:
            continue
        for jc in range(ic, n_totc):
            if counts[jc] == 0:
                continue
            jc_3d = icells_3d[jc]
            di_cell = jc_3d - ic_3d
            di_cell -= (di_cell + n_cells//2) // n_cells * n_cells
            max_di = np.max(np.abs(di_cell))
            # two cells are not neighboring
            if max_di > 1:
                continue
            indices_i = cell_list[ic, 0:counts[ic]]
            indices_j = cell_list[jc, 0:counts[jc]]
            indices_i = np.repeat(indices_i, counts[jc])
            indices_j = np.repeat(indices_j, counts[ic]).reshape(-1, counts[ic]).T.flatten()
            if ic == jc:
                ifilter = (indices_j > indices_i)
                indices_i = indices_i[ifilter]
                indices_j = indices_j[ifilter]
            ri = positions[indices_i]
            rj = positions[indices_j]
            dr = rj - ri
            ds = np.dot(dr, box_inv)
            ds -= np.floor(ds+0.5)
            dr = np.dot(ds, box)
            drdr = np.sum(dr*dr, axis=1)
            for ii in np.where(drdr < rc2)[0]:
                i = indices_i[ii]
                j = indices_j[ii]
                nbs[i, n_nbs[i]] = j
                nbs[j, n_nbs[j]] = i
                distances2[i, n_nbs[i]] = drdr[ii]
                distances2[j, n_nbs[j]] = drdr[ii]
                dr_vecs[i, n_nbs[i], :] = dr[ii]
                dr_vecs[j, n_nbs[j], :] = -dr[ii]
                n_nbs[i] += 1
                n_nbs[j] += 1
       
    return NBLS(nbs, n_nbs, distances2, dr_vecs)



################################
# systematic test neighbour list
################################
if __name__ == "__main__":
    print('OMP_NUM_THREADS:', os.environ["OMP_NUM_THREADS"])

    np.random.seed(100)

    r1 = np.zeros(3)
    r2 = np.ones(3)
    box = np.array([
         [30.0, 0.00, 0.00],
         [0.0, 40.0, 0.0],
         [0.0, 0.0, 35.0]
         ])
    box_inv = np.linalg.inv(box)
    #box_inv = np.identity(3)/30.0

    # print(spatial.distance_pbc(r1, r2, box, box_inv, 0))

    # construct inputs
    n_atoms = 4000
    r_cut = 4.0
    spositions = np.random.random((n_atoms, 3))
    positions = spositions.dot(box)

    nbs, n_nbs, distances2, dr_vecs = construct_nblist(positions, box, r_cut)

    flag = 0
    print('*****, Start testing nbs and distances2...')
    for i_atom0, neighbours in enumerate(nbs):
        neighbours = neighbours[0:n_nbs[i_atom0]]
        r0 = positions[i_atom0]
        print('Testing particle', i_atom0)
        for i_atom1 in range(n_atoms):
            if i_atom1 == i_atom0:
                continue 
            r1 = positions[i_atom1]
            dr = spatial.distance_pbc(r0, r1, box.T, box_inv.T, 0)
            if dr >= r_cut and i_atom1 in neighbours:
                print('ERROR: Wrong interacting pair:', i_atom0, i_atom1, dr)
                flag = 1
            elif dr < r_cut:
                index = np.where(neighbours==i_atom1)[0]
                if len(index) == 0:
                    print('ERROR: Missing interacting pair:', i_atom0, i_atom1, dr)
                    flag = 1
                elif len(index) > 1:
                    print('ERROR: Duplication in neighbour list:', i_atom0, i_atom1, dr)
                    flag = 1
                else:
                    if abs(np.sqrt(distances2[i_atom0, index[0]]) - dr) > 1e-6:
                        print('ERROR: Distances mismatch:', i_atom0, i_atom1, dr, np.sqrt(distances2[i_atom0, index[0]]))
                        flag = 1
    if flag == 0:
       print('Neighbour list checks out.')