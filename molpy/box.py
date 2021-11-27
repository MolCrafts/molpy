# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

from typing import Literal
import numpy as np

def factory_ndarray_dtype_list(shape):
    return np.frompyfunc(list, 0, 1)(np.empty(shape, dtype=list))

def getPos(atom):
    return getattr(atom, 'position')

map_getPos = np.vectorize(getPos, otypes=[np.ndarray])

class Box:
    
    def __init__(self, boundary_condition: Literal['ppp'], **kwargs) -> None:
               
        # ref:
        # https://docs.lammps.org/Howto_triclinic.html
        # https://hoomd-blue.readthedocs.io/en/stable/box.html
        
        """ Describes the properties of the box
        """
        
        self.boundary_condition = boundary_condition

        if 'xlo' in kwargs:
            self.defByEdge(**kwargs)
            
        elif 'a1' in kwargs:
            self.defByLatticeVectors(**kwargs)

    def defByEdge(self, xlo, xhi, ylo, yhi, zlo, zhi, xy=0, xz=0, yz=0):
        
        """ define Box via start and end which in LAMMPS and hoomd-blue style
        """
                
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
        """ define Box via lattice vector

        Args:
            a1 (np.ndarray): Must lie on the x-axis
            a2 (np.ndarray): Must lie on the xy-plane
            a3 (np.ndarray)
        """
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
    def box_size(self):
        return self._cellpar
        
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
        """wrap position(s) array back into box

        Args:
            position (np.ndarray)

        Returns:
            np.ndarray: wrapped position(s)
        """
        edge_length = np.array([self.lx, self.ly, self.lz])
        offset = np.floor_divide(position, edge_length)
        return position - offset*edge_length
    
    def displace(self, pos1, pos2):
        """find the distance between two positions in minimum image convention(mic)

        Args:
            pos1 (np.ndarray)
            pos2 (np.ndarray)

        Returns:
            np.ndarray
        """
        return self.wrap(pos1) - self.wrap(pos2)
    
    
class CellList:
    
    def __init__(self, box, rcutoff) -> None:
        
        assert isinstance(box, Box), TypeError(f'box must be an instance of Box class, but {type(box)}')
        
        self.box = box
        self.rcutoff = rcutoff
        
    def build(self):
        
        box = self.box.box_size
        rc = self.rcutoff
        
        box_inv = np.linalg.inv(box)
        len_abc_star = np.sqrt(np.sum(box_inv*box_inv, axis=0))
        h_box = 1 / len_abc_star # the "heights" of the box
        
        # cell_vec := n_cell_x, n_cell_y, n_cell_z
        self.cell_vec = np.floor_divide(h_box, rc).astype(np.int32)
        self.cell_list = factory_ndarray_dtype_list(self.cell_vec)
        self.ncell = np.prod(self.cell_vec, dtype=int)
        self.cell_size = np.diag(1./self.cell_vec).dot(box)
        self.cell_inv = np.linalg.inv(self.cell_size)
        
    def reset(self):
        
        self.build()
        
    def update(self, atoms, rebuild=True):
        
        if rebuild:
            self.build
        
        positions = np.vstack(map_getPos(atoms))
        
        wrapped_pos = self.box.wrap(positions)
        
        indices = np.floor(np.dot(wrapped_pos, self.cell_inv)).astype(int)
        
        for cell, atom in zip(self.cell_list[indices[:,0], indices[:,1], indices[:,2]], atoms):
            cell.append(atom)
            
    def getAdjCell(self, center_index, radius=1):
        
        cell_list = self.cell_list
        
        # edge length
        length = radius * 2 + 1
        
        # length^3 exclude (0, 0, 0)
        adj_cell_list = factory_ndarray_dtype_list((length**3-1))  # 1-d ndarray
        
        index = 0
        for adj_increment_index in np.ndindex((length, length, length)):
            
            adj_index = center_index + np.array(adj_increment_index) - radius
            if np.array_equal(adj_increment_index, np.array([0, 0, 0])):
                continue
            
            # wrap
            offset = np.floor_divide(adj_index, self.cell_vec)
            adj_index -= offset * self.cell_vec
            
            adj_cell_list[index] = cell_list[tuple(adj_index)]
            index += 1
                
        return adj_cell_list
        
class NeighborList:
    
    def __init__(self, cellList) -> None:
        
        self.rcutoff = cellList.rcutoff
        self.cellList = cellList
        self.box = cellList.box
        self.build()
            
    def build(self):
        
        self.neighbor_list = {} # DefaultDict(list)
        
    def reset(self):
        
        self.build()
        
    def update(self, atoms=None):
        
        if atoms is not None:
            self.cellList.update(atoms)
        
        cellList = self.cellList.cell_list
        for index, cell in np.ndenumerate(cellList):
            adj_cells = self.cellList.getAdjCell(index)  # (nadj, ) dtype=list
            for atom in cell:

                # TODO: get positions from adj cells cost half of time
                cen_pos = atom.position  # center atom's position
                adj_atoms = []
                for cell in adj_cells:
                    adj_atoms.extend(cell)
                # map_extend = np.vectorize(adj_atoms.extend)
                # map_extend(adj_cells)
                adj_atoms = np.asarray(adj_atoms)

                pair_pos = np.vstack(map_getPos(adj_atoms))

                # pair_pos = np.array(map(lambda atom: getattr(atom, 'position'), adj_atoms))
                dr = pair_pos - cen_pos
                distance = np.linalg.norm(dr)
                nei_atoms = adj_atoms[distance<self.rcutoff]
                self.neighbor_list[atom] = {
                    'neighbors': nei_atoms,
                    'dr': dr,
                    'distance': distance
                }
