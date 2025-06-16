# MolPy Core Data Structures

This document provides a complete guide to the core data structures in the MolPy framework, including tutorials and API references for the `Struct` and `Frame` modules.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Tutorials](#basic-tutorials)
3. [Advanced Extensions](#advanced-extensions)
4. [API Reference](#api-reference)
5. [Examples Collection](#examples-collection)

---

## Quick Start

MolPy provides flexible and powerful data structures for representing and manipulating molecular systems:

```python
import molpy as mp
import numpy as np

# Create atoms
atom1 = mp.Atom(name="C1", element="C", xyz=[0.0, 0.0, 0.0])
atom2 = mp.Atom(name="C2", element="C", xyz=[1.5, 0.0, 0.0])

# Create molecular structure
struct = mp.AtomicStructure(name="ethane")
struct.add_atom(atom1)
struct.add_atom(atom2)

# Add chemical bond
bond = mp.Bond(atom1, atom2, bond_type="single")
struct.add_bond(bond)

print(f"Structure contains {len(struct.atoms)} atoms and {len(struct.bonds)} bonds")
```

---

## Basic Tutorials

### 1. Entity System

The core of MolPy is the `Entity` class, which provides dictionary-like flexible data storage:

```python
import molpy as mp

# Basic Entity usage
entity = mp.Entity(name="test", value=42)

# 字典式访问
print(entity["name"])  # "test"
print(entity.get("value"))  # 42

# 属性式访问
entity.new_prop = "hello"
print(entity["new_prop"])  # "hello"

# 克隆和修改
new_entity = entity.clone(name="modified")
print(new_entity["name"])  # "modified"
```

### 2. 原子和空间操作

`Atom`类结合了Entity的灵活性和空间操作能力：

```python
# 创建原子
carbon = mp.Atom(
    name="C1",
    element="C", 
    xyz=[0.0, 0.0, 0.0],
    charge=0.0
)

# 空间操作
carbon.move([1.0, 0.0, 0.0])  # 平移
carbon.rotate(np.pi/4, [0, 0, 1])  # 绕z轴旋转45度

# 计算距离
hydrogen = mp.Atom(name="H1", element="H", xyz=[1.1, 0.0, 0.0])
distance = carbon.distance_to(hydrogen)
```

### 3. 化学键和拓扑

```python
# 创建化学键
bond = mp.Bond(carbon, hydrogen, bond_type="single", length=1.1)

# 创建角度
oxygen = mp.Atom(name="O1", element="O", xyz=[0.0, 1.0, 0.0])
angle = mp.Angle(hydrogen, carbon, oxygen, angle_type="sp3")

# 创建二面角
nitrogen = mp.Atom(name="N1", element="N", xyz=[0.0, 0.0, 1.0])
dihedral = mp.Dihedral(hydrogen, carbon, oxygen, nitrogen)
```

### 4. 分子结构构建

```python
# 方法1：逐步构建
struct = mp.AtomicStructure(name="water")

# 添加原子
o_atom = struct.def_atom(name="O", element="O", xyz=[0.0, 0.0, 0.0])
h1_atom = struct.def_atom(name="H1", element="H", xyz=[0.96, 0.0, 0.0])
h2_atom = struct.def_atom(name="H2", element="H", xyz=[-0.24, 0.93, 0.0])

# 添加化学键
struct.def_bond(o_atom, h1_atom, bond_type="single")
struct.def_bond(o_atom, h2_atom, bond_type="single")

# 方法2：批量添加
atoms = [
    mp.Atom(name=f"C{i}", element="C", xyz=[i*1.5, 0, 0]) 
    for i in range(4)
]
struct.add_atoms(atoms)

# 批量添加化学键
bonds = [mp.Bond(atoms[i], atoms[i+1]) for i in range(3)]
struct.add_bonds(bonds)
```

### 5. Frame数据结构

`Frame`提供了高效的表格式数据存储，基于xarray：

```python
import molpy as mp

# 从数组创建Frame
atoms_data = {
    'name': ['C1', 'C2', 'H1', 'H2'],
    'element': ['C', 'C', 'H', 'H'],
    'xyz': [[0,0,0], [1.5,0,0], [2.5,0,0], [3.5,0,0]],
    'charge': [0.0, 0.0, 0.0, 0.0]
}

frame = mp.Frame(atoms=atoms_data)
print(frame['atoms'])

# Frame与Struct的转换
struct = mp.MolecularStructure.from_frame(frame, name="alkane")
new_frame = struct.to_frame()
```

---

## 高级扩展

### 1. 层次结构管理

MolPy支持复杂的层次结构管理：

```python
# 创建主结构
protein = mp.AtomicStructure(name="protein")

# 创建子结构
residue1 = mp.AtomicStructure(name="ALA1")
residue2 = mp.AtomicStructure(name="GLY2")

# 建立层次关系
protein.add_child(residue1)
protein.add_child(residue2)

# 层次遍历
def print_structure(node):
    print(f"节点: {node.get('name', 'unnamed')}")

protein.traverse_dfs(print_structure)

# 层次信息
info = protein.get_hierarchy_info()
print(f"深度: {info['depth']}, 子节点数: {info['num_children']}")
```

### 2. 自定义实体类型

```python
class Residue(mp.Entity):
    """自定义残基类"""
    
    def __init__(self, name, residue_type, **kwargs):
        super().__init__(name=name, residue_type=residue_type, **kwargs)
        self._atoms = []
    
    def add_atom(self, atom):
        self._atoms.append(atom)
        atom.residue = self
    
    @property
    def atoms(self):
        return self._atoms.copy()
    
    def backbone_atoms(self):
        """返回主链原子"""
        return [atom for atom in self._atoms 
                if atom.get('name') in ['N', 'CA', 'C', 'O']]

# 使用自定义类
ala = Residue("ALA", "amino_acid", sequence_number=1)
```

### 3. 空间操作扩展

```python
class AdvancedSpatialOperations:
    """高级空间操作工具"""
    
    @staticmethod
    def align_structures(struct1, struct2, atom_pairs):
        """对齐两个结构"""
        # 计算变换矩阵
        coords1 = np.array([struct1.atoms[i].xyz for i, _ in atom_pairs])
        coords2 = np.array([struct2.atoms[j].xyz for _, j in atom_pairs])
        
        # SVD对齐算法
        centroid1 = coords1.mean(axis=0)
        centroid2 = coords2.mean(axis=0)
        
        coords1_centered = coords1 - centroid1
        coords2_centered = coords2 - centroid2
        
        H = coords1_centered.T @ coords2_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # 应用变换
        for atom in struct2.atoms:
            atom.xyz = (R @ (atom.xyz - centroid2)) + centroid1
    
    @staticmethod
    def calculate_rmsd(struct1, struct2, atom_pairs):
        """计算RMSD"""
        coords1 = np.array([struct1.atoms[i].xyz for i, _ in atom_pairs])
        coords2 = np.array([struct2.atoms[j].xyz for _, j in atom_pairs])
        
        diff = coords1 - coords2
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

# 使用高级操作
rmsd = AdvancedSpatialOperations.calculate_rmsd(struct1, struct2, pairs)
```

### 4. 容器和集合操作

```python
# 高级容器操作
class StructureCollection:
    """结构集合管理"""
    
    def __init__(self):
        self.structures = mp.Entities()
    
    def add_structure(self, struct):
        self.structures.add(struct)
    
    def filter_by_size(self, min_atoms=None, max_atoms=None):
        """按大小过滤结构"""
        filtered = []
        for struct in self.structures:
            natoms = len(struct.atoms)
            if min_atoms and natoms < min_atoms:
                continue
            if max_atoms and natoms > max_atoms:
                continue
            filtered.append(struct)
        return filtered
    
    def merge_all(self, name="merged"):
        """合并所有结构"""
        merged = mp.AtomicStructure(name=name)
        for struct in self.structures:
            merged.add_struct(struct)
        return merged

# 使用集合操作
collection = StructureCollection()
collection.add_structure(struct1)
collection.add_structure(struct2)
large_structures = collection.filter_by_size(min_atoms=100)
```

### 5. 自定义Frame处理

```python
class CustomFrameProcessor:
    """自定义Frame处理器"""
    
    @staticmethod
    def add_calculated_properties(frame):
        """添加计算属性"""
        atoms = frame['atoms']
        
        # 计算质心
        if 'xyz' in atoms.coords and 'mass' in atoms.coords:
            xyz = atoms.coords['xyz'].values
            mass = atoms.coords['mass'].values
            
            total_mass = mass.sum()
            com = np.sum(xyz * mass[:, np.newaxis], axis=0) / total_mass
            
            # 添加到frame
            frame.attrs['center_of_mass'] = com
        
        return frame
    
    @staticmethod
    def filter_atoms_by_element(frame, elements):
        """按元素过滤原子"""
        atoms = frame['atoms']
        if 'element' not in atoms.coords:
            raise ValueError("Frame中缺少element信息")
        
        mask = np.isin(atoms.coords['element'].values, elements)
        filtered_atoms = atoms.isel(index=mask)
        
        new_frame = mp.Frame()
        new_frame['atoms'] = filtered_atoms
        return new_frame

# 使用自定义处理器
enhanced_frame = CustomFrameProcessor.add_calculated_properties(frame)
carbon_frame = CustomFrameProcessor.filter_atoms_by_element(frame, ['C'])
```

---

## API参考

### Entity类

```python
class Entity(UserDict):
    """基础实体类，提供字典式存储"""
    
    def __init__(self, **kwargs):
        """初始化实体
        
        Args:
            **kwargs: 初始属性
        """
    
    def clone(self, **modify):
        """创建深拷贝
        
        Args:
            **modify: 要修改的属性
            
        Returns:
            Entity: 新的实体实例
        """
    
    def to_dict(self) -> dict:
        """转换为标准字典
        
        Returns:
            dict: 实体的字典表示
        """
```

### Atom类

```python
class Atom(Entity, SpatialMixin):
    """原子类，包含空间信息"""
    
    def __init__(self, name: str = "", xyz: Optional[ArrayLike] = None, **kwargs):
        """初始化原子
        
        Args:
            name: 原子名称
            xyz: 三维坐标
            **kwargs: 其他属性
        """
    
    @property
    def xyz(self) -> np.ndarray:
        """获取坐标"""
    
    @xyz.setter
    def xyz(self, value: ArrayLike) -> None:
        """设置坐标"""
    
    def distance_to(self, other: "SpatialMixin") -> float:
        """计算到另一个空间实体的距离"""
    
    def move(self, vector: ArrayLike) -> "SpatialMixin":
        """平移原子"""
    
    def rotate(self, theta: float, axis: ArrayLike) -> "SpatialMixin":
        """旋转原子"""
```

### Bond类

```python
class Bond(ManyBody):
    """化学键类"""
    
    def __init__(self, atom1: Atom, atom2: Atom, **kwargs):
        """初始化化学键
        
        Args:
            atom1: 第一个原子
            atom2: 第二个原子
            **kwargs: 键的属性（如bond_type, length等）
        """
    
    @property
    def atom1(self) -> Atom:
        """第一个原子"""
    
    @property
    def atom2(self) -> Atom:
        """第二个原子"""
    
    @property
    def length(self) -> float:
        """键长"""
```

### AtomicStructure类

```python
class AtomicStructure(Struct, SpatialMixin, HierarchyMixin):
    """原子结构类"""
    
    def __init__(self, name: str = "", **props):
        """初始化原子结构
        
        Args:
            name: 结构名称
            **props: 其他属性
        """
    
    @property
    def atoms(self) -> Entities:
        """原子集合"""
    
    @property
    def bonds(self) -> Entities:
        """化学键集合"""
    
    def add_atom(self, atom: Atom) -> Atom:
        """添加原子"""
    
    def def_atom(self, **props) -> Atom:
        """定义并添加原子"""
    
    def add_bond(self, bond: Bond) -> Bond:
        """添加化学键"""
    
    def def_bond(self, atom1, atom2, **kwargs) -> Bond:
        """定义并添加化学键"""
    
    def add_struct(self, struct: "AtomicStructure") -> "AtomicStructure":
        """添加子结构"""
    
    @classmethod
    def concat(cls, name: str, structs) -> "AtomicStructure":
        """连接多个结构"""
```

### Frame类

```python
class Frame(dict):
    """Frame数据结构，基于xarray"""
    
    def __init__(self, **kwargs):
        """初始化Frame
        
        Args:
            **kwargs: 数据映射
        """
    
    def to_struct(self, struct_class=None, **kwargs):
        """转换为结构对象"""
    
    @classmethod
    def from_struct(cls, struct, include_bonds=True, **kwargs):
        """从结构对象创建Frame"""
```

### HierarchyMixin类

```python
class HierarchyMixin:
    """层次结构混入类"""
    
    @property
    def parent(self):
        """父节点"""
    
    @property
    def children(self):
        """子节点列表"""
    
    @property
    def root(self):
        """根节点"""
    
    def add_child(self, child):
        """添加子节点"""
    
    def remove_child(self, child):
        """移除子节点"""
    
    def traverse_dfs(self, visit_func):
        """深度优先遍历"""
    
    def traverse_bfs(self, visit_func):
        """广度优先遍历"""
    
    def find_by_name(self, name: str):
        """按名称查找节点"""
```

---

## 示例集合

### 示例1：蛋白质结构分析

```python
import molpy as mp
import numpy as np

def analyze_protein_structure(pdb_file):
    """分析蛋白质结构"""
    
    # 加载结构（假设有相应的IO模块）
    protein = mp.load_pdb(pdb_file)
    
    # 分析二级结构
    alpha_helices = []
    beta_sheets = []
    
    for residue in protein.children:
        if residue.get('secondary_structure') == 'helix':
            alpha_helices.append(residue)
        elif residue.get('secondary_structure') == 'sheet':
            beta_sheets.append(residue)
    
    # 计算几何中心
    all_coords = protein.xyz
    geometric_center = all_coords.mean(axis=0)
    
    # 计算回旋半径
    distances = np.linalg.norm(all_coords - geometric_center, axis=1)
    radius_of_gyration = np.sqrt(np.mean(distances**2))
    
    return {
        'n_residues': len(protein.children),
        'n_helices': len(alpha_helices),
        'n_sheets': len(beta_sheets),
        'geometric_center': geometric_center,
        'radius_of_gyration': radius_of_gyration
    }
```

### 示例2：分子动力学轨迹处理

```python
def process_md_trajectory(trajectory_frames):
    """处理分子动力学轨迹"""
    
    structures = []
    
    for i, frame_data in enumerate(trajectory_frames):
        # 创建结构
        struct = mp.AtomicStructure(name=f"frame_{i}")
        
        # 从frame数据创建原子
        for atom_data in frame_data['atoms']:
            atom = mp.Atom(**atom_data)
            struct.add_atom(atom)
        
        structures.append(struct)
    
    # 分析轨迹
    trajectory_analysis = {
        'n_frames': len(structures),
        'avg_natoms': np.mean([len(s.atoms) for s in structures]),
        'coordinate_evolution': []
    }
    
    # 跟踪特定原子的坐标变化
    if structures:
        reference_atom = structures[0].atoms[0]  # 第一个原子作为参考
        
        for struct in structures:
            if len(struct.atoms) > 0:
                coord = struct.atoms[0].xyz
                trajectory_analysis['coordinate_evolution'].append(coord)
    
    return trajectory_analysis
```

### 示例3：化学反应建模

```python
class ReactionModeler:
    """化学反应建模工具"""
    
    def __init__(self):
        self.reactants = []
        self.products = []
        self.transition_states = []
    
    def add_reactant(self, struct):
        """添加反应物"""
        struct.reaction_role = "reactant"
        self.reactants.append(struct)
    
    def add_product(self, struct):
        """添加产物"""
        struct.reaction_role = "product"
        self.products.append(struct)
    
    def model_bond_breaking(self, reactant_idx, atom1_idx, atom2_idx):
        """模拟键断裂"""
        reactant = self.reactants[reactant_idx]
        
        # 找到要断裂的键
        target_bond = None
        for bond in reactant.bonds:
            if ((bond.atom1 is reactant.atoms[atom1_idx] and 
                 bond.atom2 is reactant.atoms[atom2_idx]) or
                (bond.atom1 is reactant.atoms[atom2_idx] and 
                 bond.atom2 is reactant.atoms[atom1_idx])):
                target_bond = bond
                break
        
        if target_bond:
            # 创建过渡态
            ts_struct = reactant.clone()
            ts_struct.name = f"{reactant.get('name', 'reactant')}_TS"
            
            # 逐渐拉长键
            atom1 = ts_struct.atoms[atom1_idx]
            atom2 = ts_struct.atoms[atom2_idx]
            
            direction = atom2.xyz - atom1.xyz
            direction_norm = direction / np.linalg.norm(direction)
            
            # 增加键长
            atom1.move(-0.2 * direction_norm)
            atom2.move(0.2 * direction_norm)
            
            self.transition_states.append(ts_struct)
            
            # 创建产物（键完全断裂）
            product = ts_struct.clone()
            product.name = f"{reactant.get('name', 'reactant')}_product"
            product.remove_bond(target_bond)
            
            self.add_product(product)
    
    def get_reaction_summary(self):
        """获取反应摘要"""
        return {
            'n_reactants': len(self.reactants),
            'n_products': len(self.products),
            'n_transition_states': len(self.transition_states),
            'total_atoms_reactants': sum(len(r.atoms) for r in self.reactants),
            'total_atoms_products': sum(len(p.atoms) for p in self.products)
        }

# 使用反应建模器
modeler = ReactionModeler()
# ... 添加反应物和建模反应
```

### 示例4：Frame高级操作

```python
def advanced_frame_operations():
    """Frame高级操作示例"""
    
    # 创建复杂的Frame数据
    complex_data = {
        'atoms': {
            'name': ['C1', 'C2', 'O1', 'H1', 'H2'],
            'element': ['C', 'C', 'O', 'H', 'H'],
            'xyz': [[0,0,0], [1.5,0,0], [0,1.5,0], [2.5,0,0], [0,2.5,0]],
            'charge': [0.0, 0.0, -0.4, 0.1, 0.1],
            'mass': [12.01, 12.01, 16.00, 1.008, 1.008]
        },
        'bonds': {
            'i': [0, 0, 1],
            'j': [1, 2, 3],
            'bond_type': ['single', 'single', 'single'],
            'length': [1.5, 1.4, 1.1]
        }
    }
    
    frame = mp.Frame(**complex_data)
    
    # 高级查询操作
    def find_atoms_by_element(frame, element):
        """按元素查找原子"""
        atoms = frame['atoms']
        mask = atoms.coords['element'] == element
        return atoms.isel(index=mask)
    
    # 找到所有碳原子
    carbon_atoms = find_atoms_by_element(frame, 'C')
    print(f"找到 {len(carbon_atoms.coords['name'])} 个碳原子")
    
    # 计算分子属性
    def calculate_molecular_properties(frame):
        """计算分子属性"""
        atoms = frame['atoms']
        
        # 分子量
        masses = atoms.coords['mass'].values
        molecular_weight = masses.sum()
        
        # 质心
        xyz = atoms.coords['xyz'].values
        center_of_mass = np.average(xyz, axis=0, weights=masses)
        
        # 电荷分布
        charges = atoms.coords['charge'].values
        total_charge = charges.sum()
        
        return {
            'molecular_weight': molecular_weight,
            'center_of_mass': center_of_mass,
            'total_charge': total_charge,
            'n_atoms': len(atoms.coords['name'])
        }
    
    properties = calculate_molecular_properties(frame)
    print(f"分子量: {properties['molecular_weight']:.2f}")
    print(f"质心: {properties['center_of_mass']}")
    
    return frame, properties
```

---

这份文档涵盖了MolPy核心数据结构的完整使用指南，从基础概念到高级扩展，再到详细的API参考和实用示例。你可以根据需要进一步完善特定部分或添加更多示例。
