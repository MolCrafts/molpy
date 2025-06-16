# Frame模块教程

## 概述

Frame模块提供了高效的表格式数据存储和操作功能，基于xarray构建，专门用于处理分子数据的批量操作和分析。

## 1. Frame基础

### 1.1 什么是Frame

Frame是MolPy中用于存储和操作结构化分子数据的容器，类似于pandas DataFrame但针对分子数据进行了优化：

```python
import molpy as mp
import numpy as np

# 创建基本Frame
atom_data = {
    'name': ['C1', 'C2', 'O1', 'H1'],
    'element': ['C', 'C', 'O', 'H'],
    'xyz': [[0,0,0], [1.5,0,0], [0,1.5,0], [2.5,0,0]],
    'charge': [0.0, 0.0, -0.4, 0.1]
}

frame = mp.Frame(atoms=atom_data)
print(frame['atoms'])
```

**Frame的核心特性：**
- 基于xarray.DataArray的高效存储
- 支持多维数据（坐标、张量等）
- 自动处理标量和向量数据
- 与Struct对象的双向转换

### 1.2 Frame结构

Frame本质上是一个字典，可以包含多种类型的数据：

```python
complex_frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'N1'],
        'element': ['C', 'C', 'N'],
        'xyz': [[0,0,0], [1.5,0,0], [0,1.5,0]],
        'mass': [12.01, 12.01, 14.01],
        'velocity': [[0.1,0,0], [0,0.1,0], [0,0,0.1]]
    },
    bonds={
        'i': [0, 0],
        'j': [1, 2],
        'bond_type': ['single', 'single'],
        'length': [1.5, 1.4]
    },
    properties={
        'energy': -123.45,
        'temperature': 298.15,
        'pressure': 1.0
    }
)
```

## 2. Frame操作

### 2.1 数据访问

```python
# 获取原子数据
atoms = frame['atoms']
print(f"原子数: {len(atoms.coords['name'])}")

# 访问特定属性
names = atoms.coords['name'].values
xyz_coords = atoms.coords['xyz'].values
print(f"第一个原子: {names[0]} at {xyz_coords[0]}")

# 访问单个原子
first_atom_coords = atoms.isel(index=0)
print(first_atom_coords)
```

### 2.2 数据过滤

```python
def filter_frame_by_element(frame, elements):
    """按元素过滤原子"""
    atoms = frame['atoms']
    
    # 创建掩码
    element_array = atoms.coords['element'].values
    mask = np.isin(element_array, elements)
    
    # 应用过滤
    filtered_atoms = atoms.isel(index=mask)
    
    # 创建新Frame
    new_frame = mp.Frame()
    new_frame['atoms'] = filtered_atoms
    
    return new_frame

# 只保留碳原子
carbon_frame = filter_frame_by_element(frame, ['C'])
```

### 2.3 数据转换

```python
def add_calculated_properties(frame):
    """为Frame添加计算属性"""
    atoms = frame['atoms']
    
    # 计算质心
    if 'xyz' in atoms.coords and 'mass' in atoms.coords:
        xyz = atoms.coords['xyz'].values
        mass = atoms.coords['mass'].values
        
        # 质心计算
        total_mass = mass.sum()
        center_of_mass = np.sum(xyz * mass[:, np.newaxis], axis=0) / total_mass
        
        # 添加到Frame属性
        frame.attrs = getattr(frame, 'attrs', {})
        frame.attrs['center_of_mass'] = center_of_mass
        frame.attrs['total_mass'] = total_mass
    
    # 计算分子半径
    if 'xyz' in atoms.coords:
        xyz = atoms.coords['xyz'].values
        center = xyz.mean(axis=0)
        distances = np.linalg.norm(xyz - center, axis=1)
        frame.attrs['molecular_radius'] = distances.max()
    
    return frame

# 使用函数
enhanced_frame = add_calculated_properties(frame)
print(f"质心: {enhanced_frame.attrs.get('center_of_mass')}")
```

## 3. Frame与Struct转换

### 3.1 从Struct创建Frame

```python
# 创建一个分子结构
struct = mp.AtomicStructure(name="methane")
struct.def_atom(name="C1", element="C", xyz=[0,0,0], mass=12.01)
struct.def_atom(name="H1", element="H", xyz=[1,1,1], mass=1.008)
struct.def_atom(name="H2", element="H", xyz=[-1,1,-1], mass=1.008)
struct.def_atom(name="H3", element="H", xyz=[1,-1,-1], mass=1.008)
struct.def_atom(name="H4", element="H", xyz=[-1,-1,1], mass=1.008)

# 添加化学键
for i in range(1, 5):
    struct.def_bond(struct.atoms[0], struct.atoms[i], bond_type="single")

# 转换为Frame
frame = struct.to_frame()
print("转换后的Frame:")
print(frame['atoms'])
print(frame['bonds'])
```

### 3.2 从Frame创建Struct

```python
# 从Frame重建结构
reconstructed_struct = mp.MolecularStructure.from_frame(frame, name="reconstructed")

print(f"重建的结构:")
print(f"  原子数: {len(reconstructed_struct.atoms)}")
print(f"  化学键数: {len(reconstructed_struct.bonds)}")

# 验证数据一致性
for i, atom in enumerate(reconstructed_struct.atoms):
    original_xyz = struct.atoms[i].xyz
    reconstructed_xyz = atom.xyz
    assert np.allclose(original_xyz, reconstructed_xyz), f"原子{i}坐标不匹配"
```

## 4. 高级Frame操作

### 4.1 Frame合并

```python
def merge_frames(*frames, axis='atoms'):
    """合并多个Frame"""
    if not frames:
        return mp.Frame()
    
    merged_data = {}
    
    for key in frames[0].keys():
        if key == axis:
            # 合并指定轴的数据
            arrays_to_merge = []
            for frame in frames:
                if key in frame:
                    arrays_to_merge.append(frame[key])
            
            if arrays_to_merge:
                # 使用xarray的concat功能
                import xarray as xr
                merged_array = xr.concat(arrays_to_merge, dim='index')
                merged_data[key] = merged_array
        else:
            # 其他数据直接复制第一个frame的
            merged_data[key] = frames[0][key]
    
    return mp.Frame(**merged_data)

# 创建多个Frame并合并
frame1 = mp.Frame(atoms={'name': ['C1'], 'element': ['C'], 'xyz': [[0,0,0]]})
frame2 = mp.Frame(atoms={'name': ['H1'], 'element': ['H'], 'xyz': [[1,0,0]]})

merged = merge_frames(frame1, frame2)
print(f"合并后原子数: {len(merged['atoms'].coords['name'])}")
```

### 4.2 Frame切片和子集

```python
def extract_subframe(frame, atom_indices):
    """提取Frame的子集"""
    new_frame = mp.Frame()
    
    if 'atoms' in frame:
        atoms = frame['atoms']
        # 使用xarray的isel进行索引选择
        subset_atoms = atoms.isel(index=atom_indices)
        new_frame['atoms'] = subset_atoms
    
    # 处理bonds数据，只保留涉及选定原子的键
    if 'bonds' in frame:
        bonds = frame['bonds']
        
        # 获取键的原子索引
        i_indices = bonds.coords['i'].values
        j_indices = bonds.coords['j'].values
        
        # 找到涉及选定原子的键
        valid_bonds = []
        for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
            if i in atom_indices and j in atom_indices:
                valid_bonds.append(idx)
        
        if valid_bonds:
            subset_bonds = bonds.isel(index=valid_bonds)
            new_frame['bonds'] = subset_bonds
    
    return new_frame

# 提取前3个原子的子Frame
subset_frame = extract_subframe(frame, [0, 1, 2])
```

### 4.3 Frame变换

```python
class FrameTransformer:
    """Frame变换工具类"""
    
    @staticmethod
    def translate(frame, vector):
        """平移所有原子"""
        if 'atoms' not in frame:
            return frame
        
        atoms = frame['atoms']
        if 'xyz' not in atoms.coords:
            return frame
        
        # 获取坐标数据
        xyz = atoms.coords['xyz'].values.copy()
        
        # 应用平移
        xyz += np.array(vector)
        
        # 创建新的atoms数据
        new_coords = atoms.coords.copy()
        new_coords['xyz'] = (['index', 'spatial'], xyz)
        
        # 创建新Frame
        new_frame = frame.copy()
        new_frame['atoms'] = atoms._constructor(
            data=atoms.data,
            coords=new_coords,
            dims=atoms.dims,
            name=atoms.name
        )
        
        return new_frame
    
    @staticmethod
    def rotate(frame, angle, axis, center=None):
        """旋转所有原子"""
        from scipy.spatial.transform import Rotation
        
        if 'atoms' not in frame:
            return frame
        
        atoms = frame['atoms']
        if 'xyz' not in atoms.coords:
            return frame
        
        xyz = atoms.coords['xyz'].values.copy()
        
        # 设置旋转中心
        if center is None:
            center = xyz.mean(axis=0)
        
        # 平移到原点
        xyz_centered = xyz - center
        
        # 创建旋转对象
        rotation = Rotation.from_rotvec(angle * np.array(axis))
        
        # 应用旋转
        xyz_rotated = rotation.apply(xyz_centered)
        
        # 平移回去
        xyz_final = xyz_rotated + center
        
        # 更新Frame
        new_coords = atoms.coords.copy()
        new_coords['xyz'] = (['index', 'spatial'], xyz_final)
        
        new_frame = frame.copy()
        new_frame['atoms'] = atoms._constructor(
            data=atoms.data,
            coords=new_coords,
            dims=atoms.dims,
            name=atoms.name
        )
        
        return new_frame

# 使用变换
translated_frame = FrameTransformer.translate(frame, [1.0, 0.0, 0.0])
rotated_frame = FrameTransformer.rotate(frame, np.pi/2, [0, 0, 1])
```

## 5. Frame分析功能

### 5.1 几何分析

```python
class FrameAnalyzer:
    """Frame分析工具"""
    
    @staticmethod
    def calculate_bonds_lengths(frame):
        """计算所有键长"""
        if 'atoms' not in frame or 'bonds' not in frame:
            return []
        
        atoms = frame['atoms']
        bonds = frame['bonds']
        
        xyz = atoms.coords['xyz'].values
        i_indices = bonds.coords['i'].values
        j_indices = bonds.coords['j'].values
        
        lengths = []
        for i, j in zip(i_indices, j_indices):
            length = np.linalg.norm(xyz[i] - xyz[j])
            lengths.append(length)
        
        return np.array(lengths)
    
    @staticmethod
    def calculate_angles(frame, angle_triplets):
        """计算角度"""
        if 'atoms' not in frame:
            return []
        
        xyz = frame['atoms'].coords['xyz'].values
        angles = []
        
        for i, j, k in angle_triplets:
            # 向量
            v1 = xyz[i] - xyz[j]
            v2 = xyz[k] - xyz[j]
            
            # 角度计算
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        return np.array(angles)
    
    @staticmethod
    def calculate_center_of_mass(frame):
        """计算质心"""
        if 'atoms' not in frame:
            return None
        
        atoms = frame['atoms']
        
        if 'xyz' not in atoms.coords or 'mass' not in atoms.coords:
            return None
        
        xyz = atoms.coords['xyz'].values
        mass = atoms.coords['mass'].values
        
        total_mass = mass.sum()
        center_of_mass = np.sum(xyz * mass[:, np.newaxis], axis=0) / total_mass
        
        return center_of_mass
    
    @staticmethod
    def calculate_inertia_tensor(frame):
        """计算惯性张量"""
        if 'atoms' not in frame:
            return None
        
        atoms = frame['atoms']
        
        if 'xyz' not in atoms.coords or 'mass' not in atoms.coords:
            return None
        
        xyz = atoms.coords['xyz'].values
        mass = atoms.coords['mass'].values
        
        # 计算质心
        com = FrameAnalyzer.calculate_center_of_mass(frame)
        
        # 相对于质心的坐标
        xyz_centered = xyz - com
        
        # 惯性张量计算
        I = np.zeros((3, 3))
        for i, (pos, m) in enumerate(zip(xyz_centered, mass)):
            x, y, z = pos
            I[0, 0] += m * (y*y + z*z)
            I[1, 1] += m * (x*x + z*z)
            I[2, 2] += m * (x*x + y*y)
            I[0, 1] -= m * x * y
            I[0, 2] -= m * x * z
            I[1, 2] -= m * y * z
        
        # 对称化
        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]
        
        return I

# 使用分析功能
analyzer = FrameAnalyzer()

# 计算键长
bond_lengths = analyzer.calculate_bonds_lengths(frame)
print(f"键长: {bond_lengths}")

# 计算质心
com = analyzer.calculate_center_of_mass(frame)
print(f"质心: {com}")

# 计算惯性张量
inertia = analyzer.calculate_inertia_tensor(frame)
if inertia is not None:
    print(f"惯性张量:\n{inertia}")
```

### 5.2 统计分析

```python
def analyze_frame_statistics(frame):
    """Frame统计分析"""
    stats = {}
    
    if 'atoms' in frame:
        atoms = frame['atoms']
        
        # 原子数统计
        n_atoms = len(atoms.coords['name'])
        stats['n_atoms'] = n_atoms
        
        # 元素统计
        if 'element' in atoms.coords:
            elements = atoms.coords['element'].values
            unique, counts = np.unique(elements, return_counts=True)
            stats['element_counts'] = dict(zip(unique, counts))
        
        # 坐标统计
        if 'xyz' in atoms.coords:
            xyz = atoms.coords['xyz'].values
            stats['coord_range'] = {
                'x': [xyz[:, 0].min(), xyz[:, 0].max()],
                'y': [xyz[:, 1].min(), xyz[:, 1].max()],
                'z': [xyz[:, 2].min(), xyz[:, 2].max()]
            }
            stats['coord_center'] = xyz.mean(axis=0)
            stats['coord_std'] = xyz.std(axis=0)
        
        # 质量统计
        if 'mass' in atoms.coords:
            masses = atoms.coords['mass'].values
            stats['total_mass'] = masses.sum()
            stats['avg_mass'] = masses.mean()
    
    if 'bonds' in frame:
        bonds = frame['bonds']
        stats['n_bonds'] = len(bonds.coords['i'])
        
        # 键类型统计
        if 'bond_type' in bonds.coords:
            bond_types = bonds.coords['bond_type'].values
            unique, counts = np.unique(bond_types, return_counts=True)
            stats['bond_type_counts'] = dict(zip(unique, counts))
    
    return stats

# 使用统计分析
stats = analyze_frame_statistics(frame)
print("Frame统计:")
for key, value in stats.items():
    print(f"  {key}: {value}")
```

## 6. Frame的I/O操作

### 6.1 保存和加载Frame

```python
def save_frame_to_file(frame, filename, format='json'):
    """保存Frame到文件"""
    if format == 'json':
        import json
        
        # 将Frame转换为可序列化的格式
        serializable_data = {}
        
        for key, value in frame.items():
            if hasattr(value, 'to_dict'):
                serializable_data[key] = value.to_dict()
            else:
                # 处理xarray DataArray
                coords_dict = {}
                for coord_name, coord_data in value.coords.items():
                    coords_dict[coord_name] = coord_data.values.tolist()
                
                serializable_data[key] = {
                    'coords': coords_dict,
                    'dims': value.dims,
                    'name': value.name
                }
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    elif format == 'pickle':
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(frame, f)

def load_frame_from_file(filename, format='json'):
    """从文件加载Frame"""
    if format == 'json':
        import json
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # 重建Frame
        frame_data = {}
        for key, value in data.items():
            if 'coords' in value:
                # 重建xarray DataArray
                coords = {}
                for coord_name, coord_values in value['coords'].items():
                    coords[coord_name] = coord_values
                
                frame_data[key] = coords
            else:
                frame_data[key] = value
        
        return mp.Frame(**frame_data)
    
    elif format == 'pickle':
        import pickle
        with open(filename, 'rb') as f:
            return pickle.load(f)

# 使用I/O功能
save_frame_to_file(frame, 'my_structure.json')
loaded_frame = load_frame_from_file('my_structure.json')
```

## 7. 性能优化

### 7.1 大型Frame处理

```python
def process_large_frame_efficiently(frame, chunk_size=1000):
    """高效处理大型Frame"""
    if 'atoms' not in frame:
        return
    
    atoms = frame['atoms']
    n_atoms = len(atoms.coords['name'])
    
    # 分块处理
    for start_idx in range(0, n_atoms, chunk_size):
        end_idx = min(start_idx + chunk_size, n_atoms)
        
        # 获取当前块
        chunk_atoms = atoms.isel(index=slice(start_idx, end_idx))
        
        # 处理当前块
        # 这里可以进行任何计算
        chunk_xyz = chunk_atoms.coords['xyz'].values
        
        # 示例：计算到原点的距离
        distances = np.linalg.norm(chunk_xyz, axis=1)
        
        print(f"块 {start_idx}-{end_idx}: 平均距离 = {distances.mean():.3f}")

# 处理大型Frame
process_large_frame_efficiently(frame)
```

### 7.2 内存优化

```python
def memory_efficient_frame_operations(frame):
    """内存优化的Frame操作"""
    # 使用视图而不是复制
    atoms = frame['atoms']
    
    # 原地操作
    xyz = atoms.coords['xyz']
    
    # 避免不必要的数组复制
    # 不好的做法：
    # new_xyz = xyz.values.copy()
    
    # 好的做法：使用引用
    xyz_ref = xyz.values
    
    # 对于大型数组，使用生成器
    def coordinate_generator(atoms):
        """坐标生成器"""
        xyz = atoms.coords['xyz'].values
        for coord in xyz:
            yield coord
    
    # 使用生成器处理坐标
    for coord in coordinate_generator(atoms):
        # 处理单个坐标
        distance = np.linalg.norm(coord)
```

## 8. 实际应用示例

### 8.1 分子动力学轨迹分析

```python
def analyze_md_trajectory(trajectory_frames):
    """分析分子动力学轨迹"""
    n_frames = len(trajectory_frames)
    
    # 收集所有帧的质心
    centers_of_mass = []
    
    for frame in trajectory_frames:
        com = FrameAnalyzer.calculate_center_of_mass(frame)
        if com is not None:
            centers_of_mass.append(com)
    
    centers_of_mass = np.array(centers_of_mass)
    
    # 计算质心的运动轨迹
    if len(centers_of_mass) > 1:
        displacements = np.diff(centers_of_mass, axis=0)
        distances = np.linalg.norm(displacements, axis=1)
        
        analysis = {
            'n_frames': n_frames,
            'com_trajectory': centers_of_mass,
            'total_displacement': np.linalg.norm(centers_of_mass[-1] - centers_of_mass[0]),
            'avg_step_size': distances.mean(),
            'max_step_size': distances.max()
        }
        
        return analysis
    
    return {'n_frames': n_frames}

# 使用轨迹分析
trajectory = [frame]  # 实际应用中会有多个frame
trajectory_analysis = analyze_md_trajectory(trajectory)
```

这个Frame教程提供了全面的使用指南，从基础概念到高级应用，涵盖了数据操作、分析、优化等各个方面。你可以根据具体需求选择相应的部分进行学习和应用。
