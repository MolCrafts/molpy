# Complete Frame Module Tutorial

## Overview

The Frame module provides efficient tabular data storage and manipulation functionality based on xarray, specifically designed for handling molecular data batch operations and analysis. Frame is similar to pandas DataFrame but optimized for molecular data, supporting multi-dimensional data (like coordinates, tensors) and bidirectional conversion with Struct objects.

## 1. Frame Fundamentals

### 1.1 What is Frame

Frame is a container in MolPy for storing and manipulating structured molecular data, built on xarray.DataArray:

```python
import molpy as mp
import numpy as np

# Create basic Frame
atom_data = {
    'name': ['C1', 'C2', 'O1', 'H1'],
    'element': ['C', 'C', 'O', 'H'],
    'xyz': [[0,0,0], [1.5,0,0], [0,1.5,0], [2.5,0,0]],
    'charge': [0.0, 0.0, -0.4, 0.1]
}

frame = mp.Frame(atoms=atom_data)
print("Frame created successfully")
print(f"Frame keys: {list(frame.keys())}")
```

**Core Features of Frame:**
- Efficient storage based on xarray.DataArray
- Support for multi-dimensional data (coordinates, tensors, etc.)
- Automatic handling of scalar and vector data
- Bidirectional conversion with Struct objects
- Efficient array operations and broadcasting

### 1.2 Frame Structure

Frame is essentially a dictionary that can contain various types of data:

```python
# 创建复杂Frame
complex_frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'N1'],
        'element': ['C', 'C', 'N'],
        'xyz': [[0,0,0], [1.5,0,0], [0,1.5,0]],
        'mass': [12.01, 12.01, 14.01],
        'velocity': [[0.1,0,0], [0,0.1,0], [0,0,0.1]],
        'charge': [-0.1, 0.1, -0.3]
    },
    bonds={
        'i': [0, 0],  # 原子索引
        'j': [1, 2],  # 原子索引
        'bond_type': ['single', 'single'],
        'length': [1.5, 1.4],
        'strength': [350.0, 400.0]
    },
    properties={
        'energy': -123.45,
        'temperature': 298.15,
        'pressure': 1.0,
        'dipole_moment': [0.5, 0.2, 0.0]
    }
)

print(f"Frame包含的数据组: {list(complex_frame.keys())}")
```

### 1.3 数据类型处理

Frame能智能处理不同类型的数据：

```python
# 标量数据
scalar_frame = mp.Frame(
    system_properties={
        'name': 'water',
        'temperature': 298.15,
        'total_energy': -76.4
    }
)

# 向量数据
vector_frame = mp.Frame(
    atoms={
        'position': [[0,0,0], [1,0,0], [0,1,0]],  # 3D坐标
        'force': [[0.1,0,0], [0,0.1,0], [0,0,0.1]],  # 3D力
        'mass': [1.0, 1.0, 16.0]  # 标量
    }
)

# 张量数据
tensor_frame = mp.Frame(
    atoms={
        'stress_tensor': [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 3x3张量
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
        ]
    }
)

print("不同数据类型的Frame创建成功")
```

## 2. Frame操作

### 2.1 数据访问

```python
# 创建示例Frame
frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'O1', 'H1', 'H2'],
        'element': ['C', 'C', 'O', 'H', 'H'],
        'xyz': [[0,0,0], [1.5,0,0], [0,1.5,0], [-0.5,0.5,0], [-0.5,-0.5,0]],
        'charge': [0.1, -0.1, -0.4, 0.2, 0.2]
    }
)

# 访问数据组
atoms_data = frame['atoms']
print(f"原子数据类型: {type(atoms_data)}")
print(f"原子数: {len(atoms_data.coords['name'])}")

# 访问特定属性
names = atoms_data.coords['name'].values
elements = atoms_data.coords['element'].values
coordinates = atoms_data.coords['xyz'].values
charges = atoms_data.coords['charge'].values

print(f"原子名称: {names}")
print(f"元素类型: {elements}")
print(f"坐标形状: {coordinates.shape}")
print(f"电荷: {charges}")

# 访问单个原子数据
first_atom = atoms_data.isel(index=0)
print(f"第一个原子: {first_atom}")

# 访问特定原子的特定属性
first_atom_coords = atoms_data.coords['xyz'].values[0]
print(f"第一个原子坐标: {first_atom_coords}")
```

### 2.2 数据过滤和选择

```python
# 按条件过滤原子
def filter_atoms_by_element(frame, elements):
    """按元素类型过滤原子"""
    atoms = frame['atoms']
    
    # 获取元素数组
    element_array = atoms.coords['element'].values
    
    # 创建掩码
    if isinstance(elements, str):
        elements = [elements]
    
    mask = np.isin(element_array, elements)
    
    # 应用过滤
    filtered_atoms = atoms.isel(index=mask)
    
    # 创建新Frame
    new_frame = mp.Frame()
    new_frame['atoms'] = filtered_atoms
    
    return new_frame

# 筛选碳原子
carbon_frame = filter_atoms_by_element(frame, 'C')
carbon_atoms = carbon_frame['atoms']
print(f"碳原子数: {len(carbon_atoms.coords['name'])}")
print(f"碳原子名称: {carbon_atoms.coords['name'].values}")

# 筛选多种元素
heavy_atoms_frame = filter_atoms_by_element(frame, ['C', 'O'])
heavy_atoms = heavy_atoms_frame['atoms']
print(f"重原子数: {len(heavy_atoms.coords['name'])}")
```

### 2.3 数据修改和更新

```python
# 修改原子坐标
def translate_atoms(frame, translation):
    """平移所有原子"""
    atoms = frame['atoms'].copy()  # 创建副本
    
    # 获取当前坐标
    current_coords = atoms.coords['xyz'].values
    
    # 应用平移
    new_coords = current_coords + np.array(translation)
    
    # 更新坐标
    # 注意：这里需要重新构建DataArray
    new_atoms_data = {}
    for coord_name, coord_data in atoms.coords.items():
        if coord_name == 'xyz':
            new_atoms_data[coord_name] = new_coords
        else:
            new_atoms_data[coord_name] = coord_data.values
    
    # 创建新Frame
    new_frame = mp.Frame(atoms=new_atoms_data)
    return new_frame

# 平移分子
translated_frame = translate_atoms(frame, [1, 0, 0])
original_coords = frame['atoms'].coords['xyz'].values
new_coords = translated_frame['atoms'].coords['xyz'].values

print("平移前后坐标对比:")
for i in range(min(3, len(original_coords))):
    print(f"原子 {i}: {original_coords[i]} -> {new_coords[i]}")
```

### 2.4 数据分析

```python
def analyze_frame(frame):
    """分析Frame中的原子数据"""
    if 'atoms' not in frame:
        return {}
    
    atoms = frame['atoms']
    analysis = {}
    
    # 基本统计
    analysis['原子数'] = len(atoms.coords['name'])
    
    # 元素统计
    elements = atoms.coords['element'].values
    unique_elements, counts = np.unique(elements, return_counts=True)
    analysis['元素组成'] = dict(zip(unique_elements, counts))
    
    # 坐标分析
    if 'xyz' in atoms.coords:
        coords = atoms.coords['xyz'].values
        
        # 几何中心
        center = np.mean(coords, axis=0)
        analysis['几何中心'] = center
        
        # 边界框
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        analysis['边界框'] = {
            '最小坐标': min_coords,
            '最大坐标': max_coords,
            '尺寸': max_coords - min_coords
        }
        
        # 距离统计
        n_atoms = len(coords)
        distances = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances.append(dist)
        
        if distances:
            analysis['距离统计'] = {
                '最短距离': np.min(distances),
                '最长距离': np.max(distances),
                '平均距离': np.mean(distances)
            }
    
    # 电荷分析
    if 'charge' in atoms.coords:
        charges = atoms.coords['charge'].values
        analysis['电荷分析'] = {
            '总电荷': np.sum(charges),
            '电荷范围': [np.min(charges), np.max(charges)],
            '平均绝对电荷': np.mean(np.abs(charges))
        }
    
    return analysis

# 分析Frame
analysis = analyze_frame(frame)
print("\n=== Frame分析结果 ===")
for key, value in analysis.items():
    print(f"{key}: {value}")
```

## 3. Frame与Struct的转换

### 3.1 从Struct转换到Frame

```python
# 创建一个Struct
struct = mp.AtomicStructure(name="ethanol")

# 添加原子
c1 = struct.def_atom(name="C1", element="C", xyz=[0, 0, 0], charge=0.1)
c2 = struct.def_atom(name="C2", element="C", xyz=[1.5, 0, 0], charge=-0.1)
o1 = struct.def_atom(name="O", element="O", xyz=[2.5, 0, 0], charge=-0.4)
h1 = struct.def_atom(name="H1", element="H", xyz=[0.2, 0.9, 0], charge=0.1)
h2 = struct.def_atom(name="H2", element="H", xyz=[0.2, -0.9, 0], charge=0.1)
h3 = struct.def_atom(name="H3", element="H", xyz=[2.9, 0, 0], charge=0.2)

# 添加键
struct.def_bond(c1, c2, bond_type="single")
struct.def_bond(c2, o1, bond_type="single")
struct.def_bond(c1, h1, bond_type="single")
struct.def_bond(c1, h2, bond_type="single")
struct.def_bond(o1, h3, bond_type="single")

print(f"结构: {len(struct.atoms)} 原子, {len(struct.bonds)} 键")

# 将Struct转换为Frame (需要实现to_frame方法)
def struct_to_frame(struct):
    """将AtomicStructure转换为Frame"""
    # 提取原子数据
    atom_data = {}
    if struct.atoms:
        atom_data['name'] = [atom.get('name', '') for atom in struct.atoms]
        atom_data['element'] = [atom.get('element', '') for atom in struct.atoms]
        atom_data['xyz'] = [atom.xyz.tolist() for atom in struct.atoms]
        
        # 可选属性
        if any('charge' in atom for atom in struct.atoms):
            atom_data['charge'] = [atom.get('charge', 0.0) for atom in struct.atoms]
        if any('mass' in atom for atom in struct.atoms):
            atom_data['mass'] = [atom.get('mass', 0.0) for atom in struct.atoms]
    
    # 提取键数据
    bond_data = {}
    if struct.bonds:
        atom_index = {atom: i for i, atom in enumerate(struct.atoms)}
        bond_data['i'] = [atom_index[bond.atom1] for bond in struct.bonds]
        bond_data['j'] = [atom_index[bond.atom2] for bond in struct.bonds]
        bond_data['length'] = [bond.length for bond in struct.bonds]
        bond_data['bond_type'] = [bond.get('bond_type', 'unknown') for bond in struct.bonds]
    
    # 创建Frame
    frame_data = {}
    if atom_data:
        frame_data['atoms'] = atom_data
    if bond_data:
        frame_data['bonds'] = bond_data
    
    # 添加结构属性
    frame_data['properties'] = {
        'name': struct.get('name', ''),
        'atom_count': len(struct.atoms),
        'bond_count': len(struct.bonds)
    }
    
    return mp.Frame(**frame_data)

# 转换
struct_frame = struct_to_frame(struct)
print(f"转换后的Frame包含: {list(struct_frame.keys())}")

# 验证转换
atoms = struct_frame['atoms']
print(f"Frame中原子数: {len(atoms.coords['name'])}")
print(f"原子名称: {atoms.coords['name'].values}")
```

### 3.2 从Frame转换到Struct

```python
def frame_to_struct(frame, name="from_frame"):
    """将Frame转换为AtomicStructure"""
    struct = mp.AtomicStructure(name=name)
    
    # 转换原子
    if 'atoms' in frame:
        atoms_data = frame['atoms']
        atom_objects = []
        
        names = atoms_data.coords['name'].values
        elements = atoms_data.coords.get('element', ['']*len(names))
        coords = atoms_data.coords['xyz'].values
        
        for i, (name, element, xyz) in enumerate(zip(names, elements.values if hasattr(elements, 'values') else elements, coords)):
            atom_kwargs = {
                'name': str(name),
                'element': str(element),
                'xyz': xyz.tolist() if hasattr(xyz, 'tolist') else list(xyz)
            }
            
            # 添加可选属性
            if 'charge' in atoms_data.coords:
                atom_kwargs['charge'] = float(atoms_data.coords['charge'].values[i])
            if 'mass' in atoms_data.coords:
                atom_kwargs['mass'] = float(atoms_data.coords['mass'].values[i])
            
            atom = struct.def_atom(**atom_kwargs)
            atom_objects.append(atom)
    
    # 转换键
    if 'bonds' in frame:
        bonds_data = frame['bonds']
        
        i_indices = bonds_data.coords['i'].values
        j_indices = bonds_data.coords['j'].values
        
        for i, j in zip(i_indices, j_indices):
            bond_kwargs = {}
            if 'bond_type' in bonds_data.coords:
                bond_kwargs['bond_type'] = str(bonds_data.coords['bond_type'].values[list(i_indices).index(i)])
            
            struct.def_bond(atom_objects[i], atom_objects[j], **bond_kwargs)
    
    return struct

# 转换回Struct
converted_struct = frame_to_struct(struct_frame, "converted_ethanol")
print(f"转换后的结构: {len(converted_struct.atoms)} 原子, {len(converted_struct.bonds)} 键")

# 验证转换
print("原子名称对比:")
print(f"原始: {[atom.name for atom in struct.atoms]}")
print(f"转换: {[atom.name for atom in converted_struct.atoms]}")
```

## 4. 高级Frame操作

### 4.1 Frame合并

```python
def merge_frames(*frames):
    """合并多个Frame"""
    merged_data = {}
    
    for frame in frames:
        for key, data in frame.items():
            if key not in merged_data:
                merged_data[key] = []
            merged_data[key].append(data)
    
    # 处理不同类型的数据
    final_data = {}
    for key, data_list in merged_data.items():
        if key == 'atoms':
            # 合并原子数据
            combined_atom_data = {}
            for atoms_data in data_list:
                for coord_name, coord_data in atoms_data.coords.items():
                    if coord_name not in combined_atom_data:
                        combined_atom_data[coord_name] = []
                    combined_atom_data[coord_name].extend(coord_data.values.tolist())
            final_data[key] = combined_atom_data
        elif key == 'bonds':
            # 合并键数据（需要调整原子索引）
            combined_bond_data = {}
            atom_offset = 0
            
            for i, bonds_data in enumerate(data_list):
                for coord_name, coord_data in bonds_data.coords.items():
                    if coord_name not in combined_bond_data:
                        combined_bond_data[coord_name] = []
                    
                    if coord_name in ['i', 'j']:
                        # 调整原子索引
                        adjusted_indices = [idx + atom_offset for idx in coord_data.values]
                        combined_bond_data[coord_name].extend(adjusted_indices)
                    else:
                        combined_bond_data[coord_name].extend(coord_data.values.tolist())
                
                # 更新原子偏移量
                if 'atoms' in frames[i]:
                    atom_offset += len(frames[i]['atoms'].coords['name'])
            
            final_data[key] = combined_bond_data
        else:
            # 其他数据类型的合并（例如属性）
            final_data[key] = data_list[0] if data_list else {}
    
    return mp.Frame(**final_data)

# 创建两个小分子的Frame
water_frame = mp.Frame(
    atoms={
        'name': ['O', 'H1', 'H2'],
        'element': ['O', 'H', 'H'],
        'xyz': [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]]
    },
    bonds={
        'i': [0, 0],
        'j': [1, 2],
        'bond_type': ['covalent', 'covalent']
    }
)

methane_frame = mp.Frame(
    atoms={
        'name': ['C', 'H1', 'H2', 'H3', 'H4'],
        'element': ['C', 'H', 'H', 'H', 'H'],
        'xyz': [[3, 0, 0], [3.5, 0.5, 0.5], [3.5, -0.5, 0.5], [3.5, 0.5, -0.5], [3.5, -0.5, -0.5]]
    },
    bonds={
        'i': [0, 0, 0, 0],
        'j': [1, 2, 3, 4],
        'bond_type': ['covalent', 'covalent', 'covalent', 'covalent']
    }
)

# 合并Frame
combined_frame = merge_frames(water_frame, methane_frame)
print(f"合并后原子数: {len(combined_frame['atoms'].coords['name'])}")
print(f"合并后键数: {len(combined_frame['bonds'].coords['i'])}")
```

### 4.2 Frame子集操作

```python
def get_frame_subset(frame, atom_indices):
    """获取Frame的子集"""
    if 'atoms' not in frame:
        return mp.Frame()
    
    atoms_data = frame['atoms']
    
    # 提取子集原子数据
    subset_atom_data = {}
    for coord_name, coord_data in atoms_data.coords.items():
        subset_atom_data[coord_name] = [coord_data.values[i] for i in atom_indices]
    
    # 创建原子索引映射
    old_to_new_index = {old_idx: new_idx for new_idx, old_idx in enumerate(atom_indices)}
    
    # 处理键数据
    subset_data = {'atoms': subset_atom_data}
    
    if 'bonds' in frame:
        bonds_data = frame['bonds']
        i_indices = bonds_data.coords['i'].values
        j_indices = bonds_data.coords['j'].values
        
        # 找出涉及子集原子的键
        valid_bonds = []
        for bond_idx, (i, j) in enumerate(zip(i_indices, j_indices)):
            if i in atom_indices and j in atom_indices:
                valid_bonds.append(bond_idx)
        
        if valid_bonds:
            subset_bond_data = {}
            for coord_name, coord_data in bonds_data.coords.items():
                if coord_name == 'i':
                    subset_bond_data[coord_name] = [old_to_new_index[coord_data.values[idx]] for idx in valid_bonds]
                elif coord_name == 'j':
                    subset_bond_data[coord_name] = [old_to_new_index[coord_data.values[idx]] for idx in valid_bonds]
                else:
                    subset_bond_data[coord_name] = [coord_data.values[idx] for idx in valid_bonds]
            
            subset_data['bonds'] = subset_bond_data
    
    return mp.Frame(**subset_data)

# 获取前3个原子的子集
subset_frame = get_frame_subset(combined_frame, [0, 1, 2])
print(f"子集原子数: {len(subset_frame['atoms'].coords['name'])}")
print(f"子集原子名称: {subset_frame['atoms'].coords['name'].values}")

if 'bonds' in subset_frame:
    print(f"子集键数: {len(subset_frame['bonds'].coords['i'])}")
```

### 4.3 Frame数据变换

```python
def transform_frame_coordinates(frame, transformation_matrix, translation=None):
    """对Frame中的坐标应用线性变换"""
    if 'atoms' not in frame:
        return frame
    
    atoms_data = frame['atoms']
    if 'xyz' not in atoms_data.coords:
        return frame
    
    # 获取坐标
    coords = atoms_data.coords['xyz'].values
    coords_array = np.array(coords)
    
    # 应用变换
    transformed_coords = np.dot(coords_array, transformation_matrix.T)
    
    if translation is not None:
        transformed_coords += np.array(translation)
    
    # 创建新的原子数据
    new_atom_data = {}
    for coord_name, coord_data in atoms_data.coords.items():
        if coord_name == 'xyz':
            new_atom_data[coord_name] = transformed_coords.tolist()
        else:
            new_atom_data[coord_name] = coord_data.values.tolist()
    
    # 创建新Frame
    new_frame_data = {'atoms': new_atom_data}
    
    # 复制其他数据
    for key, data in frame.items():
        if key != 'atoms':
            new_frame_data[key] = data
    
    return mp.Frame(**new_frame_data)

# 创建旋转矩阵（绕z轴旋转45度）
angle = np.pi / 4
rotation_matrix = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 1]
])

# 应用变换
transformed_frame = transform_frame_coordinates(
    combined_frame, 
    rotation_matrix, 
    translation=[1, 1, 0]
)

# 比较变换前后的坐标
original_coords = combined_frame['atoms'].coords['xyz'].values
transformed_coords = transformed_frame['atoms'].coords['xyz'].values

print("坐标变换对比（前3个原子）:")
for i in range(min(3, len(original_coords))):
    print(f"原子 {i}: {original_coords[i]} -> {transformed_coords[i]}")
```

## 5. Frame性能优化

### 5.1 批量操作

```python
def efficient_frame_operations():
    """演示高效的Frame操作"""
    
    # 创建大型Frame
    n_atoms = 1000
    atom_data = {
        'name': [f'atom_{i}' for i in range(n_atoms)],
        'element': ['C'] * n_atoms,
        'xyz': np.random.random((n_atoms, 3)).tolist(),
        'charge': np.random.uniform(-0.5, 0.5, n_atoms).tolist()
    }
    
    large_frame = mp.Frame(atoms=atom_data)
    print(f"创建了包含 {n_atoms} 个原子的Frame")
    
    # 高效的坐标操作
    atoms = large_frame['atoms']
    coords = atoms.coords['xyz'].values
    coords_array = np.array(coords)
    
    # 批量平移
    translated_coords = coords_array + np.array([1, 0, 0])
    
    # 批量缩放
    scaled_coords = coords_array * 1.5
    
    # 批量旋转
    angle = np.pi / 6
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    rotated_coords = np.dot(coords_array, rotation_matrix.T)
    
    print("批量操作完成")
    
    # 创建结果Frame
    transformed_atom_data = atom_data.copy()
    transformed_atom_data['xyz'] = rotated_coords.tolist()
    
    result_frame = mp.Frame(atoms=transformed_atom_data)
    return result_frame

# 执行高效操作
import time
start_time = time.time()
result = efficient_frame_operations()
end_time = time.time()
print(f"操作耗时: {end_time - start_time:.3f} 秒")
```

### 5.2内存优化

```python
def memory_efficient_frame_processing(frame):
    """内存高效的Frame处理"""
    
    # 使用生成器处理大型数据
    def process_atoms_lazy(atoms_data):
        names = atoms_data.coords['name'].values
        coords = atoms_data.coords['xyz'].values
        
        for name, coord in zip(names, coords):
            # 处理单个原子（避免一次性加载所有数据）
            yield {
                'name': name,
                'position': coord,
                'distance_from_origin': np.linalg.norm(coord)
            }
    
    # 处理原子数据
    if 'atoms' in frame:
        atoms_data = frame['atoms']
        
        # 计算统计信息（不存储所有中间结果）
        total_distance = 0
        max_distance = 0
        count = 0
        
        for atom_info in process_atoms_lazy(atoms_data):
            total_distance += atom_info['distance_from_origin']
            max_distance = max(max_distance, atom_info['distance_from_origin'])
            count += 1
        
        avg_distance = total_distance / count if count > 0 else 0
        
        return {
            'processed_atoms': count,
            'average_distance_from_origin': avg_distance,
            'max_distance_from_origin': max_distance
        }
    
    return {}

# 应用内存优化处理
if 'result' in locals():
    stats = memory_efficient_frame_processing(result)
    print(f"内存优化处理结果: {stats}")
```

## 6. Frame应用示例

### 6.1 分子动力学轨迹分析

```python
def create_md_trajectory_frame():
    """创建分子动力学轨迹Frame"""
    
    n_atoms = 5
    n_frames = 10
    
    # 创建轨迹数据
    trajectory_data = {
        'atoms': {
            'name': ['O', 'H1', 'H2', 'Na', 'Cl'],
            'element': ['O', 'H', 'H', 'Na', 'Cl'],
            'mass': [16.0, 1.0, 1.0, 23.0, 35.5]
        },
        'trajectory': {
            'time': list(range(n_frames)),  # 时间步
            'coordinates': [],  # 每个时间步的坐标
            'velocities': [],   # 每个时间步的速度
            'forces': []        # 每个时间步的力
        }
    }
    
    # 生成轨迹数据
    for frame_idx in range(n_frames):
        # 坐标（模拟水分子和离子的运动）
        coords = [
            [0 + 0.1*frame_idx, 0, 0],                    # O
            [0.757 + 0.1*frame_idx, 0.586, 0],          # H1
            [-0.757 + 0.1*frame_idx, 0.586, 0],         # H2
            [3 + 0.05*frame_idx, 0, 0],                  # Na
            [-3 - 0.05*frame_idx, 0, 0]                  # Cl
        ]
        
        # 速度（随机）
        velocities = np.random.uniform(-0.1, 0.1, (n_atoms, 3)).tolist()
        
        # 力（随机）
        forces = np.random.uniform(-1, 1, (n_atoms, 3)).tolist()
        
        trajectory_data['trajectory']['coordinates'].append(coords)
        trajectory_data['trajectory']['velocities'].append(velocities)
        trajectory_data['trajectory']['forces'].append(forces)
    
    return mp.Frame(**trajectory_data)

# 创建MD轨迹
md_frame = create_md_trajectory_frame()
print(f"MD轨迹Frame: {list(md_frame.keys())}")

# 分析轨迹
def analyze_md_trajectory(md_frame):
    """分析MD轨迹"""
    trajectory = md_frame['trajectory']
    
    coordinates = trajectory.coords['coordinates'].values
    times = trajectory.coords['time'].values
    
    analysis = {}
    
    # 计算每个原子的平均位置
    n_frames = len(coordinates)
    n_atoms = len(coordinates[0])
    
    atom_names = md_frame['atoms'].coords['name'].values
    
    for atom_idx in range(n_atoms):
        atom_positions = [coordinates[frame][atom_idx] for frame in range(n_frames)]
        avg_position = np.mean(atom_positions, axis=0)
        
        analysis[f'{atom_names[atom_idx]}_average_position'] = avg_position.tolist()
    
    # 计算均方位移
    initial_coords = np.array(coordinates[0])
    final_coords = np.array(coordinates[-1])
    msd = np.mean(np.sum((final_coords - initial_coords)**2, axis=1))
    analysis['mean_squared_displacement'] = msd
    
    return analysis

# 分析MD轨迹
md_analysis = analyze_md_trajectory(md_frame)
print("\nMD轨迹分析:")
for key, value in md_analysis.items():
    print(f"{key}: {value}")
```

### 6.2 量子化学计算结果

```python
def create_qm_results_frame():
    """创建量子化学计算结果Frame"""
    
    # 水分子的量子化学计算结果
    qm_data = {
        'atoms': {
            'name': ['O', 'H1', 'H2'],
            'element': ['O', 'H', 'H'],
            'xyz': [[0.0, 0.0, 0.117], [0.0, 0.757, -0.464], [0.0, -0.757, -0.464]],
            'mulliken_charge': [-0.834, 0.417, 0.417],
            'esp_charge': [-0.742, 0.371, 0.371]
        },
        'molecular_orbitals': {
            'orbital_number': list(range(1, 6)),
            'energy': [-20.55, -1.35, -0.71, -0.57, 0.20],  # Hartree
            'occupancy': [2, 2, 2, 2, 0],
            'symmetry': ['A1', 'A1', 'B2', 'A1', 'A1']
        },
        'vibrational_modes': {
            'mode_number': [1, 2, 3],
            'frequency': [1595, 3657, 3756],  # cm^-1
            'intensity': [104.5, 5.1, 57.4],  # km/mol
            'symmetry': ['A1', 'A1', 'B2']
        },
        'properties': {
            'total_energy': -76.4265,  # Hartree
            'dipole_moment': [0.0, 0.0, 1.854],  # Debye
            'homo_energy': -0.57,  # Hartree
            'lumo_energy': 0.20,   # Hartree
            'homo_lumo_gap': 0.77  # Hartree
        }
    }
    
    return mp.Frame(**qm_data)

# 创建量子化学结果Frame
qm_frame = create_qm_results_frame()
print(f"量子化学Frame: {list(qm_frame.keys())}")

# 分析量子化学结果
def analyze_qm_results(qm_frame):
    """分析量子化学计算结果"""
    analysis = {}
    
    # 电荷分析
    if 'atoms' in qm_frame:
        atoms = qm_frame['atoms']
        if 'mulliken_charge' in atoms.coords:
            mulliken_charges = atoms.coords['mulliken_charge'].values
            analysis['total_mulliken_charge'] = np.sum(mulliken_charges)
            analysis['charge_separation'] = np.max(mulliken_charges) - np.min(mulliken_charges)
    
    # 分子轨道分析
    if 'molecular_orbitals' in qm_frame:
        mo = qm_frame['molecular_orbitals']
        energies = mo.coords['energy'].values
        occupancies = mo.coords['occupancy'].values
        
        # HOMO-LUMO分析
        occupied_indices = np.where(occupancies > 0)[0]
        unoccupied_indices = np.where(occupancies == 0)[0]
        
        if len(occupied_indices) > 0 and len(unoccupied_indices) > 0:
            homo_energy = energies[occupied_indices[-1]]
            lumo_energy = energies[unoccupied_indices[0]]
            analysis['homo_lumo_gap_calculated'] = lumo_energy - homo_energy
    
    # 振动分析
    if 'vibrational_modes' in qm_frame:
        vib = qm_frame['vibrational_modes']
        frequencies = vib.coords['frequency'].values
        intensities = vib.coords['intensity'].values
        
        analysis['highest_frequency'] = np.max(frequencies)
        analysis['strongest_vibration'] = {
            'frequency': frequencies[np.argmax(intensities)],
            'intensity': np.max(intensities)
        }
    
    return analysis

# 分析量子化学结果
qm_analysis = analyze_qm_results(qm_frame)
print("\n量子化学分析:")
for key, value in qm_analysis.items():
    print(f"{key}: {value}")
```

## 7. 最佳实践

### 7.1 Frame设计原则

```python
# ✓ 好的做法：清晰的数据组织
good_frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'O'],
        'element': ['C', 'C', 'O'],
        'xyz': [[0,0,0], [1,0,0], [2,0,0]],
        'properties': {
            'charge': [0.1, -0.1, -0.4],
            'mass': [12.01, 12.01, 16.0]
        }
    },
    bonds={
        'atom_pairs': [(0,1), (1,2)],
        'types': ['single', 'single'],
        'properties': {
            'length': [1.0, 1.2],
            'strength': [300, 350]
        }
    }
)

# ✗ 避免：混乱的数据结构
# bad_frame = mp.Frame(
#     mixed_data=[1, 'string', [1,2,3], {'a': 1}],  # 类型不一致
#     nested_too_deep={'a': {'b': {'c': {'d': 1}}}},  # 过度嵌套
# )
```

### 7.2 性能优化

```python
# ✓ 好的做法：批量操作
def efficient_coordinate_processing(frame):
    atoms = frame['atoms']
    coords = np.array(atoms.coords['xyz'].values)
    
    # 批量计算距离矩阵
    n_atoms = len(coords)
    distance_matrix = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            distance_matrix[i,j] = distance_matrix[j,i] = dist
    
    return distance_matrix

# ✓ 好的做法：使用向量化操作
def vectorized_operations(frame):
    atoms = frame['atoms']
    coords = np.array(atoms.coords['xyz'].values)
    
    # 向量化的中心化
    centered_coords = coords - np.mean(coords, axis=0)
    
    # 向量化的距离计算
    distances_from_origin = np.linalg.norm(coords, axis=1)
    
    return centered_coords, distances_from_origin
```

### 7.3 错误处理和验证

```python
def validate_frame(frame):
    """验证Frame的数据完整性"""
    errors = []
    
    # 检查必要的数据组
    if 'atoms' not in frame:
        errors.append("Frame缺少atoms数据组")
        return errors
    
    atoms = frame['atoms']
    
    # 检查必要的原子属性
    required_coords = ['name', 'xyz']
    for coord in required_coords:
        if coord not in atoms.coords:
            errors.append(f"atoms缺少必要属性: {coord}")
    
    # 检查数据一致性
    if 'name' in atoms.coords and 'xyz' in atoms.coords:
        n_names = len(atoms.coords['name'].values)
        n_coords = len(atoms.coords['xyz'].values)
        if n_names != n_coords:
            errors.append(f"原子名称数({n_names})与坐标数({n_coords})不匹配")
    
    # 检查坐标维度
    if 'xyz' in atoms.coords:
        coords = atoms.coords['xyz'].values
        for i, coord in enumerate(coords):
            if len(coord) != 3:
                errors.append(f"原子{i}的坐标不是3维: {coord}")
    
    # 检查键的有效性
    if 'bonds' in frame:
        bonds = frame['bonds']
        if 'i' in bonds.coords and 'j' in bonds.coords:
            n_atoms = len(atoms.coords['name'].values)
            i_indices = bonds.coords['i'].values
            j_indices = bonds.coords['j'].values
            
            for bond_idx, (i, j) in enumerate(zip(i_indices, j_indices)):
                if i < 0 or i >= n_atoms:
                    errors.append(f"键{bond_idx}的原子索引i无效: {i}")
                if j < 0 or j >= n_atoms:
                    errors.append(f"键{bond_idx}的原子索引j无效: {j}")
                if i == j:
                    errors.append(f"键{bond_idx}连接同一个原子: {i}")
    
    return errors

# 使用验证函数
validation_errors = validate_frame(good_frame)
if validation_errors:
    print("Frame验证错误:")
    for error in validation_errors:
        print(f"- {error}")
else:
    print("✓ Frame验证通过")
```

这个完整的Frame教程涵盖了从基础概念到高级应用的所有方面，包括数据处理、转换、分析和性能优化。通过这些例子，你可以有效地使用Frame来处理复杂的分子数据。
