# MolPy核心模块示例集合

这个文档包含了MolPy核心模块的实用示例，展示了常见的使用场景和最佳实践。

## 目录

1. [基础分子构建](#基础分子构建)
2. [蛋白质结构处理](#蛋白质结构处理)
3. [分子动力学分析](#分子动力学分析)
4. [Frame高级操作](#frame高级操作)
5. [自定义扩展](#自定义扩展)
6. [性能优化示例](#性能优化示例)

---

## 基础分子构建

### 示例1：构建简单分子

```python
import molpy as mp
import numpy as np

def build_water_molecule():
    """构建水分子"""
    # 创建分子结构
    water = mp.AtomicStructure(name="water")
    
    # 添加原子
    o_atom = water.def_atom(
        name="O1",
        element="O",
        xyz=[0.0, 0.0, 0.0],
        charge=-0.834,
        mass=15.999
    )
    
    h1_atom = water.def_atom(
        name="H1",
        element="H",
        xyz=[0.757, 0.586, 0.0],
        charge=0.417,
        mass=1.008
    )
    
    h2_atom = water.def_atom(
        name="H2",
        element="H",
        xyz=[-0.757, 0.586, 0.0],
        charge=0.417,
        mass=1.008
    )
    
    # 添加化学键
    water.def_bond(o_atom, h1_atom, bond_type="single", length=0.96)
    water.def_bond(o_atom, h2_atom, bond_type="single", length=0.96)
    
    # 添加角度
    angle = mp.Angle(h1_atom, o_atom, h2_atom, angle_type="HOH")
    water.add_angle(angle)
    
    print(f"水分子构建完成:")
    print(f"  原子数: {len(water.atoms)}")
    print(f"  化学键数: {len(water.bonds)}")
    print(f"  角度数: {len(water.angles)}")
    print(f"  H-O-H角度: {np.degrees(angle.value):.1f}°")
    
    return water

water = build_water_molecule()
```

### 示例2：构建烷烃链

```python
def build_alkane_chain(n_carbons, name=None):
    """构建烷烃链"""
    if name is None:
        name = f"C{n_carbons}H{2*n_carbons+2}"
    
    alkane = mp.AtomicStructure(name=name)
    
    # 碳原子间距
    cc_distance = 1.54  # Å
    
    # 添加碳原子
    carbon_atoms = []
    for i in range(n_carbons):
        carbon = alkane.def_atom(
            name=f"C{i+1}",
            element="C",
            xyz=[i * cc_distance, 0.0, 0.0],
            hybridization="sp3"
        )
        carbon_atoms.append(carbon)
    
    # 添加C-C键
    for i in range(n_carbons - 1):
        alkane.def_bond(
            carbon_atoms[i], 
            carbon_atoms[i+1], 
            bond_type="single",
            length=cc_distance
        )
    
    # 添加氢原子
    ch_distance = 1.09  # Å
    
    for i, carbon in enumerate(carbon_atoms):
        # 确定氢原子数量
        if i == 0 or i == n_carbons - 1:  # 末端碳
            n_hydrogens = 3
        else:  # 中间碳
            n_hydrogens = 2
        
        # 添加氢原子
        for j in range(n_hydrogens):
            # 简化的氢原子位置计算
            angle = (2 * np.pi * j) / n_hydrogens
            if i == 0:  # 第一个碳
                h_pos = [
                    carbon.xyz[0] - ch_distance * np.cos(angle),
                    carbon.xyz[1] + ch_distance * np.sin(angle),
                    carbon.xyz[2]
                ]
            elif i == n_carbons - 1:  # 最后一个碳
                h_pos = [
                    carbon.xyz[0] + ch_distance * np.cos(angle),
                    carbon.xyz[1] + ch_distance * np.sin(angle),
                    carbon.xyz[2]
                ]
            else:  # 中间碳
                h_pos = [
                    carbon.xyz[0],
                    carbon.xyz[1] + ch_distance * np.cos(angle),
                    carbon.xyz[2] + ch_distance * np.sin(angle)
                ]
            
            hydrogen = alkane.def_atom(
                name=f"H{i+1}_{j+1}",
                element="H",
                xyz=h_pos
            )
            
            # 添加C-H键
            alkane.def_bond(carbon, hydrogen, bond_type="single", length=ch_distance)
    
    print(f"烷烃链 {name} 构建完成:")
    print(f"  原子数: {len(alkane.atoms)}")
    print(f"  化学键数: {len(alkane.bonds)}")
    
    return alkane

# 构建丙烷
propane = build_alkane_chain(3, "propane")
```

### 示例3：分子几何优化

```python
def simple_geometry_optimization(molecule, max_iterations=100, tolerance=1e-6):
    """简单的几何优化（仅作示例）"""
    
    def calculate_energy(mol):
        """计算简单的Lennard-Jones能量"""
        energy = 0.0
        for i, atom1 in enumerate(mol.atoms):
            for j, atom2 in enumerate(mol.atoms[i+1:], i+1):
                r = atom1.distance_to(atom2)
                if r > 0:
                    # 简化的LJ势能
                    sigma = 3.4  # Å
                    epsilon = 0.1  # kcal/mol
                    r6 = (sigma / r) ** 6
                    energy += 4 * epsilon * (r6 * r6 - r6)
        return energy
    
    def calculate_forces(mol):
        """计算原子间力"""
        forces = np.zeros((len(mol.atoms), 3))
        
        for i, atom1 in enumerate(mol.atoms):
            for j, atom2 in enumerate(mol.atoms):
                if i != j:
                    r_vec = atom2.xyz - atom1.xyz
                    r = np.linalg.norm(r_vec)
                    
                    if r > 0:
                        # LJ力计算
                        sigma = 3.4
                        epsilon = 0.1
                        r6 = (sigma / r) ** 6
                        force_mag = 24 * epsilon * (2 * r6 * r6 - r6) / r
                        forces[i] += force_mag * r_vec / r
        
        return forces
    
    # 优化循环
    step_size = 0.01
    prev_energy = calculate_energy(molecule)
    
    for iteration in range(max_iterations):
        forces = calculate_forces(molecule)
        
        # 更新原子位置
        for i, atom in enumerate(molecule.atoms):
            atom.xyz = atom.xyz + step_size * forces[i]
        
        # 计算新能量
        current_energy = calculate_energy(molecule)
        
        # 检查收敛
        if abs(current_energy - prev_energy) < tolerance:
            print(f"几何优化在第 {iteration+1} 步收敛")
            break
        
        prev_energy = current_energy
        
        if iteration % 10 == 0:
            print(f"第 {iteration} 步: 能量 = {current_energy:.6f}")
    
    return molecule

# 优化水分子几何
optimized_water = simple_geometry_optimization(water.clone())
```

---

## 蛋白质结构处理

### 示例4：构建简单多肽

```python
def build_peptide(sequence, name="peptide"):
    """构建简单多肽"""
    
    # 氨基酸骨架原子坐标模板
    backbone_template = {
        'N': [0.0, 0.0, 0.0],
        'CA': [1.458, 0.0, 0.0],
        'C': [2.009, 1.421, 0.0],
        'O': [1.251, 2.420, 0.0]
    }
    
    # 创建蛋白质结构
    protein = mp.AtomicStructure(name=name)
    
    # 残基间距离
    residue_spacing = 3.8  # Å
    
    for i, aa in enumerate(sequence):
        # 创建残基
        residue = mp.AtomicStructure(name=f"{aa}_{i+1}")
        residue.residue_type = aa
        residue.residue_number = i + 1
        
        # 添加骨架原子
        for atom_name, coord in backbone_template.items():
            adjusted_coord = [
                coord[0] + i * residue_spacing,
                coord[1],
                coord[2]
            ]
            
            atom = residue.def_atom(
                name=atom_name,
                element=atom_name[0],  # 简化：用第一个字符作为元素
                xyz=adjusted_coord,
                residue=aa,
                residue_number=i + 1,
                atom_type=atom_name
            )
        
        # 添加骨架键
        residue.def_bond(
            residue.atoms[0],  # N
            residue.atoms[1],  # CA
            bond_type="single"
        )
        residue.def_bond(
            residue.atoms[1],  # CA
            residue.atoms[2],  # C
            bond_type="single"
        )
        residue.def_bond(
            residue.atoms[2],  # C
            residue.atoms[3],  # O
            bond_type="double"
        )
        
        # 将残基添加到蛋白质
        protein.add_child(residue)
        protein.add_atoms(residue.atoms)
        protein.add_bonds(residue.bonds)
        
        # 添加肽键（除了第一个残基）
        if i > 0:
            prev_residue = protein.children[i-1]
            prev_c = prev_residue.atoms[2]  # 前一个残基的C
            curr_n = residue.atoms[0]      # 当前残基的N
            
            peptide_bond = mp.Bond(prev_c, curr_n, bond_type="peptide")
            protein.add_bond(peptide_bond)
    
    print(f"多肽 {name} 构建完成:")
    print(f"  序列: {sequence}")
    print(f"  残基数: {len(protein.children)}")
    print(f"  原子数: {len(protein.atoms)}")
    print(f"  化学键数: {len(protein.bonds)}")
    
    return protein

# 构建三肽
tripeptide = build_peptide("ALA-GLY-VAL", "tripeptide")
```

### 示例5：蛋白质结构分析

```python
def analyze_protein_structure(protein):
    """分析蛋白质结构"""
    
    analysis = {
        'basic_info': {},
        'geometry': {},
        'secondary_structure': {},
        'residue_info': []
    }
    
    # 基本信息
    analysis['basic_info'] = {
        'name': protein.get('name', 'unknown'),
        'n_residues': len(protein.children),
        'n_atoms': len(protein.atoms),
        'n_bonds': len(protein.bonds)
    }
    
    # 几何分析
    if len(protein.atoms) > 0:
        coords = protein.xyz
        
        # 几何中心
        geometric_center = coords.mean(axis=0)
        
        # 计算回旋半径
        distances = np.linalg.norm(coords - geometric_center, axis=1)
        radius_of_gyration = np.sqrt(np.mean(distances**2))
        
        # 坐标范围
        coord_ranges = {
            'x': [coords[:, 0].min(), coords[:, 0].max()],
            'y': [coords[:, 1].min(), coords[:, 1].max()],
            'z': [coords[:, 2].min(), coords[:, 2].max()]
        }
        
        analysis['geometry'] = {
            'geometric_center': geometric_center.tolist(),
            'radius_of_gyration': radius_of_gyration,
            'coordinate_ranges': coord_ranges,
            'max_dimension': max([r[1] - r[0] for r in coord_ranges.values()])
        }
    
    # 残基分析
    for i, residue in enumerate(protein.children):
        residue_info = {
            'number': i + 1,
            'type': residue.get('residue_type', 'UNK'),
            'n_atoms': len(residue.atoms),
            'backbone_atoms': [],
            'sidechain_atoms': []
        }
        
        # 分类原子
        for atom in residue.atoms:
            atom_name = atom.get('name', '')
            if atom_name in ['N', 'CA', 'C', 'O']:
                residue_info['backbone_atoms'].append(atom_name)
            else:
                residue_info['sidechain_atoms'].append(atom_name)
        
        analysis['residue_info'].append(residue_info)
    
    # 简单的二级结构预测（基于距离）
    if len(protein.children) >= 3:
        alpha_helices = []
        beta_sheets = []
        
        for i in range(len(protein.children) - 2):
            # 检查连续三个残基的CA原子距离
            try:
                ca1 = protein.children[i].get_atom_by(lambda a: a.get('name') == 'CA')
                ca2 = protein.children[i+1].get_atom_by(lambda a: a.get('name') == 'CA')
                ca3 = protein.children[i+2].get_atom_by(lambda a: a.get('name') == 'CA')
                
                if ca1 and ca2 and ca3:
                    d12 = ca1.distance_to(ca2)
                    d23 = ca2.distance_to(ca3)
                    d13 = ca1.distance_to(ca3)
                    
                    # 简化的二级结构判断
                    if 3.6 <= d13 <= 4.2:  # α螺旋特征距离
                        alpha_helices.append((i, i+2))
                    elif d13 > 6.0:  # β片层特征距离
                        beta_sheets.append((i, i+2))
            except:
                continue
        
        analysis['secondary_structure'] = {
            'alpha_helices': alpha_helices,
            'beta_sheets': beta_sheets,
            'helix_content': len(alpha_helices) / max(1, len(protein.children) - 2),
            'sheet_content': len(beta_sheets) / max(1, len(protein.children) - 2)
        }
    
    return analysis

# 分析蛋白质
protein_analysis = analyze_protein_structure(tripeptide)

print("蛋白质结构分析:")
print(f"  名称: {protein_analysis['basic_info']['name']}")
print(f"  残基数: {protein_analysis['basic_info']['n_residues']}")
print(f"  原子数: {protein_analysis['basic_info']['n_atoms']}")

if 'geometry' in protein_analysis:
    geom = protein_analysis['geometry']
    print(f"  几何中心: {geom['geometric_center']}")
    print(f"  回旋半径: {geom['radius_of_gyration']:.2f} Å")
```

---

## 分子动力学分析

### 示例6：轨迹数据处理

```python
def process_md_trajectory(trajectory_frames, analysis_functions=None):
    """处理分子动力学轨迹"""
    
    if analysis_functions is None:
        analysis_functions = [
            'center_of_mass',
            'radius_of_gyration',
            'end_to_end_distance'
        ]
    
    results = {func: [] for func in analysis_functions}
    results['frame_numbers'] = []
    results['time'] = []
    
    for i, frame in enumerate(trajectory_frames):
        results['frame_numbers'].append(i)
        results['time'].append(i * 0.002)  # 假设2fs时间步
        
        # 质心计算
        if 'center_of_mass' in analysis_functions:
            if 'atoms' in frame:
                atoms = frame['atoms']
                if 'xyz' in atoms.coords and 'mass' in atoms.coords:
                    xyz = atoms.coords['xyz'].values
                    mass = atoms.coords['mass'].values
                    total_mass = mass.sum()
                    com = np.sum(xyz * mass[:, np.newaxis], axis=0) / total_mass
                    results['center_of_mass'].append(com)
                else:
                    results['center_of_mass'].append(None)
            else:
                results['center_of_mass'].append(None)
        
        # 回旋半径计算
        if 'radius_of_gyration' in analysis_functions:
            if 'atoms' in frame:
                atoms = frame['atoms']
                if 'xyz' in atoms.coords:
                    xyz = atoms.coords['xyz'].values
                    center = xyz.mean(axis=0)
                    distances = np.linalg.norm(xyz - center, axis=1)
                    rg = np.sqrt(np.mean(distances**2))
                    results['radius_of_gyration'].append(rg)
                else:
                    results['radius_of_gyration'].append(None)
            else:
                results['radius_of_gyration'].append(None)
        
        # 端到端距离（对于聚合物链）
        if 'end_to_end_distance' in analysis_functions:
            if 'atoms' in frame:
                atoms = frame['atoms']
                if 'xyz' in atoms.coords and len(atoms.coords['xyz']) >= 2:
                    xyz = atoms.coords['xyz'].values
                    end_to_end = np.linalg.norm(xyz[0] - xyz[-1])
                    results['end_to_end_distance'].append(end_to_end)
                else:
                    results['end_to_end_distance'].append(None)
            else:
                results['end_to_end_distance'].append(None)
    
    return results

def calculate_autocorrelation(data, max_lag=None):
    """计算自相关函数"""
    data = np.array(data)
    n = len(data)
    
    if max_lag is None:
        max_lag = n // 4
    
    # 去除平均值
    data_centered = data - np.mean(data)
    
    autocorr = []
    for lag in range(max_lag):
        if lag == 0:
            autocorr.append(1.0)
        else:
            correlation = np.corrcoef(data_centered[:-lag], data_centered[lag:])[0, 1]
            autocorr.append(correlation if not np.isnan(correlation) else 0.0)
    
    return np.array(autocorr)

# 创建示例轨迹数据
def create_sample_trajectory(n_frames=100):
    """创建示例轨迹数据"""
    trajectory = []
    
    for i in range(n_frames):
        # 模拟原子在热运动中的位置变化
        n_atoms = 10
        base_coords = np.linspace([0, 0, 0], [9, 0, 0], n_atoms)
        
        # 添加随机热运动
        thermal_motion = np.random.normal(0, 0.1, (n_atoms, 3))
        coords = base_coords + thermal_motion
        
        frame_data = {
            'atoms': {
                'name': [f'C{j+1}' for j in range(n_atoms)],
                'element': ['C'] * n_atoms,
                'xyz': coords.tolist(),
                'mass': [12.01] * n_atoms
            }
        }
        
        frame = mp.Frame(**frame_data)
        trajectory.append(frame)
    
    return trajectory

# 处理示例轨迹
sample_trajectory = create_sample_trajectory(100)
trajectory_results = process_md_trajectory(sample_trajectory)

print("轨迹分析结果:")
print(f"  帧数: {len(trajectory_results['frame_numbers'])}")
print(f"  平均回旋半径: {np.mean([r for r in trajectory_results['radius_of_gyration'] if r is not None]):.3f} Å")
print(f"  平均端到端距离: {np.mean([d for d in trajectory_results['end_to_end_distance'] if d is not None]):.3f} Å")

# 计算回旋半径的自相关函数
rg_data = [r for r in trajectory_results['radius_of_gyration'] if r is not None]
if len(rg_data) > 10:
    rg_autocorr = calculate_autocorrelation(rg_data, max_lag=20)
    print(f"  回旋半径自相关函数在lag=10时的值: {rg_autocorr[10]:.3f}")
```

---

## Frame高级操作

### 示例7：Frame数据挖掘

```python
class FrameDataMiner:
    """Frame数据挖掘工具"""
    
    @staticmethod
    def find_clusters(frame, distance_threshold=3.0):
        """基于距离的原子聚类"""
        if 'atoms' not in frame:
            return []
        
        atoms = frame['atoms']
        if 'xyz' not in atoms.coords:
            return []
        
        coords = atoms.coords['xyz'].values
        n_atoms = len(coords)
        
        # 计算距离矩阵
        distance_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # 简单的聚类算法
        clusters = []
        visited = set()
        
        for i in range(n_atoms):
            if i in visited:
                continue
            
            cluster = [i]
            queue = [i]
            visited.add(i)
            
            while queue:
                current = queue.pop(0)
                for j in range(n_atoms):
                    if j not in visited and distance_matrix[current, j] <= distance_threshold:
                        cluster.append(j)
                        queue.append(j)
                        visited.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    @staticmethod
    def find_binding_sites(frame, probe_radius=1.5):
        """寻找可能的结合位点"""
        if 'atoms' not in frame:
            return []
        
        atoms = frame['atoms']
        if 'xyz' not in atoms.coords:
            return []
        
        coords = atoms.coords['xyz'].values
        
        # 创建网格
        min_coords = coords.min(axis=0) - probe_radius
        max_coords = coords.max(axis=0) + probe_radius
        
        grid_size = 1.0  # Å
        grid_points = []
        
        x_points = np.arange(min_coords[0], max_coords[0], grid_size)
        y_points = np.arange(min_coords[1], max_coords[1], grid_size)
        z_points = np.arange(min_coords[2], max_coords[2], grid_size)
        
        binding_sites = []
        
        for x in x_points:
            for y in y_points:
                for z in z_points:
                    test_point = np.array([x, y, z])
                    
                    # 检查是否与任何原子过近
                    min_distance = float('inf')
                    for coord in coords:
                        dist = np.linalg.norm(test_point - coord)
                        min_distance = min(min_distance, dist)
                    
                    # 如果距离在合理范围内，认为是潜在结合位点
                    if probe_radius <= min_distance <= probe_radius * 2:
                        binding_sites.append(test_point)
        
        return binding_sites
    
    @staticmethod
    def analyze_surface_area(frame, probe_radius=1.4):
        """简化的表面积分析"""
        if 'atoms' not in frame:
            return 0.0
        
        atoms = frame['atoms']
        if 'xyz' not in atoms.coords:
            return 0.0
        
        coords = atoms.coords['xyz'].values
        
        # 简化计算：使用凸包估算表面积
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords)
            return hull.area
        except ImportError:
            # 如果没有scipy，使用简化方法
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            dimensions = max_coords - min_coords
            
            # 估算为长方体表面积
            return 2 * (dimensions[0] * dimensions[1] + 
                       dimensions[1] * dimensions[2] + 
                       dimensions[0] * dimensions[2])

# 使用数据挖掘工具
def analyze_molecular_structure(frame):
    """综合分子结构分析"""
    miner = FrameDataMiner()
    
    analysis = {}
    
    # 聚类分析
    clusters = miner.find_clusters(frame, distance_threshold=2.0)
    analysis['clusters'] = {
        'n_clusters': len(clusters),
        'cluster_sizes': [len(cluster) for cluster in clusters],
        'largest_cluster': max(len(cluster) for cluster in clusters) if clusters else 0
    }
    
    # 结合位点分析
    binding_sites = miner.find_binding_sites(frame, probe_radius=1.5)
    analysis['binding_sites'] = {
        'n_sites': len(binding_sites),
        'sites': binding_sites[:10]  # 只保存前10个
    }
    
    # 表面积分析
    surface_area = miner.analyze_surface_area(frame)
    analysis['surface_area'] = surface_area
    
    return analysis

# 分析示例分子
molecular_analysis = analyze_molecular_structure(frame)
print("分子结构分析:")
print(f"  聚类数: {molecular_analysis['clusters']['n_clusters']}")
print(f"  最大聚类大小: {molecular_analysis['clusters']['largest_cluster']}")
print(f"  结合位点数: {molecular_analysis['binding_sites']['n_sites']}")
print(f"  表面积: {molecular_analysis['surface_area']:.2f} Ų")
```

---

## 自定义扩展

### 示例8：自定义力场

```python
class SimpleForceField:
    """简单力场实现"""
    
    def __init__(self):
        # Lennard-Jones参数
        self.lj_params = {
            'C': {'sigma': 3.4, 'epsilon': 0.086},
            'H': {'sigma': 2.5, 'epsilon': 0.030},
            'O': {'sigma': 3.2, 'epsilon': 0.152},
            'N': {'sigma': 3.3, 'epsilon': 0.170}
        }
        
        # 键参数
        self.bond_params = {
            ('C', 'C'): {'k': 350.0, 'r0': 1.54},
            ('C', 'H'): {'k': 340.0, 'r0': 1.09},
            ('C', 'O'): {'k': 450.0, 'r0': 1.43},
            ('O', 'H'): {'k': 450.0, 'r0': 0.96}
        }
        
        # 角度参数
        self.angle_params = {
            ('H', 'C', 'H'): {'k': 35.0, 'theta0': 109.5},
            ('C', 'C', 'C'): {'k': 40.0, 'theta0': 112.7},
            ('H', 'O', 'H'): {'k': 55.0, 'theta0': 104.5}
        }
    
    def calculate_lj_energy(self, structure):
        """计算Lennard-Jones能量"""
        energy = 0.0
        
        for i, atom1 in enumerate(structure.atoms):
            for j, atom2 in enumerate(structure.atoms[i+1:], i+1):
                # 获取参数
                elem1 = atom1.get('element', 'C')
                elem2 = atom2.get('element', 'C')
                
                params1 = self.lj_params.get(elem1, self.lj_params['C'])
                params2 = self.lj_params.get(elem2, self.lj_params['C'])
                
                # 组合规则
                sigma = (params1['sigma'] + params2['sigma']) / 2
                epsilon = np.sqrt(params1['epsilon'] * params2['epsilon'])
                
                # 距离
                r = atom1.distance_to(atom2)
                
                # LJ能量
                if r > 0:
                    sr = sigma / r
                    sr6 = sr ** 6
                    energy += 4 * epsilon * (sr6 * sr6 - sr6)
        
        return energy
    
    def calculate_bond_energy(self, structure):
        """计算键能"""
        energy = 0.0
        
        for bond in structure.bonds:
            elem1 = bond.atom1.get('element', 'C')
            elem2 = bond.atom2.get('element', 'C')
            
            # 获取键参数
            bond_key = tuple(sorted([elem1, elem2]))
            params = self.bond_params.get(bond_key)
            
            if params:
                r = bond.length
                r0 = params['r0']
                k = params['k']
                
                # 谐振子势能
                energy += 0.5 * k * (r - r0) ** 2
        
        return energy
    
    def calculate_angle_energy(self, structure):
        """计算角度能"""
        energy = 0.0
        
        for angle in structure.angles:
            elem1 = angle.atom1.get('element', 'C')
            elem_v = angle.vertex.get('element', 'C')
            elem2 = angle.atom2.get('element', 'C')
            
            # 获取角度参数
            angle_key = (elem1, elem_v, elem2)
            params = self.angle_params.get(angle_key)
            
            if not params:
                # 尝试反向
                angle_key = (elem2, elem_v, elem1)
                params = self.angle_params.get(angle_key)
            
            if params:
                theta = np.degrees(angle.value)
                theta0 = params['theta0']
                k = params['k']
                
                # 谐振子角度势能
                energy += 0.5 * k * (theta - theta0) ** 2
        
        return energy
    
    def calculate_total_energy(self, structure):
        """计算总能量"""
        lj_energy = self.calculate_lj_energy(structure)
        bond_energy = self.calculate_bond_energy(structure)
        angle_energy = self.calculate_angle_energy(structure)
        
        return {
            'total': lj_energy + bond_energy + angle_energy,
            'lj': lj_energy,
            'bond': bond_energy,
            'angle': angle_energy
        }

# 使用自定义力场
force_field = SimpleForceField()
energy_breakdown = force_field.calculate_total_energy(water)

print("能量分析:")
print(f"  总能量: {energy_breakdown['total']:.3f} kcal/mol")
print(f"  范德华能量: {energy_breakdown['lj']:.3f} kcal/mol")
print(f"  键能: {energy_breakdown['bond']:.3f} kcal/mol")
print(f"  角度能: {energy_breakdown['angle']:.3f} kcal/mol")
```

### 示例9：自定义分析工具

```python
class AdvancedStructureAnalyzer:
    """高级结构分析工具"""
    
    def __init__(self):
        self.periodic_table = {
            'H': 1.008, 'C': 12.01, 'N': 14.01, 'O': 16.00,
            'P': 30.97, 'S': 32.07, 'Cl': 35.45
        }
    
    def calculate_molecular_formula(self, structure):
        """计算分子式"""
        element_count = {}
        
        for atom in structure.atoms:
            element = atom.get('element', 'X')
            element_count[element] = element_count.get(element, 0) + 1
        
        # 按化学惯例排序（C, H, 然后按字母序）
        ordered_elements = []
        if 'C' in element_count:
            ordered_elements.append('C')
        if 'H' in element_count:
            ordered_elements.append('H')
        
        for element in sorted(element_count.keys()):
            if element not in ['C', 'H']:
                ordered_elements.append(element)
        
        formula = ""
        for element in ordered_elements:
            count = element_count[element]
            if count == 1:
                formula += element
            else:
                formula += f"{element}{count}"
        
        return formula
    
    def calculate_molecular_weight(self, structure):
        """计算分子量"""
        total_weight = 0.0
        
        for atom in structure.atoms:
            element = atom.get('element', 'C')
            weight = self.periodic_table.get(element, 12.01)  # 默认为碳
            total_weight += weight
        
        return total_weight
    
    def analyze_connectivity(self, structure):
        """分析连接性"""
        atom_connectivity = {}
        
        # 初始化
        for i, atom in enumerate(structure.atoms):
            atom_connectivity[i] = []
        
        # 建立连接表
        for bond in structure.bonds:
            atom1_idx = structure.atoms._data.index(bond.atom1)
            atom2_idx = structure.atoms._data.index(bond.atom2)
            
            atom_connectivity[atom1_idx].append(atom2_idx)
            atom_connectivity[atom2_idx].append(atom1_idx)
        
        # 分析连接度
        connectivity_stats = {
            'degree_distribution': {},
            'average_degree': 0.0,
            'max_degree': 0,
            'isolated_atoms': []
        }
        
        degrees = []
        for atom_idx, connections in atom_connectivity.items():
            degree = len(connections)
            degrees.append(degree)
            
            if degree == 0:
                connectivity_stats['isolated_atoms'].append(atom_idx)
            
            connectivity_stats['degree_distribution'][degree] = \
                connectivity_stats['degree_distribution'].get(degree, 0) + 1
        
        connectivity_stats['average_degree'] = np.mean(degrees) if degrees else 0
        connectivity_stats['max_degree'] = max(degrees) if degrees else 0
        
        return connectivity_stats
    
    def find_rings(self, structure, max_ring_size=8):
        """寻找环结构（简化算法）"""
        # 构建邻接列表
        adj_list = {}
        for i, atom in enumerate(structure.atoms):
            adj_list[i] = []
        
        for bond in structure.bonds:
            try:
                i = structure.atoms._data.index(bond.atom1)
                j = structure.atoms._data.index(bond.atom2)
                adj_list[i].append(j)
                adj_list[j].append(i)
            except ValueError:
                continue
        
        # 简单的环检测
        rings = []
        visited_global = set()
        
        def dfs_find_rings(start, current, path, visited):
            if len(path) > max_ring_size:
                return
            
            if current == start and len(path) > 2:
                ring = sorted(path)
                if ring not in rings:
                    rings.append(ring)
                return
            
            for neighbor in adj_list.get(current, []):
                if neighbor not in visited or (neighbor == start and len(path) > 2):
                    new_visited = visited.copy()
                    new_visited.add(neighbor)
                    dfs_find_rings(start, neighbor, path + [neighbor], new_visited)
        
        for start_atom in range(len(structure.atoms)):
            if start_atom not in visited_global:
                dfs_find_rings(start_atom, start_atom, [start_atom], {start_atom})
                visited_global.add(start_atom)
        
        return rings
    
    def generate_full_report(self, structure):
        """生成完整分析报告"""
        report = {}
        
        # 基本信息
        report['molecular_formula'] = self.calculate_molecular_formula(structure)
        report['molecular_weight'] = self.calculate_molecular_weight(structure)
        
        # 结构信息
        report['atom_count'] = len(structure.atoms)
        report['bond_count'] = len(structure.bonds)
        report['angle_count'] = len(structure.angles)
        report['dihedral_count'] = len(structure.dihedrals)
        
        # 连接性分析
        report['connectivity'] = self.analyze_connectivity(structure)
        
        # 环分析
        rings = self.find_rings(structure)
        report['rings'] = {
            'count': len(rings),
            'sizes': [len(ring) for ring in rings],
            'details': rings[:5]  # 只保留前5个环的详细信息
        }
        
        # 几何分析
        if len(structure.atoms) > 0:
            coords = structure.xyz
            
            # 键长统计
            bond_lengths = [bond.length for bond in structure.bonds]
            if bond_lengths:
                report['bond_statistics'] = {
                    'average_length': np.mean(bond_lengths),
                    'min_length': np.min(bond_lengths),
                    'max_length': np.max(bond_lengths),
                    'std_length': np.std(bond_lengths)
                }
            
            # 角度统计
            angle_values = [np.degrees(angle.value) for angle in structure.angles]
            if angle_values:
                report['angle_statistics'] = {
                    'average_angle': np.mean(angle_values),
                    'min_angle': np.min(angle_values),
                    'max_angle': np.max(angle_values),
                    'std_angle': np.std(angle_values)
                }
        
        return report

# 使用高级分析工具
analyzer = AdvancedStructureAnalyzer()
full_report = analyzer.generate_full_report(propane)

print("完整结构分析报告:")
print(f"  分子式: {full_report['molecular_formula']}")
print(f"  分子量: {full_report['molecular_weight']:.2f} g/mol")
print(f"  原子数: {full_report['atom_count']}")
print(f"  化学键数: {full_report['bond_count']}")

if 'connectivity' in full_report:
    conn = full_report['connectivity']
    print(f"  平均连接度: {conn['average_degree']:.1f}")
    print(f"  最大连接度: {conn['max_degree']}")

if 'rings' in full_report:
    rings = full_report['rings']
    print(f"  环数: {rings['count']}")
    if rings['sizes']:
        print(f"  环大小: {rings['sizes']}")
```

---

## 性能优化示例

### 示例10：大型结构优化处理

```python
class PerformanceOptimizer:
    """性能优化工具"""
    
    @staticmethod
    def batch_process_atoms(structure, operation, batch_size=1000):
        """批量处理原子，减少内存使用"""
        results = []
        
        for i in range(0, len(structure.atoms), batch_size):
            batch = structure.atoms[i:i+batch_size]
            batch_results = []
            
            for atom in batch:
                result = operation(atom)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # 可选：垃圾回收
            import gc
            gc.collect()
        
        return results
    
    @staticmethod
    def efficient_distance_calculation(structure, cutoff=10.0):
        """高效的距离计算（使用空间分割）"""
        coords = structure.xyz
        n_atoms = len(coords)
        
        # 简单的格子法
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        
        grid_size = cutoff
        grid_dims = np.ceil((max_coords - min_coords) / grid_size).astype(int)
        
        # 将原子分配到格子
        grid = {}
        for i, coord in enumerate(coords):
            grid_idx = tuple(((coord - min_coords) // grid_size).astype(int))
            
            # 确保索引在范围内
            grid_idx = tuple(np.clip(grid_idx, 0, grid_dims - 1))
            
            if grid_idx not in grid:
                grid[grid_idx] = []
            grid[grid_idx].append(i)
        
        # 计算相邻格子内的距离
        neighbor_pairs = []
        
        for grid_idx, atom_indices in grid.items():
            # 检查相邻格子
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbor_idx = (
                            grid_idx[0] + dx,
                            grid_idx[1] + dy,
                            grid_idx[2] + dz
                        )
                        
                        if neighbor_idx in grid:
                            neighbor_atoms = grid[neighbor_idx]
                            
                            for i in atom_indices:
                                for j in neighbor_atoms:
                                    if i < j:  # 避免重复计算
                                        distance = np.linalg.norm(coords[i] - coords[j])
                                        if distance <= cutoff:
                                            neighbor_pairs.append((i, j, distance))
        
        return neighbor_pairs
    
    @staticmethod
    def memory_efficient_frame_processing(frames, processor_func):
        """内存高效的Frame处理"""
        
        def frame_generator():
            """Frame生成器，避免同时加载所有帧"""
            for frame in frames:
                yield frame
        
        results = []
        
        for frame in frame_generator():
            result = processor_func(frame)
            results.append(result)
            
            # 手动清理
            del frame
        
        return results

# 使用性能优化工具
def efficient_structure_analysis(large_structure):
    """高效的大型结构分析"""
    optimizer = PerformanceOptimizer()
    
    # 批量计算原子质量
    def get_mass(atom):
        element = atom.get('element', 'C')
        masses = {'C': 12.01, 'H': 1.008, 'O': 16.00, 'N': 14.01}
        return masses.get(element, 12.01)
    
    masses = optimizer.batch_process_atoms(large_structure, get_mass, batch_size=500)
    
    # 高效距离计算
    neighbor_pairs = optimizer.efficient_distance_calculation(large_structure, cutoff=5.0)
    
    print(f"高效分析完成:")
    print(f"  原子质量总和: {sum(masses):.2f}")
    print(f"  5Å内相邻原子对: {len(neighbor_pairs)}")
    
    return {
        'masses': masses,
        'neighbor_pairs': neighbor_pairs
    }

# 性能测试
import time

def performance_benchmark():
    """性能基准测试"""
    
    # 创建大型测试结构
    large_struct = mp.AtomicStructure(name="large_test")
    
    n_atoms = 1000
    for i in range(n_atoms):
        x = (i % 10) * 2.0
        y = ((i // 10) % 10) * 2.0
        z = (i // 100) * 2.0
        
        large_struct.def_atom(
            name=f"C{i+1}",
            element="C",
            xyz=[x, y, z]
        )
    
    print(f"创建了 {len(large_struct.atoms)} 个原子的测试结构")
    
    # 测试常规方法
    start_time = time.time()
    
    # 常规距离计算
    normal_pairs = []
    coords = large_struct.xyz
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= 5.0:
                normal_pairs.append((i, j, dist))
    
    normal_time = time.time() - start_time
    
    # 测试优化方法
    start_time = time.time()
    efficient_analysis = efficient_structure_analysis(large_struct)
    optimized_time = time.time() - start_time
    
    print(f"\n性能比较:")
    print(f"  常规方法: {normal_time:.3f}s, 找到 {len(normal_pairs)} 对")
    print(f"  优化方法: {optimized_time:.3f}s, 找到 {len(efficient_analysis['neighbor_pairs'])} 对")
    print(f"  加速比: {normal_time/optimized_time:.1f}x")

# 运行性能测试
performance_benchmark()
```

这个示例集合展示了MolPy核心模块的各种应用场景，从基础分子构建到高级分析工具，再到性能优化技巧。每个示例都是独立的，可以根据具体需求选择使用。
