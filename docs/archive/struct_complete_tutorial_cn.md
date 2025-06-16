# Complete Struct Module Tutorial

## Overview

The Struct module is the core of the MolPy framework, providing fundamental data types for representing and manipulating molecular structures. This tutorial will guide you from basic usage to advanced extensions, covering all core functionality.

## 1. Basic Concepts

### 1.1 Entity System

All MolPy objects inherit from the `Entity` class, providing flexible attribute storage:

```python
import molpy as mp
import numpy as np

# Create basic entity
entity = mp.Entity(name="example", type="test")

# Dictionary-style access
print(entity["name"])  # "example"

# Dynamic property addition
entity["dynamic_prop"] = [1, 2, 3]
print(entity["dynamic_prop"])  # [1, 2, 3]

# Clone and modify
new_entity = entity.clone(name="modified", value=42)
print(new_entity["name"])  # "modified"
print(new_entity["value"])  # 42

# Shortcut call syntax
another_entity = entity(name="another", extra="data")
```

**Core Features:**
- Flexible attribute storage
- Dictionary-style access support
- Deep copy and modification functionality
- Object identity-based hashing and comparison
- Support for `__call__` shortcut syntax

### 1.2 Spatial Entities

Entities with spatial coordinates inherit from `SpatialMixin`:

```python
# Atoms are spatial entities
atom = mp.Atom(name="C1", element="C", xyz=[0.0, 0.0, 0.0])

# 空间操作
atom.move([1.0, 0.0, 0.0])  # 平移
print(f"移动后坐标: {atom.xyz}")

atom.rotate(np.pi/2, [0, 0, 1])  # 绕z轴旋转90度
print(f"旋转后坐标: {atom.xyz}")

atom.reflect([1, 0, 0])  # 沿x轴反射
print(f"反射后坐标: {atom.xyz}")

# 计算距离
atom2 = mp.Atom(name="C2", element="C", xyz=[2.0, 0.0, 0.0])
distance = atom.distance_to(atom2)
print(f"距离: {distance:.2f} Å")
```

## 2. 原子操作

### 2.1 创建原子

```python
# 基本创建
carbon = mp.Atom(
    name="CA",
    element="C",
    xyz=[0.0, 0.0, 0.0],
    residue="ALA",
    atom_type="CA",
    charge=0.0,
    mass=12.01
)

# 检查原子属性
print(f"原子名: {carbon.name}")
print(f"元素: {carbon['element']}")
print(f"坐标: {carbon.xyz}")
print(f"电荷: {carbon.get('charge', 0.0)}")

# 批量创建
atoms = []
for i in range(5):
    atom = mp.Atom(
        name=f"C{i+1}",
        element="C",
        xyz=[i * 1.5, 0.0, 0.0]
    )
    atoms.append(atom)

print(f"创建了 {len(atoms)} 个原子")
```

### 2.2 原子坐标操作

```python
# 坐标访问和修改
atom = mp.Atom(name="C1", xyz=[1, 2, 3])
print(f"原始坐标: {atom.xyz}")

# 修改坐标
atom.xyz = [4, 5, 6]
print(f"新坐标: {atom.xyz}")

# 验证坐标格式
try:
    atom.xyz = [1, 2]  # 错误：不是3D坐标
except ValueError as e:
    print(f"坐标错误: {e}")

# 链式空间操作
atom.move([1, 0, 0]).rotate(np.pi/4, [0, 0, 1]).reflect([1, 0, 0])
print(f"链式操作后坐标: {atom.xyz}")
```

## 3. 分子结构构建

### 3.1 创建分子结构

```python
# 创建空分子结构
molecule = mp.AtomicStructure(name="propane")
print(f"结构名: {molecule['name']}")
print(f"原子数: {len(molecule.atoms)}")

# 添加原子
c1 = mp.Atom(name="C1", element="C", xyz=[0.0, 0.0, 0.0])
c2 = mp.Atom(name="C2", element="C", xyz=[1.5, 0.0, 0.0])  
c3 = mp.Atom(name="C3", element="C", xyz=[3.0, 0.0, 0.0])

molecule.add_atom(c1)
molecule.add_atom(c2)
molecule.add_atom(c3)

print(f"添加后原子数: {len(molecule.atoms)}")

# 直接定义原子
h1 = molecule.def_atom(name="H1", element="H", xyz=[-0.5, 0.5, 0.5])
h2 = molecule.def_atom(name="H2", element="H", xyz=[-0.5, -0.5, 0.5])
h3 = molecule.def_atom(name="H3", element="H", xyz=[-0.5, 0.0, -0.5])

print(f"定义氢原子后总原子数: {len(molecule.atoms)}")
```

### 3.2 添加化学键

```python
# 方法1：创建Bond对象
bond1 = mp.Bond(c1, c2, bond_type="single", order=1)
molecule.add_bond(bond1)

# 方法2：直接定义键
bond2 = molecule.def_bond(c2, c3, bond_type="single", length=1.54)

# 添加C-H键
molecule.def_bond(c1, h1, bond_type="single")
molecule.def_bond(c1, h2, bond_type="single")
molecule.def_bond(c1, h3, bond_type="single")

print(f"键数: {len(molecule.bonds)}")

# 检查键长
for i, bond in enumerate(molecule.bonds):
    atom1_name = bond.atom1.name
    atom2_name = bond.atom2.name
    length = bond.length
    print(f"键 {i+1}: {atom1_name}-{atom2_name}, 长度: {length:.3f} Å")
```

### 3.3 添加角度和二面角

```python
# 添加角度 (H-C-H)
angle1 = mp.Angle(h1, c1, h2, angle_type="tetrahedral")
molecule.add_angle(angle1)

angle2 = mp.Angle(h1, c1, h3, angle_type="tetrahedral")
molecule.add_angle(angle2)

angle3 = mp.Angle(h2, c1, h3, angle_type="tetrahedral")
molecule.add_angle(angle3)

print(f"角度数: {len(molecule.angles)}")

# 检查角度值
for i, angle in enumerate(molecule.angles):
    angle_deg = np.degrees(angle.value)
    print(f"角度 {i+1}: {angle_deg:.1f}°")

# 添加二面角
if len(molecule.atoms) >= 4:
    dihedral = mp.Dihedral(h1, c1, c2, c3, dihedral_type="sp3-sp3")
    molecule.add_dihedral(dihedral)
    
    dihedral_deg = np.degrees(dihedral.value)
    print(f"二面角: {dihedral_deg:.1f}°")
```

## 4. 结构查询和操作

### 4.1 原子查找

```python
# 按名称查找
carbon_atom = molecule.atoms["C1"]
if carbon_atom:
    print(f"找到原子: {carbon_atom.name} at {carbon_atom.xyz}")

# 按条件查找
def is_carbon(atom):
    return atom.get("element") == "C"

first_carbon = molecule.atoms.get_by(is_carbon)
print(f"第一个碳原子: {first_carbon.name}")

# 获取所有特定元素的原子
carbons = [atom for atom in molecule.atoms if atom.get("element") == "C"]
hydrogens = [atom for atom in molecule.atoms if atom.get("element") == "H"]

print(f"碳原子: {[atom.name for atom in carbons]}")
print(f"氢原子: {[atom.name for atom in hydrogens]}")
```

### 4.2 键的查询

```python
# 查找包含特定原子的键
def find_bonds_with_atom(molecule, target_atom):
    result = []
    for bond in molecule.bonds:
        if bond.atom1 is target_atom or bond.atom2 is target_atom:
            result.append(bond)
    return result

c1_bonds = find_bonds_with_atom(molecule, c1)
print(f"C1参与的键数: {len(c1_bonds)}")

for bond in c1_bonds:
    other_atom = bond.atom2 if bond.atom1 is c1 else bond.atom1
    print(f"  C1-{other_atom.name}: {bond.length:.3f} Å")
```

### 4.3 结构修改

```python
# 移除原子
if len(molecule.atoms) > 3:
    removed_atom = molecule.atoms[-1]
    molecule.remove_atom(removed_atom)
    print(f"移除原子后，剩余原子数: {len(molecule.atoms)}")

# 移除键
if len(molecule.bonds) > 0:
    molecule.remove_bond(molecule.bonds[0])
    print(f"移除键后，剩余键数: {len(molecule.bonds)}")

# 添加新原子和键
new_atom = mp.Atom(name="O1", element="O", xyz=[1.5, 1.5, 0])
molecule.add_atom(new_atom)
molecule.def_bond(c2, new_atom, bond_type="single")

print(f"添加氧原子后: {len(molecule.atoms)} 原子, {len(molecule.bonds)} 键")
```

## 5. 空间操作

### 5.1 整体结构操作

```python
# 获取所有原子坐标
all_coords = molecule.xyz
print(f"坐标矩阵形状: {all_coords.shape}")
print(f"前3个原子坐标:\n{all_coords[:3]}")

# 移动整个分子
translation = np.array([2, 0, 0])
molecule.xyz = molecule.xyz + translation
print("移动后前3个原子坐标:")
print(molecule.xyz[:3])

# 也可以使用 += 操作
molecule.xyz -= translation  # 恢复原位置
```

### 5.2 几何计算

```python
def calculate_geometric_center(molecule):
    """计算几何中心"""
    coords = molecule.xyz
    return np.mean(coords, axis=0)

def calculate_radius_of_gyration(molecule):
    """计算回转半径"""
    coords = molecule.xyz
    center = calculate_geometric_center(molecule)
    distances = np.linalg.norm(coords - center, axis=1)
    return np.sqrt(np.mean(distances**2))

# 计算几何性质
center = calculate_geometric_center(molecule)
radius = calculate_radius_of_gyration(molecule)

print(f"几何中心: {center}")
print(f"回转半径: {radius:.3f} Å")
```

## 6. 高级功能

### 6.1 结构合并

```python
# 创建水分子
water = mp.AtomicStructure(name="water")
o = water.def_atom(name="O", element="O", xyz=[5, 0, 0])
h1 = water.def_atom(name="H1", element="H", xyz=[5.757, 0.586, 0])
h2 = water.def_atom(name="H2", element="H", xyz=[4.243, 0.586, 0])

# 添加O-H键
water.def_bond(o, h1, bond_type="covalent")
water.def_bond(o, h2, bond_type="covalent")

# 添加H-O-H角度
hoh_angle = mp.Angle(h1, o, h2, angle_type="bent")
water.add_angle(hoh_angle)

print(f"水分子: {len(water.atoms)} 原子, {len(water.bonds)} 键")

# 合并到主分子
original_atom_count = len(molecule.atoms)
molecule.add_struct(water)

print(f"合并后: {len(molecule.atoms)} 原子 (增加了 {len(molecule.atoms) - original_atom_count})")
print(f"合并后键数: {len(molecule.bonds)}")
```

### 6.2 层次结构管理

```python
# 创建层次结构
protein = mp.AtomicStructure(name="small_protein")
residue1 = mp.AtomicStructure(name="ALA1")
residue2 = mp.AtomicStructure(name="GLY2")

# 添加原子到残基
ca1 = residue1.def_atom(name="CA", element="C", xyz=[0, 0, 0])
cb1 = residue1.def_atom(name="CB", element="C", xyz=[1, 0, 0])
residue1.def_bond(ca1, cb1)

ca2 = residue2.def_atom(name="CA", element="C", xyz=[2, 0, 0])

# 建立层次关系
protein.add_child(residue1)
protein.add_child(residue2)

# 查看层次信息
print(f"蛋白质子结构数: {len(protein.children)}")
print(f"残基1的父结构: {residue1.parent.get('name')}")

# 获取层次信息
hierarchy_info = protein.get_hierarchy_info()
print(f"层次信息: {hierarchy_info}")

# 遍历层次结构
def print_hierarchy(struct, level=0):
    indent = "  " * level
    name = struct.get('name', 'unnamed')
    atom_count = len(struct.atoms) if hasattr(struct, 'atoms') else 0
    print(f"{indent}{name} ({atom_count} atoms)")
    
    if hasattr(struct, 'children'):
        for child in struct.children:
            print_hierarchy(child, level + 1)

print("\n层次结构:")
print_hierarchy(protein)
```

### 6.3 批量操作

```python
# 批量创建原子
def create_alkane_chain(length, name="alkane"):
    """创建烷烃链"""
    alkane = mp.AtomicStructure(name=name)
    
    carbons = []
    for i in range(length):
        carbon = alkane.def_atom(
            name=f"C{i+1}",
            element="C", 
            xyz=[i * 1.5, 0, 0]
        )
        carbons.append(carbon)
    
    # 添加C-C键
    for i in range(len(carbons) - 1):
        alkane.def_bond(carbons[i], carbons[i+1], bond_type="single")
    
    return alkane

# 创建长链烷烃
long_chain = create_alkane_chain(10, "decane")
print(f"癸烷: {len(long_chain.atoms)} 原子, {len(long_chain.bonds)} 键")

# 使用concat合并多个结构
short_chains = [
    create_alkane_chain(3, "propane"),
    create_alkane_chain(4, "butane"),
    create_alkane_chain(5, "pentane")
]

combined = mp.AtomicStructure.concat("mixed_alkanes", short_chains)
print(f"合并烷烃: {len(combined.atoms)} 原子")

# 检查层次结构
print(f"子结构数: {len(combined.children)}")
for child in combined.children:
    print(f"  {child.get('name')}: {len(child.atoms)} 原子")
```

## 7. 实用工具函数

### 7.1 结构验证

```python
def validate_structure(struct):
    """验证分子结构的合理性"""
    issues = []
    
    # 检查基本结构
    if len(struct.atoms) == 0:
        issues.append("结构中没有原子")
        return issues
    
    # 检查原子坐标
    for atom in struct.atoms:
        try:
            coords = atom.xyz
            if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                issues.append(f"原子 {atom.name} 坐标异常: {coords}")
        except Exception as e:
            issues.append(f"原子 {atom.name} 坐标访问错误: {e}")
    
    # 检查键长合理性
    for bond in struct.bonds:
        try:
            length = bond.length
            if length < 0.5:
                issues.append(f"键 {bond.atom1.name}-{bond.atom2.name} 过短: {length:.3f} Å")
            elif length > 4.0:
                issues.append(f"键 {bond.atom1.name}-{bond.atom2.name} 过长: {length:.3f} Å")
        except Exception as e:
            issues.append(f"键长计算错误: {e}")
    
    # 检查角度合理性
    for angle in struct.angles:
        try:
            angle_deg = np.degrees(angle.value)
            if angle_deg < 30 or angle_deg > 180:
                issues.append(f"角度异常: {angle_deg:.1f}°")
        except Exception as e:
            issues.append(f"角度计算错误: {e}")
    
    return issues

# 验证结构
issues = validate_structure(molecule)
if issues:
    print("结构验证发现问题:")
    for issue in issues:
        print(f"- {issue}")
else:
    print("✓ 结构验证通过")
```

### 7.2 结构分析

```python
def analyze_structure(struct):
    """分析分子结构的基本性质"""
    analysis = {}
    
    # 基本统计
    analysis["统计信息"] = {
        "原子数": len(struct.atoms),
        "键数": len(struct.bonds),
        "角度数": len(struct.angles),
        "二面角数": len(struct.dihedrals)
    }
    
    # 元素组成
    elements = {}
    for atom in struct.atoms:
        element = atom.get("element", "Unknown")
        elements[element] = elements.get(element, 0) + 1
    analysis["元素组成"] = elements
    
    # 键长分析
    if struct.bonds:
        bond_lengths = [bond.length for bond in struct.bonds]
        analysis["键长分析"] = {
            "最短键长": f"{min(bond_lengths):.3f} Å",
            "最长键长": f"{max(bond_lengths):.3f} Å",
            "平均键长": f"{np.mean(bond_lengths):.3f} Å",
            "键长标准差": f"{np.std(bond_lengths):.3f} Å"
        }
    
    # 几何分析
    if struct.atoms:
        coords = struct.xyz
        center = np.mean(coords, axis=0)
        
        # 计算边界盒
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        box_size = max_coords - min_coords
        
        analysis["几何分析"] = {
            "几何中心": f"({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})",
            "边界盒大小": f"({box_size[0]:.3f}, {box_size[1]:.3f}, {box_size[2]:.3f})",
            "最大尺寸": f"{np.max(box_size):.3f} Å"
        }
    
    return analysis

# 分析结构
analysis = analyze_structure(molecule)
print("\n=== 结构分析报告 ===")
for category, data in analysis.items():
    print(f"\n{category}:")
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"  {key}: {value}")
    else:
        print(f"  {data}")
```

### 7.3 结构导出

```python
def export_structure_info(struct, filename=None):
    """导出结构信息到文本文件"""
    lines = []
    lines.append(f"# 分子结构信息: {struct.get('name', 'unnamed')}")
    lines.append(f"# 生成时间: {import datetime; datetime.datetime.now()}")
    lines.append("")
    
    # 原子信息
    lines.append("## 原子信息")
    lines.append("Index\tName\tElement\tX\tY\tZ")
    for i, atom in enumerate(struct.atoms):
        name = atom.get('name', f'atom_{i}')
        element = atom.get('element', 'X')
        x, y, z = atom.xyz
        lines.append(f"{i}\t{name}\t{element}\t{x:.3f}\t{y:.3f}\t{z:.3f}")
    
    lines.append("")
    
    # 键信息
    lines.append("## 键信息")
    lines.append("Atom1\tAtom2\tLength\tType")
    for bond in struct.bonds:
        atom1_name = bond.atom1.get('name', 'unknown')
        atom2_name = bond.atom2.get('name', 'unknown')
        length = bond.length
        bond_type = bond.get('bond_type', 'unknown')
        lines.append(f"{atom1_name}\t{atom2_name}\t{length:.3f}\t{bond_type}")
    
    content = "\n".join(lines)
    
    if filename:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"结构信息已保存到: {filename}")
    else:
        print(content)

# 导出结构信息
export_structure_info(molecule)
```

## 8. 最佳实践

### 8.1 性能优化

```python
# ✓ 好的做法：批量操作
atoms = [mp.Atom(name=f"C{i}", xyz=[i, 0, 0]) for i in range(100)]
struct = mp.AtomicStructure()
struct.add_atoms(atoms)  # 一次性添加

# ✗ 避免：循环中的重复操作
# for atom in atoms:
#     struct.add_atom(atom)  # 每次都要处理

# ✓ 好的做法：缓存计算结果
bond_lengths = {}
for bond in struct.bonds:
    key = (bond.atom1, bond.atom2)
    if key not in bond_lengths:
        bond_lengths[key] = bond.length

# ✓ 好的做法：使用向量化操作
coords = struct.xyz
translated_coords = coords + np.array([1, 0, 0])
struct.xyz = translated_coords
```

### 8.2 错误处理

```python
def safe_add_bond(struct, atom1, atom2, **kwargs):
    """安全地添加键，包含错误检查"""
    try:
        # 检查原子是否在结构中
        if atom1 not in struct.atoms or atom2 not in struct.atoms:
            raise ValueError("原子不在结构中")
        
        # 检查是否已存在键
        for existing_bond in struct.bonds:
            if ((existing_bond.atom1 is atom1 and existing_bond.atom2 is atom2) or
                (existing_bond.atom1 is atom2 and existing_bond.atom2 is atom1)):
                print(f"警告: 键 {atom1.name}-{atom2.name} 已存在")
                return existing_bond
        
        # 创建新键
        bond = struct.def_bond(atom1, atom2, **kwargs)
        return bond
        
    except Exception as e:
        print(f"添加键失败: {e}")
        return None

# 使用安全函数
bond = safe_add_bond(molecule, c1, c2, bond_type="single")
if bond:
    print(f"成功添加键: {bond.atom1.name}-{bond.atom2.name}")
```

### 8.3 代码组织

```python
class MoleculeBuilder:
    """分子构建器类"""
    
    def __init__(self, name="molecule"):
        self.struct = mp.AtomicStructure(name=name)
        self.atom_index = {}
    
    def add_atom(self, name, element, xyz, **kwargs):
        """添加原子并建立索引"""
        atom = self.struct.def_atom(name=name, element=element, xyz=xyz, **kwargs)
        self.atom_index[name] = atom
        return atom
    
    def add_bond(self, atom1_name, atom2_name, **kwargs):
        """通过名称添加键"""
        atom1 = self.atom_index.get(atom1_name)
        atom2 = self.atom_index.get(atom2_name)
        
        if not atom1 or not atom2:
            raise ValueError(f"找不到原子: {atom1_name} 或 {atom2_name}")
        
        return self.struct.def_bond(atom1, atom2, **kwargs)
    
    def get_structure(self):
        """获取构建的结构"""
        return self.struct

# 使用构建器
builder = MoleculeBuilder("methanol")
builder.add_atom("C", "C", [0, 0, 0])
builder.add_atom("O", "O", [1.4, 0, 0])
builder.add_atom("H1", "H", [-0.5, 0.9, 0])
builder.add_atom("H2", "H", [-0.5, -0.9, 0])
builder.add_atom("H3", "H", [-0.5, 0, 0.9])
builder.add_atom("H4", "H", [1.9, 0, 0])

# 添加键
builder.add_bond("C", "O", bond_type="single")
builder.add_bond("C", "H1", bond_type="single")
builder.add_bond("C", "H2", bond_type="single")
builder.add_bond("C", "H3", bond_type="single")
builder.add_bond("O", "H4", bond_type="single")

methanol = builder.get_structure()
print(f"甲醇分子: {len(methanol.atoms)} 原子, {len(methanol.bonds)} 键")
```

这个完整的教程涵盖了Struct模块的所有主要功能，从基础概念到高级应用，包括实用工具和最佳实践。通过这些例子，你可以有效地使用MolPy构建和操作复杂的分子系统。
