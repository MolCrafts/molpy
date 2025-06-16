# Struct模块教程

## 概述

Struct模块是MolPy框架的核心，提供了表示和操作分子结构的基础数据类型。本教程将带你从基础使用到高级扩展。

## 1. 基础概念

### 1.1 Entity系统

所有MolPy对象都继承自`Entity`类，提供灵活的属性存储：

```python
import molpy as mp

# 创建基础实体
entity = mp.Entity(name="example", type="test")

# 字典式访问
print(entity["name"])  # "example"

# 属性式访问
entity.value = 42
print(entity.value)  # 42

# 动态添加属性
entity["dynamic_prop"] = [1, 2, 3]
```

**核心特性：**
- 灵活的属性存储
- 支持字典和属性两种访问方式
- 深拷贝和修改功能
- 基于对象身份的哈希和比较

### 1.2 空间实体

具有空间坐标的实体继承`SpatialMixin`：

```python
# 原子是空间实体
atom = mp.Atom(name="C1", element="C", xyz=[0.0, 0.0, 0.0])

# 空间操作
atom.move([1.0, 0.0, 0.0])  # 平移
atom.rotate(np.pi/2, [0, 0, 1])  # 旋转
atom.reflect([1, 0, 0])  # 反射

# 计算距离
atom2 = mp.Atom(name="C2", xyz=[2.0, 0.0, 0.0])
distance = atom.distance_to(atom2)
```

## 2. 分子构建

### 2.1 创建原子

```python
# 基本创建
carbon = mp.Atom(
    name="CA",
    element="C",
    xyz=[0.0, 0.0, 0.0],
    residue="ALA",
    atom_type="CA"
)

# 批量创建
atoms = []
for i in range(3):
    atom = mp.Atom(
        name=f"C{i+1}",
        element="C",
        xyz=[i * 1.5, 0.0, 0.0]
    )
    atoms.append(atom)
```

### 2.2 构建分子结构

```python
# 创建分子结构
molecule = mp.AtomicStructure(name="propane")

# 方法1：先创建原子再添加
c1 = mp.Atom(name="C1", element="C", xyz=[0.0, 0.0, 0.0])
c2 = mp.Atom(name="C2", element="C", xyz=[1.5, 0.0, 0.0])
c3 = mp.Atom(name="C3", element="C", xyz=[3.0, 0.0, 0.0])

molecule.add_atom(c1)
molecule.add_atom(c2)
molecule.add_atom(c3)

# 方法2：直接定义原子
molecule.def_atom(name="H1", element="H", xyz=[-0.5, 0.5, 0.5])
molecule.def_atom(name="H2", element="H", xyz=[-0.5, -0.5, 0.5])

# 查看结构信息
print(f"原子数: {len(molecule.atoms)}")
print(f"结构名: {molecule.get('name')}")
```

### 2.3 添加化学键

```python
# 创建化学键
bond1 = mp.Bond(c1, c2, bond_type="single", length=1.5)
bond2 = mp.Bond(c2, c3, bond_type="single", length=1.5)

molecule.add_bond(bond1)
molecule.add_bond(bond2)

# 或者直接定义化学键
molecule.def_bond(c1, molecule.atoms[-2], bond_type="single")  # C1-H1
molecule.def_bond(c1, molecule.atoms[-1], bond_type="single")  # C1-H2

print(f"化学键数: {len(molecule.bonds)}")
```

## 3. 高级结构操作

### 3.1 结构合并和分解

```python
# 创建两个分子
mol1 = mp.AtomicStructure(name="molecule1")
mol2 = mp.AtomicStructure(name="molecule2")

# ... 添加原子和键 ...

# 合并分子
complex_struct = mp.AtomicStructure(name="complex")
complex_struct.add_struct(mol1)
complex_struct.add_struct(mol2)

# 批量合并
molecules = [mol1, mol2]
combined = mp.AtomicStructure.concat("combined_system", molecules)
```

### 3.2 原子查找和过滤

```python
# 按条件查找原子
carbon_atom = molecule.get_atom_by(
    lambda atom: atom.get("element") == "C" and atom.get("name") == "C1"
)

# 使用Entities容器的高级功能
carbons = []
for atom in molecule.atoms:
    if atom.get("element") == "C":
        carbons.append(atom)

# 按名称访问
atom_by_name = molecule.atoms["C1"]  # 如果名称唯一
```

### 3.3 几何计算

```python
# 计算键长
for bond in molecule.bonds:
    length = bond.length
    print(f"键 {bond.atom1.name}-{bond.atom2.name}: {length:.3f} Å")

# 计算角度
if len(molecule.atoms) >= 3:
    angle = mp.Angle(molecule.atoms[0], molecule.atoms[1], molecule.atoms[2])
    angle_value = angle.value  # 弧度
    angle_degrees = np.degrees(angle_value)
    print(f"角度: {angle_degrees:.1f}°")

# 计算二面角
if len(molecule.atoms) >= 4:
    dihedral = mp.Dihedral(*molecule.atoms[:4])
    dihedral_value = np.degrees(dihedral.value)
    print(f"二面角: {dihedral_value:.1f}°")
```

## 4. 层次结构管理

### 4.1 构建层次结构

```python
# 创建蛋白质层次结构
protein = mp.AtomicStructure(name="protein")
chain_a = mp.AtomicStructure(name="chain_A")
chain_b = mp.AtomicStructure(name="chain_B")

# 建立层次关系
protein.add_child(chain_a)
protein.add_child(chain_b)

# 创建残基
for i in range(10):
    residue = mp.AtomicStructure(name=f"residue_{i+1}")
    chain_a.add_child(residue)
    
    # 为每个残基添加原子
    for j, atom_name in enumerate(["N", "CA", "C", "O"]):
        atom = residue.def_atom(
            name=atom_name,
            element=atom_name[0],
            xyz=[i*3.8 + j*0.5, 0, 0]
        )
```

### 4.2 层次遍历

```python
# 深度优先遍历
def print_hierarchy(node, level=0):
    indent = "  " * level
    name = node.get('name', 'unnamed')
    natoms = len(node.atoms) if hasattr(node, 'atoms') else 0
    print(f"{indent}{name} ({natoms} atoms)")

protein.traverse_dfs(lambda node: print_hierarchy(node, node.depth))

# 查找特定节点
residue_5 = protein.find_by_name("residue_5")
if residue_5:
    print(f"找到残基: {residue_5.get('name')}")

# 获取层次信息
info = chain_a.get_hierarchy_info()
print(f"Chain A - 深度: {info['depth']}, 子节点: {info['num_children']}")
```

## 5. 自定义扩展

### 5.1 自定义原子类型

```python
class ProteinAtom(mp.Atom):
    """蛋白质原子类"""
    
    def __init__(self, name, element, xyz, residue_name, residue_number, **kwargs):
        super().__init__(name=name, element=element, xyz=xyz, **kwargs)
        self.residue_name = residue_name
        self.residue_number = residue_number
    
    @property
    def is_backbone(self):
        """判断是否为主链原子"""
        return self.name in ["N", "CA", "C", "O"]
    
    @property
    def is_sidechain(self):
        """判断是否为侧链原子"""
        return not self.is_backbone

# 使用自定义原子
ca_atom = ProteinAtom("CA", "C", [0, 0, 0], "ALA", 1, b_factor=20.0)
print(f"主链原子: {ca_atom.is_backbone}")
```

### 5.2 自定义结构类型

```python
class ProteinStructure(mp.AtomicStructure):
    """蛋白质结构类"""
    
    def __init__(self, name, sequence=None, **props):
        super().__init__(name=name, **props)
        self.sequence = sequence or ""
    
    def add_residue(self, residue_name, residue_number):
        """添加残基"""
        residue = mp.AtomicStructure(name=f"{residue_name}_{residue_number}")
        residue.residue_name = residue_name
        residue.residue_number = residue_number
        self.add_child(residue)
        return residue
    
    def get_residue(self, residue_number):
        """获取指定编号的残基"""
        for child in self.children:
            if getattr(child, 'residue_number', None) == residue_number:
                return child
        return None
    
    def backbone_atoms(self):
        """获取所有主链原子"""
        backbone = []
        for residue in self.children:
            for atom in residue.atoms:
                if hasattr(atom, 'is_backbone') and atom.is_backbone:
                    backbone.append(atom)
        return backbone
    
    def calculate_secondary_structure(self):
        """计算二级结构（简化版）"""
        # 这里可以实现DSSP算法或其他二级结构预测方法
        pass

# 使用自定义蛋白质结构
protein = ProteinStructure("myprotein", sequence="ACDEFGHIKLMNPQRSTVWY")
ala_residue = protein.add_residue("ALA", 1)
```

## 6. 性能优化技巧

### 6.1 批量操作

```python
# 避免在循环中频繁调用add方法
# 不推荐
for atom_data in large_atom_list:
    atom = mp.Atom(**atom_data)
    structure.add_atom(atom)

# 推荐
atoms = [mp.Atom(**atom_data) for atom_data in large_atom_list]
structure.add_atoms(atoms)
```

### 6.2 合理使用索引

```python
# 为频繁访问的原子建立索引
atom_index = {atom.get('name'): atom for atom in structure.atoms}
ca_atom = atom_index.get('CA')  # O(1)查找

# 而不是每次都遍历
# ca_atom = structure.get_atom_by(lambda a: a.get('name') == 'CA')  # O(n)查找
```

### 6.3 内存管理

```python
# 对于大型结构，考虑使用生成器
def atom_generator(structure):
    """原子生成器，节省内存"""
    for atom in structure.atoms:
        yield atom

# 处理大型结构时使用生成器
for atom in atom_generator(large_structure):
    # 处理单个原子
    process_atom(atom)
```

## 7. 调试和验证

### 7.1 结构验证

```python
def validate_structure(structure):
    """验证结构完整性"""
    errors = []
    
    # 检查原子坐标
    for i, atom in enumerate(structure.atoms):
        try:
            coords = atom.xyz
            if not isinstance(coords, np.ndarray) or coords.shape != (3,):
                errors.append(f"原子 {i} 坐标格式错误")
        except Exception as e:
            errors.append(f"原子 {i} 坐标访问错误: {e}")
    
    # 检查化学键
    for i, bond in enumerate(structure.bonds):
        if bond.atom1 not in structure.atoms:
            errors.append(f"键 {i} 的第一个原子不在结构中")
        if bond.atom2 not in structure.atoms:
            errors.append(f"键 {i} 的第二个原子不在结构中")
    
    return errors

# 使用验证
errors = validate_structure(my_structure)
if errors:
    print("结构验证失败:")
    for error in errors:
        print(f"  - {error}")
```

### 7.2 调试信息

```python
def debug_structure(structure):
    """打印结构调试信息"""
    print(f"结构名称: {structure.get('name', 'unnamed')}")
    print(f"原子数: {len(structure.atoms)}")
    print(f"化学键数: {len(structure.bonds)}")
    print(f"角度数: {len(structure.angles)}")
    print(f"二面角数: {len(structure.dihedrals)}")
    
    # 原子统计
    element_count = {}
    for atom in structure.atoms:
        element = atom.get('element', 'unknown')
        element_count[element] = element_count.get(element, 0) + 1
    
    print("元素统计:")
    for element, count in element_count.items():
        print(f"  {element}: {count}")
    
    # 层次信息
    if hasattr(structure, 'get_hierarchy_info'):
        hier_info = structure.get_hierarchy_info()
        print(f"层次深度: {hier_info['depth']}")
        print(f"子节点数: {hier_info['num_children']}")

# 使用调试
debug_structure(my_structure)
```

这个教程涵盖了Struct模块的主要功能和使用模式。你可以根据具体需求选择相应的部分进行学习和应用。
