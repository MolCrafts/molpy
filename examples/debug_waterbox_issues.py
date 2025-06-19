#!/usr/bin/env python3
"""
Debug script for waterbox issues:
1. 坐标深复制问题
2. 电荷字段问题  
3. 键和角输出问题
"""

import molpy as mp
import numpy as np
from pathlib import Path

print("=== 1. Testing deep copy issue ===")

class SPCE(mp.AtomicStructure):
    def __init__(self, name="spce", molid=1):
        super().__init__(name=name)
        o = self.def_atom(
            name="o", molid=molid, type="O", q=-0.8476, xyz=[0.00000, 0.00000, 0.00000]
        )
        h1 = self.def_atom(
            name="h1",
            molid=molid,
            type="H",
            q=0.4238,
            xyz=[0.8164904, 0.5773590, 0.00000]
        )
        h2 = self.def_atom(
            name="h2",
            molid=molid,
            type="H",
            q=0.4238,
            xyz=[-0.8164904, 0.5773590, 0.00000]
        )
        self.def_bond(o, h1)
        self.def_bond(o, h2)
        
        # Create and add angle
        angle = mp.Angle(h1, o, h2, theta0=109.47, k=1000.0)
        self.add_angle(angle)

# 创建模板分子
spce_template = SPCE(name="spce_template", molid=1)

print("Template molecule created:")
print(f"Atoms: {len(spce_template.atoms)}")
print(f"Bonds: {len(spce_template.bonds)}")
print(f"Angles: {len(spce_template.angles)}")

print("\nTemplate atoms properties:")
for i, atom in enumerate(spce_template.atoms):
    print(f"  Atom {i}: name={atom.get('name')}, type={atom.get('type')}, q={atom.get('q')}, xyz={atom.get('xyz')}")

# 测试深复制
print("\n=== Testing deep copy behavior ===")

# 方法1: 使用__call__()方法
water1 = spce_template(molid=2)
print(f"\nWater1 (using __call__): name={water1.get('name')}")

# 手动更新原子的molid（因为molid是在原子级别设置的）
for atom in water1.atoms:
    atom['molid'] = 2

print("Water1 atoms:")
for i, atom in enumerate(water1.atoms):
    print(f"  Atom {i}: name={atom.get('name')}, molid={atom.get('molid')}, q={atom.get('q')}, xyz={atom.get('xyz')}")

# 移动water1
spatial_water1 = mp.SpatialWrapper(water1)
spatial_water1.move([3.0, 0.0, 0.0])

print("\nAfter moving water1:")
print("Template atoms (should NOT change):")
for i, atom in enumerate(spce_template.atoms):
    print(f"  Atom {i}: name={atom.get('name')}, xyz={atom.get('xyz')}")

print("Water1 atoms (should be moved):")
for i, atom in enumerate(water1.atoms):
    print(f"  Atom {i}: name={atom.get('name')}, xyz={atom.get('xyz')}")

# 测试电荷字段
print("\n=== Testing charge field ===")
frame = spce_template.to_frame()
print(f"Frame atoms keys: {list(frame['atoms'].keys())}")
if 'q' in frame['atoms']:
    print(f"Charges in frame: {frame['atoms']['q']}")
else:
    print("No 'q' field found in frame!")

# 测试bonds和angles
print("\n=== Testing bonds and angles ===")
if 'bonds' in frame:
    print(f"Bonds in frame: {len(frame['bonds'])} bonds")
    print(f"Bond keys: {list(frame['bonds'].keys())}")
else:
    print("No bonds found in frame!")

if 'angles' in frame:
    print(f"Angles in frame: {len(frame['angles'])} angles")
    print(f"Angle keys: {list(frame['angles'].keys())}")
else:
    print("No angles found in frame!")

print("\n=== Testing system composition ===")

system = mp.System()
ff = mp.ForceField(name="spce", unit="real")
system.set_forcefield(ff)
system.def_box(np.diag([10.0, 10.0, 10.0]))

# 添加两个分子
system.add_struct(spce_template)
system.add_struct(water1)

print(f"System structures: {len(system._struct)}")
frame_system = system.to_frame()
print(f"System frame atoms: {len(frame_system['atoms']) if 'atoms' in frame_system else 0}")
if 'atoms' in frame_system:
    print(f"Atoms keys: {list(frame_system['atoms'].keys())}")
    if 'q' in frame_system['atoms']:
        print(f"Charges: {frame_system['atoms']['q']}")
    if 'xyz' in frame_system['atoms']:
        print(f"First few coordinates: {frame_system['atoms']['xyz'][:6]}")
