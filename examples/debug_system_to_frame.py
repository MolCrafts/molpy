#!/usr/bin/env python3
"""
Debug script for System.to_frame() issue
"""

import molpy as mp
import numpy as np

print("=== Testing System.to_frame() ===")

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

# 创建系统
system = mp.System()
ff = mp.ForceField(name="spce", unit="real")
system.set_forcefield(ff)
system.def_box(np.diag([10.0, 10.0, 10.0]))

# 创建两个分子
mol1 = SPCE(name="mol1", molid=1)
mol2 = SPCE(name="mol2", molid=2)

print(f"Mol1: {len(mol1.atoms)} atoms, {len(mol1.bonds)} bonds, {len(mol1.angles)} angles")
print(f"Mol2: {len(mol2.atoms)} atoms, {len(mol2.bonds)} bonds, {len(mol2.angles)} angles")

# 添加到系统
system.add_struct(mol1)
system.add_struct(mol2)

print(f"\nSystem has {len(system._struct)} structures")

# 手动测试合并
print("\n=== Manual merge test ===")
combined = mp.AtomicStructure("combined")
print(f"Empty combined: {len(combined.atoms)} atoms")

combined.add_struct(mol1)
print(f"After adding mol1: {len(combined.atoms)} atoms, {len(combined.bonds)} bonds, {len(combined.angles)} angles")

combined.add_struct(mol2)
print(f"After adding mol2: {len(combined.atoms)} atoms, {len(combined.bonds)} bonds, {len(combined.angles)} angles")

# 测试to_frame
print("\n=== Testing to_frame ===")
frame = combined.to_frame()
print(f"Combined frame: {len(frame['atoms']) if 'atoms' in frame else 0} atoms")
if 'atoms' in frame:
    print(f"Atoms keys: {list(frame['atoms'].keys())}")
    if 'q' in frame['atoms']:
        print(f"Charges: {frame['atoms']['q']}")

# 测试系统to_frame
print("\n=== Testing system.to_frame() ===")
sys_frame = system.to_frame()
print(f"System frame: {len(sys_frame['atoms']) if 'atoms' in sys_frame else 0} atoms")
if 'atoms' in sys_frame:
    print(f"Atoms keys: {list(sys_frame['atoms'].keys())}")
    if 'q' in sys_frame['atoms']:
        print(f"Charges: {sys_frame['atoms']['q']}")
