#!/usr/bin/env python3
"""
Debug LAMMPS writer issue
"""

import molpy as mp
import numpy as np

# 创建一个简单的系统
system = mp.System()
ff = mp.ForceField(name="test", unit="real")
system.set_forcefield(ff)
system.def_box(np.diag([10.0, 10.0, 10.0]))

# 创建一个分子
struct = mp.AtomicStructure("test")
o = struct.def_atom(name="o", type="O", q=-0.8476, xyz=[0.0, 0.0, 0.0])
h1 = struct.def_atom(name="h1", type="H", q=0.4238, xyz=[1.0, 0.0, 0.0])
struct.def_bond(o, h1)

system.add_struct(struct)

# 转换为frame
frame = system.to_frame()

print("=== Frame analysis ===")
print(f"Frame type: {type(frame)}")
print(f"Frame keys: {list(frame.keys())}")
print(f"Atoms type: {type(frame['atoms'])}")
if hasattr(frame['atoms'], 'sizes'):
    print(f"Atoms sizes: {frame['atoms'].sizes}")
else:
    print(f"Atoms is dict with keys: {list(frame['atoms'].keys())}")
    print(f"Length of atoms: {len(frame['atoms']['id']) if 'id' in frame['atoms'] else 'no id field'}")

# 检查bonds和angles
if 'bonds' in frame:
    print(f"Bonds type: {type(frame['bonds'])}")
    if hasattr(frame['bonds'], 'sizes'):
        print(f"Bonds sizes: {frame['bonds'].sizes}")
    else:
        print(f"Bonds is dict with keys: {list(frame['bonds'].keys())}")
        print(f"Length of bonds: {len(frame['bonds']['id']) if 'id' in frame['bonds'] else 'no id field'}")
else:
    print("No bonds in frame")

if 'angles' in frame:
    print(f"Angles type: {type(frame['angles'])}")
    if hasattr(frame['angles'], 'sizes'):
        print(f"Angles sizes: {frame['angles'].sizes}")
    else:
        print(f"Angles is dict with keys: {list(frame['angles'].keys())}")
        print(f"Length of angles: {len(frame['angles']['id']) if 'id' in frame['angles'] else 'no id field'}")
else:
    print("No angles in frame")

# 检查电荷
if 'q' in frame['atoms']:
    print(f"Charges: {frame['atoms']['q']}")
    print(f"Charge values: {frame['atoms']['q'].values if hasattr(frame['atoms']['q'], 'values') else frame['atoms']['q']}")
