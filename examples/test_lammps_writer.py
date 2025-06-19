#!/usr/bin/env python3
"""
Test the fixed LAMMPS writer
"""

import molpy as mp
import numpy as np
from pathlib import Path

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

# 创建输出目录
output_dir = Path("./data/test_lammps")
output_dir.mkdir(exist_ok=True)

# 导出LAMMPS文件
data_file = output_dir / "test.data"
mp.io.write_lammps_data(data_file, frame)

print(f"✅ LAMMPS data file written: {data_file}")
print(f"File size: {data_file.stat().st_size} bytes")

# 读取并显示文件内容
print("\n=== File content ===")
with open(data_file, 'r') as f:
    content = f.read()
    print(content)
