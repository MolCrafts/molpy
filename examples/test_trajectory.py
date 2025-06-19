#!/usr/bin/env python3
"""
Test the new LAMMPS trajectory API
"""

import molpy as mp
import numpy as np
from pathlib import Path
from typing import Optional

print("=== Testing LAMMPS Trajectory API ===")

# 创建一些示例frames用于写入trajectory
frames = []

# 创建基础分子结构
def create_water_frame(timestep: int, translation: Optional[np.ndarray] = None) -> mp.Frame:
    """Create a water molecule frame with optional translation."""
    # 创建水分子原子数据
    atoms_data = {
        'id': [1, 2, 3],
        'molid': [1, 1, 1],
        'type': ['O', 'H', 'H'],
        'q': [-0.8476, 0.4238, 0.4238],
        'x': [0.0, 0.8164904, -0.8164904],
        'y': [0.0, 0.5773590, 0.5773590], 
        'z': [0.0, 0.0, 0.0]
    }
    
    # 应用平移
    if translation is not None:
        atoms_data['x'] = [x + translation[0] for x in atoms_data['x']]
        atoms_data['y'] = [y + translation[1] for y in atoms_data['y']]
        atoms_data['z'] = [z + translation[2] for z in atoms_data['z']]
    
    # 创建box
    box = mp.Box(np.diag([10.0, 10.0, 10.0]))
    
    # 创建frame
    frame = mp.Frame({'atoms': atoms_data}, box=box, timestep=timestep)
    
    return frame

# 创建一系列frames (水分子在移动)
for i in range(5):
    translation = np.array([i * 0.5, 0.0, 0.0])  # 沿x轴移动
    frame = create_water_frame(timestep=i * 100, translation=translation)
    frames.append(frame)
    print(f"Created frame {i}: timestep={i*100}, translation={translation}")

# 创建输出目录
output_dir = Path("./data/trajectory_test")
output_dir.mkdir(exist_ok=True)

# 测试trajectory writer
print("\n=== Testing Trajectory Writer ===")
from molpy.io.trajectory.lammps import LammpsTrajectoryWriter

traj_file = output_dir / "water_traj.lammpstrj"

try:
    with LammpsTrajectoryWriter(traj_file, atom_style="full") as writer:
        for frame in frames:
            timestep = getattr(frame, 'timestep', 0)
            writer.write_frame(frame, timestep)
    print(f"✅ Trajectory written to: {traj_file}")
    print(f"File size: {traj_file.stat().st_size} bytes")
except Exception as e:
    print(f"❌ Error writing trajectory: {e}")
    import traceback
    traceback.print_exc()

# 显示文件内容的一部分
if traj_file.exists():
    print("\n=== Trajectory File Content (first 50 lines) ===")
    with open(traj_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:50]):
            print(f"{i+1:2d}: {line.rstrip()}")
        if len(lines) > 50:
            print(f"... ({len(lines) - 50} more lines)")

# 测试trajectory reader
print("\n=== Testing Trajectory Reader ===")
try:
    from molpy.io.trajectory.lammps import LammpsTrajectoryReader
    
    with LammpsTrajectoryReader(traj_file) as reader:
        print(f"Number of frames in trajectory: {len(reader)}")
        
        # 读取第一帧
        frame0 = reader.read_frame(0)
        print(f"Frame 0 timestep: {getattr(frame0, 'timestep', 'unknown')}")
        print(f"Frame 0 atoms: {frame0['atoms']}")
        print(f"Frame 0 box: {frame0.box}")
        
        # 读取最后一帧
        if len(reader) > 1:
            frame_last = reader.read_frame(len(reader) - 1)
            print(f"Last frame timestep: {getattr(frame_last, 'timestep', 'unknown')}")
            
        # 测试迭代器
        print("\nIterating through all frames:")
        for i, frame in enumerate(reader):
            timestep = getattr(frame, 'timestep', 'unknown')
            atoms = frame['atoms']
            x_coords = atoms['x'].values if hasattr(atoms['x'], 'values') else atoms['x']
            print(f"  Frame {i}: timestep={timestep}, first atom x={x_coords[0]:.3f}")
            
except Exception as e:
    print(f"❌ Error reading trajectory: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 Trajectory API test completed!")
