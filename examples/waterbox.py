# Standard Python script for waterbox generation
import molpy as mp
import numpy as np
from pathlib import Path

data_path = Path("./data/waterbox")

class SPCE(mp.AtomicStruct):

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

ff = mp.ForceField(name="spce", unit="real")
atomstyle = ff.def_atomstyle("full")
o_type = atomstyle.def_type("O", mass=15.999)
h_type = atomstyle.def_type("H", mass=1.008)

bondstyle = ff.def_bondstyle("harmonic")
bondstyle.def_type(
    o_type, h_type, k=1000.0, r0=1.0
)

anglestyle = ff.def_anglestyle("harmonic")
anglestyle.def_type(
    h_type, o_type, h_type, k=1000.0, theta0=109.47
)

pairstyle = ff.def_pairstyle("lj/charmm/coul/long", inner=9.0, outer=10.0, cutoff=10.0, mix="arithmetic")
pairstyle.def_type(
    o_type, o_type, epsilon=0.1554, sigma=3.1656
)
pairstyle.def_type(
    h_type, h_type, epsilon=0.0, sigma=0.0
)

system = mp.System()
system.set_forcefield(ff)

# 设置盒子大小 (3x3x3 的小盒子用于测试)
box_size = 10.0
system.def_box(np.diag([box_size, box_size, box_size]))

# 创建 3x3x3 = 27 个水分子
n_molecules = 0
spacing = 3.0  # 分子间距

# 创建一个模板分子并使用typifier
spce = mp.SpatialWrapper(SPCE(name="spce_template", molid=1))

for i in range(3):
    for j in range(3):
        for k in range(3):
            # 计算唯一的molid
            molid = n_molecules + 1
            
            # 使用模板的__call__()方法创建新实例
            water_mol = spce(molid=molid)
            
            # 手动更新所有原子的molid
            for atom in water_mol.atoms:
                atom['molid'] = molid
            
            # 用 SpatialWrapper 包装并移动到指定位置
            position = [i * spacing, j * spacing, k * spacing]
            spatial_water = mp.SpatialWrapper(water_mol)
            spatial_water.move(position)
            
            # 添加到系统（移动后的分子已经通过wrapper更新了原始对象的坐标）
            system.add_struct(water_mol)
            n_molecules += 1

print(f"创建了 {n_molecules} 个水分子")

# 检查系统内容 - 使用正确的属性访问方式
total_atoms = 0
total_bonds = 0
total_angles = 0

if hasattr(system, '_struct'):
    total_atoms = sum(len(struct.atoms) for struct in system._struct)
    total_bonds = sum(len(struct.bonds) for struct in system._struct)
    total_angles = sum(len(struct.angles) for struct in system._struct)
    
    print(f"系统包含 {total_atoms} 个原子")
    print(f"系统包含 {total_bonds} 个键")
    print(f"系统包含 {total_angles} 个角")
else:
    print("系统结构信息不可用")

# Direct export to LAMMPS
print("=== Direct LAMMPS Export ===")

# Create output directory
data_path.mkdir(exist_ok=True)

# Convert system to frame
frame = system.to_frame()
print(frame["atoms"]["xyz"])
print(f"Frame generated with {len(frame['atoms'])} atoms")

# Export LAMMPS data file
data_file = data_path / "water_box.data"
mp.io.write_lammps_data(data_file, frame)
print(f"✅ LAMMPS data file: {data_file}")

# Export LAMMPS forcefield file  
ff_file = data_path / "water_box.ff"
mp.io.write_lammps_forcefield(ff_file, ff)
print(f"✅ LAMMPS forcefield file: {ff_file}")

# Show created files
print(f"\nCreated files:")
for f in data_path.glob("water_box.*"):
    print(f"  - {f.name} ({f.stat().st_size} bytes)")

print(f"\n🎉 LAMMPS files exported successfully!")
print(f"📁 Location: {data_path}")
print(f"💧 System: {n_molecules} water molecules, {total_atoms} atoms")


