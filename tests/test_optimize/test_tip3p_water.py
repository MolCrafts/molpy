"""Test TIP3P water molecule optimization with typifier."""

import numpy as np
import pytest

from molpy.core.atomistic import Atomistic
from molpy.optimize import LBFGS
from molpy.optimize.potential_wrappers import (
    AnglePotentialWrapper,
    BondPotentialWrapper,
)
from molpy.potential.angle import AngleHarmonic
from molpy.potential.base import Potentials
from molpy.potential.bond import BondHarmonic


def test_tip3p_water_optimization():
    """Test LBFGS optimization of distorted TIP3P water molecule.

    Creates a water molecule with:
    - 3 atoms (O-H-H) in distorted right-angle configuration
    - Bond and angle potentials from TIP3P force field
    - Uses manual type assignment
    - Verifies final structure has reasonable bond lengths and angle

    Note: TIP3P uses nm units, not Å
    """
    # Create distorted water molecule (O at origin, Hs stretched in nm units)
    struct = Atomistic()
    o_atom = struct.def_atom(symbol="O", xyz=[0.0, 0.0, 0.0])
    h1_atom = struct.def_atom(
        symbol="H", xyz=[0.12, 0.0, 0.0]
    )  # 1.2 Å = 0.12 nm, stretched
    h2_atom = struct.def_atom(
        symbol="H", xyz=[0.0, 0.12, 0.0]
    )  # 1.2 Å = 0.12 nm, stretched

    # Define bonds
    b1 = struct.def_bond(o_atom, h1_atom)
    b2 = struct.def_bond(o_atom, h2_atom)

    # Define angle
    a1 = struct.def_angle(h1_atom, o_atom, h2_atom)

    # Manually assign types (for simplicity, not using typifier yet)
    o_atom["type"] = 0  # tip3p-O
    h1_atom["type"] = 1  # tip3p-H
    h2_atom["type"] = 1  # tip3p-H
    b1["type"] = 0  # O-H bond type
    b2["type"] = 0  # O-H bond type
    a1["type"] = 0  # H-O-H angle type

    # TIP3P force field parameters from tip3p.xml
    # Bond: O-H length=0.09572 nm, k=462750.4 kJ/mol/nm²
    # Angle: H-O-H angle=1.82421813418 rad = 104.5199948597°, k=836.8 kJ/mol/rad²
    #
    # NOTE: Angle potential internally uses DEGREES as the standard unit
    # Convert from radians in XML: 1.82421813418 rad = 104.5199948597°

    bond_potential = BondHarmonic(
        k=np.array([462750.4]),  # kJ/mol/nm²
        r0=np.array([0.09572]),  # nm
    )

    angle_potential = AngleHarmonic(
        k=np.array([836.8]),  # kJ/mol/rad²
        theta0=np.array([104.5199948597]),  # degrees (converted from 1.82421813418 rad)
    )

    # Wrap potentials for Frame interface
    bond_wrapped = BondPotentialWrapper(bond_potential)
    angle_wrapped = AnglePotentialWrapper(angle_potential)

    # Combine potentials
    combined = Potentials([bond_wrapped, angle_wrapped])

    # Debug callback to track progress
    energies_log = []

    def log_progress(opt, struct):
        e_total = opt.get_energy(struct)
        forces = opt.get_forces(struct)
        fmax = np.max(np.abs(forces))

        # Calculate current angle
        bond1 = struct.xyz[1] - struct.xyz[0]
        bond2 = struct.xyz[2] - struct.xyz[0]
        cos_a = np.dot(bond1, bond2) / (np.linalg.norm(bond1) * np.linalg.norm(bond2))
        angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

        energies_log.append((e_total, fmax, angle))
        if len(energies_log) % 50 == 0:
            print(
                f"Step {len(energies_log)}: E={e_total:.2f}, fmax={fmax:.2f}, angle={angle:.1f}°"
            )

    # Optimize
    opt = LBFGS(combined, maxstep=0.005, memory=20)  # Small steps in nm units
    opt.attach(log_progress, interval=1)
    result = opt.run(struct, fmax=5.0, steps=500, inplace=True)

    # Check final geometry
    o_pos = struct.xyz[0]
    h1_pos = struct.xyz[1]
    h2_pos = struct.xyz[2]

    # Check bond lengths (should be close to 0.09572 nm in TIP3P)
    bond1_vec = h1_pos - o_pos
    bond2_vec = h2_pos - o_pos
    bond1_length = np.linalg.norm(bond1_vec)
    bond2_length = np.linalg.norm(bond2_vec)

    print(f"Final bond lengths: {bond1_length:.5f} nm, {bond2_length:.5f} nm")
    print(f"                    ({bond1_length * 10:.4f} Å, {bond2_length * 10:.4f} Å)")
    assert (
        abs(bond1_length - 0.09572) < 0.01
    ), f"O-H1 bond should be ~0.09572 nm, got {bond1_length:.5f} nm"
    assert (
        abs(bond2_length - 0.09572) < 0.01
    ), f"O-H2 bond should be ~0.09572 nm, got {bond2_length:.5f} nm"

    # Check angle (should be close to 104.52° = 1.824 rad)
    cos_angle = np.dot(bond1_vec, bond2_vec) / (bond1_length * bond2_length)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    print(f"Final H-O-H angle: {angle_rad:.4f} rad = {angle_deg:.2f}°")
    # TIP3P target: 1.82421813418 rad (~104.52°)
    assert (
        abs(angle_rad - 1.82421813418) < 0.15
    ), f"H-O-H angle should be ~1.824 rad (~104.52°), got {angle_rad:.4f} rad ({angle_deg:.2f}°)"

    print(f"✓ TIP3P water optimized successfully in {result.nsteps} steps")
    print(f"  Final energy: {result.energy:.4f}")
    print(f"  Final fmax: {result.fmax:.4f}")


if __name__ == "__main__":
    test_tip3p_water_optimization()
