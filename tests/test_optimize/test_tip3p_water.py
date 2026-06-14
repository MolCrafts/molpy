"""Test TIP3P water molecule optimization through a molrs ForceField."""

import numpy as np

from molpy.core.atomistic import Atomistic
from molpy.core.forcefield import AngleHarmonicStyle, BondHarmonicStyle, ForceField
from molpy.optimize import LBFGS, ForceFieldPotential


def _build_tip3p_forcefield() -> ForceField:
    """TIP3P bond + angle force field (nm / kJ units, matching tip3p.xml).

    Bond: O-H length 0.09572 nm, k = 462750.4 kJ/mol/nm².
    Angle: H-O-H 104.52°, k = 836.8 kJ/mol/rad².
    """
    ff = ForceField(name="tip3p", units="real")
    astyle = ff.def_atomstyle("full")
    o_type = astyle.def_type("OW", mass=15.999)
    h_type = astyle.def_type("HW", mass=1.008)

    bondstyle = ff.def_style(BondHarmonicStyle())
    bondstyle.def_type(o_type, h_type, k=462750.4, r0=0.09572)

    anglestyle = ff.def_style(AngleHarmonicStyle())
    anglestyle.def_type(h_type, o_type, h_type, k=836.8, theta0=104.5199948597)

    return ff


def test_tip3p_water_optimization():
    """LBFGS optimization of a distorted TIP3P water molecule via its force field.

    The molecule starts in a distorted right-angle configuration with stretched
    bonds; the force field is built natively in molrs and evaluated each step
    through :class:`ForceFieldPotential`. The optimized geometry must recover the
    TIP3P bond length and H-O-H angle. (TIP3P uses nm units, not Å.)
    """
    struct = Atomistic()
    # Atoms carry string type labels that match the force-field type names so
    # molrs can match topology rows to FF types when compiling the potentials.
    o_atom = struct.def_atom(symbol="O", xyz=[0.0, 0.0, 0.0], type="OW")
    h1_atom = struct.def_atom(symbol="H", xyz=[0.12, 0.0, 0.0], type="HW")
    h2_atom = struct.def_atom(symbol="H", xyz=[0.0, 0.12, 0.0], type="HW")

    struct.def_bond(o_atom, h1_atom, type="OW-HW")
    struct.def_bond(o_atom, h2_atom, type="OW-HW")
    struct.def_angle(h1_atom, o_atom, h2_atom, type="HW-OW-HW")

    ff = _build_tip3p_forcefield()
    potential = ForceFieldPotential(ff)

    opt = LBFGS(potential, maxstep=0.005, memory=20)
    result = opt.run(struct, fmax=5.0, steps=500, inplace=True)

    o_pos = struct.xyz[0]
    h1_pos = struct.xyz[1]
    h2_pos = struct.xyz[2]

    bond1_vec = h1_pos - o_pos
    bond2_vec = h2_pos - o_pos
    bond1_length = np.linalg.norm(bond1_vec)
    bond2_length = np.linalg.norm(bond2_vec)

    assert abs(bond1_length - 0.09572) < 0.01, (
        f"O-H1 bond should be ~0.09572 nm, got {bond1_length:.5f} nm"
    )
    assert abs(bond2_length - 0.09572) < 0.01, (
        f"O-H2 bond should be ~0.09572 nm, got {bond2_length:.5f} nm"
    )

    cos_angle = np.dot(bond1_vec, bond2_vec) / (bond1_length * bond2_length)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    assert abs(angle_rad - 1.82421813418) < 0.15, (
        f"H-O-H angle should be ~1.824 rad (~104.52°), got {angle_rad:.4f} rad"
    )

    assert result.converged or result.nsteps == 500


if __name__ == "__main__":
    test_tip3p_water_optimization()
