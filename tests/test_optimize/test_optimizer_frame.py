"""Frame-native optimizer behaviour: run accepts a molrs.Frame, mutates its
coordinate columns in place (inplace=True) or leaves the input untouched
(inplace=False), and returns an OptimizationResult carrying a molrs.Frame."""

import molrs
import numpy as np

from molpy.core.atomistic import Atomistic
from molpy.core.forcefield import AngleHarmonicStyle, BondHarmonicStyle, ForceField
from molpy.optimize import LBFGS, ForceFieldPotential, OptimizationResult


def _tip3p_ff() -> ForceField:
    ff = ForceField(name="tip3p", units="real")
    astyle = ff.def_atomstyle("full")
    o = astyle.def_type("OW", mass=15.999)
    h = astyle.def_type("HW", mass=1.008)
    ff.def_style(BondHarmonicStyle()).def_type(o, h, k=462750.4, r0=0.09572)
    ff.def_style(AngleHarmonicStyle()).def_type(h, o, h, k=836.8, theta0=104.5199948597)
    return ff


def _water_frame() -> "molrs.Frame":
    s = Atomistic()
    o = s.def_atom(symbol="O", xyz=[0.0, 0.0, 0.0], type="OW")
    h1 = s.def_atom(symbol="H", xyz=[0.12, 0.0, 0.0], type="HW")
    h2 = s.def_atom(symbol="H", xyz=[0.0, 0.12, 0.0], type="HW")
    s.def_bond(o, h1, type="OW-HW")
    s.def_bond(o, h2, type="OW-HW")
    s.def_angle(h1, o, h2, type="HW-OW-HW")
    return s.to_frame()


def _coords(frame: "molrs.Frame") -> np.ndarray:
    a = frame["atoms"]
    return np.column_stack([np.asarray(a["x"]), np.asarray(a["y"]), np.asarray(a["z"])])


def test_run_accepts_frame_and_returns_frame_result():
    frame = _water_frame()
    opt = LBFGS(ForceFieldPotential(_tip3p_ff()), maxstep=0.005, memory=20)
    result = opt.run(frame, fmax=5.0, steps=50, inplace=True)
    assert isinstance(result, OptimizationResult)
    assert isinstance(result.frame, molrs.Frame)


def test_inplace_true_mutates_input_frame():
    frame = _water_frame()
    before = _coords(frame).copy()
    opt = LBFGS(ForceFieldPotential(_tip3p_ff()), maxstep=0.005, memory=20)
    result = opt.run(frame, fmax=5.0, steps=50, inplace=True)
    assert result.frame is frame
    assert not np.allclose(_coords(frame), before)  # coordinates moved


def test_inplace_false_leaves_input_unchanged():
    frame = _water_frame()
    before = _coords(frame).copy()
    opt = LBFGS(ForceFieldPotential(_tip3p_ff()), maxstep=0.005, memory=20)
    result = opt.run(frame, fmax=5.0, steps=50, inplace=False)
    assert result.frame is not frame
    assert np.allclose(_coords(frame), before)  # input untouched
    assert not np.allclose(_coords(result.frame), before)  # copy optimized


def test_position_bridge_roundtrip():
    frame = _water_frame()
    opt = LBFGS(ForceFieldPotential(_tip3p_ff()))
    pos = opt.get_positions(frame)
    opt.set_positions(frame, pos)
    assert np.allclose(opt.get_positions(frame), pos)
