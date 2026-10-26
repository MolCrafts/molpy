//! Geometric Lennard-Jones mixing (OPLS-AA's), as OpenMM prices it.
//!
//! OpenMM's `app.ForceField` mixes every `<NonbondedForce>` Lennard-Jones
//! pair Lorentz-Berthelot (`σᵢⱼ = (σᵢ + σⱼ)/2`, `εᵢⱼ = √(εᵢεⱼ)`), its 1-4
//! exceptions included, and ignores foyer's `<ForceField
//! combining_rule="geometric">` attribute. Nothing else in its XML schema
//! states a geometric σ: a `<CustomNonbondedForce>` can mix per-type σ and ε
//! any way, but excludes whole bonded neighbours (`bondCutoff`) and scales
//! none, so it cannot price OPLS's ½-weighted 1-4 pairs; a
//! `<LennardJonesForce>` prices them but mixes arithmetic, and would need an
//! `<NBFixPair>` for every type pair (OPLS-AA's 813 types: 237 658 rows).
//!
//! So the writer keeps the `<NonbondedForce>` rows (charges, σ, ε,
//! `coulomb14scale`, `lj14scale`) and the root's `combining_rule`, and adds
//! [`SCRIPT`], a `<Script>` OpenMM runs at the end of `createSystem` —
//! foyer's patch of the `System`:
//!
//! - a `CustomNonbondedForce` named `GeometricLennardJones` prices the Lennard-Jones
//!   of every pair at `σᵢⱼ = √(σᵢσⱼ)`, `εᵢⱼ = √(εᵢεⱼ)`, with the
//!   `NonbondedForce`'s cutoff, switching and dispersion correction, its
//!   method's periodicity (`NoCutoff`, `CutoffNonPeriodic`, or
//!   `CutoffPeriodic` under `CutoffPeriodic`, `Ewald` and `PME`; `LJPME` is
//!   refused), and an exclusion per exception;
//! - the `NonbondedForce` keeps the charges (and so `Ewald` / `PME`) at
//!   `ε = 0`;
//! - each 1-4 exception keeps its `lj14scale·√(εᵢεⱼ)` and takes `σ = √(σᵢσⱼ)`.
//!
//! The script runs once per `System` (a second copy, from a second
//! molrs-written file, finds the force and does nothing). The reader reads
//! this exact script as the geometric rule it applies, and refuses every other
//! `<Script>`.

/// The `<Script>` body (Python, run by OpenMM's `createSystem` with the
/// `System` as `sys`).
pub(crate) const SCRIPT: &str = r#"# Geometric Lennard-Jones mixing (combining_rule="geometric"), written by molrs.
import math as _math
import openmm as _mm
from openmm import unit as _u
_methods = {
    _mm.NonbondedForce.NoCutoff: _mm.CustomNonbondedForce.NoCutoff,
    _mm.NonbondedForce.CutoffNonPeriodic: _mm.CustomNonbondedForce.CutoffNonPeriodic,
    _mm.NonbondedForce.CutoffPeriodic: _mm.CustomNonbondedForce.CutoffPeriodic,
    _mm.NonbondedForce.Ewald: _mm.CustomNonbondedForce.CutoffPeriodic,
    _mm.NonbondedForce.PME: _mm.CustomNonbondedForce.CutoffPeriodic,
}
_done = any(_f.getName() == "GeometricLennardJones" for _f in sys.getForces())
for _nb in [] if _done else [_f for _f in sys.getForces() if isinstance(_f, _mm.NonbondedForce)]:
    if _nb.getNonbondedMethod() not in _methods:
        raise ValueError("combining_rule=\"geometric\": LJPME mixes its dispersion arithmetic; use PME")
    _lj = _mm.CustomNonbondedForce("4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)")
    _lj.setName("GeometricLennardJones")
    _lj.addPerParticleParameter("sigma")
    _lj.addPerParticleParameter("epsilon")
    _lj.setNonbondedMethod(_methods[_nb.getNonbondedMethod()])
    _lj.setCutoffDistance(_nb.getCutoffDistance())
    _lj.setUseSwitchingFunction(_nb.getUseSwitchingFunction())
    _lj.setSwitchingDistance(_nb.getSwitchingDistance())
    _lj.setUseLongRangeCorrection(_nb.getUseDispersionCorrection())
    _sigma = []
    for _i in range(_nb.getNumParticles()):
        _q, _s, _e = _nb.getParticleParameters(_i)
        _sigma.append(_s.value_in_unit(_u.nanometer))
        _lj.addParticle([_sigma[-1], _e.value_in_unit(_u.kilojoule_per_mole)])
        _nb.setParticleParameters(_i, _q, _s, 0.0)
    for _k in range(_nb.getNumExceptions()):
        _a, _b, _qq, _s, _e = _nb.getExceptionParameters(_k)
        _lj.addExclusion(_a, _b)
        if _e.value_in_unit(_u.kilojoule_per_mole) != 0.0:
            _nb.setExceptionParameters(_k, _a, _b, _qq, _math.sqrt(_sigma[_a] * _sigma[_b]), _e)
    sys.addForce(_lj)
"#;
