# Testing & Scientific Validation Standards

Project-specific test requirements beyond the structure documented in CLAUDE.md
("Testing Guidelines"). Consumed by `/mol:test`, `/mol:impl`, and the `tester` /
`scientist` agents. Migrated from the former local `molpy-tester`, `molpy-test`,
`molpy-review` skills and `molpy-scientist` agent (2026-06-10).

## Coverage targets

- ≥ 80% per module overall; **≥ 90% for `core/`**.
- Tests must be deterministic. Never modify tests to make them pass — fix the
  implementation (unless the test itself is wrong).

## Required test categories per module type

Every new module: happy path, edge cases (empty input, single atom, invalid
parameters), correct return types, and — for copy-first helpers — `.copy()`
isolation (caller input untouched; see the test patterns in CLAUDE.md; the core
data-model API itself mutates in place by design).

Scientific code (`potential/`, `compute/`, `typifier/`, `core/ops/`):

- Numerical validation against known analytical values
  (e.g. argon LJ, SPC/E water).
- Unit consistency: output units match documentation.
- Limiting cases: r→0, r→∞, single atom, uniform distribution.
- Force consistency: F = −dV/dr verified by numerical gradient check.

I/O modules: round-trip (write → read → compare), malformed-input handling with
clear errors, format-spec compliance.

Parsers: known molecules parse correctly; invalid input gives clear errors;
round-trip parse → serialize → parse → compare.

Builders: topology correctness after building (bonds/angles consistent);
external-tool paths marked `@pytest.mark.external`.

## Physical limits that must hold (validation targets)

- Non-bonded: V(r→∞) → 0. Bonded: V(r_eq) = minimum, F(r_eq) = 0.
- RDF: g(r→∞) → 1 in a homogeneous system; coordination N(r) = ∫4πr²ρg(r)dr.
- MSD: MSD(t=0) = 0; MSD ~ 6Dt (3D) at long times.
- Combining rules must match the declared force field
  (Lorentz-Berthelot vs geometric); cutoff handling explicit (shift/switch/truncate).
- Equilibrium sanity values: C–C ≈ 1.54 Å, H–O–H ≈ 104.52°.

## Unit systems & constants (reference)

| System | Energy | Length | Time | Mass | Charge |
|---|---|---|---|---|---|
| real (LAMMPS/AMBER) | kcal/mol | Å | fs | g/mol | e |
| metal (LAMMPS) | eV | Å | ps | g/mol | e |
| GROMACS | kJ/mol | nm | ps | u | e |

Conversions: 1 kcal/mol = 4.184 kJ/mol = 0.04336 eV; 1 Å = 0.1 nm;
kB = 1.987204e-3 kcal/(mol·K); Coulomb constant C = 332.0637 in kcal·Å/(mol·e²).

Common potential forms (watch the convention): harmonic bond
V = (1/2)k(r−r0)² [LAMMPS uses V = K(r−r0)² with K = k/2]; Morse
V = D[1−exp(−α(r−r0))]²; LJ 12-6 V = 4ε[(σ/r)¹²−(σ/r)⁶]; harmonic angle
V = (1/2)k(θ−θ0)²; cosine dihedral V = K[1+d·cos(nφ)]. Every equation must
trace to a published reference (see docs-style.md).
