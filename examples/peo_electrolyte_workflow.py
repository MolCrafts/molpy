"""PEO polymer-electrolyte workflow — eight stages, a LAMMPS data file each.

This example walks the full "BigSMILES -> simulation-ready box" pipeline for a
poly(ethylene oxide) (PEO) electrolyte. **Every stage prints its structure and
writes a LAMMPS data file** (``stepN_*.data``) so each stage can be loaded
directly into OVITO / VMD / LAMMPS to make a figure.

Stages
------
1. BigSMILES input          -> the decoded monomer (one repeat unit)
2. Parse repeat unit        -> the PEO repeat unit  -CH2-CH2-O-  (shown x2)
3. Build polymer chain      -> idealized straight all-trans (-CH2-CH2-O-)n chain
4. Optimize conformer       -> MMFF94 + L-BFGS relaxed 3D chain + topology
5. Pack simulation box      -> Packmol/Molpack-packed PEO chains (+ Li+ / TFSI-)
6. Assign force field       -> OPLS-AA atom types, charges, LJ parameters
7. Export engine-ready      -> LAMMPS / GROMACS / OpenMM input sets
8. Simulation-ready config  -> final periodic electrolyte box

The chain is first laid out as a clean, straight all-trans zigzag (step 3 — the
idealized "artificial" structure that looks good on a figure), then relaxed with
molrs's built-in MMFF94 force field and L-BFGS optimizer (step 4) into a
realistic conformer. The box is then
packed for real with Packmol through molpy's :class:`Molpack`, which preserves
every molecule's full topology with offset connectivity. Packmol must be on
PATH; the OPLS-AA force field ships with molpy. The TFSI- anion uses a 3D
conformer (RDKit/molrs) and is best-effort; if it fails the ion is skipped.

Run from anywhere::

    python examples/peo_electrolyte_workflow.py
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import numpy as np

import molpy
import molrs
from molpy.builder import polymer
from molpy.core.atomistic import Atomistic
from molpy.core.box import Box
from molpy.core.frame import Block, Frame
from molpy.io import write_lammps_data

# ---------------------------------------------------------------------------
# Configuration — tweak these to change the figure
# ---------------------------------------------------------------------------
REPEAT_UNIT = "{[<]CCO[>]}"  # PEO repeat unit, molpy G-BigSMILES dialect
BIGSMILES_LABEL = "O{[>][<]CCO[>][<]}H"  # the human-facing BigSMILES string
DEGREE_OF_POLYMERIZATION = 12  # n in (-CH2-CH2-O-)n
N_CHAINS = 8  # PEO chains packed into the box (doubled; box unchanged)
N_LI = 8  # Li+ ions (doubled)
N_TFSI = 8  # TFSI- ions (doubled)
BOX_LENGTH = 36.0  # cubic box edge, angstrom (unchanged)
SEED = 42
OPT_FMAX = 0.05  # L-BFGS force convergence (eV/A-equivalent in MMFF units)
OPT_MAX_STEPS = 1000  # L-BFGS step cap
PACK_STEPS = 200  # Packmol optimization loops

# Bundled OPLS-AA force field, located via the installed package (CWD-independent).
FF_PATH = Path(molpy.__file__).parent / "data" / "forcefield" / "oplsaa.xml"
# Outputs land next to this script, regardless of where it is run from.
OUT_DIR = Path(__file__).resolve().parent / "peo_workflow_output"


# ---------------------------------------------------------------------------
# Pretty-printing helpers (figure-friendly)
# ---------------------------------------------------------------------------
def banner(step: int, title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\nSTEP {step}.  {title}\n{line}")


def icon(art: str) -> None:
    """Print a small ASCII motif for the step."""
    print(art.strip("\n"))
    print()


def formula(struct: Atomistic) -> str:
    counts = Counter(a["element"] for a in struct.atoms)
    order = ["C", "H", "O", "N", "S", "F", "Li"]
    parts = [f"{el}{counts[el]}" for el in order if counts.get(el)]
    parts += [f"{el}{n}" for el, n in counts.items() if el not in order]
    return "".join(parts)


def coords(struct: Atomistic) -> np.ndarray:
    return np.array([[a["x"], a["y"], a["z"]] for a in struct.atoms], dtype=float)


def radius_of_gyration(xyz: np.ndarray) -> float:
    center = xyz.mean(axis=0)
    return float(np.sqrt(((xyz - center) ** 2).sum(axis=1).mean()))


# ---------------------------------------------------------------------------
# Idealized all-trans linear chain (the "artificial", pre-optimization figure)
# ---------------------------------------------------------------------------
def _norm(v) -> np.ndarray:
    v = np.asarray(v, float)
    length = np.linalg.norm(v)
    return v / length if length > 1e-9 else v


def _adjacency(frame: Frame) -> list[list[int]]:
    bonds = frame["bonds"]
    adj: list[list[int]] = [[] for _ in range(frame["atoms"].nrows)]
    ai = np.asarray(bonds["atomi"]).astype(int)
    aj = np.asarray(bonds["atomj"]).astype(int)
    for i, j in zip(ai, aj):
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _backbone_order(adj: list[list[int]], elements: list[str]) -> list[int]:
    """Walk the heavy-atom (C/O) path from one terminus to the other."""
    heavy = {i for i, e in enumerate(elements) if e != "H"}

    def heavy_neighbors(i: int) -> list[int]:
        return [j for j in adj[i] if j in heavy]

    ends = [i for i in heavy if len(heavy_neighbors(i)) == 1]
    start = ends[0] if ends else next(iter(heavy))
    order, prev, cur = [start], None, start
    while True:
        nxt = [j for j in heavy_neighbors(cur) if j != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        order.append(cur)
    return order


def _place_substituents(
    center: np.ndarray, heavy_dirs: list[np.ndarray], n_sub: int, dist: float
) -> list[np.ndarray]:
    """Tetrahedral placement of ``n_sub`` H atoms around a heavy atom."""
    if n_sub == 0:
        return []
    if len(heavy_dirs) >= 2:  # mid-backbone atom: H's stick out of the plane
        u1, u2 = heavy_dirs[0], heavy_dirs[1]
        bis = _norm(u1 + u2)
        nrm = _norm(np.cross(u1, u2))
        if n_sub == 1:
            return [center + (-bis) * dist]
        beta = np.deg2rad(54.75)
        return [
            center + _norm(-bis * np.cos(beta) + nrm * np.sin(beta)) * dist,
            center + _norm(-bis * np.cos(beta) - nrm * np.sin(beta)) * dist,
        ]
    u = heavy_dirs[0]  # terminal atom: H's on a tetrahedral cone around -u
    axis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, u)) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    e1 = _norm(np.cross(u, axis))
    e2 = _norm(np.cross(u, e1))
    out = []
    for k in range(n_sub):
        phi = 2 * np.pi * k / n_sub + 0.3
        direction = -u / 3.0 + (e1 * np.cos(phi) + e2 * np.sin(phi)) * (
            np.sqrt(8) / 3.0
        )
        out.append(center + _norm(direction) * dist)
    return out


def build_linear_peo(n: int) -> Atomistic:
    """Build a straight, all-trans PEO chain (idealized, pre-optimization).

    Topology (and H-capping) comes from the polymer builder; only the
    coordinates are replaced with an extended zigzag backbone plus tetrahedral
    hydrogens. This is the clean "artificial" chain to show before relaxation.
    """
    chain = polymer(f"{REPEAT_UNIT}|{n}|", optimize=False, random_seed=SEED)
    frame = chain.to_frame()
    elements = list(np.asarray(frame["atoms"]["element"]).astype(str))
    adj = _adjacency(frame)
    backbone = _backbone_order(adj, elements)

    pos = np.zeros((len(elements), 3))
    half_angle = np.deg2rad(111.0 / 2)  # backbone bond angle ~111 deg
    bond_len = {("C", "C"): 1.53, ("C", "O"): 1.43, ("O", "C"): 1.43}
    for k in range(1, len(backbone)):
        a, b = backbone[k - 1], backbone[k]
        length = bond_len.get((elements[a], elements[b]), 1.50)
        dx = length * np.sin(half_angle)
        dy = ((-1) ** (k + 1)) * length * np.cos(half_angle)
        pos[b] = pos[a] + np.array([dx, dy, 0.0])

    heavy = {i for i, e in enumerate(elements) if e != "H"}
    for c in backbone:
        heavy_dirs = [_norm(pos[j] - pos[c]) for j in adj[c] if j in heavy]
        hydrogens = [j for j in adj[c] if elements[j] == "H"]
        dist = 0.97 if elements[c] == "O" else 1.10
        for h, hp in zip(
            hydrogens, _place_substituents(pos[c], heavy_dirs, len(hydrogens), dist)
        ):
            pos[h] = hp

    for atom, p in zip(chain.atoms, pos):
        atom["x"], atom["y"], atom["z"] = float(p[0]), float(p[1]), float(p[2])
    return chain


def mmff_optimize(struct: Atomistic, *, fmax: float, max_steps: int) -> dict:
    """Relax ``struct`` in place with molrs's built-in MMFF94 + L-BFGS optimizer.

    Builds MMFF94 potentials for the molecule, minimizes the Cartesian
    coordinates with L-BFGS, and writes the optimized positions back onto the
    atoms. Returns a small report dict (energy / Rg before and after).
    """
    coords0 = np.asarray(molrs.extract_coords(struct.to_frame())).reshape(-1, 3)
    potentials = molrs.build_mmff_potentials(struct)  # MMFF94
    energy0 = float(potentials.calc_energy(coords0.reshape(-1)))

    optimizer = molrs.LBFGS(potentials, fmax=fmax, max_steps=max_steps)  # L-BFGS
    coords1, report = optimizer.run(coords0)
    coords1 = np.asarray(coords1).reshape(-1, 3)
    energy1 = float(potentials.calc_energy(coords1.reshape(-1)))

    for atom, xyz in zip(struct.atoms, coords1):  # write optimized coords back
        atom["x"], atom["y"], atom["z"] = float(xyz[0]), float(xyz[1]), float(xyz[2])

    return {
        "energy_before": energy0,
        "energy_after": energy1,
        "rg_before": radius_of_gyration(coords0),
        "rg_after": radius_of_gyration(coords1),
        "converged": bool(getattr(report, "converged", False)),
        "steps": int(getattr(report, "n_steps", max_steps)),
        "fmax": float(getattr(report, "final_fmax", float("nan"))),
    }


# ---------------------------------------------------------------------------
# LAMMPS data emission — the artifact every step produces
# ---------------------------------------------------------------------------
def struct_to_frame(
    struct: Atomistic,
    *,
    box_length: float | None = None,
    mol_ids: list[int] | None = None,
    use_ff_type: bool = False,
) -> Frame:
    """Build a LAMMPS-ready Frame (atoms + bonds/angles/dihedrals + box).

    Works off the molrs-native ``to_frame()`` (which supplies a contiguous atom
    order plus 0-based connectivity indices ``atomi``/``atomj``/...), then adds
    the columns the LAMMPS writer needs: a fresh unique ``id``, ``mol_id``,
    ``type`` (OPLS type when ``use_ff_type`` is set and present, else element),
    ``charge``, and element-pattern connectivity ``type`` labels (e.g. ``C-O-C``).
    """
    base = struct.to_frame()
    atoms = base["atoms"]
    cols = set(atoms.keys())
    n = atoms.nrows

    elements = np.asarray(atoms["element"]).astype(str)
    x = np.asarray(atoms["x"]).astype(float)
    y = np.asarray(atoms["y"]).astype(float)
    z = np.asarray(atoms["z"]).astype(float)
    atype = (
        np.asarray(atoms["type"]).astype(str)
        if use_ff_type and "type" in cols
        else elements
    )
    charge = (
        np.asarray(atoms["charge"]).astype(float)
        if "charge" in cols
        else np.zeros(n, dtype=float)
    )

    if box_length is None:  # single molecule — snug cubic box with padding
        span = float(max(np.ptp(x), np.ptp(y), np.ptp(z))) + 6.0
        x, y, z = x - (x.min() - 3.0), y - (y.min() - 3.0), z - (z.min() - 3.0)
        box_len = span
    else:
        box_len = box_length

    if mol_ids is not None and len(mol_ids) == n:
        mol_col = np.array(mol_ids, dtype=int)
    else:
        mol_col = np.ones(n, dtype=int)

    frame = Frame(
        {
            "atoms": Block(
                {
                    "id": np.arange(1, n + 1, dtype=int),  # fresh, unique
                    "mol_id": mol_col,
                    "type": atype,
                    "charge": charge,
                    "element": elements,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
        }
    )

    connectivity = [
        ("bonds", ["atomi", "atomj"]),
        ("angles", ["atomi", "atomj", "atomk"]),
        ("dihedrals", ["atomi", "atomj", "atomk", "atoml"]),
    ]
    for block_name, keys in connectivity:
        if block_name not in base.keys() or base[block_name].nrows == 0:
            continue
        block = base[block_name]
        if not all(key in block.keys() for key in keys):
            continue
        idx_cols = [np.asarray(block[key]).astype(int) for key in keys]
        type_labels = np.array(
            ["-".join(elements[col[r]] for col in idx_cols) for r in range(block.nrows)]
        )
        data = {key: idx_cols[j] for j, key in enumerate(keys)}
        data["type"] = type_labels
        frame[block_name] = Block(data)

    frame.box = Box(np.array([box_len] * 3, dtype=float), origin=np.zeros(3))
    return frame


def write_step(frame: Frame, name: str) -> Path:
    """Write a LAMMPS data file and return its path."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence untyped-improper notices
        write_lammps_data(path, frame, atom_style="full")
    counts = frame["atoms"].nrows
    extras = {
        key: frame[key].nrows
        for key in ("bonds", "angles", "dihedrals")
        if key in frame
    }
    tail = ", ".join(f"{k}={v}" for k, v in extras.items())
    print(
        f"  -> LAMMPS data : {path.name}  ({counts} atoms{', ' + tail if tail else ''})"
    )
    return path


# ===========================================================================
# STEP 1 — BigSMILES input
# ===========================================================================
def step1_bigsmiles() -> None:
    banner(1, "BigSMILES input")
    icon(
        r"""
       "O{[>][<]CCO[>][<]}H"                  -CH2-CH2-O-
        +-------------------+    parse         /===========\
        |  t e x t  string  |  --------->      |  monomer  | n
        +-------------------+                  \===========/
        """
    )
    monomer = polymer(f"{REPEAT_UNIT}|1|", optimize=False, random_seed=SEED)
    print(f"  BigSMILES label : {BIGSMILES_LABEL}")
    print(f"  molpy build form: {REPEAT_UNIT}|n|   (G-BigSMILES dialect)")
    print(f"  decoded monomer : {formula(monomer)}  (one H-capped repeat unit)")
    write_step(struct_to_frame(monomer), "step1_input_monomer.data")


# ===========================================================================
# STEP 2 — Parse repeat unit
# ===========================================================================
def step2_repeat_unit() -> None:
    banner(2, "Parse repeat unit  ->  PEO  -CH2-CH2-O-")
    icon(
        r"""
        [               ]
        [ -CH2-CH2-O-   ]
        [               ] n      (two units shown, linked head-to-tail)
        """
    )
    from molpy.parser import parse_monomer

    unit = parse_monomer(REPEAT_UNIT)
    print(f"  repeat unit (topology) : {formula(unit)}  backbone C-C-O")
    print(f"      atoms : {len(list(unit.atoms))}   bonds : {len(list(unit.bonds))}")

    # A 2-unit fragment carries 3D coordinates -> a viewable bracketed fragment.
    fragment = polymer(f"{REPEAT_UNIT}|2|", optimize=False, random_seed=SEED)
    print(f"  bracketed fragment x2  : {formula(fragment)}")
    write_step(struct_to_frame(fragment), "step2_repeat_unit.data")


# ===========================================================================
# STEP 3 — Build polymer chain
# ===========================================================================
def step3_build_chain() -> Atomistic:
    banner(
        3, f"Build polymer chain  ->  (-CH2-CH2-O-)n  with n={DEGREE_OF_POLYMERIZATION}"
    )
    icon(
        r"""
        H-[ CH2-CH2-O ]-[ CH2-CH2-O ]- ... -[ CH2-CH2-O ]-H
              C   C  O      C   C  O            C   C  O
          all-trans zigzag — straight, idealized (artificial)
        """
    )
    # Hand-built extended all-trans chain: clean and straight for the figure,
    # NOT yet physically relaxed (that is step 4).
    chain = build_linear_peo(DEGREE_OF_POLYMERIZATION)
    counts = Counter(a["element"] for a in chain.atoms)
    xyz = coords(chain)
    print(f"  molecular formula : {formula(chain)}")
    print(f"  total atoms       : {len(list(chain.atoms))}")
    print(f"      carbon  (C)   : {counts.get('C', 0)}   <- backbone CH2")
    print(f"      oxygen  (O)   : {counts.get('O', 0)}   <- ether linkage")
    print(f"      hydrogen(H)   : {counts.get('H', 0)}")
    print(f"  bonds             : {len(list(chain.bonds))}")
    print(f"  geometry          : idealized all-trans (straight), pre-optimization")
    print(f"      Rg            : {radius_of_gyration(xyz):.2f} A  (extended)")
    print(f"      end-to-end x  : {np.ptp(xyz[:, 0]):.1f} A")
    write_step(struct_to_frame(chain), "step3_chain.data")
    return chain


# ===========================================================================
# STEP 4 — Optimize conformer (MMFF + L-BFGS) + topology
# ===========================================================================
def step4_optimize_conformer(chain: Atomistic) -> Atomistic:
    banner(4, "Optimize conformer  (MMFF94 + L-BFGS)  + topology")
    icon(
        r"""
        idealized straight             relaxed conformer
        ===========        MMFF+L-BFGS      ~~~~~~~
        all-trans (stiff)  ----------->    realistic backbone
                                           (clean bonds/angles/dihedrals)
        """
    )
    report = mmff_optimize(chain, fmax=OPT_FMAX, max_steps=OPT_MAX_STEPS)
    print("  MMFF94 + L-BFGS minimization")
    print(
        f"      energy : {report['energy_before']:.1f}  ->  {report['energy_after']:.1f}"
        "   (MMFF units)"
    )
    print(
        f"      Rg     : {report['rg_before']:.2f}  ->  {report['rg_after']:.2f} A"
        "   (relaxed from the idealized straight chain)"
    )
    print(
        f"      L-BFGS : converged={report['converged']}  "
        f"steps={report['steps']}  fmax={report['fmax']:.3f}"
    )

    chain.get_topo()  # (re)derive angles/dihedrals from the bond graph
    xyz = coords(chain)
    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    print("  3D conformation (optimized)")
    print(
        f"      bounding box (A): "
        f"x[{mins[0]:6.2f},{maxs[0]:6.2f}] "
        f"y[{mins[1]:6.2f},{maxs[1]:6.2f}] "
        f"z[{mins[2]:6.2f},{maxs[2]:6.2f}]"
    )
    print("  topology")
    print(f"      bonds     : {len(list(chain.bonds))}")
    print(f"      angles    : {len(list(chain.angles))}")
    print(f"      dihedrals : {len(list(chain.dihedrals))}")
    write_step(struct_to_frame(chain), "step4_optimized_conformer.data")
    return chain


# ===========================================================================
# Ion sub-structures (optional)
# ===========================================================================
def _li_struct() -> Atomistic:
    ion = Atomistic()
    ion.def_atom(element="Li", xyz=[0.0, 0.0, 0.0], charge=1.0)
    return ion


def _tfsi_struct() -> Atomistic:
    from molpy.conformer import Conformer
    from molpy.parser import parse_molecule

    anion = parse_molecule("[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F")
    anion_3d, _ = Conformer(seed=SEED).generate(anion)
    return anion_3d


# ===========================================================================
# STEP 5 — Pack simulation box (real Packmol via Molpack)
# ===========================================================================
def step5_pack_box(chain: Atomistic) -> Frame:
    banner(5, "Pack simulation box  (Packmol / Molpack)")
    icon(
        r"""
        +----------------------------+  periodic box
        |  ~~~~~      ~~~~~           |  ~  PEO chain
        |      ~~~~~        ~~~~~     |  *  Li+
        |   *      ~~~~~        *     |  o  TFSI-
        |       o        *      o    |
        +----------------------------+
        """
    )
    from molpy.pack import InsideBoxConstraint, Molpack

    # Uniform column schema for every target so Packmol can merge species
    # (struct_to_frame yields the same atom columns for chain, Li+, and TFSI-).
    species: list[tuple[str, Atomistic, int]] = [("PEO", chain, N_CHAINS)]
    try:
        species.append(("Li+", _li_struct(), N_LI))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  (Li+ skipped: {exc})")
    try:
        species.append(("TFSI-", _tfsi_struct(), N_TFSI))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  (TFSI- skipped: {exc})")

    box = InsideBoxConstraint(length=np.array([BOX_LENGTH] * 3), origin=np.zeros(3))
    packer = Molpack(workdir=OUT_DIR / "packmol")
    for _name, struct, number in species:
        packer.add_target(struct_to_frame(struct), number=number, constraint=box)

    print(f"  box            : {BOX_LENGTH:.1f}^3 A (periodic)")
    print("  composition")
    for name, _struct, number in species:
        print(f"      {name:<6} x {number}")

    packed = packer.optimize(max_steps=PACK_STEPS, seed=SEED, pbc=[BOX_LENGTH] * 3)
    packed.box = Box(np.array([BOX_LENGTH] * 3, dtype=float), origin=np.zeros(3))

    elem_counts = Counter(str(e) for e in packed["atoms"]["element"])
    print(f"  packed (Packmol): {packed['atoms'].nrows} atoms")
    print(
        "      by element : "
        + ", ".join(f"{el}:{n}" for el, n in sorted(elem_counts.items()))
    )
    n_bonds = packed["bonds"].nrows if "bonds" in packed else 0
    print(f"      topology   : bonds={n_bonds} (per-molecule connectivity preserved)")
    print("  layout         : real Packmol placement, separated, not a network")
    write_step(packed, "step5_packed_box.data")
    return packed


# ===========================================================================
# STEP 6 — Assign force field
# ===========================================================================
def step6_assign_forcefield(chain: Atomistic):
    banner(6, "Assign force field  (OPLS-AA)")
    icon(
        r"""
          [type] [q] [type]            +--------------+
            C ----- O ----- C   ===>   |  ForceField  |
          [type] [q] [type]            |   .xml file  |
                                       +--------------+
        """
    )
    from molpy.io.forcefield.xml import XMLForceFieldReader
    from molpy.typifier import OplsTypifier

    ff = XMLForceFieldReader(str(FF_PATH)).read()
    typed = OplsTypifier(ff).typify(chain)

    print(f"  force field    : OPLS-AA  ({FF_PATH.name})")
    print("  per-atom tags  (type / charge / sigma / epsilon)")
    seen: set[str] = set()
    for atom in typed.atoms:
        atype = str(atom.get("type", "?"))
        if atype in seen:
            continue
        seen.add(atype)
        print(
            f"      {atom['element']:<2} type={atype:<9} "
            f"q={atom.get('charge', 0.0):+.3f}  "
            f"sigma={atom.get('sigma', 0.0):.3f}  "
            f"eps={atom.get('epsilon', 0.0):.4f}"
        )
        if len(seen) >= 8:
            break
    total_q = sum(float(a.get("charge", 0.0)) for a in typed.atoms)
    print(f"  unique atom types: {len({str(a.get('type')) for a in typed.atoms})}")
    print(f"  net charge       : {total_q:+.3f} e")
    write_step(struct_to_frame(typed, use_ff_type=True), "step6_forcefield.data")
    return typed, ff


# ===========================================================================
# STEP 7 — Export engine-ready system
# ===========================================================================
def step7_export(typed: Atomistic, ff):
    banner(7, "Export engine-ready system")
    icon(
        r"""
        +- LAMMPS -+  +- GROMACS -+  +- OpenMM -+
        | .data    |  | .gro/.top |  | .xml/.py |   ==>  minimized box
        | .in      |  | .mdp      |  | .pdb     |
        +----------+  +-----------+  +----------+
        """
    )
    from molpy.io.emit import emit_all

    engines_dir = OUT_DIR / "engine_inputs"
    results = emit_all(typed, ff, engines_dir, prefix="peo_electrolyte")
    for engine in ("lammps", "gromacs", "openmm"):
        names = ", ".join(p.name for p in results.get(engine, []))
        print(f"  {engine.upper():<8}: {names}")
    print(f"  multi-engine dir : {engines_dir}/")
    # The canonical, fully-typed LAMMPS data for this step:
    write_step(struct_to_frame(typed, use_ff_type=True), "step7_engine_ready.data")
    return results


# ===========================================================================
# STEP 8 — Simulation-ready configuration
# ===========================================================================
def step8_summary(chain: Atomistic, packed: Frame, results) -> None:
    banner(8, "Simulation-ready configuration")
    icon(
        r"""
        +----------------------------+
        |  ~~~~~      ~~~~~           |
        |      ~~~~~        ~~~~~     |   [OK]  ready to run
        |   *      ~~~~~        *     |
        +----------------------------+
        """
    )
    n_box = packed["atoms"].nrows
    engines = ", ".join(sorted(results)) if results else "none"
    print(f"  repeat unit    : -CH2-CH2-O-  (PEO)")
    print(f"  chain length   : n = {DEGREE_OF_POLYMERIZATION}  ({formula(chain)})")
    print(f"  chains in box  : {N_CHAINS}")
    print(f"  box            : {BOX_LENGTH:.1f}^3 A, periodic")
    print(f"  atoms in box   : {n_box}")
    print(f"  force field    : OPLS-AA (typed)")
    print(f"  engine inputs  : {engines}")
    # Final clean periodic box (the Packmol-packed deliverable configuration):
    write_step(packed, "step8_final_box.data")
    print("  status         : [OK] periodic polymer-electrolyte box ready")

    print("\n" + "-" * 72)
    print("All LAMMPS data files written to:")
    print(f"  {OUT_DIR}/")
    for path in sorted(OUT_DIR.glob("step*.data")):
        print(f"    {path.name}")


# ===========================================================================
def main() -> None:
    step1_bigsmiles()
    step2_repeat_unit()
    chain = step3_build_chain()
    # step 4 optimizes the chain IN PLACE and returns the same object, so the
    # chain handed to packing below carries the MMFF/L-BFGS-relaxed conformation.
    chain = step4_optimize_conformer(chain)
    packed = step5_pack_box(chain)  # packs the optimized conformer
    typed, ff = step6_assign_forcefield(chain)
    results = step7_export(typed, ff)
    step8_summary(chain, packed, results)


if __name__ == "__main__":
    main()
