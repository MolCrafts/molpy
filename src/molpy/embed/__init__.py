"""3D coordinate generation for molpy molecules.

Thin Python wrapper around the molrs Rust ``embed`` pipeline, exposed via the
``molrs`` binary extension. The heavy lifting — fragment / distance-geometry
build, energy minimisation, rotor search, stereo guard — runs inside molrs;
this module only marshals :class:`molpy.Atomistic` across that boundary.

The main-trunk ``molpy.compute.Generate3D`` is a thin :class:`Compute` wrapper
over this function. The RDKit adapter (:mod:`molpy.adapter.rdkit`), which also
hosts the optional RDKit ``Generate3D`` / ``OptimizeGeometry`` operators,
remains available as an optional external backend.
"""

from __future__ import annotations

import molrs

from molpy.core.atomistic import Atomistic

from .report import EmbedReport, StageReport

__all__ = ["EmbedReport", "StageReport", "generate_3d"]


def _atomistic_from_graph(g: "molrs.Graph") -> Atomistic:
    """Build a fresh graph-backed :class:`Atomistic` from a molrs result graph.

    Copies every node property (element + coordinates the embedder produced)
    and rebuilds bonds by node index. Replaces the old ``from_molrs`` now that
    ``Atomistic`` *is* a molrs graph — construction goes through ``def_atom`` /
    ``def_bond`` so the molpy-side registries stay in sync with the backing graph.
    """
    out = Atomistic()
    for i in range(g.n_atoms):
        out.def_atom(**{k: g.get_atom_prop(i, k) for k in g.atom_keys(i)})
    atoms = list(out.atoms)
    for b in range(g.n_bonds):
        i, j = g.get_bond_atoms(b)
        attrs = {k: g.get_bond_prop(b, k) for k in g.bond_keys(b)}
        out.def_bond(atoms[i], atoms[j], **attrs)
    return out


def generate_3d(
    mol: Atomistic,
    *,
    speed: str = "medium",
    add_hydrogens: bool = True,
    rng_seed: int | None = None,
) -> tuple[Atomistic, EmbedReport]:
    """Generate 3D coordinates via the molrs ``embed`` Rust pipeline.

    Parameters
    ----------
    mol:
        Input molecular graph. Element symbols and bond orders are required;
        coordinates may be missing.
    speed:
        Quality preset, one of ``"fast"``, ``"medium"``, or ``"better"``.
    add_hydrogens:
        Add explicit hydrogens before embedding.
    rng_seed:
        Optional deterministic RNG seed.

    Returns
    -------
    tuple[Atomistic, EmbedReport]
        Fresh atomistic structure with generated coordinates plus a per-stage
        report. The input ``mol`` is not mutated.
    """
    if len(list(mol.atoms)) == 0:
        raise ValueError("cannot generate 3D structure for empty molecule")

    # ``mol`` is already a molrs graph (Atomistic is-a molrs.Graph), so it can be
    # embedded directly — no to_molrs conversion. Work on a copy so the input is
    # not mutated, and fold each atom's formal charge into a ``"formal_charge"``
    # prop so add_hydrogens gets [N+]/[N-] hydrogen counts right (molrs reads
    # that key during valence filling).
    work = mol.copy()
    for i in range(work.n_atoms):
        keys = work.atom_keys(i)
        if "formal_charge" in keys:
            continue
        for src in ("formal_charge", "charge"):
            if src in keys:
                v = work.get_atom_prop(i, src)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    work.set_atom_prop(i, "formal_charge", float(v))
                break
    # (Bond orders are stored as float at the source — see
    # _LinkPropProxy._g_set — so molrs reads C=O / aromatic bonds correctly.)

    native_opts = molrs.EmbedOptions(
        speed=speed, add_hydrogens=bool(add_hydrogens), seed=rng_seed
    )
    result = molrs.generate_3d(work, native_opts)
    out = _atomistic_from_graph(result.mol)
    report = EmbedReport.from_native(result.report)
    return out, report
