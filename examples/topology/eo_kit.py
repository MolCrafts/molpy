"""Shared ethylene-glycol kit for every topology example.

Guide: docs/user-guide/topology/index.md

All scripts under ``examples/topology/`` import from here so chemistry stays
consistent: one EO template, one ether reaction, one crosslink reaction.
"""

from __future__ import annotations

import molpy as mp
from molpy.builder.assembly import (
    MonomerLibrary,
    PolymerBuilder,
    ResiduePlacer,
    SiteMap,
)
from molpy.conformer import Conformer
from molpy.core import fields
from molpy.parser import parse_molecule

# Main-chain growth (ether condensation).
ETHER = "[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]"

# Statistical C–C crosslink after SITE x / h are marked.
XLINK = "[C;%x:1][H;%h].[C;%x:2][H;%h]>>[C:1][C:2]"

# End-link / agent coupling can reuse ETHER when ends carry a/b hydroxyls.
# Second-network crosslink uses a distinct site namespace.
XLINK2 = "[C;%y:1][H;%k].[C;%y:2][H;%k]>>[C:1][C:2]"


def ethylene_glycol(*, seed: int = 42) -> mp.Atomistic:
    """Bifunctional EO: OCCO, hydroxyl O labelled a / b."""
    eo, _ = Conformer(add_hydrogens=True, seed=seed).generate(parse_molecule("OCCO"))
    SiteMap(eo).label_elements("O", "a", "b")
    return eo


def monofunctional_cap(*, end: str = "b", seed: int = 7) -> mp.Atomistic:
    """Single-OH cap (methanol).

    ``end="a"`` starts a path (first reactant); ``end="b"`` terminates it
    (second reactant). Telechelics need one of each.
    """
    if end not in ("a", "b"):
        raise ValueError("end must be 'a' or 'b'")
    cap, _ = Conformer(add_hydrogens=True, seed=seed).generate(parse_molecule("CO"))
    SiteMap(cap).label_elements("O", end)
    return cap


def trifunctional_core(*, seed: int = 1) -> mp.Atomistic:
    """Star / branch core: glycerol OCC(O)CO, three O SITE a."""
    core, _ = Conformer(add_hydrogens=True, seed=seed).generate(
        parse_molecule("OCC(O)CO")
    )
    oxygens = [a for a in core.atoms if a.get(fields.ELEMENT) == "O"]
    if len(oxygens) < 3:
        raise RuntimeError("expected three oxygens on OCC(O)CO")
    SiteMap(core).label_atoms(oxygens[:3], "a", "a", "a")
    return core


def tetrafunctional_agent(*, seed: int = 2) -> mp.Atomistic:
    """Four-arm agent: pentaerythritol-like C(CO)4 motif via C(CO)(CO)(CO)CO."""
    # 2,2-bis(hydroxymethyl)propane-1,3-diol ≈ four primary OH
    agent, _ = Conformer(add_hydrogens=True, seed=seed).generate(
        parse_molecule("C(CO)(CO)(CO)CO")
    )
    oxygens = [a for a in agent.atoms if a.get(fields.ELEMENT) == "O"]
    if len(oxygens) < 4:
        raise RuntimeError("expected four oxygens on C(CO)(CO)(CO)CO")
    SiteMap(agent).label_atoms(oxygens[:4], "a", "a", "a", "a")
    return agent


def branch_unit(*, seed: int = 3) -> mp.Atomistic:
    """Comb junction: three OH (two chain + one graft) as a,a,b."""
    unit, _ = Conformer(add_hydrogens=True, seed=seed).generate(
        parse_molecule("OCC(O)CO")
    )
    oxygens = [a for a in unit.atoms if a.get(fields.ELEMENT) == "O"]
    SiteMap(unit).label_atoms(oxygens[:3], "a", "a", "b")
    return unit


def eo_builder(
    *,
    extra: dict[str, mp.Atomistic] | None = None,
    seed: int = 42,
) -> PolymerBuilder:
    library: dict[str, mp.Atomistic] = {"EO": ethylene_glycol(seed=seed)}
    if extra:
        library.update(extra)
    return PolymerBuilder(
        MonomerLibrary(library),
        mp.Reaction(ETHER),
        placer=ResiduePlacer(),
    )


def full_library(*, seed: int = 42) -> dict[str, mp.Atomistic]:
    """All named templates used across the topology suite."""
    return {
        "EO": ethylene_glycol(seed=seed),
        "CAPA": monofunctional_cap(end="a", seed=seed + 1),
        "CAPB": monofunctional_cap(end="b", seed=seed + 2),
        "X3": trifunctional_core(seed=seed + 3),
        "X4": tetrafunctional_agent(seed=seed + 4),
        "BR": branch_unit(seed=seed + 5),
    }


def report(name: str, polymer: mp.Atomistic) -> None:
    n_at = polymer.n_atoms
    n_bd = len(list(polymer.bonds))
    res_ids = {
        int(a[fields.RES_ID]) for a in polymer.atoms if a.get(fields.RES_ID) is not None
    }
    n_res = len(res_ids) if res_ids else 0
    shape = "cyclic" if n_res and n_bd >= n_at else "acyclic"
    print(
        f"{name:16s}  residues={n_res:3d}  atoms={n_at:4d}  bonds={n_bd:4d}  ({shape})"
    )


def mark_backbone_crosslink_sites(
    strand: mp.Atomistic,
    *,
    step: int = 2,
    site: str = "x",
    leaving: str = "h",
) -> list:
    carbons = [
        a
        for a in strand.atoms
        if a.get(fields.ELEMENT) == "C"
        and any(n.get(fields.ELEMENT) == "H" for n in strand.get_neighbors(a))
    ]
    return SiteMap(strand).every_nth(
        carbons, step, site, leaving=leaving, fold_charge=True
    )


def mark_residue_crosslink_sites(
    strand: mp.Atomistic,
    res_names: set[str],
    *,
    site: str = "x",
    leaving: str = "h",
) -> list:
    """Mark carbons only on residues whose RES_NAME is in ``res_names``."""
    carbons = [
        a
        for a in strand.atoms
        if str(a.get(fields.RES_NAME)) in res_names
        and a.get(fields.ELEMENT) == "C"
        and any(n.get(fields.ELEMENT) == "H" for n in strand.get_neighbors(a))
    ]
    return SiteMap(strand).every_nth(
        carbons, 1, site, leaving=leaving, fold_charge=True
    )


def mark_end_hydroxyls(
    strand: mp.Atomistic,
    *,
    site: str = "x",
    leaving: str = "h",
) -> list:
    """Mark remaining hydroxyl O (still bonded to H) as crosslink sites."""
    ends = [
        a
        for a in strand.atoms
        if a.get(fields.ELEMENT) == "O"
        and any(n.get(fields.ELEMENT) == "H" for n in strand.get_neighbors(a))
    ]
    return SiteMap(strand).every_nth(ends, 1, site, leaving=leaving, fold_charge=True)
