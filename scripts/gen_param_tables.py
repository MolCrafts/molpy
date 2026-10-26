#!/usr/bin/env python3
"""Translate the antechamber / AMBER parameter tables into committed Rust data.

molrs parses NO parameter text at runtime. This script reads the upstream tables
from `$AMBERHOME` and emits typed Rust `const`s into
`crates/molrs/src/ff/params/`. The emitted `.rs` is the single in-repo source
of truth: the raw `.DAT` / `.DEF` files are NOT vendored.

Fourteen tables, from two upstream directories:

  dat/antechamber/
    BCCPARM.DAT          -> bccparm.rs        (corrections + CORR aliases)
    BCCPARM_ABCG2.DAT    -> bccparm_abcg2.rs  (corrections + CORR aliases)
    GASPARM.DAT          -> gasparm.rs        (Gasteiger PEOE parameters)
    PARMCHK.DAT          -> parmchk.rs     (parmchk2's EQUA / CORR table)
    PARM_BLBA_GAFF{,2}.DAT -> gaff_empirical.rs (empirical bond / angle constants)
    ATOMTYPE_{BCC,ABCG2,GAS,GFF,GFF2,AMBER,SYBYL}.DEF -> atomtype_*.rs
  dat/leap/parm/
    gaff.dat             -> gaff.rs           (MASS/BOND/ANGLE/DIHE/IMPROPER/NONBON)
    gaff2.dat            -> gaff2.rs          (idem)

The committed tables are AmberTools 26.1's (`gaff2.dat` version 2.2.30):
`--check` regenerates them into a temporary directory and fails unless every
one is byte-identical to the committed file.

All seven `.DEF` files share ONE ATD / WILDATOM grammar, so there is one parser
here and seven outputs. Both `gaff*.dat` are AMBER's sectioned `parm` format, so
they likewise share one parser (`parse_parm`) and two outputs. A `MASS` row
restated identically (gaff2.dat 2.2.30 lists `sp` and `sq` twice) is one row;
restated differently it is an error.

`PARMCHK.DAT` is also read for the ATD tables: it supplies the
`AtdRule.alternate` column (see `parse_parmchk`). The `.DEF` files emit only the
phase-1 name of a conjugated system (`cc`); antechamber renames one colour of
each conjugated system to a partner (`cd`) that no `.DEF` row declares, and
`PARMCHK.DAT` is where that pairing is written down.

The environment (`f9`) and atom-property (`f8`) mini-languages are PRE-PARSED
into a static AST — that is the whole point: molrs walks the AST, it never
re-parses a string. Pattern atom names are resolved here too (`EW` / WILDATOM /
element symbol -> `PatternAtom`), so an unknown name is a generator error rather
than a rule that silently never matches.

The other tables under `ff/params/` are not this script's: `oplsaa.rs` is
generated from GROMACS's `oplsaa.ff` by `cargo mrs-gen-opls`
(`crates/molrs/examples/gen_opls_params.rs`); `oplsaa_typing.rs`, `mmff.rs`, `uff.rs`,
`clpol.rs` and `mod.rs` are hand-maintained source.

Run it where AmberTools, `rustfmt` (which formats the output as `cargo fmt`
does) and molrs (its periodic table names the elements) are installed:

    AMBERHOME=/path/to/amber python3 scripts/gen_param_tables.py
    AMBERHOME=... python3 scripts/gen_param_tables.py --out-dir /tmp/tables
    AMBERHOME=... python3 scripts/gen_param_tables.py --check
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import ClassVar

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "crates/molrs/src/ff/params"

# Every emitted table opens by saying where it came from — which is the ONLY place
# provenance belongs. It is not a directory name, and it is not the table's name:
# "how it arrived" is not "what it is", and these tables are ordinary committed
# source, reviewed, grepped and stepped through like any other.
HEADER = """//! {title}
//!
//! DO NOT HAND-EDIT — regenerate with `scripts/gen_param_tables.py`, which emits
//! this table from AmberTools' own `.DAT` / `.DEF` files. That is where the table
//! came FROM; it is not what the table IS — this is ordinary source, not a build
//! artefact.
//!
//! Source: `$AMBERHOME/{source}` (AmberTools).
"""

#: The two upstream directories this generator reads.
ANTECHAMBER_DIR = "dat/antechamber"
PARM_DIR = "dat/leap/parm"

# rustfmt's `struct_lit_width` default (18) explodes every flat data row across
# six lines, which destroys the one-row-per-line grep-ability that is half the
# point of committing these tables. Pin the tabular consts; the nested ATD rules
# are genuinely tree-shaped and are left to rustfmt.
RUSTFMT_SKIP = "#[rustfmt::skip]"

# --- Element symbol <-> atomic number --------------------------------------
# molrs's periodic table (molpy.core.Element). A name is an element only when
# it is the element's exact symbol (`Cl`, never `CL` or `chlorine`).


@cache
def atomic_number(symbol: str) -> int | None:
    """The atomic number of the element whose symbol is exactly ``symbol``."""
    from molpy.core import Element

    try:
        element = Element(symbol)
    except (ValueError, KeyError):
        return None
    return element.number if element.symbol == symbol else None


def element_symbol(z: int) -> str:
    from molpy.core import Element

    return Element(z).symbol


# Atom-property tokens, mapped to the `AtomProp` Rust variant. The lowercase
# forms (`sb`/`db`/`tb`) count aromatic + delocalized bonds too; the uppercase
# forms are strict. Anything else is a generator error.
ATOM_PROPS = {
    "RG": "Rg", "RG3": "Rg3", "RG4": "Rg4", "RG5": "Rg5", "RG6": "Rg6",
    "RG7": "Rg7", "RG8": "Rg8", "RG9": "Rg9", "RG10": "Rg10",
    "NR": "Nr",
    "AR1": "Ar1", "AR2": "Ar2", "AR3": "Ar3", "AR4": "Ar4", "AR5": "Ar5",
    "SB": "SbStrict", "sb": "SbAny",
    "DB": "DbStrict", "db": "DbAny",
    "TB": "TbStrict", "tb": "TbAny",
    "AB": "Ab", "DL": "Dl",
}

ENV_BOND_TYPES = {
    "any": "Any", "ANY": "Any", "Any": "Any",
    "SB": "Single", "sb": "Single",
    "DB": "Double", "db": "Double",
    "TB": "Triple", "tb": "Triple",
    "AB": "Aromatic", "ab": "Aromatic",
}

FLOAT_RE = re.compile(r"^[+-]?\d+\.\d+$")

# (file, module, const, gaff_namespace)
#
# `gaff_namespace` — is this `.DEF` written in the atom-type namespace that
# PARMCHK.DAT's `equivalent_flag` column describes? Only `-at gaff` / `-at gaff2`
# are, and the column may only be applied to those two.
#
# It is a NAMESPACE fact, not a per-table special case, and it is the same fact
# the WILDATOM aliases already carry: a name means what the FILE that declares it
# says it means (`XB` is `C3 N2 N3 O2 S2 P2` in ATOMTYPE_BCC.DEF and `N P` in
# ATOMTYPE_GFF.DEF). PARMCHK.DAT's lowercase PARM rows are GAFF types, so its
# `cg` is GAFF's inner sp carbon of a conjugated system — but `ATOMTYPE_GAS.DEF`
# ALSO spells a type `cg`, and there it is a guanidinium carbon (`C` bonded to
# three `N3`), which is not half of anything: GASPARM.DAT has no `ch` row at all.
# The two `cg`s are one PARM row upstream and cannot be told apart inside
# PARMCHK.DAT — every other column of that row (mass, group, atomic number) is
# identical — so the scope has to be stated here.
#
# This is the same trap as ATOMTYPE_AMBER.DEF's `CC`/`CD` (parm94's histidine
# carbons, `equivalent_flag` 0), one table over: a shared SPELLING is not a
# shared meaning. The flag column catches the AMBER pair on its own; only the
# namespace catches the GAS one.
ATOMTYPE_FILES = [
    ("ATOMTYPE_BCC.DEF", "atomtype_bcc", "ATOMTYPE_BCC", False),
    ("ATOMTYPE_ABCG2.DEF", "atomtype_abcg2", "ATOMTYPE_ABCG2", False),
    ("ATOMTYPE_GAS.DEF", "atomtype_gas", "ATOMTYPE_GAS", False),
    ("ATOMTYPE_GFF.DEF", "atomtype_gff", "ATOMTYPE_GFF", True),
    ("ATOMTYPE_GFF2.DEF", "atomtype_gff2", "ATOMTYPE_GFF2", True),
    ("ATOMTYPE_AMBER.DEF", "atomtype_amber", "ATOMTYPE_AMBER", False),
    ("ATOMTYPE_SYBYL.DEF", "atomtype_sybyl", "ATOMTYPE_SYBYL", False),
]

#: The conjugated pairing table. Read for its `equivalent_flag` column — the only
#: in-repo source of the phase-2 names (`cd`, `cf`, ...) — AND, since
#: chem-perceive-10, emitted whole as `parmchk.rs`: it is parmchk2's atom-type
#: substitution table (EQUA / CORR rows, per-arity penalties, weights, defaults)
#: and the `improper_flag` column that decides which atoms carry an improper.
PARMCHK_FILE = "PARMCHK.DAT"

# (file, const) — the empirical bond/angle constant tables, one per force field.
# `PARM PC` is the Badger exponent m; `PARM BL` the per-element-pair reference
# length + ln(Kij); `PARM BA` the per-element angle C / Z factors.
BLBA_FILES = [
    ("PARM_BLBA_GAFF.DAT", "EMPIRICAL_GAFF"),
    ("PARM_BLBA_GAFF2.DAT", "EMPIRICAL_GAFF2"),
]

# (file, module, const prefix) — AMBER `parm`-format force-field parameter files.
# One grammar (`parse_parm`), two outputs. Unlike the `.DEF` tables these live
# under `dat/leap/parm/`, because they are LEaP's force fields rather than
# antechamber's typing rules.
PARM_FILES = [
    ("gaff.dat", "gaff", "GAFF"),
    ("gaff2.dat", "gaff2", "GAFF2"),
]

class GrammarError(Exception):
    """An upstream construct this generator refuses to guess at."""


# ---------------------------------------------------------------------------
# Shared lexing helpers
# ---------------------------------------------------------------------------

def is_comment(line: str) -> bool:
    """True for the banner / rule / comment lines every antechamber table carries."""
    return not line or line[0] in "#-=" or line.startswith("//")


def rust_float(token: str) -> str:
    """Emit a source float token verbatim when it is already a valid Rust literal.

    Keeps the committed table byte-comparable with the upstream text (`-0.0753`
    stays `-0.0753`), and round-trips exactly. Anything unusual falls back to a
    shortest-round-trip repr so we never silently lose a digit.
    """
    if FLOAT_RE.match(token):
        return token
    return repr(float(token))


def rust_str(s: str) -> str:
    if '"' in s or "\\" in s:
        raise GrammarError(f"unescapable string literal {s!r}")
    return f'"{s}"'


def opt(value: str | None) -> str:
    return f"Some({value})" if value is not None else "None"


# ---------------------------------------------------------------------------
# BCCPARM.DAT / BCCPARM_ABCG2.DAT
# ---------------------------------------------------------------------------

def parse_bccparm(path: Path) -> tuple[list[tuple[str, str, int, str]], list[tuple[str, str]]]:
    """Parse a `BCCPARM*.DAT` -> (correction rows, CORR aliases).

    Row layout: `index  left  right  bond_type  delta`. A `CORR a b` row declares
    that atom type `a` borrows `b`'s corrections. Trailing `#` comments on CORR
    rows are ignored (they are prose, not fields).
    """
    corrections: list[tuple[str, str, int, str]] = []
    aliases: list[tuple[str, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if f[0].upper() == "CORR":
            if len(f) < 3:
                raise GrammarError(f"{path.name}:{lineno}: short CORR row: {raw}")
            aliases.append((f[1], f[2]))
            continue
        if len(f) < 5:
            raise GrammarError(f"{path.name}:{lineno}: short parameter row: {raw}")
        corrections.append((f[1], f[2], int(f[3]), rust_float(f[4])))
    return corrections, aliases


def emit_bccparm(path: Path, prefix: str) -> str:
    corrections, aliases = parse_bccparm(path)
    L = [HEADER.format(
        title=f"AM1-BCC bond charge corrections — `{path.name}`.",
        source=f"{ANTECHAMBER_DIR}/{path.name}",
    )]
    w = L.append
    w("use crate::ff::params::{BccAlias, BccCorrectionRow};")
    w("")
    w(f"/// The {len(corrections)} oriented bond charge corrections of `{path.name}`.")
    w("///")
    w("/// A row `(left, right, bond_type, delta)` adds `+delta` to the `left` atom and")
    w("/// `-delta` to the `right` atom. A bond encountered in the reverse orientation")
    w("/// applies the same magnitude with the sign flipped.")
    w(RUSTFMT_SKIP)
    w(f"pub const {prefix}_CORRECTIONS: &[BccCorrectionRow] = &[")
    for left, right, bond_type, delta in corrections:
        w(f"    BccCorrectionRow {{ left: {rust_str(left)}, right: {rust_str(right)}, "
          f"bond_type: {bond_type}, delta: {delta} }},")
    w("];")
    w("")
    if aliases:
        w(f"/// The {len(aliases)} `CORR` alias rows of `{path.name}`.")
        w("///")
        w("/// `atom_type` carries no corrections of its own; look `reference` up instead.")
    else:
        w(f"/// `{path.name}` declares no `CORR` alias rows.")
    w(RUSTFMT_SKIP)
    w(f"pub const {prefix}_ALIASES: &[BccAlias] = &[")
    for atom_type, reference in aliases:
        w(f"    BccAlias {{ atom_type: {rust_str(atom_type)}, reference: {rust_str(reference)} }},")
    w("];")
    w("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# GASPARM.DAT
# ---------------------------------------------------------------------------

def emit_gasparm(path: Path) -> str:
    """Parse `GASPARM.DAT` -> Gasteiger PEOE rows.

    Columns are `a b c d formal_charge`. `a`/`b`/`c` are the electronegativity
    polynomial coefficients; `d` is chi+ (the normalisation denominator, NOT a
    quartic coefficient); `formal_charge` is the seed charge q0. The three
    meanings get three distinct field names — never an anonymous `[f64; 5]`.
    """
    rows = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if f[0] != "GASPARM":
            raise GrammarError(f"{path.name}:{lineno}: unknown row kind {f[0]!r}")
        if len(f) < 7:
            raise GrammarError(f"{path.name}:{lineno}: short GASPARM row: {raw}")
        rows.append((f[1], *[rust_float(v) for v in f[2:7]]))

    L = [HEADER.format(
        title=f"Gasteiger–Marsili PEOE parameters — `{path.name}`.",
        source=f"{ANTECHAMBER_DIR}/{path.name}",
    )]
    w = L.append
    w("use crate::ff::params::GasteigerRow;")
    w("")
    w(f"/// The {len(rows)} Gasteiger PEOE parameter rows of `{path.name}`.")
    w(RUSTFMT_SKIP)
    w("pub const GASTEIGER_PARAMS: &[GasteigerRow] = &[")
    for atom_type, a, b, c, chi_plus, seed in rows:
        w(f"    GasteigerRow {{ atom_type: {rust_str(atom_type)}, a: {a}, b: {b}, c: {c}, "
          f"chi_plus: {chi_plus}, seed_charge: {seed} }},")
    w("];")
    w("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# PARMCHK.DAT — the conjugated pairing (the `alternate` column)
# ---------------------------------------------------------------------------

#: The header's own enumeration of the pairing, e.g.
#:     #   equivalent_flag:  1 for cc/ce/cg/nc/ne/pc/pe
#:     #                     2 for cd/cf/ch/nd/nf/pd/pf
#: The trailing `0 for others` line cannot match: it names no `a/b/c` list.
EQUIVALENT_FLAG_RE = re.compile(
    r"^#\s*(?:equivalent_flag:\s*)?([12])\s+for\s+(\w+(?:/\w+)+)\s*$"
)


@dataclass(frozen=True)
class Equivalents:
    """The conjugated pairing `PARMCHK.DAT` declares, read from BOTH of its halves."""

    #: phase-1 atom type -> the phase-2 type antechamber renames it to.
    pairs: dict[str, str]
    #: the one flagged pair the header does not name (`cp` -> `cq`): the biphenyl
    #: bridge carbons, which antechamber colours in a pass of their own.
    bridge: dict[str, str]
    #: atom type -> its `equivalent_flag` column, verbatim (sign included).
    flags: dict[str, int]


def parse_parmchk(path: Path) -> Equivalents:
    """Read the conjugated pairing out of `PARMCHK.DAT`.

    Two halves of the same fact, and BOTH are needed:

    * the **header** enumerates the pairing by name (`1 for cc/ce/cg/nc/ne/pc/pe`
      / `2 for cd/cf/ch/nd/nf/pd/pf`), zipped position-wise into `pairs`;
    * the per-type **`equivalent_flag` column** of each `PARM` row says whether
      that particular type takes part.

    They are cross-checked against each other, so an upstream edit to either one
    is a generator error rather than a silently wrong `alternate` column.
    """
    pairs: dict[str, str] = {}
    flags: dict[str, int] = {}
    phases: dict[int, list[str]] = {}

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        header = EQUIVALENT_FLAG_RE.match(raw.rstrip())
        if header:
            phase, names = int(header[1]), header[2].split("/")
            if phase in phases:
                raise GrammarError(f"{path.name}:{lineno}: phase {phase} declared twice")
            phases[phase] = names
            continue
        if not raw.startswith("PARM"):
            continue
        f = raw.split()
        if len(f) < 7:
            raise GrammarError(f"{path.name}:{lineno}: short PARM row: {raw}")
        flags[f[1]] = int(f[5])

    if sorted(phases) != [1, 2]:
        raise GrammarError(
            f"{path.name}: expected the header to declare both equivalent_flag phases, "
            f"found {sorted(phases)}"
        )
    if len(phases[1]) != len(phases[2]):
        raise GrammarError(
            f"{path.name}: the two equivalent_flag phases have different lengths "
            f"({phases[1]} vs {phases[2]}); the pairing is positional and does not zip"
        )

    for phase, names in phases.items():
        for name in names:
            if name not in flags:
                raise GrammarError(f"{path.name}: header names `{name}`, which has no PARM row")
            # The sign of the column distinguishes ring (-) from chain (+) types;
            # the phase is its magnitude.
            if abs(flags[name]) != phase:
                raise GrammarError(
                    f"{path.name}: header puts `{name}` in phase {phase} but its "
                    f"equivalent_flag column says {flags[name]}"
                )
    pairs = dict(zip(phases[1], phases[2], strict=True))

    # The flagged types the header leaves out are the bridge pair (`cp` -1 /
    # `cq` -2). `atomtype.c` colours them in `cpadjust`, separately from the
    # header's pairs (`atadjust`). The pairing is positional only when there is
    # one of each, so anything else is a generator error.
    named = set(phases[1]) | set(phases[2])
    rest = {
        phase: sorted(t for t, f in flags.items() if abs(f) == phase and t not in named)
        for phase in (1, 2)
    }
    if len(rest[1]) != len(rest[2]) or len(rest[1]) > 1:
        raise GrammarError(
            f"{path.name}: expected at most one equivalent_flag pair outside the header, "
            f"found {rest[1]} / {rest[2]}"
        )
    bridge = dict(zip(rest[1], rest[2], strict=True))
    return Equivalents(pairs=pairs, flags=flags, bridge=bridge)


def alternate_of(atom_type: str, eq: Equivalents | None) -> str | None:
    """The phase-2 name `atom_type` is renamed to on the other colour of a
    conjugated system (the header's pairs, `atadjust`), or `None`.

    `eq` is `None` for a `.DEF` outside the GAFF namespace (see `ATOMTYPE_FILES`),
    where PARMCHK.DAT's column says nothing about the types the file declares.
    Within that namespace, three independent gates — three different things that
    must NOT be paired:

    * the **flag column**, looked up case-sensitively, is what a type must carry to
      take part at all. `ATOMTYPE_AMBER.DEF` has real `CC` and `CD` rows — parm94's
      histidine carbons — whose `equivalent_flag` is `0`. Keying on the *spelling*
      of a type instead of on its column would pair them, 2-colour the AMBER table,
      and break a column that reproduces antechamber 37/37 today.
    * the **header's enumeration** is what supplies the partner's name. A nonzero
      flag alone is not a pairing: upstream flags `cp` / `cq` (-1 / -2) but does not
      list them in the header — they are coloured in a pass of their own
      (`bridge_of`), and never together with `cc` / `ce` / ….
    * the **namespace** decides whether the column applies to this file at all.
    """
    if eq is None:
        return None
    if abs(eq.flags.get(atom_type, 0)) != 1:
        return None
    return eq.pairs.get(atom_type)


def bridge_of(atom_type: str, eq: Equivalents | None) -> str | None:
    """The name a biphenyl bridge carbon (`cp`) takes on the other colour of
    its bridge system (`cq`, `atomtype.c`'s `cpadjust`), or `None`."""
    if eq is None:
        return None
    return eq.bridge.get(atom_type)


def emit_alternate(atom_type: str, eq: Equivalents | None) -> str:
    """The `alternate` column of an `AtdRule` row."""
    if (name := alternate_of(atom_type, eq)) is not None:
        return f"Some(Alternate {{ atom_type: {rust_str(name)}, pass: AlternatePass::Conjugated }})"
    if (name := bridge_of(atom_type, eq)) is not None:
        return f"Some(Alternate {{ atom_type: {rust_str(name)}, pass: AlternatePass::Bridge }})"
    return "None"


# ---------------------------------------------------------------------------
# PARMCHK.DAT — the substitution table itself (parmchk.rs)
# ---------------------------------------------------------------------------
#
# The file's own header names the PARM columns:
#
#     #   atomtype    improper_flag group_id  mass  equivalent_flag  atomic_num
#
# and an `EQUA` / `CORR` row carries the atom type it maps to plus NINE penalty
# columns. The file's trailing note pins their order — "GENERAL_SIMILARITY is
# listed in the 11th colume of CORR lines" — and the 11th token of a CORR row is
# the last of the nine, so the columns are the DEFAULT_* block's own order:
#
#     bl  blf  ba  baf  ba_ctr  baf_ctr  tor  tor_ctr  general_similarity
#
# A `-1` means "not tabulated"; the consumer substitutes the matching DEFAULT_*.

#: The nine penalty columns of an EQUA / CORR row, in file order.
CORR_COLUMNS = [
    "bond_length",
    "bond_force",
    "angle_centre",
    "angle_centre_force",
    "angle",
    "angle_force",
    "torsion_centre",
    "torsion",
    "similarity",
]

#: `WEIGHT_*` / `DEFAULT_*` scalars, mapped to their Rust field names. Anything
#: else in those blocks is a generator error rather than a silently dropped knob.
PARMCHK_SCALARS = {
    "WEIGHT_BL": "weight_bond_length",
    "WEIGHT_BLF": "weight_bond_force",
    "WEIGHT_BA": "weight_angle",
    "WEIGHT_BAF": "weight_angle_force",
    "WEIGHT_X": "weight_wildcard",
    "WEIGHT_X3": "weight_wildcard_centre",
    "WEIGHT_BA_CTR": "weight_angle_centre",
    "WEIGHT_TOR_CTR": "weight_torsion_centre",
    "WEIGHT_IMPROPER": "weight_improper",
    "WEIGHT_GROUP": "weight_group",
    "WEIGHT_EQUTYPE": "weight_equivalent",
    "DEFAULT_BL": "default_bond_length",
    "DEFAULT_BLF": "default_bond_force",
    "DEFAULT_BA": "default_angle",
    "DEFAULT_BAF": "default_angle_force",
    "DEFAULT_BA_CTR": "default_angle_centre",
    "DEFAULT_BAF_CTR": "default_angle_centre_force",
    "DEFAULT_TOR": "default_torsion",
    "DEFAULT_TOR_CTR": "default_torsion_centre",
    "DEFAULT_FRACT1": "default_fraction_1",
    "DEFAULT_FRACT2": "default_fraction_2",
    "THRESHOLD_BA": "threshold_angle",
}


@dataclass
class ParmchkType:
    """One `PARM` block: the atom type's flags plus its EQUA / CORR rows."""

    name: str
    improper: bool
    group: int
    mass: str
    equivalent_flag: int
    atomic_number: int
    #: `(to, [9 penalties])` — an equivalent type carries penalty columns too
    #: (`c5` does), but parmchk2 charges nothing for substituting one.
    equa: list[tuple[str, list[str] | None]] = field(default_factory=list)
    corr: list[tuple[str, list[str]]] = field(default_factory=list)


def parse_parmchk_table(path: Path) -> tuple[list[ParmchkType], dict[str, str]]:
    """`PARMCHK.DAT` -> its PARM blocks + the WEIGHT_* / DEFAULT_* scalars."""
    types: list[ParmchkType] = []
    scalars: dict[str, str] = {}
    current: ParmchkType | None = None

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        f = raw.split()
        if not f:
            continue
        head = f[0]
        if head == "PARM":
            if len(f) < 7:
                raise GrammarError(f"{path.name}:{lineno}: short PARM row: {raw}")
            current = ParmchkType(
                name=f[1],
                improper=bool(int(f[2])),
                group=int(f[3]),
                mass=rust_float(f[4]),
                equivalent_flag=int(f[5]),
                atomic_number=int(f[6]),
            )
            types.append(current)
        elif head in ("EQUA", "EUQA"):  # the file misspells one of them
            if current is None:
                raise GrammarError(f"{path.name}:{lineno}: EQUA before any PARM row")
            if len(f) >= 11:
                current.equa.append((f[1], [rust_float(v) for v in f[2:11]]))
            else:
                for name in f[1:]:
                    current.equa.append((name, None))
        elif head == "CORR":
            if current is None:
                raise GrammarError(f"{path.name}:{lineno}: CORR before any PARM row")
            # "For the CORR lines, if no penalty parameter presents, the default
            # value will be used" (the file's own header). A bare `CORR c3` is
            # therefore nine untabulated columns, not a malformed row.
            if len(f) == 2:
                current.corr.append((f[1], ["-1.0"] * 9))
            elif len(f) >= 11:
                current.corr.append((f[1], [rust_float(v) for v in f[2:11]]))
            else:
                raise GrammarError(
                    f"{path.name}:{lineno}: a CORR row carries either no penalty column "
                    f"or all nine (the 11th is GENERAL_SIMILARITY, per the file's own "
                    f"note); this one has {len(f) - 2}: {raw}"
                )
        elif head in PARMCHK_SCALARS:
            scalars[PARMCHK_SCALARS[head]] = rust_float(f[1])

    missing = sorted(set(PARMCHK_SCALARS.values()) - set(scalars))
    if missing:
        raise GrammarError(f"{path.name}: no value for {missing}")
    return types, scalars


def emit_parmchk(path: Path, eq: Equivalents) -> str:
    types, scalars = parse_parmchk_table(path)
    impropers = sum(1 for t in types if t.improper)

    L = [HEADER.format(
        title=f"parmchk2's atom-type substitution table — `{path.name}`.",
        source=f"{ANTECHAMBER_DIR}/{path.name}",
    )]
    w = L.append
    w("use crate::ff::params::{ParmchkCorr, ParmchkTable, ParmchkType, ParmchkWeights};")
    w("")
    w("/// The penalty weights and per-arity defaults (`WEIGHT_*` / `DEFAULT_*`).")
    w("pub const PARMCHK_WEIGHTS: ParmchkWeights = ParmchkWeights {")
    for field_name in PARMCHK_SCALARS.values():
        w(f"    {field_name}: {scalars[field_name]},")
    w("};")
    w("")
    w(f"/// The {len(types)} `PARM` blocks of `{path.name}`, in file order.")
    w("///")
    w(f"/// {impropers} of them carry `improper: true` — the column that decides whether a")
    w("/// 3-coordinate atom of that type is an improper CENTRE at all (`c3` is not; `ca`,")
    w("/// `c`, `n` and `na` are).")
    w(RUSTFMT_SKIP)
    w("pub const PARMCHK_TYPES: &[ParmchkType] = &[")
    for t in types:
        equa = ", ".join(rust_str(name) for name, _ in t.equa)
        corr = ", ".join(
            f"ParmchkCorr {{ to: {rust_str(name)}, penalties: [{', '.join(cols)}] }}"
            for name, cols in t.corr
        )
        alternate = alternate_of(t.name, eq)
        w(f"    ParmchkType {{ name: {rust_str(t.name)}, improper: {str(t.improper).lower()}, "
          f"group: {t.group}, mass: {t.mass}, "
          f"equivalent_flag: {t.equivalent_flag}, atomic_number: {t.atomic_number}, "
          f"alternate: {opt(rust_str(alternate)) if alternate else 'None'}, "
          f"equivalent: &[{equa}], corresponding: &[{corr}] }},")
    w("];")
    w("")
    w(f"/// `{path.name}` as one typed table.")
    w("pub const PARMCHK: ParmchkTable = ParmchkTable {")
    w(f"    name: {rust_str(path.name)},")
    w("    types: PARMCHK_TYPES,")
    w("    weights: PARMCHK_WEIGHTS,")
    w("};")
    w("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# PARM_BLBA_GAFF*.DAT — the empirical bond / angle constants (gaff_empirical.rs)
# ---------------------------------------------------------------------------
#
#   PARM  PC  <m>                                  the Badger exponent
#   PARM  BL  <e1> <z1> <e2> <z2> <r_ref> <ln_k>   per element PAIR
#   PARM  BA  <e>  <z>  <c> <z_factor>             per element

def parse_blba(path: Path) -> tuple[str, list[tuple[int, int, str, str]], list[tuple[int, str, str]]]:
    power: str | None = None
    bonds: list[tuple[int, int, str, str]] = []
    angles: list[tuple[int, str, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        f = raw.split()
        if not f or f[0] != "PARM":
            continue
        kind = f[1]
        if kind == "PC":
            power = rust_float(f[2])
        elif kind == "BL":
            if len(f) < 8:
                raise GrammarError(f"{path.name}:{lineno}: short BL row: {raw}")
            bonds.append((int(f[3]), int(f[5]), rust_float(f[6]), rust_float(f[7])))
        elif kind == "BA":
            if len(f) < 6:
                raise GrammarError(f"{path.name}:{lineno}: short BA row: {raw}")
            angles.append((int(f[3]), rust_float(f[4]), rust_float(f[5])))
        else:
            raise GrammarError(f"{path.name}:{lineno}: unknown row kind `{kind}`")
    if power is None:
        raise GrammarError(f"{path.name}: no `PARM PC` row — the Badger exponent is missing")
    return power, bonds, angles


def emit_empirical_from(src: Path) -> str:
    L = [HEADER.format(
        title="GAFF empirical bond / angle constants (Wang2004 Eqs. 3 and 5).",
        source=f"{ANTECHAMBER_DIR}/PARM_BLBA_GAFF*.DAT",
    )]
    w = L.append
    w("use crate::ff::params::{EmpiricalAngleRow, EmpiricalBondRow, EmpiricalTable};")
    w("")
    for filename, const in BLBA_FILES:
        path = src / filename
        power, bonds, angles = parse_blba(path)
        lower = const.lower()
        w(f"/// The {len(bonds)} `BL` rows of `{path.name}`: reference length + `ln(Kij)`.")
        w(RUSTFMT_SKIP)
        w(f"const {const}_BONDS: &[EmpiricalBondRow] = &[")
        for z1, z2, r_ref, ln_k in bonds:
            w(f"    EmpiricalBondRow {{ z1: {z1}, z2: {z2}, r_ref: {r_ref}, ln_k: {ln_k} }},")
        w("];")
        w("")
        w(f"/// The {len(angles)} `BA` rows of `{path.name}`: the angle C / Z factors.")
        w(RUSTFMT_SKIP)
        w(f"const {const}_ANGLES: &[EmpiricalAngleRow] = &[")
        for z, c, zf in angles:
            w(f"    EmpiricalAngleRow {{ z: {z}, c: {c}, z_factor: {zf} }},")
        w("];")
        w("")
        w(f"/// `{path.name}` as one typed table (Badger exponent m = {power}).")
        w(f"pub const {const}: EmpiricalTable = EmpiricalTable {{")
        w(f"    name: {rust_str(path.name)},")
        w(f"    bond_power: {power},")
        w(f"    bonds: {const}_BONDS,")
        w(f"    angles: {const}_ANGLES,")
        w("};")
        w("")
        del lower
    return "\n".join(L)


# ---------------------------------------------------------------------------
# ATOMTYPE_*.DEF — the ATD / WILDATOM grammar (one parser, seven outputs)
# ---------------------------------------------------------------------------

@dataclass
class Pattern:
    """One node of an environment expression, e.g. `C3[AR1](O1,O1)`."""

    name: str
    degree: int | None
    prop: str | None
    label: str | None
    children: list[Pattern] = field(default_factory=list)


@dataclass
class Rule:
    """One `ATD` row."""

    atom_type: str
    residue: str
    atomic_number: int | None
    degree: int | None
    hydrogen_count: int | None
    ewd_count: int | None
    prop: str | None
    env: list[Pattern] | None
    env_bonds: list[tuple[str, str, str]] | None


def find_matching(s: str, start: int, op: str, cl: str) -> int:
    depth = 0
    for i in range(start, len(s)):
        if s[i] == op:
            depth += 1
        elif s[i] == cl:
            depth -= 1
            if depth == 0:
                return i
    raise GrammarError(f"unmatched `{op}` in `{s}`")


def split_top_level(s: str, sep: str = ",") -> list[str]:
    out, start, depth = [], 0, 0
    for i, ch in enumerate(s):
        if ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth -= 1
        elif ch == sep and depth == 0:
            out.append(s[start:i].strip())
            start = i + 1
    out.append(s[start:].strip())
    return out


def parse_atom_pattern(s: str, pos: int) -> tuple[Pattern, int]:
    """`Name[Degree][Property]<Label>(Children...)` — every part after Name optional."""
    name_start = pos
    while pos < len(s) and s[pos].isalpha():
        pos += 1
    if pos == name_start:
        raise GrammarError(f"expected an atom name in `{s}`")
    name = s[name_start:pos]

    degree_start = pos
    while pos < len(s) and s[pos].isdigit():
        pos += 1
    degree = int(s[degree_start:pos]) if pos > degree_start else None

    prop = label = None
    children: list[Pattern] = []
    while pos < len(s):
        ch = s[pos]
        if ch == "[":
            end = find_matching(s, pos, "[", "]")
            prop = s[pos:end + 1]
            pos = end + 1
        elif ch == "<":
            end = find_matching(s, pos, "<", ">")
            label = s[pos + 1:end]
            pos = end + 1
        elif ch == "(":
            end = find_matching(s, pos, "(", ")")
            # Consecutive groups CONCATENATE: `S4(O1)(O1)(O2)` means an S of
            # degree 4 with three neighbours, exactly like `S4(O1,O1,O2)`.
            # (ATOMTYPE_ABCG2.DEF:62, the sulfonate/sulfate-ester oxygen `3S`,
            # is the only row upstream that spells it this way.)
            children.extend(parse_pattern_list(s[pos + 1:end]))
            pos = end + 1
        else:
            break
    return Pattern(name, degree, prop, label, children), pos


def parse_pattern_list(s: str) -> list[Pattern]:
    out = []
    for part in split_top_level(s):
        if not part:
            continue
        pattern, pos = parse_atom_pattern(part, 0)
        if pos != len(part):
            raise GrammarError(f"trailing junk in environment fragment `{part}`")
        out.append(pattern)
    return out


def parse_environment(s: str) -> list[Pattern]:
    if not (s.startswith("(") and s.endswith(")")):
        raise GrammarError(f"environment must be parenthesised: `{s}`")
    return parse_pattern_list(s[1:-1])


def parse_env_bonds(s: str) -> list[tuple[str, str, str]]:
    """`a1:a2:any[,...]` — a bond constraint between two <labelled> environment atoms."""
    out = []
    for raw in s.split(","):
        parts = raw.split(":")
        if len(parts) != 3:
            raise GrammarError(f"environment bond must be `a:b:TYPE`: `{raw}`")
        if parts[2] not in ENV_BOND_TYPES:
            raise GrammarError(f"unknown environment bond type `{parts[2]}`")
        out.append((parts[0], parts[1], parts[2]))
    return out


def parse_prop_expr(s: str) -> list[list[tuple[int | None, str, bool | None]]]:
    """`[A.B,C]` -> AND of ORs: every comma-group must hold; a group holds if any
    dot-unit holds. A unit is `[count]PROP[' | '']`, where `'` additionally requires
    the bond back to the previous atom to be of that type and `''` requires it not to be.
    """
    if not (s.startswith("[") and s.endswith("]")):
        raise GrammarError(f"atom property must be bracketed: `{s}`")
    constraints = []
    for group in s[1:-1].split(","):
        units = []
        for unit in group.split("."):
            digits = 0
            while digits < len(unit) and unit[digits].isdigit():
                digits += 1
            count = int(unit[:digits]) if digits else None
            prop = unit[digits:]
            relation: bool | None = None
            if prop.endswith("''"):
                prop, relation = prop[:-2], False
            elif prop.endswith("'"):
                prop, relation = prop[:-1], True
            if prop not in ATOM_PROPS:
                raise GrammarError(f"unknown atom property `{prop}` in `{s}`")
            units.append((count, prop, relation))
        constraints.append(units)
    return constraints


def parse_wildatom_spec(token: str) -> tuple[int, int | None]:
    """`C3` -> (6, Some(3)); `O` -> (8, None)."""
    split = next((i for i, c in enumerate(token) if c.isdigit()), len(token))
    symbol, degree = token[:split], token[split:]
    z = atomic_number(symbol)
    if z is None:
        raise GrammarError(f"unknown WILDATOM element symbol `{symbol}`")
    return z, int(degree) if degree else None


def parse_atomtype_def(path: Path) -> tuple[dict[str, list[tuple[int, int | None]]], list[Rule]]:
    wildatoms: dict[str, list[tuple[int, int | None]]] = {}
    rules: list[Rule] = []

    def num(f: list[str], i: int) -> int | None:
        v = f[i] if i < len(f) else None
        return None if v in (None, "*", "&") else int(v)

    def text(f: list[str], i: int) -> str | None:
        v = f[i] if i < len(f) else None
        return None if v in (None, "*", "&") else v

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if is_comment(line):
            continue
        f = line.split()
        if f[0] == "WILDATOM":
            if len(f) < 3:
                raise GrammarError(f"{path.name}:{lineno}: short WILDATOM row")
            wildatoms[f[1]] = [parse_wildatom_spec(t) for t in f[2:]]
            continue
        if f[0] != "ATD":
            continue  # prose in the trailing "rules of defining an atom" section
        if len(f) < 3:
            raise GrammarError(f"{path.name}:{lineno}: short ATD row: {raw}")
        atom_type = f[1]
        # Every table ends with a constraint-free fall-through row — `ATD DU &`
        # (BCC / ABCG2 / GAS / GFF / GFF2 / AMBER) or `ATD ANY &` (SYBYL). It is
        # a NAMED RULE OF THE TABLE, not an invented fallback, and antechamber
        # reaches it: `-at amber` types nitromethane's two nitro oxygens and
        # DMSO's sulfoxide sulfur `DU`. Dropping it made molrs error where
        # antechamber answers, so it is emitted like any other row — with every
        # column unconstrained, which is what `&` in the residue column means.
        residue = text(f, 2) or "*"
        env = text(f, 8)
        env_bonds = text(f, 9)
        prop = text(f, 7)
        rules.append(Rule(
            atom_type=atom_type,
            residue=residue,
            atomic_number=num(f, 3),
            degree=num(f, 4),
            hydrogen_count=num(f, 5),
            ewd_count=num(f, 6),
            prop=prop,
            env=parse_environment(env) if env else None,
            env_bonds=parse_env_bonds(env_bonds) if env_bonds else None,
        ))
    return wildatoms, rules


# --- ATD -> Rust ------------------------------------------------------------

def wild_const(name: str) -> str:
    return f"WILD_{name.upper()}"


def emit_prop_expr(s: str) -> str:
    constraints = parse_prop_expr(s)
    out = []
    for units in constraints:
        rendered = []
        for count, prop, relation in units:
            rel = None
            if relation is True:
                rel = "PropRelation::BondedToPrev"
            elif relation is False:
                rel = "PropRelation::NotBondedToPrev"
            rendered.append(
                f"PropUnit {{ count: {opt(str(count) if count is not None else None)}, "
                f"prop: AtomProp::{ATOM_PROPS[prop]}, relation: {opt(rel)} }}"
            )
        out.append(f"PropConstraint {{ units: &[{', '.join(rendered)}] }}")
    return f"PropExpr {{ constraints: &[{', '.join(out)}] }}"


def emit_pattern(p: Pattern, wildatoms: dict) -> str:
    """Resolve the pattern's name here, so molrs never looks a name up at runtime."""
    if p.name == "EW":
        atom = "PatternAtom::ElectronWithdrawing"
    elif p.name in wildatoms:
        atom = f"PatternAtom::Wild({wild_const(p.name)})"
    elif atomic_number(p.name) is not None:
        atom = f"PatternAtom::Element({atomic_number(p.name)})"
    else:
        raise GrammarError(
            f"pattern atom `{p.name}` is neither EW, a WILDATOM of this file, nor an element"
        )
    children = ", ".join(emit_pattern(c, wildatoms) for c in p.children)
    return (
        f"AtomPattern {{ atom: {atom}, "
        f"degree: {opt(str(p.degree) if p.degree is not None else None)}, "
        f"property: {opt(emit_prop_expr(p.prop) if p.prop else None)}, "
        f"label: {opt(rust_str(p.label) if p.label else None)}, "
        f"children: &[{children}] }}"
    )


PARAM_TYPES = [
    "Alternate", "AlternatePass", "AtdRule", "AtdTable", "AtomPattern", "AtomProp", "EnvBond", "EnvBondType",
    "PatternAtom", "PropConstraint", "PropExpr", "PropRelation", "PropUnit",
    "WildAtom", "WildAtomSpec",
]


def emit_atomtype(path: Path, const: str, eq: Equivalents | None) -> str:
    wildatoms, rules = parse_atomtype_def(path)

    L: list[str] = []
    w = L.append

    for name, specs in wildatoms.items():
        pretty = " ".join(
            f"{element_symbol(z)}{d if d is not None else ''}" for z, d in specs
        )
        w(f"/// `WILDATOM {name} {pretty}`")
        w(RUSTFMT_SKIP)
        w(f"const {wild_const(name)}: &[WildAtomSpec] = &[")
        for z, degree in specs:
            w(f"    WildAtomSpec {{ z: {z}, degree: {opt(str(degree) if degree is not None else None)} }},")
        w("];")
        w("")

    w(f"/// The {len(wildatoms)} `WILDATOM` aliases declared by `{path.name}`.")
    w(RUSTFMT_SKIP)
    w("pub const WILDATOMS: &[WildAtom] = &[")
    for name in wildatoms:
        w(f"    WildAtom {{ name: {rust_str(name)}, specs: {wild_const(name)} }},")
    w("];")
    w("")

    paired = sum(1 for r in rules if emit_alternate(r.atom_type, eq) != "None")
    w(f"/// The {len(rules)} `ATD` rules of `{path.name}`, in file order.")
    w("///")
    w("/// Order is significant: the FIRST rule that matches wins — which is why the")
    w("/// table's own last row is the constraint-free fall-through (`DU`, or `ANY` in")
    w("/// `ATOMTYPE_SYBYL.DEF`). It matches anything nothing above it matched, and")
    w("/// antechamber does reach it: `-at amber` types nitromethane's nitro oxygens")
    w("/// `DU`. It is a rule of the table, not a fallback the engine invents.")
    w("///")
    if paired:
        w(f"/// {paired} of these rules carry an `alternate`: the phase-2 name")
        w("/// `PARMCHK.DAT` pairs their atom type with. The rule emits the phase-1 name;")
        w("/// the typifier's 2-colouring passes rename one colour of each conjugated")
        w("/// (or biphenyl-bridge) system to the alternate, which is the only way a type")
        w("/// no ATD row declares (`cd`, `cq`) is ever assigned.")
    else:
        w("/// No rule here carries an `alternate`: `PARMCHK.DAT`'s `equivalent_flag`")
        w("/// column describes the GAFF atom-type namespace, and this file is not written")
        w("/// in it. Nothing in this table is ever renamed by the 2-colouring pass.")
    w("pub const RULES: &[AtdRule] = &[")
    for r in rules:
        env = (
            "&[" + ", ".join(emit_pattern(p, wildatoms) for p in r.env) + "]"
            if r.env is not None else None
        )
        env_bonds = None
        if r.env_bonds is not None:
            bonds = ", ".join(
                f"EnvBond {{ a: {rust_str(a)}, b: {rust_str(b)}, "
                f"bond: EnvBondType::{ENV_BOND_TYPES[t]} }}"
                for a, b, t in r.env_bonds
            )
            env_bonds = f"&[{bonds}]"
        w("    AtdRule {")
        w(f"        atom_type: {rust_str(r.atom_type)},")
        w(f"        alternate: {emit_alternate(r.atom_type, eq)},")
        w(f"        residue: {rust_str(r.residue)},")
        w(f"        atomic_number: {opt(str(r.atomic_number) if r.atomic_number is not None else None)},")
        w(f"        degree: {opt(str(r.degree) if r.degree is not None else None)},")
        w(f"        hydrogen_count: {opt(str(r.hydrogen_count) if r.hydrogen_count is not None else None)},")
        w(f"        ewd_count: {opt(str(r.ewd_count) if r.ewd_count is not None else None)},")
        w(f"        atom_property: {opt(emit_prop_expr(r.prop) if r.prop else None)},")
        w(f"        environment: {opt(env)},")
        w(f"        environment_bonds: {opt(env_bonds)},")
        w("    },")
    w("];")
    w("")
    w(f"/// `{path.name}` as one typed table.")
    w(f"pub const {const}: AtdTable = AtdTable {{")
    w(f"    name: {rust_str(path.name)},")
    w("    wildatoms: WILDATOMS,")
    w("    rules: RULES,")
    w("};")
    w("")

    # Import exactly the row types this file mentions — `PropRelation` only shows
    # up in the tables that actually use a `'` / `''` unit, and an unused import
    # is a `-D warnings` build failure.
    body = "\n".join(L)
    used = [t for t in PARAM_TYPES if re.search(rf"\b{t}\b", body)]
    head = HEADER.format(
        title=f"Antechamber atom-type definition rules — `{path.name}`.",
        source=f"{ANTECHAMBER_DIR}/{path.name}",
    )
    return f"{head}\nuse crate::ff::params::{{{', '.join(used)}}};\n\n{body}"


# ---------------------------------------------------------------------------
# gaff.dat / gaff2.dat — AMBER's `parm` format (one parser, two outputs)
# ---------------------------------------------------------------------------
#
# The `parm` format is POSITIONAL, not labelled: no section carries a heading,
# and each is terminated by a blank line. Reading it therefore means walking the
# sections in the order the format fixes them (LEaP's own reader does the same):
#
#     title
#     MASS        rows      -> emitted
#     hydrophilic list      (one line, no blank terminator)   -> not emitted
#     BOND        rows      -> emitted
#     ANGLE       rows      -> emitted
#     DIHE        rows      -> emitted
#     IMPROPER    rows      -> emitted
#     10-12 H-bond rows     -> not emitted
#     equivalence rows      -> not emitted (and guarded: see `parse_parm`)
#     `MOD4  RE` label      (declares the units of the NONBON columns)
#     NONBON      rows      -> emitted
#     END
#
# Three sections carry no force-field parameter and are not emitted. That is not
# a silent drop — each is accounted for:
#
# * the **hydrophilic list** is a solvation hint for LEaP's `solvateBox`, not a
#   parameter, and antechamber ignores it too;
# * the **10-12 H-bond** section is the pre-ff94 hydrogen-bond potential, dead
#   since 1994; both files carry exactly one row there, the `hw`/`ow` fast-water
#   flag, and `parse_parm` errors if that ever changes rather than guessing at
#   the column meanings of a format nothing documents any more;
# * the **equivalence** section would make one atom type borrow another's NONBON
#   row. It is empty in both files, and `parse_parm` errors if it stops being —
#   a non-empty one silently changes what a NONBON lookup means.
#
# The atom-type columns are FIXED-WIDTH (`A2,1X,A2,...`), which is load-bearing:
# a wildcard row reads `X -c -c -X `, and splitting that on whitespace yields
# `['X', '-c', '-c', '-X']` — four wrong types. They are cut by column instead.

#: The `X` atom type of a DIHE / IMPROPER row: matches any type. Emitted as
#: `None`, so a consumer cannot read a wildcard slot as a concrete type by
#: accident. gaff.dat and gaff2.dat carry 615 wildcard rows each and spec 11's
#: parmchk2-style fallback matching is built on them.
PARM_WILDCARD = "X"


@dataclass(frozen=True)
class ParmTables:
    """The six parameter sections of one `parm` file, in file order."""

    #: `(atom_type, mass, polarizability)`
    masses: list[tuple[str, str, str]]
    #: `(i, j, force_constant, length)`
    bonds: list[tuple[str, str, str, str]]
    #: `(i, j, k, force_constant, angle_deg)`
    angles: list[tuple[str, str, str, str, str]]
    #: `(i, j, k, l, divisor, barrier, phase_deg, periodicity, more_terms)`
    dihedrals: list[tuple[str, str, str, str, int, str, str, int, bool]]
    #: `(i, j, k, l, barrier, phase_deg, periodicity)` — `k` is the CENTRAL atom
    impropers: list[tuple[str, str, str, str, str, str, int]]
    #: `(atom_type, r_min_half, epsilon)`
    nonbonded: list[tuple[str, str, str]]


class ParmCursor:
    """A line cursor over a `parm` file, because the format is positional."""

    def __init__(self, path: Path) -> None:
        self.name = path.name
        self.lines = path.read_text(encoding="utf-8").splitlines()
        self.pos = 0

    def line(self) -> tuple[int, str]:
        """The next line, consumed. Returns `(lineno, text)`."""
        if self.pos >= len(self.lines):
            raise GrammarError(f"{self.name}: file ended early")
        self.pos += 1
        return self.pos, self.lines[self.pos - 1]

    def section(self) -> list[tuple[int, str]]:
        """Rows up to (and consuming) the blank line that terminates the section."""
        rows: list[tuple[int, str]] = []
        while self.pos < len(self.lines):
            self.pos += 1
            raw = self.lines[self.pos - 1]
            if not raw.strip():
                return rows
            rows.append((self.pos, raw))
        raise GrammarError(f"{self.name}: unterminated section at end of file")


def parm_atom_types(name: str, lineno: int, raw: str, arity: int) -> list[str]:
    """The fixed-width `A2,1X,A2,...` atom-type columns of a bonded row."""
    types = []
    for n in range(arity):
        if n and raw[3 * n - 1 : 3 * n] != "-":
            raise GrammarError(
                f"{name}:{lineno}: expected `-` between atom-type columns: {raw!r}"
            )
        token = raw[3 * n : 3 * n + 2].strip()
        if not token:
            raise GrammarError(f"{name}:{lineno}: empty atom-type column {n}: {raw!r}")
        types.append(token)
    return types


def parm_values(name: str, lineno: int, raw: str, arity: int, count: int) -> list[str]:
    """The first `count` numeric columns after the atom types.

    Taken POSITIONALLY, exactly as the FORTRAN format reads them: the trailing
    provenance columns of a gaff row (`SOURCE1`, a sample count, an RMS error)
    are prose and are not parameters.
    """
    fields = raw[3 * arity - 1 :].split()
    if len(fields) < count:
        raise GrammarError(f"{name}:{lineno}: expected {count} values: {raw!r}")
    out = []
    for value in fields[:count]:
        try:
            float(value)
        except ValueError as exc:
            raise GrammarError(f"{name}:{lineno}: non-numeric column {value!r}") from exc
        out.append(value)
    return out


def parm_integer(name: str, lineno: int, token: str, what: str) -> int:
    """A column the format declares integral (`IDIVF`, `PN`), verified as such."""
    value = float(token)
    if value != int(value):
        raise GrammarError(f"{name}:{lineno}: {what} must be an integer, got {token!r}")
    return int(value)


def no_wildcards(name: str, lineno: int, section: str, types: list[str]) -> None:
    """`X` may appear in DIHE and IMPROPER rows only.

    The other sections index by concrete atom type, and a consumer of this table
    relies on that: it is why `ParmBondRow` can hold a `&str` where a dihedral row
    has to hold an `Option<&str>`.
    """
    if PARM_WILDCARD in types:
        raise GrammarError(
            f"{name}:{lineno}: a wildcard `{PARM_WILDCARD}` appeared in the {section} "
            f"section, which upstream indexes by concrete atom type only"
        )


def no_reversed_duplicate(
    name: str, section: str, keys: list[tuple[str, ...]]
) -> None:
    """A bonded term matches its row in either orientation, so a table holding
    BOTH `a-b-c` and `c-b-a` would make the lookup order-dependent. Neither gaff
    table does; if one ever did, the resolution rule would have to be derived
    from LEaP rather than guessed at here."""
    seen = set(keys)
    for key in keys:
        reverse = tuple(reversed(key))
        if reverse != key and reverse in seen:
            raise GrammarError(
                f"{name}: the {section} section holds both `{'-'.join(key)}` and its "
                f"reverse `{'-'.join(reverse)}`; which one an unordered term matches is "
                f"then undefined"
            )


def parse_parm(path: Path) -> ParmTables:
    """Parse an AMBER `parm` file (`gaff.dat`, `gaff2.dat`) into its six sections."""
    cur = ParmCursor(path)
    name = path.name
    cur.line()  # title

    masses = []
    for lineno, raw in cur.section():
        atom_type = raw[:2].strip()
        if not atom_type:
            raise GrammarError(f"{name}:{lineno}: MASS row has no atom type: {raw!r}")
        no_wildcards(name, lineno, "MASS", [atom_type])
        mass, polarizability = parm_values(name, lineno, raw, 1, 2)
        row = (atom_type, rust_float(mass), rust_float(polarizability))
        seen = [m for m in masses if m[0] == atom_type]
        if seen:
            if seen[0] != row:
                raise GrammarError(f"{name}:{lineno}: MASS {atom_type} restated differently")
            continue
        masses.append(row)

    cur.line()  # hydrophilic-atom list: a solvation hint, not a parameter

    bonds, bond_keys = [], []
    for lineno, raw in cur.section():
        i, j = parm_atom_types(name, lineno, raw, 2)
        no_wildcards(name, lineno, "BOND", [i, j])
        force_constant, length = parm_values(name, lineno, raw, 2, 2)
        bonds.append((i, j, rust_float(force_constant), rust_float(length)))
        bond_keys.append((i, j))
    no_reversed_duplicate(name, "BOND", bond_keys)

    angles, angle_keys = [], []
    for lineno, raw in cur.section():
        i, j, k = parm_atom_types(name, lineno, raw, 3)
        no_wildcards(name, lineno, "ANGLE", [i, j, k])
        force_constant, angle = parm_values(name, lineno, raw, 3, 2)
        angles.append((i, j, k, rust_float(force_constant), rust_float(angle)))
        angle_keys.append((i, j, k))
    no_reversed_duplicate(name, "ANGLE", angle_keys)

    dihedrals, dihedral_keys = [], []
    for lineno, raw in cur.section():
        i, j, k, l = parm_atom_types(name, lineno, raw, 4)
        divisor, barrier, phase, pn = parm_values(name, lineno, raw, 4, 4)
        idivf = parm_integer(name, lineno, divisor, "IDIVF")
        if idivf < 1:
            raise GrammarError(f"{name}:{lineno}: IDIVF must be positive, got {idivf}")
        periodicity = parm_integer(name, lineno, pn, "PN")
        # A NEGATIVE periodicity is upstream's continuation flag: another cosine
        # term for the same quartet follows on the next line. The magnitude is
        # the periodicity; the sign is a structural fact about the file, so it is
        # emitted as its own `more_terms` field rather than as a signed number a
        # kernel could feed straight into `cos(n*phi)`.
        dihedrals.append((
            i, j, k, l, idivf,
            rust_float(barrier), rust_float(phase),
            abs(periodicity), periodicity < 0,
        ))
        dihedral_keys.append((i, j, k, l))
    no_reversed_duplicate(name, "DIHE", dihedral_keys)
    check_dihedral_terms(name, dihedrals)

    impropers = []
    for lineno, raw in cur.section():
        i, j, k, l = parm_atom_types(name, lineno, raw, 4)
        barrier, phase, pn = parm_values(name, lineno, raw, 4, 3)
        periodicity = parm_integer(name, lineno, pn, "PN")
        if periodicity < 1:
            raise GrammarError(
                f"{name}:{lineno}: an IMPROPER row carries no multi-term continuation "
                f"flag, so its PN must be positive; got {periodicity}"
            )
        impropers.append((i, j, k, l, rust_float(barrier), rust_float(phase), periodicity))

    hbond = cur.section()
    for lineno, raw in hbond:
        fields = raw.split()
        if fields[:2] != ["hw", "ow"]:
            raise GrammarError(
                f"{name}:{lineno}: the 10-12 H-bond section holds a row this generator "
                f"cannot read ({raw!r}). Upstream it holds only the `hw ow` fast-water "
                f"flag; the 10-12 potential itself has been dead since ff94 and its "
                f"column layout is not something to guess at"
            )

    equivalence = cur.section()
    if equivalence:
        raise GrammarError(
            f"{name}: the nonbonded equivalence section is no longer empty "
            f"({len(equivalence)} rows). Those rows make one atom type borrow another's "
            f"NONBON parameters, which changes what every NONBON lookup means — they "
            f"have to be emitted and honoured, not skipped"
        )

    _, label = cur.line()
    if label.split() != ["MOD4", "RE"]:
        raise GrammarError(
            f"{name}: expected the NONBON label `MOD4  RE`, got {label!r}. `RE` is what "
            f"declares the two columns to be R* and epsilon; under `AC` they would be "
            f"the A/C coefficients of the 12-6 form instead"
        )

    nonbonded = []
    for lineno, raw in cur.section():
        fields = raw.split()
        if len(fields) < 3:
            raise GrammarError(f"{name}:{lineno}: short NONBON row: {raw!r}")
        no_wildcards(name, lineno, "NONBON", [fields[0]])
        nonbonded.append((fields[0], rust_float(fields[1]), rust_float(fields[2])))

    return ParmTables(
        masses=masses,
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        impropers=impropers,
        nonbonded=nonbonded,
    )


def check_dihedral_terms(name: str, dihedrals: list) -> None:
    """The multi-term continuation flag must agree with the file's own grouping.

    A dihedral quartet with several cosine terms is written as consecutive rows,
    every one but the last carrying a negative PN. Two independent facts — the
    sign column and the row adjacency — say the same thing, so they are checked
    against each other: a consumer that groups terms by quartet key (as the
    ForceField population path does) must land on the same grouping as one that
    follows the sign.
    """
    for idx, row in enumerate(dihedrals):
        key, more_terms = row[:4], row[8]
        if not more_terms:
            continue
        if idx + 1 >= len(dihedrals):
            raise GrammarError(
                f"{name}: the last DIHE row flags a continuation term that never comes"
            )
        if dihedrals[idx + 1][:4] != key:
            raise GrammarError(
                f"{name}: the DIHE row `{'-'.join(key)}` flags a continuation term "
                f"(negative PN) but the next row is a different quartet "
                f"`{'-'.join(dihedrals[idx + 1][:4])}`"
            )


class TypeInterner:
    """Names the atom types of one `parm` table, and numbers them.

    The number is the type's row in the `MASS` section — the section that
    declares the atom types — so `ParmTable::name_of` is one array read. Every
    type a bonded row names must be declared there; one that is not is an
    upstream defect, not something to invent an index for.

    Types are emitted as a named `const` (`c3` -> `T_C3`), never as a bare
    number: the whole point of committing these tables is that a row can be read
    and grepped, and `i: T_C3, j: T_C3` reads as well as `i: "c3", j: "c3"` did
    while costing one byte instead of a 16-byte fat pointer.
    """

    #: An atom type is not always a Rust identifier: `gaff2.dat` declares one
    #: literally named `n+` (a protonated nitrogen). The mangling is explicit and
    #: an unmapped character is a generator error — an ad-hoc mangle could quietly
    #: fold two atom types onto one `const` and swap their parameters.
    MANGLE: ClassVar[dict[str, str]] = {"+": "_PLUS", "-": "_MINUS", "*": "_STAR"}

    def __init__(self, name: str, masses: list[tuple[str, str, str]]) -> None:
        self.name = name
        self.index = {atom_type: n for n, (atom_type, _, _) in enumerate(masses)}
        if len(self.index) != len(masses):
            raise GrammarError(f"{name}: the MASS section declares a type twice")
        if len(self.index) > 256:
            raise GrammarError(
                f"{name}: {len(self.index)} atom types no longer fit a `ParmType(u8)`"
            )
        self.consts = {t: self._mangle(t) for t in self.index}
        collisions = len(self.consts) - len(set(self.consts.values()))
        if collisions:
            raise GrammarError(
                f"{name}: {collisions} atom types mangle onto the same const name; two types "
                f"sharing one const would silently swap their parameters"
            )

    def _mangle(self, atom_type: str) -> str:
        out = []
        for ch in atom_type:
            if ch.isalnum():
                out.append(ch.upper())
            elif ch in self.MANGLE:
                out.append(self.MANGLE[ch])
            else:
                raise GrammarError(
                    f"{self.name}: the atom type `{atom_type}` carries `{ch}`, which is neither "
                    f"alphanumeric nor a character this generator knows how to mangle into a "
                    f"Rust identifier"
                )
        return "T_" + "".join(out)

    def const(self, atom_type: str) -> str:
        """The `const` naming `atom_type`, e.g. `T_C3` (and `n+` -> `T_N_PLUS`)."""
        if atom_type not in self.consts:
            raise GrammarError(
                f"{self.name}: a parameter row names the atom type `{atom_type}`, which the "
                f"MASS section never declares — so the table has no mass, no index and no "
                f"name for it"
            )
        return self.consts[atom_type]

    def slot(self, token: str) -> str:
        """A DIHE / IMPROPER atom-type slot: `X` -> `None`, else `Some(T_..)`."""
        return "None" if token == PARM_WILDCARD else f"Some({self.const(token)})"

    def declarations(self) -> list[str]:
        out = []
        for atom_type, n in self.index.items():
            out.append(f"/// `{atom_type}` — row {n} of the `MASS` section.")
            out.append(f"const {self.const(atom_type)}: ParmType = ParmType({n});")
        return out


def emit_parm(path: Path, prefix: str) -> str:
    t = parse_parm(path)
    types = TypeInterner(path.name, t.masses)
    source = f"{PARM_DIR}/{path.name}"
    wildcards = sum(
        1
        for row in (*t.dihedrals, *t.impropers)
        if PARM_WILDCARD in row[:4]
    )

    L = [HEADER.format(
        title=f"AMBER general force field parameters — `{path.name}`.",
        source=source,
    )]
    w = L.append
    # These tables are MEASURED numbers. gaff2's `c1-c1` bond is 1.4426 A long and
    # clippy reads that as a fat-fingered `LOG2_E`; three more rows land near PI.
    # The lint cannot tell data from a mistyped constant, and there is no constant
    # here to mistype.
    w("#![allow(clippy::approx_constant)]")
    w("")
    w("use crate::ff::params::{")
    w("    ParmAngleRow, ParmBondRow, ParmDihedralRow, ParmImproperRow, ParmMassRow,")
    w("    ParmNonbondedRow, ParmTable, ParmType,")
    w("};")
    w("")
    w(f"// The {len(t.masses)} atom types of `{path.name}`, interned to their MASS row.")
    for line in types.declarations():
        w(line)
    w("")

    w(f"/// The {len(t.masses)} `MASS` rows of `{path.name}` — the atom-type declarations.")
    w("///")
    w("/// This section's row order IS the [`ParmType`] numbering.")
    w(RUSTFMT_SKIP)
    w(f"pub const {prefix}_MASSES: &[ParmMassRow] = &[")
    for atom_type, mass, polarizability in t.masses:
        w(f"    ParmMassRow {{ atom_type: {rust_str(atom_type)}, mass: {mass}, "
          f"polarizability: {polarizability} }},")
    w("];")
    w("")

    w(f"/// The {len(t.bonds)} `BOND` rows of `{path.name}`.")
    w(RUSTFMT_SKIP)
    w(f"pub const {prefix}_BONDS: &[ParmBondRow] = &[")
    for i, j, force_constant, length in t.bonds:
        w(f"    ParmBondRow {{ i: {types.const(i)}, j: {types.const(j)}, "
          f"force_constant: {force_constant}, length: {length} }},")
    w("];")
    w("")

    w(f"/// The {len(t.angles)} `ANGLE` rows of `{path.name}`.")
    w(RUSTFMT_SKIP)
    w(f"pub const {prefix}_ANGLES: &[ParmAngleRow] = &[")
    for i, j, k, force_constant, angle in t.angles:
        w(f"    ParmAngleRow {{ i: {types.const(i)}, j: {types.const(j)}, k: {types.const(k)}, "
          f"force_constant: {force_constant}, angle_deg: {angle} }},")
    w("];")
    w("")

    wild_dihedrals = sum(1 for row in t.dihedrals if PARM_WILDCARD in row[:4])
    w(f"/// The {len(t.dihedrals)} `DIHE` rows of `{path.name}`, in file order.")
    w("///")
    w(f"/// {wild_dihedrals} of them carry a wildcard slot (`None`), and the order is")
    w("/// significant: consecutive rows with the same quartet are the cosine terms of one")
    w("/// torsion, all but the last flagged [`more_terms`](ParmDihedralRow::more_terms).")
    w(RUSTFMT_SKIP)
    w(f"pub const {prefix}_DIHEDRALS: &[ParmDihedralRow] = &[")
    for i, j, k, l, divisor, barrier, phase, periodicity, more_terms in t.dihedrals:
        w(f"    ParmDihedralRow {{ i: {types.slot(i)}, j: {types.slot(j)}, k: {types.slot(k)}, "
          f"l: {types.slot(l)}, divisor: {divisor}, barrier: {barrier}, phase_deg: {phase}, "
          f"periodicity: {periodicity}, more_terms: {str(more_terms).lower()} }},")
    w("];")
    w("")

    wild_impropers = sum(1 for row in t.impropers if PARM_WILDCARD in row[:4])
    w(f"/// The {len(t.impropers)} `IMPROPER` rows of `{path.name}`, in file order.")
    w("///")
    w(f"/// {wild_impropers} of them carry a wildcard slot (`None`). The CENTRAL atom is")
    w("/// [`k`](ParmImproperRow::k), the third — that is AMBER's convention, not molrs's.")
    w(RUSTFMT_SKIP)
    w(f"pub const {prefix}_IMPROPERS: &[ParmImproperRow] = &[")
    for i, j, k, l, barrier, phase, periodicity in t.impropers:
        w(f"    ParmImproperRow {{ i: {types.slot(i)}, j: {types.slot(j)}, k: {types.slot(k)}, "
          f"l: {types.slot(l)}, barrier: {barrier}, phase_deg: {phase}, "
          f"periodicity: {periodicity} }},")
    w("];")
    w("")

    w(f"/// The {len(t.nonbonded)} `NONBON` rows of `{path.name}` (`MOD4  RE`: R* and epsilon).")
    w(RUSTFMT_SKIP)
    w(f"pub const {prefix}_NONBONDED: &[ParmNonbondedRow] = &[")
    for atom_type, r_min_half, epsilon in t.nonbonded:
        w(f"    ParmNonbondedRow {{ atom_type: {types.const(atom_type)}, "
          f"r_min_half: {r_min_half}, epsilon: {epsilon} }},")
    w("];")
    w("")

    w(f"/// `{path.name}` as one typed table — {wildcards} of its rows are wildcard rows.")
    w(f"pub const {prefix}: ParmTable = ParmTable {{")
    w(f"    name: {rust_str(path.name)},")
    w(f"    masses: {prefix}_MASSES,")
    w(f"    bonds: {prefix}_BONDS,")
    w(f"    angles: {prefix}_ANGLES,")
    w(f"    dihedrals: {prefix}_DIHEDRALS,")
    w(f"    impropers: {prefix}_IMPROPERS,")
    w(f"    nonbonded: {prefix}_NONBONDED,")
    w("};")
    w("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def rustfmt(paths: list[Path]) -> None:
    """Format in place, so the committed output is `cargo fmt --check`-clean and
    `--check` compares bytes that a formatter cannot perturb."""
    exe = shutil.which("rustfmt")
    if exe is None:
        raise SystemExit("rustfmt not found on PATH; cannot emit fmt-stable tables")
    subprocess.run([exe, "--edition", "2024", *[str(p) for p in paths]], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="where to write the emitted .rs (default: the committed tree)")
    ap.add_argument("--check", action="store_true",
                    help="emit into a temporary directory and fail unless every table "
                         "is byte-identical to the committed one; write nothing")
    args = ap.parse_args()

    amberhome = os.environ.get("AMBERHOME")
    if not amberhome:
        raise SystemExit("AMBERHOME is not set; point it at an AmberTools install")
    root = Path(amberhome)
    src = root / ANTECHAMBER_DIR
    parm = root / PARM_DIR
    if not src.is_dir():
        raise SystemExit(f"no antechamber tables under {src}")
    if not parm.is_dir():
        raise SystemExit(f"no LEaP parm tables under {parm}")

    # The conjugated pairing feeds every ATD table's `alternate` column, so it is
    # read once, up front: a malformed PARMCHK.DAT must fail before anything is
    # emitted, not leave six good tables and one silently unpaired.
    equivalents = parse_parmchk(src / PARMCHK_FILE)

    staged: dict[str, str] = {}
    staged["bccparm.rs"] = emit_bccparm(src / "BCCPARM.DAT", "BCC")
    staged["bccparm_abcg2.rs"] = emit_bccparm(src / "BCCPARM_ABCG2.DAT", "ABCG2")
    staged["gasparm.rs"] = emit_gasparm(src / "GASPARM.DAT")
    staged["parmchk.rs"] = emit_parmchk(src / PARMCHK_FILE, equivalents)
    staged["gaff_empirical.rs"] = emit_empirical_from(src)
    for filename, module, const, gaff_namespace in ATOMTYPE_FILES:
        staged[f"{module}.rs"] = emit_atomtype(
            src / filename, const, equivalents if gaff_namespace else None
        )
    for filename, module, prefix in PARM_FILES:
        staged[f"{module}.rs"] = emit_parm(parm / filename, prefix)

    # Emit into a tempdir first: a mid-run GrammarError must not leave a
    # half-written table behind in the repo.
    with tempfile.TemporaryDirectory(prefix="param-tables-") as tmp:
        tmp_paths = []
        for name, text in staged.items():
            p = Path(tmp) / name
            p.write_text(text, encoding="utf-8")
            tmp_paths.append(p)
        rustfmt(tmp_paths)

        if args.check:
            stale = [p.name for p in tmp_paths
                     if not (OUT_DIR / p.name).is_file()
                     or (OUT_DIR / p.name).read_bytes() != p.read_bytes()]
            for name in stale:
                print(f"  differs  {OUT_DIR.relative_to(REPO) / name}")
            if stale:
                print(f"{len(stale)} of {len(tmp_paths)} tables differ from {amberhome}")
                return 1
            print(f"{len(tmp_paths)} tables byte-identical to {amberhome}")
            return 0

        args.out_dir.mkdir(parents=True, exist_ok=True)
        for p in tmp_paths:
            shutil.copyfile(p, args.out_dir / p.name)

    for name in sorted(staged):
        out = args.out_dir / name
        print(f"  wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}"
              f"  ({len(out.read_text(encoding="utf-8").splitlines())} lines)")
    print(f"{len(staged)} files -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
