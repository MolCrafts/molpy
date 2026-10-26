//! antechamber's post-typing passes: 2-colouring the conjugated systems.
//!
//! An `ATOMTYPE_GFF*.DEF` row can only ever emit the **phase-1** name of a
//! conjugated system — `cc`, `ce`, `cg`, `nc`, `ne`, `pc`, `pe`, and the
//! biphenyl bridge carbon `cp`. The names antechamber answers for the other
//! half — `cd`, `cf`, `ch`, `nd`, `nf`, `pd`, `pf`, `cq` — appear in no `.DEF`
//! row at all, so no amount of rule matching can produce them. antechamber
//! assigns them after the rules, in `atomtype.c`'s `atadjust` and `cpadjust`,
//! and this module is a transcription of those two passes.
//!
//! The pass owns only the *colouring*. The names it colours with are data — the
//! rule's [`alternate`](AtdRule::alternate), which the generator reads out of
//! `PARMCHK.DAT`'s `equivalent_flag` column, together with the pass that owns
//! the pair — so the engine stays table-generic: it never spells a GAFF type,
//! and a table whose rules carry no `alternate` (BCC, ABCG2, GAS, AMBER, SYBYL)
//! passes through untouched.
//!
//! # `atadjust` — the conjugated names
//!
//! Every atom whose rule carries a [`Conjugated`](AlternatePass::Conjugated)
//! alternate takes part. The first of them (in atom order) is coloured `+1`;
//! then the bond list is swept, in bond order, once per participating atom
//! but one. In a sweep, a bond between two participants with one end coloured
//! colours the other end: the same colour across a single bond (1, or 7
//! aromatic single), the opposite one across a double or triple (2, 8, 3).
//! A bond of any other type (9, delocalized) colours nothing. While a sweep has
//! coloured nothing yet, a bond between two uncoloured participants first
//! colours its *first* atom `+1` — that is how a second conjugated system gets
//! its seed. An atom left at `-1` is renamed to its rule's alternate.
//!
//! The colouring therefore follows bond order, never ring position (2-pyridone is
//! `cc cd cd cc`: the bond between its two middle carbons is single), and the
//! subgraph is the conjugated one, not the aromatic one (1,4-benzoquinone is
//! `o c cc cd c o cc cd`; hexatriene is `c2 ce ce cf cf c2`). Where a system's
//! parity is inconsistent — an odd number of double bonds round a ring, as in
//! azulene's perimeter — the answer is whatever the sweep order makes it, which
//! is why the sweep is transcribed rather than replaced by a search.
//!
//! # `cpadjust` — the bridge carbons
//!
//! The same, over the atoms whose rule carries a
//! [`Bridge`](AlternatePass::Bridge) alternate (`cp`), with two differences: only
//! the first of them is ever seeded, and a bond keeps the colour only when it is
//! a plain single (1) — every other bond, an aromatic single (7) included, flips
//! it. So biphenyl's bridge is `cp cp`, and o-terphenyl's middle ring, whose two
//! bridge carbons share an aromatic bond, is `cp … cq`.

use molrs::core::NodeId;

use super::facts::MolFacts;
use crate::ff::params::{AlternatePass, AtdRule};

/// The final atom type of every atom, in `atom_ids` order.
///
/// `assigned[i]` is the rule that matched `atom_ids[i]` — its phase-1 answer.
/// The two passes rename the atoms they colour `-1` to their own rule's
/// alternate. Note "its **own** rule's": the two colours are not two names but
/// two *phases*. Vinylacetylene (`C=C-C#C`) is one system holding a `ce` and a
/// `cg`, and antechamber answers `ce cg`.
pub(super) fn resolve_types(
    atom_ids: &[NodeId],
    assigned: &[&'static AtdRule],
    facts: &MolFacts,
) -> Vec<&'static str> {
    let n = atom_ids.len();
    let in_pass = |pass: AlternatePass| -> Vec<bool> {
        assigned
            .iter()
            .map(|rule| rule.alternate.is_some_and(|alt| alt.pass == pass))
            .collect()
    };
    let conjugated = colour(&in_pass(AlternatePass::Conjugated), &facts.bonds, true);
    let bridge = colour(&in_pass(AlternatePass::Bridge), &facts.bonds, false);

    (0..n)
        .map(|i| match assigned[i].alternate {
            Some(alt) if conjugated[i] == -1 || bridge[i] == -1 => alt.atom_type,
            _ => assigned[i].atom_type,
        })
        .collect()
}

/// One colouring pass over the participating atoms `member`: `atadjust` when
/// `conjugated`, `cpadjust` otherwise. Returns each atom's colour, `0` for an
/// atom the pass never reached.
fn colour(member: &[bool], bonds: &[(usize, usize, i32)], conjugated: bool) -> Vec<i32> {
    let mut colour = vec![0i32; member.len()];
    let Some(first) = member.iter().position(|m| *m) else {
        return colour;
    };
    colour[first] = 1;
    let sweeps = member.iter().filter(|m| **m).count() - 1;
    // Across this bond, the colour of the far end given the near one's — `None`
    // when the bond colours nothing.
    let across = |bond_type: i32, near: i32| -> Option<i32> {
        if conjugated {
            match bond_type {
                1 | 7 => Some(near),
                2 | 8 | 3 => Some(-near),
                _ => None,
            }
        } else if bond_type == 1 {
            Some(near)
        } else {
            Some(-near)
        }
    };
    for _ in 0..sweeps {
        let mut coloured = false;
        for &(i, j, bond_type) in bonds {
            if !(member[i] && member[j]) {
                continue;
            }
            if conjugated && !coloured && colour[i] == 0 && colour[j] == 0 {
                colour[i] = 1;
            }
            if colour[i] == 0 && colour[j] != 0 {
                coloured = true;
                if let Some(c) = across(bond_type, colour[j]) {
                    colour[i] = c;
                }
            }
            if colour[j] == 0 && colour[i] != 0 {
                coloured = true;
                if let Some(c) = across(bond_type, colour[i]) {
                    colour[j] = c;
                }
            }
        }
    }
    colour
}
