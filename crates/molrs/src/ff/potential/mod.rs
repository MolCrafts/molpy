//! Kernels: a force field's terms evaluated on coordinates — the
//! [`Potential`] traits, the generic form kernels and the built-in forms they
//! price (`bond`, `angle`, `dihedral`, `improper`, `pair`), PME's
//! reciprocal-space sum and `cmap`, and
//! the weights a non-bonded kernel takes on close pairs ([`PairWeights`],
//! [`SpecialWeights`]).
//!
//! What a style *is* (its spec) is the IR's ([`crate::ff::ir`]); which kernel
//! prices a style is the style registry's ([`crate::ff::style_registry`]);
//! binding a force field's styles to a frame is the compiler's
//! ([`PotentialCompiler`](crate::ff::compile::PotentialCompiler)); which
//! integrator calls a kernel is `md`'s and `optimize`'s.
//!
//! A [`Potential`] stores pre-resolved topology indices and parameters.
//! Callers pass only flat coordinates — no [`Frame`] in the hot loop.
//! Construction from a [`Frame`] happens once via
//! [`PotentialCompiler::compile`](crate::ff::compile::PotentialCompiler::compile).

pub(crate) mod flat_coords;

pub mod angle;
pub mod bond;
pub mod cmap;
pub mod dihedral;
mod error;
pub mod form_kernel;
pub mod improper;
pub mod kspace;
mod neighbor_pairs;
pub mod pair;
pub(crate) mod param_reads;
pub mod soft;

pub use error::CompileError;

use std::collections::{HashMap, HashSet};

use ndarray::{Array1, Array2, ArrayView2};

use crate::ff::ir::SpecialBonds;
use molrs::core::Block;
use molrs::core::BondDistanceWeights;
use molrs::core::Frame;
use molrs::core::Neighbors;
use molrs::core::Virial;
use molrs::core::keys::{ATOMI, ATOMJ, IS_14, MOL_ID};
use molrs::core::schema::PAIR_OVERRIDE_COLUMNS;
use molrs::core::schema::block_names::{ATOMS, PAIR_OVERRIDES, PAIRS};
use molrs::op::{F, Idx};

/// Above this many atoms, [`intramolecular_pairs`] refuses rather than
/// enumerating.
///
/// The list is every pair in the molecule with no cutoff, so it is `N(N-1)/2`
/// rows — at 50 000 atoms that is 1.2 billion rows and about 11 GiB, which
/// arrives as an OOM kill rather than as an answer. 20 000 is where the block
/// is still under 2 GiB and the caller is still plausibly asking for what this
/// function is for: the intramolecular pairs of one molecule, in free space.
/// A periodic or larger system wants a neighbour list and
/// [`PotentialCompiler::compile_typed`](crate::ff::compile::PotentialCompiler::compile_typed).
///
/// molwasm caps the same path at 2 000 for its own memory budget; this is
/// the native ceiling, not a duplicate of that policy.
pub const MAX_ATOMS_FOR_A_FULL_PAIR_LIST: usize = 20_000;

/// `atomi` + `atomj` + `is_14`, as they land in the block's columns.
const BYTES_PER_PAIR_ROW: usize = 4 + 4 + 1;

/// Build the intramolecular non-bonded `pairs` block (`atomi`, `atomj`, `is_14`)
/// from a frame's bonds: every `i < j` pair, excluding 1-2 and 1-3 pairs and
/// flagging 1-4 pairs, each class its **bond-graph distance**
/// ([`Topology::close_pairs`](molrs::core::Topology::close_pairs)) — the
/// classification [`PotentialCompiler::compile_typed`](crate::ff::compile::PotentialCompiler::compile_typed)
/// weights by and LAMMPS's `special_bonds` applies, so an `angles` or
/// `dihedrals` block that is absent or lists only some terms changes nothing.
///
/// This is the neighbour list that [`PotentialCompiler::compile`](crate::ff::compile::PotentialCompiler::compile) hands to every
/// pair kernel; every force field (GAFF/LAMMPS, OPLS, MMFF, …) shares this
/// one path. Per-pair scaling of the flagged 1-4 pairs is applied by the
/// pair kernels using the force field's 1-4 weight, not baked into this list.
///
/// # Per-pair overrides
///
/// Override cells (`lj_scale`, `coul_scale`, `epsilon`, `sigma`,
/// `charge_product`) already on the frame move onto the same pairs of the
/// new list. They are read from `pairs` when that block is present, and from
/// `pair_overrides` otherwise — the AMBER prmtop reader's 1-4 rows whose
/// `SCEE` / `SCNB` differ from the field's. An override on a pair the new
/// list excludes is an [`Err`], since the list would drop it.
///
/// # Why this needs the force field's weights
///
/// Which 1-2 / 1-3 pairs belong in the list *is* a force-field decision, so
/// this function does not make it on its own. LAMMPS's `special_bonds fene`
/// (`[0, 1, 1]`) keeps 1-3 pairs at full strength, and a FENE chain without
/// them has nothing holding it open. `special` answers it, via
/// [`SpecialBonds::compiled_inclusion`](crate::ff::ir::SpecialBonds::compiled_inclusion),
/// which is also where weights this list cannot express become an [`Err`]
/// rather than a silently different force field.
///
/// [`SpecialBonds::EXCLUDE_12_13`](crate::ff::ir::SpecialBonds::EXCLUDE_12_13)
/// excludes both classes.
///
/// # Candidate pairs
///
/// With `neighbors: None` the candidates are every `i < j` pair of the
/// frame's atoms (refused past [`MAX_ATOMS_FOR_A_FULL_PAIR_LIST`]). With
/// `Some(table)` they are the pairs of that cutoff-bounded neighbour table,
/// by the same rules: each unordered pair appears once, as `(lo, hi)`, in the
/// order the table first reports it, so a full (both-direction) table and a
/// half-shell one give the same block.
pub fn intramolecular_pairs(
    frame: &Frame,
    special: &SpecialBonds,
    neighbors: Option<&Neighbors>,
) -> Result<Block, String> {
    if let Some(neighbors) = neighbors {
        return neighbor_pairs::over_neighbors(frame, special, neighbors);
    }
    let [keep_12, keep_13] = special.compiled_inclusion()?;
    let n_atoms = frame.get(ATOMS).and_then(|b| b.n_rows()).unwrap_or(0);
    if n_atoms > MAX_ATOMS_FOR_A_FULL_PAIR_LIST {
        return Err(format!(
            "intramolecular_pairs: {n_atoms} atoms would enumerate {} pairs \
             (~{} GiB) — this list is every pair in the molecule, with no cutoff. \
             Above {MAX_ATOMS_FOR_A_FULL_PAIR_LIST} atoms build a neighbour list \
             instead and evaluate through PotentialCompiler::compile_typed.",
            n_atoms * (n_atoms - 1) / 2,
            (n_atoms * (n_atoms - 1) / 2 * BYTES_PER_PAIR_ROW) >> 30,
        ));
    }
    let classes = bond_classes(frame, n_atoms)?;

    let mut pi: Vec<Idx> = Vec::new();
    let mut pj: Vec<Idx> = Vec::new();
    let mut p14: Vec<bool> = Vec::new();
    for a in 0..n_atoms {
        for b in (a + 1)..n_atoms {
            let class = classes.get(&(a, b)).copied();
            if (!keep_12 && class == Some(1)) || (!keep_13 && class == Some(2)) {
                continue;
            }
            pi.push(a as Idx);
            pj.push(b as Idx);
            p14.push(class == Some(3));
        }
    }

    let mut pairs = Block::new();
    if !pi.is_empty() {
        let overrides = carried_overrides(frame, &pi, &pj)?;
        pairs
            .insert(ATOMI, Array1::from_vec(pi).into_dyn(), None)
            .expect("fresh pairs block");
        pairs
            .insert(ATOMJ, Array1::from_vec(pj).into_dyn(), None)
            .expect("fresh pairs block");
        pairs
            .insert(IS_14, Array1::from_vec(p14).into_dyn(), None)
            .expect("fresh pairs block");
        for (key, values, valid) in overrides {
            pairs
                .insert(key, Array1::from_vec(values).into_dyn(), Some(valid))
                .map_err(|e| e.to_string())?;
        }
    } else if let Some((i, j)) = override_pairs(frame)?.into_iter().next() {
        return Err(excluded_override(i, j));
    }
    Ok(pairs)
}

/// The `pairs` block [`PotentialCompiler::compile`](crate::ff::compile::PotentialCompiler::compile)
/// should price.
///
/// No `pairs` block: [`intramolecular_pairs`] builds one (bond graph, or the
/// pairs of `neighbors`) and carries `pair_overrides` onto it. A `pairs`
/// block and no neighbor table: that block, unchanged. A `pairs` block and a
/// neighbor table: the intersection. Intramolecular rows of the existing
/// list stay as written (`is_14` and override cells included), so a GROMACS
/// exclusion or a `[ pairs ]` flag the bond graph would not assign survives.
/// Intermolecular rows are authoritative only when the list already contains
/// a cross-`mol_id` pair, or the frame has a single `mol_id` (no column
/// counts as one molecule). Otherwise the list is the intramolecular one a
/// reader writes above [`MAX_ATOMS_FOR_A_FULL_PAIR_LIST`], and intermolecular
/// candidates come from the neighbor table under `special`.
///
/// # Errors
///
/// The errors of [`intramolecular_pairs`]. A `pairs` row whose endpoints fall
/// outside `atoms.mol_id`. Two rows of one unordered pair. A column other
/// than `atomi`, `atomj`, `is_14` and the override columns on a list that the
/// intersection has to extend with a neighbor pair. A kept row of a block
/// with no bool `is_14` column.
pub fn install_pairs(
    frame: &Frame,
    special: &SpecialBonds,
    neighbors: Option<&Neighbors>,
) -> Result<Block, String> {
    let Some(existing) = frame.get(PAIRS) else {
        return intramolecular_pairs(frame, special, neighbors);
    };
    let Some(neighbors) = neighbors else {
        return Ok(existing.clone());
    };
    intersect_pairs(frame, existing, special, neighbors)
}

/// One output row of [`intersect_pairs`]: an existing `pairs` row, or a
/// neighbor pair the capped list does not hold.
enum KeptPair {
    Row(usize),
    Fresh { lo: Idx, hi: Idx, is_14: bool },
}

/// `neighbors` cut by the evaluation list `existing`.
fn intersect_pairs(
    frame: &Frame,
    existing: &Block,
    special: &SpecialBonds,
    neighbors: &Neighbors,
) -> Result<Block, String> {
    let (ai, aj) = pair_endpoints(existing)?;
    let mut index: HashMap<(usize, usize), usize> = HashMap::with_capacity(ai.len());
    for (r, (&a, &b)) in ai.iter().zip(aj.iter()).enumerate() {
        let pair = (a.min(b) as usize, a.max(b) as usize);
        if index.insert(pair, r).is_some() {
            return Err(format!(
                "install_pairs: atoms {} and {} are on two pairs rows",
                pair.0, pair.1
            ));
        }
    }
    let mol = molecule_ids(frame)?;
    let closed = intermolecular_list_is_closed(mol.as_ref(), &ai, &aj)?;
    let [keep_12, keep_13] = if closed {
        [true, true]
    } else {
        special.compiled_inclusion()?
    };
    let classes = if closed {
        HashMap::new()
    } else {
        let n_atoms = frame.get(ATOMS).and_then(|b| b.n_rows()).unwrap_or(0);
        bond_classes(frame, n_atoms)?
    };

    let mut seen = HashSet::new();
    let mut kept = Vec::new();
    for (&a, &b) in neighbors
        .query_point_indices()
        .iter()
        .zip(neighbors.point_indices())
    {
        let (lo, hi) = (a.min(b) as usize, a.max(b) as usize);
        if lo == hi || !seen.insert((lo, hi)) {
            continue;
        }
        let same = same_molecule(mol.as_ref(), lo, hi)?;
        if same || closed {
            if let Some(&row) = index.get(&(lo, hi)) {
                kept.push(KeptPair::Row(row));
            }
        } else {
            let class = classes.get(&(lo, hi)).copied();
            if (!keep_12 && class == Some(1)) || (!keep_13 && class == Some(2)) {
                continue;
            }
            kept.push(KeptPair::Fresh {
                lo: lo as Idx,
                hi: hi as Idx,
                is_14: class == Some(3),
            });
        }
    }
    materialize_intersection(existing, &kept)
}

/// `(atomi, atomj)` of a `pairs` block. A column-less block (no rows) has
/// neither, and that is an empty list.
fn pair_endpoints(block: &Block) -> Result<(Vec<Idx>, Vec<Idx>), String> {
    if block.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let column = |key: &str| -> Result<Vec<Idx>, String> {
        block
            .get(key)
            .and_then(|c| c.get::<u64>())
            .map(|c| c.iter().copied().collect())
            .ok_or_else(|| format!("install_pairs: pairs block missing \"{key}\""))
    };
    Ok((column(ATOMI)?, column(ATOMJ)?))
}

/// `atoms.mol_id`, when the frame has that column.
fn molecule_ids(frame: &Frame) -> Result<Option<ndarray::ArrayView1<'_, Idx>>, String> {
    let Some(atoms) = frame.get(ATOMS) else {
        return Ok(None);
    };
    let Some(col) = atoms.get(MOL_ID) else {
        return Ok(None);
    };
    let ids = col
        .get::<u64>()
        .ok_or_else(|| "install_pairs: atoms.mol_id must be an unsigned column".to_string())?;
    ids.view()
        .into_dimensionality()
        .map(Some)
        .map_err(|_| "install_pairs: atoms.mol_id must be one-dimensional".to_string())
}

/// Whether intermolecular membership of `pairs` is closed.
///
/// One distinct `mol_id` (or no column) has no intermolecular pair, so the
/// list is the whole answer. Several molecules are closed when a row already
/// crosses two of them — the reader included those pairs — and open when
/// every row is intramolecular, which is the list written once the system
/// exceeds [`MAX_ATOMS_FOR_A_FULL_PAIR_LIST`].
fn intermolecular_list_is_closed(
    mol: Option<&ndarray::ArrayView1<'_, Idx>>,
    ai: &[Idx],
    aj: &[Idx],
) -> Result<bool, String> {
    let Some(mol) = mol else {
        return Ok(true);
    };
    if mol.iter().collect::<HashSet<_>>().len() <= 1 {
        return Ok(true);
    }
    for (&a, &b) in ai.iter().zip(aj.iter()) {
        let (ma, mb) = molecule_pair(mol, a as usize, b as usize)?;
        if ma != mb {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether atoms `lo` and `hi` share a molecule. No `mol_id` column means
/// they do.
fn same_molecule(
    mol: Option<&ndarray::ArrayView1<'_, Idx>>,
    lo: usize,
    hi: usize,
) -> Result<bool, String> {
    let Some(mol) = mol else {
        return Ok(true);
    };
    let (a, b) = molecule_pair(mol, lo, hi)?;
    Ok(a == b)
}

fn molecule_pair(
    mol: &ndarray::ArrayView1<'_, Idx>,
    lo: usize,
    hi: usize,
) -> Result<(Idx, Idx), String> {
    match (mol.get(lo), mol.get(hi)) {
        (Some(&a), Some(&b)) => Ok((a, b)),
        _ => Err(format!(
            "install_pairs: atoms {lo} and {hi} are outside the atoms mol_id column ({})",
            mol.len()
        )),
    }
}

/// The columns an intersection may extend with a fresh neighbor row.
const INSTALL_COLUMNS: [&str; 8] = [
    ATOMI,
    ATOMJ,
    IS_14,
    "epsilon",
    "sigma",
    "charge_product",
    "lj_scale",
    "coul_scale",
];

fn materialize_intersection(existing: &Block, kept: &[KeptPair]) -> Result<Block, String> {
    let rows: Vec<usize> = kept
        .iter()
        .filter_map(|row| match row {
            KeptPair::Row(r) => Some(*r),
            KeptPair::Fresh { .. } => None,
        })
        .collect();
    if rows.len() == kept.len() {
        return existing.select_rows(&rows).map_err(|e| e.to_string());
    }
    if let Some(key) = existing.keys().find(|key| !INSTALL_COLUMNS.contains(key)) {
        return Err(format!(
            "install_pairs: pairs column '{key}' has no form on a neighbor pair the list does \
             not already hold"
        ));
    }
    let is_14 = existing.get(IS_14).and_then(|c| c.get::<bool>());
    if kept.iter().any(|row| matches!(row, KeptPair::Row(_))) && is_14.is_none() {
        return Err("install_pairs: pairs block missing a bool \"is_14\" column".to_string());
    }
    let n = kept.len();
    let mut pi = Vec::with_capacity(n);
    let mut pj = Vec::with_capacity(n);
    let mut p14 = Vec::with_capacity(n);
    let (ai, aj) = if kept.iter().any(|row| matches!(row, KeptPair::Row(_))) {
        pair_endpoints(existing)?
    } else {
        (Vec::new(), Vec::new())
    };
    for row in kept {
        match row {
            KeptPair::Row(r) => {
                pi.push(ai[*r].min(aj[*r]));
                pj.push(ai[*r].max(aj[*r]));
                p14.push(is_14.expect("checked above")[*r]);
            }
            KeptPair::Fresh { lo, hi, is_14 } => {
                pi.push(*lo);
                pj.push(*hi);
                p14.push(*is_14);
            }
        }
    }
    let mut out = Block::new();
    if n == 0 {
        return Ok(out);
    }
    out.insert(ATOMI, Array1::from_vec(pi).into_dyn(), None)
        .map_err(|e| e.to_string())?;
    out.insert(ATOMJ, Array1::from_vec(pj).into_dyn(), None)
        .map_err(|e| e.to_string())?;
    out.insert(IS_14, Array1::from_vec(p14).into_dyn(), None)
        .map_err(|e| e.to_string())?;
    for key in PAIR_OVERRIDE_COLUMNS {
        let Some(col) = existing.get(key) else {
            continue;
        };
        let values = col
            .get::<f64>()
            .ok_or_else(|| format!("pairs: the per-pair override column '{key}' must be float"))?;
        let valid = existing.validity(key);
        let mut new_values = Vec::with_capacity(n);
        let mut new_valid = Vec::with_capacity(n);
        for row in kept {
            match row {
                KeptPair::Row(r) => {
                    let holds = valid.is_none_or(|mask| mask[*r]);
                    new_values.push(if holds { values[*r] } else { 0.0 });
                    new_valid.push(holds);
                }
                KeptPair::Fresh { .. } => {
                    new_values.push(0.0);
                    new_valid.push(false);
                }
            }
        }
        if new_valid.iter().any(|&holds| holds) {
            out.insert(
                key,
                Array1::from_vec(new_values).into_dyn(),
                Some(new_valid),
            )
            .map_err(|e| e.to_string())?;
        }
    }
    Ok(out)
}

/// The bond distance (1, 2 or 3) of every pair at most three bonds apart in
/// `frame`'s `bonds`: [`Topology::close_pairs`](molrs::core::Topology::close_pairs).
/// A frame with no atoms has none.
pub(crate) fn bond_classes(
    frame: &Frame,
    n_atoms: usize,
) -> Result<HashMap<(usize, usize), usize>, String> {
    if n_atoms == 0 {
        return Ok(HashMap::new());
    }
    let topology = molrs::core::Topology::from_frame(frame).map_err(|e| e.to_string())?;
    Ok(topology.close_pairs(3))
}

/// One override column of a new pair list: its name, values and validity.
type OverrideColumn = (&'static str, Vec<F>, Vec<bool>);

/// The per-pair override cells ([`PAIR_OVERRIDE_COLUMNS`]) of the frame's
/// own rows, moved onto the rows `(pi, pj)` of the new list: a reader's
/// per-pair data (an AMBER torsion's own SCEE / SCNB, a GROMACS `[ pairs ]`
/// row's parameters) survives building the full list. Read from `pairs`
/// when that block is present, and from `pair_overrides` otherwise. An
/// override on a pair the new list leaves out (1-2 / 1-3) is an `Err`: it
/// would be dropped.
fn carried_overrides(frame: &Frame, pi: &[Idx], pj: &[Idx]) -> Result<Vec<OverrideColumn>, String> {
    let Some(old) = override_source(frame) else {
        return Ok(Vec::new());
    };
    let (Some(oi), Some(oj)) = (
        old.get(ATOMI).and_then(|c| c.get::<u64>()),
        old.get(ATOMJ).and_then(|c| c.get::<u64>()),
    ) else {
        return Ok(Vec::new());
    };
    let row_of: HashMap<(Idx, Idx), usize> = pi
        .iter()
        .zip(pj)
        .enumerate()
        .map(|(r, (&a, &b))| ((a, b), r))
        .collect();
    let mut out = Vec::new();
    for key in PAIR_OVERRIDE_COLUMNS {
        let Some(col) = old.get(key) else {
            continue;
        };
        let values = col.get::<f64>().ok_or_else(|| {
            format!("pairs or pair_overrides: the per-pair override column '{key}' must be float")
        })?;
        let valid = old.validity(key);
        let mut new_values = vec![0.0; pi.len()];
        let mut new_valid = vec![false; pi.len()];
        // One row owns a pair, across every override column. A second row
        // last-writing the cell is the split `exceptions::plan` already rejects.
        let mut row_of_pair: HashMap<(Idx, Idx), usize> = HashMap::new();
        for (r, (&a, &b)) in oi.iter().zip(oj.iter()).enumerate() {
            if valid.is_some_and(|m| !m[r]) {
                continue;
            }
            let v = values[r];
            if !v.is_finite() {
                return Err(format!(
                    "pairs or pair_overrides row {r}: '{key}' = {v}; a null cell is a \
                     validity mask, not a non-finite value"
                ));
            }
            let pair = (a.min(b), a.max(b));
            match row_of_pair.get(&pair) {
                Some(&prev) if prev != r => {
                    return Err(format!(
                        "pairs or pair_overrides: atoms {} and {} carry per-pair overrides \
                         on two rows; a pair has one set of exception parameters",
                        pair.0, pair.1
                    ));
                }
                None => {
                    row_of_pair.insert(pair, r);
                }
                _ => {}
            }
            let at = *row_of
                .get(&pair)
                .ok_or_else(|| excluded_override(pair.0 as usize, pair.1 as usize))?;
            new_values[at] = v;
            new_valid[at] = true;
        }
        if new_valid.iter().any(|&v| v) {
            out.push((key, new_values, new_valid));
        }
    }
    Ok(out)
}

/// `pairs` when the frame has it, otherwise `pair_overrides`.
fn override_source(frame: &Frame) -> Option<&Block> {
    frame.get(PAIRS).or_else(|| frame.get(PAIR_OVERRIDES))
}

/// The pairs of the frame's override rows that carry an override cell.
fn override_pairs(frame: &Frame) -> Result<Vec<(usize, usize)>, String> {
    let Some(old) = override_source(frame) else {
        return Ok(Vec::new());
    };
    let (Some(oi), Some(oj)) = (
        old.get(ATOMI).and_then(|c| c.get::<u64>()),
        old.get(ATOMJ).and_then(|c| c.get::<u64>()),
    ) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for key in PAIR_OVERRIDE_COLUMNS {
        if old.get(key).is_none() {
            continue;
        }
        let valid = old.validity(key);
        for (r, (&a, &b)) in oi.iter().zip(oj.iter()).enumerate() {
            if valid.is_none_or(|m| m[r]) {
                out.push((a.min(b) as usize, a.max(b) as usize));
            }
        }
    }
    Ok(out)
}

fn excluded_override(i: usize, j: usize) -> String {
    format!(
        "intramolecular_pairs: the frame's pairs or pair_overrides row for atoms {i} and {j} \
         carries a per-pair override, and the force field's special_bonds exclude that pair \
         (1-2 or 1-3), so the override would be dropped"
    )
}

/// `(lo, hi)` end-atom pairs of a topology block (bond ends, angle i–k,
/// dihedral i–l), one per row in row order. A missing block or column yields
/// none.
pub(crate) fn end_pairs(
    frame: &Frame,
    block: &str,
    col_a: &str,
    col_b: &str,
) -> Vec<(usize, usize)> {
    let Some(b) = frame.get(block) else {
        return Vec::new();
    };
    let (Some(a_col), Some(b_col)) = (
        b.get(col_a).and_then(|c| c.get::<u64>()),
        b.get(col_b).and_then(|c| c.get::<u64>()),
    ) else {
        return Vec::new();
    };
    a_col
        .iter()
        .zip(b_col.iter())
        .map(|(&i, &j)| {
            let (i, j) = (i as usize, j as usize);
            (i.min(j), i.max(j))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Accumulator
// ---------------------------------------------------------------------------

/// What an evaluation adds into: the running energy, a force buffer, and —
/// when the caller asked for it — the virial `Σ f ⊗ r`.
///
/// Every evaluation primitive ([`Potential::evaluate`],
/// [`IndexedTerms::evaluate_terms`], [`PairDriven::evaluate_pairs`]) takes
/// one of these and *adds* to it, so energy-only, forces-only, energy and
/// forces, and energy-forces-virial are one call each: build the accumulator
/// for the channels wanted, call the primitive, read the channels back. An
/// aggregate passes the same accumulator to every member, which is what lets
/// a whole step own one force buffer instead of one allocation per member.
///
/// # The channels
///
/// * **energy** — always tallied. It is one add per term.
/// * **forces** — always tallied, into a buffer of `coords.len()` (`3·N`):
///   one the accumulator owns ([`new`](Self::new)), or the caller's
///   ([`adding_into`](Self::adding_into)). Not optional, because no kernel
///   has a cheaper energy-only path: `dE/dr` is the energy's own derivative,
///   evaluated in the same expression, and skipping it would save the stores
///   and nothing else. An energy-only caller reads [`energy`](Self::energy)
///   and drops the buffer.
/// * **virial** — only when requested ([`with_virial`](Self::with_virial)).
///   It is the one channel that costs real work per pair, so a kernel that
///   can tally one asks [`forces_with_virial`](Self::forces_with_virial)
///   whether to, once per call, and runs a loop monomorphised on the answer
///   — the inner loop never branches on it.
///
/// # A missing virial is `None`, never zero
///
/// A kernel adds forces through exactly one of two doors.
/// [`forces_with_virial`](Self::forces_with_virial) is for a kernel that
/// tallies the virial of what it adds; [`forces_without_virial`](Self::forces_without_virial)
/// is for every other one, and taking it marks a requested virial *lost* —
/// [`virial`](Self::virial) then answers `None` for the whole evaluation. A
/// virial missing a term is not a small error, it is a different quantity,
/// and a pressure computed from it would look entirely plausible. Because the
/// door is the declaration, an aggregate needs no bookkeeping of its own: a
/// bonded member, an external field or a compiled pair list loses the virial
/// by adding its forces, and a member that adds no force has no virial to
/// lose.
#[derive(Debug)]
pub struct Accumulator<'a> {
    energy: F,
    forces: ForceBuffer<'a>,
    virial: VirialChannel,
}

/// Where an [`Accumulator`]'s forces land.
#[derive(Debug)]
enum ForceBuffer<'a> {
    /// Zeroed at construction and handed back by [`Accumulator::into_forces`].
    Owned(Vec<F>),
    /// The caller's buffer, added into.
    Borrowed(&'a mut [F]),
}

/// The state of an [`Accumulator`]'s virial channel.
#[derive(Clone, Copy, Debug)]
enum VirialChannel {
    /// Not requested: kernels skip the tally.
    Off,
    /// Requested, and every force added so far came with its virial.
    Open(Virial),
    /// Requested, and some force was added without one.
    Lost,
}

impl Accumulator<'static> {
    /// Energy and forces into a zeroed buffer of `n_coords` (`3·N`)
    /// components the accumulator owns.
    pub fn new(n_coords: usize) -> Self {
        Self {
            energy: 0.0,
            forces: ForceBuffer::Owned(vec![0.0; n_coords]),
            virial: VirialChannel::Off,
        }
    }
}

impl<'a> Accumulator<'a> {
    /// Energy and forces added into `forces`, the caller's buffer — which
    /// keeps whatever it already held.
    pub fn adding_into(forces: &'a mut [F]) -> Self {
        Self {
            energy: 0.0,
            forces: ForceBuffer::Borrowed(forces),
            virial: VirialChannel::Off,
        }
    }

    /// Also tally the virial `Σ f ⊗ r` — see [`virial`](Self::virial).
    pub fn with_virial(mut self) -> Self {
        self.virial = VirialChannel::Open(Virial::ZERO);
        self
    }

    /// The energy added so far.
    pub fn energy(&self) -> F {
        self.energy
    }

    /// The force buffer: `-∇E` of everything added so far, plus whatever a
    /// borrowed buffer held before.
    pub fn forces(&self) -> &[F] {
        match &self.forces {
            ForceBuffer::Owned(v) => v,
            ForceBuffer::Borrowed(s) => s,
        }
    }

    /// The force buffer as a vector: an owned one moves out; a borrowed one is
    /// copied (its caller already holds it, so has no reason to ask).
    pub fn into_forces(self) -> Vec<F> {
        match self.forces {
            ForceBuffer::Owned(v) => v,
            ForceBuffer::Borrowed(s) => s.to_vec(),
        }
    }

    /// The virial `Σ f ⊗ r` of everything added so far.
    ///
    /// `None` when it was not requested, or when some contribution added
    /// forces without one (see the type's documentation) — never a zero
    /// standing in for a term nobody tallied.
    pub fn virial(&self) -> Option<Virial> {
        match self.virial {
            VirialChannel::Open(w) => Some(w),
            VirialChannel::Off | VirialChannel::Lost => None,
        }
    }

    /// Kernel side: add `energy` to the running total.
    #[inline]
    pub fn add_energy(&mut self, energy: F) {
        self.energy += energy;
    }

    /// Kernel side: the force buffer, for a kernel that does **not** tally the
    /// virial of what it adds. A requested virial is lost from here on.
    #[inline]
    pub fn forces_without_virial(&mut self) -> &mut [F] {
        if let VirialChannel::Open(_) = self.virial {
            self.virial = VirialChannel::Lost;
        }
        self.forces_mut()
    }

    /// Kernel side: the force buffer and, when the virial was requested and
    /// is still whole, the running virial to add this kernel's `Σ f ⊗ r` to.
    ///
    /// Ask once per call and branch once on the answer, outside the loop.
    #[inline]
    pub fn forces_with_virial(&mut self) -> (&mut [F], Option<&mut Virial>) {
        let virial = match &mut self.virial {
            VirialChannel::Open(w) => Some(w),
            VirialChannel::Off | VirialChannel::Lost => None,
        };
        let forces: &mut [F] = match &mut self.forces {
            ForceBuffer::Owned(v) => v,
            ForceBuffer::Borrowed(s) => s,
        };
        (forces, virial)
    }

    fn forces_mut(&mut self) -> &mut [F] {
        match &mut self.forces {
            ForceBuffer::Owned(v) => v,
            ForceBuffer::Borrowed(s) => s,
        }
    }
}

// ---------------------------------------------------------------------------
// Potential trait
// ---------------------------------------------------------------------------

/// Energy and forces from coordinates alone.
///
/// A `Potential` is **molecule-bound**: its per-element parameters are expanded
/// against the molecule's topology once at [`PotentialCompiler::compile`](crate::ff::compile::PotentialCompiler::compile)
/// (string type labels resolved to per-bond/angle/… arrays). Evaluation
/// therefore takes only coordinates — there is no per-call topology resolution.
///
/// This is the whole of what every potential can do. Two capabilities that only
/// some have are separate traits, so that a caller which needs one says so in a
/// type instead of asking at run time:
///
/// * [`IndexedTerms`] — the rows are named by an index table the caller may
///   replace. Every bonded kernel.
/// * [`PairDriven`] — the sum runs over whatever pairs a neighbour search turns
///   up. Every pair kernel.
///
/// Each of the three has exactly one evaluation primitive, and all three add
/// into an [`Accumulator`]; which channels come back is the accumulator's
/// business, not a family of methods'.
///
/// [`ForceTerm`] is the three of them as one value, chosen when the kernel is
/// built. It exists because a `Box<dyn Potential>` cannot be asked which of the
/// two it also is, and `terms()` is the wrong place to ask: its job is to
/// return a table, and it allocates one per member per call.
///
/// The geometry optimizer ([`crate::optimize::Lbfgs`]) depends on this trait —
/// not the other way around.
pub trait Potential: Send + Sync {
    /// Add this potential's energy and forces (`-∇E`) at `coords` — flat
    /// `[x0, y0, z0, x1, …]` — into `acc`.
    ///
    /// A kernel that tallies no virial adds its forces through
    /// [`Accumulator::forces_without_virial`]; one that does, through
    /// [`Accumulator::forces_with_virial`].
    ///
    /// Required: there is no default, so every implementor states how it
    /// adds its energy and forces.
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>);
}

/// A potential whose rows are named by an index table the caller can replace.
///
/// Implementing this is a declaration that *which atoms* is separable from
/// *what the parameters are*: row `r` of [`terms`](IndexedTerms::terms) belongs
/// with row `r` of the parameters, and a caller may hand back a different table
/// naming different atoms for the same rows. That is what lets a periodic
/// régime point a bond at a periodic copy without the kernel ever learning that
/// copies exist.
///
/// Every bonded kernel implements it; no pair kernel does, because a pair
/// kernel's rows are not fixed — they are whatever the neighbour search found.
pub trait IndexedTerms: Potential {
    /// The atom indices this kernel resolved at construction, `(n_terms,
    /// arity)` — arity 2 for a bond, 3 for an angle, 4 for a dihedral or an
    /// improper.
    fn terms(&self) -> Array2<u32>;

    /// [`Potential::evaluate`] with `terms` in place of the indices resolved
    /// at construction.
    ///
    /// `terms` must have the shape [`terms`](IndexedTerms::terms) returned: the
    /// row *set* is fixed — it is the terms the force field declared — and only
    /// which atoms each row names may differ.
    ///
    /// There is no default. A default that ignored `terms` would be a correct
    /// fallback for a kernel holding no indices and a silently wrong answer for
    /// one that does, and only the second kind is in this trait.
    fn evaluate_terms(&self, coords: &[F], terms: ArrayView2<'_, u32>, acc: &mut Accumulator<'_>);
}

/// A potential summed over whatever pairs a neighbour search turns up.
///
/// Every pair kernel implements it. Nothing else should: the two required
/// methods are the ones a caller maintaining a live neighbour table needs
/// answered honestly, and a potential that ignores the table cannot answer
/// them.
pub trait PairDriven: Potential {
    /// Add this kernel's contribution over the pair table `pairs` into `acc`,
    /// scaling each pair.
    ///
    /// * **`factor`** is one weight per pair, aligned with the table's rows, or
    ///   empty meaning all ones. A weight of exactly zero *skips* the pair
    ///   rather than multiplying it, because a bonded pair sits at bond length
    ///   where a repulsive term is enormous. Carrying the weights per pair is
    ///   what lets a caller stop rebuilding the pair table once per distinct
    ///   weight.
    /// * **the virial**, when `acc` asks for one, is tallied in the same loop
    ///   that makes the forces. It cannot be recovered from the forces
    ///   afterwards: under periodic boundaries `Σ_a f_a ⊗ x_a` over the stored
    ///   coordinates depends on where the cell's origin happens to be, and a
    ///   sum over separations does not — so a kernel tallies it here, where
    ///   both terms of each pair are still in hand. A kernel that cannot (one
    ///   bound to a compiled pair list) adds through
    ///   [`Accumulator::forces_without_virial`], and the virial reads `None`.
    ///
    /// No default: one that ignored `factor` would be right for a potential
    /// that does not sum over the pair table and would silently lose a force
    /// field's exclusions for one that does. Only the second kind is in this
    /// trait, so the question is asked of every implementor.
    fn evaluate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        acc: &mut Accumulator<'_>,
    );

    /// Whether this kernel's parameters are bound to one fixed pair list.
    ///
    /// Such a kernel answers for **that** list and no other. Handed a
    /// neighbour table it does not read it — it returns the frozen sum — and
    /// that is not a cheaper route to the same number, it is a different
    /// question silently unanswered. Worse, the caller goes on maintaining and
    /// reporting a live list that provably does not enter the answer.
    ///
    /// A periodic force path refuses such a member at construction. No default:
    /// a kernel built from one pair list and a kernel keyed on the atoms are
    /// the same Rust type in molrs (the list is a private source variant), so
    /// the answer cannot be inferred and has to be given.
    fn binds_a_fixed_pair_list(&self) -> bool;

    /// Whether pricing this member needs a neighbour table handed to it each
    /// step: true for a kernel over a type table (`PairLjCut`, a
    /// `compile_typed` kernel), false for one bound to a fixed pair list. An
    /// aggregate asks its members ([`Potentials`]).
    fn reads_a_neighbor_table(&self) -> bool {
        !self.binds_a_fixed_pair_list()
    }
}

/// One member of a force evaluation, with its role fixed when it was built.
///
/// A force evaluation sums members that play different parts — a bonded term
/// reads an index table, a pair term reads a neighbour table, an external field
/// reads neither. Which part a member plays is settled by its constructor, and
/// this is where that answer is kept, so no loop has to re-derive it per step
/// and no method has to carry a default that is wrong for half its implementors.
pub enum ForceTerm {
    /// A bonded term: evaluated against an index table the caller supplies.
    Indexed(Box<dyn IndexedTerms>),
    /// A non-bonded term: evaluated against a neighbour table, with per-pair
    /// weights.
    Pair(Box<dyn PairDriven>),
    /// Everything else — an external field, a restraint, a constant force.
    /// Evaluated from coordinates alone.
    Plain(Box<dyn Potential>),
}

impl ForceTerm {
    /// A bonded term — one whose rows are named by an index table.
    pub fn indexed(p: impl IndexedTerms + 'static) -> Self {
        ForceTerm::Indexed(Box::new(p))
    }

    /// A non-bonded term — one summed over a neighbour table.
    pub fn pair(p: impl PairDriven + 'static) -> Self {
        ForceTerm::Pair(Box::new(p))
    }

    /// Anything else — evaluated from coordinates alone.
    pub fn plain(p: impl Potential + 'static) -> Self {
        ForceTerm::Plain(Box::new(p))
    }

    /// This member as a plain potential, whatever part it plays.
    pub fn as_potential(&self) -> &dyn Potential {
        match self {
            ForceTerm::Indexed(p) => &**p,
            ForceTerm::Pair(p) => &**p,
            ForceTerm::Plain(p) => &**p,
        }
    }

    /// The index table, for a bonded member.
    pub fn terms(&self) -> Option<Array2<u32>> {
        match self {
            ForceTerm::Indexed(p) => Some(p.terms()),
            _ => None,
        }
    }

    /// Whether this member is bound to one fixed pair list — see
    /// [`PairDriven::binds_a_fixed_pair_list`]. A member that reads no pair
    /// table is bound to none.
    pub fn binds_a_fixed_pair_list(&self) -> bool {
        match self {
            ForceTerm::Pair(p) => p.binds_a_fixed_pair_list(),
            _ => false,
        }
    }
}

impl std::fmt::Debug for ForceTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ForceTerm::Indexed(_) => "ForceTerm::Indexed",
            ForceTerm::Pair(_) => "ForceTerm::Pair",
            ForceTerm::Plain(_) => "ForceTerm::Plain",
        })
    }
}

impl Potential for ForceTerm {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        self.as_potential().evaluate(coords, acc);
    }
}

impl Potential for Box<dyn Potential> {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        (**self).evaluate(coords, acc);
    }
}

// ---------------------------------------------------------------------------
// Potentials collection
// ---------------------------------------------------------------------------

/// A kernel built for a neighbour-driven evaluation, and which of a force
/// field's special-bonds weight sets scales it. `None` for a bonded kernel:
/// it *is* the bonded interaction, not a scaled copy of one.
pub type ScaledTerm = (ForceTerm, Option<crate::ff::ir::SpecialClass>);

/// One member of a neighbour-driven evaluation: the kernel, and the weights
/// its non-bonded term takes.
///
/// The weights travel with the member because a force field may scale close
/// van-der-Waals and electrostatic neighbours differently, and in molrs those
/// are separate kernels.
pub type WeightedTerm = (ForceTerm, Option<PairWeights>);

/// The weights a neighbour-driven non-bonded member applies: by bond distance
/// (`special_bonds`), and zero on the pairs the 1-4 exceptions kernel prices
/// in its place (a `pairs` row with per-pair overrides).
///
/// A neighbour table finds every pair inside the cutoff, so a pair that an
/// exception prices would otherwise be priced twice — once here at its
/// class's weight, once there at its own.
#[derive(Debug, Clone, PartialEq)]
pub struct PairWeights {
    by_distance: BondDistanceWeights,
    replaced: Vec<(usize, usize)>,
}

impl PairWeights {
    /// Weights by bond distance, with `replaced` pairs (either order) at 0.
    pub fn new(by_distance: BondDistanceWeights, replaced: Vec<(usize, usize)>) -> Self {
        Self {
            by_distance,
            replaced,
        }
    }

    /// The weights by bond distance.
    pub fn by_distance(&self) -> &BondDistanceWeights {
        &self.by_distance
    }

    /// The pairs weighted 0 whatever their bond distance.
    pub fn replaced(&self) -> &[(usize, usize)] {
        &self.replaced
    }

    /// The weights on `topo`'s close pairs: per atom, its partners whose
    /// weight is not 1 — the lists
    /// [`Topology::special_weights`](molrs::core::Topology::special_weights) gives,
    /// with the replaced pairs set to 0.
    pub fn special_weights(&self, topo: &molrs::core::Topology) -> SpecialWeights {
        let mut lists = topo.special_weights(&self.by_distance);
        for &(i, j) in &self.replaced {
            for (a, b) in [(i, j), (j, i)] {
                if let Some(list) = lists.get_mut(a) {
                    match list.binary_search_by_key(&b, |&(p, _)| p) {
                        Ok(k) => list[k].1 = 0.0,
                        Err(k) => list.insert(k, (b, 0.0)),
                    }
                }
            }
        }
        SpecialWeights::new(&lists)
    }
}

/// The weights a force field puts on close non-bonded neighbours.
///
/// A bonded pair's non-bonded term is not wanted at full strength: 1-2 and 1-3
/// are normally excluded outright and 1-4 scaled, because the bonded terms
/// already describe those interactions. A compiled intramolecular list carries
/// that by *omitting* the excluded rows and baking the 1-4 factor into the
/// parameters — which works only while that exact list is the one being
/// evaluated. A neighbour table has no such memory: it finds every pair inside
/// the cutoff, bonded or not.
///
/// So the weights have to be applied at evaluation time, and this holds them.
/// Build it from [`PairWeights::special_weights`], or from
/// [`Topology::special_weights`](molrs::core::Topology::special_weights), which
/// walks the bond graph.
///
/// # Why it splits the table rather than scaling in the kernel
///
/// Energy is a sum over pairs, so scaling a group of pairs and scaling their
/// contribution are the same number — which means the weights can be applied
/// *outside* the kernels, and no kernel has to learn that a force field has
/// exclusions.
///
/// The obvious cheaper trick — evaluate everything, then subtract what should
/// not have been counted — is not available. A 1-2 pair sits at bond length,
/// where a Lennard-Jones term is enormous; subtracting it from a total of
/// ordinary size cancels away the very digits the answer is made of.
#[derive(Clone, Debug)]
pub struct SpecialWeights {
    /// Per owned atom, its special partners sorted by index, with weights.
    per_atom: Vec<Vec<(u32, F)>>,
    /// Whether every list is empty. A fact about the table, not about a step.
    nothing_scaled: bool,
}

/// Nothing scaled — the right default, and not the one `derive` would give.
///
/// `nothing_scaled` is a cached answer, and a derived `bool` default is
/// `false`: an empty table would have claimed to scale something. It cost only
/// a slower path, because an empty lookup answers 1.0 for every pair — but a
/// cached fact that disagrees with the thing it caches is a trap whoever
/// trusts it next will fall into.
impl Default for SpecialWeights {
    fn default() -> Self {
        Self {
            per_atom: Vec::new(),
            nothing_scaled: true,
        }
    }
}

impl SpecialWeights {
    /// Take the per-atom lists a bond-graph walk produced.
    pub fn new(special: &[Vec<(usize, F)>]) -> Self {
        let per_atom: Vec<Vec<(u32, F)>> = special
            .iter()
            .map(|l| l.iter().map(|&(p, w)| (p as u32, w)).collect())
            .collect();
        Self {
            nothing_scaled: per_atom.iter().all(|l| l.is_empty()),
            per_atom,
        }
    }

    /// The weight on the pair of atoms `(i, j)`. `1.0` when the two
    /// are far enough apart in the bond graph to interact normally.
    pub fn weight(&self, i: usize, j: usize) -> F {
        let Some(list) = self.per_atom.get(i) else {
            return 1.0;
        };
        match list.binary_search_by_key(&(j as u32), |&(p, _)| p) {
            Ok(k) => list[k].1,
            Err(_) => 1.0,
        }
    }

    /// True when nothing is scaled, so a caller can skip the weights entirely.
    ///
    /// Answered from a flag set at construction: it is a property of the table
    /// and was being recomputed by scanning every atom, once per member, once
    /// per step.
    pub fn is_empty(&self) -> bool {
        self.nothing_scaled
    }

    /// Fill `out` with one weight per row of `pairs`.
    ///
    /// # Why a column and not a split
    ///
    /// Partitioning the table into a full-strength one and a group per
    /// distinct weight is also correct, because energy is a sum over pairs and
    /// scaling a group is the same as scaling its contribution. It costs the
    /// table being rebuilt — allocated, re-pushed column by column — once per
    /// weight per member per step. Measured at 4 096 atoms that is four times
    /// the kernel it prepares input for, and thirty megabytes a step.
    ///
    /// A weight is one number per pair. Handing the kernel that number is one
    /// pass over a buffer the caller keeps.
    /// The per-pair weights for `pairs`, or an empty slice when nothing is
    /// scaled.
    ///
    /// The empty slice is not a table of ones: a kernel reads it as "no weights
    /// apply" and skips the multiply entirely, which is the common case and the
    /// one worth not paying for. `scratch` is the caller's buffer, reused
    /// across steps — [`fill_factors`](Self::fill_factors) is what fills it.
    pub fn factors_for<'a>(&self, pairs: &Neighbors, scratch: &'a mut Vec<F>) -> &'a [F] {
        if self.is_empty() {
            return &[];
        }
        self.fill_factors(pairs, scratch);
        scratch
    }

    pub fn fill_factors(&self, pairs: &Neighbors, out: &mut Vec<F>) {
        let i_col = pairs.query_point_indices();
        let j_col = pairs.point_indices();
        out.clear();
        out.reserve(i_col.len());
        for p in 0..i_col.len() {
            let (i, j) = (i_col[p] as usize, j_col[p] as usize);
            // An atom and its own periodic image are a real interaction, and
            // no bond-graph weight describes it: the walk is root-inclusive,
            // so `weight(i, i)` would answer 0 — the weight of an atom with
            // itself, a different question from the one this pair asks.
            out.push(if i == j { 1.0 } else { self.weight(i, j) });
        }
    }
}

/// Aggregates multiple potentials; energy/forces are summed.
///
/// Every member evaluates in one shared unit system. Unit conversion is the
/// caller's job (`UnitPreset`), never this type's.
pub struct Potentials {
    /// Each member with the part it plays, settled when it was built.
    ///
    /// The role is settled at construction, so it is kept as the member's
    /// [`ForceTerm`] variant rather than recovered per step from `terms()`.
    inner: Vec<ForceTerm>,
    /// Number of atoms the kernels were compiled against (`coords.len() / 3`).
    /// `0` when unknown (e.g. built incrementally via [`Potentials::push`]).
    n_atoms: usize,
}

impl std::fmt::Debug for Potentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Potentials")
            .field("len", &self.inner.len())
            .finish()
    }
}

impl Potentials {
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
            n_atoms: 0,
        }
    }

    /// Add a member. Which part it plays is [`ForceTerm`]'s to say, and its
    /// constructor already said it.
    pub fn push(&mut self, member: ForceTerm) {
        self.inner.push(member);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether every pair member is bound to its pair list, so the
    /// collection can be priced from coordinates alone
    /// ([`Potential::evaluate`]). A pair kernel built over a type table —
    /// `PairLjCut`, a `compile_typed` kernel — reads a neighbour table each
    /// step and adds nothing to a coordinates-only evaluation; a door that
    /// evaluates from coordinates (a minimizer, `Potentials::evaluate`)
    /// asks here first.
    ///
    /// # Errors
    ///
    /// [`CompileError::Invalid`] naming how many members read a neighbour
    /// table, and where they are priced instead.
    pub fn require_pair_lists(&self) -> Result<(), CompileError> {
        let unbound = self
            .inner
            .iter()
            .filter(|m| matches!(m, ForceTerm::Pair(p) if p.reads_a_neighbor_table()))
            .count();
        if unbound == 0 {
            return Ok(());
        }
        Err(CompileError::Invalid(format!(
            "{unbound} pair member(s) read a neighbour table and add nothing to an evaluation \
             from coordinates alone: price them through an MD evaluator with a NeighborList \
             or VerletSkin, or compile the force field over the frame's pair list"
        )))
    }

    /// Number of atoms the kernels were compiled against, i.e. the expected
    /// `coords.len() / 3`. Returns `0` if unknown (built via [`push`]).
    ///
    /// [`push`]: Potentials::push
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Record the compiled atom count (used by [`PotentialCompiler::compile`](crate::ff::compile::PotentialCompiler::compile)).
    pub fn set_n_atoms(&mut self, n_atoms: usize) {
        self.n_atoms = n_atoms;
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// The members, in the order the force field's styles produced them.
    ///
    /// A caller that needs to treat members differently — a periodic régime
    /// rebinding the bonded terms onto copies while the nonbonded one reads a
    /// neighbour table — needs them one at a time, because the index table a
    /// member wants is the member's own. Summing them all is what
    /// [`Potential::evaluate`] on the aggregate is for.
    pub fn members(&self) -> &[ForceTerm] {
        &self.inner
    }

    /// Give up the members, for a caller that wants to own them individually.
    pub fn into_members(self) -> Vec<ForceTerm> {
        self.inner
    }
}

impl Default for Potentials {
    fn default() -> Self {
        Self::new()
    }
}

/// Make the aggregate usable wherever a single [`Potential`] is expected (the
/// geometry optimizer in [`crate::optimize`], an MD integrator, another
/// [`Potentials`]): every member adds into the one accumulator it is handed —
/// one buffer for the whole aggregate, no allocation per member.
impl Potential for Potentials {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        for member in &self.inner {
            member.evaluate(coords, acc);
        }
    }
}

/// An aggregate is pair-driven when it is asked to be: it forwards to the
/// members that read a pair table and evaluates the rest the ordinary way.
///
/// The split is the member's own [`ForceTerm`] variant, which its
/// constructor chose.
impl PairDriven for Potentials {
    /// Every member adds into the same accumulator.
    ///
    /// A per-pair weight belongs to a member that reads the pair table. A
    /// bonded member ignores the table, and its own interaction is the thing
    /// the weights exist to avoid double-counting — scaling it would be
    /// scaling the wrong side of that.
    ///
    /// The virial needs no bookkeeping here: a pair member tallies its own,
    /// and every other member adds its forces through
    /// [`Accumulator::forces_without_virial`], which makes the aggregate's
    /// virial `None` — the honest answer when one member cannot report.
    fn evaluate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        acc: &mut Accumulator<'_>,
    ) {
        for member in &self.inner {
            match member {
                ForceTerm::Pair(p) => p.evaluate_pairs(coords, pairs, factor, acc),
                other => other.evaluate(coords, acc),
            }
        }
    }

    /// True if *any* member is. One compiled kernel is enough to make the
    /// aggregate's answer independent of the table it is handed.
    fn binds_a_fixed_pair_list(&self) -> bool {
        self.inner.iter().any(ForceTerm::binds_a_fixed_pair_list)
    }

    fn reads_a_neighbor_table(&self) -> bool {
        self.inner
            .iter()
            .any(|m| matches!(m, ForceTerm::Pair(p) if p.reads_a_neighbor_table()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::compile::PotentialCompiler;
    use crate::ff::potential::fixtures::OneShot;

    /// A pair's class is its bond distance: a chain that states only its
    /// bonds, or bonds plus a partial `angles` / `dihedrals` block, gives the
    /// same list as one stating every angle and dihedral — the classes the
    /// neighbour-driven door weights by.
    #[test]
    fn pair_classes_follow_bond_distance_not_the_listed_terms() {
        use molrs::core::keys::{ATOMK, ATOML};
        use molrs::core::schema::block_names::{ANGLES, BONDS, DIHEDRALS};
        let block = |cols: &[(&str, Vec<Idx>)]| {
            let mut b = Block::new();
            for (k, v) in cols {
                b.insert(*k, Array1::from_vec(v.clone()).into_dyn(), None)
                    .unwrap();
            }
            b
        };
        // Chain 0-1-2-3-4.
        let chain = |angles: bool, dihedrals: bool| {
            let mut f = Frame::new();
            f.insert(ATOMS, block(&[("id", vec![0, 1, 2, 3, 4])]));
            f.insert(
                BONDS,
                block(&[(ATOMI, vec![0, 1, 2, 3]), (ATOMJ, vec![1, 2, 3, 4])]),
            );
            if angles {
                f.insert(
                    ANGLES,
                    block(&[(ATOMI, vec![0]), (ATOMJ, vec![1]), (ATOMK, vec![2])]),
                );
            }
            if dihedrals {
                f.insert(
                    DIHEDRALS,
                    block(&[
                        (ATOMI, vec![0]),
                        (ATOMJ, vec![1]),
                        (ATOMK, vec![2]),
                        (ATOML, vec![3]),
                    ]),
                );
            }
            f
        };
        let listed = |f: &Frame| {
            let p = intramolecular_pairs(f, &SpecialBonds::EXCLUDE_12_13, None).unwrap();
            let i = p.get(ATOMI).unwrap().get::<u64>().unwrap().to_owned();
            let j = p.get(ATOMJ).unwrap().get::<u64>().unwrap().to_owned();
            let f14 = p.get(IS_14).unwrap().get::<bool>().unwrap().to_owned();
            (i, j, f14)
        };
        let expected = listed(&chain(false, false));
        // 0-3, 0-4, 1-4: the 1-4 pairs are 0-3 and 1-4.
        assert_eq!(expected.0.as_slice().unwrap(), &[0, 0, 1]);
        assert_eq!(expected.1.as_slice().unwrap(), &[3, 4, 4]);
        assert_eq!(expected.2.as_slice().unwrap(), &[true, false, true]);
        assert_eq!(listed(&chain(true, true)), expected);
        assert_eq!(listed(&chain(true, false)), expected);
    }

    /// The default scales nothing, and says so.
    ///
    /// `nothing_scaled` caches an answer, and a cached fact that disagrees with
    /// the thing it caches is worse than no cache: a derived `bool` default is
    /// `false`, so an empty table claimed to have weights and a provider with
    /// no neighbour list refused to build. The Python suite found it; this
    /// keeps it found.
    #[test]
    fn an_empty_weight_table_scales_nothing() {
        assert!(SpecialWeights::default().is_empty());
        assert!(SpecialWeights::new(&[]).is_empty());
        assert!(SpecialWeights::new(&[vec![], vec![]]).is_empty());
        assert!(!SpecialWeights::new(&[vec![(1_usize, 0.5_f64)], vec![]]).is_empty());
        // And an absent entry still answers full strength.
        assert_eq!(SpecialWeights::default().weight(0, 1), 1.0);
    }

    use crate::ff::forcefield::ForceField;
    use crate::ff::ir::Params;
    use molrs::core::Block;
    use molrs::op::Idx;
    use ndarray::Array1;

    struct DummyPotential {
        value: F,
    }

    impl Potential for DummyPotential {
        fn evaluate(&self, _coords: &[F], acc: &mut Accumulator<'_>) {
            acc.add_energy(self.value);
            for f in acc.forces_without_virial() {
                *f += self.value;
            }
        }
    }

    fn make_atoms_only_frame() -> Frame {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![0.0 as F, 2.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "y",
                Array1::from_vec(vec![0.0 as F, 0.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "z",
                Array1::from_vec(vec![0.0 as F, 0.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("atoms", atoms);
        frame
    }

    /// A four-atom frame carrying one term of every bonded arity, so a single
    /// force field produces a bond, an angle, a dihedral and an improper
    /// kernel and the rebinding contract can be checked on all four at once.
    fn make_all_bonded_frame() -> Frame {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        // A non-planar, non-collinear arrangement: a dihedral and an improper
        // are both undefined on degenerate geometry, and a test that fed them
        // one would be asserting against NaN.
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![0.0 as F, 1.5, 2.1, 3.4]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "y",
                Array1::from_vec(vec![0.0 as F, 0.2, 1.4, 1.1]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "z",
                Array1::from_vec(vec![0.0 as F, 0.9, 0.3, 1.8]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(); 4]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("atoms", atoms);

        let idx = |v: Vec<Idx>| Array1::from_vec(v).into_dyn();
        let ty = |n: usize, name: &str| Array1::from_vec(vec![name.to_string(); n]).into_dyn();

        let mut bonds = Block::new();
        bonds.insert("atomi", idx(vec![0, 1, 2]), None).unwrap();
        bonds.insert("atomj", idx(vec![1, 2, 3]), None).unwrap();
        bonds.insert("type", ty(3, "A-A"), None).unwrap();
        frame.insert("bonds", bonds);

        let mut angles = Block::new();
        angles.insert("atomi", idx(vec![0, 1]), None).unwrap();
        angles.insert("atomj", idx(vec![1, 2]), None).unwrap();
        angles.insert("atomk", idx(vec![2, 3]), None).unwrap();
        angles.insert("type", ty(2, "A-A-A"), None).unwrap();
        frame.insert("angles", angles);

        let mut dihedrals = Block::new();
        dihedrals.insert("atomi", idx(vec![0]), None).unwrap();
        dihedrals.insert("atomj", idx(vec![1]), None).unwrap();
        dihedrals.insert("atomk", idx(vec![2]), None).unwrap();
        dihedrals.insert("atoml", idx(vec![3]), None).unwrap();
        dihedrals.insert("type", ty(1, "A-A-A-A"), None).unwrap();
        frame.insert("dihedrals", dihedrals);

        let mut impropers = Block::new();
        impropers.insert("atomi", idx(vec![1]), None).unwrap();
        impropers.insert("atomj", idx(vec![0]), None).unwrap();
        impropers.insert("atomk", idx(vec![2]), None).unwrap();
        impropers.insert("atoml", idx(vec![3]), None).unwrap();
        impropers.insert("type", ty(1, "A-A-A-A"), None).unwrap();
        frame.insert("impropers", impropers);

        frame
    }

    /// Handing a kernel back the very indices it resolved at construction must
    /// change nothing — bit for bit.
    ///
    /// This is the contract that makes a periodic régime possible: if the two
    /// entry points can disagree on identical input, they can disagree on
    /// remapped input too, and the difference would be indistinguishable from
    /// physics. Bit equality rather than a tolerance because it is meant to be
    /// the same arithmetic in the same order — a kernel that had grown a
    /// second copy of its force expression would show up here as a few ulp.
    #[test]
    fn rebinding_a_kernel_to_its_own_indices_changes_nothing() {
        let mut ff = ForceField::new("all-bonded");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-A",
                &["A", "A"],
                Params::from_iter([("k", 300.0), ("r0", 1.5)]),
            )
            .unwrap();
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-A-A",
                &["A", "A", "A"],
                Params::from_iter([("k", 50.0), ("theta0", 109.5)]),
            )
            .unwrap();
        ff.def_style("dihedral", "opls", Params::new())
            .unwrap()
            .def_type(
                "A-A-A-A",
                &["A", "A", "A", "A"],
                Params::from_iter([("k1", 1.3), ("k2", -0.05), ("k3", 0.24), ("k4", 0.0)]),
            )
            .unwrap();
        ff.def_style("improper", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-A-A-A",
                &["A", "A", "A", "A"],
                Params::from_iter([("k", 10.0), ("chi0", 0.0)]),
            )
            .unwrap();

        let frame = make_all_bonded_frame();
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();

        let mut checked = 0;
        for (m, member) in pots.members().iter().enumerate() {
            let Some(terms) = member.terms() else {
                continue;
            };
            let ForceTerm::Indexed(pot) = member else {
                unreachable!("only an indexed member answers with a table")
            };
            let (e0, f0) = pot.energy_forces_at(&coords);
            let (e1, f1) = pot.over_terms(&coords, terms.view());
            assert_eq!(
                e0.to_bits(),
                e1.to_bits(),
                "member {m}: energy {e0} vs {e1} on identical indices"
            );
            assert_eq!(f0.len(), f1.len(), "member {m}: force length");
            for (c, (a, b)) in f0.iter().zip(&f1).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "member {m}: force component {c}: {a} vs {b}"
                );
            }
            assert!(
                e0.is_finite() && e0 != 0.0,
                "member {m} contributed nothing, so it asserted nothing"
            );
            checked += 1;
        }
        assert_eq!(
            checked, 4,
            "expected a bond, an angle, a dihedral and an improper to answer `terms()`"
        );
    }

    fn make_bond_frame() -> Frame {
        let mut frame = make_atoms_only_frame();
        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn(), None)
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn(), None)
            .unwrap();
        bonds
            .insert(
                "type",
                Array1::from_vec(vec!["A-A".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("bonds", bonds);
        frame
    }

    fn make_lj_frame() -> Frame {
        // Two atoms 2 Å apart, each of per-atom `type` "A" (the kernel reads
        // per-atom LJ params keyed by it and Lorentz-Berthelot-combines).
        let mut frame = make_atoms_only_frame();
        frame
            .get_mut("atoms")
            .unwrap()
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(), "A".to_string()]).into_dyn(),
                None,
            )
            .unwrap();

        // Build the neighbour list the real way — `intramolecular_pairs` owns
        // the atomi/atomj/is_14 convention; with no bonds it yields the single
        // unexcluded (0,1) pair, is_14 = false.
        let pairs = intramolecular_pairs(&frame, &SpecialBonds::EXCLUDE_12_13, None).unwrap();
        frame.insert("pairs", pairs);

        frame
    }

    #[test]
    fn test_potentials_collection() {
        let mut pots = Potentials::new();
        pots.push(ForceTerm::plain(DummyPotential { value: 1.0 }));
        pots.push(ForceTerm::plain(DummyPotential { value: 2.0 }));

        assert_eq!(pots.len(), 2);

        let coords: Vec<F> = vec![0.0; 6];
        assert!((pots.energy_at(&coords) - 3.0).abs() < 1e-5);

        let (_, forces) = pots.energy_forces_at(&coords);
        for f in &forces {
            assert!((*f - 3.0).abs() < 1e-5);
        }
    }

    /// Prices each row of the table it is handed at 1, and nothing without one.
    struct PairCounting;

    impl Potential for PairCounting {
        fn evaluate(&self, _coords: &[F], _acc: &mut Accumulator<'_>) {}
    }

    impl PairDriven for PairCounting {
        fn evaluate_pairs(
            &self,
            _coords: &[F],
            pairs: &Neighbors,
            _factor: &[F],
            acc: &mut Accumulator<'_>,
        ) {
            // No force, so no virial to lose.
            acc.add_energy(pairs.query_point_indices().len() as F);
        }

        fn binds_a_fixed_pair_list(&self) -> bool {
            false
        }
    }

    #[test]
    fn one_pair_table_is_shared_with_every_member() {
        let pairs = Neighbors::from_pairs(
            [
                molrs::core::NeighborPair {
                    i: 0,
                    j: 1,
                    dist_sq: 1.0,
                    disp: [1.0, 0.0, 0.0],
                },
                molrs::core::NeighborPair {
                    i: 0,
                    j: 2,
                    dist_sq: 4.0,
                    disp: [2.0, 0.0, 0.0],
                },
            ],
            molrs::core::NeighborColumns::FULL,
            molrs::core::QueryMode::SelfQuery { n_points: 3 },
        );
        let mut pots = Potentials::new();
        pots.push(ForceTerm::pair(PairCounting));
        pots.push(ForceTerm::pair(PairCounting));
        let coords: Vec<F> = vec![0.0; 9];
        let (e, _, w) = pots.over_pairs_virial(&coords, &pairs);
        assert!((e - 4.0).abs() < 1e-12);
        assert_eq!(
            w,
            Some(Virial::ZERO),
            "every member tallied, so the sum is reported"
        );

        // A member that adds forces without a virial makes the aggregate's
        // `None` — not a total missing one term.
        pots.push(ForceTerm::plain(DummyPotential { value: 1.0 }));
        let (e, _, w) = pots.over_pairs_virial(&coords, &pairs);
        assert!((e - 5.0).abs() < 1e-12);
        assert_eq!(w, None);
    }

    /// The accumulator's channels: a borrowed buffer keeps what it held, an
    /// unrequested virial is `None`, and a requested one survives only while
    /// every contribution came with one.
    #[test]
    fn an_accumulator_reports_only_the_channels_it_was_given() {
        let mut buf = vec![1.0; 6];
        let mut acc = Accumulator::adding_into(&mut buf);
        DummyPotential { value: 2.0 }.evaluate(&[0.0; 6], &mut acc);
        assert_eq!(acc.energy(), 2.0);
        assert_eq!(acc.virial(), None, "not requested");
        assert_eq!(acc.into_forces(), vec![3.0; 6]);
        assert_eq!(buf, vec![3.0; 6], "added into, not overwritten");

        let mut acc = Accumulator::new(6).with_virial();
        assert_eq!(
            acc.virial(),
            Some(Virial::ZERO),
            "requested and nothing added"
        );
        if let (_, Some(w)) = acc.forces_with_virial() {
            w.add_outer([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        }
        assert_eq!(acc.virial().map(|w| w.components[0]), Some(2.0));
        acc.forces_without_virial();
        assert_eq!(acc.virial(), None, "a contribution without one loses it");
        assert!(
            acc.forces_with_virial().1.is_none(),
            "and nobody tallies into a lost one"
        );
    }

    #[test]
    fn unknown_style_kernel_is_error() {
        // A style whose (category, name) has no kernel -> Err from PotentialCompiler::compile.
        let mut ff = ForceField::new("test");
        ff.def_style("bond", "nonexistent", Params::new())
            .unwrap()
            .def_type("A-A", &["A", "A"], Params::from_iter([("k", 1.0)]))
            .unwrap();
        let frame = make_bond_frame();
        let err = PotentialCompiler::new(&ff).compile(&frame).unwrap_err();
        assert!(err.to_string().contains("no kernel"), "{err}");
    }

    #[test]
    fn registering_a_style_extends_dispatch() {
        // A custom (category, name) with no built-in kernel becomes usable by
        // registering its constructor — no edit to PotentialCompiler required.
        fn my_constructor(
            _sp: &Params,
            _tp: &[(&str, &Params)],
            _f: &Frame,
        ) -> Result<ForceTerm, crate::ff::potential::CompileError> {
            Ok(ForceTerm::plain(DummyPotential { value: 42.0 }))
        }
        crate::ff::style_registry::register_style(
            crate::ff::ir::StyleSpec::new("pair", "test/custom"),
            Some(crate::ff::style_registry::Kernel::constructor(
                my_constructor,
            )),
        )
        .unwrap();

        let mut ff = ForceField::new("test");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "test/custom", Params::new())
            .unwrap()
            .def_type("A", &["A"], Params::new())
            .unwrap();
        let frame = make_lj_frame();
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();
        // the custom kernel ran: DummyPotential yields its constant value.
        assert!((pots.energy_at(&coords) - 42.0).abs() < 1e-9);
    }

    #[test]
    fn atom_style_is_skipped() {
        // Atom styles carry types/charges, not a pairwise kernel -> skipped.
        let mut ff = ForceField::new("test");
        ff.def_style("atom", "full", Params::new()).unwrap();
        let frame = make_atoms_only_frame();
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        assert_eq!(pots.len(), 0);
    }

    #[test]
    fn test_compile_requires_types() {
        let mut ff = ForceField::new("test");
        ff.def_style("bond", "harmonic", Params::new()).unwrap();
        let frame = make_bond_frame();
        let err = PotentialCompiler::new(&ff)
            .compile(&frame)
            .expect_err("expected compile to fail");
        assert!(err.to_string().contains("has no type definitions"));
    }

    #[test]
    fn test_compile_energy() {
        let mut ff = ForceField::new("test");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_iter([("epsilon", 1.0), ("sigma", 1.0)]),
            )
            .unwrap();

        let frame = make_lj_frame();
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();

        let (energy, _) = pots.energy_forces_at(&coords);
        let expected: F = 4.0 * (1.0 / 4096.0 - 1.0 / 64.0);
        assert!((energy - expected).abs() < 1e-5);
    }

    #[test]
    fn a_pair_type_is_found_through_its_endpoints_not_its_name() {
        let mut ff = ForceField::new("test");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap()
            .def_type(
                "anything",
                &["A"],
                Params::from_iter([("epsilon", 1.0), ("sigma", 1.0)]),
            )
            .unwrap();

        let frame = make_lj_frame();
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();

        let (energy, _) = pots.energy_forces_at(&coords);
        let expected: F = 4.0 * (1.0 / 4096.0 - 1.0 / 64.0);
        assert!((energy - expected).abs() < 1e-5);
    }

    #[test]
    fn test_compile_forces() {
        let mut ff = ForceField::new("test");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_iter([("epsilon", 1.0), ("sigma", 1.0)]),
            )
            .unwrap();

        let frame = make_lj_frame();
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();

        let (_, forces) = pots.energy_forces_at(&coords);

        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-5);
        }
    }

    #[test]
    fn lj_cut_combines_distinct_types_lorentz_berthelot() {
        // Two atoms 2.5 Å apart of distinct types A and B; the kernel must
        // Lorentz-Berthelot-combine their per-atom params.
        let mut frame = make_atoms_only_frame();
        frame
            .get_mut("atoms")
            .unwrap()
            .insert(
                "x",
                Array1::from_vec(vec![0.0 as F, 2.5 as F]).into_dyn(),
                None,
            )
            .unwrap();
        frame
            .get_mut("atoms")
            .unwrap()
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(), "B".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert(
            "pairs",
            intramolecular_pairs(&frame, &SpecialBonds::EXCLUDE_12_13, None).unwrap(),
        );

        let mut ff = ForceField::new("test");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "lj/cut", Params::new())
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_iter([("epsilon", 1.0), ("sigma", 1.0)]),
            )
            .unwrap()
            .def_type(
                "B",
                &["B"],
                Params::from_iter([("epsilon", 4.0), ("sigma", 3.0)]),
            )
            .unwrap();

        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();
        let (energy, _) = pots.energy_forces_at(&coords);

        // ε = √(1·4) = 2, σ = (1+3)/2 = 2, at r = 2.5.
        let (eps, sigma, r) = (2.0_f64, 2.0_f64, 2.5_f64);
        let sr6 = (sigma / r).powi(6);
        let expected = 4.0 * eps * (sr6 * sr6 - sr6);
        assert!(
            (energy - expected).abs() < 1e-9,
            "energy={energy} expected={expected}"
        );
    }

    #[test]
    fn lj_cut_applies_special_bonds_14_scaling() {
        // A 4-atom chain 0-1-2-3 (bonds 0-1-2-3, angles, one dihedral) so the
        // neighbour list excludes 1-2/1-3 and flags only the (0,3) pair is_14.
        // With special_bonds lj 1-4 = 0.5, the (0,3) LJ energy is halved.
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![0.0 as F, 1.0, 2.0, 3.0]).into_dyn(),
                None,
            )
            .unwrap();
        for col in ["y", "z"] {
            atoms
                .insert(col, Array1::from_vec(vec![0.0 as F; 4]).into_dyn(), None)
                .unwrap();
        }
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(); 4]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("atoms", atoms);

        let mut bonds = Block::new();
        bonds
            .insert(
                "atomi",
                Array1::from_vec(vec![0 as Idx, 1, 2]).into_dyn(),
                None,
            )
            .unwrap();
        bonds
            .insert(
                "atomj",
                Array1::from_vec(vec![1 as Idx, 2, 3]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("bonds", bonds);

        let mut angles = Block::new();
        angles
            .insert(
                "atomi",
                Array1::from_vec(vec![0 as Idx, 1]).into_dyn(),
                None,
            )
            .unwrap();
        angles
            .insert(
                "atomk",
                Array1::from_vec(vec![2 as Idx, 3]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("angles", angles);

        let mut dihedrals = Block::new();
        dihedrals
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn(), None)
            .unwrap();
        dihedrals
            .insert("atoml", Array1::from_vec(vec![3 as Idx]).into_dyn(), None)
            .unwrap();
        frame.insert("dihedrals", dihedrals);

        // The real neighbour list: only (0,3), flagged is_14.
        let pairs = intramolecular_pairs(&frame, &SpecialBonds::EXCLUDE_12_13, None).unwrap();
        assert_eq!(
            pairs.n_rows(),
            Some(1),
            "expected exactly the (0,3) 1-4 pair"
        );
        frame.insert("pairs", pairs);

        let mut ff = ForceField::new("test");
        ff.def_style("pair", "lj/cut", Params::new())
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_iter([("epsilon", 1.0), ("sigma", 1.0)]),
            )
            .unwrap();
        let mut sb = crate::ff::ir::SpecialBonds::EXCLUDE_12_13;
        sb.lj[2] = 0.5;
        ff.set_special_bonds(sb);

        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();
        let (energy, _) = pots.energy_forces_at(&coords);

        // (0,3) at r = 3, ε = σ = 1, scaled by the 0.5 1-4 weight.
        let sr6 = (1.0_f64 / 3.0).powi(6);
        let expected = 0.5 * 4.0 * (sr6 * sr6 - sr6);
        assert!(
            (energy - expected).abs() < 1e-12,
            "energy={energy} expected={expected}"
        );
    }

    #[test]
    fn test_compile_empty_ff() {
        let ff = ForceField::new("test");
        let frame = make_lj_frame();
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();

        let (energy, forces) = pots.energy_forces_at(&coords);
        assert!(energy.abs() < 1e-5);
        assert_eq!(forces.len(), 6);
        assert!(forces.iter().all(|x| x.abs() < 1e-5));
    }

    #[test]
    fn test_compile_skips_absent_topology() {
        // A style whose topology block is absent from the frame contributes
        // nothing (no rows of that kind) — skipped, not an error.
        let mut ff = ForceField::new("test");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_iter([("epsilon", 1.0), ("sigma", 1.0)]),
            )
            .unwrap();

        let frame = make_atoms_only_frame();
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        assert_eq!(pots.len(), 0);
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();
        assert!(pots.energy_at(&coords).abs() < 1e-9);
    }

    /// A frame too large for a full pair list is refused, not enumerated.
    ///
    /// `N(N-1)/2` with no cutoff is the shape of this list, and the failure
    /// mode past a certain `N` is the OOM killer, which tells the caller
    /// nothing about what to do instead. The refusal names the alternative.
    /// An overlay on a bond-graph 1-4 is carried onto that pair. A `pairs`
    /// block already present wins, and the overlay is not read.
    #[test]
    fn pair_overrides_are_carried_onto_the_bond_graph_list() {
        let chain = || {
            let mut frame = Frame::new();
            let mut atoms = Block::new();
            atoms
                .insert("x", Array1::from_vec(vec![0.0_f64; 4]).into_dyn(), None)
                .unwrap();
            frame.insert(ATOMS, atoms);
            let mut bonds = Block::new();
            bonds
                .insert(ATOMI, Array1::from_vec(vec![0_u64, 1, 2]).into_dyn(), None)
                .unwrap();
            bonds
                .insert(ATOMJ, Array1::from_vec(vec![1_u64, 2, 3]).into_dyn(), None)
                .unwrap();
            frame.insert("bonds", bonds);
            frame
        };
        let overlay = |scale: f64| {
            let mut block = Block::new();
            block
                .insert(ATOMI, Array1::from_vec(vec![0_u64]).into_dyn(), None)
                .unwrap();
            block
                .insert(ATOMJ, Array1::from_vec(vec![3_u64]).into_dyn(), None)
                .unwrap();
            block
                .insert("lj_scale", Array1::from_vec(vec![scale]).into_dyn(), None)
                .unwrap();
            block
        };
        let scale_of = |pairs: &Block| -> f64 {
            let i = pairs.get(ATOMI).unwrap().get::<u64>().unwrap();
            let j = pairs.get(ATOMJ).unwrap().get::<u64>().unwrap();
            let row = (0..i.len()).find(|&r| i[[r]] == 0 && j[[r]] == 3).unwrap();
            pairs.get("lj_scale").unwrap().get::<f64>().unwrap()[[row]]
        };

        let mut frame = chain();
        frame.insert(PAIR_OVERRIDES, overlay(0.5));
        let built = intramolecular_pairs(&frame, &SpecialBonds::EXCLUDE_12_13, None).unwrap();
        assert!(
            built
                .get(IS_14)
                .unwrap()
                .get::<bool>()
                .unwrap()
                .iter()
                .all(|&f| f)
        );
        assert_eq!(scale_of(&built), 0.5);

        frame.insert(PAIRS, overlay(0.2));
        let built = intramolecular_pairs(&frame, &SpecialBonds::EXCLUDE_12_13, None).unwrap();
        assert_eq!(scale_of(&built), 0.2);
    }

    #[test]
    fn a_repeated_or_non_finite_overlay_row_is_refused() {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![0.0_f64; 4]).into_dyn(), None)
            .unwrap();
        frame.insert(ATOMS, atoms);
        let mut bonds = Block::new();
        bonds
            .insert(ATOMI, Array1::from_vec(vec![0_u64, 1, 2]).into_dyn(), None)
            .unwrap();
        bonds
            .insert(ATOMJ, Array1::from_vec(vec![1_u64, 2, 3]).into_dyn(), None)
            .unwrap();
        frame.insert("bonds", bonds);

        let mut twice = Block::new();
        twice
            .insert(ATOMI, Array1::from_vec(vec![0_u64, 0]).into_dyn(), None)
            .unwrap();
        twice
            .insert(ATOMJ, Array1::from_vec(vec![3_u64, 3]).into_dyn(), None)
            .unwrap();
        twice
            .insert(
                "lj_scale",
                Array1::from_vec(vec![0.5_f64, 0.25]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert(PAIR_OVERRIDES, twice);
        let err = intramolecular_pairs(&frame, &SpecialBonds::EXCLUDE_12_13, None).unwrap_err();
        assert!(err.contains("two rows"), "{err}");

        let mut nan = Block::new();
        nan.insert(ATOMI, Array1::from_vec(vec![0_u64]).into_dyn(), None)
            .unwrap();
        nan.insert(ATOMJ, Array1::from_vec(vec![3_u64]).into_dyn(), None)
            .unwrap();
        nan.insert(
            "lj_scale",
            Array1::from_vec(vec![f64::NAN]).into_dyn(),
            None,
        )
        .unwrap();
        frame.insert(PAIR_OVERRIDES, nan);
        let err = intramolecular_pairs(&frame, &SpecialBonds::EXCLUDE_12_13, None).unwrap_err();
        assert!(err.contains("non-finite"), "{err}");
    }

    /// A list that omits one intramolecular pair and flags another `is_14`
    /// the bond graph would not keeps both facts. A second molecule's
    /// neighbor is kept only when the list is open.
    #[test]
    fn install_pairs_keeps_an_authoritative_list_and_opens_a_capped_one() {
        use molrs::core::{NeighborColumns, NeighborPair, QueryMode};

        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![0.0_f64; 5]).into_dyn(), None)
            .unwrap();
        // 0-1-2-3 one molecule, atom 4 another.
        atoms
            .insert(
                MOL_ID,
                Array1::from_vec(vec![0_u64, 0, 0, 0, 1]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert(ATOMS, atoms);
        let mut bonds = Block::new();
        bonds
            .insert(ATOMI, Array1::from_vec(vec![0_u64, 1, 2]).into_dyn(), None)
            .unwrap();
        bonds
            .insert(ATOMJ, Array1::from_vec(vec![1_u64, 2, 3]).into_dyn(), None)
            .unwrap();
        frame.insert("bonds", bonds);

        // (0, 2) is bond distance 2, which the bond graph would exclude, and
        // the list flags it 1-4. (0, 3) is the bond-graph 1-4 and is omitted.
        let mut listed = Block::new();
        listed
            .insert(ATOMI, Array1::from_vec(vec![0_u64]).into_dyn(), None)
            .unwrap();
        listed
            .insert(ATOMJ, Array1::from_vec(vec![2_u64]).into_dyn(), None)
            .unwrap();
        listed
            .insert(IS_14, Array1::from_vec(vec![true]).into_dyn(), None)
            .unwrap();
        listed
            .insert(
                "lj_scale",
                Array1::from_vec(vec![0.25_f64]).into_dyn(),
                None,
            )
            .unwrap();
        let neighbors = Neighbors::from_pairs(
            [(0_u32, 2_u32), (0, 3), (0, 4)]
                .into_iter()
                .map(|(i, j)| NeighborPair {
                    i,
                    j,
                    dist_sq: 1.0,
                    disp: [1.0, 0.0, 0.0],
                }),
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points: 5 },
        );
        let rows = |block: &Block| -> (Vec<u64>, Vec<u64>, Vec<bool>) {
            (
                block
                    .get(ATOMI)
                    .unwrap()
                    .get::<u64>()
                    .unwrap()
                    .iter()
                    .copied()
                    .collect(),
                block
                    .get(ATOMJ)
                    .unwrap()
                    .get::<u64>()
                    .unwrap()
                    .iter()
                    .copied()
                    .collect(),
                block
                    .get(IS_14)
                    .unwrap()
                    .get::<bool>()
                    .unwrap()
                    .iter()
                    .copied()
                    .collect(),
            )
        };

        // Open: two molecules and no cross-molecule row. (0, 4) comes from
        // the neighbor table; (0, 3) stays omitted.
        frame.insert(PAIRS, listed.clone());
        let open = install_pairs(&frame, &SpecialBonds::EXCLUDE_12_13, Some(&neighbors)).unwrap();
        assert_eq!(rows(&open), (vec![0, 0], vec![2, 4], vec![true, false]));
        let scale = open.get("lj_scale").unwrap().get::<f64>().unwrap();
        let valid = open.validity("lj_scale").unwrap();
        assert_eq!(scale[[0]], 0.25);
        assert!(valid[0] && !valid[1]);

        // Closed by a cross-molecule row, (1, 4), that the neighbor table
        // does not contain. (0, 4) is then omitted: a closed list adds no
        // intermolecular pair it does not already hold. (0, 3) stays omitted.
        let mut closed_list = Block::new();
        closed_list
            .insert(ATOMI, Array1::from_vec(vec![0_u64, 1]).into_dyn(), None)
            .unwrap();
        closed_list
            .insert(ATOMJ, Array1::from_vec(vec![2_u64, 4]).into_dyn(), None)
            .unwrap();
        closed_list
            .insert(IS_14, Array1::from_vec(vec![true, false]).into_dyn(), None)
            .unwrap();
        closed_list
            .insert(
                "lj_scale",
                Array1::from_vec(vec![0.25_f64, 0.0]).into_dyn(),
                Some(vec![true, false]),
            )
            .unwrap();
        frame.insert(PAIRS, closed_list);
        let closed = install_pairs(&frame, &SpecialBonds::EXCLUDE_12_13, Some(&neighbors)).unwrap();
        assert_eq!(rows(&closed), (vec![0], vec![2], vec![true]));
        assert_eq!(
            closed.get("lj_scale").unwrap().get::<f64>().unwrap()[[0]],
            0.25
        );
    }

    #[test]
    fn a_frame_too_large_for_a_full_pair_list_is_refused() {
        use molrs::core::Block;
        use ndarray::Array1;

        let n = MAX_ATOMS_FOR_A_FULL_PAIR_LIST + 1;
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from(vec!["a".to_string(); n]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("atoms", atoms);

        let err = intramolecular_pairs(&frame, &SpecialBonds::EXCLUDE_12_13, None)
            .expect_err("a list this size is not an answer");
        assert!(err.contains("neighbour list"), "{err}");
        assert!(err.contains("compile_typed"), "{err}");

        // And one atom under the ceiling still builds, so the bound is a
        // ceiling and not an off-by-one that refuses the supported case.
        let mut small = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from(vec!["a".to_string(); MAX_ATOMS_FOR_A_FULL_PAIR_LIST]).into_dyn(),
                None,
            )
            .unwrap();
        small.insert("atoms", atoms);
        assert!(intramolecular_pairs(&small, &SpecialBonds::EXCLUDE_12_13, None).is_ok());
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::{Accumulator, IndexedTerms, PairDriven, Potential};
    use molrs::core::{Neighbors, Virial};
    use molrs::op::F;
    use ndarray::ArrayView2;

    /// One evaluation in fresh storage, for a test that wants the totals:
    /// a new [`Accumulator`], one call of the primitive, its channels read
    /// back. Test-only — the API is the primitive and the accumulator.
    pub(crate) trait OneShot: Potential {
        /// [`Potential::evaluate`]: energy and forces.
        fn energy_forces_at(&self, coords: &[F]) -> (F, Vec<F>) {
            let mut acc = Accumulator::new(coords.len());
            self.evaluate(coords, &mut acc);
            (acc.energy(), acc.into_forces())
        }

        /// [`Potential::evaluate`]: energy.
        fn energy_at(&self, coords: &[F]) -> F {
            self.energy_forces_at(coords).0
        }

        /// [`PairDriven::evaluate_pairs`], unweighted: energy and forces.
        fn over_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>)
        where
            Self: PairDriven,
        {
            let (e, f, _) = self.over_pairs_virial(coords, pairs);
            (e, f)
        }

        /// [`PairDriven::evaluate_pairs`], unweighted, with the virial
        /// requested.
        fn over_pairs_virial(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>, Option<Virial>)
        where
            Self: PairDriven,
        {
            let mut acc = Accumulator::new(coords.len()).with_virial();
            self.evaluate_pairs(coords, pairs, &[], &mut acc);
            let (e, w) = (acc.energy(), acc.virial());
            (e, acc.into_forces(), w)
        }

        /// [`IndexedTerms::evaluate_terms`]: energy and forces.
        fn over_terms(&self, coords: &[F], terms: ArrayView2<'_, u32>) -> (F, Vec<F>)
        where
            Self: IndexedTerms,
        {
            let mut acc = Accumulator::new(coords.len());
            self.evaluate_terms(coords, terms, &mut acc);
            (acc.energy(), acc.into_forces())
        }
    }

    impl<P: Potential + ?Sized> OneShot for P {}

    /// Central-difference check that every force component is `-dE/dx`.
    pub(crate) fn assert_forces_are_negative_gradient(pot: &dyn Potential, coords: &[F], tol: F) {
        let h: F = 1e-6;
        let (_, f) = pot.energy_forces_at(coords);
        let mut worst: F = 0.0;
        for i in 0..coords.len() {
            let mut plus = coords.to_vec();
            let mut minus = coords.to_vec();
            plus[i] += h;
            minus[i] -= h;
            let numeric = -(pot.energy_at(&plus) - pot.energy_at(&minus)) / (2.0 * h);
            worst = worst.max((f[i] - numeric).abs());
        }
        assert!(worst < tol, "max |F + dE/dx| = {worst:.3e}");
    }
}
