//! A frame's special pairs (its `pairs` block) as a force field reads them:
//! the non-bonded exceptions.
//!
//! A pair no row names is priced at full strength. A row names a pair the
//! pair styles price otherwise: by its class (`n_bonds` 1, 2 or 3, weighted
//! by the force field's `special_bonds`) or by its own override cells. The
//! rows are topology, generated with the angles and dihedrals
//! ([`Topology::pairs_block`](molrs::core::Topology::pairs_block)); the
//! compiler, the engine writers and MD read them through this.

use std::collections::HashSet;

use crate::ff::ir::{SpecialBonds, SpecialClass};
use crate::ff::potential::SpecialWeights;
use molrs::core::Frame;
use molrs::core::keys::{ATOMI, ATOMJ, N_BONDS};
use molrs::core::schema::PAIR_OVERRIDE_COLUMNS;
use molrs::core::schema::block_names::{ATOMS, BONDS, PAIRS};
use molrs::op::F;

/// One row's override cells, in [`PAIR_OVERRIDE_COLUMNS`] order.
pub(crate) type Cells = [Option<F>; 5];
pub(crate) const EPS: usize = 0;
pub(crate) const SIGMA: usize = 1;
pub(crate) const QQ: usize = 2;
pub(crate) const LJ_SCALE: usize = 3;
pub(crate) const COUL_SCALE: usize = 4;

/// One `pairs` row.
#[derive(Clone, Debug)]
pub(crate) struct SpecialPair {
    /// The lower atom row.
    pub(crate) i: usize,
    /// The higher atom row.
    pub(crate) j: usize,
    /// Bonds along the shortest path between the two atoms.
    pub(crate) n_bonds: usize,
    pub(crate) cells: Cells,
}

impl SpecialPair {
    /// Whether the row states one of the pair's parameters (`epsilon`,
    /// `sigma`, `charge_product`), which no pair style's kernel reads.
    pub(crate) fn has_parameters(&self) -> bool {
        self.cells[EPS].is_some() || self.cells[SIGMA].is_some() || self.cells[QQ].is_some()
    }

    /// The weight of the pair in the styles of `class`: the row's own scale,
    /// else `special_bonds` for its bond distance, else full strength.
    pub(crate) fn weight(&self, class: SpecialClass, special: &SpecialBonds) -> F {
        let (cell, table) = match class {
            SpecialClass::Vdw => (self.cells[LJ_SCALE], special.lj),
            SpecialClass::Coulomb => (self.cells[COUL_SCALE], special.coul),
        };
        cell.unwrap_or(match self.n_bonds {
            d @ 1..=3 => table[d - 1],
            _ => 1.0,
        })
    }
}

/// The rows of a frame's `pairs` block, sorted by `(i, j)`.
#[derive(Clone, Debug, Default)]
pub(crate) struct SpecialPairs {
    rows: Vec<SpecialPair>,
}

impl SpecialPairs {
    /// The frame's special pairs; `None` when it has no `pairs` block.
    ///
    /// # Errors
    ///
    /// A missing or non-unsigned `atomi` / `atomj` / `n_bonds` column, an end
    /// outside the atoms, a pair of an atom with itself, two rows of one pair,
    /// and an override column that is not float or holds a non-finite value.
    pub(crate) fn of(frame: &Frame) -> Result<Option<Self>, String> {
        let Some(block) = frame.get(PAIRS) else {
            return Ok(None);
        };
        let n = block.n_rows().unwrap_or(0);
        if n == 0 {
            return Ok(Some(Self::default()));
        }
        let index = |key: &str| -> Result<Vec<usize>, String> {
            let col = block
                .get(key)
                .ok_or_else(|| format!("pairs: missing the \"{key}\" column"))?;
            let values = col
                .get::<u64>()
                .ok_or_else(|| format!("pairs: \"{key}\" must be an unsigned column"))?;
            Ok(values.iter().map(|&v| v as usize).collect())
        };
        let (ai, aj, hops) = (index(ATOMI)?, index(ATOMJ)?, index(N_BONDS)?);
        let n_atoms = frame.get(ATOMS).and_then(|b| b.n_rows()).unwrap_or(0);
        let mut cells = vec![[None; 5]; n];
        for (c, key) in PAIR_OVERRIDE_COLUMNS.iter().enumerate() {
            let Some(col) = block.get(key) else {
                continue;
            };
            let values = col
                .get::<f64>()
                .ok_or_else(|| format!("pairs: the override column \"{key}\" must be float"))?;
            let valid = block.validity(key);
            for (r, row) in cells.iter_mut().enumerate() {
                if valid.is_some_and(|v| !v[r]) {
                    continue;
                }
                let v = values[r];
                if !v.is_finite() {
                    return Err(format!("pairs: row {r} has a non-finite \"{key}\" ({v})"));
                }
                row[c] = Some(v);
            }
        }
        let mut rows = Vec::with_capacity(n);
        for r in 0..n {
            let (a, b) = (ai[r], aj[r]);
            if a >= n_atoms || b >= n_atoms {
                return Err(format!(
                    "pairs: row {r} names atoms {a} and {b}, and the frame has {n_atoms}"
                ));
            }
            if a == b {
                return Err(format!("pairs: row {r} pairs atom {a} with itself"));
            }
            rows.push(SpecialPair {
                i: a.min(b),
                j: a.max(b),
                n_bonds: hops[r],
                cells: cells[r],
            });
        }
        rows.sort_by_key(|p| (p.i, p.j));
        if let Some(w) = rows
            .windows(2)
            .find(|w| (w[0].i, w[0].j) == (w[1].i, w[1].j))
        {
            return Err(format!(
                "pairs: atoms {} and {} are on two rows; a pair has one",
                w[0].i, w[0].j
            ));
        }
        Ok(Some(Self { rows }))
    }

    pub(crate) fn rows(&self) -> &[SpecialPair] {
        &self.rows
    }

    /// Refuse a frame whose `bonds` name a pair no row does: the pairs were
    /// never generated, or a bond was added after they were.
    pub(crate) fn require_bonds_covered(&self, frame: &Frame) -> Result<(), String> {
        let Some(bonds) = frame.get(BONDS) else {
            return Ok(());
        };
        if bonds.n_rows().unwrap_or(0) == 0 {
            return Ok(());
        }
        let column = |key: &str| {
            bonds
                .get(key)
                .and_then(|c| c.get::<u64>())
                .ok_or_else(|| format!("bonds: missing an unsigned \"{key}\" column"))
        };
        let (bi, bj) = (column(ATOMI)?, column(ATOMJ)?);
        let listed: HashSet<(usize, usize)> = self.rows.iter().map(|p| (p.i, p.j)).collect();
        for (&a, &b) in bi.iter().zip(bj.iter()) {
            let (a, b) = (a as usize, b as usize);
            if !listed.contains(&(a.min(b), a.max(b))) {
                return Err(format!(
                    "pairs: the bond between atoms {a} and {b} has no pairs row, so its \
                     non-bonded term would be priced at full strength. Generate the pairs \
                     with the angles and dihedrals (Atomistic::generate_topology)"
                ));
            }
        }
        Ok(())
    }

    /// The weights of the styles of `class`: per atom, its partners whose
    /// weight is not 1. A pair in `replaced` is priced elsewhere (the
    /// exceptions kernel) and weighs 0 here.
    pub(crate) fn weights(
        &self,
        class: SpecialClass,
        special: &SpecialBonds,
        replaced: &HashSet<(usize, usize)>,
        n_atoms: usize,
    ) -> SpecialWeights {
        let mut lists: Vec<Vec<(usize, F)>> = vec![Vec::new(); n_atoms];
        let mut put = |i: usize, j: usize, w: F| {
            lists[i].push((j, w));
            lists[j].push((i, w));
        };
        let mut seen = HashSet::with_capacity(self.rows.len());
        for p in &self.rows {
            let w = if replaced.contains(&(p.i, p.j)) {
                0.0
            } else {
                p.weight(class, special)
            };
            seen.insert((p.i, p.j));
            if w != 1.0 {
                put(p.i, p.j, w);
            }
        }
        for &(i, j) in replaced {
            if !seen.contains(&(i, j)) {
                put(i, j, 0.0);
            }
        }
        for list in &mut lists {
            list.sort_unstable_by_key(|&(p, _)| p);
        }
        SpecialWeights::new(&lists)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Block;
    use molrs::op::Idx;
    use ndarray::Array1;

    const AMBER: SpecialBonds = SpecialBonds {
        lj: [0.0, 0.0, 0.5],
        coul: [0.0, 0.0, 1.0 / 1.2],
    };

    fn ids(values: &[Idx]) -> ndarray::ArrayD<Idx> {
        Array1::from_vec(values.to_vec()).into_dyn()
    }

    /// `n_atoms` atoms, `bonds`, and a `pairs` block of `(i, j, n_bonds)` with
    /// `coul_scale` cells (`None`: null).
    fn frame(
        n_atoms: usize,
        bonds: &[(Idx, Idx)],
        rows: &[(Idx, Idx, Idx)],
        coul: &[Option<F>],
    ) -> Frame {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![0.0; n_atoms]).into_dyn(), None)
            .unwrap();
        frame.insert(ATOMS, atoms);
        if !bonds.is_empty() {
            let mut b = Block::new();
            let (i, j): (Vec<Idx>, Vec<Idx>) = bonds.iter().copied().unzip();
            b.insert(ATOMI, ids(&i), None).unwrap();
            b.insert(ATOMJ, ids(&j), None).unwrap();
            frame.insert(BONDS, b);
        }
        let mut p = Block::new();
        p.insert(
            ATOMI,
            ids(&rows.iter().map(|r| r.0).collect::<Vec<_>>()),
            None,
        )
        .unwrap();
        p.insert(
            ATOMJ,
            ids(&rows.iter().map(|r| r.1).collect::<Vec<_>>()),
            None,
        )
        .unwrap();
        p.insert(
            N_BONDS,
            ids(&rows.iter().map(|r| r.2).collect::<Vec<_>>()),
            None,
        )
        .unwrap();
        if !coul.is_empty() {
            let values: Vec<F> = coul.iter().map(|c| c.unwrap_or(0.0)).collect();
            let valid: Vec<bool> = coul.iter().map(Option::is_some).collect();
            p.insert(
                "coul_scale",
                Array1::from_vec(values).into_dyn(),
                Some(valid),
            )
            .unwrap();
        }
        frame.insert(PAIRS, p);
        frame
    }

    fn read(frame: &Frame) -> Result<SpecialPairs, String> {
        SpecialPairs::of(frame).map(|p| p.expect("the frame has a pairs block"))
    }

    /// A row's own scale beats its class's weight; a class outside 1-3 (an
    /// exclusion a format names at its own distance) weighs 1 without one.
    #[test]
    fn a_row_weighs_its_cell_else_its_class_else_one() {
        let row = |n_bonds, coul| SpecialPair {
            i: 0,
            j: 1,
            n_bonds,
            cells: [None, None, None, None, coul],
        };
        let vdw = |p: &SpecialPair| p.weight(SpecialClass::Vdw, &AMBER);
        let coulomb = |p: &SpecialPair| p.weight(SpecialClass::Coulomb, &AMBER);
        assert_eq!(vdw(&row(3, None)), 0.5);
        assert_eq!(coulomb(&row(3, None)), 1.0 / 1.2);
        assert_eq!(coulomb(&row(3, Some(0.25))), 0.25);
        assert_eq!(vdw(&row(3, Some(0.25))), 0.5);
        assert_eq!(coulomb(&row(1, None)), 0.0);
        assert_eq!(coulomb(&row(0, None)), 1.0);
        assert_eq!(coulomb(&row(4, Some(0.0))), 0.0);
    }

    /// Rows come back ordered by pair, `i < j` whichever end a row wrote
    /// first, with their null cells null.
    #[test]
    fn rows_are_read_in_pair_order_with_their_cells() {
        let f = frame(4, &[], &[(3, 1, 2), (0, 2, 4)], &[None, Some(0.0)]);
        let pairs = read(&f).unwrap();
        let got: Vec<(usize, usize, usize, Option<F>)> = pairs
            .rows()
            .iter()
            .map(|p| (p.i, p.j, p.n_bonds, p.cells[COUL_SCALE]))
            .collect();
        assert_eq!(got, [(0, 2, 4, Some(0.0)), (1, 3, 2, None)]);
        assert!(SpecialPairs::of(&Frame::new()).unwrap().is_none());
    }

    #[test]
    fn a_malformed_block_is_refused_by_name() {
        let refused = |f: &Frame, needle: &str| {
            let err = read(f).unwrap_err();
            assert!(err.contains(needle), "{needle}: {err}");
        };
        let mut no_class = frame(3, &[], &[(0, 1, 1)], &[]);
        no_class.get_mut(PAIRS).unwrap().remove(N_BONDS);
        refused(&no_class, "\"n_bonds\"");
        refused(&frame(3, &[], &[(0, 3, 1)], &[]), "the frame has 3");
        refused(&frame(3, &[], &[(1, 1, 0)], &[]), "with itself");
        refused(&frame(3, &[], &[(0, 1, 1), (1, 0, 1)], &[]), "two rows");
        refused(&frame(3, &[], &[(0, 1, 1)], &[Some(F::NAN)]), "non-finite");
    }

    /// A bond whose pair has no row is refused (the special pairs were not
    /// generated); one with a row passes.
    #[test]
    fn a_bond_without_a_row_is_refused() {
        let covered = frame(
            3,
            &[(0, 1), (2, 1)],
            &[(0, 1, 1), (1, 2, 1), (0, 2, 2)],
            &[],
        );
        read(&covered)
            .unwrap()
            .require_bonds_covered(&covered)
            .unwrap();
        let open = frame(3, &[(0, 1), (2, 1)], &[(0, 1, 1)], &[]);
        let err = read(&open)
            .unwrap()
            .require_bonds_covered(&open)
            .unwrap_err();
        assert!(err.contains("atoms 2 and 1"), "{err}");
        assert!(err.contains("generate_topology"), "{err}");
    }

    /// The weights list every pair whose weight is not 1, both ways; a pair
    /// the exceptions kernel prices weighs 0, listed or not.
    #[test]
    fn weights_list_what_is_not_one_and_zero_what_is_replaced() {
        let f = frame(
            5,
            &[],
            &[(0, 1, 1), (0, 3, 3), (1, 2, 3), (2, 4, 0)],
            &[None, None, Some(1.0), None],
        );
        let pairs = read(&f).unwrap();
        let replaced = HashSet::from([(0, 3), (3, 4)]);
        let w = pairs.weights(SpecialClass::Coulomb, &AMBER, &replaced, 5);
        let mut got: Vec<(usize, usize, F)> = w.pairs().collect();
        got.sort_by_key(|&(i, j, _)| (i, j));
        assert_eq!(got, [(0, 1, 0.0), (0, 3, 0.0), (3, 4, 0.0)]);
        assert_eq!(w.weight(2, 1), 1.0);
        assert_eq!(w.weight(1, 0), 0.0);
    }
}
