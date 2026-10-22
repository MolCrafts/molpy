//! The validator — a standalone judge of a Frame against the vocabulary: [`Validator`].

use super::violation::{
    InstancePath, MAX_CELL_VIOLATIONS_PER_COLUMN, SchemaReport, Violation, ViolationKind,
};
use super::{block, column, relation_endpoints};
use crate::core::BlockAccess;
use crate::core::FrameAccess;
use std::collections::HashMap;

/// Judges a frame against the canonical vocabulary.
///
/// A key the vocabulary does not know is unconstrained (and refused only in a
/// closed block); a key it owns has one dtype — `x` is `Float` and no caller
/// makes it otherwise.
///
/// `Validator` deliberately does **not** live on `Frame`. It is its own type,
/// it takes a frame as a parameter, and it depends on nothing above `core`: it
/// knows nothing of `io`, `ff` or `compute`.
///
/// It is generic over [`FrameAccess`] rather than tied to `Frame`, so the same
/// validator judges an owned `Frame` and a borrowed `FrameView` without
/// changes — including the `FrameView` that crosses the CXX bridge.
#[derive(Debug, Clone, Copy, Default)]
pub struct Validator;

impl Validator {
    /// The canonical validator: the committed vocabulary.
    pub fn canonical() -> Self {
        Validator
    }

    /// Every violation in `frame`. Never fails; an empty report means the frame
    /// conforms.
    ///
    /// Collects exhaustively — it does not stop at the first problem, because a
    /// caller fixing a file wants the whole list, not one round trip per bad
    /// column.
    pub fn check<FA: FrameAccess + ?Sized>(&self, frame: &FA) -> SchemaReport {
        let mut report = SchemaReport::new();

        let names: Vec<String> = frame.block_keys().iter().map(|s| s.to_string()).collect();
        let mut nrows: HashMap<String, usize> = HashMap::new();
        for name in &names {
            if let Some(n) = frame
                .visit_block(name, |b: &dyn BlockAccess| b.n_rows())
                .flatten()
            {
                nrows.insert(name.clone(), n);
            }
        }

        for name in &names {
            self.check_columns(frame, name, &mut report);
            self.check_endpoints(frame, name, &nrows, &mut report);
        }

        report.sort();
        report
    }

    /// `Err` iff [`check`](Self::check) is non-empty.
    pub fn validate<FA: FrameAccess + ?Sized>(&self, frame: &FA) -> Result<(), SchemaReport> {
        self.check(frame).into_result()
    }

    fn check_columns<FA: FrameAccess + ?Sized>(
        &self,
        frame: &FA,
        name: &str,
        report: &mut SchemaReport,
    ) {
        let spec = block(name);
        let found = frame.visit_block(name, |b: &dyn BlockAccess| {
            let keys: Vec<String> = b.column_keys().iter().map(|s| s.to_string()).collect();
            let mut out = Vec::with_capacity(keys.len());
            for k in keys {
                let dtype = b.column_dtype(&k);
                let shape = b.column_shape(&k);
                out.push((k, dtype, shape));
            }
            out
        });
        let Some(cols) = found else { return };

        for (col, dtype, shape) in &cols {
            if let (Some(expected), Some(found)) = (column(col).map(|s| s.dtype), *dtype)
                && found != expected
            {
                report.push(Violation::column(
                    name,
                    col,
                    ViolationKind::WrongDtype { expected, found },
                ));
            }
            if let (Some(spec), Some(shape)) = (column(col), shape.as_ref())
                && !spec.shape.admits(shape)
            {
                report.push(Violation::column(
                    name,
                    col,
                    ViolationKind::WrongShape {
                        expected: spec.shape,
                        found: shape.clone(),
                    },
                ));
            }
            if let Some(s) = spec
                && !s.open
                && column(col).is_none()
            {
                report.push(Violation::column(name, col, ViolationKind::UnknownColumn));
            }
        }

        for key in spec.map(|s| s.required).unwrap_or(&[]) {
            if !cols.iter().any(|(c, _, _)| c == key) {
                report.push(Violation::column(name, *key, ViolationKind::MissingColumn));
            }
        }
    }

    fn check_endpoints<FA: FrameAccess + ?Sized>(
        &self,
        frame: &FA,
        name: &str,
        nrows: &HashMap<String, usize>,
        report: &mut SchemaReport,
    ) {
        // A block with no spec is legal (MolGraph mints one per relation
        // kind); `relation_endpoints` infers its endpoints from the columns
        // it carries, and adds every reference the block declares
        // (`targets`), so those get range-checked too.
        let Some((refs, declared, rows)) = frame.visit_block(name, |b: &dyn BlockAccess| {
            let declared: Vec<(&str, &str)> = b.targets();
            let refs = relation_endpoints(name, |k| b.contains_key(k), &declared);
            let declared: Vec<String> = declared.iter().map(|(c, _)| c.to_string()).collect();
            (refs, declared, b.n_rows().unwrap_or(0))
        }) else {
            return;
        };
        for reference in refs {
            // An absolute target (`/frame/atoms`) lies outside this frame; the
            // record that holds both sections checks it.
            if !reference.is_local() {
                continue;
            }
            let Some(&target_rows) = nrows.get(&reference.target) else {
                // The default `atoms` rule is a convention; a declared target
                // must exist wherever the referencing block has rows.
                if rows > 0 && declared.contains(&reference.column) {
                    report.push(Violation::column(
                        name,
                        &reference.column,
                        ViolationKind::MissingTarget {
                            target: reference.target.clone(),
                        },
                    ));
                }
                continue;
            };
            self.check_range(
                frame,
                name,
                &reference.column,
                &reference.target,
                target_rows,
                report,
            );
        }
    }

    fn check_range<FA: FrameAccess + ?Sized>(
        &self,
        frame: &FA,
        name: &str,
        col: &str,
        target: &str,
        target_rows: usize,
        report: &mut SchemaReport,
    ) {
        // A wrong-dtype endpoint column has already been reported by
        // `check_columns`. Reading it as uint here would return None and
        // silently skip the range check — which is exactly the failure this
        // module exists to remove, so the dtype report is what covers it.
        let Some(values) = frame.column(name, col).and_then(|c| c.get::<u64>()) else {
            return;
        };
        // A null row references nothing (a `virtual_sites` row built from
        // fewer atoms leaves its trailing endpoints null).
        let mask: Option<Vec<bool>> = frame
            .visit_block(name, |b: &dyn BlockAccess| {
                b.validity(col).map(<[bool]>::to_vec)
            })
            .flatten();
        let mut reported = 0usize;
        let mut extra = 0usize;
        for (row, &v) in values.iter().enumerate() {
            if (v as usize) < target_rows || mask.as_ref().is_some_and(|m| !m[row]) {
                continue;
            }
            if reported < MAX_CELL_VIOLATIONS_PER_COLUMN {
                report.push(Violation {
                    path: InstancePath::Cell {
                        block: name.to_string(),
                        col: col.to_string(),
                        row,
                    },
                    kind: ViolationKind::IndexOutOfRange {
                        value: v,
                        target: target.to_string(),
                        target_nrows: target_rows,
                    },
                });
                reported += 1;
            } else {
                extra += 1;
            }
        }
        if extra > 0 {
            report.push(Violation::column(
                name,
                col,
                ViolationKind::TruncatedCells { extra },
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Block;
    use crate::core::Frame;
    use crate::op::{F, Idx};
    use ndarray::Array1;

    /// An `atoms` block of `n` rows carrying only `x`.
    fn atoms(n: usize) -> Block {
        let mut b = Block::new();
        b.insert("x", Array1::from_vec(vec![0.0 as F; n]).into_dyn(), None)
            .unwrap();
        b
    }

    fn uint_block(cols: &[(&str, &[Idx])]) -> Block {
        let mut b = Block::new();
        for (key, values) in cols {
            b.insert(*key, Array1::from_vec(values.to_vec()).into_dyn(), None)
                .unwrap();
        }
        b
    }

    fn out_of_range(block: &str, col: &str, value: Idx, target_nrows: usize) -> Violation {
        Violation {
            path: InstancePath::Cell {
                block: block.to_string(),
                col: col.to_string(),
                row: 0,
            },
            kind: ViolationKind::IndexOutOfRange {
                value,
                target: "atoms".to_string(),
                target_nrows,
            },
        }
    }

    #[test]
    fn check_reports_a_canonical_bond_endpoint_past_the_atoms() {
        // 3 atoms are rows 0..=2, so atomj = 3 is one past the end.
        let mut frame = Frame::new();
        frame.insert("atoms", atoms(3));
        frame.insert("bonds", uint_block(&[("atomi", &[0]), ("atomj", &[3])]));

        let report = Validator::canonical().check(&frame);
        let found: Vec<&Violation> = report.iter().collect();
        assert_eq!(found, vec![&out_of_range("bonds", "atomj", 3, 3)]);
    }

    #[test]
    fn check_reports_an_inferred_relation_endpoint_past_the_atoms() {
        // `ports` has no BlockSpec; its endpoint columns are inferred from
        // the keys it carries. atomi = 5 is past rows 0..=1.
        let mut frame = Frame::new();
        frame.insert("atoms", atoms(2));
        frame.insert("ports", uint_block(&[("atomi", &[5]), ("atomj", &[1])]));

        let report = Validator::canonical().check(&frame);
        let found: Vec<&Violation> = report.iter().collect();
        assert_eq!(found, vec![&out_of_range("ports", "atomi", 5, 2)]);
    }

    #[test]
    fn check_reads_members_ibead_into_atoms_and_leaves_an_undeclared_atom_alone() {
        // `members.ibead` references `atoms`; `atom` is an opaque handle until
        // a target is declared for it.
        let mut frame = Frame::new();
        frame.insert("atoms", atoms(2));
        frame.insert("members", uint_block(&[("ibead", &[9]), ("atom", &[10])]));

        let report = Validator::canonical().check(&frame);
        let found: Vec<&Violation> = report.iter().collect();
        assert_eq!(found, vec![&out_of_range("members", "ibead", 9, 2)]);
    }

    #[test]
    fn check_range_checks_a_declared_target_and_reports_a_missing_one() {
        let mut frame = Frame::new();
        frame.insert("atoms", atoms(2));
        let mut sites = uint_block(&[("site", &[1, 3])]);
        sites.set_target("site", "atoms").unwrap();
        frame.insert("refs", sites);
        let report = Validator::canonical().check(&frame);
        assert_eq!(report.iter().count(), 1, "{report}");
        assert!(report.to_string().contains("site"), "{report}");

        let mut frame = Frame::new();
        let mut sites = uint_block(&[("site", &[0])]);
        sites.set_target("site", "sites").unwrap();
        frame.insert("refs", sites);
        let report = Validator::canonical().check(&frame);
        assert!(report.to_string().contains("sites"), "{report}");
    }

    #[test]
    fn check_ranges_every_cmap_endpoint_and_requires_all_five() {
        let mut frame = Frame::new();
        frame.insert("atoms", atoms(3));
        frame.insert(
            "cmaps",
            uint_block(&[
                ("atomi", &[0]),
                ("atomj", &[1]),
                ("atomk", &[2]),
                ("atoml", &[1]),
                ("atomm", &[0]),
            ]),
        );
        let report = Validator::canonical().check(&frame);
        assert!(report.is_empty(), "{report}");

        frame.get_mut("cmaps").unwrap().remove("atomm");
        frame
            .get_mut("cmaps")
            .unwrap()
            .insert("atomm", Array1::from_vec(vec![3 as Idx]).into_dyn(), None)
            .unwrap();
        let report = Validator::canonical().check(&frame);
        assert!(report.to_string().contains("atomm"), "{report}");

        frame.get_mut("cmaps").unwrap().remove("atomm");
        let report = Validator::canonical().check(&frame);
        assert!(report.to_string().contains("atomm"), "{report}");
    }

    #[test]
    fn check_skips_a_null_endpoint() {
        let mut frame = Frame::new();
        frame.insert("atoms", atoms(3));
        let mut sites = uint_block(&[
            ("atomi", &[0]),
            ("atomj", &[1]),
            ("atomk", &[2]),
            ("atoml", &[0]),
        ]);
        // atoml is null on the one row, its filler pointing past the atoms.
        sites
            .insert(
                "atoml",
                Array1::from_vec(vec![99 as Idx]).into_dyn(),
                Some(vec![false]),
            )
            .unwrap();
        frame.insert("virtual_sites", sites);
        let report = Validator::canonical().check(&frame);
        assert!(report.is_empty(), "{report}");
    }
}
