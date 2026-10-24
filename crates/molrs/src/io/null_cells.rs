//! Null cells on the way out: a writer never writes one.
//!
//! A column may hold null cells (its validity mask, [`Block::validity`]):
//! [`Frame::concat`] leaves every row of a part that lacks the column null,
//! as a nullable column read from a record does. The cell's filler (a `0`,
//! an empty string) is no value, so a writer that wrote it would write a
//! number or a blank field nobody stated — LAMMPS reads atom ID `0` as an
//! error, `grompp` reads a blank atom name as the next field.
//!
//! What each writer does with a null cell, by column:
//!
//! - **A label the format can generate** (an atom's `name`, `res_name`,
//!   `res_id`; a PDB occupancy or B-factor): the cell is unset, and the
//!   writer generates the label as it does for a frame without the column
//!   ([`AtomLabels`](crate::io::pdb::labels::AtomLabels): the element
//!   numbered within its residue, `MOL`, the molecule's residue number).
//! - **Anything else it writes** (an `id`, a `type`, a charge, a mass, a
//!   coordinate, a topology row's endpoints): refused by
//!   [`refuse_null_cells`], naming the column and the rows. An `id` is not
//!   generated beside the stated ones: an ID identifies its atom to other
//!   files, and a generated one could repeat a stated one. A frame without
//!   the column is numbered 1..N by the writers that do so.
//!
//! [`Block::validity`]: crate::core::Block::validity
//! [`Frame::concat`]: crate::core::Frame::concat

use crate::io::invalid_data;
use molrs::core::FrameAccess;

/// How many runs of null rows an error lists before it elides the rest.
const LISTED_RUNS: usize = 5;

/// The rows of `block.key` that hold no value, ascending; empty when the
/// column has none, or the block or column is absent.
pub(crate) fn null_rows(frame: &impl FrameAccess, block: &str, key: &str) -> Vec<usize> {
    frame
        .visit_block(block, |b| {
            b.validity(key).map(|mask| {
                mask.iter()
                    .enumerate()
                    .filter(|(_, valid)| !**valid)
                    .map(|(row, _)| row)
                    .collect()
            })
        })
        .flatten()
        .unwrap_or_default()
}

/// `Err` naming the first of `keys` in `block` that holds a null cell, and
/// its rows; `Ok` when none does. For the columns a writer writes and
/// cannot generate (module docs).
///
/// # Errors
///
/// `InvalidData`: `block.key is null in 15 of 40 rows (rows 25-39) …`.
pub(crate) fn refuse_null_cells(
    frame: &impl FrameAccess,
    block: &str,
    keys: &[&str],
) -> std::io::Result<()> {
    for key in keys {
        let rows = null_rows(frame, block, key);
        if rows.is_empty() {
            continue;
        }
        let n = frame
            .visit_block(block, |b| b.n_rows().unwrap_or(0))
            .unwrap_or(0);
        return Err(invalid_data(format!(
            "{block}.{key} is null in {} of {n} rows ({}, 0-based): a writer writes no \
             placeholder (a 0, an empty field) for a missing value. Frame.concat leaves the \
             rows of a part that lacks the column null; give those rows a value first",
            rows.len(),
            runs(&rows)
        )));
    }
    Ok(())
}

/// [`refuse_null_cells`] over every column of `block`: for a writer that
/// writes the whole block (XYZ, the LAMMPS dump and molecule writers). An
/// absent block has no columns and passes.
pub(crate) fn refuse_null_cells_in_block(
    frame: &impl FrameAccess,
    block: &str,
) -> std::io::Result<()> {
    let keys: Vec<String> = frame
        .visit_block(block, |b| {
            b.column_keys().into_iter().map(str::to_owned).collect()
        })
        .unwrap_or_default();
    let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
    refuse_null_cells(frame, block, &keys)
}

/// `rows` (ascending) as runs: `rows 3, 7-9, 12`, eliding after
/// [`LISTED_RUNS`].
fn runs(rows: &[usize]) -> String {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for &row in rows {
        match spans.last_mut() {
            Some((_, end)) if *end + 1 == row => *end = row,
            _ => spans.push((row, row)),
        }
    }
    let mut text: Vec<String> = spans
        .iter()
        .take(LISTED_RUNS)
        .map(|&(a, b)| {
            if a == b {
                a.to_string()
            } else {
                format!("{a}-{b}")
            }
        })
        .collect();
    if spans.len() > LISTED_RUNS {
        text.push("…".to_owned());
    }
    let noun = if rows.len() == 1 { "row" } else { "rows" };
    format!("{noun} {}", text.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::{Block, Frame};
    use ndarray::Array1;

    fn frame_with_nulls(mask: Vec<bool>) -> Frame {
        let n = mask.len();
        let mut atoms = Block::new();
        atoms
            .insert(
                "charge",
                Array1::from_vec(vec![0.0; n]).into_dyn(),
                Some(mask),
            )
            .unwrap();
        atoms
            .insert("x", Array1::from_vec(vec![0.0; n]).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame
    }

    /// Every frame writer refuses a null cell in a column it writes; none
    /// writes the filler.
    #[test]
    fn every_frame_writer_refuses_a_null_cell_it_would_write() {
        use crate::io::{
            write_cif, write_cube, write_dcd, write_lammps_dump, write_lammps_dump_local,
            write_lammps_molecule, write_mol2, write_poscar, write_trr, write_xsf, write_xtc,
            write_xyz,
        };
        use molrs::core::{Cell, SimBox};

        let mut atoms = Block::new();
        let f = |v: Vec<f64>| Array1::from_vec(v).into_dyn();
        let s = |v: &[&str]| Array1::from_vec(v.iter().map(|x| x.to_string()).collect()).into_dyn();
        atoms
            .insert("id", Array1::from_vec(vec![1u64, 2]).into_dyn(), None)
            .unwrap();
        atoms
            .insert("x", f(vec![0.0, 1.5]), Some(vec![true, false]))
            .unwrap();
        atoms.insert("y", f(vec![0.0, 0.0]), None).unwrap();
        atoms.insert("z", f(vec![0.0, 0.0]), None).unwrap();
        atoms.insert("charge", f(vec![0.0, 0.0]), None).unwrap();
        atoms.insert("element", s(&["C", "C"]), None).unwrap();
        atoms.insert("name", s(&["C1", "C2"]), None).unwrap();
        atoms.insert("type", s(&["C", "C"]), None).unwrap();
        atoms
            .insert(
                "atomic_number",
                Array1::from_vec(vec![6u64, 6]).into_dyn(),
                None,
            )
            .unwrap();
        let mut bonds = Block::new();
        bonds
            .insert(
                "atomi",
                Array1::from_vec(vec![0u64]).into_dyn(),
                Some(vec![false]),
            )
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1u64]).into_dyn(), None)
            .unwrap();
        let mut grid = Block::new();
        grid.insert(
            "value",
            ndarray::ArrayD::<f64>::zeros(ndarray::IxDyn(&[2, 2, 2])),
            None,
        )
        .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);
        frame.insert("grid", grid);
        let h = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        frame.simbox = Some(SimBox::new(Cell::Matrix(h), [0.0; 3], [true; 3]).unwrap());

        let mut buf = Vec::new();
        let doors: Vec<(&str, std::io::Error)> = vec![
            ("xyz", write_xyz(&mut buf, &frame).unwrap_err()),
            ("mol2", write_mol2(&mut buf, &frame).unwrap_err()),
            ("cif", write_cif(&mut buf, &frame).unwrap_err()),
            ("dcd", write_dcd(&mut buf, &frame).unwrap_err()),
            ("trr", write_trr(&mut buf, &frame).unwrap_err()),
            ("xtc", write_xtc(&mut buf, &frame).unwrap_err()),
            ("poscar", write_poscar(&mut buf, &frame).unwrap_err()),
            ("xsf", write_xsf(&mut buf, &frame).unwrap_err()),
            ("cube", write_cube(&mut buf, &frame).unwrap_err()),
            (
                "dump",
                write_lammps_dump(&mut buf, &frame, None).unwrap_err(),
            ),
            (
                "molecule",
                write_lammps_molecule(&mut buf, &frame).unwrap_err(),
            ),
        ];
        for (door, err) in doors {
            assert!(err.to_string().contains("atoms.x is null"), "{door}: {err}");
        }
        let err = write_lammps_dump_local(&mut buf, &[frame]).unwrap_err();
        assert!(
            err.to_string().contains("bonds.atomi is null"),
            "dump local: {err}"
        );
    }

    #[test]
    fn a_null_cell_is_refused_by_column_and_rows() {
        let frame = frame_with_nulls(vec![true, false, false, false, true, false]);
        assert_eq!(null_rows(&frame, "atoms", "charge"), [1, 2, 3, 5]);
        let err = refuse_null_cells(&frame, "atoms", &["x", "charge"]).unwrap_err();
        let message = err.to_string();
        assert!(
            message.starts_with("atoms.charge is null in 4 of 6 rows (rows 1-3, 5, 0-based)"),
            "{message}"
        );
    }

    #[test]
    fn columns_without_nulls_and_absent_ones_pass() {
        let frame = frame_with_nulls(vec![true; 3]);
        refuse_null_cells(&frame, "atoms", &["x", "charge", "id"]).unwrap();
        refuse_null_cells(&frame, "bonds", &["atomi"]).unwrap();
        assert!(null_rows(&frame, "atoms", "id").is_empty());
    }

    #[test]
    fn many_runs_are_elided() {
        let rows: Vec<usize> = (0..20).step_by(2).collect();
        assert_eq!(runs(&rows), "rows 0, 2, 4, 6, 8, …");
        assert_eq!(runs(&[7]), "row 7");
    }

    fn strings(v: &[&str]) -> ndarray::ArrayD<String> {
        Array1::from_vec(v.iter().map(|s| (*s).to_owned()).collect()).into_dyn()
    }

    /// A water stated whole (ids, names, residue) then a methane's C-H
    /// with none of those columns, joined by `Frame::concat`: the
    /// methane's `id`, `name`, `res_name` and `res_id` cells are null.
    fn concat_of_unlike_parts() -> Frame {
        concat_parts(false)
    }

    /// As [`concat_of_unlike_parts`]; with `type_ids`, the water also carries
    /// numeric `type_id` columns (a prmtop's numbering) that the methane lacks.
    fn concat_parts(type_ids: bool) -> Frame {
        let part = |elements: &[&str], types: &[&str], q: &[f64], bonds: &[[u64; 2]]| {
            let n = elements.len();
            let mut atoms = Block::new();
            atoms.insert("element", strings(elements), None).unwrap();
            let masses = elements
                .iter()
                .map(|symbol| {
                    crate::core::Element::by_symbol(symbol)
                        .unwrap()
                        .atomic_mass() as f64
                })
                .collect::<Vec<_>>();
            atoms
                .insert("mass", Array1::from_vec(masses).into_dyn(), None)
                .unwrap();
            atoms.insert("type", strings(types), None).unwrap();
            atoms
                .insert("charge", Array1::from_vec(q.to_vec()).into_dyn(), None)
                .unwrap();
            for (d, key) in ["x", "y", "z"].iter().enumerate() {
                let v: Vec<f64> = (0..n).map(|i| (i + d) as f64).collect();
                atoms
                    .insert(*key, Array1::from_vec(v).into_dyn(), None)
                    .unwrap();
            }
            let mut b = Block::new();
            let col = |k: usize| {
                Array1::from_vec(bonds.iter().map(|r| r[k] as molrs::op::Idx).collect()).into_dyn()
            };
            b.insert("atomi", col(0), None).unwrap();
            b.insert("atomj", col(1), None).unwrap();
            let bond_types: Vec<String> = bonds
                .iter()
                .map(|r| format!("{}-{}", types[r[0] as usize], types[r[1] as usize]))
                .collect();
            let bond_types: Vec<&str> = bond_types.iter().map(String::as_str).collect();
            b.insert("type", strings(&bond_types), None).unwrap();
            let mut frame = Frame::new();
            frame.insert("atoms", atoms);
            frame.insert("bonds", b);
            frame
        };
        let mut water = part(
            &["O", "H", "H"],
            &["OW", "HW", "HW"],
            &[-0.8, 0.4, 0.4],
            &[[0, 1], [0, 2]],
        );
        let atoms = water.get_mut("atoms").unwrap();
        atoms
            .insert(
                "id",
                Array1::from_vec(vec![1 as molrs::op::Idx, 2, 3]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert("name", strings(&["OW", "HW1", "HW2"]), None)
            .unwrap();
        atoms
            .insert("res_name", strings(&["SOL"; 3]), None)
            .unwrap();
        atoms
            .insert(
                "res_id",
                Array1::from_vec(vec![7 as molrs::op::Idx; 3]).into_dyn(),
                None,
            )
            .unwrap();
        if type_ids {
            let ids = |v: Vec<molrs::op::Idx>| Array1::from_vec(v).into_dyn();
            water
                .get_mut("atoms")
                .unwrap()
                .insert("type_id", ids(vec![1, 2, 2]), None)
                .unwrap();
            water
                .get_mut("bonds")
                .unwrap()
                .insert("type_id", ids(vec![1, 1]), None)
                .unwrap();
        }
        let ch = part(&["C", "H"], &["CT", "HC"], &[-0.1, 0.1], &[[0, 1]]);
        Frame::concat([&water, &ch]).unwrap()
    }

    /// A `type_id` on one part only leaves null cells after the concat; the
    /// string `type` wins over it, so the LAMMPS writer numbers the types from
    /// the labels and does not refuse the stale numbering's gaps.
    #[test]
    fn a_partial_type_id_beside_string_types_is_not_refused() {
        let mut frame = concat_parts(true);
        assert_eq!(null_rows(&frame, "bonds", "type_id"), [2]);
        let atoms = frame.get_mut("atoms").unwrap();
        atoms.remove("id").unwrap();
        let mol: Vec<molrs::op::Idx> = vec![1, 1, 1, 2, 2];
        atoms
            .insert("mol_id", Array1::from_vec(mol).into_dyn(), None)
            .unwrap();
        let text =
            crate::io::writer::capture(|sink| crate::io::write_lammps_data(sink, &frame)).unwrap();
        assert!(text.contains("4 atom types"), "{text}");
        assert!(text.contains("2 bond types"), "{text}");
    }

    /// LAMMPS: a null `id` is refused by column and rows, never written as
    /// 0; without the column the atoms are numbered 1..N.
    #[test]
    fn the_lammps_data_writer_refuses_a_null_id() {
        let mut frame = concat_of_unlike_parts();
        let mol: Vec<molrs::op::Idx> = vec![1, 1, 1, 2, 2];
        frame
            .get_mut("atoms")
            .unwrap()
            .insert("mol_id", Array1::from_vec(mol).into_dyn(), None)
            .unwrap();
        let err = crate::io::writer::capture(|sink| crate::io::write_lammps_data(sink, &frame))
            .unwrap_err();
        assert!(
            err.to_string()
                .starts_with("atoms.id is null in 2 of 5 rows (rows 3-4, 0-based)"),
            "{err}"
        );
        frame.get_mut("atoms").unwrap().remove("id").unwrap();
        let text =
            crate::io::writer::capture(|sink| crate::io::write_lammps_data(sink, &frame)).unwrap();
        let ids: Vec<&str> = text
            .lines()
            .skip_while(|l| !l.starts_with("Atoms"))
            .skip(2)
            .take(5)
            .map(|l| l.split_whitespace().next().unwrap())
            .collect();
        assert_eq!(ids, ["1", "2", "3", "4", "5"]);
    }

    /// A null charge is refused by every writer that writes charges.
    #[test]
    fn a_null_charge_is_refused() {
        let mut frame = concat_of_unlike_parts();
        let atoms = frame.get_mut("atoms").unwrap();
        atoms.remove("id").unwrap();
        atoms
            .set_validity("charge", vec![true, true, true, false, true])
            .unwrap();
        atoms
            .insert(
                "mol_id",
                Array1::from_vec(vec![1 as molrs::op::Idx, 1, 1, 2, 2]).into_dyn(),
                None,
            )
            .unwrap();
        let err = crate::io::writer::capture(|sink| crate::io::write_lammps_data(sink, &frame))
            .unwrap_err();
        assert!(
            err.to_string()
                .starts_with("atoms.charge is null in 1 of 5 rows (row 3"),
            "{err}"
        );
    }

    /// PDB and GRO: an unset name, residue name or residue number is
    /// generated as for a frame without the column — the element numbered
    /// within its residue, `MOL`, the molecule's number — never a blank
    /// field; the stated ones are kept.
    #[test]
    fn coordinate_writers_generate_unset_labels() {
        let frame = concat_of_unlike_parts();
        let gro =
            crate::io::writer::capture(|sink| crate::io::write_gro(sink, &frame)).unwrap_err();
        assert!(gro.to_string().starts_with("atoms.id is null"), "{gro}");
        let mut frame = frame;
        frame.get_mut("atoms").unwrap().remove("id").unwrap();
        let gro = crate::io::writer::capture(|sink| crate::io::write_gro(sink, &frame)).unwrap();
        let fields: Vec<(String, String, String)> = gro
            .lines()
            .skip(2)
            .take(5)
            .map(|l| {
                (
                    l[0..5].trim().to_owned(),
                    l[5..10].trim().to_owned(),
                    l[10..15].trim().to_owned(),
                )
            })
            .collect();
        let want = [
            ("7", "SOL", "OW"),
            ("7", "SOL", "HW1"),
            ("7", "SOL", "HW2"),
            ("2", "MOL", "C1"),
            ("2", "MOL", "H1"),
        ];
        assert_eq!(
            fields,
            want.map(|(a, b, c)| (a.to_owned(), b.to_owned(), c.to_owned()))
        );
        let pdb = crate::io::writer::capture(|sink| crate::io::write_pdb(sink, &frame)).unwrap();
        let atoms: Vec<&str> = pdb.lines().filter(|l| l.starts_with("ATOM")).collect();
        assert_eq!(atoms[3][12..16].trim(), "C1");
        assert_eq!(&atoms[3][17..20], "MOL");
        assert_eq!(atoms[3][22..26].trim(), "2");
    }

    /// The `.top` `[ atoms ]` names the atoms as the `.gro` does, so
    /// `grompp` finds them alike; a null type is refused.
    #[cfg(feature = "ff")]
    #[test]
    fn the_top_writer_names_atoms_as_the_gro_writer() {
        use crate::ff::forcefield::ForceField;
        use crate::ff::ir::Params;
        use crate::io::gromacs::GromacsTopForcefieldWriter;

        let mut frame = concat_of_unlike_parts();
        frame.get_mut("atoms").unwrap().remove("id").unwrap();
        let mut angles = Block::new();
        for (key, atom) in [("atomi", 1), ("atomj", 0), ("atomk", 2)] {
            angles
                .insert(
                    key,
                    Array1::from_vec(vec![atom as molrs::op::Idx]).into_dyn(),
                    None,
                )
                .unwrap();
        }
        angles.insert("type", strings(&["HW-OW-HW"]), None).unwrap();
        frame.insert("angles", angles);
        let mut ff = ForceField::new("t");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 0.5],
        });
        const TYPES: [(&str, f64); 4] =
            [("OW", 16.0), ("HW", 1.008), ("CT", 12.011), ("HC", 1.008)];
        let atoms = ff.def_style("atom", "full", Params::new()).unwrap();
        for (t, m) in TYPES {
            atoms
                .def_type(t, &[], Params::from_iter([("mass", m)].iter().copied()))
                .unwrap();
        }
        let lj = ff.def_style("pair", "lj/cut", Params::new()).unwrap();
        for (t, _) in TYPES {
            lj.def_type(
                t,
                &[t],
                Params::from_iter([("epsilon", 0.1), ("sigma", 3.0)].iter().copied()),
            )
            .unwrap();
        }
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "HW-OW-HW",
                &["HW", "OW", "HW"],
                Params::from_iter([("k", 55.0), ("theta0", 104.5)].iter().copied()),
            )
            .unwrap();
        let bond = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        for (name, i, j) in [("OW-HW", "OW", "HW"), ("CT-HC", "CT", "HC")] {
            bond.def_type(
                name,
                &[i, j],
                Params::from_iter([("k", 450.0), ("r0", 1.0)].iter().copied()),
            )
            .unwrap();
        }
        let top = crate::io::writer::capture(|sink| {
            GromacsTopForcefieldWriter::new().write_system(sink, &ff, &frame)
        })
        .unwrap();
        let rows: Vec<Vec<&str>> = top
            .lines()
            .filter(|l| l.split_whitespace().count() == 8 && !l.starts_with(';'))
            .map(|l| l.split_whitespace().collect())
            .collect();
        let labels: Vec<[&str; 3]> = rows.iter().map(|r| [r[2], r[3], r[4]]).collect();
        assert_eq!(
            labels,
            [
                ["7", "SOL", "OW"],
                ["7", "SOL", "HW1"],
                ["7", "SOL", "HW2"],
                ["2", "MOL", "C1"],
                ["2", "MOL", "H1"],
            ]
        );

        let types = frame.get_mut("atoms").unwrap();
        types
            .set_validity("type", vec![true, true, true, false, false])
            .unwrap();
        let err = crate::io::writer::capture(|sink| {
            GromacsTopForcefieldWriter::new().write_system(sink, &ff, &frame)
        })
        .unwrap_err();
        assert!(
            err.starts_with("atoms.type is null in 2 of 5 rows"),
            "{err}"
        );
    }
}
