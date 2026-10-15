//! A force field's parameters of a typed frame, written onto the frame as
//! columns: one value per relation row and per atom.
//!
//! A typed frame names each row's type (`bonds.type`, `atoms.type`, …) and the
//! force field holds that type's parameters. [`ForceField::materialize_params`]
//! resolves the one against the other and writes the numbers next to the rows
//! they price — `bonds.<prefix>k`, `bonds.<prefix>r0`, `atoms.<prefix>sigma`, …
//! — so a consumer that works per row (a learned model reading a reference
//! field, an analysis comparing two fields on one molecule) never repeats the
//! type lookup a kernel does.
//!
//! Nothing here knows a style. The columns are the parameters the force field
//! stores for each type, whatever they are called, in the force-field IR's
//! units (LAMMPS standard: degrees, un-halved `K`, the force field's `units`
//! preset). A style the IR gains tomorrow is written by the same code.

use std::collections::{BTreeMap, HashMap};

use ndarray::Array1;

use crate::ff::forcefield::{ForceField, Style};
use crate::ff::ir::{Params, pair_key};
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::schema::block_names::{ANGLES, ATOMS, BONDS, CMAPS, DIHEDRALS, IMPROPERS};
use molrs::op::F;

/// The relation categories and the frame block each one's rows live in.
const RELATIONS: [(&str, &str); 5] = [
    ("bond", BONDS),
    ("angle", ANGLES),
    ("dihedral", DIHEDRALS),
    ("improper", IMPROPERS),
    ("cmap", CMAPS),
];

/// One column being built: a value and a validity flag per row.
struct ParamColumn {
    values: Vec<F>,
    valid: Vec<bool>,
}

impl ParamColumn {
    fn new(rows: usize) -> Self {
        Self {
            values: vec![0.0; rows],
            valid: vec![false; rows],
        }
    }
}

/// The columns of one block, keyed by parameter name, in name order.
type ParamColumnMap = BTreeMap<String, ParamColumn>;

impl ForceField {
    /// Write the parameters this force field gives each row of `frame` as
    /// columns named `<prefix><parameter>`, and return the `(block, column)`
    /// pairs written, in order.
    ///
    /// - **Relations.** Each row of `bonds`, `angles`, `dihedrals`,
    ///   `impropers` and `cmaps` gets the numeric parameters of the type its
    ///   `type` column names, under the one style of its category that
    ///   defines that type: `bonds.<prefix>k`, `bonds.<prefix>r0`;
    ///   `dihedrals.<prefix>k1`, `<prefix>periodicity1`, `<prefix>phase1`, …
    /// - **Atoms.** Each atom gets the numeric parameters of its `atoms.type`
    ///   under every `atom` style (`<prefix>mass`) and, from every `pair`
    ///   style that holds per-type rows, those of its type's self row
    ///   (`<prefix>epsilon`, `<prefix>sigma` under `lj/cut`). A pair style
    ///   with no rows (`coul/cut`) writes nothing: its numbers are not per
    ///   type. A cross row (NBFIX) has no per-atom form and is not written;
    ///   the style's `mixing` rule combines the self rows.
    ///
    /// The values are the stored ones, in the force-field IR's units (LAMMPS
    /// standard: angles in degrees, `K` without a ½, the force field's
    /// `units` preset). A row whose type lacks a parameter another row's type
    /// has — a two-term torsion beside a three-term one, an estimated term's
    /// `estimate_penalty` — gets a null cell (see `Block::validity`). String
    /// parameters (metadata) and array parameters (a CMAP `grid`) have no
    /// per-row scalar and are not written. A column already present under a
    /// written name is replaced. Charges are per-atom frame data (a charge
    /// model writes `atoms.charge`), not a type parameter, unless the field
    /// stores them per type.
    ///
    /// A block absent from the frame, and a category the field has no style
    /// for, are skipped.
    ///
    /// # Errors
    ///
    /// A block with rows but no string `type` column; a row or atom whose type
    /// no style of its category defines, or two styles define; an atom whose
    /// type has no self row in a pair style that has rows; two styles writing
    /// one column of `atoms`.
    pub fn materialize_params(
        &self,
        frame: &mut Frame,
        prefix: &str,
    ) -> Result<Vec<(String, String)>, String> {
        let mut written = Vec::new();
        for (category, block_name) in RELATIONS {
            let styles: Vec<&Style> = self.get_styles(category);
            if styles.is_empty() {
                continue;
            }
            let Some(block) = frame.get(block_name) else {
                continue;
            };
            let columns = relation_columns(category, block_name, &styles, block)?;
            written.extend(write(frame, block_name, prefix, columns)?);
        }
        if let Some(block) = frame.get(ATOMS) {
            let columns = atom_columns(self, block)?;
            written.extend(write(frame, ATOMS, prefix, columns)?);
        }
        Ok(written)
    }
}

/// The rows' `type` labels, or `None` for a block with no rows.
fn type_labels(block: &Block, block_name: &str) -> Result<Option<Vec<String>>, String> {
    let rows = block.n_rows().unwrap_or(0);
    if rows == 0 {
        return Ok(None);
    }
    let labels = block
        .get("type")
        .and_then(|c| c.get::<String>())
        .ok_or_else(|| {
            format!("materialize_params: the {block_name} block has no string \"type\" column")
        })?;
    Ok(Some(labels.iter().cloned().collect()))
}

/// Copy every numeric parameter of `params` into row `row` of `columns`.
fn fill(columns: &mut ParamColumnMap, rows: usize, row: usize, params: &Params) {
    for (key, value) in params
        .iter()
        .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
    {
        let column = columns
            .entry(key.to_owned())
            .or_insert_with(|| ParamColumn::new(rows));
        column.values[row] = value;
        column.valid[row] = true;
    }
}

/// The parameter columns of one relation block: each row under the style of
/// `category` that defines its type.
fn relation_columns(
    category: &str,
    block_name: &str,
    styles: &[&Style],
    block: &Block,
) -> Result<ParamColumnMap, String> {
    let mut columns = ParamColumnMap::new();
    let Some(labels) = type_labels(block, block_name)? else {
        return Ok(columns);
    };
    // type name -> (style, params); a name two styles define is ambiguous.
    let mut by_name: HashMap<String, Vec<(&str, Params)>> = HashMap::new();
    for style in styles {
        for (name, params) in style.defs().kernel_type_params()? {
            by_name
                .entry(name)
                .or_default()
                .push((style.name(), params));
        }
    }
    let rows = labels.len();
    for (row, label) in labels.iter().enumerate() {
        let params = match by_name.get(label.as_str()).map(Vec::as_slice) {
            Some([(_, params)]) => params,
            Some(several) => {
                let names: Vec<&str> = several.iter().map(|(s, _)| *s).collect();
                return Err(format!(
                    "materialize_params: {block_name} row {row}: type '{label}' is defined by \
                     {} {category} styles ({})",
                    names.len(),
                    names.join(", ")
                ));
            }
            None => {
                return Err(format!(
                    "materialize_params: {block_name} row {row}: no {category} style defines \
                     type '{label}'"
                ));
            }
        };
        fill(&mut columns, rows, row, params);
    }
    Ok(columns)
}

/// The per-atom parameter columns: every `atom` style's type row and every
/// row-holding `pair` style's self row of each atom's type.
fn atom_columns(ff: &ForceField, block: &Block) -> Result<ParamColumnMap, String> {
    let mut columns = ParamColumnMap::new();
    let sources: Vec<&Style> = ff
        .styles()
        .iter()
        .filter(|s| matches!(s.category(), "atom" | "pair"))
        .filter(|s| !s.defs().collect_type_params().is_empty())
        .collect();
    if sources.is_empty() {
        return Ok(columns);
    }
    let Some(labels) = type_labels(block, ATOMS)? else {
        return Ok(columns);
    };
    let rows = labels.len();
    for style in sources {
        let rows_of: HashMap<String, Params> =
            style.defs().kernel_type_params()?.into_iter().collect();
        let mut own = ParamColumnMap::new();
        for (row, label) in labels.iter().enumerate() {
            let key = match style.category() {
                "pair" => pair_key(label, label)?,
                _ => label.clone(),
            };
            let params = rows_of.get(&key).ok_or_else(|| {
                let what = if style.category() == "pair" {
                    "self row"
                } else {
                    "type"
                };
                format!(
                    "materialize_params: atom {row}: {} style '{}' has no {what} for type \
                     '{label}'",
                    style.category(),
                    style.name()
                )
            })?;
            fill(&mut own, rows, row, params);
        }
        for (key, column) in own {
            if columns.contains_key(&key) {
                return Err(format!(
                    "materialize_params: two styles write the per-atom parameter '{key}' \
                     ({} style '{}' is the second)",
                    style.category(),
                    style.name()
                ));
            }
            columns.insert(key, column);
        }
    }
    Ok(columns)
}

/// Insert `columns` into `frame`'s block `block_name` as `<prefix><name>`.
fn write(
    frame: &mut Frame,
    block_name: &str,
    prefix: &str,
    columns: ParamColumnMap,
) -> Result<Vec<(String, String)>, String> {
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    let block = frame
        .get_mut(block_name)
        .expect("the caller read this block");
    let mut written = Vec::with_capacity(columns.len());
    for (key, column) in columns {
        let name = format!("{prefix}{key}");
        block
            .insert(
                name.clone(),
                Array1::from_vec(column.values).into_dyn(),
                Some(column.valid),
            )
            .map_err(|e| format!("materialize_params: {block_name}.{name}: {e}"))?;
        written.push((block_name.to_owned(), name));
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::ir::Params;
    use molrs::op::Idx;

    fn uint(values: &[Idx]) -> ndarray::ArrayD<Idx> {
        Array1::from_vec(values.to_vec()).into_dyn()
    }

    fn strings(values: &[&str]) -> ndarray::ArrayD<String> {
        Array1::from_vec(values.iter().map(|s| (*s).to_owned()).collect()).into_dyn()
    }

    /// A-B-A with bonds of two types, one torsion-free angle, and LJ self rows.
    fn system() -> (ForceField, Frame) {
        let mut ff = ForceField::new("t");
        let atoms = ff.def_style("atom", "full", Params::new()).unwrap();
        atoms
            .def_type("A", &[], Params::from_iter([("mass", 12.0)]))
            .unwrap();
        atoms
            .def_type("B", &[], Params::from_iter([("mass", 1.0)]))
            .unwrap();
        let lj = ff.def_style("pair", "lj/cut", Params::new()).unwrap();
        lj.def_type(
            "A",
            &["A"],
            Params::from_iter([("epsilon", 0.1), ("sigma", 3.4)]),
        )
        .unwrap();
        lj.def_type(
            "B",
            &["B"],
            Params::from_iter([("epsilon", 0.02), ("sigma", 2.5)]),
        )
        .unwrap();
        ff.def_style("pair", "coul/cut", Params::from_iter([("coulomb", 332.0)]))
            .unwrap();
        let bonds = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        bonds
            .def_type(
                "A-B",
                &["A", "B"],
                Params::from_iter([("k", 300.0), ("r0", 1.1)]),
            )
            .unwrap();
        let morse = ff.def_style("bond", "morse", Params::new()).unwrap();
        morse
            .def_type(
                "A-A",
                &["A", "A"],
                Params::from_iter([("d0", 80.0), ("alpha", 2.0), ("r0", 1.5)]),
            )
            .unwrap();

        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("type", strings(&["A", "A", "B"]), None)
            .unwrap();
        frame.insert(ATOMS, atoms);
        let mut bonds = Block::new();
        bonds.insert("atomi", uint(&[0, 1]), None).unwrap();
        bonds.insert("atomj", uint(&[1, 2]), None).unwrap();
        bonds
            .insert("type", strings(&["A-A", "A-B"]), None)
            .unwrap();
        frame.insert(BONDS, bonds);
        (ff, frame)
    }

    fn float(frame: &Frame, block: &str, key: &str) -> Vec<F> {
        frame
            .get(block)
            .unwrap()
            .get(key)
            .unwrap()
            .get::<f64>()
            .unwrap()
            .iter()
            .copied()
            .collect()
    }

    /// Each bond row carries its own style's parameters; a parameter only the
    /// other style has is a null cell, not a zero.
    #[test]
    fn relation_rows_carry_their_types_parameters_and_null_where_absent() {
        let (ff, mut frame) = system();
        let written = ff.materialize_params(&mut frame, "ref_").unwrap();
        let bonds: Vec<&str> = written
            .iter()
            .filter(|(b, _)| b == BONDS)
            .map(|(_, c)| c.as_str())
            .collect();
        assert_eq!(bonds, ["ref_alpha", "ref_d0", "ref_k", "ref_r0"]);
        assert_eq!(float(&frame, BONDS, "ref_r0"), [1.5, 1.1]);
        assert_eq!(float(&frame, BONDS, "ref_k")[1], 300.0);
        let block = frame.get(BONDS).unwrap();
        assert_eq!(block.validity("ref_k"), Some(&[false, true][..]));
        assert_eq!(block.validity("ref_d0"), Some(&[true, false][..]));
        assert_eq!(block.validity("ref_r0"), None);
    }

    /// Atoms get their atom-style row and their LJ self row; `coul/cut` has no
    /// per-type rows and writes nothing.
    #[test]
    fn atoms_carry_mass_and_their_lj_self_row() {
        let (ff, mut frame) = system();
        ff.materialize_params(&mut frame, "ref_").unwrap();
        assert_eq!(float(&frame, ATOMS, "ref_mass"), [12.0, 12.0, 1.0]);
        assert_eq!(float(&frame, ATOMS, "ref_sigma"), [3.4, 3.4, 2.5]);
        assert_eq!(float(&frame, ATOMS, "ref_epsilon"), [0.1, 0.1, 0.02]);
        assert!(frame.get(ATOMS).unwrap().get("ref_coulomb").is_none());
    }

    /// A row typed by no style of the field is not this field's frame.
    #[test]
    fn a_row_no_style_defines_is_an_error() {
        let (ff, mut frame) = system();
        let bonds = frame.get_mut(BONDS).unwrap();
        bonds
            .insert("type", strings(&["A-A", "B-B"]), None)
            .unwrap();
        let err = ff.materialize_params(&mut frame, "ref_").unwrap_err();
        assert!(
            err.contains("bonds row 1") && err.contains("'B-B'"),
            "{err}"
        );
    }

    /// An atom whose type has no self row in a row-holding pair style.
    #[test]
    fn an_atom_without_a_self_row_is_an_error() {
        let (ff, mut frame) = system();
        let atoms = frame.get_mut(ATOMS).unwrap();
        atoms
            .insert("type", strings(&["A", "A", "C"]), None)
            .unwrap();
        let err = ff.materialize_params(&mut frame, "ref_").unwrap_err();
        assert!(err.contains("atom 2"), "{err}");
    }

    /// Writing twice replaces the columns rather than failing or stacking.
    #[test]
    fn a_second_write_replaces_the_columns() {
        let (ff, mut frame) = system();
        ff.materialize_params(&mut frame, "").unwrap();
        let again = ff.materialize_params(&mut frame, "").unwrap();
        assert!(again.contains(&(BONDS.to_owned(), "k".to_owned())));
        assert_eq!(float(&frame, BONDS, "r0"), [1.5, 1.1]);
    }
}
