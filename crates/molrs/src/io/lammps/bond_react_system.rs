//! The file set of a LAMMPS `fix bond/react` (REACTER) run: the system's data
//! file, its force-field include and every template's molecule and map files.

use crate::core::keys::LAMMPS_TYPE_MASSES;
use crate::io::invalid_data;
use crate::io::lammps::LammpsForcefieldWriter;
use crate::io::lammps::bond_react::BondReactTemplate;
use crate::io::lammps::data::write_lammps_data;
use crate::io::lammps::molecule::write_lammps_molecule;
use crate::io::writer::ForceFieldWriter;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};

use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::MetaValue;
use molrs::core::TypeLabels;
use molrs::core::keys;
use molrs::ff::forcefield::ForceField;
use molrs::op::{F, Idx};
use ndarray::Array1;

/// The typed blocks a template and the system share a numbering for.
const TYPED_BLOCKS: [&str; 5] = ["atoms", "bonds", "angles", "dihedrals", "impropers"];

/// Topology rows a template lost because their type is not a label of the
/// unified inventory (an empty, `"None"` or purely numeric `type`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DroppedRows {
    /// The reaction's name.
    pub template: String,
    /// The block the rows were in (`"bonds"`, `"angles"`, …).
    pub block: String,
    /// How many rows were dropped, summed over the pre and post templates.
    pub rows: usize,
}

/// What [`write_lammps_bond_react_system`] wrote.
#[derive(Debug, Clone)]
pub struct BondReactSystem {
    /// The type numbering shared by the data file, the force-field include
    /// and every template.
    pub labels: TypeLabels,
    /// The data file, `{workdir}/{stem}.data`.
    pub data_path: PathBuf,
    /// The force-field include, `{workdir}/{stem}.ff`, written without a
    /// `units` line: the input reads it after the data file.
    pub ff_path: PathBuf,
    /// Template topology rows left out for want of a type label.
    pub dropped: Vec<DroppedRows>,
}

/// Whether `label` names a type: non-empty, not the `"None"` placeholder,
/// and not a bare number (which is already an id, not a label).
fn is_label(label: &str) -> bool {
    !label.is_empty() && label != "None" && !label.bytes().all(|b| b.is_ascii_digit())
}

/// Every type label `frames` use, per typed block, sorted.
fn unified_labels(frames: &[&Frame]) -> HashMap<&'static str, Vec<String>> {
    TYPED_BLOCKS
        .iter()
        .map(|&block| {
            let mut all = BTreeSet::new();
            for frame in frames {
                if let Some(types) = frame
                    .get(block)
                    .and_then(|b| b.get(keys::TYPE))
                    .and_then(|c| c.get::<String>())
                {
                    all.extend(types.iter().filter(|t| is_label(t)).cloned());
                }
            }
            (block, all.into_iter().collect())
        })
        .collect()
}

/// The mass of each atom label the frames use: the first row's `mass`
/// column. The data file needs one for every declared type, a type only a
/// template uses included; a label no frame states a mass for is left out,
/// and the data writer refuses its type by name (no mass is derived from the
/// species, none defaulted).
fn label_masses(frames: &[&Frame]) -> HashMap<String, F> {
    let mut out = HashMap::new();
    for frame in frames {
        let Some(atoms) = frame.get("atoms") else {
            continue;
        };
        let (Some(types), Some(mass)) = (
            atoms.get(keys::TYPE).and_then(|c| c.get::<String>()),
            atoms.get(keys::MASS).and_then(|c| c.get::<f64>()),
        ) else {
            continue;
        };
        for (label, &m) in types.iter().zip(mass.iter()) {
            out.entry(label.clone()).or_insert(m);
        }
    }
    out
}

/// A template frame ready for `write_lammps_molecule`: atom ids its row
/// numbers, every typed row's `type_id` its label's unified id, and topology
/// rows without a unified label dropped (their count per block returned).
fn numbered_template(
    frame: &Frame,
    ids: &HashMap<&'static str, HashMap<&str, Idx>>,
    which: &str,
) -> Result<(Frame, Vec<(&'static str, usize)>)> {
    let mut out = frame.clone();
    let mut dropped = Vec::new();
    for block_name in TYPED_BLOCKS {
        let Some(block) = out.get(block_name) else {
            continue;
        };
        let n = block.n_rows().unwrap_or(0);
        if n == 0 {
            continue;
        }
        let Some(types) = block.get(keys::TYPE).and_then(|c| c.get::<String>()) else {
            continue;
        };
        let map = &ids[block_name];
        let labels: Vec<String> = types.iter().cloned().collect();
        let keep: Vec<usize> = (0..n)
            .filter(|&i| map.contains_key(labels[i].as_str()))
            .collect();
        let mut block: Block = if keep.len() < n {
            if block_name == "atoms" {
                let bad = labels.iter().find(|l| !map.contains_key(l.as_str()));
                return Err(invalid_data(format!(
                    "the {which} template has an atom without a type label ({bad:?}); \
                     type every template atom"
                )));
            }
            dropped.push((block_name, n - keep.len()));
            block
                .select_rows(&keep)
                .map_err(|e| invalid_data(e.to_string()))?
        } else {
            block.clone()
        };
        let type_ids: Vec<Idx> = keep.iter().map(|&i| map[labels[i].as_str()]).collect();
        block
            .insert(keys::TYPE_ID, Array1::from_vec(type_ids).into_dyn(), None)
            .map_err(|e| invalid_data(e.to_string()))?;
        out.insert(block_name, block);
    }
    if let Some(atoms) = out.get_mut("atoms") {
        let n = atoms.n_rows().unwrap_or(0) as Idx;
        atoms
            .insert(keys::ID, Array1::from_iter(1..=n).into_dyn(), None)
            .map_err(|e| invalid_data(e.to_string()))?;
    }
    Ok((out, dropped))
}

/// Write the file set of a `fix bond/react` run into `workdir` (created if
/// missing): `{stem}.data` for `frame`, `{stem}.ff` for `forcefield`, and
/// `{name}_pre.mol`, `{name}_post.mol` and `{name}.map` for each named
/// template, `stem` being `workdir`'s own file name.
///
/// Every type label `frame` and the templates use (their string `type`
/// columns) is declared in the data file, so the data file, the `.ff`
/// include and the templates number types alike; template topology rows
/// whose type is not such a label are left out of the molecule files and
/// reported in [`BondReactSystem::dropped`]. The include carries the
/// styles and coefficients ([`LammpsForcefieldWriter`]) but no `units`
/// line: the input sets `units` and `atom_style`, reads the data file, then
/// includes it. Every check runs before the first file is written.
///
/// # Errors
///
/// [`ErrorKind::InvalidInput`] for a `workdir` without a file name or two
/// templates of one name. [`ErrorKind::InvalidData`] for a template whose
/// map is malformed ([`write_lammps_bond_react_map`](crate::io::write_lammps_bond_react_map)) or that has an atom
/// without a type label; for a `frame` whose own labels are not the unified
/// numbering (a purely numeric label beside named ones); and for a
/// `forcefield` the include cannot be written from — its source is the
/// [`ForceFieldWriteError`](crate::io::writer::ForceFieldWriteError), a typed
/// refusal kept. Any file's I/O error.
pub fn write_lammps_bond_react_system(
    workdir: impl AsRef<Path>,
    frame: &Frame,
    forcefield: &ForceField,
    templates: &[(String, BondReactTemplate)],
) -> Result<BondReactSystem> {
    let workdir = workdir.as_ref();
    let stem = workdir
        .file_name()
        .ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("{} names no directory", workdir.display()),
            )
        })?
        .to_owned();
    let mut seen = HashSet::new();
    if let Some((name, _)) = templates.iter().find(|(name, _)| !seen.insert(name)) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("two templates are named {name:?}"),
        ));
    }
    // Every map is checked before any file is written.
    let maps = templates
        .iter()
        .map(|(_, t)| t.map_file_text())
        .collect::<Result<Vec<String>>>()?;

    let mut all: Vec<&Frame> = vec![frame];
    for (_, t) in templates {
        all.push(&t.pre);
        all.push(&t.post);
    }
    let unified = unified_labels(&all);

    let mut system = frame.clone();
    for block in TYPED_BLOCKS {
        TypeLabels::declare(&mut system, block, &unified[block]).map_err(invalid_data)?;
    }
    let labels = TypeLabels::from_frame(&system).map_err(invalid_data)?;
    for block in TYPED_BLOCKS {
        let declared = labels.block(block).and_then(|b| b.labels()).unwrap_or(&[]);
        if !unified[block].is_empty() && declared != unified[block].as_slice() {
            return Err(invalid_data(format!(
                "the system's {block} labels {declared:?} are not the unified numbering \
                 {:?}; give every {block} row a named type",
                unified[block]
            )));
        }
    }
    let ids: HashMap<&'static str, HashMap<&str, Idx>> = TYPED_BLOCKS
        .iter()
        .map(|&block| {
            let map = unified[block]
                .iter()
                .enumerate()
                .map(|(i, label)| (label.as_str(), (i + 1) as Idx))
                .collect();
            (block, map)
        })
        .collect();

    // The include is read after `read_data` (its coefficients need the box),
    // where LAMMPS refuses a `units` line: the input states the units.
    let mut include = Vec::new();
    LammpsForcefieldWriter::new(&labels)
        .with_skip_units(true)
        .write(&mut include, forcefield)?;
    let mut numbered = Vec::with_capacity(templates.len());
    for (_, template) in templates {
        numbered.push([
            numbered_template(&template.pre, &ids, "pre")?,
            numbered_template(&template.post, &ids, "post")?,
        ]);
    }

    std::fs::create_dir_all(workdir)?;
    let base = workdir.join(&stem);
    let data_path = base.with_extension("data");
    // A type only a template uses is declared in the data file, and LAMMPS
    // reads its mass from there: the templates' masses go into the system's
    // per-type masses, beside any it already states.
    let mut masses = match system.meta.get(LAMMPS_TYPE_MASSES) {
        Some(MetaValue::Json(serde_json::Value::Object(map))) => map.clone(),
        _ => serde_json::Map::new(),
    };
    for (label, m) in label_masses(&all) {
        masses.entry(label).or_insert(serde_json::Value::from(m));
    }
    system.meta.insert(
        LAMMPS_TYPE_MASSES,
        MetaValue::Json(serde_json::Value::Object(masses)),
    );
    write_lammps_data(&data_path, &system)?;
    let ff_path = base.with_extension("ff");
    std::fs::write(&ff_path, include)?;

    let mut dropped = Vec::new();
    for (((name, _), map), parts) in templates.iter().zip(maps).zip(numbered) {
        let mut lost: HashMap<&'static str, usize> = HashMap::new();
        for (which, (frame, gone)) in ["pre", "post"].into_iter().zip(parts) {
            for (block, rows) in gone {
                *lost.entry(block).or_default() += rows;
            }
            write_lammps_molecule(
                workdir.join(format!("{name}_{which}.mol")).as_path(),
                &frame,
            )?;
        }
        for block in TYPED_BLOCKS {
            if let Some(&rows) = lost.get(block) {
                dropped.push(DroppedRows {
                    template: name.clone(),
                    block: block.to_owned(),
                    rows,
                });
            }
        }
        std::fs::write(workdir.join(format!("{name}.map")), map)?;
    }

    Ok(BondReactSystem {
        labels,
        ff_path,
        data_path,
        dropped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::lammps::bond_react::tests::{coupling, strings, template};
    use molrs::ff::ir::Params;

    /// Every label the system and `coupling` use; `hc-oh` left out when
    /// `complete` is false.
    fn forcefield(complete: bool) -> ForceField {
        let lj = |eps: f64, sigma: f64| Params::from_iter([("epsilon", eps), ("sigma", sigma)]);
        let bond = |k: f64, r0: f64| Params::from_iter([("k", k), ("r0", r0)]);
        let mut ff = ForceField::new("hand");
        ff.set_units("real");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 9.0)]))
            .unwrap()
            .def_type("c3", &["c3"], lj(0.1078, 3.39771))
            .unwrap()
            .def_type("hc", &["hc"], lj(0.0157, 2.64953))
            .unwrap()
            .def_type("oh", &["oh"], lj(0.093, 3.242871))
            .unwrap();
        let bonds = ff
            .def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("c3-c3", &["c3", "c3"], bond(300.0, 1.53))
            .unwrap()
            .def_type("c3-oh", &["c3", "oh"], bond(320.0, 1.41))
            .unwrap();
        if complete {
            bonds
                .def_type("hc-oh", &["hc", "oh"], bond(550.0, 0.96))
                .unwrap();
        }
        ff
    }

    /// A force field the include cannot be written from is refused before
    /// any file is written.
    #[test]
    fn an_incomplete_forcefield_writes_nothing() {
        let dir = std::env::temp_dir().join(format!(
            "molrs-bond-react-incomplete-{}",
            std::process::id()
        ));
        let workdir = dir.join("rxn");
        let system = template(&["c3", "c3"], &[0, 0], &[(0, 1, "c3-c3")]);
        let err = write_lammps_bond_react_system(
            &workdir,
            &system,
            &forcefield(false),
            &[("rxn1".into(), coupling())],
        )
        .unwrap_err();
        let written = workdir.exists();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(err.to_string().contains("hc-oh"), "{err}");
        assert!(!written);
    }

    #[test]
    fn the_system_declares_template_only_types_and_numbers_templates_alike() {
        let dir = std::env::temp_dir().join(format!("molrs-bond-react-{}", std::process::id()));
        let workdir = dir.join("rxn");
        let mut system = template(&["c3", "c3"], &[0, 0], &[(0, 1, "c3-c3")]);
        system
            .get_mut("atoms")
            .unwrap()
            .insert(
                "mol_id",
                Array1::from_vec(vec![1 as Idx, 1]).into_dyn(),
                None,
            )
            .unwrap();
        let mut t = coupling();
        for part in [&mut t.pre, &mut t.post] {
            let n = part["atoms"].n_rows().unwrap();
            part.get_mut("atoms")
                .unwrap()
                .insert(
                    "mol_id",
                    Array1::from_vec(vec![1 as Idx; n]).into_dyn(),
                    None,
                )
                .unwrap();
        }
        // An untyped template angle is dropped and reported.
        let mut angles = Block::new();
        for (k, v) in [("atomi", 0), ("atomj", 1), ("atomk", 2)] {
            angles
                .insert(k, Array1::from_vec(vec![v as Idx]).into_dyn(), None)
                .unwrap();
        }
        angles.insert("type", strings(&["None"]), None).unwrap();
        t.post.insert("angles", angles);

        let out = write_lammps_bond_react_system(
            &workdir,
            &system,
            &forcefield(true),
            &[("rxn1".into(), t)],
        )
        .unwrap();
        let data = std::fs::read_to_string(&out.data_path).unwrap();
        let include = std::fs::read_to_string(&out.ff_path).unwrap();
        let pre = std::fs::read_to_string(workdir.join("rxn1_pre.mol")).unwrap();
        let post = std::fs::read_to_string(workdir.join("rxn1_post.mol")).unwrap();
        let map_exists = workdir.join("rxn1.map").exists();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(out.data_path, workdir.join("rxn.data"));
        assert_eq!(out.ff_path, workdir.join("rxn.ff"));
        assert!(map_exists);
        // The include numbers by the shared labels, template-only ones too,
        // and leaves `units` to the input.
        assert!(include.contains("bond_coeff c3-oh 320"), "{include}");
        assert!(include.contains("pair_coeff oh oh"), "{include}");
        assert!(!include.contains("units"), "{include}");
        // atoms c3=1, hc=2, oh=3; bonds c3-c3=1, c3-oh=2, hc-oh=3.
        assert!(data.contains("3 atom types"), "{data}");
        // `oh` only a template uses: its mass is the template's, not 1.
        let masses = &data[data.find("Masses\n\n").unwrap()..];
        let oh: f64 = masses.lines().nth(4).unwrap()[2..].parse().unwrap();
        assert!((oh - 15.999).abs() < 0.01, "{data}");
        assert!(data.contains("3 bond types"), "{data}");
        assert!(
            data.contains("Bond Type Labels\n\n1 c3-c3\n2 c3-oh\n3 hc-oh\n"),
            "{data}"
        );
        assert!(pre.contains("Types\n\n1 1\n2 3\n3 2\n"), "{pre}");
        assert!(post.contains("Types\n\n1 3\n2 1\n3 2\n"), "{post}");
        assert!(post.contains("Bonds\n\n1 2 1 2\n"), "{post}");
        assert!(!post.contains("Angles"), "{post}");
        assert_eq!(
            out.dropped,
            [DroppedRows {
                template: "rxn1".into(),
                block: "angles".into(),
                rows: 1
            }]
        );
        let atoms = out.labels.block("atoms").unwrap();
        assert_eq!(atoms.labels().unwrap(), ["c3", "hc", "oh"]);
    }
}
