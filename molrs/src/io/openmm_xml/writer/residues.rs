//! Residue templates from a typed frame
//! ([`OpenmmXmlWriter::write_system_str`]).
//!
//! OpenMM's `createSystem` matches every residue of the PDB topology to a
//! `<Residue>` template by element and bond graph, then takes each atom's
//! type (and, under `<UseAttributeFromResidue>`, its charge) from the
//! template. The residues and atom names are the PDB writer's
//! ([`AtomLabels`]), so the templates describe exactly the topology OpenMM
//! reads from [`write_pdb`](crate::io::write_pdb) of the same frame.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use molrs::core::Frame;

use super::{Charges, OpenmmXmlWriter, esc};
use crate::ff::forcefield::{AtomType, ForceField};
use crate::io::pdb::labels::AtomLabels;
use crate::io::writer::ForceFieldWriteError;

/// The residue templates of one frame and what the force field needs from
/// it.
pub(super) struct Templates {
    /// The `<Residue>` elements, whole.
    pub(super) xml: String,
    /// The element of each atom type the frame uses (the frame's).
    pub(super) elements: HashMap<String, String>,
    /// Where `<NonbondedForce>` takes its charges from.
    pub(super) charges: Charges,
}

/// One residue's template content; residues with equal content share a
/// template.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Content {
    res_name: String,
    /// `(name, type, charge bits)` per atom.
    atoms: Vec<(String, String, Option<u64>)>,
    /// Bonds inside the residue, by local index, `(lo, hi)` sorted.
    bonds: Vec<(usize, usize)>,
    /// One local index per bond leaving the residue, sorted.
    external: Vec<usize>,
}

impl Templates {
    pub(super) fn of(
        w: &OpenmmXmlWriter,
        ff: &ForceField,
        frame: &Frame,
    ) -> Result<Self, ForceFieldWriteError> {
        let atoms = frame.get("atoms").ok_or("frame has no atoms block")?;
        let n = atoms.n_rows().unwrap_or(0);
        let strings = |key: &str| atoms.get(key).and_then(|c| c.as_string());
        let types = strings("type").ok_or("atoms has no string type column")?;
        let elements = strings("element").ok_or(
            "atoms has no string element column: OpenMM matches residues to templates by \
             element",
        )?;
        let charges = atoms.get("charge").and_then(|c| c.as_float());
        let types: Vec<&str> = types.iter().map(|t| t.as_str()).collect();
        let elements: Vec<&str> = elements.iter().map(|e| e.trim()).collect();
        let charges: Option<Vec<f64>> = charges.map(|q| q.iter().copied().collect());

        let element_of = type_elements(ff, &types, &elements)?;
        let bonds = bonds(frame, n)?;
        let labels = AtomLabels::pdb(frame, n).map_err(|e| e.to_string())?;
        let residues = labels.residues();
        let mut residue_of = vec![0; n];
        for (r, run) in residues.iter().enumerate() {
            residue_of[run.clone()].fill(r);
        }

        let mut inside: Vec<Vec<(usize, usize)>> = vec![Vec::new(); residues.len()];
        let mut leaving: Vec<Vec<usize>> = vec![Vec::new(); residues.len()];
        for &(i, j) in &bonds {
            let (ri, rj) = (residue_of[i], residue_of[j]);
            let (li, lj) = (i - residues[ri].start, j - residues[rj].start);
            if ri == rj {
                inside[ri].push((li.min(lj), li.max(lj)));
            } else {
                leaving[ri].push(li);
                leaving[rj].push(lj);
            }
        }

        // One template per distinct content, in order of first appearance.
        let mut seen: HashSet<Content> = HashSet::new();
        let mut per_name: HashMap<String, usize> = HashMap::new();
        let mut xml = String::new();
        for (r, run) in residues.iter().enumerate() {
            let content = Content {
                res_name: labels.res_names[run.start].clone(),
                atoms: run
                    .clone()
                    .map(|a| {
                        (
                            labels.names[a].clone(),
                            types[a].to_owned(),
                            charges.as_ref().map(|q| q[a].to_bits()),
                        )
                    })
                    .collect(),
                bonds: sorted(std::mem::take(&mut inside[r])),
                external: sorted(std::mem::take(&mut leaving[r])),
            };
            unique_names(&content, &labels, run)?;
            if seen.contains(&content) {
                continue;
            }
            let k = per_name.entry(content.res_name.clone()).or_insert(0);
            *k += 1;
            let name = match *k {
                1 => content.res_name.clone(),
                k => format!("{}_{k}", content.res_name),
            };
            xml.push_str(&template_xml(w, &name, &content));
            seen.insert(content);
        }

        let charges = match &charges {
            Some(q) => Charges::Residues {
                nonzero: q.iter().any(|&q| q != 0.0),
            },
            None => Charges::TypesOnly,
        };
        Ok(Self {
            xml,
            elements: element_of,
            charges,
        })
    }
}

/// Each used type's element, from the frame: one per type, and the type's
/// own `element` when it has one.
fn type_elements(
    ff: &ForceField,
    types: &[&str],
    elements: &[&str],
) -> Result<HashMap<String, String>, ForceFieldWriteError> {
    let defined: HashMap<&str, &AtomType> = ff
        .get_atomtypes()
        .into_iter()
        .filter(|t| t.params.get_str("type_") != Some("*"))
        .map(|t| (t.name.as_str(), t))
        .collect();
    let mut element_of: HashMap<String, String> = HashMap::new();
    for (i, (&ty, &element)) in types.iter().zip(elements).enumerate() {
        let atom_type = defined.get(ty).ok_or_else(|| {
            format!("atom {i} has type '{ty}', which the force field does not define")
        })?;
        if element.is_empty() {
            return Err(format!(
                "atom {i} has no element: OpenMM matches residues to templates by element"
            )
            .into());
        }
        if let Some(own) = atom_type.params.get_str("element")
            && !own.trim().eq_ignore_ascii_case(element)
        {
            return Err(format!(
                "atom {i} of type '{ty}' is element {element}, but the type's element is {own}"
            )
            .into());
        }
        match element_of.get(ty) {
            Some(other) if !other.eq_ignore_ascii_case(element) => {
                return Err(format!(
                    "atoms of type '{ty}' are elements {other} and {element}: an OpenMM atom \
                     type has one element"
                )
                .into());
            }
            Some(_) => {}
            None => {
                element_of.insert(ty.to_owned(), element.to_owned());
            }
        }
    }
    Ok(element_of)
}

/// The frame's bonds, each once, `(lo, hi)`; none without a `bonds` block.
fn bonds(frame: &Frame, n: usize) -> Result<Vec<(usize, usize)>, ForceFieldWriteError> {
    let Some(block) = frame.get("bonds") else {
        return Ok(Vec::new());
    };
    let column = |key: &str| -> Result<Vec<usize>, String> {
        block
            .get(key)
            .and_then(|c| c.as_uint())
            .map(|c| c.iter().map(|&v| v as usize).collect())
            .ok_or_else(|| format!("bonds has no index column {key}"))
    };
    let pairs = column("atomi")?
        .into_iter()
        .zip(column("atomj")?)
        .map(|(i, j)| {
            if i >= n || j >= n || i == j {
                Err(format!(
                    "bond ({i}, {j}) does not join two of the {n} atoms"
                ))
            } else {
                Ok((i.min(j), i.max(j)))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut pairs = sorted(pairs);
    pairs.dedup();
    Ok(pairs)
}

fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort_unstable();
    v
}

/// OpenMM refuses a template with two atoms of one name.
fn unique_names(
    content: &Content,
    labels: &AtomLabels,
    run: &Range<usize>,
) -> Result<(), ForceFieldWriteError> {
    let mut names: HashSet<&str> = HashSet::new();
    for (name, _, _) in &content.atoms {
        if !names.insert(name) {
            return Err(format!(
                "residue {} {} (atoms {}..{}) has two atoms named '{name}': an OpenMM residue \
                 template names its atoms uniquely; give the frame a `name` column unique \
                 within each residue",
                content.res_name, labels.res_seqs[run.start], run.start, run.end
            )
            .into());
        }
    }
    Ok(())
}

/// One `<Residue>` element.
fn template_xml(w: &OpenmmXmlWriter, name: &str, content: &Content) -> String {
    let mut xml = format!("    <Residue name=\"{}\">\n", esc(name));
    for (atom, ty, charge) in &content.atoms {
        let charge = charge
            .map(|q| format!(" charge=\"{}\"", w.fmt_f(f64::from_bits(q))))
            .unwrap_or_default();
        xml.push_str(&format!(
            "      <Atom name=\"{}\" type=\"{}\"{charge}/>\n",
            esc(atom),
            esc(ty)
        ));
    }
    let atom_name = |i: usize| esc(&content.atoms[i].0);
    for &(i, j) in &content.bonds {
        xml.push_str(&format!(
            "      <Bond atomName1=\"{}\" atomName2=\"{}\"/>\n",
            atom_name(i),
            atom_name(j)
        ));
    }
    for &i in &content.external {
        xml.push_str(&format!(
            "      <ExternalBond atomName=\"{}\"/>\n",
            atom_name(i)
        ));
    }
    xml.push_str("    </Residue>\n");
    xml
}

#[cfg(test)]
mod tests {
    use crate::ff::forcefield::ForceField;
    use crate::ff::ir::Params;
    use crate::io::{
        read_openmm_forcefield_str, read_pdb_str, write_openmm_forcefield_str,
        write_openmm_system_str, write_pdb_str,
    };
    use molrs::core::{Block, Frame};
    use molrs::op::Idx;
    use ndarray::{ArrayD, IxDyn};
    use serde_json::Value;

    const OPLS_XML: &str = include_str!("../../../ff/testdata/openmm/opls.xml");
    const OPLS_JSON: &str = include_str!("../../../ff/testdata/openmm/opls.json");

    fn column<T: Clone>(v: Vec<T>) -> ArrayD<T> {
        let n = v.len();
        ArrayD::from_shape_vec(IxDyn(&[n]), v).unwrap()
    }

    fn strs(v: &[&str]) -> ArrayD<String> {
        column(v.iter().map(|s| s.to_string()).collect())
    }

    fn json_strings(v: &Value, key: &str) -> Vec<String> {
        v[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_owned())
            .collect()
    }

    /// `copies` 1-propanols of the OPLS fixture (types, elements, charges,
    /// positions, bonds), each copy 50 Å further along x; no names and no
    /// residue columns.
    fn propanols(copies: usize) -> Frame {
        let v: Value = serde_json::from_str(OPLS_JSON).unwrap();
        let types = json_strings(&v, "types");
        let elements = json_strings(&v, "elements");
        let charges: Vec<f64> = v["charges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| q.as_f64().unwrap())
            .collect();
        let positions: Vec<Vec<f64>> = v["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                p.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect()
            })
            .collect();
        let bonds: Vec<(Idx, Idx)> = v["bonds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| (b[0].as_u64().unwrap() as Idx, b[1].as_u64().unwrap() as Idx))
            .collect();
        let n = types.len();
        let repeat = |v: &Vec<String>| -> Vec<String> {
            (0..copies).flat_map(|_| v.iter().cloned()).collect()
        };
        let mut atoms = Block::new();
        atoms.insert("type", column(repeat(&types))).unwrap();
        atoms.insert("element", column(repeat(&elements))).unwrap();
        atoms
            .insert(
                "charge",
                column((0..copies).flat_map(|_| charges.iter().copied()).collect()),
            )
            .unwrap();
        for (k, key) in ["x", "y", "z"].into_iter().enumerate() {
            let values = (0..copies)
                .flat_map(|c| {
                    positions
                        .iter()
                        .map(move |p| p[k] + if k == 0 { 50.0 * c as f64 } else { 0.0 })
                })
                .collect();
            atoms.insert(key, column::<f64>(values)).unwrap();
        }
        let mut block = Block::new();
        let shifted = |c: usize, i: Idx| i + (c * n) as Idx;
        block
            .insert(
                "atomi",
                column(
                    (0..copies)
                        .flat_map(|c| bonds.iter().map(move |b| shifted(c, b.0)))
                        .collect(),
                ),
            )
            .unwrap();
        block
            .insert(
                "atomj",
                column(
                    (0..copies)
                        .flat_map(|c| bonds.iter().map(move |b| shifted(c, b.1)))
                        .collect(),
                ),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", block);
        frame
    }

    fn opls() -> ForceField {
        read_openmm_forcefield_str(OPLS_XML).unwrap()
    }

    fn system(ff: &ForceField, frame: &Frame) -> String {
        write_openmm_system_str(ff, frame, None).unwrap_or_else(|e| panic!("{e}"))
    }

    fn system_err(ff: &ForceField, frame: &Frame) -> String {
        write_openmm_system_str(ff, frame, None)
            .expect_err("expected the system to be refused")
            .into()
    }

    /// `(name, [(atom name, type, charge)], [(bond names)], [external names])`
    /// per `<Residue>`.
    type Template = (
        String,
        Vec<(String, String, Option<f64>)>,
        Vec<(String, String)>,
        Vec<String>,
    );

    fn templates(xml: &str) -> Vec<Template> {
        let doc = roxmltree::Document::parse(xml).unwrap();
        let attr = |n: &roxmltree::Node, k: &str| n.attribute(k).unwrap().to_owned();
        doc.descendants()
            .filter(|n| n.has_tag_name("Residue"))
            .map(|r| {
                let kids: Vec<_> = r.children().filter(|c| c.is_element()).collect();
                let of = |tag: &'static str| kids.iter().filter(move |c| c.has_tag_name(tag));
                (
                    attr(&r, "name"),
                    of("Atom")
                        .map(|a| {
                            (
                                attr(a, "name"),
                                attr(a, "type"),
                                a.attribute("charge").map(|q| q.parse().unwrap()),
                            )
                        })
                        .collect(),
                    of("Bond")
                        .map(|b| (attr(b, "atomName1"), attr(b, "atomName2")))
                        .collect(),
                    of("ExternalBond").map(|b| attr(b, "atomName")).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn one_template_per_distinct_molecule_named_as_the_pdb_names_it() {
        let frame = propanols(2);
        let xml = system(&opls(), &frame);
        let found = templates(&xml);
        assert_eq!(found.len(), 1, "two identical propanols share one template");
        let (name, atoms, bonds, external) = &found[0];
        assert_eq!(name, "MOL");
        let v: Value = serde_json::from_str(OPLS_JSON).unwrap();
        let names: Vec<&str> = atoms.iter().map(|a| a.0.as_str()).collect();
        assert_eq!(
            names,
            [
                "C1", "C2", "C3", "O1", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"
            ]
        );
        let types: Vec<String> = atoms.iter().map(|a| a.1.clone()).collect();
        assert_eq!(types, json_strings(&v, "types"));
        let charges: Vec<f64> = atoms.iter().map(|a| a.2.unwrap()).collect();
        let want: Vec<f64> = v["charges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| q.as_f64().unwrap())
            .collect();
        assert_eq!(charges, want);
        assert_eq!(bonds.len(), 11);
        assert!(bonds.contains(&("O1".to_owned(), "H8".to_owned())));
        assert!(external.is_empty());

        // The PDB of the same frame: one residue per molecule, the same names.
        let pdb = read_pdb_str(&write_pdb_str(&frame).unwrap()).unwrap();
        let atoms = pdb.get("atoms").unwrap();
        let pdb_names: Vec<String> = atoms
            .get("name")
            .and_then(|c| c.as_string())
            .unwrap()
            .iter()
            .cloned()
            .collect();
        assert_eq!(&pdb_names[..12], names.as_slice());
        assert_eq!(&pdb_names[12..], names.as_slice());
        let res_ids: Vec<Idx> = atoms
            .get("res_id")
            .and_then(|c| c.as_uint())
            .unwrap()
            .iter()
            .copied()
            .collect();
        assert_eq!(res_ids, [[1; 12], [2; 12]].concat());
        let res_names = atoms.get("res_name").and_then(|c| c.as_string()).unwrap();
        assert!(res_names.iter().all(|r| r == "MOL"));

        // Charges come from the templates; the types keep their elements.
        let doc = roxmltree::Document::parse(&xml).unwrap();
        let nonbonded = doc
            .descendants()
            .find(|n| n.has_tag_name("NonbondedForce"))
            .unwrap();
        assert!(
            nonbonded
                .children()
                .any(|c| c.has_tag_name("UseAttributeFromResidue"))
        );
        assert!(
            nonbonded
                .children()
                .filter(|c| c.has_tag_name("Atom"))
                .all(|a| a.attribute("charge").is_none())
        );
        assert!(
            doc.descendants()
                .filter(|n| n.has_tag_name("Type"))
                .all(|t| t.attribute("element").is_some())
        );
    }

    #[test]
    fn the_system_reads_back_as_the_force_field() {
        let ff = opls();
        let frame = propanols(1);
        let back = read_openmm_forcefield_str(&system(&ff, &frame)).unwrap();
        for category in ["bond", "angle", "dihedral", "pair"] {
            // (type, [(key, value)]) per row, sorted.
            let rows = |f: &ForceField| -> Vec<(String, Vec<(String, f64)>)> {
                let mut rows: Vec<_> = f
                    .get_styles(category)
                    .into_iter()
                    .flat_map(|s| {
                        s.type_rows()
                            .into_iter()
                            .map(|(name, _, p)| {
                                let mut values: Vec<_> =
                                    p.iter().map(|(k, v)| (k.to_owned(), v)).collect();
                                values.sort_by(|a, b| a.0.cmp(&b.0));
                                (name.to_owned(), values)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                rows.sort_by(|a, b| a.0.cmp(&b.0));
                rows
            };
            let (got, want) = (rows(&back), rows(&ff));
            assert_eq!(got.len(), want.len(), "{category}");
            for ((name, values), (want_name, want_values)) in got.iter().zip(&want) {
                assert_eq!(name, want_name, "{category}");
                assert_eq!(values.len(), want_values.len(), "{category} {name}");
                for ((k, v), (_, w)) in values.iter().zip(want_values) {
                    // Degrees → radians → degrees: within an ulp or so.
                    assert!(
                        (v - w).abs() <= 1e-12 * w.abs().max(1.0),
                        "{name} {k}: {v} {w}"
                    );
                }
            }
        }
        // Charges live on the templates: the types read back without them.
        assert!(
            back.get_atomtypes()
                .iter()
                .all(|t| t.params.get("charge").is_none())
        );

        // A frame without charges: the types' charges, written and read back.
        let mut bare = propanols(1);
        bare.get_mut("atoms").unwrap().remove("charge");
        let xml = system(&ff, &bare);
        assert!(templates(&xml)[0].1.iter().all(|a| a.2.is_none()));
        let back = read_openmm_forcefield_str(&xml).unwrap();
        for t in ff.get_atomtypes() {
            let again = back.get_atomtypes().into_iter().find(|b| b.name == t.name);
            assert_eq!(
                again.and_then(|b| b.params.get("charge")),
                t.params.get("charge"),
                "{}",
                t.name
            );
        }
        // Everything but the templates is the force field's own document.
        let alone = write_openmm_forcefield_str(&ff, None).unwrap();
        let start = xml.find("  <Residues>").unwrap();
        let end = xml.find("  </Residues>\n").unwrap() + "  </Residues>\n".len();
        assert_eq!(format!("{}{}", &xml[..start], &xml[end..]), alone);
    }

    /// CH2-CH2 with one residue per CH2: identical residues joined by a bond
    /// that leaves each.
    fn ethylene_units() -> (ForceField, Frame) {
        let mut ff = ForceField::new("units");
        let atom = |mass: f64, q: f64, e: &str| {
            let mut p = Params::from_pairs(&[("mass", mass), ("charge", q)]);
            p.set_str("element", e);
            p
        };
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], atom(12.011, -0.12, "C"))
            .unwrap()
            .def_type("HC", &[], atom(1.008, 0.06, "H"))
            .unwrap();
        ff.def_style("pair", "coul/cut", Params::new()).unwrap();
        let mut atoms = Block::new();
        atoms
            .insert("type", strs(&["CT", "HC", "HC", "CT", "HC", "HC"]))
            .unwrap();
        atoms
            .insert("element", strs(&["C", "H", "H", "C", "H", "H"]))
            .unwrap();
        atoms
            .insert("res_id", column::<Idx>(vec![1, 1, 1, 2, 2, 2]))
            .unwrap();
        let mut bonds = Block::new();
        bonds
            .insert("atomi", column::<Idx>(vec![0, 0, 0, 3, 3]))
            .unwrap();
        bonds
            .insert("atomj", column::<Idx>(vec![1, 2, 3, 4, 5]))
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);
        (ff, frame)
    }

    #[test]
    fn a_bond_between_residues_is_an_external_bond_of_each() {
        let (ff, frame) = ethylene_units();
        let found = templates(&system(&ff, &frame));
        assert_eq!(found.len(), 1);
        let (_, atoms, bonds, external) = &found[0];
        assert_eq!(atoms.len(), 3);
        assert_eq!(bonds.len(), 2);
        assert_eq!(external, &["C1".to_owned()]);
    }

    #[test]
    fn residues_of_one_name_with_other_content_are_numbered() {
        let mut frame = propanols(2);
        let atoms = frame.get_mut("atoms").unwrap();
        let mut q: Vec<f64> = atoms
            .get("charge")
            .and_then(|c| c.as_float())
            .unwrap()
            .iter()
            .copied()
            .collect();
        q[12] += 0.01;
        q[13] -= 0.01;
        atoms.insert("charge", column(q)).unwrap();
        let names: Vec<String> = templates(&system(&opls(), &frame))
            .into_iter()
            .map(|t| t.0)
            .collect();
        assert_eq!(names, ["MOL", "MOL_2"]);
    }

    #[test]
    fn what_openmm_cannot_match_is_refused_by_name() {
        let ff = opls();

        let mut frame = propanols(1);
        let atoms = frame.get_mut("atoms").unwrap();
        atoms.insert("name", strs(&["C"; 12])).unwrap();
        assert!(system_err(&ff, &frame).contains("two atoms named 'C'"));

        let mut frame = propanols(1);
        let atoms = frame.get_mut("atoms").unwrap();
        let mut types: Vec<String> = atoms
            .get("type")
            .and_then(|c| c.as_string())
            .unwrap()
            .iter()
            .cloned()
            .collect();
        types[0] = "opls_999".into();
        atoms.insert("type", column(types)).unwrap();
        assert!(system_err(&ff, &frame).contains("'opls_999'"));

        let mut frame = propanols(1);
        let atoms = frame.get_mut("atoms").unwrap();
        let mut elements: Vec<String> = atoms
            .get("element")
            .and_then(|c| c.as_string())
            .unwrap()
            .iter()
            .cloned()
            .collect();
        elements[0] = "N".into();
        atoms.insert("element", column(elements)).unwrap();
        assert!(system_err(&ff, &frame).contains("the type's element is C"));

        let mut frame = propanols(1);
        frame.get_mut("atoms").unwrap().remove("element");
        assert!(system_err(&ff, &frame).contains("element column"));
    }

    #[test]
    fn charges_must_be_somewhere_and_priced() {
        let (_, mut frame) = ethylene_units();
        // Types with neither charges nor elements: both from the frame.
        let bare = |pair: &str| {
            let mut ff = ForceField::new("bare");
            ff.def_style("atom", "full", Params::new())
                .unwrap()
                .def_type("CT", &[], Params::from_pairs(&[("mass", 12.011)]))
                .unwrap()
                .def_type("HC", &[], Params::from_pairs(&[("mass", 1.008)]))
                .unwrap();
            let style = ff.def_style("pair", pair, Params::new()).unwrap();
            if pair == "lj/cut" {
                style
                    .def_type(
                        "CT",
                        &["CT"],
                        Params::from_pairs(&[("sigma", 3.5), ("epsilon", 0.066)]),
                    )
                    .unwrap()
                    .def_type(
                        "HC",
                        &["HC"],
                        Params::from_pairs(&[("sigma", 2.5), ("epsilon", 0.03)]),
                    )
                    .unwrap();
            }
            ff
        };
        assert!(system_err(&bare("coul/cut"), &frame).contains("no charge column"));
        frame
            .get_mut("atoms")
            .unwrap()
            .insert("charge", column(vec![-0.12, 0.06, 0.06, -0.12, 0.06, 0.06]))
            .unwrap();
        let xml = system(&bare("coul/cut"), &frame);
        assert!(xml.contains("<Type name=\"CT\" class=\"CT\" element=\"C\""));
        // Frame charges with no Coulomb style to price them.
        assert!(system_err(&bare("lj/cut"), &frame).contains("no Coulomb pair style"));
    }
}
