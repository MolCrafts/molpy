//! OpenMM force-field XML reader (OpenMM's own `<ForceField>` files, and the OPLS-AA / CL&P / foyer packs in the same schema).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use ndarray::ArrayD;
use roxmltree::Node;

use crate::core::constants::VACUUM_DIELECTRIC;
use crate::core::unit_factors::{KCAL_ANGSTROM2_TO_KJ_NM2, KCAL_TO_KJ, NM_TO_ANGSTROM};
use crate::ff::forcefield::ForceField;
use crate::ff::ir::CombiningRule;
use crate::ff::ir::torsion::rb_polynomial;
use crate::ff::ir::{ONE_FOUR, ONE_FOUR_EPSILON14, has_own_one_four};
use crate::ff::ir::{Params, SpecialBonds};
use crate::ff::typifier::OplsAaTypifier;
use crate::io::reader::ForceFieldReader;
use molrs::core::TypeName;

/// The energy expressions of a `<CustomTorsionForce>` read as `improper
/// harmonic`, whitespace removed: OpenMM's CHARMM ports, and molrs's writer
/// form for `chi0 ≠ 0`.
pub(crate) const HARMONIC_IMPROPER_SIGNED: &str = "k*(theta-theta0)^2";
/// See [`HARMONIC_IMPROPER_SIGNED`].
pub(crate) const HARMONIC_IMPROPER_ABS: &str = "k*(abs(theta)-theta0)^2";

/// Read a [`ForceField`] from an OpenMM force-field XML file.
pub fn read_openmm_forcefield(path: &str) -> Result<ForceField, String> {
    OpenmmXmlReader::new().read(path)
}

/// Read a [`ForceField`] from OpenMM force-field XML text — the inverse of
/// [`write_openmm_forcefield_str`](crate::io::write_openmm_forcefield_str).
pub fn read_openmm_forcefield_str(text: &str) -> Result<ForceField, String> {
    OpenmmXmlReader::new().read_str(text)
}

/// Reader for OpenMM `<ForceField>` XML (nm, kJ/mol, radians).
///
/// Parses an OpenMM `<ForceField>` (nm, kJ/mol, radians, OpenMM's factors)
/// into a molrs [`ForceField`] in the force-field IR, whose definitions follow
/// LAMMPS's (`real`: Å, kcal/mol, degrees for angle-valued parameters, e; no
/// hidden ½). Every section OpenMM's `app.ForceField` builds a force from is
/// either read exactly or refused by name; nothing that carries energy is
/// skipped.
///
/// ```xml
/// <ForceField name="OPLS-AA" combining_rule="geometric">
///   <AtomTypes>
///     <Type name="opls_001" class="opls_001" element="C" mass="12.011"/>
///   </AtomTypes>
///   <HarmonicBondForce>
///     <Bond class1="OW" class2="HW" length="0.09572" k="502080.0"/>   <!-- nm, kJ/mol/nm² -->
///   </HarmonicBondForce>
///   <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
///     <Atom type="opls_001" charge="0.5" sigma="0.375" epsilon="0.43932"/> <!-- e, nm, kJ/mol -->
///   </NonbondedForce>
/// </ForceField>
/// ```
///
/// # Naming vocabularies
///
/// Bonded forces key on the **class** attribute (`class1`, …) or the **type**
/// attribute (`type1`, …) of a row; the label is stored as written, and an
/// empty attribute (`class2=""`) is OpenMM's wildcard, stored as `""`. A row
/// naming neither is ignored by OpenMM and refused here. Nonbonded rows key on
/// types: a `<NonbondedForce>` / `<LennardJonesForce>` `<Atom class=…>` row
/// applies to every `<AtomTypes>` type of that class, as in OpenMM.
///
/// # Sections
///
/// | OpenMM | IR | Conversion |
/// |---|---|---|
/// | `<HarmonicBondForce><Bond length k>` | `bond harmonic` | `r0 = 10·length`; OpenMM's ½k → `k = k/(2·418.4)` |
/// | `<HarmonicAngleForce><Angle angle k>` | `angle harmonic` | `theta0` = angle in degrees; `k = k/(2·4.184)` |
/// | `<AmoebaUreyBradleyForce><UreyBradley k d>` | `angle charmm`, joined with the angle row of the same three labels (either direction) | OpenMM adds a bond of force constant `2k`, so `k` is un-halved: `k_ub = k/418.4`, `r_ub = 10·d`; with no angle row, `k = 0` |
/// | `<PeriodicTorsionForce><Proper k_m periodicity_m phase_m>` | `dihedral periodic` | `k_m/4.184`, phases in degrees |
/// | `<PeriodicTorsionForce><Proper c0..c3>` (CL&P / foyer spelling) | `dihedral opls` | `k_n = c_{n−1}/4.184` |
/// | `<PeriodicTorsionForce><Improper …>` | `improper periodic` (one term) | stored in the order OpenMM prices |
/// | `<RBTorsionForce><Proper c0..c5>` | `dihedral multi/harmonic` (`c5 = 0`) or `dihedral nharmonic` (N = 6) | `A_{n+1} = (−1)ⁿ C_n / 4.184` (`cos(φ − 180°) = −cos φ`), exact, constant included |
/// | `<CustomTorsionForce energy="k*(theta-theta0)^2">` `<Improper>` | `improper harmonic` | `k/4.184`; OpenMM's θ is signed, LAMMPS's χ = \|φ\|, which agree at `theta0 = 0` only — another `theta0` is refused |
/// | `<CustomTorsionForce energy="k*(abs(theta)-theta0)^2">` `<Improper>` | `improper harmonic` | `k/4.184`, `chi0` = theta0 in degrees (the writer's form for `chi0 ≠ 0`) |
/// | `<CMAPTorsionForce><Map>` + `<Torsion map>` | `cmap charmm` | element `(i, j)` of OpenMM's map (`energy[i + N·j]` at φ = 2πi/N, ψ = 2πj/N) is molrs `grid[(i + N/2) mod N][(j + N/2) mod N]` (φ-major from −180°), ÷ 4.184; N must be even |
/// | `<NonbondedForce coulomb14scale lj14scale><Atom charge sigma epsilon>` | `pair lj/cut` (`mixing` = the root's foyer `combining_rule`, else OpenMM's `arithmetic`) + `pair coul/cut`; `charge` on `atom full` | `sigma` × 10, `epsilon` ÷ 4.184; `special_bonds` `[0, 0, scale]` |
/// | `<LennardJonesForce lj14scale><Atom sigma epsilon [sigma14 epsilon14]>` | `pair lj/charmm` (`mixing arithmetic`) + `pair coul/charmm`; `charge` from the `<NonbondedForce>` beside it, whose `epsilon` must be 0 | as above; `sigma14`/`epsilon14` absent → LAMMPS's two-number `pair_coeff` |
/// | `<LennardJonesForce><NBFixPair sigma epsilon>` | `pair lj/charmm` cross row | as above; OpenMM prices a 1-4 NBFIX pair with the NBFIX row too, which is LAMMPS's cross row without `epsilon14` / `sigma14` |
///
/// Every Coulomb style takes OpenMM's constant, `ONE_4PI_EPS0` =
/// 138.93545764438198 kJ·nm/(mol·e²) = [`OPENMM_ONE_4PI_EPS0`](crate::core::constants::OPENMM_ONE_4PI_EPS0) in kcal·Å/(mol·e²)
/// (LAMMPS `real`'s `qqr2e` is 332.06371, 9.9·10⁻⁹ below it). OpenMM's cutoffs
/// and switching are `createSystem` arguments, not file values, so no style
/// here has a `cutoff` (and `lj/charmm` / `coul/charmm` no `inner`): the
/// caller states them, as for `NoCutoff` a cutoff beyond every pair.
///
/// # 1-4 pairs
///
/// OpenMM prices every bond-graph 1-4 pair once: `<NonbondedForce>` at
/// `coulomb14scale` (Coulomb) and `lj14scale` (its own Lennard-Jones), and a
/// `<LennardJonesForce>` at its `lj14scale` with the `sigma14`/`epsilon14`
/// of the two types mixed Lorentz-Berthelot (the NBFIX row for an NBFIX pair).
/// That is `special_bonds lj [0, 0, lj14scale] coul [0, 0, coulomb14scale]`
/// exactly when no type's `sigma14`/`epsilon14` differs from its
/// `sigma`/`epsilon`. Otherwise the 1-4 Lennard-Jones is not LAMMPS's
/// `special_bonds` pricing (which takes the regular `epsilon`/`sigma`): the
/// field keeps `epsilon14`/`sigma14` on `lj/charmm` and its `special_bonds`,
/// and declares the style param `one_four = "epsilon14"`
/// ([`ONE_FOUR_EPSILON14`]):
/// a frame's 1-4 pairs are OpenMM's exceptions, which the IR holds as per-pair
/// override rows on `pairs`.
/// [`ForceField::materialize_one_four`](crate::ff::forcefield::ForceField::materialize_one_four)
/// writes them;
/// compiling such a field for a frame whose 1-4 pairs lack them is refused.
///
/// # Refused
///
/// `<Script>` / `<InitializationScript>` (code), every `Custom*Force` other
/// than the harmonic improper above, `<Proper>` rows of a
/// `<CustomTorsionForce>`, `<Improper>` rows of a `<RBTorsionForce>`,
/// `ordering="smirnoff"`, a multi-term periodic improper, an odd CMAP size, a
/// Urey–Bradley row with a wildcard, an `<NBFixPair>` of one type with itself,
/// a `<NonbondedForce>` with non-zero `epsilon` beside a `<LennardJonesForce>`
/// (two Lennard-Jones forces), and every force OpenMM's `app.ForceField`
/// does not build from this schema (AMOEBA multipoles, GBSA, Drude, …). The
/// residue templates (`<Residues>`, `<Patches>`) and `<Info>` carry no
/// parameters and are skipped.
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenmmXmlReader;

impl OpenmmXmlReader {
    pub fn new() -> Self {
        Self
    }

    /// Read an OPLS-AA typifier from OpenMM force-field XML text: the typing
    /// annotations Foyer adds to each `<AtomTypes><Type>` (`def` SMARTS,
    /// `overrides`, `priority`, `layer`) and the force field of the same
    /// document — a caller's OPLS-AA / CL&P file, where
    /// [`OplsAaTypifier::oplsaa`] is the shipped set.
    ///
    /// # Errors
    ///
    /// Either half failing to read: a malformed document, a `<Type>` without
    /// `name` / `class`, a non-integer `priority` / `layer`, an `overrides`
    /// naming an undeclared type or forming a cycle, or what
    /// [`read_str`](ForceFieldReader::read_str) refuses.
    pub fn read_opls_typifier_str(&self, text: &str) -> Result<OplsAaTypifier, String> {
        let meta = super::opls_typing::read_typing_metadata(text)?;
        let ff = self.read_str(text)?;
        Ok(OplsAaTypifier::new(meta, ff))
    }
}

impl ForceFieldReader for OpenmmXmlReader {
    fn read_str(&self, text: &str) -> Result<ForceField, String> {
        let doc = roxmltree::Document::parse(text)
            .map_err(|e| format!("OpenMM XML parse error: {}", e))?;
        let root = doc.root_element();
        if root.tag_name().name() != "ForceField" {
            return Err(format!(
                "root element must be <ForceField>, got <{}>",
                root.tag_name().name()
            ));
        }

        // molpy / foyer convention: missing ``name`` → ``"Unknown"``.
        let mut ff = ForceField::new(root.attribute("name").unwrap_or("Unknown"));
        let mut raw = Raw::default();

        for sec in root.children().filter(Node::is_element) {
            match sec.tag_name().name() {
                "AtomTypes" => {
                    for t in sec.children().filter(Node::is_element) {
                        require_tag(&t, "Type")?;
                        let name = require_str(&t, "name")?.to_owned();
                        let mass = opt_f64(&t, "mass")?.unwrap_or(0.0);
                        raw.atom_rows.push(AtomTypeRow {
                            name,
                            mass,
                            class: t.attribute("class").map(str::to_owned),
                            element: t.attribute("element").map(str::to_owned),
                            def: t.attribute("def").map(str::to_owned),
                            desc: t.attribute("desc").map(str::to_owned),
                            doi: t.attribute("doi").map(str::to_owned),
                            overrides: t.attribute("overrides").map(str::to_owned),
                        });
                    }
                }
                "HarmonicBondForce" => parse_bonds(&mut ff, &sec)?,
                "HarmonicAngleForce" => parse_angles(&mut raw, &sec)?,
                "AmoebaUreyBradleyForce" => parse_urey_bradley(&mut raw, &sec)?,
                "RBTorsionForce" => parse_rb_torsions(&mut ff, &sec)?,
                // OpenMM k{m}/periodicity{m}/phase{m}, or CL&P c0..c3 — per row.
                "PeriodicTorsionForce" => parse_periodic_torsions(&mut ff, &sec)?,
                "CustomTorsionForce" => parse_custom_torsions(&mut ff, &sec)?,
                "CMAPTorsionForce" => parse_cmap(&mut ff, &sec)?,
                "NonbondedForce" => parse_nonbonded(&mut raw, &sec)?,
                "LennardJonesForce" => parse_lennard_jones(&mut raw, &sec)?,
                // Residue templates and provenance carry no parameters.
                "Residues" | "Patches" | "Info" => {}
                other => {
                    return Err(format!(
                        "<{other}>: no force-field IR form (molrs reads OpenMM's \
                         HarmonicBond, HarmonicAngle, AmoebaUreyBradley, PeriodicTorsion, \
                         RBTorsion, CMAPTorsion, Nonbonded and LennardJones forces, and \
                         the harmonic-improper CustomTorsionForce)"
                    ));
                }
            }
        }

        build_angles(&mut ff, &raw)?;
        // `combining_rule` is a foyer attribute (OpenMM's NonbondedForce
        // always mixes Lorentz-Berthelot; foyer patches the system for a
        // geometric rule): OPLS-AA mixes σ geometrically.
        let combining_rule = root.attribute("combining_rule");
        build_nonbonded(&mut ff, &raw, combining_rule)?;
        ensure_class_wildcards(&mut ff, &raw.atom_rows)?;
        Ok(ff)
    }
}

/// What is collected over the whole document before it can be defined.
#[derive(Default)]
struct Raw {
    atom_rows: Vec<AtomTypeRow>,
    angles: Vec<AngleRow>,
    urey_bradley: Vec<UreyBradleyRow>,
    /// `(coulomb14scale, lj14scale)` of the `<NonbondedForce>` tags.
    nonbonded_scales: Option<(f64, f64)>,
    nonbonded: Vec<NonbondedRow>,
    /// `lj14scale` of the `<LennardJonesForce>` tags.
    lj_scale: Option<f64>,
    lj_atoms: Vec<LjRow>,
    nbfix: Vec<NbfixRow>,
}

/// One `<Type>` row of `<AtomTypes>` (mass + string metadata).
struct AtomTypeRow {
    name: String,
    mass: f64,
    class: Option<String>,
    element: Option<String>,
    def: Option<String>,
    desc: Option<String>,
    doi: Option<String>,
    overrides: Option<String>,
}

/// A nonbonded row's key: `type="…"` or `class="…"`.
#[derive(Clone)]
enum AtomKey {
    Type(String),
    Class(String),
}

impl AtomKey {
    fn of(node: &Node) -> Result<Self, String> {
        match (node.attribute("type"), node.attribute("class")) {
            (Some(t), None) => Ok(Self::Type(t.to_owned())),
            (None, Some(c)) => Ok(Self::Class(c.to_owned())),
            _ => Err(format!(
                "<{}> needs exactly one of `type` and `class`",
                node.tag_name().name()
            )),
        }
    }

    /// The atom types this key names, in `<AtomTypes>` order.
    fn types(&self, atom_rows: &[AtomTypeRow]) -> Result<Vec<String>, String> {
        match self {
            Self::Type(t) => Ok(vec![t.clone()]),
            Self::Class(c) => {
                let types: Vec<String> = atom_rows
                    .iter()
                    .filter(|r| r.class.as_deref() == Some(c.as_str()))
                    .map(|r| r.name.clone())
                    .collect();
                if types.is_empty() {
                    return Err(format!("class=\"{c}\" names no <AtomTypes> type"));
                }
                Ok(types)
            }
        }
    }
}

/// One `<Atom>` row of `<NonbondedForce>`, in molrs units.
struct NonbondedRow {
    key: AtomKey,
    charge: Option<f64>,
    sigma: f64,
    epsilon: f64,
}

/// One `<Atom>` row of `<LennardJonesForce>`, in molrs units.
struct LjRow {
    key: AtomKey,
    sigma: f64,
    epsilon: f64,
    sigma14: Option<f64>,
    epsilon14: Option<f64>,
}

/// One `<NBFixPair>`, in molrs units.
struct NbfixRow {
    keys: [AtomKey; 2],
    sigma: f64,
    epsilon: f64,
}

/// One `<Angle>` row, in molrs units.
struct AngleRow {
    ends: [String; 3],
    k: f64,
    theta0: f64,
}

/// One `<UreyBradley>` row, in molrs units.
struct UreyBradleyRow {
    ends: [String; 3],
    k_ub: f64,
    r_ub: f64,
}

/// Build the atom style (`full`: mass + charge per type) and the pair styles.
///
/// - `<NonbondedForce>` alone: `lj/cut` (per-type ε/σ, `mixing` from
///   `combining_rule`, else OpenMM's `arithmetic`) + `coul/cut`;
/// - with `<LennardJonesForce>`: `lj/charmm` (per-type ε/σ/ε₁₄/σ₁₄, NBFIX
///   cross rows, `arithmetic`) + `coul/charmm`, the `<NonbondedForce>`
///   supplying charges only.
///
/// String metadata on each atom type matches molpy's reader contract:
/// ``type_`` (type name), ``class``, ``element``, ``smarts`` (the XML
/// ``def``), ``desc``, ``doi``, ``overrides``.
fn build_nonbonded(
    ff: &mut ForceField,
    raw: &Raw,
    combining_rule: Option<&str>,
) -> Result<(), String> {
    let atom_rows = &raw.atom_rows;
    // Per-type charge from the NonbondedForce rows (a class row covers every
    // type of its class).
    let mut charge_of: HashMap<String, f64> = HashMap::new();
    let mut nb_rows: Vec<(String, f64, f64)> = Vec::new();
    for r in &raw.nonbonded {
        for ty in r.key.types(atom_rows)? {
            if let Some(q) = r.charge {
                if let Some(&other) = charge_of.get(&ty)
                    && other != q
                {
                    return Err(format!(
                        "<NonbondedForce> rows for type \"{ty}\" give different charges"
                    ));
                }
                charge_of.insert(ty.clone(), q);
            }
            nb_rows.push((ty, r.sigma, r.epsilon));
        }
    }

    if !atom_rows.is_empty() {
        let atom = ff
            .def_style("atom", "full", Params::new())
            .map_err(|e| e.to_string())?;
        for row in atom_rows {
            let mut params = Params::from_pairs(&[("mass", row.mass)]);
            if let Some(&q) = charge_of.get(&row.name) {
                params.set("charge", q);
            }
            params.set_str("type_", &row.name);
            let strings = [
                ("class", &row.class),
                ("element", &row.element),
                ("smarts", &row.def),
                ("desc", &row.desc),
                ("doi", &row.doi),
                ("overrides", &row.overrides),
            ];
            for (key, value) in strings {
                if let Some(value) = value {
                    params.set_str(key, value);
                }
            }
            atom.def_type(&row.name, &[], params)
                .map_err(|e| e.to_string())?;
        }
    }

    let coulomb = |ff: &mut ForceField, name: &str| -> Result<(), String> {
        ff.def_style(
            "pair",
            name,
            Params::from_pairs(&[
                ("coulomb", crate::core::constants::openmm_coulomb_real()),
                ("dielectric", VACUUM_DIELECTRIC),
            ]),
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
    };
    let (coul14, nb_lj14) = raw.nonbonded_scales.unwrap_or((0.0, 0.0));

    let Some(lj14) = raw.lj_scale else {
        if raw.nonbonded_scales.is_none() {
            return Ok(());
        }
        let rule = combining_rule.map_or(Ok(CombiningRule::Arithmetic), foyer_rule)?;
        let mut lj_params = Params::new();
        lj_params.set_str("mixing", rule.name());
        let lj = ff
            .def_style("pair", "lj/cut", lj_params)
            .map_err(|e| e.to_string())?;
        for (ty, sigma, epsilon) in &nb_rows {
            lj.def_type(
                ty,
                &[ty],
                Params::from_pairs(&[("epsilon", *epsilon), ("sigma", *sigma)]),
            )
            .map_err(|e| e.to_string())?;
        }
        coulomb(ff, "coul/cut")?;
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.0, nb_lj14],
            coul: [0.0, 0.0, coul14],
        });
        return Ok(());
    };

    // <LennardJonesForce>: OpenMM computes the NonbondedForce's own LJ as
    // well, so a non-zero epsilon there is a second Lennard-Jones force.
    if let Some((ty, _, eps)) = nb_rows.iter().find(|(_, _, e)| *e != 0.0) {
        return Err(format!(
            "<NonbondedForce> type \"{ty}\" has epsilon = {} kJ/mol beside a \
             <LennardJonesForce>: OpenMM prices both Lennard-Jones forces, which no \
             single pair style is",
            eps * KCAL_TO_KJ.get()
        ));
    }
    if let Some(rule) = combining_rule
        && foyer_rule(rule)? != CombiningRule::Arithmetic
    {
        return Err(format!(
            "<ForceField combining_rule=\"{rule}\">: <LennardJonesForce> mixes \
             Lorentz-Berthelot (arithmetic) by construction"
        ));
    }
    let mut lj_params = Params::new();
    lj_params.set_str("mixing", "arithmetic");
    let lj = ff
        .def_style("pair", "lj/charmm", lj_params)
        .map_err(|e| e.to_string())?;
    for r in &raw.lj_atoms {
        let mut p = Params::from_pairs(&[("epsilon", r.epsilon), ("sigma", r.sigma)]);
        // OpenMM takes each of sigma14 / epsilon14 alone; LAMMPS's
        // `pair_coeff` gives both or neither.
        match (r.epsilon14, r.sigma14) {
            (None, None) => {}
            (e14, s14) => {
                p.set("epsilon14", e14.unwrap_or(r.epsilon));
                p.set("sigma14", s14.unwrap_or(r.sigma));
            }
        }
        for ty in r.key.types(atom_rows)? {
            lj.def_type(&ty, &[&ty], p.clone())
                .map_err(|e| e.to_string())?;
        }
    }
    for r in &raw.nbfix {
        let (a, b) = (r.keys[0].types(atom_rows)?, r.keys[1].types(atom_rows)?);
        for ta in &a {
            for tb in &b {
                if ta == tb {
                    return Err(format!(
                        "<NBFixPair> of type \"{ta}\" with itself: its self pair would \
                         differ from the epsilon/sigma it mixes with, which no pair \
                         style row holds"
                    ));
                }
                lj.def_type(
                    TypeName::pair(ta, tb)?.as_str(),
                    &[ta, tb],
                    Params::from_pairs(&[("epsilon", r.epsilon), ("sigma", r.sigma)]),
                )
                .map_err(|e| e.to_string())?;
            }
        }
    }
    let own_one_four = lj.type_rows().iter().any(|(_, _, p)| has_own_one_four(p));
    if own_one_four && lj14 != 0.0 {
        lj.set_str_param(ONE_FOUR, ONE_FOUR_EPSILON14);
    }
    // The Coulomb comes from a <NonbondedForce>; without one OpenMM prices
    // no charges.
    if raw.nonbonded_scales.is_some() {
        coulomb(ff, "coul/charmm")?;
    }
    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, lj14],
        coul: [0.0, 0.0, coul14],
    });
    Ok(())
}

/// Class-only bond/angle endpoints need a placeholder AtomType with
/// ``type_="*"`` and ``class=<class>`` so TypeClassIndex / class-keyed
/// matching can resolve them (molpy XML reader parity).
///
/// Placeholders are inserted in ascending class-name order (byte-wise `str`
/// ordering), so the resulting atom-type order is identical on every read.
fn ensure_class_wildcards(ff: &mut ForceField, atom_rows: &[AtomTypeRow]) -> Result<(), String> {
    let real_names: HashSet<String> = atom_rows.iter().map(|r| r.name.clone()).collect();
    let mut endpoint_classes: BTreeSet<String> = BTreeSet::new();

    // Classes declared on AtomTypes that are not themselves type names.
    for row in atom_rows {
        if let Some(ref c) = row.class
            && !real_names.contains(c)
        {
            endpoint_classes.insert(c.clone());
        }
    }
    // Bonded endpoint labels that aren't real atom-type names (class-keyed
    // rows). The wildcard `""` is no class.
    let mut add = |part: &String| {
        if !part.is_empty() && !real_names.contains(part) {
            endpoint_classes.insert(part.clone());
        }
    };
    for bt in ff.get_bondtypes() {
        for part in [&bt.itom, &bt.jtom] {
            add(part);
        }
    }
    for at in ff.get_angletypes() {
        for part in [&at.itom, &at.jtom, &at.ktom] {
            add(part);
        }
    }
    for dt in ff.get_dihedraltypes() {
        for part in [&dt.itom, &dt.jtom, &dt.ktom, &dt.ltom] {
            add(part);
        }
    }

    if endpoint_classes.is_empty() {
        return Ok(());
    }

    // Prefer the existing "full" atom style; create one only if needed.
    if ff.get_style("atom", "full").is_none() && ff.get_styles("atom").is_empty() {
        ff.def_style("atom", "full", Params::new())
            .map_err(|e| e.to_string())?;
    }
    let style_name = if ff.get_style("atom", "full").is_some() {
        "full".to_owned()
    } else {
        ff.get_styles("atom")
            .first()
            .map(|s| s.name().to_owned())
            .unwrap_or_else(|| "full".to_owned())
    };
    let atom = ff
        .def_style("atom", &style_name, Params::new())
        .map_err(|e| e.to_string())?;

    for class_name in endpoint_classes {
        if atom.get_atomtype(&class_name).is_some() {
            continue;
        }
        atom.def_type(&class_name, &[], Params::new())
            .map_err(|e| e.to_string())?;
        atom.set_type_str_param(&class_name, "type_", "*");
        atom.set_type_str_param(&class_name, "class", &class_name);
    }
    Ok(())
}

/// Endpoint `n` of a bonded row: its `class{n}` or `type{n}` attribute, as
/// written (`""` is OpenMM's wildcard). OpenMM ignores a row naming neither;
/// it is refused here.
fn class_or_type<'a>(node: &'a Node, n: usize) -> Result<&'a str, String> {
    let class_key = format!("class{n}");
    let type_key = format!("type{n}");
    node.attribute(class_key.as_str())
        .or_else(|| node.attribute(type_key.as_str()))
        .ok_or_else(|| {
            format!(
                "<{}> names no `class{n}` / `type{n}` (OpenMM ignores such a row)",
                node.tag_name().name()
            )
        })
}

fn endpoints<'a, const N: usize>(node: &'a Node) -> Result<[&'a str; N], String> {
    let mut out = [""; N];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = class_or_type(node, i + 1)?;
    }
    Ok(out)
}

fn is_wildcard(label: &str) -> bool {
    label.is_empty()
}

fn parse_bonds(ff: &mut ForceField, sec: &Node) -> Result<(), String> {
    let style = ff
        .def_style("bond", "harmonic", Params::new())
        .map_err(|e| e.to_string())?;
    for b in sec.children().filter(Node::is_element) {
        require_tag(&b, "Bond")?;
        let ends = endpoints::<2>(&b)?;
        let r0 = require_f64(&b, "length")? * NM_TO_ANGSTROM.get();
        // OpenMM ½k in kJ/mol/nm² → LAMMPS K = k/2 in kcal/mol/Å².
        let k = require_f64(&b, "k")? / KCAL_ANGSTROM2_TO_KJ_NM2.get() / 2.0;
        style
            .def_type(
                TypeName::join(&ends)?.as_str(),
                &ends,
                Params::from_pairs(&[("k", k), ("r0", r0)]),
            )
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn parse_angles(raw: &mut Raw, sec: &Node) -> Result<(), String> {
    for a in sec.children().filter(Node::is_element) {
        require_tag(&a, "Angle")?;
        let ends = endpoints::<3>(&a)?.map(str::to_owned);
        raw.angles.push(AngleRow {
            ends,
            // OpenMM ½k in kJ/mol/rad² → LAMMPS K = k/2 in kcal/mol/rad².
            k: require_f64(&a, "k")? / KCAL_TO_KJ.get() / 2.0,
            theta0: require_f64(&a, "angle")?.to_degrees(),
        });
    }
    Ok(())
}

/// `<AmoebaUreyBradleyForce><UreyBradley class1..3|type1..3 k d>`. OpenMM's
/// generator adds, per matching angle, a `HarmonicBondForce` term between the
/// end atoms with force constant `2k` (`AmoebaUreyBradleyForceBuilder`), so
/// the energy is `k(r₁₃ − d)²`: `k` is already LAMMPS's un-halved `K_ub`.
fn parse_urey_bradley(raw: &mut Raw, sec: &Node) -> Result<(), String> {
    for u in sec.children().filter(Node::is_element) {
        require_tag(&u, "UreyBradley")?;
        let ends = endpoints::<3>(&u)?;
        if ends.iter().any(|e| is_wildcard(e)) {
            return Err(format!(
                "<UreyBradley> {}: a wildcard Urey-Bradley row has no angle row to join",
                ends.join("-")
            ));
        }
        raw.urey_bradley.push(UreyBradleyRow {
            ends: ends.map(str::to_owned),
            k_ub: require_f64(&u, "k")? / KCAL_ANGSTROM2_TO_KJ_NM2.get(),
            r_ub: require_f64(&u, "d")? * NM_TO_ANGSTROM.get(),
        });
    }
    Ok(())
}

/// Define the angle rows: those a Urey–Bradley row names (in either
/// direction) as `angle charmm`, the others as `angle harmonic`, in file
/// order; a Urey–Bradley row without an angle row is an `angle charmm` row
/// with `k = 0`.
fn build_angles(ff: &mut ForceField, raw: &Raw) -> Result<(), String> {
    let key = |e: &[String; 3]| -> [String; 3] {
        let rev = [e[2].clone(), e[1].clone(), e[0].clone()];
        if rev < *e { rev } else { e.clone() }
    };
    let mut ub: BTreeMap<[String; 3], (f64, f64)> = BTreeMap::new();
    let mut ub_order: Vec<[String; 3]> = Vec::new();
    for u in &raw.urey_bradley {
        let k = key(&u.ends);
        match ub.get(&k) {
            Some(&(k_ub, r_ub)) if (k_ub, r_ub) != (u.k_ub, u.r_ub) => {
                return Err(format!(
                    "<UreyBradley> {}: two rows with different k / d",
                    u.ends.join("-")
                ));
            }
            Some(_) => {}
            None => {
                ub.insert(k.clone(), (u.k_ub, u.r_ub));
                ub_order.push(u.ends.clone());
            }
        }
    }
    if !raw.angles.is_empty() && raw.angles.iter().any(|a| !ub.contains_key(&key(&a.ends))) {
        ff.def_style("angle", "harmonic", Params::new())
            .map_err(|e| e.to_string())?;
    }
    let mut joined: HashSet<[String; 3]> = HashSet::new();
    for a in &raw.angles {
        let ends: Vec<&str> = a.ends.iter().map(String::as_str).collect();
        let name = TypeName::join(&ends)?;
        let (style, params) = match ub.get(&key(&a.ends)) {
            Some(&(k_ub, r_ub)) => {
                joined.insert(key(&a.ends));
                (
                    "charmm",
                    Params::from_pairs(&[
                        ("k", a.k),
                        ("theta0", a.theta0),
                        ("k_ub", k_ub),
                        ("r_ub", r_ub),
                    ]),
                )
            }
            None => (
                "harmonic",
                Params::from_pairs(&[("k", a.k), ("theta0", a.theta0)]),
            ),
        };
        ff.def_style("angle", style, Params::new())
            .map_err(|e| e.to_string())?
            .def_type(name.as_str(), &ends, params)
            .map_err(|e| e.to_string())?;
    }
    for ends in ub_order {
        if joined.contains(&key(&ends)) {
            continue;
        }
        let (k_ub, r_ub) = ub[&key(&ends)];
        let e: Vec<&str> = ends.iter().map(String::as_str).collect();
        ff.def_style("angle", "charmm", Params::new())
            .map_err(|e| e.to_string())?
            .def_type(
                TypeName::join(&e)?.as_str(),
                &e,
                Params::from_pairs(&[("k", 0.0), ("theta0", 0.0), ("k_ub", k_ub), ("r_ub", r_ub)]),
            )
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// `<RBTorsionForce>`: OpenMM's `Σₙ₌₀⁵ Cₙ cosⁿ(φ − 180°)` is `Σ Aₙ₊₁ cosⁿφ`
/// with `Aₙ₊₁ = (−1)ⁿ Cₙ` — LAMMPS `dihedral multi/harmonic` when `C5 = 0`,
/// `dihedral nharmonic` (N = 6) otherwise, exactly, constant included.
/// `<Improper>` rows (OpenMM's RB improper) have no IR form.
fn parse_rb_torsions(ff: &mut ForceField, sec: &Node) -> Result<(), String> {
    for d in sec.children().filter(Node::is_element) {
        let tag = d.tag_name().name();
        let ends = endpoints::<4>(&d)?;
        if tag == "Improper" {
            return Err(format!(
                "RBTorsionForce <Improper> {}: a Ryckaert-Bellemans improper has no IR form \
                 (no improper style is a cosine polynomial)",
                ends.join("-")
            ));
        }
        require_tag(&d, "Proper")?;
        let mut c = [0.0; 6];
        for (n, slot) in c.iter_mut().enumerate() {
            *slot = require_f64(&d, &format!("c{n}"))? / KCAL_TO_KJ.get();
        }
        let (style, params) = rb_polynomial(c);
        ff.def_style("dihedral", style, Params::new())
            .map_err(|e| e.to_string())?
            .def_type(TypeName::join(&ends)?.as_str(), &ends, params)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// `<PeriodicTorsionForce>` children, in either of the two spellings found in the
/// wild, decided per row:
///
/// - **OpenMM's own** — `k{m}`, `periodicity{m}`, `phase{m}` for `m = 1, 2, …`,
///   `E = Σ k_m [1 + cos(n_m φ − γ_m)]` in kJ/mol and radians. That is molrs's
///   `dihedral periodic` form term for term, so only the energy unit changes. A
///   `<Proper>` goes to `dihedral periodic`; an `<Improper>` to
///   `improper periodic`, whose kernel holds one term.
/// - **CL&P / foyer** — OPLS Fourier `c0..c3` in kJ/mol under this tag, read as
///   `dihedral opls` `k1..k4` in kcal/mol.
///
/// A row carrying neither spelling, or both, is an error. An `<Improper>` is
/// stored in the order whose dihedral OpenMM prices ([`improper_order`]).
fn parse_periodic_torsions(ff: &mut ForceField, sec: &Node) -> Result<(), String> {
    let section = sec.tag_name().name();
    let ordering = sec.attribute("ordering").unwrap_or("default");
    for d in sec.children().filter(Node::is_element) {
        let tag = d.tag_name().name();
        if tag != "Improper" && tag != "Proper" {
            return Err(format!(
                "{section}: unexpected child <{tag}> (expected <Proper> or <Improper>)"
            ));
        }
        let mut classes = endpoints::<4>(&d)?;
        if tag == "Improper" {
            classes = improper_order(classes, ordering, section)?;
        }
        let label = classes.join("-");
        let terms = periodic_terms(&d)?;
        let clp = (0..4).any(|i| d.attribute(format!("c{i}").as_str()).is_some());
        let params = match (terms.is_empty(), clp) {
            (false, true) => {
                return Err(format!(
                    "PeriodicTorsionForce <{tag}> {label}: carries both OpenMM \
                     k1/periodicity1/phase1 and CL&P c0..c3 terms"
                ));
            }
            (true, false) => {
                return Err(format!(
                    "PeriodicTorsionForce <{tag}> {label}: carries neither OpenMM \
                     k1/periodicity1/phase1 nor CL&P c0..c3 terms"
                ));
            }
            (true, true) if tag == "Improper" => {
                return Err(format!(
                    "PeriodicTorsionForce <Improper> {label}: CL&P c0..c3 is a proper-torsion form"
                ));
            }
            (true, true) => {
                let c = |i: usize| -> Result<f64, String> {
                    Ok(opt_f64(&d, &format!("c{i}"))?.unwrap_or(0.0) / KCAL_TO_KJ.get())
                };
                let (f1, f2, f3, f4) = (c(0)?, c(1)?, c(2)?, c(3)?);
                (
                    "dihedral",
                    "opls",
                    Params::from_pairs(&[("k1", f1), ("k2", f2), ("k3", f3), ("k4", f4)]),
                )
            }
            (false, false) if tag == "Improper" => {
                if terms.len() != 1 {
                    return Err(format!(
                        "PeriodicTorsionForce <Improper> {label}: {} terms, but the \
                         periodic improper holds one",
                        terms.len()
                    ));
                }
                let (k, n, phase) = terms[0];
                (
                    "improper",
                    "periodic",
                    Params::from_pairs(&[("k", k), ("periodicity", n), ("phase", phase)]),
                )
            }
            (false, false) => {
                let mut pairs: Vec<(String, f64)> = Vec::with_capacity(3 * terms.len());
                for (m, (k, n, phase)) in terms.iter().enumerate() {
                    let m = m + 1;
                    pairs.push((format!("k{m}"), *k));
                    pairs.push((format!("periodicity{m}"), *n));
                    pairs.push((format!("phase{m}"), *phase));
                }
                let refs: Vec<(&str, f64)> = pairs.iter().map(|(k, v)| (k.as_str(), *v)).collect();
                ("dihedral", "periodic", Params::from_pairs(&refs))
            }
        };
        let (category, style_name, params) = params;
        ff.def_style(category, style_name, Params::new())
            .map_err(|e| e.to_string())?
            .def_type(TypeName::join(&classes)?.as_str(), &classes, params)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// `<CustomTorsionForce energy=…>`: the harmonic improper of OpenMM's CHARMM
/// ports, recognised by its exact expression (whitespace aside) with per-torsion
/// parameters `k`, `theta0` and no global parameter, is `improper harmonic`.
/// OpenMM's `CustomTorsion` default ordering is `charmm`.
fn parse_custom_torsions(ff: &mut ForceField, sec: &Node) -> Result<(), String> {
    let energy: String = require_str(sec, "energy")?
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let signed = match energy.as_str() {
        HARMONIC_IMPROPER_SIGNED => true,
        HARMONIC_IMPROPER_ABS => false,
        _ => {
            return Err(format!(
                "<CustomTorsionForce energy=\"{energy}\">: no IR form (molrs reads the \
                 harmonic improper \"{HARMONIC_IMPROPER_SIGNED}\" and \
                 \"{HARMONIC_IMPROPER_ABS}\")"
            ));
        }
    };
    let ordering = sec.attribute("ordering").unwrap_or("charmm");
    let mut per_torsion: Vec<&str> = Vec::new();
    for child in sec.children().filter(Node::is_element) {
        match child.tag_name().name() {
            "PerTorsionParameter" => per_torsion.push(
                child
                    .attribute("name")
                    .ok_or("<PerTorsionParameter> missing required attribute `name`")?,
            ),
            "GlobalParameter" => {
                return Err(format!(
                    "<CustomTorsionForce energy=\"{energy}\">: global parameter `{}` has no \
                     IR form",
                    require_str(&child, "name")?
                ));
            }
            _ => {}
        }
    }
    per_torsion.sort_unstable();
    if per_torsion != ["k", "theta0"] {
        return Err(format!(
            "<CustomTorsionForce energy=\"{energy}\">: per-torsion parameters \
             {per_torsion:?}, expected k and theta0"
        ));
    }
    for d in sec.children().filter(Node::is_element) {
        match d.tag_name().name() {
            "PerTorsionParameter" => continue,
            "Improper" => {}
            "Proper" => {
                return Err(format!(
                    "<CustomTorsionForce energy=\"{energy}\"> <Proper> {}: a harmonic \
                     proper dihedral (LAMMPS `dihedral quadratic`) has no IR form",
                    endpoints::<4>(&d)?.join("-")
                ));
            }
            other => {
                return Err(format!("<CustomTorsionForce>: unexpected child <{other}>"));
            }
        }
        let classes = improper_order(endpoints::<4>(&d)?, ordering, "CustomTorsionForce")?;
        let label = classes.join("-");
        let k = require_f64(&d, "k")? / KCAL_TO_KJ.get();
        let theta0 = require_f64(&d, "theta0")?;
        // OpenMM's θ is the signed dihedral; LAMMPS's χ is |φ|. Their
        // energies agree for every geometry only at θ0 = 0 (θ² = |θ|²).
        if signed && theta0 != 0.0 {
            return Err(format!(
                "<CustomTorsionForce energy=\"{HARMONIC_IMPROPER_SIGNED}\"> <Improper> \
                 {label}: theta0 = {theta0} rad ≠ 0 prices the signed dihedral θ, which \
                 LAMMPS's improper harmonic (χ = |φ|) does not"
            ));
        }
        if !(0.0..=std::f64::consts::PI).contains(&theta0) {
            return Err(format!(
                "<CustomTorsionForce> <Improper> {label}: theta0 = {theta0} rad is outside \
                 [0, π], where |θ| lies"
            ));
        }
        ff.def_style("improper", "harmonic", Params::new())
            .map_err(|e| e.to_string())?
            .def_type(
                TypeName::join(&classes)?.as_str(),
                &classes,
                Params::from_pairs(&[("k", k), ("chi0", theta0.to_degrees())]),
            )
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// `<CMAPTorsionForce>`: `<Map>` energies (kJ/mol), OpenMM's layout
/// `energy[i + N·j]` at φ = 2πi/N, ψ = 2πj/N (`CMAPTorsionForce::addMap`),
/// and `<Torsion map class1..class5>` rows naming them. Each torsion row is a
/// `cmap charmm` row whose `grid` is its map in molrs's layout (φ-major from
/// −180°): molrs `[p][q]` = OpenMM `(i, j)` with `p = (i + N/2) mod N`,
/// `q = (j + N/2) mod N`. An odd N puts no OpenMM node on molrs's −180° grid.
fn parse_cmap(ff: &mut ForceField, sec: &Node) -> Result<(), String> {
    let mut maps: Vec<ArrayD<f64>> = Vec::new();
    for m in sec.children().filter(|n| n.has_tag_name("Map")) {
        let values = m
            .text()
            .unwrap_or("")
            .split_whitespace()
            .map(|v| {
                v.parse::<f64>()
                    .map_err(|_| format!("<Map> {}: not a number: {v:?}", maps.len()))
            })
            .collect::<Result<Vec<f64>, String>>()?;
        let n = (values.len() as f64).sqrt().round() as usize;
        if n * n != values.len() || n < 2 {
            return Err(format!(
                "<Map> {}: {} values are no N×N map (N ≥ 2)",
                maps.len(),
                values.len()
            ));
        }
        if !n.is_multiple_of(2) {
            return Err(format!(
                "<Map> {}: N = {n} is odd, so OpenMM's grid (origin 0) has no node at \
                 molrs's −180°",
                maps.len()
            ));
        }
        let mut grid = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let (p, q) = ((i + n / 2) % n, (j + n / 2) % n);
                grid[p * n + q] = values[i + n * j] / KCAL_TO_KJ.get();
            }
        }
        maps.push(ArrayD::from_shape_vec(vec![n, n], grid).map_err(|e| e.to_string())?);
    }
    for t in sec.children().filter(Node::is_element) {
        match t.tag_name().name() {
            "Map" => continue,
            "Torsion" => {}
            other => return Err(format!("<CMAPTorsionForce>: unexpected child <{other}>")),
        }
        let ends = endpoints::<5>(&t)?;
        let index = require_str(&t, "map")?;
        let grid = index
            .parse::<usize>()
            .ok()
            .and_then(|i| maps.get(i))
            .ok_or_else(|| {
                format!(
                    "<Torsion> {}: map=\"{index}\" names no <Map> ({} in the force)",
                    ends.join("-"),
                    maps.len()
                )
            })?;
        let mut params = Params::new();
        params.set_array(crate::ff::ir::CMAP_GRID, grid.clone());
        ff.def_style("cmap", "charmm", Params::new())
            .map_err(|e| e.to_string())?
            .def_type(TypeName::join(&ends)?.as_str(), &ends, params)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn parse_nonbonded(raw: &mut Raw, sec: &Node) -> Result<(), String> {
    let scales = (
        require_f64(sec, "coulomb14scale")?,
        require_f64(sec, "lj14scale")?,
    );
    // OpenMM merges repeated tags and refuses different 1-4 scales.
    if let Some(prev) = raw.nonbonded_scales
        && prev != scales
    {
        return Err("two <NonbondedForce> tags with different 1-4 scales".into());
    }
    raw.nonbonded_scales = Some(scales);
    for a in sec.children().filter(Node::is_element) {
        // UseAttributeFromResidue (charges on the residue templates) is a
        // system-building rule, not a parameter.
        if a.tag_name().name() != "Atom" {
            continue;
        }
        raw.nonbonded.push(NonbondedRow {
            key: AtomKey::of(&a)?,
            // TIP3P et al. omit charge here — do not invent 0.0.
            charge: opt_f64(&a, "charge")?,
            sigma: require_f64(&a, "sigma")? * NM_TO_ANGSTROM.get(),
            epsilon: require_f64(&a, "epsilon")? / KCAL_TO_KJ.get(),
        });
    }
    Ok(())
}

fn parse_lennard_jones(raw: &mut Raw, sec: &Node) -> Result<(), String> {
    let scale = require_f64(sec, "lj14scale")?;
    if let Some(prev) = raw.lj_scale
        && prev != scale
    {
        return Err("two <LennardJonesForce> tags with different lj14scale".into());
    }
    raw.lj_scale = Some(scale);
    for a in sec.children().filter(Node::is_element) {
        match a.tag_name().name() {
            "Atom" => raw.lj_atoms.push(LjRow {
                key: AtomKey::of(&a)?,
                sigma: require_f64(&a, "sigma")? * NM_TO_ANGSTROM.get(),
                epsilon: require_f64(&a, "epsilon")? / KCAL_TO_KJ.get(),
                sigma14: opt_f64(&a, "sigma14")?.map(|s| s * NM_TO_ANGSTROM.get()),
                epsilon14: opt_f64(&a, "epsilon14")?.map(|e| e / KCAL_TO_KJ.get()),
            }),
            "NBFixPair" => {
                let key = |n: usize| -> Result<AtomKey, String> {
                    match (
                        a.attribute(format!("type{n}").as_str()),
                        a.attribute(format!("class{n}").as_str()),
                    ) {
                        (Some(t), None) => Ok(AtomKey::Type(t.to_owned())),
                        (None, Some(c)) => Ok(AtomKey::Class(c.to_owned())),
                        _ => Err(format!("<NBFixPair> needs one of type{n} / class{n}")),
                    }
                };
                raw.nbfix.push(NbfixRow {
                    keys: [key(1)?, key(2)?],
                    sigma: require_f64(&a, "sigma")? * NM_TO_ANGSTROM.get(),
                    epsilon: require_f64(&a, "epsilon")? / KCAL_TO_KJ.get(),
                });
            }
            "UseAttributeFromResidue" => {}
            other => return Err(format!("<LennardJonesForce>: unexpected child <{other}>")),
        }
    }
    Ok(())
}

/// The stored order of an OpenMM `<Improper>` row `[c1, c2, c3, c4]` (centre
/// first), under its force's `ordering`: the order whose dihedral OpenMM
/// prices, which is the one molrs prices.
///
/// - `default` / `amber`: OpenMM evaluates `(c2, c3, c1, c4)` — AMBER's order,
///   centre third. (Where `c2` and `c3` match the same atom types OpenMM picks
///   which of the two goes first by element and index; a caller that lists the
///   atoms of such a row decides the same way.)
/// - `charmm`: `(c1, c2, c3, c4)` for a row without wildcards, the
///   `default` order for one with.
/// - `smirnoff` averages three permutations, which no single dihedral is.
pub(crate) fn improper_order<'a>(
    classes: [&'a str; 4],
    ordering: &str,
    force: &str,
) -> Result<[&'a str; 4], String> {
    let [c1, c2, c3, c4] = classes;
    let wildcard = classes.iter().any(|c| is_wildcard(c));
    match ordering {
        "default" | "amber" => Ok([c2, c3, c1, c4]),
        "charmm" if wildcard => Ok([c2, c3, c1, c4]),
        "charmm" => Ok(classes),
        "smirnoff" => Err(format!(
            "{force} ordering=\"smirnoff\": the improper {} averages three atom \
             permutations, which no single molrs improper prices",
            classes.join("-")
        )),
        other => Err(format!(
            "{force}: unknown improper ordering {other:?} (OpenMM's are default, amber, \
             charmm, smirnoff)"
        )),
    }
}

/// OpenMM's indexed terms `(k [kcal/mol], periodicity, phase [deg])`, from
/// `m = 1` until `periodicity{m}` is absent. A started term must carry all
/// three keys.
fn periodic_terms(node: &Node) -> Result<Vec<(f64, f64, f64)>, String> {
    let mut terms = Vec::new();
    for m in 1.. {
        let n = opt_f64(node, &format!("periodicity{m}"))?;
        let k = opt_f64(node, &format!("k{m}"))?;
        let phase = opt_f64(node, &format!("phase{m}"))?;
        match (n, k, phase) {
            (None, None, None) => break,
            (Some(n), Some(k), Some(phase)) => {
                terms.push((k / KCAL_TO_KJ.get(), n, phase.to_degrees()));
            }
            _ => {
                return Err(format!(
                    "<{}> term {m} needs all of periodicity{m}, k{m}, phase{m}",
                    node.tag_name().name()
                ));
            }
        }
    }
    Ok(terms)
}

// --- attribute helpers (total: missing/malformed → Err) -------------------

fn require_tag(node: &Node, expect: &str) -> Result<(), String> {
    let got = node.tag_name().name();
    if got == expect {
        Ok(())
    } else {
        Err(format!("expected <{}>, got <{}>", expect, got))
    }
}

fn require_str<'a>(node: &'a Node, attr: &str) -> Result<&'a str, String> {
    node.attribute(attr).ok_or_else(|| {
        format!(
            "<{}> missing required attribute `{}`",
            node.tag_name().name(),
            attr
        )
    })
}

fn require_f64(node: &Node, attr: &str) -> Result<f64, String> {
    let raw = require_str(node, attr)?;
    raw.parse::<f64>().map_err(|_| {
        format!(
            "<{}> attribute `{}` is not a number: {:?}",
            node.tag_name().name(),
            attr,
            raw
        )
    })
}

fn opt_f64(node: &Node, attr: &str) -> Result<Option<f64>, String> {
    match node.attribute(attr) {
        None => Ok(None),
        Some(raw) => raw.parse::<f64>().map(Some).map_err(|_| {
            format!(
                "<{}> attribute `{}` is not a number: {:?}",
                node.tag_name().name(),
                attr,
                raw
            )
        }),
    }
}

/// foyer's `<ForceField combining_rule>`: `lorentz` (Lorentz-Berthelot,
/// [`CombiningRule::Arithmetic`]) or a canonical rule name.
fn foyer_rule(rule: &str) -> Result<CombiningRule, String> {
    match rule {
        "lorentz" => Ok(CombiningRule::Arithmetic),
        other => {
            CombiningRule::parse(other).map_err(|e| format!("<ForceField combining_rule>: {e}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny but genuine OPLS-AA/GROMACS XML excerpt (rows copied from molpy's
    /// bundled `oplsaa.xml`), exercising every section. Used for conversion and
    /// edge-case unit tests; full-file parity lives in the bm-molrs-molpy harness.
    const MINI: &str = r#"<ForceField name="OPLS-AA" combining_rule="geometric">
  <AtomTypes>
    <Type name="opls_001" class="opls_001" element="C" mass="12.011"/>
    <Type name="opls_002" class="opls_002" element="O" mass="15.9994"/>
  </AtomTypes>
  <HarmonicBondForce>
    <Bond class1="OW" class2="HW" length="0.09572" k="502080.0"/>
  </HarmonicBondForce>
  <HarmonicAngleForce>
    <Angle class1="HW" class2="OW" class3="HW" angle="1.91113553093" k="627.6"/>
  </HarmonicAngleForce>
  <RBTorsionForce>
    <Proper class1="Br" class2="C" class3="CT" class4="HC" c0="0.75312" c1="2.25936" c2="0.0" c3="-3.01248" c4="0.0" c5="0.0"/>
  </RBTorsionForce>
  <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
    <Atom type="opls_001" charge="0.5" sigma="0.375" epsilon="0.43932"/>
    <Atom type="opls_002" charge="-0.5" sigma="0.296" epsilon="0.87864"/>
  </NonbondedForce>
</ForceField>"#;

    /// `<RBTorsionForce>` with the given `c0..c5` (kJ/mol) on Br-C-CT-HC.
    fn rb_row(c: [&str; 6]) -> String {
        format!(
            r#"<ForceField name="x"><RBTorsionForce>
    <Proper class1="Br" class2="C" class3="CT" class4="HC" c0="{}" c1="{}" c2="{}" c3="{}" c4="{}" c5="{}"/>
</RBTorsionForce></ForceField>"#,
            c[0], c[1], c[2], c[3], c[4], c[5]
        )
    }

    fn rb_params(xml: &str, style: &str) -> Params {
        let ff = OpenmmXmlReader::new().read_str(xml).unwrap();
        dihedral_types(ff.get_style("dihedral", style).unwrap())[0]
            .params
            .clone()
    }

    /// RB `Σ Cₙ cosⁿ(φ − 180°)` is `Σ Aₙ₊₁ cosⁿφ`, `Aₙ₊₁ = (−1)ⁿ Cₙ`: the MINI
    /// row (C0 = 0.75312, C1 = 2.25936, C3 = −3.01248 kJ/mol) is
    /// `multi/harmonic` A1 = 0.18, A2 = −0.54, A4 = 0.72 kcal/mol, A3 = A5 = 0.
    #[test]
    fn rb_row_reads_as_multi_harmonic_in_kcal() {
        let p = rb_params(MINI, "multi/harmonic");
        for (key, want) in [
            ("a1", 0.18),
            ("a2", -0.54),
            ("a3", 0.0),
            ("a4", 0.72),
            ("a5", 0.0),
        ] {
            let got = p.get(key).unwrap();
            assert!((got - want).abs() < 1e-12, "{key}: got {got}, want {want}");
        }
        assert!(p.get("a6").is_none());
    }

    /// A constant offset (ΣC ≠ 0) is part of the polynomial form: kept, not
    /// refused.
    #[test]
    fn rb_row_with_nonzero_sum_keeps_its_constant() {
        let p = rb_params(
            &rb_row(["4.184", "0", "0", "0", "0", "0"]),
            "multi/harmonic",
        );
        assert!((p.get("a1").unwrap() - 1.0).abs() < 1e-12);
    }

    /// C5 ≠ 0 is past `multi/harmonic`'s cos⁴φ: `nharmonic` with N = 6,
    /// A6 = −C5.
    #[test]
    fn rb_row_with_c5_reads_as_nharmonic() {
        let p = rb_params(&rb_row(["0", "0", "0", "0", "0", "4.184"]), "nharmonic");
        assert!((p.get("a6").unwrap() + 1.0).abs() < 1e-12);
        assert_eq!(p.get("a1"), Some(0.0));
    }

    /// The RB series equals OpenMM's formula at any φ.
    #[test]
    fn rb_multi_harmonic_prices_openmm_formula() {
        use crate::ff::ir::torsion::MultiHarmonicForm;
        let c = [1.3, -0.7, 2.1, 0.4, -1.9, 0.0];
        let strs = c.map(|v| v.to_string());
        let xml = rb_row(strs.each_ref().map(|v| v.as_str()));
        let p = rb_params(&xml, "multi/harmonic");
        let series = MultiHarmonicForm::from_params(&p).to_series();
        for phi in [-2.9, -1.0, 0.0, 0.4, 1.7, 3.1] {
            let psi: f64 = phi - std::f64::consts::PI;
            let openmm: f64 = c
                .iter()
                .enumerate()
                .map(|(n, cn)| cn * psi.cos().powi(n as i32))
                .sum();
            let got = series.energy(phi) * KCAL_TO_KJ.get();
            assert!((got - openmm).abs() < 1e-12, "φ = {phi}: {got} vs {openmm}");
        }
    }

    /// An RB `<Improper>` has no improper form.
    #[test]
    fn rb_improper_is_refused() {
        let xml = r#"<ForceField><RBTorsionForce><Improper class1="A" class2="B" class3="C" class4="D" c0="1" c1="0" c2="0" c3="0" c4="0" c5="0"/></RBTorsionForce></ForceField>"#;
        let err = OpenmmXmlReader::new().read_str(xml).unwrap_err();
        assert!(err.contains("RBTorsionForce <Improper>"), "{err}");
    }

    #[test]
    fn reads_all_sections_with_molrs_units() {
        let ff = OpenmmXmlReader::new().read_str(MINI).unwrap();

        // bond: length 0.09572 nm → 0.9572 Å; OpenMM ½k 502080 kJ/mol/nm² →
        // LAMMPS K = 502080 / 418.4 / 2 = 600 kcal/mol/Å².
        let bond = ff.get_style("bond", "harmonic").unwrap();
        let bt = bond.get_bondtype("OW", "HW").unwrap();
        assert!((bt.params.get("r0").unwrap() - 0.9572).abs() < 1e-9);
        assert!((bt.params.get("k").unwrap() - 600.0).abs() < 1e-9);

        // angle: theta0 rad → deg; OpenMM ½k 627.6 → K = 627.6 / 4.184 / 2 = 75.
        let angle = ff.get_style("angle", "harmonic").unwrap();
        let at = &angle_types(angle)[0];
        assert!((at.params.get("theta0").unwrap() - 109.5).abs() < 1e-8);
        assert!((at.params.get("k").unwrap() - 75.0).abs() < 1e-9);

        // RB → multi/harmonic a1..a5.
        let dih = ff.get_style("dihedral", "multi/harmonic").unwrap();
        assert!(dihedral_types(dih)[0].params.get("a5").is_some());

        // pair lj/cut: sigma 0.375 nm → 3.75 Å; epsilon 0.43932 kJ → /4.184 kcal.
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let pt = lj.get_pairtype("opls_001", None).unwrap();
        assert!((pt.params.get("sigma").unwrap() - 3.75).abs() < 1e-9);
        assert!((pt.params.get("epsilon").unwrap() - 0.43932 / 4.184).abs() < 1e-9);

        // OpenMM's NonbondedForce mixes Lorentz-Berthelot unless the foyer
        // root says otherwise (MINI: geometric).
        assert_eq!(lj.params().get_str("mixing"), Some("geometric"));

        // coul/cut with OpenMM's own Coulomb constant.
        let coul = ff.get_style("pair", "coul/cut").unwrap();
        assert_eq!(
            coul.params().get("coulomb"),
            Some(crate::core::constants::openmm_coulomb_real())
        );
        assert!((crate::core::constants::openmm_coulomb_real() - 332.06371329919216).abs() < 1e-12);

        // The 1-4 scales live on the ForceField's special_bonds (1-2/1-3
        // excluded) — the single source the pair kernels consume.
        let sb = ff.special_bonds();
        assert_eq!(sb.lj, [0.0, 0.0, 0.5]);
        assert_eq!(sb.coul, [0.0, 0.0, 0.5]);

        // atom style carries mass + charge per opls type.
        let atom = ff.get_style("atom", "full").unwrap();
        let a1 = atom.get_atomtype("opls_001").unwrap();
        assert!((a1.params.get("mass").unwrap() - 12.011).abs() < 1e-9);
        assert!((a1.params.get("charge").unwrap() - 0.5).abs() < 1e-12);
        let a2 = atom.get_atomtype("opls_002").unwrap();
        assert!((a2.params.get("charge").unwrap() + 0.5).abs() < 1e-12);
    }

    #[test]
    fn missing_required_attr_errors() {
        let xml = r#"<ForceField name="x"><HarmonicBondForce>
            <Bond class1="OW" class2="HW" length="0.1"/>
        </HarmonicBondForce></ForceField>"#;
        let err = OpenmmXmlReader::new().read_str(xml).unwrap_err();
        assert!(err.contains('k'), "err: {err}");
    }

    #[test]
    fn non_numeric_attr_errors() {
        let xml = r#"<ForceField name="x"><HarmonicBondForce>
            <Bond class1="OW" class2="HW" length="oops" k="1.0"/>
        </HarmonicBondForce></ForceField>"#;
        let err = OpenmmXmlReader::new().read_str(xml).unwrap_err();
        assert!(err.contains("not a number"), "err: {err}");
    }

    #[test]
    fn wrong_root_errors() {
        let err = OpenmmXmlReader::new()
            .read_str(r#"<System name="x"/>"#)
            .unwrap_err();
        assert!(err.contains("ForceField"), "err: {err}");
    }

    #[test]
    fn unknown_section_errors() {
        let xml = r#"<ForceField name="x"><MysteryForce/></ForceField>"#;
        let err = OpenmmXmlReader::new().read_str(xml).unwrap_err();
        assert!(
            err.contains("<MysteryForce>: no force-field IR form"),
            "err: {err}"
        );
    }

    /// Two `<NonbondedForce>` rows for one type with different charges: the
    /// per-type charge is ambiguous, so the read fails and names the type
    /// rather than silently keeping the first row.
    #[test]
    fn conflicting_nonbonded_charges_for_one_type_error() {
        let row = r#"    <Atom type="opls_001" charge="0.5" sigma="0.375" epsilon="0.43932"/>
"#;
        assert_eq!(MINI.matches(row).count(), 1, "fixture edit must be unique");
        let xml = MINI.replacen(
            row,
            r#"    <Atom type="opls_001" charge="0.5" sigma="0.375" epsilon="0.43932"/>
    <Atom type="opls_001" charge="0.25" sigma="0.375" epsilon="0.43932"/>
"#,
            1,
        );
        let err = OpenmmXmlReader::new().read_str(&xml).unwrap_err();
        assert!(
            err.contains("opls_001"),
            "error should name the type: {err}"
        );
        assert!(err.contains("charge"), "error should mention charge: {err}");
    }

    /// Class-only bond endpoints get one `type_="*"` placeholder AtomType per
    /// class. The contract: placeholders are appended in ascending class-name
    /// order, identically on every read (no HashSet iteration order leaking
    /// into the ForceField). Ten classes, listed in a deliberately unsorted
    /// order, so a coincidentally sorted hash order is ~1/10! per read; the
    /// 20 repeated reads each get a freshly seeded `RandomState`.
    #[test]
    fn class_wildcard_placeholders_are_sorted_by_class_name() {
        let xml = r#"<ForceField name="x">
  <AtomTypes>
    <Type name="opls_900" class="opls_900" element="C" mass="12.011"/>
  </AtomTypes>
  <HarmonicBondForce>
    <Bond class1="zz" class2="CT" length="0.1" k="1.0"/>
    <Bond class1="OH" class2="a1" length="0.1" k="1.0"/>
    <Bond class1="Q9" class2="br" length="0.1" k="1.0"/>
    <Bond class1="M" class2="cl" length="0.1" k="1.0"/>
    <Bond class1="HC" class2="Na" length="0.1" k="1.0"/>
  </HarmonicBondForce>
</ForceField>"#;
        // Byte-wise (String Ord) ascending order, written out by hand.
        let expected: Vec<&str> = vec!["CT", "HC", "M", "Na", "OH", "Q9", "a1", "br", "cl", "zz"];

        let placeholder_names = |text: &str| -> Vec<String> {
            let ff = OpenmmXmlReader::new().read_str(text).unwrap();
            let atom = ff.get_style("atom", "full").unwrap();
            atom_types(atom)
                .iter()
                .filter(|t| t.params.get_str("type_") == Some("*"))
                .map(|t| t.name.clone())
                .collect()
        };

        let first = placeholder_names(xml);
        assert_eq!(first, expected, "placeholders not in ascending class order");

        for i in 0..20 {
            let again = placeholder_names(xml);
            assert_eq!(
                again, first,
                "read #{i} produced a different placeholder order"
            );
        }
    }

    /// One `<PeriodicTorsionForce>` holding *rows* (raw XML children).
    fn periodic_section(rows: &str) -> String {
        format!(
            r#"<ForceField name="x"><PeriodicTorsionForce>
{rows}
</PeriodicTorsionForce></ForceField>"#
        )
    }

    /// OpenMM's own spelling — `E = Σ k_m [1 + cos(n_m φ − γ_m)]`, kJ/mol and
    /// radians — is the same form as molrs `dihedral periodic`, so it reads
    /// term by term: 4.184 kJ/mol → 1 kcal/mol, 2.092 → 0.5, phases in degrees.
    #[test]
    fn openmm_periodic_proper_reads_as_multi_term_periodic() {
        let xml = periodic_section(
            r#"<Proper class1="HC" class2="CT" class3="CT" class4="HC" periodicity1="3" k1="4.184" phase1="0.0" periodicity2="1" k2="2.092" phase2="3.141592653589793"/>"#,
        );
        let ff = OpenmmXmlReader::new().read_str(&xml).unwrap();
        let dih = ff
            .get_style("dihedral", "periodic")
            .expect("dihedral periodic");
        let p = &dihedral_types(dih)[0].params;
        for (key, want) in [
            ("k1", 1.0),
            ("periodicity1", 3.0),
            ("phase1", 0.0),
            ("k2", 0.5),
            ("periodicity2", 1.0),
            ("phase2", 180.0),
        ] {
            let got = p.get(key).unwrap_or_else(|| panic!("missing {key}"));
            assert!((got - want).abs() < 1e-12, "{key}: got {got}, want {want}");
        }
        assert!(p.get("k3").is_none());
    }

    /// CL&P's `c0..c3` spelling under the same tag still reads as OPLS Fourier.
    #[test]
    fn clp_fourier_proper_still_reads_as_opls() {
        let xml = periodic_section(
            r#"<Proper class1="CT" class2="CT" class3="CT" class4="CT" c0="5.4392" c1="-0.2092" c2="0.8368" c3="0.0"/>"#,
        );
        let ff = OpenmmXmlReader::new().read_str(&xml).unwrap();
        let p = &dihedral_types(ff.get_style("dihedral", "opls").unwrap())[0].params;
        assert!((p.get("k1").unwrap() - 1.3).abs() < 1e-12);
        assert!((p.get("k3").unwrap() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn proper_with_neither_spelling_is_an_error() {
        let xml = periodic_section(r#"<Proper class1="A" class2="B" class3="C" class4="D"/>"#);
        let err = OpenmmXmlReader::new().read_str(&xml).unwrap_err();
        assert!(err.contains("periodicity1") && err.contains("c0"), "{err}");
    }

    #[test]
    fn proper_with_both_spellings_is_an_error() {
        let xml = periodic_section(
            r#"<Proper class1="A" class2="B" class3="C" class4="D" c0="1.0" periodicity1="2" k1="1.0" phase1="0.0"/>"#,
        );
        assert!(OpenmmXmlReader::new().read_str(&xml).is_err());
    }

    /// A term index needs all three of `k{m}`, `periodicity{m}`, `phase{m}`.
    #[test]
    fn incomplete_periodic_term_is_an_error() {
        let xml = periodic_section(
            r#"<Proper class1="A" class2="B" class3="C" class4="D" periodicity1="2" k1="1.0"/>"#,
        );
        let err = OpenmmXmlReader::new().read_str(&xml).unwrap_err();
        assert!(err.contains("phase1"), "{err}");
    }

    /// OpenMM impropers live under the same force and are read.
    /// The row lists the centre (`N`) first and OpenMM prices the dihedral
    /// `(c2, c3, c1, c4)`, so it is stored in that order: AMBER's, centre third.
    #[test]
    fn openmm_improper_reads_as_periodic_improper() {
        let xml = periodic_section(
            r#"<Improper class1="N" class2="C" class3="O" class4="CT" periodicity1="2" k1="43.932" phase1="3.141592653589793"/>"#,
        );
        let ff = OpenmmXmlReader::new().read_str(&xml).unwrap();
        let imp = ff
            .get_style("improper", "periodic")
            .expect("improper periodic");
        let t = &improper_types(imp)[0];
        assert_eq!(
            [
                t.itom.as_str(),
                t.jtom.as_str(),
                t.ktom.as_str(),
                t.ltom.as_str()
            ],
            ["C", "O", "N", "CT"]
        );
        assert_eq!(t.name, "C-O-N-CT");
        assert!((t.params.get("k").unwrap() - 10.5).abs() < 1e-12);
        assert!((t.params.get("periodicity").unwrap() - 2.0).abs() < 1e-12);
        assert_eq!(t.params.get("phase"), Some(180.0));
    }

    /// Under `ordering="charmm"` OpenMM prices a wildcard-free row as written,
    /// so it is stored as written; `smirnoff` (three averaged permutations) is
    /// refused.
    #[test]
    fn improper_ordering_charmm_keeps_the_row_and_smirnoff_is_refused() {
        let row = r#"<Improper class1="N" class2="C" class3="O" class4="CT" periodicity1="2" k1="4.184" phase1="0.0"/>"#;
        let charmm = format!(
            r#"<ForceField name="x"><PeriodicTorsionForce ordering="charmm">{row}</PeriodicTorsionForce></ForceField>"#
        );
        let ff = OpenmmXmlReader::new().read_str(&charmm).unwrap();
        let t = &improper_types(ff.get_style("improper", "periodic").unwrap())[0];
        assert_eq!(
            [
                t.itom.as_str(),
                t.jtom.as_str(),
                t.ktom.as_str(),
                t.ltom.as_str()
            ],
            ["N", "C", "O", "CT"]
        );
        let smirnoff = charmm.replace("charmm", "smirnoff");
        let err = OpenmmXmlReader::new().read_str(&smirnoff).unwrap_err();
        assert!(err.contains("smirnoff"), "{err}");
    }

    /// The periodic improper kernel holds one term; a second is refused, not dropped.
    #[test]
    fn multi_term_improper_is_an_error() {
        let xml = periodic_section(
            r#"<Improper class1="C" class2="O" class3="N" class4="CT" periodicity1="2" k1="1.0" phase1="0.0" periodicity2="1" k2="1.0" phase2="0.0"/>"#,
        );
        assert!(OpenmmXmlReader::new().read_str(&xml).is_err());
    }

    /// `<PeriodicImproperForce>` is no OpenMM force; it is refused as an
    /// unknown section.
    #[test]
    fn a_periodic_improper_force_section_is_refused() {
        let xml = r#"<ForceField name="x"><PeriodicImproperForce>
<Improper class1="C" class2="O" class3="N" class4="CT" periodicity1="2" k1="43.932" phase1="3.141592653589793"/>
</PeriodicImproperForce></ForceField>"#;
        let err = OpenmmXmlReader::new().read_str(xml).unwrap_err();
        assert!(err.contains("PeriodicImproperForce"), "{err}");
    }

    #[test]
    fn unknown_periodic_torsion_child_is_an_error() {
        let xml = periodic_section(r#"<Torsion class1="A" class2="B" class3="C" class4="D"/>"#);
        assert!(OpenmmXmlReader::new().read_str(&xml).is_err());
    }

    /// A CHARMM-style excerpt: charges from `<NonbondedForce>` (ε = 0), the
    /// van der Waals from `<LennardJonesForce>` with 1-4 parameters and an
    /// NBFIX row, a Urey–Bradley term, a harmonic `<CustomTorsionForce>`
    /// improper and a 4×4 CMAP.
    const CHARMM: &str = r#"<ForceField>
  <AtomTypes>
    <Type name="CT1" class="CT1" element="C" mass="12.011"/>
    <Type name="HB1" class="HB1" element="H" mass="1.008"/>
    <Type name="NH1" class="NH1" element="N" mass="14.007"/>
    <Type name="C" class="C" element="C" mass="12.011"/>
  </AtomTypes>
  <HarmonicAngleForce>
    <Angle type1="NH1" type2="CT1" type3="HB1" angle="1.8849555921538759" k="401.664"/>
    <Angle type1="NH1" type2="CT1" type3="C" angle="1.8849555921538759" k="418.4"/>
  </HarmonicAngleForce>
  <AmoebaUreyBradleyForce>
    <UreyBradley type1="NH1" type2="CT1" type3="HB1" k="20920.0" d="0.214"/>
    <UreyBradley type1="HB1" type2="CT1" type3="NH1" k="20920.0" d="0.214"/>
    <UreyBradley type1="C" type2="CT1" type3="C" k="8368.0" d="0.25"/>
  </AmoebaUreyBradleyForce>
  <CustomTorsionForce energy="k * (theta - theta0)^2">
    <PerTorsionParameter name="k"/>
    <PerTorsionParameter name="theta0"/>
    <Improper k="83.68" theta0="0.0" type1="NH1" type2="C" type3="CT1" type4="HB1"/>
  </CustomTorsionForce>
  <CMAPTorsionForce>
    <Map>0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15</Map>
    <Torsion map="0" type1="C" type2="NH1" type3="CT1" type4="C" type5="NH1"/>
  </CMAPTorsionForce>
  <NonbondedForce coulomb14scale="1.0" lj14scale="1.0">
    <UseAttributeFromResidue name="charge"/>
    <Atom type="CT1" sigma="1.0" epsilon="0.0"/>
    <Atom type="HB1" sigma="1.0" epsilon="0.0"/>
    <Atom class="NH1" sigma="1.0" epsilon="0.0"/>
    <Atom type="C" charge="0.51" sigma="1.0" epsilon="0.0"/>
  </NonbondedForce>
  <LennardJonesForce lj14scale="1.0">
    <Atom type="CT1" sigma="0.35635948725613575" epsilon="0.133888" sigma14="0.3385415128933289" epsilon14="0.04184"/>
    <Atom type="HB1" sigma="0.2351972615890496" epsilon="0.092048"/>
    <Atom class="NH1" sigma="0.329632525712" epsilon="0.8368"/>
    <Atom type="C" sigma="0.356359487256" epsilon="0.46024"/>
    <NBFixPair type1="NH1" type2="HB1" sigma="0.28" epsilon="0.2"/>
  </LennardJonesForce>
</ForceField>"#;

    fn charmm() -> ForceField {
        OpenmmXmlReader::new().read_str(CHARMM).unwrap()
    }

    /// `<LennardJonesForce>` is `lj/charmm`: per-type ε/σ (and ε₁₄/σ₁₄ when the
    /// row has them), a class row on every type of the class, NBFIX as a
    /// cross row without 1-4 parameters of its own (OpenMM prices a 1-4 NBFIX
    /// pair with the NBFIX row), Lorentz-Berthelot; the Coulomb half is
    /// `coul/charmm` with OpenMM's constant; the 1-4 weights are the two
    /// forces' scales; ε₁₄ ≠ ε declares `one_four = "epsilon14"`.
    #[test]
    fn lennard_jones_force_reads_as_lj_charmm() {
        let ff = charmm();
        let lj = ff.get_style("pair", "lj/charmm").expect("lj/charmm");
        assert_eq!(lj.params().get_str("mixing"), Some("arithmetic"));
        assert_eq!(lj.params().get_str("one_four"), Some("epsilon14"));
        assert!(ff.get_style("pair", "lj/cut").is_none());
        let rows = pair_types(lj);
        let row = |i: &str, j: &str| {
            rows.iter()
                .find(|t| t.itom == i && t.jtom == j)
                .unwrap_or_else(|| panic!("{i}-{j}"))
        };
        let ct1 = &row("CT1", "CT1").params;
        assert!((ct1.get("epsilon").unwrap() - 0.032).abs() < 1e-15);
        assert!((ct1.get("sigma").unwrap() - 3.5635948725613575).abs() < 1e-14);
        assert!((ct1.get("epsilon14").unwrap() - 0.01).abs() < 1e-15);
        assert!((ct1.get("sigma14").unwrap() - 3.385415128933289).abs() < 1e-14);
        assert!(row("HB1", "HB1").params.get("epsilon14").is_none());
        assert!((row("NH1", "NH1").params.get("epsilon").unwrap() - 0.2).abs() < 1e-15);
        let fix = &row("NH1", "HB1").params;
        assert!((fix.get("sigma").unwrap() - 2.8).abs() < 1e-14);
        assert!(fix.get("epsilon14").is_none() && fix.get("sigma14").is_none());

        let coul = ff.get_style("pair", "coul/charmm").expect("coul/charmm");
        assert_eq!(
            coul.params().get("coulomb"),
            Some(crate::core::constants::openmm_coulomb_real())
        );
        assert_eq!(ff.special_bonds().lj, [0.0, 0.0, 1.0]);
        assert_eq!(ff.special_bonds().coul, [0.0, 0.0, 1.0]);
        let atom = ff.get_style("atom", "full").unwrap();
        assert_eq!(
            atom.get_atomtype("C").unwrap().params.get("charge"),
            Some(0.51)
        );
        assert_eq!(atom.get_atomtype("CT1").unwrap().params.get("charge"), None);
    }

    /// No row with 1-4 parameters of its own: LAMMPS's `special_bonds`
    /// pricing is OpenMM's, so no `one_four` is declared.
    #[test]
    fn lennard_jones_force_without_own_one_four_declares_nothing() {
        let xml = CHARMM.replace(r#" sigma14="0.3385415128933289" epsilon14="0.04184""#, "");
        let ff = OpenmmXmlReader::new().read_str(&xml).unwrap();
        let lj = ff.get_style("pair", "lj/charmm").unwrap();
        assert_eq!(lj.params().get_str("one_four"), None);
    }

    /// Without a `<NonbondedForce>` OpenMM prices no Coulomb: no Coulomb style.
    #[test]
    fn lennard_jones_force_alone_has_no_coulomb_style() {
        let start = CHARMM.find("  <NonbondedForce").unwrap();
        let end = CHARMM.find("</NonbondedForce>").unwrap() + "</NonbondedForce>".len();
        let xml = format!("{}{}", &CHARMM[..start], &CHARMM[end..]);
        let ff = OpenmmXmlReader::new().read_str(&xml).unwrap();
        assert!(ff.get_style("pair", "lj/charmm").is_some());
        assert!(ff.get_style("pair", "coul/charmm").is_none());
    }

    /// OpenMM computes the NonbondedForce's own Lennard-Jones as well: a
    /// non-zero ε there beside a `<LennardJonesForce>` is two LJ forces.
    #[test]
    fn nonbonded_epsilon_beside_lennard_jones_force_is_refused() {
        let xml = CHARMM.replace(
            r#"<Atom type="HB1" sigma="1.0" epsilon="0.0"/>"#,
            r#"<Atom type="HB1" sigma="1.0" epsilon="0.1"/>"#,
        );
        let err = OpenmmXmlReader::new().read_str(&xml).unwrap_err();
        assert!(
            err.contains("HB1") && err.contains("LennardJonesForce"),
            "{err}"
        );
    }

    #[test]
    fn nbfix_of_a_type_with_itself_is_refused() {
        let xml = CHARMM.replace(r#"type1="NH1" type2="HB1""#, r#"type1="HB1" type2="HB1""#);
        let err = OpenmmXmlReader::new().read_str(&xml).unwrap_err();
        assert!(err.contains("NBFixPair"), "{err}");
    }

    /// OpenMM's UB builder adds a bond of force constant `2k` per angle, so
    /// the file's `k` is LAMMPS's un-halved `K_ub`: 20920 kJ/mol/nm² = 50
    /// kcal/mol/Å², d = 0.214 nm = 2.14 Å. The angle row of the same types (in
    /// either direction) becomes `angle charmm`; an angle without UB stays
    /// `harmonic`; a UB row without an angle row is `angle charmm` with k = 0.
    #[test]
    fn urey_bradley_joins_its_angle_as_angle_charmm() {
        let ff = charmm();
        let ch = ff.get_style("angle", "charmm").expect("angle charmm");
        let rows = angle_types(ch);
        let ub = rows.iter().find(|t| t.name == "NH1-CT1-HB1").unwrap();
        for (key, want) in [
            ("k", 401.664 / 4.184 / 2.0),
            ("theta0", 108.0),
            ("k_ub", 50.0),
            ("r_ub", 2.14),
        ] {
            let got = ub.params.get(key).unwrap();
            assert!((got - want).abs() < 1e-12, "{key}: {got} vs {want}");
        }
        let lone = rows.iter().find(|t| t.name == "C-CT1-C").unwrap();
        assert_eq!(lone.params.get("k"), Some(0.0));
        assert!((lone.params.get("k_ub").unwrap() - 20.0).abs() < 1e-12);
        let h = ff.get_style("angle", "harmonic").expect("angle harmonic");
        assert_eq!(angle_types(h).len(), 1);
        assert_eq!(angle_types(h)[0].name, "NH1-CT1-C");
    }

    #[test]
    fn conflicting_urey_bradley_rows_are_refused() {
        let xml = CHARMM.replace(
            r#"<UreyBradley type1="HB1" type2="CT1" type3="NH1" k="20920.0" d="0.214"/>"#,
            r#"<UreyBradley type1="HB1" type2="CT1" type3="NH1" k="20000.0" d="0.214"/>"#,
        );
        assert!(OpenmmXmlReader::new().read_str(&xml).is_err());
    }

    /// `k*(theta-theta0)^2` (spaces aside) is `improper harmonic`, k in
    /// kcal/mol/rad² with no ½ on either side; `charmm` ordering keeps a
    /// wildcard-free row as written.
    #[test]
    fn custom_torsion_harmonic_improper_reads_as_improper_harmonic() {
        let ff = charmm();
        let imp = ff
            .get_style("improper", "harmonic")
            .expect("improper harmonic");
        let t = &improper_types(imp)[0];
        assert_eq!(t.name, "NH1-C-CT1-HB1");
        assert!((t.params.get("k").unwrap() - 20.0).abs() < 1e-12);
        assert_eq!(t.params.get("chi0"), Some(0.0));
    }

    fn custom(energy: &str, theta0: &str) -> String {
        format!(
            r#"<ForceField><CustomTorsionForce energy="{energy}"><PerTorsionParameter name="k"/><PerTorsionParameter name="theta0"/><Improper k="83.68" theta0="{theta0}" type1="A" type2="B" type3="C" type4="D"/></CustomTorsionForce></ForceField>"#
        )
    }

    /// OpenMM's θ is signed; LAMMPS's χ = |φ|. They agree at θ0 = 0 only, so
    /// `k*(theta-theta0)^2` with θ0 = π is refused; molrs's `abs` form holds
    /// any χ0.
    #[test]
    fn signed_harmonic_improper_off_zero_is_refused_and_abs_form_reads() {
        let err = OpenmmXmlReader::new()
            .read_str(&custom("k*(theta-theta0)^2", "3.141592653589793"))
            .unwrap_err();
        assert!(err.contains("signed"), "{err}");
        let ff = OpenmmXmlReader::new()
            .read_str(&custom("k*(abs(theta)-theta0)^2", "3.141592653589793"))
            .unwrap();
        let t = &improper_types(ff.get_style("improper", "harmonic").unwrap())[0];
        assert!((t.params.get("chi0").unwrap() - 180.0).abs() < 1e-12);
    }

    #[test]
    fn other_custom_forces_are_refused_by_name() {
        let err = OpenmmXmlReader::new()
            .read_str(&custom("k*(1+cos(theta-theta0))", "0"))
            .unwrap_err();
        assert!(err.contains("k*(1+cos(theta-theta0))"), "{err}");
        for tag in [
            "CustomBondForce",
            "CustomNonbondedForce",
            "Script",
            "GBSAOBCForce",
        ] {
            let xml = format!("<ForceField><{tag}/></ForceField>");
            let err = OpenmmXmlReader::new().read_str(&xml).unwrap_err();
            assert!(err.contains(tag), "{err}");
        }
        let proper = r#"<ForceField><CustomTorsionForce energy="k*(theta-theta0)^2"><PerTorsionParameter name="k"/><PerTorsionParameter name="theta0"/><Proper k="1" theta0="0" type1="A" type2="B" type3="C" type4="D"/></CustomTorsionForce></ForceField>"#;
        let err = OpenmmXmlReader::new().read_str(proper).unwrap_err();
        assert!(err.contains("<Proper>"), "{err}");
    }

    /// OpenMM's `energy[i + N·j]` is at φ = 2πi/N, ψ = 2πj/N; molrs `[p][q]` at
    /// φ = −180° + pΔ, ψ = −180° + qΔ. With N = 4 (Δ = 90°) and OpenMM's
    /// value `i + 4j`: molrs `[0][0]` (−180°, −180°) is OpenMM (2, 2) = 10;
    /// `[2][0]` (0°, −180°) is OpenMM (0, 2) = 8; `[0][2]` (−180°, 0°) is
    /// OpenMM (2, 0) = 2; `[3][1]` (90°, −90°) is OpenMM (1, 3) = 13. Values ÷ 4.184.
    #[test]
    fn cmap_map_is_shifted_by_half_and_transposed_into_phi_major() {
        let ff = charmm();
        let rows = ff.get_cmaptypes();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "C-NH1-CT1-C-NH1");
        let grid = rows[0].params.get_array("grid").unwrap();
        assert_eq!(grid.shape(), &[4, 4]);
        for (p, q, omm) in [
            (0, 0, 10.0),
            (2, 0, 8.0),
            (0, 2, 2.0),
            (3, 1, 13.0),
            (2, 2, 0.0),
        ] {
            assert_eq!(grid[[p, q]], omm / 4.184, "[{p}][{q}]");
        }
    }

    #[test]
    fn odd_cmap_size_and_unknown_map_are_refused() {
        let odd = CHARMM.replace(
            "<Map>0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15</Map>",
            "<Map>0 1 2 3 4 5 6 7 8</Map>",
        );
        let err = OpenmmXmlReader::new().read_str(&odd).unwrap_err();
        assert!(err.contains("odd"), "{err}");
        let missing = CHARMM.replace(r#"<Torsion map="0""#, r#"<Torsion map="3""#);
        let err = OpenmmXmlReader::new().read_str(&missing).unwrap_err();
        assert!(err.contains("map=\"3\""), "{err}");
    }

    /// A bonded row naming neither `class{n}` nor `type{n}` is ignored by
    /// OpenMM; it is refused rather than read as a wildcard.
    #[test]
    fn a_row_without_endpoints_is_refused() {
        let xml = r#"<ForceField><HarmonicBondForce><Bond class1="A" length="0.1" k="1"/></HarmonicBondForce></ForceField>"#;
        let err = OpenmmXmlReader::new().read_str(xml).unwrap_err();
        assert!(err.contains("class2"), "{err}");
    }

    /// OpenMM's wildcard is an empty attribute, kept as `""`.
    #[test]
    fn an_empty_class_is_the_wildcard() {
        let xml = periodic_section(
            r#"<Improper class1="C" class2="" class3="" class4="O" periodicity1="2" k1="43.932" phase1="3.141592653589793"/>"#,
        );
        let ff = OpenmmXmlReader::new().read_str(&xml).unwrap();
        let t = &improper_types(ff.get_style("improper", "periodic").unwrap())[0];
        assert_eq!(
            [
                t.itom.as_str(),
                t.jtom.as_str(),
                t.ktom.as_str(),
                t.ltom.as_str()
            ],
            ["", "", "C", "O"]
        );
    }

    // -- small helpers to reach into StyleDefs for assertions --
    use crate::ff::forcefield::{
        AngleType, AtomType, DihedralType, ImproperType, PairType, Style, StyleDefs,
    };
    fn pair_types(s: &Style) -> &[PairType] {
        match s.defs() {
            StyleDefs::Pair(v) => v,
            _ => unreachable!(),
        }
    }
    fn improper_types(s: &Style) -> &[ImproperType] {
        match s.defs() {
            StyleDefs::Improper(v) => v,
            _ => unreachable!(),
        }
    }
    fn atom_types(s: &Style) -> &[AtomType] {
        match s.defs() {
            StyleDefs::Atom(v) => v,
            _ => unreachable!(),
        }
    }
    fn angle_types(s: &Style) -> &[AngleType] {
        match s.defs() {
            StyleDefs::Angle(v) => v,
            _ => unreachable!(),
        }
    }
    fn dihedral_types(s: &Style) -> &[DihedralType] {
        match s.defs() {
            StyleDefs::Dihedral(v) => v,
            _ => unreachable!(),
        }
    }
}
