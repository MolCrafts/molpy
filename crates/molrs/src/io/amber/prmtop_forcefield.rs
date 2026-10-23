//! AMBER prmtop force-field reader.

use std::collections::{BTreeSet, HashMap};

use crate::core::constants::VACUUM_DIELECTRIC;
use crate::core::constants::{AMBER_COULOMB, CHARMM_COULOMB};
use crate::core::constants::{AMBER_SCEE, AMBER_SCNB};
use crate::ff::forcefield::ForceField;
use crate::ff::ir::CombiningRule;
use crate::ff::ir::{Params, SpecialBonds};
use crate::ff::potential::pair::lj_ab_to_sigma_epsilon;
use crate::io::amber::prmtop::{frame_from_sections, parse_flag_sections};
use crate::io::amber::prmtop_tables::{
    TorsionRow, TorsionTables, TorsionTerm, atom_number, atom_type_names, canonical_terms,
    chamber_impropers, cmap_terms, decode_torsions, is_chamber, parse_tokens, proper_type_names,
    section, table_value, torsion_rows,
};
use crate::io::reader::ForceFieldReader;
use crate::io::{Source, invalid_data};
use molrs::core::TypeName;
use molrs::core::keys;
use molrs::core::{Block, Frame};
use molrs::op::{F, Idx};

/// `(sigma_Å, epsilon_kcal_per_mol)` of one LJ entry.
type Lj = (f64, f64);

/// Reader for AMBER prmtop force-field parameter tables.
///
/// Builds a molrs [`ForceField`] from the parameter tables of an AMBER
/// topology (prmtop / parm7), and of a CHARMM topology ParmEd's `chamber`
/// wrote in the same format. Structure/connectivity lives in
/// [`read_prmtop`](crate::io::read_prmtop); this module owns styles + type params only.
/// AMBER is read, never written: there is no prmtop writer.
///
/// # Amber FileFormats (I/O) vs the force-field IR
///
/// **Parsing** follows <https://ambermd.org/FileFormats.php> (and the
/// expanded Swails prmtop appendix for section layout). **Stored potentials**
/// are the force-field IR's, whose definitions follow LAMMPS — AMBER's own
/// for every bonded term but for the angle unit:
///
/// | Term | Amber prmtop storage | LAMMPS style | molrs store |
/// |------|----------------------|--------------|-------------|
/// | Bond | `RK` in `E = RK·(r−r₀)²` (no ½) | `bond_style harmonic`, same | `bond harmonic`: `k = RK`, `r0` |
/// | Angle | `TK` in `E = TK·(θ−θ₀)²` (no ½), `θ₀` rad | `angle_style harmonic`, same | `angle harmonic`: `k = TK`, `theta0` in degrees |
/// | Dihedral | `PK·[1 + cos(nφ − δ)]`, `δ` rad; rows of one quartet and negative-`PN` chains are one torsion | `dihedral_style fourier` | `dihedral periodic`: `k<m>/periodicity<m>/phase<m>` (degrees), terms sorted by periodicity |
/// | Improper | same form; 4th pointer negative | `improper_style cvff` (one term) | `improper periodic`, AMBER's atom order (centre third); a multi-term improper is one `<quartet>@<n>` type per term |
/// | LJ | `A/r¹² − B/r⁶` via ICO | `lj/cut` σ/ε | `σ = 2^{−1/6} r_min`, `ε = B²/(4A)`; self rows from the diagonal, an explicit cross row for each off-diagonal entry that is not Lorentz–Berthelot (NBFIX) |
/// | 1-4 scales | `SCEE`/`SCNB` divisors per torsion type (default 1.2 / 2.0) | `special_bonds` | `coul_14 = 1/SCEE`, `lj_14 = 1/SCNB` of the divisor most 1-4 rows carry; the frame's `pairs` carry `coul_scale` / `lj_scale` for the pairs that differ |
///
/// A chamber prmtop (`%FLAG CTITLE`) adds CHARMM's terms:
///
/// | Term | chamber prmtop storage | LAMMPS style | molrs store |
/// |------|------------------------|--------------|-------------|
/// | Urey–Bradley | `CHARMM_UREY_BRADLEY*`: `K_ub·(r₁₃ − r_ub)²` per angle | `angle_style charmm` | every angle `angle charmm`: `k = TK`, `theta0`, `k_ub`, `r_ub` (0, 0 for an angle without one) |
/// | Improper | `CHARMM_IMPROPER*`: `K_ψ·(ψ − ψ₀)²`, centre first | `improper_style harmonic` | `improper harmonic`: `k = K_ψ`, `chi0` in degrees, file order |
/// | CMAP | `CHARMM_CMAP_*` (`CMAP_*` in an AMBER ff19SB file) | `fix cmap` | `cmap charmm`, one type per map (`cmap_terms`) |
/// | LJ | `LENNARD_JONES_ACOEF/BCOEF` + `LENNARD_JONES_14_ACOEF/BCOEF` | `lj/charmm/coul/charmm` | `lj/charmm`: `epsilon`, `sigma`, and `epsilon14`, `sigma14` with `one_four = "epsilon14"` when the 1-4 table differs |
/// | Coulomb | `CHARGE` = `q·√332.0716` | `coul/charmm` | `coul/charmm`: `coulomb = 332.0716` |
///
/// Refused by name: polarizable (`IPOL > 0`), 12-6-4 (`LENNARD_JONES_CCOEF`)
/// and 10-12 hydrogen-bond (non-zero `HBOND_ACOEF/BCOEF`, a negative ICO)
/// prmtops; a 1-4 row on a negative-`PN` chain (`one_four_weights`: sander
/// prices that pair once per chained term); two terms of one improper with
/// one periodicity; a CHARMM improper with ψ₀ other than 0° or 180° (LAMMPS
/// prices |ψ|); a Urey–Bradley term that is not on exactly one angle.
///
/// Every phase within 0.004 rad of ±π is ±π exactly, as sander's `rdparm`
/// takes it (`amber_phase`): tleap writes π as `3.14159400`. A type name
/// that stands for two LJ classes or masses is split (`atom_type_names`).
///
/// Notes:
/// - OpenMM multiplies Amber bond/angle `RK`/`TK` by 2 when loading into its ½k
///   kernels; molrs, like LAMMPS, keeps them. Dihedral `PK` is never doubled.
/// - Swails’ appendix writes bond/angle as ½k and torsion as `k cos(…)`; those
///   equations disagree with Amber parameter files, OpenMM’s converter, and
///   the FileFormats `parm.dat` section. We follow FileFormats + OpenMM.
/// - `%COMMENT` lines are skipped. Section order is free (flag map).
#[derive(Debug, Default, Clone, Copy)]
pub struct AmberPrmtopForcefieldReader;

impl AmberPrmtopForcefieldReader {
    /// The reader; the format has no settings.
    pub fn new() -> Self {
        Self
    }

    /// The whole AMBER system `src` holds (a path or a whole file's bytes):
    /// the force field, and the structure frame of
    /// [`read_prmtop`](crate::io::read_prmtop) plus a `pairs` block holding the 1-4 pairs
    /// sander weighs otherwise than the field's `special_bonds` — a torsion
    /// type whose `SCEE` / `SCNB` differ from the divisor most 1-4 rows carry
    /// (GLYCAM beside ff14SB), a pair two rows list, a 1-4 pair no row lists
    /// — each with its own `coul_scale` (Σ 1/SCEE) / `lj_scale` (Σ 1/SCNB)
    /// where it differs, null where it agrees. No `pairs` block when every
    /// pair agrees, or the file has no `SCEE_SCALE_FACTOR` /
    /// `SCNB_SCALE_FACTOR` (pre-Amber-11). It is not a pair list: build the
    /// full one with `ff::potential::intramolecular_pairs`, which keeps these
    /// cells.
    ///
    /// # Errors
    ///
    /// The OS error of a path that cannot be read;
    /// [`InvalidData`](std::io::ErrorKind::InvalidData) for what
    /// [`read`](ForceFieldReader::read) and the structure reader refuse, and a
    /// 1-4 row on a bonded pair or an angle's ends (sander prices that pair as
    /// 1-4, molrs excludes it).
    pub fn read_system<'a>(
        &self,
        src: impl Into<Source<'a>>,
    ) -> std::io::Result<(ForceField, Frame)> {
        system(&src.into().bytes()?).map_err(invalid_data)
    }
}

/// The force field and frame of prmtop bytes
/// ([`AmberPrmtopForcefieldReader::read_system`]).
fn system(bytes: &[u8]) -> Result<(ForceField, Frame), String> {
    let sections = parse_flag_sections(bytes).map_err(|e| e.to_string())?;
    let ff = build_forcefield(&sections)?;
    let mut frame = frame_from_sections(&sections).map_err(|e| e.to_string())?;
    if let Some(pairs) = one_four_pairs(&sections, &frame)? {
        frame.insert(molrs::core::schema::block_names::PAIRS, pairs);
    }
    Ok((ff, frame))
}

impl ForceFieldReader for AmberPrmtopForcefieldReader {
    fn read<'a>(&self, src: impl Into<Source<'a>>) -> std::io::Result<ForceField> {
        let bytes = src.into().bytes()?;
        let sections = parse_flag_sections(&*bytes)?;
        build_forcefield(&sections).map_err(invalid_data)
    }
}

fn ico_lookup(n_types: usize, iac_i: usize, iac_j: usize, nb_index: &[i64]) -> Result<i64, String> {
    let index = n_types
        .saturating_mul(iac_i.saturating_sub(1))
        .saturating_add(iac_j.saturating_sub(1));
    let nb = *nb_index.get(index).unwrap_or(&0);
    if nb < 0 {
        return Err("10-12 interactions are not supported".into());
    }
    Ok(nb)
}

/// One A/B table read through the ICO matrix: the self `(σ, ε)` of each LJ
/// class (1-based, index 0 unused) and every off-diagonal `(i < j)` entry.
struct LjTable {
    self_params: Vec<Lj>,
    cross: Vec<((usize, usize), Lj)>,
}

fn read_lj_table(
    n_types: usize,
    nb_index: &[i64],
    acoef: &[f64],
    bcoef: &[f64],
) -> Result<LjTable, String> {
    let entry = |i: usize, j: usize| -> Result<Option<Lj>, String> {
        let nb = ico_lookup(n_types, i, j, nb_index)?;
        if nb == 0 {
            return Ok(None);
        }
        Ok(Some(lj_ab_to_sigma_epsilon(
            entry(acoef, "LENNARD_JONES_ACOEF", nb)?,
            entry(bcoef, "LENNARD_JONES_BCOEF", nb)?,
        )))
    };
    let mut self_params = vec![(1.0, 0.0); n_types + 1];
    for (t, slot) in self_params.iter_mut().enumerate().skip(1) {
        if let Some(p) = entry(t, t)? {
            *slot = p;
        }
    }
    let mut cross = Vec::new();
    for i in 1..=n_types {
        for j in (i + 1)..=n_types {
            if let Some(p) = entry(i, j)? {
                cross.push(((i, j), p));
            }
        }
    }
    Ok(LjTable { self_params, cross })
}

fn rel(got: f64, exp: f64) -> f64 {
    let denom = got.abs().max(exp.abs());
    if denom == 0.0 {
        0.0
    } else {
        (got - exp).abs() / denom
    }
}

/// Whether `(σ, ε)` prices as the Lorentz–Berthelot mix of two self terms.
///
/// An entry with ε = 0 between types whose mix has ε = 0 (a type with no van
/// der Waals term, GAFF's `ho`, beside any other) is that mix whatever its σ:
/// neither prices anything. tleap writes a zero `A`/`B` for it, which reads
/// as σ = 1 ([`lj_ab_to_sigma_epsilon`]), not the mixed σ.
fn is_lorentz_berthelot(p: Lj, a: Lj, b: Lj) -> bool {
    let epsilon = (a.1 * b.1).sqrt();
    if p.1 == 0.0 && epsilon == 0.0 {
        return true;
    }
    rel(p.0, 0.5 * (a.0 + b.0)) <= 1e-6 && rel(p.1, epsilon) <= 1e-6
}

/// The LJ rows of a file: `(name, σ, ε, σ14, ε14)` self rows (one per type
/// name, from its class's diagonal) and `(a, b, σ, ε, σ14, ε14)` cross rows,
/// `a < b` bytewise, one per pair of type names on two classes whose
/// off-diagonal entry — in the regular table or, with `one_four`, the 1-4
/// table — is not the Lorentz–Berthelot mix of the classes' self terms
/// (CHARMM NBFIX, ParmEd `changeLJPair`). A Lorentz–Berthelot entry gives no
/// row: `mixing = arithmetic` reproduces it.
///
/// Atoms sharing a type name share its LJ class (the atom-type definition
/// makes a second class under one name a `TypeConflict`), so their rows are
/// equal and the definition rule collapses them to one pair type. Several
/// names on one LJ class each get the cross rows of that class.
#[allow(clippy::type_complexity)]
fn lj_rows(
    atom_types: &[String],
    type_index: &[i64],
    regular: &LjTable,
    one_four: Option<&LjTable>,
) -> Result<(Vec<(String, Lj, Lj)>, Vec<(String, String, Lj, Lj)>), String> {
    let n_types = regular.self_params.len() - 1;
    let class = |i: usize| -> Result<usize, String> {
        match type_index.get(i).copied() {
            Some(t) if t >= 1 && t as usize <= n_types => Ok(t as usize),
            Some(t) => Err(format!(
                "ATOM_TYPE_INDEX of atom {} is {t}, outside the {n_types} Lennard-Jones types",
                i + 1
            )),
            None => Err(format!("ATOM_TYPE_INDEX has no entry for atom {}", i + 1)),
        }
    };
    let self14 = |t: usize| one_four.map_or(regular.self_params[t], |o| o.self_params[t]);
    let mut names_of: Vec<BTreeSet<&str>> = vec![BTreeSet::new(); n_types + 1];
    let mut selves = Vec::new();
    for (i, name) in atom_types.iter().enumerate() {
        if name.is_empty() {
            continue;
        }
        let t = class(i)?;
        names_of[t].insert(name.as_str());
        selves.push((name.clone(), regular.self_params[t], self14(t)));
    }
    let cross14: HashMap<(usize, usize), Lj> = one_four
        .map(|o| o.cross.iter().copied().collect())
        .unwrap_or_default();
    let mut cross = Vec::new();
    for &((i, j), p) in &regular.cross {
        let (si, sj) = (regular.self_params[i], regular.self_params[j]);
        let p14 = cross14.get(&(i, j)).copied().unwrap_or(p);
        let mixed14 = one_four.is_none() || is_lorentz_berthelot(p14, self14(i), self14(j));
        if is_lorentz_berthelot(p, si, sj) && mixed14 {
            continue;
        }
        for a in &names_of[i] {
            for b in &names_of[j] {
                let (a, b) = if a <= b { (a, b) } else { (b, a) };
                cross.push(((*a).to_owned(), (*b).to_owned(), p, p14));
            }
        }
    }
    Ok((selves, cross))
}

/// Entry `pointer` (1-based, as prmtop pointers are) of the section `name`.
///
/// # Errors
///
/// A pointer outside the section, named with the section: a parameter the
/// file does not hold is never read as 0.
fn entry(table: &[f64], name: &str, pointer: i64) -> Result<f64, String> {
    usize::try_from(pointer - 1)
        .ok()
        .and_then(|i| table.get(i).copied())
        .ok_or_else(|| format!("{name} has no entry {pointer} (it holds {})", table.len()))
}

/// Whether two A/B tables are the same numbers.
fn same_table(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| rel(*x, *y) <= 1e-12)
}

fn periodic_params(terms: &[TorsionTerm]) -> Params {
    let mut owned = Vec::with_capacity(3 * terms.len());
    for (m, t) in terms.iter().enumerate() {
        let i = m + 1;
        owned.push((format!("k{i}"), t.k));
        owned.push((format!("periodicity{i}"), t.periodicity));
        owned.push((format!("phase{i}"), t.phase.to_degrees()));
    }
    let refs: Vec<(&str, f64)> = owned.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    Params::from_iter(refs.iter().copied())
}

fn first_i64(sections: &HashMap<String, Vec<String>>, key: &str) -> Result<Option<i64>, String> {
    Ok(section::<i64>(sections, key)?.first().copied())
}

/// What no force field here can hold, refused by name.
fn refuse_unsupported(sections: &HashMap<String, Vec<String>>) -> Result<(), String> {
    if sections.contains_key("LENNARD_JONES_CCOEF") {
        return Err("12-6-4 Lennard-Jones prmtop files are not supported".into());
    }
    if first_i64(sections, "IPOL")?.unwrap_or(0) > 0 {
        return Err(
            "polarizable (IPOL > 0) prmtop files are not supported: the induced dipoles have \
             no Class-I form"
                .into(),
        );
    }
    for flag in ["HBOND_ACOEF", "HBOND_BCOEF"] {
        if section::<f64>(sections, flag)?.iter().any(|&v| v != 0.0) {
            return Err(format!(
                "10-12 interactions are not supported (%FLAG {flag} has a non-zero entry)"
            ));
        }
    }
    Ok(())
}

fn def<'s>(
    ff: &'s mut ForceField,
    category: &str,
    name: &str,
    params: Params,
) -> Result<&'s mut crate::ff::forcefield::Style, String> {
    ff.def_style(category, name, params)
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Build ForceField
// ---------------------------------------------------------------------------

fn build_forcefield(sections: &HashMap<String, Vec<String>>) -> Result<ForceField, String> {
    let pointers = sections
        .get("POINTERS")
        .ok_or_else(|| "POINTERS section missing".to_string())?;
    let ptr_vals: Vec<i64> = parse_tokens(pointers)?;
    // NATOM, NTYPES, … — first two always present when POINTERS is valid.
    let n_atom = *ptr_vals.first().unwrap_or(&0) as usize;
    let n_types = *ptr_vals.get(1).unwrap_or(&0) as usize;
    refuse_unsupported(sections)?;
    let chamber = is_chamber(sections);

    let atom_types = atom_type_names(sections, n_atom)?;
    let type_name = |atom: usize| atom_types.get(atom).map(String::as_str).unwrap_or_default();

    let masses: Vec<f64> = section::<f64>(sections, "MASS")?;
    let type_index: Vec<i64> = parse_tokens(
        sections
            .get("ATOM_TYPE_INDEX")
            .ok_or_else(|| "%FLAG ATOM_TYPE_INDEX section missing".to_string())?,
    )?;

    let mut ff = ForceField::new(if chamber { "CHARMM" } else { "AMBER" });
    // prmtop stores Å, kcal/mol and e: LAMMPS `real`.
    ff.set_units("real");

    // Atom types: one per name; id from ATOM_TYPE_INDEX (the LJ class). Every
    // atom defines its type, so two atoms of one name with a different class
    // or mass are a TypeConflict.
    {
        let style = def(&mut ff, "atom", "full", Params::new())?;
        for (i, name) in atom_types.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            let id = type_index
                .get(i)
                .copied()
                .ok_or_else(|| format!("ATOM_TYPE_INDEX has no entry for atom {}", i + 1))?
                as f64;
            let mass = entry(&masses, "MASS", i as i64 + 1)?;
            style
                .def_type(name, &[], Params::from_iter([("id", id), ("mass", mass)]))
                .map_err(|e| e.to_string())?;
        }
    }

    // Bonds: one type per endpoint type pair in `TypeName::orient`'s spelling;
    // k = RK (AMBER's and LAMMPS's K). Every bond row defines its type, so two
    // rows under one name with different k/r0 are a TypeConflict.
    let bond_k = section::<f64>(sections, "BOND_FORCE_CONSTANT")?;
    let bond_r0 = section::<f64>(sections, "BOND_EQUIL_VALUE")?;
    let mut bond_ptrs = section::<i64>(sections, "BONDS_INC_HYDROGEN")?;
    bond_ptrs.extend(section::<i64>(sections, "BONDS_WITHOUT_HYDROGEN")?);
    {
        let style = def(&mut ff, "bond", "harmonic", Params::new())?;
        for chunk in bond_ptrs.as_chunks::<3>().0 {
            let a = chunk[0];
            let b = chunk[1];
            if a < 0 || b < 0 {
                return Err(format!("negative bonded atom pointers ({a}, {b})"));
            }
            let i = (a / 3) as usize;
            let j = (b / 3) as usize;
            let ends = TypeName::orient(&[type_name(i), type_name(j)]);
            let k = entry(&bond_k, "BOND_FORCE_CONSTANT", chunk[2])?;
            let r0 = entry(&bond_r0, "BOND_EQUIL_VALUE", chunk[2])?;
            style
                .def_type(
                    TypeName::join(&ends)?.as_str(),
                    &ends,
                    Params::from_iter([("k", k), ("r0", r0)]),
                )
                .map_err(|e| e.to_string())?;
        }
    }

    // Angles: k = TK; theta0 radians in prmtop → degrees. As for bonds, every
    // angle row defines its type. A chamber file's angles are CHARMM's, each
    // with its Urey–Bradley term (0, 0 without one): `angle charmm`.
    let angle_k = section::<f64>(sections, "ANGLE_FORCE_CONSTANT")?;
    let angle_eq = section::<f64>(sections, "ANGLE_EQUIL_VALUE")?;
    let mut angle_ptrs = section::<i64>(sections, "ANGLES_INC_HYDROGEN")?;
    angle_ptrs.extend(section::<i64>(sections, "ANGLES_WITHOUT_HYDROGEN")?);
    let mut angles: Vec<([usize; 3], i64)> = Vec::with_capacity(angle_ptrs.len() / 4);
    for chunk in angle_ptrs.as_chunks::<4>().0 {
        let (a, b, c) = (chunk[0], chunk[1], chunk[2]);
        if a < 0 || b < 0 || c < 0 {
            return Err(format!("negative angle atom pointers ({a}, {b}, {c})"));
        }
        angles.push((
            [(a / 3) as usize, (b / 3) as usize, (c / 3) as usize],
            chunk[3],
        ));
    }
    let urey_bradley = if chamber {
        urey_bradley_of_angles(&angles, &chamber_urey_bradleys(sections, n_atom)?)?
    } else {
        vec![None; angles.len()]
    };
    {
        let style = def(
            &mut ff,
            "angle",
            if chamber { "charmm" } else { "harmonic" },
            Params::new(),
        )?;
        for ((atoms, tid), ub) in angles.iter().zip(&urey_bradley) {
            let ends = TypeName::orient(&atoms.map(type_name));
            let k = entry(&angle_k, "ANGLE_FORCE_CONSTANT", *tid)?;
            let theta0 = entry(&angle_eq, "ANGLE_EQUIL_VALUE", *tid)?.to_degrees();
            let mut params = Params::from_iter([("k", k), ("theta0", theta0)]);
            if chamber {
                let (k_ub, r_ub) = ub.unwrap_or((0.0, 0.0));
                params.set("k_ub", k_ub);
                params.set("r_ub", r_ub);
            }
            style
                .def_type(TypeName::join(&ends)?.as_str(), &ends, params)
                .map_err(|e| e.to_string())?;
        }
    }

    // Torsions. FileFormats: 3rd pointer negative → no 1-4; 4th negative →
    // improper. Potential: PK·[1 + cos(n·φ − phase)], phase stored in degrees
    // — LAMMPS fourier / molrs periodic, no form factor on K. The rows of one
    // atom quartet (and each row's negative-PN chain) are one torsion; every
    // torsion defines its type; a further distinct set of terms on one
    // quartet is the type `<quartet>@<n>` (`proper_type_names`).
    let dih_k = section::<f64>(sections, "DIHEDRAL_FORCE_CONSTANT")?;
    let dih_phase = section::<f64>(sections, "DIHEDRAL_PHASE")?;
    let dih_per = section::<f64>(sections, "DIHEDRAL_PERIODICITY")?;
    let mut dih_ptrs = section::<i64>(sections, "DIHEDRALS_INC_HYDROGEN")?;
    dih_ptrs.extend(section::<i64>(sections, "DIHEDRALS_WITHOUT_HYDROGEN")?);
    let tables = TorsionTables {
        k: &dih_k,
        periodicity: &dih_per,
        phase: &dih_phase,
    };
    let torsions = decode_torsions(&dih_ptrs, &atom_types, Some(tables))?;

    // 1-4: the divisor most 1-4 rows carry is the field's weight; the frame
    // reader gives the pairs whose rows carry another their own scales.
    let scee = section::<f64>(sections, "SCEE_SCALE_FACTOR")?;
    let scnb = section::<f64>(sections, "SCNB_SCALE_FACTOR")?;
    // A file without SCEE/SCNB (pre-Amber-11) is an AMBER force field's.
    let one_four = one_four_weights(
        &dih_ptrs,
        n_atom,
        &scee,
        &scnb,
        &dih_per,
        Some((AMBER_SCEE, AMBER_SCNB)),
    )?
    .ok_or("1-4 weights unknown although AMBER's divisors stand in")?;
    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, one_four.lj],
        coul: [0.0, 0.0, one_four.coul],
    });

    {
        let style = def(&mut ff, "dihedral", "periodic", Params::new())?;
        let names = proper_type_names(&torsions, &atom_types)?;
        for (t, name) in torsions.iter().zip(names) {
            let Some(name) = name else { continue };
            let ends = t.types(&atom_types);
            style
                .def_type(&name, &ends, periodic_params(&canonical_terms(&t.terms)))
                .map_err(|e| e.to_string())?;
        }
    }
    let mut periodic_impropers: BTreeSet<String> = BTreeSet::new();
    if torsions.iter().any(|t| t.improper) {
        let style = def(&mut ff, "improper", "periodic", Params::new())?;
        for t in torsions.iter().filter(|t| t.improper) {
            let ends = t.types(&atom_types);
            for (name, term) in t.improper_rows(&atom_types)? {
                let term = term.ok_or_else(|| format!("improper {name} has no terms"))?;
                style
                    .def_type(
                        &name,
                        &ends,
                        Params::from_iter([
                            ("k", term.k),
                            ("periodicity", term.periodicity),
                            ("phase", term.phase.to_degrees()),
                        ]),
                    )
                    .map_err(|e| e.to_string())?;
                periodic_impropers.insert(name);
            }
        }
    }

    if chamber {
        def_charmm_impropers(&mut ff, sections, &atom_types, &periodic_impropers)?;
    }
    if let Some(cmap) = cmap_terms(sections, &atom_types)? {
        let style = def(&mut ff, "cmap", "charmm", Params::new())?;
        for (name, ends, n, grid) in cmap.maps {
            let grid = ndarray::Array2::from_shape_vec((n, n), grid)
                .map_err(|e| e.to_string())?
                .into_dyn();
            let mut params = Params::new();
            params.set("grid", grid);
            let ends: Vec<&str> = ends.iter().map(String::as_str).collect();
            style
                .def_type(&name, &ends, params)
                .map_err(|e| e.to_string())?;
        }
    }

    // Pair LJ from the full ICO matrix. FileFormats:
    // index = ICO[NTYPES*(IAC(i)-1) + IAC(j)] (1-based Fortran).
    // No pair style gets a `cutoff`: a prmtop carries none (it lives in the
    // mdin), so the caller declares it.
    let nb_index = section::<i64>(sections, "NONBONDED_PARM_INDEX")?;
    let acoef = section::<f64>(sections, "LENNARD_JONES_ACOEF")?;
    let bcoef = section::<f64>(sections, "LENNARD_JONES_BCOEF")?;
    let regular = read_lj_table(n_types, &nb_index, &acoef, &bcoef)?;
    if chamber {
        let a14 = section::<f64>(sections, "LENNARD_JONES_14_ACOEF")?;
        let b14 = section::<f64>(sections, "LENNARD_JONES_14_BCOEF")?;
        // The 1-4 table matters only where it differs from the regular one.
        let distinct = !(a14.is_empty() && b14.is_empty())
            && !(same_table(&a14, &acoef) && same_table(&b14, &bcoef));
        let table14 = if distinct {
            Some(read_lj_table(n_types, &nb_index, &a14, &b14)?)
        } else {
            None
        };
        def_lj_charmm(
            &mut ff,
            &atom_types,
            &type_index,
            &regular,
            table14.as_ref(),
        )?;
        def(
            &mut ff,
            "pair",
            "coul/charmm",
            Params::from_iter([
                ("coulomb", CHARMM_COULOMB),
                ("dielectric", VACUUM_DIELECTRIC),
            ]),
        )?;
    } else {
        let (selves, cross) = lj_rows(&atom_types, &type_index, &regular, None)?;
        // The off-diagonal entries that are no row are Lorentz–Berthelot
        // mixes, so the style states that rule rather than lean on a default.
        let mut lj = Params::new();
        lj.set("mixing", CombiningRule::Arithmetic.name());
        let style = def(&mut ff, "pair", "lj/cut", lj)?;
        for (tname, (sigma, epsilon), _) in &selves {
            style
                .def_type(
                    tname,
                    &[tname],
                    Params::from_iter([("epsilon", *epsilon), ("sigma", *sigma)]),
                )
                .map_err(|e| e.to_string())?;
        }
        for (a, b, (sigma, epsilon), _) in &cross {
            style
                .def_type(
                    TypeName::pair(a, b)?.as_str(),
                    &[a, b],
                    Params::from_iter([("epsilon", *epsilon), ("sigma", *sigma)]),
                )
                .map_err(|e| e.to_string())?;
        }
        def(
            &mut ff,
            "pair",
            "coul/cut",
            Params::from_iter([
                ("coulomb", AMBER_COULOMB),
                ("dielectric", VACUUM_DIELECTRIC),
            ]),
        )?;
    }

    Ok(ff)
}

/// AMBER's 1-4 pairs and their weights.
///
/// sander prices the 1-4 pair `(i, l)` of every proper torsion row whose
/// 3rd pointer is not negative, once per row (never an improper's, whatever
/// its 3rd pointer — checked with sander), at `1/SCEE` (Coulomb) and `1/SCNB`
/// (van der Waals) of the row's type. `coul` / `lj` are the force field's
/// `special_bonds` 1-4 weights: the reciprocal of the divisor most rows
/// carry (the first such value on a tie). `pairs` maps each 1-4 pair
/// `(lo, hi)` (0-based) to its summed `(coul, lj)` weight.
#[derive(Debug, Clone, PartialEq)]
struct OneFourWeights {
    coul: f64,
    lj: f64,
    pairs: std::collections::BTreeMap<(usize, usize), (f64, f64)>,
}

/// [`OneFourWeights`] of the torsion rows `pointers`.
///
/// A file without `SCEE_SCALE_FACTOR` / `SCNB_SCALE_FACTOR` (pre-Amber-11)
/// states no divisors; they are then its force field's. `default` is the
/// `(SCEE, SCNB)` the caller assumes for such a file (the force field takes
/// AMBER's, `core::constants::AMBER_SCEE` / `AMBER_SCNB`); with `None` the weights of such a file —
/// and the field weights of a file with no 1-4 row — are unknown,
/// `Ok(None)` (the per-pair rows assume nothing).
///
/// # Errors
///
/// A 1-4 row whose type has no divisor or a non-positive one, naming the
/// flag and the type; and a 1-4 row whose
/// type continues a multi-term chain (negative periodicity), on which sander
/// prices the 1-4 pair once per chained term and with a Coulomb factor
/// inconsistent with its other 1-4 pairs — no force field holds that.
fn one_four_weights(
    pointers: &[i64],
    n_atoms: usize,
    scee: &[f64],
    scnb: &[f64],
    periodicity: &[f64],
    default: Option<(f64, f64)>,
) -> Result<Option<OneFourWeights>, String> {
    let rows: Vec<TorsionRow> = torsion_rows(pointers, n_atoms)?
        .into_iter()
        .filter(|r| !r.exclude_14 && !r.improper)
        .collect();
    let divisor = |values: &[f64], flag: &str, tid: i64, default: Option<f64>| {
        if values.is_empty() {
            return Ok(default);
        }
        let v = usize::try_from(tid - 1)
            .ok()
            .and_then(|i| values.get(i).copied())
            .ok_or_else(|| format!("{flag} type {tid} is out of range"))?;
        if v <= 0.0 {
            return Err(format!("{flag} type {tid} has non-positive divisor {v}"));
        }
        Ok(Some(v))
    };
    let mut pairs = std::collections::BTreeMap::new();
    // How many 1-4 rows carry each SCEE / SCNB divisor.
    let (mut ce, mut cn): (Vec<Tally>, Vec<Tally>) = (Vec::new(), Vec::new());
    for row in &rows {
        let [i, _, _, l] = row.atoms;
        let chained = usize::try_from(row.tid - 1)
            .ok()
            .and_then(|i| periodicity.get(i))
            .is_some_and(|&pn| pn < 0.0);
        if chained {
            return Err(format!(
                "dihedral type {} continues a multi-term chain (negative periodicity) on a row \
                 with a 1-4 pair (atoms {} and {}): sander prices that pair once per chained term \
                 and at a Coulomb factor unlike its other 1-4 pairs, which no force field holds",
                row.tid,
                i + 1,
                l + 1
            ));
        }
        let e = divisor(scee, "SCEE_SCALE_FACTOR", row.tid, default.map(|d| d.0))?;
        let n = divisor(scnb, "SCNB_SCALE_FACTOR", row.tid, default.map(|d| d.1))?;
        let (Some(e), Some(n)) = (e, n) else {
            return Ok(None);
        };
        tally(&mut ce, e);
        tally(&mut cn, n);
        let w = pairs.entry((i.min(l), i.max(l))).or_insert((0.0, 0.0));
        w.0 += 1.0 / e;
        w.1 += 1.0 / n;
    }
    let dominant = |c: &[Tally], default: Option<f64>| {
        c.iter()
            .fold(None::<Tally>, |best, &(v, n)| match best {
                Some((_, m)) if m >= n => best,
                _ => Some((v, n)),
            })
            .map(|(v, _)| v)
            .or(default)
    };
    let (Some(e), Some(n)) = (
        dominant(&ce, default.map(|d| d.0)),
        dominant(&cn, default.map(|d| d.1)),
    ) else {
        return Ok(None);
    };
    Ok(Some(OneFourWeights {
        coul: 1.0 / e,
        lj: 1.0 / n,
        pairs,
    }))
}

/// A chamber Urey–Bradley term: the angle's end atoms (0-based), `K_ub`
/// (kcal/mol/Å², no ½) and `r_ub` (Å).
#[derive(Debug, Clone, Copy, PartialEq)]
struct UreyBradley {
    ends: (usize, usize),
    k_ub: f64,
    r_ub: f64,
}

/// `CHARMM_UREY_BRADLEY` (atom, atom, type; 1-based atom numbers) with its
/// force-constant and equilibrium tables; empty when the file has none.
fn chamber_urey_bradleys(
    sections: &HashMap<String, Vec<String>>,
    n_atoms: usize,
) -> Result<Vec<UreyBradley>, String> {
    const FLAG: &str = "CHARMM_UREY_BRADLEY";
    let rows: Vec<i64> = section(sections, FLAG)?;
    let k: Vec<f64> = section(sections, "CHARMM_UREY_BRADLEY_FORCE_CONSTANT")?;
    let r: Vec<f64> = section(sections, "CHARMM_UREY_BRADLEY_EQUIL_VALUE")?;
    if !rows.len().is_multiple_of(3) {
        return Err(format!(
            "%FLAG {FLAG} has {} entries, not triples",
            rows.len()
        ));
    }
    rows.as_chunks::<3>()
        .0
        .iter()
        .map(|&[i, j, t]| {
            Ok(UreyBradley {
                ends: (
                    atom_number(FLAG, i, n_atoms)?,
                    atom_number(FLAG, j, n_atoms)?,
                ),
                k_ub: table_value("CHARMM_UREY_BRADLEY_FORCE_CONSTANT", &k, t)?,
                r_ub: table_value("CHARMM_UREY_BRADLEY_EQUIL_VALUE", &r, t)?,
            })
        })
        .collect()
}

/// A value and how many times it was seen.
type Tally = (f64, usize);

/// Count `v` in `counts` (values equal to 1e-6 relative are one value).
fn tally(counts: &mut Vec<Tally>, v: f64) {
    match counts
        .iter_mut()
        .find(|(u, _)| (u - v).abs() <= 1e-6 * u.abs().max(v.abs()))
    {
        Some(slot) => slot.1 += 1,
        None => counts.push((v, 1)),
    }
}

/// The `pairs` rows of [`AmberPrmtopForcefieldReader::read_system`]: the 1-4
/// pairs of `frame` (the structure reader's) whose summed sander weight
/// differs from the field's (`one_four_weights`); `None` when none does.
///
/// sander prices a pair at full weight unless `EXCLUDED_ATOMS_LIST` names it,
/// plus `1/SCEE`, `1/SCNB` of its type for every torsion row that lists it as
/// its 1-4 pair. For the 1-4 pairs of the topology (the end atoms of a
/// proper, neither bonded nor an angle's ends) that sum is the pair's
/// weight. A 1-4 row on a bonded pair or an angle's ends is an `Err`: molrs
/// excludes those pairs.
fn one_four_pairs(
    sections: &HashMap<String, Vec<String>>,
    frame: &Frame,
) -> Result<Option<Block>, String> {
    use std::collections::{BTreeMap, HashSet};
    let n_atoms = frame.get("atoms").and_then(|b| b.n_rows()).unwrap_or(0);
    let mut dihe_ptrs = section::<i64>(sections, "DIHEDRALS_INC_HYDROGEN")?;
    dihe_ptrs.extend(section::<i64>(sections, "DIHEDRALS_WITHOUT_HYDROGEN")?);
    // No divisors in the file (pre-Amber-11): no pair is priced on its own.
    let Some(weights) = one_four_weights(
        &dihe_ptrs,
        n_atoms,
        &section::<f64>(sections, "SCEE_SCALE_FACTOR")?,
        &section::<f64>(sections, "SCNB_SCALE_FACTOR")?,
        &section::<f64>(sections, "DIHEDRAL_PERIODICITY")?,
        None,
    )?
    else {
        return Ok(None);
    };
    let key = |a: Idx, b: Idx| (a.min(b) as usize, a.max(b) as usize);
    // The (first, last) atom pairs of a frame block's rows.
    let ends = |block: &str, last: &str| -> Vec<(usize, usize)> {
        let Some(b) = frame.get(block) else {
            return Vec::new();
        };
        match (
            b.get(keys::ATOMI).and_then(|c| c.get::<u64>()),
            b.get(last).and_then(|c| c.get::<u64>()),
        ) {
            (Some(i), Some(j)) => i.iter().zip(j.iter()).map(|(&a, &b)| key(a, b)).collect(),
            _ => Vec::new(),
        }
    };
    let close: HashSet<(usize, usize)> = ends("bonds", keys::ATOMJ)
        .into_iter()
        .chain(ends("angles", keys::ATOMK))
        .collect();
    if let Some(&(i, j)) = weights.pairs.keys().find(|p| close.contains(p)) {
        return Err(format!(
            "atoms {} and {} are a torsion's 1-4 pair (a non-negative 3rd pointer) and \
             also bonded or an angle's ends: sander prices that pair as 1-4, molrs \
             excludes it",
            i + 1,
            j + 1
        ));
    }
    let excluded: Option<HashSet<(usize, usize)>> = frame
        .get("exclusions")
        .map(|_| ends("exclusions", keys::ATOMJ).into_iter().collect());
    let mut topology_14: BTreeMap<(usize, usize), (F, F)> = BTreeMap::new();
    for pair in ends("dihedrals", keys::ATOML) {
        if close.contains(&pair) {
            continue;
        }
        let (coul, lj) = weights.pairs.get(&pair).copied().unwrap_or((0.0, 0.0));
        // Without an exclusion list every such pair is taken as excluded.
        let full = match &excluded {
            Some(set) if !set.contains(&pair) => 1.0,
            _ => 0.0,
        };
        topology_14.insert(pair, (coul + full, lj + full));
    }
    let differs = |a: F, b: F| (a - b).abs() > 1e-12 * a.abs().max(b.abs()).max(1.0);
    let rows: Vec<((usize, usize), (F, F))> = topology_14
        .into_iter()
        .filter(|&(_, (c, l))| differs(c, weights.coul) || differs(l, weights.lj))
        .collect();
    if rows.is_empty() {
        return Ok(None);
    }
    let column_err = |e: molrs::core::BlockError| e.to_string();
    let mut block = Block::new();
    for (column, pick) in [(keys::ATOMI, 0usize), (keys::ATOMJ, 1usize)] {
        let ids: Vec<Idx> = rows
            .iter()
            .map(|r| if pick == 0 { r.0.0 } else { r.0.1 } as Idx)
            .collect();
        block
            .insert(column, ndarray::Array1::from_vec(ids).into_dyn(), None)
            .map_err(column_err)?;
    }
    block
        .insert(
            keys::IS_14,
            ndarray::Array1::from_vec(vec![true; rows.len()]).into_dyn(),
            None,
        )
        .map_err(column_err)?;
    for (column, pick, field) in [
        ("coul_scale", 0usize, weights.coul),
        ("lj_scale", 1usize, weights.lj),
    ] {
        let values: Vec<F> = rows
            .iter()
            .map(|r| if pick == 0 { r.1.0 } else { r.1.1 })
            .collect();
        let valid: Vec<bool> = values.iter().map(|&v| differs(v, field)).collect();
        if valid.iter().any(|&v| v) {
            block
                .insert(
                    column,
                    ndarray::Array1::from_vec(values).into_dyn(),
                    Some(valid),
                )
                .map_err(column_err)?;
        }
    }
    Ok(Some(block))
}

/// The Urey–Bradley `(k_ub, r_ub)` of each angle: the chamber term whose two
/// atoms are its end atoms. A term on no angle, or on the ends of two (a
/// four-membered ring), has no `angle charmm` form and is an `Err`.
fn urey_bradley_of_angles(
    angles: &[([usize; 3], i64)],
    terms: &[UreyBradley],
) -> Result<Vec<Option<(f64, f64)>>, String> {
    let mut by_ends: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (n, (atoms, _)) in angles.iter().enumerate() {
        let (i, k) = (atoms[0], atoms[2]);
        by_ends.entry((i.min(k), i.max(k))).or_default().push(n);
    }
    let mut out = vec![None; angles.len()];
    for ub in terms {
        let (i, k) = ub.ends;
        let key = (i.min(k), i.max(k));
        match by_ends.get(&key).map(Vec::as_slice) {
            Some(&[n]) => {
                if out[n].is_some() {
                    return Err(format!(
                        "atoms {} and {} carry two Urey-Bradley terms",
                        key.0 + 1,
                        key.1 + 1
                    ));
                }
                out[n] = Some((ub.k_ub, ub.r_ub));
            }
            Some(_) => {
                return Err(format!(
                    "Urey-Bradley term on atoms {} and {} spans the ends of several angles; \
                     `angle charmm` gives each angle its own term, which would count it more \
                     than once",
                    key.0 + 1,
                    key.1 + 1
                ));
            }
            None => {
                return Err(format!(
                    "Urey-Bradley term on atoms {} and {} is on no angle; `angle charmm` holds \
                     it with its angle",
                    key.0 + 1,
                    key.1 + 1
                ));
            }
        }
    }
    Ok(out)
}

/// A chamber file's CHARMM impropers as `improper harmonic`
/// (`K (χ − chi0)²`, LAMMPS's `K`, centre first as the file lists them).
///
/// LAMMPS prices χ = |φ|, CHARMM the signed φ: the two agree when ψ₀ is 0°
/// or 180°, and any other ψ₀ is an `Err`.
fn def_charmm_impropers(
    ff: &mut ForceField,
    sections: &HashMap<String, Vec<String>>,
    atom_types: &[String],
    periodic: &BTreeSet<String>,
) -> Result<(), String> {
    let impropers = chamber_impropers(sections, atom_types.len())?;
    if impropers.is_empty() {
        return Ok(());
    }
    let style = def(ff, "improper", "harmonic", Params::new())?;
    for imp in impropers {
        let ends = imp.atoms.map(|a| atom_types[a].as_str());
        let name = TypeName::join(&ends)?.to_string();
        if periodic.contains(&name) {
            return Err(format!(
                "improper type {name} is both an AMBER (periodic) and a CHARMM (harmonic) \
                 improper; one name cannot select two styles"
            ));
        }
        let chi0 = imp.psi0.to_degrees();
        let on_axis = chi0.abs() < 1e-6 || (chi0.abs() - 180.0).abs() < 1e-6;
        if !on_axis {
            return Err(format!(
                "CHARMM improper {name}: psi0 = {chi0}° — LAMMPS's improper harmonic prices \
                 |psi|, which equals CHARMM's signed psi only for psi0 of 0° or 180°"
            ));
        }
        style
            .def_type(
                &name,
                &ends,
                Params::from_iter([("k", imp.k), ("chi0", chi0)]),
            )
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// A chamber file's Lennard-Jones as `lj/charmm`: `epsilon`, `sigma` from the
/// regular table and, when its 1-4 table differs, `epsilon14`, `sigma14` from
/// it with the style's `one_four = "epsilon14"` (the 1-4 pairs, weighted by
/// `special_bonds`, are meant at ε₁₄/σ₁₄); cross rows where either table's
/// off-diagonal entry is not Lorentz–Berthelot.
fn def_lj_charmm(
    ff: &mut ForceField,
    atom_types: &[String],
    type_index: &[i64],
    regular: &LjTable,
    one_four: Option<&LjTable>,
) -> Result<(), String> {
    let (selves, cross) = lj_rows(atom_types, type_index, regular, one_four)?;
    let mut style_params = Params::new();
    style_params.set("mixing", CombiningRule::Arithmetic.name());
    if one_four.is_some() {
        style_params.set("one_four", "epsilon14");
    }
    let params = |p: Lj, p14: Lj| {
        let mut out = Params::from_iter([("epsilon", p.1), ("sigma", p.0)]);
        if one_four.is_some() {
            out.set("epsilon14", p14.1);
            out.set("sigma14", p14.0);
        }
        out
    };
    let style = def(ff, "pair", "lj/charmm", style_params)?;
    for (name, p, p14) in &selves {
        style
            .def_type(name, &[name], params(*p, *p14))
            .map_err(|e| e.to_string())?;
    }
    for (a, b, p, p14) in &cross {
        style
            .def_type(TypeName::pair(a, b)?.as_str(), &[a, b], params(*p, *p14))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The structure frame of `text` with the `pairs` rows
    /// [`AmberPrmtopForcefieldReader::read_system`] adds (the test prmtops hold
    /// no LJ tables, so the force field itself is not read).
    fn read_system(text: &str) -> Frame {
        let sections = parse_flag_sections(text.as_bytes()).unwrap();
        let mut frame = frame_from_sections(&sections).unwrap();
        if let Some(pairs) = one_four_pairs(&sections, &frame).unwrap() {
            frame.insert("pairs", pairs);
        }
        frame
    }

    use crate::io::amber::prmtop::tests::chain5_prmtop;

    /// SCEE = SCNB = 1 on every 1-4 row of a chamber file: uniform, no `pairs`.
    #[test]
    fn a_uniform_chamber_file_has_no_pairs() {
        let (_, frame) = AmberPrmtopForcefieldReader::new()
            .read_system(CHAMBER_MINI.as_bytes())
            .unwrap();
        assert!(frame.get("pairs").is_none());
    }

    /// [`chain5_prmtop`] with the dihedral tables and SCEE/SCNB: type 1
    /// (1.2 / 2.0) and type 2 (`scee2` / `scnb2`).
    fn chain5_scaled(torsions: &str, scee2: &str, scnb2: &str) -> String {
        chain5_prmtop(torsions)
            + &format!(
                "\
%FLAG DIHEDRAL_FORCE_CONSTANT
%FORMAT(5E16.8)
  1.00000000E+00  1.00000000E+00
%FLAG DIHEDRAL_PERIODICITY
%FORMAT(5E16.8)
  3.00000000E+00  2.00000000E+00
%FLAG DIHEDRAL_PHASE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00
%FLAG SCEE_SCALE_FACTOR
%FORMAT(5E16.8)
  1.20000000E+00  {scee2}
%FLAG SCNB_SCALE_FACTOR
%FORMAT(5E16.8)
  2.00000000E+00  {scnb2}
"
            )
    }

    /// The `(atomi, atomj, coul_scale, lj_scale)` of each `pairs` row, a null
    /// cell as `None`.
    fn pair_scales(frame: &Frame) -> Vec<(Idx, Idx, Option<F>, Option<F>)> {
        let pairs = frame.get("pairs").expect("pairs");
        let ai = pairs.get("atomi").unwrap().get::<u64>().unwrap();
        let aj = pairs.get("atomj").unwrap().get::<u64>().unwrap();
        let cell = |k: &str, r: usize| -> Option<F> {
            let col = pairs.get(k)?;
            if pairs.validity(k).is_some_and(|m| !m[r]) {
                return None;
            }
            Some(col.get::<f64>().unwrap()[[r]])
        };
        (0..ai.len())
            .map(|r| (ai[[r]], aj[[r]], cell("coul_scale", r), cell("lj_scale", r)))
            .collect()
    }

    /// Two 1-4 pairs, (0, 3) at type 1's 1.2 / 2.0 and (1, 4) at type 2's
    /// 1.0 / 1.0: the field takes the first divisor on the tie, and the
    /// frame's `pairs` carry the other pair's own weights.
    #[test]
    fn a_non_uniform_scee_gives_the_odd_pairs_their_scales() {
        let text = chain5_scaled(
            "       0       3       6       9       1       3       6       9      12       2",
            "1.00000000E+00",
            "1.00000000E+00",
        );
        let frame = read_system(&text);
        assert_eq!(pair_scales(&frame), vec![(1, 4, Some(1.0), Some(1.0))]);
        let flags = frame.get("pairs").unwrap().get("is_14").unwrap();
        assert!(flags.get::<bool>().unwrap().iter().all(|&f| f));
    }

    /// Only the weight that differs gets a cell: SCNB 2.0 on both types.
    #[test]
    fn a_pair_carries_only_the_scale_that_differs() {
        let text = chain5_scaled(
            "       0       3       6       9       1       3       6       9      12       2",
            "1.00000000E+00",
            "2.00000000E+00",
        );
        assert_eq!(
            pair_scales(&read_system(&text)),
            vec![(1, 4, Some(1.0), None)]
        );
    }

    /// A 1-4 pair two rows list is priced twice by sander: its scales are the
    /// sum; a 1-4 pair of the topology no row lists (every row's 3rd pointer
    /// negative, no exclusion list) is priced at 0.
    #[test]
    fn listed_twice_sums_and_listed_never_is_zero() {
        let text = chain5_scaled(
            "       0       3       6       9       1       0       3       6       9       1\n       3       6      -9      12       1",
            "1.20000000E+00",
            "2.00000000E+00",
        );
        let got = pair_scales(&read_system(&text));
        assert_eq!(got.len(), 2);
        let (i, j, c, l) = got[0];
        assert_eq!((i, j), (0, 3));
        assert!((c.unwrap() - 2.0 / 1.2).abs() < 1e-15 && (l.unwrap() - 1.0).abs() < 1e-15);
        assert_eq!(got[1], (1, 4, Some(0.0), Some(0.0)));
    }

    /// Three 1-4 rows: two at type 2 (1.0 / 1.0), one at type 1 (1.2 / 2.0).
    /// The field's weights are type 2's, the majority; each pair's summed
    /// weight is its rows'.
    #[test]
    fn one_four_weights_take_the_majority_divisor() {
        // Atoms 0..6; rows (0,3) type 1, (1,4) type 2, (2,5) type 2.
        let rows = [0, 3, 6, 9, 1, 3, 6, 9, 12, 2, 6, 9, 12, 15, 2];
        let w = one_four_weights(&rows, 6, &[1.2, 1.0], &[2.0, 1.0], &[3.0, 2.0], None)
            .unwrap()
            .unwrap();
        assert_eq!((w.coul, w.lj), (1.0, 1.0));
        assert_eq!(w.pairs[&(0, 3)], (1.0 / 1.2, 0.5));
        assert_eq!(w.pairs[&(1, 4)], (1.0, 1.0));
        assert_eq!(w.pairs.len(), 3);
    }

    /// No SCEE/SCNB section: the divisors are the caller's `default`, and
    /// unknown without one. An improper's row and a suppressed row (negative
    /// 3rd pointer) list no 1-4 pair.
    #[test]
    fn one_four_weights_default_and_skip_impropers() {
        let rows = [
            0, 3, 6, 9, 1, 0, 3, -6, 9, 1, 0, 3, -6, -9, 1, 0, 3, 6, -9, 1,
        ];
        assert_eq!(one_four_weights(&rows, 4, &[], &[], &[2.0], None), Ok(None));
        let w = one_four_weights(&rows, 4, &[], &[], &[2.0], Some((1.2, 2.0)))
            .unwrap()
            .unwrap();
        assert_eq!((w.coul, w.lj), (1.0 / 1.2, 0.5));
        assert_eq!(w.pairs.len(), 1);
        assert_eq!(w.pairs[&(0, 3)], (1.0 / 1.2, 0.5));
    }

    #[test]
    fn one_four_weights_refuse_a_chained_1_4_row_and_a_zero_divisor() {
        let rows = [0, 3, 6, 9, 1];
        let err =
            one_four_weights(&rows, 4, &[1.2, 1.2], &[2.0, 2.0], &[-3.0, 2.0], None).unwrap_err();
        assert!(err.contains("multi-term chain"), "{err}");
        let err = one_four_weights(&rows, 4, &[0.0], &[2.0], &[3.0], None).unwrap_err();
        assert!(err.contains("SCEE_SCALE_FACTOR type 1"), "{err}");
    }

    #[test]
    fn empty_prmtop_errors() {
        let r = AmberPrmtopForcefieldReader::new().read("%VERSION 1\n".as_bytes());
        assert!(r.is_err());
    }

    /// sander expands a negative PN into the next type, in a prmtop too
    /// (checked: one PN of an ff14SB dipeptide made negative adds the next
    /// type's term to every row of that type).
    #[test]
    fn multiterm_expansion() {
        use crate::io::amber::prmtop_tables::TorsionTables;
        // type 1 has PN=-2 → continues to type 2 (PN=+3)
        let (k, per, phase) = ([1.0, 2.0], [-2.0, 3.0], [0.0, 0.5]);
        let tables = TorsionTables {
            k: &k,
            periodicity: &per,
            phase: &phase,
        };
        let periods = |tid| -> Vec<f64> {
            tables
                .chain(tid)
                .unwrap()
                .iter()
                .map(|t| t.periodicity)
                .collect()
        };
        assert_eq!(periods(1), vec![2.0, 3.0]);
        assert_eq!(periods(2), vec![3.0]);
        assert!(tables.chain(3).is_err());
    }

    #[test]
    fn comment_lines_ignored() {
        // Minimal broken POINTERS after a comment — parser must not treat
        // %COMMENT as a numeric token.
        let text = "\
%VERSION VERSION_STAMP = V0001.000
%FLAG POINTERS
%COMMENT ignore me
%FORMAT(10I8)
       0       0
";
        // Missing sections → may fail later, but not on the comment token.
        let sections = parse_flag_sections(text.as_bytes()).unwrap();
        let vals: Vec<i64> = parse_tokens(sections.get("POINTERS").unwrap()).unwrap();
        assert_eq!(vals, vec![0, 0]);
    }

    /// 4-atom c3–c3–c3–hc chain, NTYPES=2.
    ///
    /// c3 self LJ from published GAFF R*=1.9080 Å, ε=0.1094 kcal/mol
    /// (`A=1043080.23`, `B=675.612248`). hc from GAFF R*=1.4870 Å, ε=0.0157.
    /// Off-diagonal ICO is Lorentz–Berthelot of the two self terms.
    /// Goldens are hand-derived from the A/B closed form; no AmberTools.
    const GAFF_MINI: &str = "\
%VERSION VERSION_STAMP = V0001.000
%FLAG TITLE
%FORMAT(20a4)
GAFF_MINI
%FLAG POINTERS
%FORMAT(10I8)
       4       2       1       2       1       1       0       1       0       0
       0       1       2       1       1       2       2       2       2       0
       0       0       0       0       0       0       0       0       4       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
C1  C2  C3  H1
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.20100000E+01  1.20100000E+01  1.20100000E+01  1.00800000E+00
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
c3  c3  c3  hc
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       1       1       2
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1       2       2       3
%FLAG BOND_FORCE_CONSTANT
%FORMAT(5E16.8)
  3.00000000E+02  3.40000000E+02
%FLAG BOND_EQUIL_VALUE
%FORMAT(5E16.8)
  1.53500000E+00  1.09000000E+00
%FLAG ANGLE_FORCE_CONSTANT
%FORMAT(5E16.8)
  5.00000000E+01  4.00000000E+01
%FLAG ANGLE_EQUIL_VALUE
%FORMAT(5E16.8)
  1.91113553E+00  1.91113553E+00
%FLAG DIHEDRAL_FORCE_CONSTANT
%FORMAT(5E16.8)
  1.50000000E-01  2.00000000E-01
%FLAG DIHEDRAL_PERIODICITY
%FORMAT(5E16.8)
  3.00000000E+00  2.00000000E+00
%FLAG DIHEDRAL_PHASE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00
%FLAG SCEE_SCALE_FACTOR
%FORMAT(5E16.8)
  1.20000000E+00  1.20000000E+00
%FLAG SCNB_SCALE_FACTOR
%FORMAT(5E16.8)
  2.00000000E+00  2.00000000E+00
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  1.04308023E+06  9.717081166135172E+04  7.516077034091E+03
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  6.75612248E+02  1.2691914994192737E+02  2.17257827878E+01
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
       6       9       2
%FLAG BONDS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       1       3       6       1
%FLAG ANGLES_INC_HYDROGEN
%FORMAT(10I8)
       3       6       9       2
%FLAG ANGLES_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       6       1
%FLAG DIHEDRALS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG DIHEDRALS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       6       9       1
";

    fn read_ff(text: &str) -> ForceField {
        AmberPrmtopForcefieldReader::new()
            .read(text.as_bytes())
            .unwrap_or_else(|e| panic!("read: {e}"))
    }

    fn read_err(text: &str) -> String {
        AmberPrmtopForcefieldReader::new()
            .read(text.as_bytes())
            .expect_err("expected Err from read")
            .to_string()
    }

    fn without_flag(text: &str, flag: &str) -> String {
        let mut out = String::new();
        let mut skipping = false;
        for line in text.lines() {
            if line.starts_with("%FLAG") {
                skipping = line.split_whitespace().nth(1) == Some(flag);
            }
            if !skipping {
                out.push_str(line);
                out.push('\n');
            }
        }
        out
    }

    fn rel_close(got: f64, expected: f64, tol: f64) -> bool {
        (got - expected).abs() <= tol * expected.abs()
    }

    /// A pointer past its parameter section, or a type index past the
    /// Lennard-Jones types, is refused by name — never a zero bond or a
    /// σ = 1, ε = 0 atom.
    #[test]
    fn a_pointer_outside_its_section_is_refused() {
        let bond = GAFF_MINI.replace(
            "%FORMAT(10I8)\n       6       9       2\n",
            "%FORMAT(10I8)\n       6       9       3\n",
        );
        let err = read_err(&bond);
        assert!(err.contains("BOND_FORCE_CONSTANT has no entry 3"), "{err}");

        let lj = GAFF_MINI.replace(
            "       1       1       1       2\n%FLAG NONBONDED_PARM_INDEX",
            "       1       1       1       5\n%FLAG NONBONDED_PARM_INDEX",
        );
        assert_ne!(lj, GAFF_MINI);
        let err = read_err(&lj);
        assert!(err.contains("ATOM_TYPE_INDEX"), "{err}");
    }

    #[test]
    fn pair_styles_are_registered_lj_cut_and_coul_cut() {
        let ff = read_ff(GAFF_MINI);
        assert!(
            ff.get_style("pair", "lj/cut").is_some(),
            "lj/cut pair style"
        );
        let coul = ff
            .get_style("pair", "coul/cut")
            .expect("coul/cut pair style");
        assert!(
            ff.get_style("pair", "lj/cut/coul/long").is_none(),
            "combined lj/cut/coul/long must not be registered"
        );

        let coulomb = coul
            .params()
            .get("coulomb")
            .and_then(|v| v.as_num())
            .expect("coulomb");
        let dielectric = coul
            .params()
            .get("dielectric")
            .and_then(|v| v.as_num())
            .expect("dielectric");
        assert!(
            (coulomb - 332.052_217_29).abs() < 1e-10,
            "coulomb={coulomb}"
        );
        assert!((dielectric - 1.0).abs() < 1e-12, "dielectric={dielectric}");

        for style in ff.get_styles("pair") {
            // A prmtop carries no cutoff (it lives in the mdin); the caller
            // declares one.
            assert!(
                style
                    .params()
                    .get("cutoff")
                    .and_then(|v| v.as_num())
                    .is_none(),
                "{} carries an invented cutoff",
                style.name()
            );
            assert!(
                style
                    .params()
                    .get("cutoff_lj")
                    .and_then(|v| v.as_num())
                    .is_none(),
                "{} still has cutoff_lj",
                style.name()
            );
            assert!(
                style
                    .params()
                    .get("cutoff_coul")
                    .and_then(|v| v.as_num())
                    .is_none(),
                "{} still has cutoff_coul",
                style.name()
            );
        }
    }

    #[test]
    fn special_bonds_are_reciprocal_divisors() {
        let ff = read_ff(GAFF_MINI);
        let sb = ff.special_bonds().unwrap();
        assert!((sb.lj[2] - 0.5).abs() < 1e-12, "lj_14={}", sb.lj[2]);
        assert!(
            (sb.coul[2] - 1.0 / 1.2).abs() < 1e-12,
            "coul_14={}",
            sb.coul[2]
        );
    }

    #[test]
    fn one_four_divisors_default_when_absent() {
        let text = without_flag(
            &without_flag(GAFF_MINI, "SCEE_SCALE_FACTOR"),
            "SCNB_SCALE_FACTOR",
        );
        let ff = read_ff(&text);
        let sb = ff.special_bonds().unwrap();
        assert!((sb.lj[2] - 0.5).abs() < 1e-12, "lj_14={}", sb.lj[2]);
        assert!(
            (sb.coul[2] - 1.0 / 1.2).abs() < 1e-12,
            "coul_14={}",
            sb.coul[2]
        );
    }

    /// Two 1-4 rows on one pair with divisors 1.2 and 1.0: a non-uniform file
    /// is read. The field takes the divisor most 1-4
    /// rows carry — a tie here, so the first — and the frame reader gives the
    /// pair its own weights (`1/1.2 + 1/1.0`).
    #[test]
    fn a_non_uniform_scee_takes_the_divisor_most_rows_carry() {
        let text = GAFF_MINI
            .replace(
                "  1.20000000E+00  1.20000000E+00",
                "  1.20000000E+00  1.00000000E+00",
            )
            .replace(
                "       0       3       6       9       1",
                "       0       3       6       9       1       0       3       6       9       2",
            );
        let sb = *read_ff(&text).special_bonds().unwrap();
        assert!((sb.coul[2] - 1.0 / 1.2).abs() < 1e-15, "{sb:?}");
        assert!((sb.lj[2] - 0.5).abs() < 1e-15, "{sb:?}");
    }

    #[test]
    fn one_four_divisors_ignore_suppressed_torsions() {
        let text = GAFF_MINI
            .replace(
                "  1.20000000E+00  1.20000000E+00",
                "  1.20000000E+00  1.00000000E+00",
            )
            .replace(
                "       0       3       6       9       1",
                "       0       3       6       9       1       0       3      -6       9       2",
            );
        let ff = read_ff(&text);
        let sb = ff.special_bonds().unwrap();
        assert!((sb.lj[2] - 0.5).abs() < 1e-12, "lj_14={}", sb.lj[2]);
        assert!(
            (sb.coul[2] - 1.0 / 1.2).abs() < 1e-12,
            "coul_14={}",
            sb.coul[2]
        );
    }

    #[test]
    fn a_non_positive_one_four_divisor_is_refused() {
        let text = GAFF_MINI
            .replace(
                "  1.20000000E+00  1.20000000E+00",
                "  1.20000000E+00  0.00000000E+00",
            )
            .replace(
                "       0       3       6       9       1",
                "       0       3       6       9       1       0       3       6       9       2",
            );
        let err = read_err(&text);
        assert!(
            err.contains("SCEE_SCALE_FACTOR"),
            "error should name the flag: {err}"
        );
        assert!(err.contains("2"), "error should name used type id 2: {err}");
    }

    #[test]
    fn rejects_12_6_4() {
        let text = format!(
            "{GAFF_MINI}%FLAG LENNARD_JONES_CCOEF\n%FORMAT(5E16.8)\n  \
             1.00000000E+00  0.00000000E+00  0.00000000E+00\n"
        );
        let err = read_err(&text);
        assert!(err.contains("12-6-4"), "error should name 12-6-4: {err}");
    }

    #[test]
    fn an_improper_keeps_its_central_atom_third() {
        // i=C1 (c3), j=H1 (hc), k=C2 (c3, the centre), l=C3 (c3): j > k by
        // name, which reverses a proper but must not reverse an improper.
        let text = GAFF_MINI.replace(
            "       0       3       6       9       1",
            "       0       9       3      -6       1",
        );
        let ff = read_ff(&text);
        let style = ff
            .get_style("improper", "periodic")
            .expect("improper style");
        let rows = style.type_rows();
        let names: Vec<&str> = rows.iter().map(|(n, _, _)| n.as_ref()).collect();
        assert_eq!(names, vec!["c3-hc-c3-c3"]);
    }

    /// A negative-PN chain on an improper (types 1 → 2) is one `improper
    /// periodic` row per term, named `<quartet>@<n>`.
    #[test]
    fn a_multi_term_improper_is_one_type_per_term() {
        let text = GAFF_MINI
            .replace(
                "       0       3       6       9       1",
                "       0       3      -6      -9       1",
            )
            .replace(
                "  3.00000000E+00  2.00000000E+00",
                " -2.00000000E+00  3.00000000E+00",
            );
        let ff = read_ff(&text);
        let style = ff
            .get_style("improper", "periodic")
            .expect("improper style");
        let mut rows: Vec<(String, Vec<f64>)> = style
            .type_rows()
            .into_iter()
            .map(|(n, _, p)| {
                let v = ["k", "periodicity", "phase"]
                    .map(|k| p.get(k).and_then(|v| v.as_num()).unwrap());
                (n.to_owned(), v.to_vec())
            })
            .collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            rows,
            vec![
                ("c3-c3-c3-hc@2".to_owned(), vec![0.15, 2.0, 0.0]),
                ("c3-c3-c3-hc@3".to_owned(), vec![0.2, 3.0, 0.0]),
            ]
        );
    }

    /// A negative-PN chain on a row that carries a 1-4 pair is refused: sander
    /// prices that pair once per chained term, at a Coulomb factor unlike its
    /// other 1-4 pairs.
    #[test]
    fn a_chain_on_a_1_4_row_is_refused() {
        let text = GAFF_MINI.replace(
            "  3.00000000E+00  2.00000000E+00",
            " -3.00000000E+00  2.00000000E+00",
        );
        let err = read_err(&text);
        assert!(err.contains("multi-term chain"), "{err}");
    }

    #[test]
    fn rejects_negative_ico() {
        let text = GAFF_MINI.replace(
            "       1       2       2       3",
            "       1      -1       2       3",
        );
        let err = read_err(&text);
        assert!(
            err.contains("10-12 interactions are not supported"),
            "{err}"
        );
    }

    #[test]
    fn rejects_nonzero_hbond() {
        let text = format!("{GAFF_MINI}%FLAG HBOND_ACOEF\n%FORMAT(5E16.8)\n  1.00000000E+00\n");
        let err = read_err(&text);
        assert!(
            err.contains("10-12 interactions are not supported"),
            "{err}"
        );
    }

    #[test]
    fn decode_lj_types_self_terms_match_closed_form() {
        let ff = read_ff(GAFF_MINI);
        let lj = ff.get_style("pair", "lj/cut").expect("lj/cut pair style");
        let pt = lj.get_pairtype("c3", None).expect("c3 self pair type");
        let eps = pt
            .params
            .get("epsilon")
            .and_then(|v| v.as_num())
            .expect("epsilon");
        let sigma = pt
            .params
            .get("sigma")
            .and_then(|v| v.as_num())
            .expect("sigma");
        assert!(rel_close(eps, 0.109400, 1e-6), "epsilon={eps} vs 0.109400");
        assert!(
            rel_close(sigma, 3.3996695, 1e-6),
            "sigma={sigma} vs 3.3996695"
        );
    }

    /// An off-diagonal ICO entry that is not Lorentz–Berthelot (CHARMM NBFIX,
    /// ParmEd `changeLJPair`) is an explicit `lj/cut` cross row with the
    /// entry's own σ/ε.
    #[test]
    fn decode_lj_types_keeps_nbfix_cross_terms_as_cross_rows() {
        let nbfix = GAFF_MINI
            .replace(
                "  1.04308023E+06  9.717081166135172E+04  7.516077034091E+03",
                "  1.04308023E+06  9.814251977796524E+04  7.516077034091E+03",
            )
            .replace(
                "  6.75612248E+02  1.2691914994192737E+02  2.17257827878E+01",
                "  6.75612248E+02  1.281883414413926E+02  2.17257827878E+01",
            );
        let ff = read_ff(&nbfix);
        let lj = ff.get_style("pair", "lj/cut").expect("lj/cut pair style");
        let cross = lj.get_pairtype("c3", Some("hc")).expect("c3-hc cross row");
        assert_eq!(
            (cross.endpoints[0].as_str(), cross.endpoints[1].as_str()),
            ("c3", "hc")
        );
        // σ = (A/B)^{1/6}, ε = B²/(4A) of the off-diagonal entry.
        let (a, b) = (9.814251977796524e4_f64, 1.281883414413926e2_f64);
        let (sigma, eps) = ((a / b).powf(1.0 / 6.0), b * b / (4.0 * a));
        assert!(rel_close(
            cross.params.get("sigma").and_then(|v| v.as_num()).unwrap(),
            sigma,
            1e-12
        ));
        assert!(rel_close(
            cross
                .params
                .get("epsilon")
                .and_then(|v| v.as_num())
                .unwrap(),
            eps,
            1e-12
        ));
        // The self rows are untouched.
        let c3 = lj.get_pairtype("c3", None).unwrap();
        assert!(rel_close(
            c3.params.get("epsilon").and_then(|v| v.as_num()).unwrap(),
            0.1094,
            1e-6
        ));
    }

    /// The `lj/cut` style states the rule its missing off-diagonal rows are
    /// mixed by, Lorentz–Berthelot, rather than leaning on a kernel default.
    #[test]
    fn the_lj_style_states_arithmetic_mixing() {
        let ff = read_ff(GAFF_MINI);
        let lj = ff.get_style("pair", "lj/cut").expect("lj/cut pair style");
        assert_eq!(
            lj.params().get("mixing").and_then(|v| v.as_text()),
            Some("arithmetic")
        );
    }

    /// A Lorentz–Berthelot off-diagonal entry is what mixing gives: no row.
    #[test]
    fn decode_lj_types_emits_no_row_for_a_mixed_entry() {
        let ff = read_ff(GAFF_MINI);
        let lj = ff.get_style("pair", "lj/cut").expect("lj/cut pair style");
        assert!(lj.get_pairtype("c3", Some("hc")).is_none());
        for pt in ff.get_types("pair") {
            assert_eq!(
                pt.endpoints[0], pt.endpoints[1],
                "unexpected cross pair type {}-{}",
                pt.endpoints[0], pt.endpoints[1]
            );
        }
    }

    /// `text` with the single occurrence of `from` replaced by `to`.
    fn replaced_once(text: &str, from: &str, to: &str) -> String {
        assert_eq!(
            text.matches(from).count(),
            1,
            "fixture edit must be unique: {from:?}"
        );
        text.replacen(from, to, 1)
    }

    /// [`GAFF_MINI`] with a third bond-parameter row equal to row 1
    /// (`RK = 300`, `r₀ = 1.535`), and the second c3–c3 bond pointing at it.
    /// Two table rows, one parameter set, one type name.
    fn gaff_mini_with_equal_bond_rows_under_one_name() -> String {
        let text = replaced_once(
            GAFF_MINI,
            "  3.00000000E+02  3.40000000E+02\n",
            "  3.00000000E+02  3.40000000E+02  3.00000000E+02\n",
        );
        let text = replaced_once(
            &text,
            "  1.53500000E+00  1.09000000E+00\n",
            "  1.53500000E+00  1.09000000E+00  1.53500000E+00\n",
        );
        replaced_once(
            &text,
            "       0       3       1       3       6       1\n",
            "       0       3       1       3       6       3\n",
        )
    }

    #[test]
    fn two_bond_rows_with_equal_k_and_r0_under_one_name_define_one_type() {
        let ff = read_ff(&gaff_mini_with_equal_bond_rows_under_one_name());
        let c3c3: Vec<_> = ff
            .get_types("bond")
            .into_iter()
            .filter(|t| t.name == "c3-c3")
            .collect();
        assert_eq!(c3c3.len(), 1);
        // k = RK: AMBER's K is LAMMPS's.
        assert_eq!(
            c3c3[0].params.get("k").and_then(|v| v.as_num()),
            Some(300.0)
        );
        assert_eq!(
            c3c3[0].params.get("r0").and_then(|v| v.as_num()),
            Some(1.535)
        );
    }

    /// The second c3–c3 bond points at row 2 (`RK = 340`, `r₀ = 1.09`): two
    /// parameter sets under one type name is a conflict, not a silent
    /// first-wins drop.
    #[test]
    fn two_bond_rows_with_unequal_params_under_one_name_are_an_error() {
        let text = replaced_once(
            GAFF_MINI,
            "       0       3       1       3       6       1\n",
            "       0       3       1       3       6       2\n",
        );
        let err = read_err(&text);
        assert!(err.contains("c3-c3"), "error should name the type: {err}");
    }

    /// No `%FLAG ATOM_TYPE_INDEX` at all is a malformed file, reported as a
    /// missing section, not as a per-atom gap.
    #[test]
    fn absent_atom_type_index_section_is_reported_missing() {
        let err = read_err(&without_flag(GAFF_MINI, "ATOM_TYPE_INDEX"));
        assert!(
            err.contains("%FLAG ATOM_TYPE_INDEX"),
            "error should name the section: {err}"
        );
        assert!(err.contains("missing"), "error should say missing: {err}");
    }

    /// A present `ATOM_TYPE_INDEX` with fewer entries than atoms names the
    /// first atom lacking an entry, and is a different error from the absent
    /// section.
    #[test]
    fn short_atom_type_index_names_the_atom_without_entry() {
        let text = replaced_once(
            GAFF_MINI,
            "%FLAG ATOM_TYPE_INDEX\n%FORMAT(10I8)\n       1       1       1       2\n",
            "%FLAG ATOM_TYPE_INDEX\n%FORMAT(10I8)\n       1       1       1\n",
        );
        let err = read_err(&text);
        assert!(err.contains("atom 4"), "error should name atom 4: {err}");

        let missing = read_err(&without_flag(GAFF_MINI, "ATOM_TYPE_INDEX"));
        assert_ne!(
            err, missing,
            "short section and absent section must be distinct errors"
        );
    }

    /// The bond table row number is file layout, not a parameter.
    #[test]
    fn bond_types_carry_no_row_id() {
        let ff = read_ff(GAFF_MINI);
        let bonds = ff.get_types("bond");
        assert!(!bonds.is_empty());
        for bt in bonds {
            assert_eq!(
                bt.params.get("id").and_then(|v| v.as_num()),
                None,
                "{} carries an id",
                bt.name
            );
        }
    }

    /// The angle table row number is file layout, not a parameter.
    #[test]
    fn angle_types_carry_no_row_id() {
        let ff = read_ff(GAFF_MINI);
        let angles = ff.get_types("angle");
        assert!(!angles.is_empty());
        for at in angles {
            assert_eq!(
                at.params.get("id").and_then(|v| v.as_num()),
                None,
                "{} carries an id",
                at.name
            );
        }
    }

    /// A five-atom chamber (CHARMM) prmtop, hand-written: every number in it
    /// is hand-chosen, so each test below is a hand value.
    const CHAMBER_MINI: &str = include_str!("testdata/chamber_mini.parm7");

    fn type_params(ff: &ForceField, category: &str, style: &str) -> Vec<(String, Params)> {
        let mut rows: Vec<(String, Params)> = ff
            .get_style(category, style)
            .unwrap_or_else(|| panic!("no {category} {style}"))
            .type_rows()
            .into_iter()
            .map(|(n, _, p)| (n.to_owned(), p.clone()))
            .collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        rows
    }

    /// Every angle of a chamber file is `angle charmm`: its Urey-Bradley
    /// term where the file has one (K_ub 5.4, r_ub 2.4 on atoms 1-3), 0 and 0
    /// where it has none.
    #[test]
    fn chamber_angles_are_angle_charmm_with_their_urey_bradley() {
        let ff = read_ff(CHAMBER_MINI);
        assert_eq!(ff.name, "CHARMM");
        assert!(ff.get_style("angle", "harmonic").is_none());
        let rows = type_params(&ff, "angle", "charmm");
        let names: Vec<&str> = rows.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["C1-N1-C2", "C2-C3-N2", "C3-C2-N1"]);
        let p = &rows[0].1;
        assert_eq!(p.get("k").and_then(|v| v.as_num()), Some(50.0));
        assert!((p.get("theta0").and_then(|v| v.as_num()).unwrap() - 109.5).abs() < 1e-12);
        assert_eq!(
            (
                p.get("k_ub").and_then(|v| v.as_num()),
                p.get("r_ub").and_then(|v| v.as_num())
            ),
            (Some(5.4), Some(2.4))
        );
        for (name, p) in &rows[1..] {
            assert_eq!(
                (
                    p.get("k_ub").and_then(|v| v.as_num()),
                    p.get("r_ub").and_then(|v| v.as_num())
                ),
                (Some(0.0), Some(0.0)),
                "{name}"
            );
        }
    }

    /// A CHARMM improper is `improper harmonic` in the file's order (centre
    /// first), `k = K_psi` (CHARMM's un-halved constant is LAMMPS's).
    #[test]
    fn chamber_impropers_are_improper_harmonic_in_file_order() {
        let ff = read_ff(CHAMBER_MINI);
        let rows = type_params(&ff, "improper", "harmonic");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, "N1-C1-C2-C3");
        assert_eq!(
            (
                rows[0].1.get("k").and_then(|v| v.as_num()),
                rows[0].1.get("chi0").and_then(|v| v.as_num())
            ),
            (Some(20.0), Some(0.0))
        );
        let ends = ff.get_style("improper", "harmonic").unwrap().type_rows()[0]
            .1
            .clone();
        assert_eq!(ends, vec!["N1", "C1", "C2", "C3"]);
    }

    /// A CHARMM improper whose psi0 is not 0° or 180° is refused: LAMMPS prices
    /// |psi|, CHARMM the signed psi.
    #[test]
    fn a_chamber_improper_off_the_axis_is_refused() {
        let text = replaced_once(
            CHAMBER_MINI,
            "%FLAG CHARMM_IMPROPER_PHASE\n%FORMAT(5E16.8)\n  0.00000000E+00",
            "%FLAG CHARMM_IMPROPER_PHASE\n%FORMAT(5E16.8)\n  5.00000000E-01",
        );
        let err = read_err(&text);
        assert!(err.contains("psi0") && err.contains("N1-C1-C2-C3"), "{err}");
    }

    /// The CMAP map is a `cmap charmm` type named by its five atom types, its
    /// grid the file's values in the file's (φ-major) order.
    #[test]
    fn chamber_cmap_is_a_cmap_charmm_type() {
        let ff = read_ff(CHAMBER_MINI);
        let style = ff.get_style("cmap", "charmm").expect("cmap charmm");
        let rows = style.type_rows();
        assert_eq!(rows.len(), 1);
        let (name, ends, params) = &rows[0];
        assert_eq!(*name, "C1-N1-C2-C3-N2");
        assert_eq!(ends, &vec!["C1", "N1", "C2", "C3", "N2"]);
        let grid = params.get("grid").and_then(|v| v.as_array()).unwrap();
        assert_eq!(grid.shape(), &[2, 2]);
        assert_eq!(
            grid.iter().copied().collect::<Vec<_>>(),
            vec![0.1, 0.2, 0.3, 0.4]
        );
    }

    /// Chamber Lennard-Jones is `lj/charmm`: the regular table's σ/ε, and —
    /// the 1-4 table differing — `epsilon14`/`sigma14` with
    /// `one_four = "epsilon14"`. A cross row stands where either table's
    /// off-diagonal entry is not Lorentz–Berthelot (here the 1-4 one:
    /// ε₁₄ 0.05 against the mix √(0.01·0.2)). Coulomb is `coul/charmm` at
    /// CHARMM's 332.0716; the 1-4 weights are 1/SCEE = 1/SCNB = 1.
    #[test]
    fn chamber_lennard_jones_is_lj_charmm_with_its_1_4_table() {
        let ff = read_ff(CHAMBER_MINI);
        let lj = ff.get_style("pair", "lj/charmm").expect("lj/charmm");
        assert_eq!(
            lj.params().get("one_four").and_then(|v| v.as_text()),
            Some("epsilon14")
        );
        assert_eq!(
            lj.params().get("mixing").and_then(|v| v.as_text()),
            Some("arithmetic")
        );
        let s = |rmin: f64| rmin * 2f64.powf(-1.0 / 6.0);
        let c1 = lj.get_pairtype("C1", None).unwrap();
        for (key, want) in [
            ("epsilon", 0.1),
            ("sigma", s(2.0)),
            ("epsilon14", 0.01),
            ("sigma14", s(1.8)),
        ] {
            assert!(
                rel_close(
                    c1.params.get(key).and_then(|v| v.as_num()).unwrap(),
                    want,
                    1e-12
                ),
                "C1 {key}"
            );
        }
        let cross = lj.get_pairtype("C1", Some("N1")).expect("C1-N1 cross row");
        for (key, want) in [
            ("epsilon", 0.02f64.sqrt()),
            ("sigma", s(2.5)),
            ("epsilon14", 0.05),
            ("sigma14", s(2.4)),
        ] {
            assert!(
                rel_close(
                    cross.params.get(key).and_then(|v| v.as_num()).unwrap(),
                    want,
                    1e-12
                ),
                "C1-N1 {key}"
            );
        }
        // N1 and N2 share class 2; C2, C3 share class 1 with C1.
        assert!(lj.get_pairtype("C2", Some("N2")).is_some());
        let coul = ff.get_style("pair", "coul/charmm").expect("coul/charmm");
        assert_eq!(
            coul.params().get("coulomb").and_then(|v| v.as_num()),
            Some(332.0716)
        );
        assert!(ff.get_style("pair", "lj/cut").is_none());
        let sb = ff.special_bonds().unwrap();
        assert_eq!((sb.lj, sb.coul), ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]));
    }

    /// A chamber file whose 1-4 table is the regular one has no 1-4 columns
    /// and no `one_four`.
    #[test]
    fn an_equal_1_4_table_adds_nothing() {
        let a = "  4.0960000000000000E+02  8.4293697021788080E+03  1.0628820000000001E+05";
        let b = "  1.2800000000000001E+01  6.9053396600248790E+01  2.9160000000000002E+02";
        let text = replaced_once(
            &replaced_once(
                CHAMBER_MINI,
                "  1.1568313814261765E+01  1.8260173718028282E+03  1.0628820000000001E+05",
                a,
            ),
            "  6.8024448000000000E-01  1.9110297599999996E+01  2.9160000000000002E+02",
            b,
        );
        let ff = read_ff(&text);
        let lj = ff.get_style("pair", "lj/charmm").unwrap();
        assert_eq!(lj.params().get("one_four").and_then(|v| v.as_text()), None);
        for pt in ff.get_types("pair") {
            assert!(
                pt.params
                    .get("epsilon14")
                    .and_then(|v| v.as_num())
                    .is_none(),
                "{}",
                pt.name
            );
        }
        assert!(
            lj.get_pairtype("C1", Some("N1")).is_none(),
            "the mix needs no row"
        );
    }

    /// A Urey-Bradley term on no angle has no `angle charmm` form.
    #[test]
    fn a_urey_bradley_term_on_no_angle_is_refused() {
        let text = replaced_once(
            CHAMBER_MINI,
            "%FLAG CHARMM_UREY_BRADLEY\n%FORMAT(10I8)\n       1       3       1",
            "%FLAG CHARMM_UREY_BRADLEY\n%FORMAT(10I8)\n       1       4       1",
        );
        let err = read_err(&text);
        assert!(err.contains("on no angle"), "{err}");
    }

    /// tleap writes π as 3.14159400; sander prices it as π, and so does the
    /// reader: the phase is 180° exactly, not 180.0000153°.
    #[test]
    fn a_phase_next_to_pi_is_pi() {
        let text = GAFF_MINI.replace(
            "%FLAG DIHEDRAL_PHASE\n%FORMAT(5E16.8)\n  0.00000000E+00  0.00000000E+00",
            "%FLAG DIHEDRAL_PHASE\n%FORMAT(5E16.8)\n  3.14159400E+00  0.00000000E+00",
        );
        let ff = read_ff(&text);
        let rows = type_params(&ff, "dihedral", "periodic");
        assert_eq!(
            rows[0].1.get("phase1").and_then(|v| v.as_num()),
            Some(180.0)
        );
    }

    /// Two atoms of one AMBER_ATOM_TYPE in two LJ classes (a chamber file cuts
    /// CHARMM's types to four characters) are two types, `<name>~<class>`.
    #[test]
    fn one_type_name_on_two_classes_is_two_types() {
        let text = replaced_once(CHAMBER_MINI, "C1  N1  C2  C3  N2", "C1  N1  C1  C3  C1");
        let ff = read_ff(&text);
        let atoms: Vec<String> = ff
            .get_style("atom", "full")
            .unwrap()
            .type_rows()
            .into_iter()
            .map(|(n, _, _)| n.to_owned())
            .collect();
        let mut atoms = atoms;
        atoms.sort();
        assert_eq!(atoms, vec!["C1~1", "C1~2", "C3", "N1"]);
    }
}
