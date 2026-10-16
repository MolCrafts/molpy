//! AMBER frcmod force-field writer.

use crate::core::constants::AMBER_COULOMB;
use crate::core::constants::VACUUM_DIELECTRIC;
use crate::core::constants::{AMBER_SCEE, AMBER_SCNB};
use crate::ff::forcefield::{ForceField, Style};
use crate::ff::ir::CombiningRule;
use crate::ff::ir::Engine;
use crate::ff::ir::Params;
use crate::ff::style_registry::refuse_style;
use crate::ff::typifier::Provenance;
use crate::io::writer::{ForceFieldWriteError, ForceFieldWriter};

/// Section headers, in file order.
const SECTIONS: [&str; 6] = ["MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON"];

/// Writer for AMBER frcmod parameter files.
///
/// `AmberFrcmodWriter::new()`, then [`ForceFieldWriter::write`] /
/// [`ForceFieldWriter::write_str`].
///
/// The inverse of the force-field half of
/// [`AmberPrmtopForcefieldReader`](crate::io::amber::AmberPrmtopForcefieldReader):
/// it writes a [`ForceField`] as the six parameter sections of an AMBER
/// frcmod file, so tleap can load a molrs force field with
/// `loadamberparams`. The force-field IR follows the LAMMPS standard, which for these terms is
/// AMBER's own — Å, kcal/mol, amu, un-halved `K`, degrees — so every number is
/// written as stored.
///
/// # Output
///
/// A title line, then `MASS`, `BOND`, `ANGLE`, `DIHE`, `IMPROPER` and
/// `NONBON`, each closed by a blank line and always present (an empty
/// section is a header and its blank line).
///
/// | molrs style | Section | Row (file units) |
/// |---|---|---|
/// | `atom/full` | `MASS` | `T  mass` |
/// | `bond/harmonic` | `BOND` | `T1-T2  RK = k  R0 = r0` |
/// | `angle/harmonic` | `ANGLE` | `T1-T2-T3  TK = k  THETA0 = theta0` |
/// | `dihedral/periodic` | `DIHE` | one row per term: `T1-T2-T3-T4  IDIVF = 1  PK = k_m  PHASE = phase_m  PN = ±n_m` |
/// | `improper/periodic` | `IMPROPER` | `T1-T2-T3-T4  PK = k  PHASE = phase  PN = n`, the atoms in the stored (AMBER) order |
/// | `pair/lj/cut` (self rows) | `NONBON` | `  T  R*/2 = σ·2^(1/6)/2  EPSILON = ε` |
///
/// `dihedral/periodic` takes `k{m}`/`periodicity{m}`/`phase{m}`, or a single
/// `k`/`periodicity`/`phase` triple; the prmtop reader writes the first, the
/// GAFF typifier the second. A multi-term torsion is written in AMBER's
/// convention: `PN` is negative on every term but the last. In the bonded
/// rows each atom type is padded to frcmod's two-character field, and a
/// wildcard endpoint (`X` or empty) is written `X `; `MASS` and `NONBON` rows
/// are whitespace-delimited and take longer types (the ion types `Li+`).
///
/// `pair/coul/cut` has no section: tleap takes the Coulomb constants from
/// AMBER itself, so only the constants the AMBER readers declare
/// (`coulomb = AMBER_COULOMB`, `dielectric = 1`, `delta = 0`, no types) are
/// accepted, and nothing is written.
///
/// # Estimated terms
///
/// A term the GAFF typifier estimated rather than matched carries the four
/// provenance keys of [`Provenance`] (`estimated`, `estimate_penalty`,
/// `estimate_method`, `estimate_analog`). They are metadata, not parameters:
/// the row is written from its parameters and the provenance becomes the
/// row's trailing comment, as parmchk2 writes it —
/// `same as c3-os, penalty score= 2.5` (or `estimated (empirical), …` when no
/// analog was copied). AMBER reads the fixed fields and ignores the rest of
/// the line.
///
/// # Refusals
///
/// A pair style's `cutoff` is a run setting (AMBER keeps it in the mdin, not in
/// any parameter file), so it is not force-field data this writer drops: it is
/// not written. What a frcmod cannot express is an `Err` naming it, never a
/// silent drop:
///
/// - any other style;
/// - a style parameter (e.g. an `lj/cut` `shift`), or `lj/cut` mixing other
///   than `arithmetic` (AMBER combines by Lorentz–Berthelot);
/// - a type parameter with no column, or a missing one;
/// - an explicit `lj/cut` cross row (NBFIX has no frcmod section);
/// - an atom type longer than two characters;
/// - a non-positive or non-integer periodicity;
/// - declared units other than `real`, or declared special-bond weights other
///   than AMBER's (1-2 / 1-3 excluded, 1-4 LJ 1/SCNB, Coulomb 1/SCEE), which
///   tleap supplies itself.
#[derive(Debug, Clone, Default)]
pub struct AmberFrcmodWriter;

impl AmberFrcmodWriter {
    pub fn new() -> Self {
        Self
    }

    /// Refuse a declared unit system or 1-2/1-3/1-4 weights tleap would not
    /// reproduce.
    fn check_declarations(ff: &ForceField) -> Result<(), String> {
        if let Some(units) = ff.declared_units()
            && units != "real"
        {
            return Err(format!(
                "force field units '{units}': frcmod is in real units (Å, kcal/mol)"
            ));
        }
        if let Some(sb) = ff.declared_special_bonds() {
            let amber_lj = [0.0, 0.0, 1.0 / AMBER_SCNB];
            let amber_coul = [0.0, 0.0, 1.0 / AMBER_SCEE];
            let close =
                |a: &[f64; 3], b: &[f64; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-12);
            if !close(&sb.lj, &amber_lj) || !close(&sb.coul, &amber_coul) {
                return Err(format!(
                    "special-bond weights (lj {:?}, coul {:?}) are not AMBER's (lj {amber_lj:?}, \
                     coul {amber_coul:?}); a frcmod cannot carry them",
                    sb.lj, sb.coul
                ));
            }
        }
        Ok(())
    }

    /// Refuse the style-level parameters of `style`, which no section holds.
    fn check_style_params(style: &Style) -> Result<(), String> {
        let what = format!("{}/{}", style.category(), style.name());
        let p = style.params();
        // A cutoff is a run setting, not parameter data (see the module doc).
        let params = || {
            p.iter()
                .filter(|(key, _)| style.category() != "pair" || *key != "cutoff")
        };
        match (style.category(), style.name()) {
            ("pair", "coul/cut") => {
                for (key, value) in params() {
                    let implied = match key {
                        "coulomb" => AMBER_COULOMB,
                        "dielectric" => VACUUM_DIELECTRIC,
                        "delta" => 0.0,
                        _ => {
                            return Err(format!("{what} style param '{key}' has no frcmod column"));
                        }
                    };
                    if value != implied {
                        return Err(format!(
                            "{what} {key} = {value}: tleap implies {key} = {implied}"
                        ));
                    }
                }
                if let Some((name, _, _)) = style.type_rows().first() {
                    return Err(format!("{what} type '{name}' has no frcmod section"));
                }
            }
            _ => {
                if let Some((key, _)) = params().next() {
                    return Err(format!("{what} style param '{key}' has no frcmod column"));
                }
            }
        }
        for (key, value) in p.iter_strings() {
            let arithmetic =
                || CombiningRule::parse(value).is_ok_and(|m| m == CombiningRule::Arithmetic);
            if !(what == "pair/lj/cut" && key == "mixing" && arithmetic()) {
                return Err(format!(
                    "{what} style param '{key}' = '{value}' has no frcmod column (AMBER \
                     mixing is arithmetic)"
                ));
            }
        }
        Ok(())
    }

    /// Refuse a numeric type parameter outside `allowed`. A key that prices
    /// nothing ([`Params::iter_parameters`]) is never refused: the
    /// provenance of an estimated term ([`Provenance::KEYS`]) is written as
    /// the row's comment, an annotation not at all.
    fn check_type_params(p: &Params, allowed: &[&str], what: &str) -> Result<(), String> {
        match p.iter_parameters().find(|(key, _)| !allowed.contains(key)) {
            Some((key, _)) => Err(format!("{what}: parameter '{key}' has no frcmod column")),
            None => Ok(()),
        }
    }

    /// The trailing comment of an estimated row, parmchk2's way (`same as
    /// <analog>, penalty score= <penalty>`); empty for a matched term.
    fn provenance_comment(p: &Params) -> String {
        match Provenance::from_params(p) {
            None => String::new(),
            Some(estimate) if estimate.analog.is_empty() => format!(
                "  estimated ({}), penalty score= {:.1}",
                estimate.method.as_str(),
                estimate.penalty
            ),
            Some(estimate) => format!(
                "  same as {}, penalty score= {:.1}",
                estimate.analog, estimate.penalty
            ),
        }
    }

    /// A parameter the row cannot be written without.
    fn need(p: &Params, key: &str, what: &str) -> Result<f64, String> {
        p.get(key).ok_or_else(|| format!("{what} has no {key}"))
    }

    /// A periodicity as frcmod's `PN`: a positive integer.
    fn periodicity(n: f64, what: &str) -> Result<f64, String> {
        if n.fract() != 0.0 || !n.is_finite() || n <= 0.0 {
            return Err(format!("{what}: periodicity {n} is not a positive integer"));
        }
        Ok(n)
    }

    /// The fixed-width type field: each type padded to two characters, a
    /// wildcard written `X `, joined with `-`.
    fn type_field(ends: &[&str], what: &str) -> Result<String, String> {
        let mut parts = Vec::with_capacity(ends.len());
        for end in ends {
            let end = if end.is_empty() { "X" } else { end };
            if end.chars().count() > 2 || end.contains(char::is_whitespace) {
                return Err(format!(
                    "{what}: atom type '{end}' does not fit frcmod's two-character type field"
                ));
            }
            parts.push(format!("{end:<2}"));
        }
        Ok(parts.join("-"))
    }

    /// The type field of a one-type row (`MASS`, `NONBON`). These rows are read
    /// whitespace-delimited, so a type may be longer than two characters (the
    /// AMBER ion types `Li+`, `Na+`, `Cl-`); only the bonded rows' fixed-width
    /// fields are limited to two.
    fn single_type_field(name: &str, what: &str) -> Result<String, String> {
        if name.is_empty() || name.contains(char::is_whitespace) {
            return Err(format!(
                "{what}: atom type '{name}' is not a frcmod type name"
            ));
        }
        Ok(format!("{name:<2}"))
    }

    /// The `(k, n, d)` cosine terms of a `dihedral/periodic` type, in
    /// the order the kernel reads them.
    fn dihedral_terms(p: &Params, what: &str) -> Result<Vec<(f64, f64, f64)>, String> {
        let mut terms = Vec::new();
        let mut allowed: Vec<String> = Vec::new();
        let mut m = 1;
        while let Some(k) = p.get(&format!("k{m}")) {
            let n = Self::need(p, &format!("periodicity{m}"), what)?;
            let d = p.get(&format!("phase{m}")).unwrap_or(0.0);
            terms.push((k, Self::periodicity(n, what)?, d));
            allowed.extend([
                format!("k{m}"),
                format!("periodicity{m}"),
                format!("phase{m}"),
            ]);
            m += 1;
        }
        if terms.is_empty() {
            let k = Self::need(p, "k", what)?;
            let n = Self::need(p, "periodicity", what)?;
            let d = p.get("phase").unwrap_or(0.0);
            terms.push((k, Self::periodicity(n, what)?, d));
            allowed.extend(["k", "periodicity", "phase"].map(String::from));
        }
        let allowed: Vec<&str> = allowed.iter().map(String::as_str).collect();
        Self::check_type_params(p, &allowed, what)?;
        Ok(terms)
    }

    /// The rows of `style`, with the index of the section they belong to.
    fn style_rows(style: &Style) -> Result<(usize, Vec<String>), ForceFieldWriteError> {
        let (category, name) = (style.category(), style.name());
        let section = match (category, name) {
            ("atom", "full") => 0,
            ("bond", "harmonic") => 1,
            ("angle", "harmonic") => 2,
            ("dihedral", "periodic") => 3,
            ("improper", "periodic") => 4,
            ("pair", "lj/cut") => 5,
            ("pair", "coul/cut") => return Ok((5, Vec::new())),
            _ => return Err(refuse_style(Engine::AmberFrcmod, category, name).into()),
        };
        let mut rows = Vec::new();
        for (type_name, ends, p) in style.type_rows() {
            let what = format!("{category}/{name} type '{type_name}'");
            let need = |key: &str| Self::need(p, key, &what);
            let note = Self::provenance_comment(p);
            match section {
                0 => {
                    // `id` is the prmtop LJ class index: a label, not a parameter.
                    Self::check_type_params(p, &["mass", "id"], &what)?;
                    let key = Self::single_type_field(type_name, &what)?;
                    rows.push(format!("{key}  {:.6}", need("mass")?));
                }
                1 => {
                    Self::check_type_params(p, &["k", "r0"], &what)?;
                    let key = Self::type_field(&ends, &what)?;
                    rows.push(format!(
                        "{key}  {:.6}  {:.6}{note}",
                        need("k")?,
                        need("r0")?
                    ));
                }
                2 => {
                    Self::check_type_params(p, &["k", "theta0"], &what)?;
                    let key = Self::type_field(&ends, &what)?;
                    rows.push(format!(
                        "{key}  {:.6}  {:.6}{note}",
                        need("k")?,
                        need("theta0")?
                    ));
                }
                3 => {
                    let key = Self::type_field(&ends, &what)?;
                    let terms = Self::dihedral_terms(p, &what)?;
                    let last = terms.len() - 1;
                    for (m, (k, n, d)) in terms.into_iter().enumerate() {
                        let pn = if m == last { n } else { -n };
                        rows.push(format!("{key}  1  {k:.6}  {d:.6}  {pn:.1}{note}"));
                    }
                }
                4 => {
                    Self::check_type_params(p, &["k", "periodicity", "phase"], &what)?;
                    let key = Self::type_field(&ends, &what)?;
                    let n = Self::periodicity(need("periodicity")?, &what)?;
                    let d = p.get("phase").unwrap_or(0.0);
                    // tleap reads an IMPROPER row's PK from fixed columns: it
                    // must start at 0-based column 15 or later, past the unused
                    // IDIVF slot that a DIHE row fills with `  1`. Two spaces
                    // put PK at column 13, and tleap then stores K = 1e5 without
                    // a warning; four put it at 15.
                    rows.push(format!("{key}    {:.6}  {d:.6}  {n:.1}{note}", need("k")?));
                }
                _ => {
                    if ends[0] != ends[1] {
                        return Err(format!(
                            "{what}: an explicit cross row ({} with {}) is an NBFIX, which a \
                             frcmod cannot express",
                            ends[0], ends[1]
                        )
                        .into());
                    }
                    Self::check_type_params(p, &["sigma", "epsilon"], &what)?;
                    let key = Self::single_type_field(ends[0], &what)?;
                    let r_min_half = need("sigma")? * 2f64.powf(1.0 / 6.0) / 2.0;
                    rows.push(format!("  {key}  {r_min_half:.6}  {:.6}", need("epsilon")?));
                }
            }
        }
        Ok((section, rows))
    }
}

impl ForceFieldWriter for AmberFrcmodWriter {
    fn write_str(&self, ff: &ForceField) -> Result<String, ForceFieldWriteError> {
        Self::check_declarations(ff)?;
        let mut sections: [Vec<String>; 6] = Default::default();
        for style in ff.styles() {
            let (section, rows) = Self::style_rows(style)?;
            Self::check_style_params(style)?;
            sections[section].extend(rows);
        }
        let mut out = format!("{} force field, written by molrs\n", ff.name);
        for (header, rows) in SECTIONS.iter().zip(&sections) {
            out.push_str(header);
            out.push('\n');
            for row in rows {
                out.push_str(row);
                out.push('\n');
            }
            out.push('\n');
        }
        Ok(out)
    }
}

/// Write `forcefield` to `path` as an AMBER frcmod file.
pub fn write_frcmod(path: &str, forcefield: &ForceField) -> Result<(), ForceFieldWriteError> {
    AmberFrcmodWriter::new().write(forcefield, path)
}

/// `forcefield` as the text of an AMBER frcmod file.
pub fn write_frcmod_str(forcefield: &ForceField) -> Result<String, ForceFieldWriteError> {
    AmberFrcmodWriter::new().write_str(forcefield)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::forcefield::ForceField;
    use crate::ff::ir::Params;

    /// The rows under `header`, up to the blank line that closes the section.
    fn section<'a>(text: &'a str, header: &str) -> Vec<&'a str> {
        text.lines()
            .skip_while(|l| l.trim() != header)
            .skip(1)
            .take_while(|l| !l.trim().is_empty())
            .collect()
    }

    /// Split a row into its fixed-width type field and the numbers after it.
    fn fields(row: &str, width: usize) -> (&str, Vec<f64>) {
        let (key, rest) = row.split_at(width);
        let values = rest
            .split_whitespace()
            .map(|t| t.parse().unwrap_or_else(|e| panic!("{t:?}: {e}")))
            .collect();
        (key, values)
    }

    fn write(ff: &ForceField) -> String {
        write_frcmod_str(ff).unwrap_or_else(|e| panic!("write: {e}"))
    }

    /// One `category/style` with one type over `ends`.
    fn one_type(category: &str, style: &str, ends: &[&str], params: Params) -> ForceField {
        let mut ff = ForceField::new("t");
        let name = ends.join("-");
        ff.def_style(category, style, Params::new())
            .unwrap()
            .def_type(&name, ends, params)
            .unwrap();
        ff
    }

    #[test]
    fn bond_row_writes_k_as_rk() {
        let ff = one_type(
            "bond",
            "harmonic",
            &["c3", "os"],
            Params::from_pairs(&[("k", 640.0), ("r0", 1.43)]),
        );
        let text = write(&ff);
        let rows = section(&text, "BOND");
        assert_eq!(rows.len(), 1, "{text}");
        let (key, v) = fields(rows[0], 5);
        assert_eq!(key, "c3-os");
        assert_eq!(v, vec![640.0, 1.43]);
    }

    /// An estimated term's provenance keys are metadata, not parameters: the
    /// row is written from `k` / `r0`, and the provenance is its comment.
    #[test]
    fn estimated_term_writes_its_provenance_as_the_row_comment() {
        let mut params = Params::from_pairs(&[("k", 640.0), ("r0", 1.43)]);
        Provenance::analogy(2.5, "c3-os").apply_to(&mut params);
        let ff = one_type("bond", "harmonic", &["c3", "oh"], params);
        let text = write(&ff);
        let rows = section(&text, "BOND");
        assert_eq!(rows.len(), 1, "{text}");
        let (values, comment) = rows[0].split_once("  same as").unwrap();
        let (key, v) = fields(values, 5);
        assert_eq!(key, "c3-oh");
        assert_eq!(v, vec![640.0, 1.43]);
        assert_eq!(comment, " c3-os, penalty score= 2.5");
    }

    /// An empirical estimate copied no row: the comment names the method.
    #[test]
    fn empirical_estimate_comment_names_the_method() {
        let mut params = Params::from_pairs(&[("k", 640.0), ("r0", 1.43)]);
        Provenance::empirical(12.0).apply_to(&mut params);
        let ff = one_type("bond", "harmonic", &["c3", "oh"], params);
        let text = write(&ff);
        let rows = section(&text, "BOND");
        assert!(
            rows[0].ends_with("  estimated (empirical), penalty score= 12.0"),
            "{}",
            rows[0]
        );
    }

    #[test]
    fn one_character_type_is_padded_to_two() {
        let ff = one_type(
            "bond",
            "harmonic",
            &["c", "o"],
            Params::from_pairs(&[("k", 2.0), ("r0", 1.2)]),
        );
        let text = write(&ff);
        let (key, _) = fields(section(&text, "BOND")[0], 5);
        assert_eq!(key, "c -o ");
    }

    #[test]
    fn angle_row_writes_k_and_degrees_as_stored() {
        let ff = one_type(
            "angle",
            "harmonic",
            &["c3", "os", "c3"],
            Params::from_pairs(&[("k", 100.0), ("theta0", 90.0)]),
        );
        let text = write(&ff);
        let rows = section(&text, "ANGLE");
        assert_eq!(rows.len(), 1, "{text}");
        let (key, v) = fields(rows[0], 8);
        assert_eq!(key, "c3-os-c3");
        assert_eq!(v, vec![100.0, 90.0]);
    }

    #[test]
    fn two_term_periodic_dihedral_negates_every_pn_but_the_last() {
        let ff = one_type(
            "dihedral",
            "periodic",
            &["c3", "os", "c3", "h1"],
            Params::from_pairs(&[
                ("k1", 0.38),
                ("periodicity1", 3.0),
                ("phase1", 0.0),
                ("k2", 0.1),
                ("periodicity2", 2.0),
                ("phase2", 180.0),
            ]),
        );
        let text = write(&ff);
        let rows = section(&text, "DIHE");
        assert_eq!(rows.len(), 2, "{text}");
        let (key0, v0) = fields(rows[0], 11);
        let (key1, v1) = fields(rows[1], 11);
        assert_eq!(key0, "c3-os-c3-h1");
        assert_eq!(key1, "c3-os-c3-h1");
        // IDIVF, PK, PHASE (deg), PN
        assert_eq!(v0, vec![1.0, 0.38, 0.0, -3.0]);
        assert_eq!(v1[..2], [1.0, 0.1]);
        assert!((v1[2] - 180.0).abs() < 1e-6, "{v1:?}");
        assert_eq!(v1[3], 2.0);
    }

    #[test]
    fn wildcard_endpoint_is_written_as_x() {
        let ff = one_type(
            "improper",
            "periodic",
            &["X", "o", "c", "o"],
            Params::from_pairs(&[("k", 1.1), ("periodicity", 2.0), ("phase", 180.0)]),
        );
        let text = write(&ff);
        let (key, v) = fields(section(&text, "IMPROPER")[0], 11);
        assert_eq!(key, "X -o -c -o ");
        assert_eq!(v[0], 1.1);
        assert!((v[1] - 180.0).abs() < 1e-6, "{v:?}");
        assert_eq!(v[2], 2.0);
    }

    #[test]
    fn improper_pk_starts_where_tleap_reads_it() {
        // tleap reads PK from fixed columns (0-based >= 15); at column 13 it
        // silently stores K = 1e5. Whitespace-split parsing can't see this,
        // so pin the column itself.
        let ff = one_type(
            "improper",
            "periodic",
            &["o", "os", "c", "os"],
            Params::from_pairs(&[("k", 1.1), ("periodicity", 2.0), ("phase", 180.0)]),
        );
        let text = write(&ff);
        let row = section(&text, "IMPROPER")[0];
        let pk = row[11..].find(|c: char| !c.is_whitespace()).unwrap() + 11;
        assert!(pk >= 15, "PK starts at column {pk}: {row:?}");
        assert!(row[pk..].starts_with("1.100000"), "{row:?}");
    }

    #[test]
    fn lj_sigma_becomes_half_r_min() {
        let sigma = 3.4;
        let mut ff = ForceField::new("t");
        ff.def_style("pair", "lj/cut", Params::new())
            .unwrap()
            .def_type(
                "c3",
                &["c3"],
                Params::from_pairs(&[("sigma", sigma), ("epsilon", 0.1094)]),
            )
            .unwrap();
        let text = write(&ff);
        let rows = section(&text, "NONBON");
        assert_eq!(rows.len(), 1, "{text}");
        let (key, v) = fields(rows[0], 4);
        assert_eq!(key, "  c3");
        assert!(
            (v[0] - sigma * 2f64.powf(1.0 / 6.0) / 2.0).abs() < 1e-6,
            "{v:?}"
        );
        assert_eq!(v[1], 0.1094);
    }

    #[test]
    fn unsupported_style_is_refused_by_name() {
        let ff = one_type(
            "bond",
            "morse",
            &["c3", "os"],
            Params::from_pairs(&[("d0", 1.0), ("alpha", 2.0), ("r0", 1.4)]),
        );
        let err = write_frcmod_str(&ff).unwrap_err();
        assert!(
            err.contains("AMBER frcmod has no form for bond `morse`"),
            "{err}"
        );
    }

    #[test]
    fn lj_cross_type_is_refused() {
        let ff = one_type(
            "pair",
            "lj/cut",
            &["c3", "os"],
            Params::from_pairs(&[("sigma", 3.0), ("epsilon", 0.1)]),
        );
        let err = write_frcmod_str(&ff).unwrap_err();
        assert!(err.contains("c3-os"), "{err}");
    }

    #[test]
    fn style_param_frcmod_would_drop_is_refused() {
        let mut ff = ForceField::new("t");
        ff.def_style("pair", "lj/cut", Params::from_pairs(&[("shift", 1.0)]))
            .unwrap();
        let err = write_frcmod_str(&ff).unwrap_err();
        assert!(err.contains("shift"), "{err}");
    }

    #[test]
    fn pair_cutoff_is_a_run_setting_and_not_written() {
        let mut ff = ForceField::new("t");
        ff.def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 9.0)]))
            .unwrap();
        let text = write_frcmod_str(&ff).unwrap();
        assert!(!text.contains('9'), "{text}");
    }

    #[test]
    fn an_ion_type_longer_than_two_characters_is_a_mass_and_nonbon_row() {
        let mut ff = ForceField::new("t");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("Li+", &[], Params::from_pairs(&[("mass", 6.94)]))
            .unwrap();
        ff.def_style("pair", "lj/cut", Params::new())
            .unwrap()
            .def_type(
                "Li+",
                &["Li+"],
                Params::from_pairs(&[("sigma", 2.0), ("epsilon", 0.01)]),
            )
            .unwrap();
        let text = write_frcmod_str(&ff).unwrap();
        assert!(text.contains("Li+  6.940000"), "{text}");
        assert!(text.contains("  Li+  "), "{text}");
    }

    #[test]
    fn type_longer_than_two_characters_is_refused() {
        let ff = one_type(
            "bond",
            "harmonic",
            &["c3x", "os"],
            Params::from_pairs(&[("k", 2.0), ("r0", 1.2)]),
        );
        let err = write_frcmod_str(&ff).unwrap_err();
        assert!(err.contains("c3x"), "{err}");
    }

    /// The text under each frcmod section keyword (`MASS`, `BOND`, …), keyed
    /// by the lower-cased keyword: enough to tell which section a row landed in.
    fn sections(text: &str) -> std::collections::BTreeMap<String, String> {
        const KEYWORDS: &[&str] = &["MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON"];
        let mut out = std::collections::BTreeMap::new();
        let mut current: Option<String> = None;
        let mut body: Vec<&str> = Vec::new();
        for line in text.lines() {
            let upper = line.trim().to_ascii_uppercase();
            if KEYWORDS.contains(&upper.as_str()) {
                if let Some(sec) = current.take() {
                    out.insert(sec, body.join("\n").trim().to_string());
                }
                current = Some(upper.to_ascii_lowercase());
                body.clear();
            } else if current.is_some() {
                body.push(line);
            }
        }
        if let Some(sec) = current {
            out.insert(sec, body.join("\n").trim().to_string());
        }
        out
    }

    #[test]
    fn each_row_lands_in_its_section() {
        let mut ff = ForceField::new("t");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("c3", &[], Params::from_pairs(&[("mass", 12.01)]))
            .unwrap();
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "c3-os",
                &["c3", "os"],
                Params::from_pairs(&[("k", 640.0), ("r0", 1.43)]),
            )
            .unwrap();
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "c3-os-c3",
                &["c3", "os", "c3"],
                Params::from_pairs(&[("k", 100.0), ("theta0", 109.0)]),
            )
            .unwrap();
        ff.def_style("dihedral", "periodic", Params::new())
            .unwrap()
            .def_type(
                "c3-os-c3-h1",
                &["c3", "os", "c3", "h1"],
                Params::from_pairs(&[("k1", 0.38), ("periodicity1", 3.0), ("phase1", 0.0)]),
            )
            .unwrap();
        ff.def_style("improper", "periodic", Params::new())
            .unwrap()
            .def_type(
                "c-o-c-o",
                &["c", "o", "c", "o"],
                Params::from_pairs(&[("k", 1.1), ("periodicity", 2.0), ("phase", 180.0)]),
            )
            .unwrap();
        ff.def_style("pair", "lj/cut", Params::new())
            .unwrap()
            .def_type(
                "c3",
                &["c3"],
                Params::from_pairs(&[("sigma", 3.4), ("epsilon", 0.1094)]),
            )
            .unwrap();

        let parsed = sections(&write(&ff));
        let body = |s: &str| parsed.get(s).cloned().unwrap_or_default();
        assert!(body("mass").starts_with("c3 "), "{parsed:?}");
        assert!(body("bond").starts_with("c3-os "), "{parsed:?}");
        assert!(body("angle").starts_with("c3-os-c3 "), "{parsed:?}");
        assert!(body("dihe").starts_with("c3-os-c3-h1 "), "{parsed:?}");
        assert!(body("improper").starts_with("c -o -c -o "), "{parsed:?}");
        assert!(body("nonbon").starts_with("c3 "), "{parsed:?}");
    }
}
