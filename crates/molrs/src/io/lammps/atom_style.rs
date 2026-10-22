//! `atom_style` → data-file Atoms column layout.
//!
//! Layouts follow the *Atoms* section table in
//! <https://docs.lammps.org/read_data.html>. Accelerated suffixes (`/kk`,
//! `/gpu`, …) are stripped before lookup.
//!
//! Each style is a sequence of [`DataField`]s; optional trailing image flags
//! (`nx ny nz`) are handled separately by the parser.

pub(crate) use molrs::core::type_labels::is_int_token;

/// One column in a data-file Atoms line (excluding optional image flags).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DataField {
    Id,
    Type,
    Mol,
    Charge,
    X,
    Y,
    Z,
    Bodyflag,
    Mass,
    Diameter,
    Density,
    Volume,
    /// Generic 0/1 finite-size flag (ellipsoid / line / triangle).
    ShapeFlag,
    Mux,
    Muy,
    Muz,
    /// Magnetic spin direction + magnitude (`spin` style).
    Spx,
    Spy,
    Spz,
    Sp,
    Rho,
    Esph,
    Cv,
    /// DPD internal temperature.
    Theta,
    Espin,
    Eradius,
    /// RHEO status integer.
    Status,
    /// RHEO thermal energy.
    Energy,
    TemplateIndex,
    TemplateAtom,
    /// eDPD temperature / heat capacity pair slots (parsed as float extras).
    EdpdTemp,
    EdpdCv,
    /// SMD: molecule already covered by Mol; remaining pre-xyz slots.
    SmdVolume,
    SmdMass,
    SmdKradius,
    SmdCradius,
    SmdX0,
    SmdY0,
    SmdZ0,
    /// Dielectric extras after dipole.
    Area,
    Ed,
    Em,
    Epsilon,
    Curvature,
}

/// Fixed column layout for one atom style.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AtomStyleLayout {
    /// Base fields (no image flags).
    pub fields: &'static [DataField],
    /// When true (hybrid), allow extra tokens between base fields and optional
    /// trailing image flags.
    pub flexible_tail: bool,
}

impl AtomStyleLayout {
    pub(crate) fn min_cols(self) -> usize {
        self.fields.len()
    }
}

// ---- Field sequences (read_data table) ------------------------------------

const ATOMIC: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const CHARGE: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::Charge,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

/// dpd / mdpd: id type <scalar> x y z  (scalar is theta / rho, not charge)
const SCALAR_XYZ: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::Theta, // dpd; mdpd reuses as rho via Rho below — see layout match
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const MDPD: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::Rho,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const MOLECULAR: &[DataField] = &[
    DataField::Id,
    DataField::Mol,
    DataField::Type,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const FULL: &[DataField] = &[
    DataField::Id,
    DataField::Mol,
    DataField::Type,
    DataField::Charge,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const BODY: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::Bodyflag,
    DataField::Mass,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const SPHERE: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::Diameter,
    DataField::Density,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const ELLIPSOID: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::ShapeFlag,
    DataField::Density,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const PERI: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::Volume,
    DataField::Density,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const EDPD: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::EdpdTemp,
    DataField::EdpdCv,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const LINE: &[DataField] = &[
    DataField::Id,
    DataField::Mol,
    DataField::Type,
    DataField::ShapeFlag,
    DataField::Density,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const TEMPLATE: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::Mol,
    DataField::TemplateIndex,
    DataField::TemplateAtom,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const SPH: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::Rho,
    DataField::Esph,
    DataField::Cv,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const ELECTRON: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::Charge,
    DataField::Espin,
    DataField::Eradius,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const DIPOLE: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::Charge,
    DataField::X,
    DataField::Y,
    DataField::Z,
    DataField::Mux,
    DataField::Muy,
    DataField::Muz,
];

const SPIN: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::X,
    DataField::Y,
    DataField::Z,
    DataField::Spx,
    DataField::Spy,
    DataField::Spz,
    DataField::Sp,
];

const RHEO: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::Status,
    DataField::Rho,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const RHEO_THERMAL: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::Status,
    DataField::Rho,
    DataField::Energy,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const SMD: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::Mol,
    DataField::SmdVolume,
    DataField::SmdMass,
    DataField::SmdKradius,
    DataField::SmdCradius,
    DataField::SmdX0,
    DataField::SmdY0,
    DataField::SmdZ0,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

const DIELECTRIC: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::Charge,
    DataField::X,
    DataField::Y,
    DataField::Z,
    DataField::Mux,
    DataField::Muy,
    DataField::Muz,
    DataField::Area,
    DataField::Ed,
    DataField::Em,
    DataField::Epsilon,
    DataField::Curvature,
];

const HYBRID: &[DataField] = &[
    DataField::Id,
    DataField::Type,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

fn layout(fields: &'static [DataField], flexible_tail: bool) -> AtomStyleLayout {
    AtomStyleLayout {
        fields,
        flexible_tail,
    }
}

/// Strip accelerator package suffixes (`/kk`, `/gpu`, …) and lowercase.
///
/// Real multi-segment styles such as `bpm/sphere` and `rheo/thermal` are kept.
pub(crate) fn normalize_atom_style(raw: &str) -> String {
    let s = raw.trim().to_ascii_lowercase();
    for suffix in ["/kk", "/gpu", "/omp", "/opt", "/intel"] {
        if let Some(stripped) = s.strip_suffix(suffix) {
            return stripped.to_string();
        }
    }
    s
}

/// Parse the `# <style>` hint after an "Atoms" section header line.
pub(crate) fn parse_atoms_style_hint(section_line: &str) -> Option<String> {
    let raw = section_line
        .split_once('#')
        .map(|(_, after)| after.split_whitespace().next().unwrap_or(""))
        .filter(|s| !s.is_empty())?;
    Some(normalize_atom_style(raw))
}

// bpm/sphere: id mol type diameter density x y z
const BPM_SPHERE: &[DataField] = &[
    DataField::Id,
    DataField::Mol,
    DataField::Type,
    DataField::Diameter,
    DataField::Density,
    DataField::X,
    DataField::Y,
    DataField::Z,
];

/// Return the fixed column layout for a normalised atom style name.
pub(crate) fn layout_for_atom_style(style: &str) -> Option<AtomStyleLayout> {
    Some(match style {
        "atomic" => layout(ATOMIC, false),
        "charge" => layout(CHARGE, false),
        "dpd" => layout(SCALAR_XYZ, false),
        "mdpd" => layout(MDPD, false),
        "bond" | "angle" | "molecular" => layout(MOLECULAR, false),
        "full" | "amoeba" => layout(FULL, false),
        "body" => layout(BODY, false),
        "sphere" => layout(SPHERE, false),
        "bpm/sphere" => layout(BPM_SPHERE, false),
        "ellipsoid" => layout(ELLIPSOID, false),
        "peri" => layout(PERI, false),
        "edpd" => layout(EDPD, false),
        "line" | "tri" => layout(LINE, false),
        "template" => layout(TEMPLATE, false),
        "sph" => layout(SPH, false),
        "electron" => layout(ELECTRON, false),
        "dipole" => layout(DIPOLE, false),
        "spin" => layout(SPIN, false),
        "rheo" => layout(RHEO, false),
        "rheo/thermal" => layout(RHEO_THERMAL, false),
        "smd" => layout(SMD, false),
        "dielectric" => layout(DIELECTRIC, false),
        s if s.starts_with("hybrid") => layout(HYBRID, true),
        _ => return None,
    })
}

/// Infer a layout from column count when no usable style hint is present.
///
/// Image flags add exactly 3 columns (handled by the atom-line parser against
/// `min_cols`). Ambiguous counts that need a style comment (`body`/`sphere`
/// at 7 cols) default to the most common molecular layouts.
pub(crate) fn layout_from_column_count(n: usize) -> std::io::Result<AtomStyleLayout> {
    use crate::io::invalid_data;
    match n {
        5 | 8 => Ok(layout(ATOMIC, false)),
        // charge vs molecular: disambiguated per-line by the data reader.
        6 | 9 => Ok(layout(CHARGE, false)),
        7 | 10 => Ok(layout(FULL, false)),
        _ => Err(invalid_data(format!(
            "Invalid Atoms line: unsupported column count {n} without a known \
             atom_style comment (expected 5–10 for common styles, or a style \
             hint such as `Atoms # angle`)"
        ))),
    }
}

/// True if the token looks like a non-integer float (charge disambiguation).
pub(crate) fn is_noninteger_float_token(token: &str) -> bool {
    token.contains('.') || token.contains('e') || token.contains('E')
}

/// Frame column key for a data-file Atoms field (canonical names where they exist).
pub(crate) fn field_column_key(field: DataField) -> &'static str {
    use molrs::core::keys;
    match field {
        DataField::Id => keys::ID,
        // LAMMPS numbers its types; the label lives in `type`.
        DataField::Type => keys::TYPE_ID,
        DataField::Mol => keys::MOL_ID,
        DataField::Charge => keys::CHARGE,
        DataField::X => keys::X,
        DataField::Y => keys::Y,
        DataField::Z => keys::Z,
        DataField::Bodyflag => "bodyflag",
        DataField::Mass => keys::MASS,
        DataField::Diameter => "diameter",
        DataField::Density => "density",
        DataField::Volume => "volume",
        DataField::ShapeFlag => "shape_flag",
        DataField::Mux => keys::MUX,
        DataField::Muy => keys::MUY,
        DataField::Muz => keys::MUZ,
        DataField::Spx => "spx",
        DataField::Spy => "spy",
        DataField::Spz => "spz",
        DataField::Sp => "sp",
        DataField::Rho => "rho",
        DataField::Esph => "esph",
        DataField::Cv => "cv",
        DataField::Theta => "theta",
        DataField::Espin => "espin",
        DataField::Eradius => "eradius",
        DataField::Status => "status",
        DataField::Energy => "energy",
        DataField::TemplateIndex => "template_index",
        DataField::TemplateAtom => "template_atom",
        DataField::EdpdTemp => "edpd_temp",
        DataField::EdpdCv => "edpd_cv",
        DataField::SmdVolume => "smd_volume",
        DataField::SmdMass => "smd_mass",
        DataField::SmdKradius => "smd_kradius",
        DataField::SmdCradius => "smd_cradius",
        DataField::SmdX0 => "x0",
        DataField::SmdY0 => "y0",
        DataField::SmdZ0 => "z0",
        DataField::Area => "area",
        DataField::Ed => "ed",
        DataField::Em => "em",
        DataField::Epsilon => "epsilon",
        DataField::Curvature => "curvature",
    }
}

/// Whether this field is always required for any write (core coordinates + ids).
pub(crate) fn is_core_field(field: DataField) -> bool {
    matches!(
        field,
        DataField::Id | DataField::Type | DataField::X | DataField::Y | DataField::Z
    )
}

/// Styles considered when inferring a write layout, most-specific first by field
/// count (tie-break: earlier in this list wins only after count comparison).
const WRITE_STYLE_CANDIDATES: &[&str] = &[
    "dielectric",
    "smd",
    "dipole",
    "electron",
    "sph",
    "spin",
    "rheo/thermal",
    "rheo",
    "bpm/sphere",
    "line",
    "tri",
    "template",
    "ellipsoid",
    "peri",
    "edpd",
    "sphere",
    "body",
    "full",
    "molecular",
    "charge",
    "dpd",
    "mdpd",
    "atomic",
];

/// Infer the best `Atoms # <style>` layout for writing, given a predicate that
/// reports whether a Frame has a column for each [`DataField`].
///
/// Among layouts whose non-core fields are all present, pick the one with the
/// most fields (most specific). Falls back to `atomic`.
pub(crate) fn infer_write_style(
    has_field: impl Fn(DataField) -> bool,
) -> (&'static str, AtomStyleLayout) {
    let mut best_name: &'static str = "atomic";
    let mut best = layout(ATOMIC, false);
    let mut best_n = best.min_cols();

    for &name in WRITE_STYLE_CANDIDATES {
        let Some(lay) = layout_for_atom_style(name) else {
            continue;
        };
        let ok = lay.fields.iter().all(|&f| is_core_field(f) || has_field(f));
        if ok && lay.min_cols() > best_n {
            best_name = name;
            best = lay;
            best_n = lay.min_cols();
        }
    }

    // bond/angle share molecular layout — emit the molecular keyword.
    if matches!(best_name, "bond" | "angle") {
        best_name = "molecular";
    }
    (best_name, best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accelerator_suffixes_are_stripped_and_real_slashes_are_kept() {
        assert_eq!(normalize_atom_style(" Full/kk "), "full");
        assert_eq!(normalize_atom_style("charge/omp"), "charge");
        assert_eq!(normalize_atom_style("bpm/sphere"), "bpm/sphere");
    }

    #[test]
    fn the_style_hint_is_the_first_token_after_the_hash() {
        assert_eq!(
            parse_atoms_style_hint("Atoms # charge").as_deref(),
            Some("charge")
        );
        assert_eq!(
            parse_atoms_style_hint("Atoms # full/gpu extra").as_deref(),
            Some("full")
        );
        assert_eq!(parse_atoms_style_hint("Atoms"), None);
        assert_eq!(parse_atoms_style_hint("Atoms #"), None);
    }

    #[test]
    fn layouts_follow_the_read_data_table() {
        let full = layout_for_atom_style("full").unwrap();
        assert_eq!(
            full.fields,
            &[
                DataField::Id,
                DataField::Mol,
                DataField::Type,
                DataField::Charge,
                DataField::X,
                DataField::Y,
                DataField::Z
            ]
        );
        assert_eq!(full.min_cols(), 7);
        assert!(!full.flexible_tail);
        assert!(
            layout_for_atom_style("hybrid full charge")
                .unwrap()
                .flexible_tail
        );
        assert!(layout_for_atom_style("no-such-style").is_none());
    }

    #[test]
    fn column_counts_map_to_the_common_layouts_and_reject_the_rest() {
        assert_eq!(layout_from_column_count(5).unwrap().fields, ATOMIC);
        assert_eq!(
            layout_from_column_count(8).unwrap().fields,
            ATOMIC,
            "with image flags"
        );
        assert_eq!(layout_from_column_count(6).unwrap().fields, CHARGE);
        assert_eq!(layout_from_column_count(7).unwrap().fields, FULL);
        assert!(layout_from_column_count(4).is_err());
        assert!(layout_from_column_count(11).is_err());
    }

    #[test]
    fn integer_tokens_allow_one_leading_sign_only() {
        assert!(is_int_token("42"));
        assert!(is_int_token("-1"));
        assert!(is_int_token("+7"));
        assert!(!is_int_token(""));
        assert!(!is_int_token("1.0"));
        assert!(!is_int_token("--1"));
    }
}
