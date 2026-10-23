//! The two decisions a LAMMPS data file shares with its input script: the
//! `atom_style` of its `Atoms` section and the `units` its numbers are in.
//!
//! The input script must say the same `atom_style` and `units` as the data
//! file it reads, so the data writer owns both: a caller either names them
//! ([`LammpsDataWriteOptions`]) or asks what the writer will infer
//! ([`infer_lammps_atom_style`]) and writes that into its script.

use crate::io::invalid_data;
use crate::io::lammps::atom_style::{
    AtomStyleLayout, DataField, field_column_key, infer_write_style, is_core_field,
    layout_for_atom_style, normalize_atom_style,
};
use crate::io::lammps::data::frame_has_atom_field;
use crate::io::lammps::units::{LammpsUnitConverter, parse_lammps_units_style};
use molrs::core::keys::{self, LAMMPS_UNITS};
use molrs::core::{FrameAccess, PresetDim};
use molrs::op::F;

/// The unit style of a frame that states none: molrs's own convention
/// (Å, g/mol, e, Å/fs), which is LAMMPS `real`.
pub(crate) const DEFAULT_FRAME_UNITS: &str = "real";

/// The topology blocks a data file can carry.
const TOPOLOGY_BLOCKS: [&str; 5] = ["bonds", "angles", "dihedrals", "impropers", "cmaps"];

/// How a LAMMPS data file is written: its `atom_style` and `units`.
///
/// The default (`None`, `None`) is the writer's long-standing behaviour: the
/// style inferred from the frame's columns, the frame's numbers unchanged.
///
/// ```
/// use molrs::io::lammps::LammpsDataWriteOptions;
///
/// let options = LammpsDataWriteOptions {
///     atom_style: Some("full".into()),
///     units: Some("metal".into()),
/// };
/// assert_ne!(options, LammpsDataWriteOptions::default());
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LammpsDataWriteOptions {
    /// The `atom_style` of the `Atoms` section (accelerator suffixes such as
    /// `/kk` are dropped). `None` infers it from the frame's columns
    /// ([`infer_lammps_atom_style`]).
    ///
    /// A named style is written exactly: every column it needs must be in
    /// `frame["atoms"]`, or the write fails naming the missing ones. The
    /// writer never fills one in — a `charge` of 0 or a `mol_id` of 1 would
    /// be a silent physics choice (`fix rigid molecule`, `neigh_modify
    /// exclude molecule/intra`, a charged system written neutral). A style
    /// that cannot hold the frame's topology (`charge` with bonds, `bond`
    /// with angles) and a `hybrid` style are refused too.
    pub atom_style: Option<String>,
    /// The LAMMPS `units` style the file is written in: `lj`, `real` or
    /// `metal`, the styles the force-field writer converts to
    /// ([`LammpsUnitConverter`], `from → lj → to`). Positions, box, masses,
    /// charges, velocities and the dipole and radius columns are converted
    /// from the frame's units — its `units` meta preset, else its
    /// `lammps_units` meta, else `real` — and the title line records the style
    /// (`units = metal`) as `write_data` does, which the reader puts back in
    /// `lammps_units`. `None` writes the frame's numbers unchanged and states
    /// no units.
    pub units: Option<String>,
}

/// The `atom_style` [`write_lammps_data`](crate::io::write_lammps_data)
/// writes `frame` in when none is named: among the styles whose columns
/// `frame["atoms"]` has, the one with the most columns (`full` for a frame
/// with `mol_id` and `charge`, `atomic` for one with neither). Write it into
/// the input script's `atom_style` line.
///
/// ```
/// use molrs::core::{Block, Frame};
/// use molrs::io::lammps::infer_lammps_atom_style;
/// use molrs::op::F;
/// use ndarray::array;
///
/// let mut atoms = Block::new();
/// for key in ["x", "y", "z"] {
///     atoms.insert(key, array![0.0 as F].into_dyn(), None).unwrap();
/// }
/// atoms.insert("type", array!["Ar".to_string()].into_dyn(), None).unwrap();
/// let mut frame = Frame::new();
/// frame.insert("atoms", atoms);
/// assert_eq!(infer_lammps_atom_style(&frame), "atomic");
/// ```
pub fn infer_lammps_atom_style(frame: &impl FrameAccess) -> &'static str {
    infer_write_style(|f| frame_has_atom_field(frame, f)).0
}

/// The style name the `Atoms` header states and its column layout: the
/// named style, checked against the frame, or the inferred one.
pub(crate) fn resolve_write_style(
    frame: &impl FrameAccess,
    requested: Option<&str>,
) -> std::io::Result<(String, AtomStyleLayout)> {
    let Some(raw) = requested else {
        let (name, layout) = infer_write_style(|f| frame_has_atom_field(frame, f));
        return Ok((name.to_owned(), layout));
    };
    let name = normalize_atom_style(raw);
    if name.starts_with("hybrid") {
        return Err(invalid_data(format!(
            "atom_style `{raw}`: the data writer cannot write a hybrid style \
             (its sub-styles' columns are not known); name a single style"
        )));
    }
    let layout = layout_for_atom_style(&name)
        .ok_or_else(|| invalid_data(format!("unknown LAMMPS atom_style `{raw}`")))?;
    let missing: Vec<&str> = layout
        .fields
        .iter()
        .copied()
        .filter(|&f| !is_core_field(f) && !frame_has_atom_field(frame, f))
        .map(field_column_key)
        .collect();
    if !missing.is_empty() {
        return Err(invalid_data(format!(
            "atom_style `{name}` needs the frame['atoms'] column(s) {missing:?}, which the \
             frame lacks; the writer fills none in. Add them (a 'charge' of 0.0 for a \
             neutral system, a 'mol_id' per molecule), or write the style the frame \
             holds: `{}` (infer_lammps_atom_style)",
            infer_lammps_atom_style(frame)
        )));
    }
    check_topology_fits(frame, &name)?;
    Ok((name, layout))
}

/// The topology blocks `read_data` accepts under atom style `style`.
fn topology_capacity(style: &str) -> &'static [&'static str] {
    match style {
        "bond" | "bpm/sphere" => &TOPOLOGY_BLOCKS[..1],
        "angle" => &TOPOLOGY_BLOCKS[..2],
        "molecular" | "full" | "amoeba" => &TOPOLOGY_BLOCKS,
        _ => &[],
    }
}

/// Refuse a named style whose atoms cannot carry the frame's topology:
/// `read_data` would reject the file (`atom_style charge` with `Bonds`).
fn check_topology_fits(frame: &impl FrameAccess, style: &str) -> std::io::Result<()> {
    let allowed = topology_capacity(style);
    // A Drude spring is written as a bond.
    let sections = TOPOLOGY_BLOCKS
        .iter()
        .map(|&block| (block, block))
        .chain([crate::io::lammps::drude::SPRINGS]);
    for (block, section) in sections {
        let n = frame
            .visit_block(block, |b| b.n_rows().unwrap_or(0))
            .unwrap_or(0);
        if n > 0 && !allowed.contains(&section) {
            return Err(invalid_data(format!(
                "atom_style `{style}` cannot hold frame['{block}'] ({n} rows): LAMMPS \
                 read_data refuses that section under this style"
            )));
        }
    }
    Ok(())
}

/// The per-dimension factors from the frame's units to the file's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DataUnits {
    /// The style the title line records; `None` when no `units` was named.
    pub title: Option<&'static str>,
    from: &'static str,
    to: &'static str,
    pub length: F,
    pub mass: F,
    pub charge: F,
    pub velocity: F,
}

impl DataUnits {
    /// No conversion, no units stated.
    const UNCHANGED: Self = Self {
        title: None,
        from: DEFAULT_FRAME_UNITS,
        to: DEFAULT_FRAME_UNITS,
        length: 1.0,
        mass: 1.0,
        charge: 1.0,
        velocity: 1.0,
    };

    /// Whether the numbers change on the way out.
    fn converts(&self) -> bool {
        self.from != self.to
    }

    /// The factor of one `Atoms` column, or an error naming a column whose
    /// dimension the writer does not convert (only when it must convert).
    pub(crate) fn field_factor(&self, field: DataField) -> std::io::Result<F> {
        if !self.converts() {
            return Ok(1.0);
        }
        use DataField as D;
        let factor = match field {
            D::Id
            | D::Type
            | D::Mol
            | D::Bodyflag
            | D::ShapeFlag
            | D::Status
            | D::TemplateIndex
            | D::TemplateAtom
            | D::Espin => Some(1.0),
            D::X
            | D::Y
            | D::Z
            | D::Diameter
            | D::Eradius
            | D::SmdKradius
            | D::SmdCradius
            | D::SmdX0
            | D::SmdY0
            | D::SmdZ0 => Some(self.length),
            D::Mass | D::SmdMass => Some(self.mass),
            D::Charge => Some(self.charge),
            D::Mux | D::Muy | D::Muz => Some(self.charge * self.length),
            _ => None,
        };
        factor.ok_or_else(|| {
            invalid_data(format!(
                "frame['atoms'] column '{}' has no unit conversion from `{}` to `{}`; \
                 write this atom style in the frame's own units (units=None)",
                field_column_key(field),
                self.from,
                self.to
            ))
        })
    }
}

/// The LAMMPS unit style `frame`'s numbers are in: its `units` meta preset,
/// else the `lammps_units` a data file's title stated, else `real`.
fn frame_units(frame: &impl FrameAccess) -> std::io::Result<&'static str> {
    let meta = frame.meta_ref();
    let stated = if let Some(value) = meta.get(keys::UNITS) {
        keys::units_preset(value).ok_or_else(|| {
            invalid_data(format!(
                "frame meta '{}' names no preset, so the data writer cannot convert from it",
                keys::UNITS
            ))
        })?
    } else if let Some(value) = meta.get(LAMMPS_UNITS) {
        value
            .as_str()
            .ok_or_else(|| invalid_data(format!("frame meta '{LAMMPS_UNITS}' must be a string")))?
    } else {
        return Ok(DEFAULT_FRAME_UNITS);
    };
    parse_lammps_units_style(stated).map_err(|e| invalid_data(format!("the frame's units: {e}")))
}

/// The conversion from `frame`'s units to `target` (a LAMMPS `units` style).
pub(crate) fn resolve_units(
    frame: &impl FrameAccess,
    target: Option<&str>,
) -> std::io::Result<DataUnits> {
    let Some(raw) = target else {
        return Ok(DataUnits::UNCHANGED);
    };
    let to = parse_lammps_units_style(raw)
        .map_err(|e| invalid_data(format!("LAMMPS data file units: {e}")))?;
    let from = frame_units(frame)?;
    let converter = LammpsUnitConverter::canonical()
        .map_err(|e| invalid_data(format!("lammps unit system: {e}")))?;
    let one = |dimension| {
        converter
            .convert(1.0, from, to, dimension)
            .map_err(invalid_data)
    };
    Ok(DataUnits {
        title: Some(to),
        from,
        to,
        length: one(PresetDim::Length)?,
        mass: one(PresetDim::Mass)?,
        charge: one(PresetDim::Charge)?,
        velocity: one(PresetDim::Velocity)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::{Frame, MetaValue};

    /// Two charged atoms in one molecule, bonded, moving.
    const FULL: &str = concat!(
        "LAMMPS data file\n\n2 atoms\n1 bonds\n1 atom types\n1 bond types\n\n",
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
        "Masses\n\n1 12.0\n\n",
        "Atoms # full\n\n",
        "1 1 1 -0.5 1.0 2.0 3.0\n",
        "2 1 1 0.5 4.0 5.0 6.0\n",
        "\nVelocities\n\n",
        "1 0.1 0.2 0.3\n",
        "2 -0.1 0.0 0.5\n",
        "\nBonds\n\n1 1 1 2\n",
    );

    /// An LJ fluid: no charge, no molecule.
    const ATOMIC: &str = concat!(
        "LAMMPS data file\n\n2 atoms\n1 atom types\n\n",
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
        "Masses\n\n1 39.948\n\n",
        "Atoms # atomic\n\n",
        "1 1 1.0 2.0 3.0\n",
        "2 1 4.0 5.0 6.0\n",
    );

    fn frame(text: &str) -> Frame {
        crate::io::read_lammps_data((text).as_bytes()).expect("parse")
    }

    fn write(
        frame: &Frame,
        atom_style: Option<&str>,
        units: Option<&str>,
    ) -> std::io::Result<String> {
        crate::io::writer::capture(|sink| {
            use crate::io::writer::FrameWriter;
            crate::io::lammps::LammpsDataWriter::new(sink)
                .with_options(
                    LammpsDataWriteOptions {
                        atom_style: atom_style.map(str::to_owned),
                        units: units.map(str::to_owned),
                    }
                    .clone(),
                )
                .write(frame)
        })
    }

    fn floats(frame: &Frame, key: &str) -> Vec<F> {
        frame
            .column("atoms", key)
            .and_then(|c| c.get::<f64>())
            .unwrap_or_else(|| panic!("no float column {key}"))
            .iter()
            .copied()
            .collect()
    }

    fn atoms_lines(text: &str) -> Vec<&str> {
        text.split("Atoms #")
            .nth(1)
            .expect("an Atoms section")
            .lines()
            .skip(2)
            .take_while(|l| !l.is_empty())
            .collect()
    }

    fn close(a: &[F], b: &[F]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= 1e-9 * y.abs().max(1.0))
    }

    // ---- atom_style ------------------------------------------------------

    #[test]
    fn inference_names_the_style_the_default_writer_writes() {
        for (text, style) in [(FULL, "full"), (ATOMIC, "atomic")] {
            let f = frame(text);
            assert_eq!(infer_lammps_atom_style(&f), style);
            let out =
                crate::io::writer::capture(|sink| crate::io::write_lammps_data(sink, &f)).unwrap();
            assert!(out.contains(&format!("Atoms # {style}\n")), "{out}");
        }
    }

    #[test]
    fn the_default_options_are_the_old_writer_byte_for_byte() {
        for text in [FULL, ATOMIC] {
            let f = frame(text);
            let old =
                crate::io::writer::capture(|sink| crate::io::write_lammps_data(sink, &f)).unwrap();
            assert_eq!(write(&f, None, None).unwrap(), old);
            assert!(old.starts_with("# LAMMPS data file generated by molrs\n"));
            // Naming the inferred style changes nothing either.
            let inferred = infer_lammps_atom_style(&f);
            assert_eq!(write(&f, Some(inferred), None).unwrap(), old);
        }
    }

    #[test]
    fn a_named_style_missing_columns_is_refused_naming_them() {
        let lj = frame(ATOMIC);
        let err = write(&lj, Some("full"), None).unwrap_err().to_string();
        assert!(
            err.contains("\"mol_id\"") && err.contains("\"charge\""),
            "{err}"
        );
        assert!(
            err.contains("`atomic`"),
            "names the style the frame holds: {err}"
        );
        let err = write(&lj, Some("charge"), None).unwrap_err().to_string();
        assert!(err.contains("column(s) [\"charge\"]"), "{err}");
    }

    #[test]
    fn a_named_style_is_written_exactly() {
        let f = frame(FULL);
        let molecular = {
            let mut f = f.clone();
            f.remove("bonds");
            f
        };
        let out = write(&molecular, Some("charge"), None).unwrap();
        assert!(out.contains("Atoms # charge\n"), "{out}");
        assert_eq!(atoms_lines(&out), ["1 1 -0.5 1 2 3", "2 1 0.5 4 5 6"]);
        let out = write(&f, Some("molecular"), None).unwrap();
        assert_eq!(atoms_lines(&out), ["1 1 1 1 2 3", "2 1 1 4 5 6"]);
        // Accelerator suffixes are not part of the style.
        let out = write(&f, Some("Full/kk"), None).unwrap();
        assert!(out.contains("Atoms # full\n"), "{out}");
    }

    #[test]
    fn a_named_style_that_cannot_hold_the_topology_is_refused() {
        let f = frame(FULL);
        let err = write(&f, Some("charge"), None).unwrap_err().to_string();
        assert!(err.contains("frame['bonds']"), "{err}");
        assert!(write(&f, Some("bond"), None).is_ok());
    }

    #[test]
    fn hybrid_and_unknown_styles_are_refused() {
        let f = frame(FULL);
        let err = write(&f, Some("hybrid full dipole"), None).unwrap_err();
        assert!(err.to_string().contains("hybrid"), "{err}");
        let err = write(&f, Some("nonsense"), None).unwrap_err();
        assert!(err.to_string().contains("nonsense"), "{err}");
    }

    // ---- units -----------------------------------------------------------

    #[test]
    fn metal_converts_velocities_and_records_the_style_for_the_reader() {
        let f = frame(FULL);
        let out = write(&f, None, Some("metal")).unwrap();
        assert!(
            out.starts_with("# LAMMPS data file generated by molrs, units = metal\n"),
            "{out}"
        );
        let back = frame(&out);
        assert_eq!(
            back.meta.get(LAMMPS_UNITS).and_then(MetaValue::as_str),
            Some("metal")
        );
        // Å/fs → Å/ps; Å, g/mol and e are metal's too.
        assert!(close(&floats(&back, keys::VX), &[100.0, -100.0]));
        assert!(close(&floats(&back, keys::VZ), &[300.0, 500.0]));
        assert!(close(&floats(&back, keys::X), &floats(&f, keys::X)));
        assert!(close(&floats(&back, keys::CHARGE), &[-0.5, 0.5]));
        assert!(out.contains("Masses\n\n1 12\n"), "{out}");
    }

    #[test]
    fn a_converted_file_read_back_converts_home() {
        let f = frame(FULL);
        for units in ["metal"] {
            let there = frame(&write(&f, None, Some(units)).unwrap());
            // The reader's `lammps_units` is the frame's units on the way back.
            let home = frame(&write(&there, None, Some("real")).unwrap());
            for key in [
                keys::X,
                keys::Y,
                keys::Z,
                keys::CHARGE,
                keys::VX,
                keys::VY,
                keys::VZ,
            ] {
                assert!(
                    close(&floats(&home, key), &floats(&f, key)),
                    "{units} {key}: {:?} vs {:?}",
                    floats(&home, key),
                    floats(&f, key)
                );
            }
        }
    }

    /// A frame written as `lj` needs the system's σ, ε and mass, which the
    /// writer is not given: refused, never relabelled.
    #[test]
    fn lj_is_refused_without_a_reference() {
        let f = frame(FULL);
        let err = write(&f, Some("full"), Some("lj")).unwrap_err();
        assert!(err.to_string().contains("lj reference"), "{err}");
    }

    #[test]
    fn the_frames_own_units_are_the_source() {
        let mut f = frame(FULL);
        f.meta.insert(LAMMPS_UNITS, "metal");
        let same = frame(&write(&f, None, Some("metal")).unwrap());
        assert!(close(&floats(&same, keys::VX), &[0.1, -0.1]));
        let real = frame(&write(&f, None, Some("real")).unwrap());
        assert!(close(&floats(&real, keys::VX), &[1e-4, -1e-4]));
    }

    #[test]
    fn an_unsupported_unit_style_is_refused_by_name() {
        let f = frame(FULL);
        for units in ["nano", "si", "furlong"] {
            let err = write(&f, None, Some(units)).unwrap_err().to_string();
            assert!(
                err.contains(units) && err.contains("lj, real, metal"),
                "{err}"
            );
        }
        let mut stated = frame(FULL);
        stated.meta.insert(LAMMPS_UNITS, "nano");
        let err = write(&stated, None, Some("real")).unwrap_err().to_string();
        assert!(
            err.contains("frame's units") && err.contains("nano"),
            "{err}"
        );
    }

    #[test]
    fn a_column_without_a_known_dimension_is_refused_only_when_converting() {
        let sphere = frame(concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
            "Atoms # sphere\n\n1 1 2.0 1.5 1.0 2.0 3.0\n",
        ));
        assert!(write(&sphere, Some("sphere"), Some("real")).is_ok());
        let err = write(&sphere, Some("sphere"), Some("metal"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("'density'") && err.contains("units=None"),
            "{err}"
        );
    }
}
