//! LAMMPS data file format reader and writer.

use crate::io::frame_columns::insert_column_of_length;
use crate::io::frame_index::{FrameIndexBuilder, FrameOffset};
use crate::io::invalid_data;
use crate::io::lammps::atom_style::{
    AtomStyleLayout, DataField, field_column_key, is_int_token, is_noninteger_float_token,
    layout_for_atom_style, layout_from_column_count, normalize_atom_style, parse_atoms_style_hint,
};
use crate::io::lammps::box_bounds::{BoxBounds, simbox_from_bounds};
use crate::io::lammps::columns::OptionalColumn;
use crate::io::lammps::data_options::{
    DataUnits, LammpsDataWriteOptions, resolve_units, resolve_write_style,
};
use crate::io::lammps::drude::{SPRINGS, drude_flags, lammps_type_labels, refuse_virtual_sites};
use crate::io::lammps::fields::{
    TypeRef, invert_type_labels, labels_to_meta, parse_f, parse_i, tokenize,
};
use crate::io::reader::{FrameReader, Reader};
use crate::io::writer::FrameWriter;
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::FrameAccess;
use molrs::core::SimBox;
use molrs::core::keys;
use molrs::op::{F, I, Idx, Pbc3};
use ndarray::ArrayViewD;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::OnceLock;

use crate::core::keys::{LAMMPS_COEFFS_TEXT, LAMMPS_UNITS};

// ============================================================================
// Header
// ============================================================================

#[derive(Debug, Clone, Default)]
struct LammpsDataHeader {
    n_atoms: usize,
    n_bonds: usize,
    n_angles: usize,
    n_dihedrals: usize,
    n_impropers: usize,
    /// `N crossterms` (or `N cmap crossterms`): the rows of the `CMAP`
    /// section LAMMPS `fix cmap` reads.
    n_crossterms: usize,
    n_atom_types: usize,
    n_bond_types: usize,
    n_angle_types: usize,
    n_dihedral_types: usize,
    n_improper_types: usize,
    bounds: BoxBounds,
    /// The `units = <style>` field of the title line, as `write_data` writes
    /// it; `None` when the title does not state one.
    units: Option<String>,
}

// ============================================================================
// Streaming atom columns
// ============================================================================

/// Column-oriented atom buffer filled directly from the Atoms section.
struct AtomColumns {
    id: Vec<I>,
    type_refs: Vec<TypeRef>,
    x: Vec<F>,
    y: Vec<F>,
    z: Vec<F>,
    mol: OptionalColumn<I>,
    charge: OptionalColumn<F>,
    bodyflag: OptionalColumn<I>,
    mass: OptionalColumn<F>,
    diameter: OptionalColumn<F>,
    density: OptionalColumn<F>,
    volume: OptionalColumn<F>,
    shape_flag: OptionalColumn<I>,
    mux: OptionalColumn<F>,
    muy: OptionalColumn<F>,
    muz: OptionalColumn<F>,
    spx: OptionalColumn<F>,
    spy: OptionalColumn<F>,
    spz: OptionalColumn<F>,
    sp: OptionalColumn<F>,
    rho: OptionalColumn<F>,
    esph: OptionalColumn<F>,
    cv: OptionalColumn<F>,
    theta: OptionalColumn<F>,
    espin: OptionalColumn<I>,
    eradius: OptionalColumn<F>,
    status: OptionalColumn<I>,
    energy: OptionalColumn<F>,
    template_index: OptionalColumn<I>,
    template_atom: OptionalColumn<I>,
    edpd_temp: OptionalColumn<F>,
    edpd_cv: OptionalColumn<F>,
    smd_volume: OptionalColumn<F>,
    smd_mass: OptionalColumn<F>,
    smd_kradius: OptionalColumn<F>,
    smd_cradius: OptionalColumn<F>,
    smd_x0: OptionalColumn<F>,
    smd_y0: OptionalColumn<F>,
    smd_z0: OptionalColumn<F>,
    area: OptionalColumn<F>,
    ed: OptionalColumn<F>,
    em: OptionalColumn<F>,
    epsilon: OptionalColumn<F>,
    curvature: OptionalColumn<F>,
    ix: OptionalColumn<I>,
    iy: OptionalColumn<I>,
    iz: OptionalColumn<I>,
    vx: OptionalColumn<F>,
    vy: OptionalColumn<F>,
    vz: OptionalColumn<F>,
}

impl AtomColumns {
    fn with_capacity(n: usize) -> Self {
        Self {
            id: Vec::with_capacity(n),
            type_refs: Vec::with_capacity(n),
            x: Vec::with_capacity(n),
            y: Vec::with_capacity(n),
            z: Vec::with_capacity(n),
            mol: OptionalColumn::with_capacity(n),
            charge: OptionalColumn::with_capacity(n),
            bodyflag: OptionalColumn::with_capacity(n),
            mass: OptionalColumn::with_capacity(n),
            diameter: OptionalColumn::with_capacity(n),
            density: OptionalColumn::with_capacity(n),
            volume: OptionalColumn::with_capacity(n),
            shape_flag: OptionalColumn::with_capacity(n),
            mux: OptionalColumn::with_capacity(n),
            muy: OptionalColumn::with_capacity(n),
            muz: OptionalColumn::with_capacity(n),
            spx: OptionalColumn::with_capacity(n),
            spy: OptionalColumn::with_capacity(n),
            spz: OptionalColumn::with_capacity(n),
            sp: OptionalColumn::with_capacity(n),
            rho: OptionalColumn::with_capacity(n),
            esph: OptionalColumn::with_capacity(n),
            cv: OptionalColumn::with_capacity(n),
            theta: OptionalColumn::with_capacity(n),
            espin: OptionalColumn::with_capacity(n),
            eradius: OptionalColumn::with_capacity(n),
            status: OptionalColumn::with_capacity(n),
            energy: OptionalColumn::with_capacity(n),
            template_index: OptionalColumn::with_capacity(n),
            template_atom: OptionalColumn::with_capacity(n),
            edpd_temp: OptionalColumn::with_capacity(n),
            edpd_cv: OptionalColumn::with_capacity(n),
            smd_volume: OptionalColumn::with_capacity(n),
            smd_mass: OptionalColumn::with_capacity(n),
            smd_kradius: OptionalColumn::with_capacity(n),
            smd_cradius: OptionalColumn::with_capacity(n),
            smd_x0: OptionalColumn::with_capacity(n),
            smd_y0: OptionalColumn::with_capacity(n),
            smd_z0: OptionalColumn::with_capacity(n),
            area: OptionalColumn::with_capacity(n),
            ed: OptionalColumn::with_capacity(n),
            em: OptionalColumn::with_capacity(n),
            epsilon: OptionalColumn::with_capacity(n),
            curvature: OptionalColumn::with_capacity(n),
            ix: OptionalColumn::with_capacity(n),
            iy: OptionalColumn::with_capacity(n),
            iz: OptionalColumn::with_capacity(n),
            vx: OptionalColumn::with_capacity(n),
            vy: OptionalColumn::with_capacity(n),
            vz: OptionalColumn::with_capacity(n),
        }
    }

    fn len(&self) -> usize {
        self.id.len()
    }

    fn push_defaults_for_absent_optionals(&mut self) {
        // After streaming a row that only touches present fields, pad optionals
        // that were not written this row. Called once per row after field walk.
        let n = self.id.len();
        let pad_i = |c: &mut OptionalColumn<I>| c.values.resize(n, 0);
        let pad_f = |c: &mut OptionalColumn<F>| c.values.resize(n, 0.0);
        pad_i(&mut self.mol);
        pad_f(&mut self.charge);
        pad_i(&mut self.bodyflag);
        pad_f(&mut self.mass);
        pad_f(&mut self.diameter);
        pad_f(&mut self.density);
        pad_f(&mut self.volume);
        pad_i(&mut self.shape_flag);
        pad_f(&mut self.mux);
        pad_f(&mut self.muy);
        pad_f(&mut self.muz);
        pad_f(&mut self.spx);
        pad_f(&mut self.spy);
        pad_f(&mut self.spz);
        pad_f(&mut self.sp);
        pad_f(&mut self.rho);
        pad_f(&mut self.esph);
        pad_f(&mut self.cv);
        pad_f(&mut self.theta);
        pad_i(&mut self.espin);
        pad_f(&mut self.eradius);
        pad_i(&mut self.status);
        pad_f(&mut self.energy);
        pad_i(&mut self.template_index);
        pad_i(&mut self.template_atom);
        pad_f(&mut self.edpd_temp);
        pad_f(&mut self.edpd_cv);
        pad_f(&mut self.smd_volume);
        pad_f(&mut self.smd_mass);
        pad_f(&mut self.smd_kradius);
        pad_f(&mut self.smd_cradius);
        pad_f(&mut self.smd_x0);
        pad_f(&mut self.smd_y0);
        pad_f(&mut self.smd_z0);
        pad_f(&mut self.area);
        pad_f(&mut self.ed);
        pad_f(&mut self.em);
        pad_f(&mut self.epsilon);
        pad_f(&mut self.curvature);
        pad_i(&mut self.ix);
        pad_i(&mut self.iy);
        pad_i(&mut self.iz);
        pad_f(&mut self.vx);
        pad_f(&mut self.vy);
        pad_f(&mut self.vz);
    }

    fn into_block(self, atom_type_labels: &HashMap<String, String>) -> std::io::Result<Block> {
        let n = self.id.len();
        let label_to_id = invert_type_labels(atom_type_labels);
        let types: Vec<I> = self
            .type_refs
            .iter()
            .map(|t| t.resolve(&label_to_id))
            .collect();

        let mut block = Block::new();
        insert_column_of_length(
            &mut block,
            keys::ID,
            self.id.iter().map(|&v| v as Idx).collect(),
            n,
        )?;
        insert_column_of_length(
            &mut block,
            keys::TYPE_ID,
            types.iter().map(|&v| v as Idx).collect(),
            n,
        )?;
        insert_column_of_length(&mut block, keys::X, self.x, n)?;
        insert_column_of_length(&mut block, keys::Y, self.y, n)?;
        insert_column_of_length(&mut block, keys::Z, self.z, n)?;

        macro_rules! opt_i {
            ($col:expr, $key:expr) => {
                if $col.present {
                    insert_column_of_length(&mut block, $key, $col.values, n)?;
                }
            };
        }
        // Unsigned columns of the canonical vocabulary (ids, not quantities).
        macro_rules! opt_u {
            ($col:expr, $key:expr) => {
                if $col.present {
                    insert_column_of_length(
                        &mut block,
                        $key,
                        $col.values.iter().map(|&v| v as Idx).collect(),
                        n,
                    )?;
                }
            };
        }
        macro_rules! opt_f {
            ($col:expr, $key:expr) => {
                if $col.present {
                    insert_column_of_length(&mut block, $key, $col.values, n)?;
                }
            };
        }

        opt_u!(self.mol, keys::MOL_ID);
        opt_f!(self.charge, keys::CHARGE);
        opt_i!(self.bodyflag, "bodyflag");
        opt_f!(self.mass, keys::MASS);
        opt_f!(self.diameter, "diameter");
        opt_f!(self.density, "density");
        opt_f!(self.volume, "volume");
        opt_i!(self.shape_flag, "shape_flag");
        opt_f!(self.mux, keys::MUX);
        opt_f!(self.muy, keys::MUY);
        opt_f!(self.muz, keys::MUZ);
        opt_f!(self.spx, "spx");
        opt_f!(self.spy, "spy");
        opt_f!(self.spz, "spz");
        opt_f!(self.sp, "sp");
        opt_f!(self.rho, "rho");
        opt_f!(self.esph, "esph");
        opt_f!(self.cv, "cv");
        opt_f!(self.theta, "theta");
        opt_i!(self.espin, "espin");
        opt_f!(self.eradius, "eradius");
        opt_i!(self.status, "status");
        opt_f!(self.energy, "energy");
        opt_i!(self.template_index, "template_index");
        opt_i!(self.template_atom, "template_atom");
        opt_f!(self.edpd_temp, "edpd_temp");
        opt_f!(self.edpd_cv, "edpd_cv");
        opt_f!(self.smd_volume, "smd_volume");
        opt_f!(self.smd_mass, "smd_mass");
        opt_f!(self.smd_kradius, "smd_kradius");
        opt_f!(self.smd_cradius, "smd_cradius");
        opt_f!(self.smd_x0, "x0");
        opt_f!(self.smd_y0, "y0");
        opt_f!(self.smd_z0, "z0");
        opt_f!(self.area, "area");
        opt_f!(self.ed, "ed");
        opt_f!(self.em, "em");
        opt_f!(self.epsilon, "epsilon");
        opt_f!(self.curvature, "curvature");
        opt_i!(self.ix, "ix");
        opt_i!(self.iy, "iy");
        opt_i!(self.iz, "iz");
        opt_f!(self.vx, keys::VX);
        opt_f!(self.vy, keys::VY);
        opt_f!(self.vz, keys::VZ);

        Ok(block)
    }
}

// ============================================================================
// Topology
// ============================================================================

struct TopologyTerm {
    type_ref: TypeRef,
    members: [I; 5],
    n_members: u8,
}

impl TopologyTerm {
    fn members(&self) -> &[I] {
        &self.members[..self.n_members as usize]
    }
}

// ============================================================================
// Atoms line → columns
// ============================================================================

fn resolve_layout(
    tokens: &[&str],
    known: Option<AtomStyleLayout>,
    style_known: bool,
) -> std::io::Result<AtomStyleLayout> {
    let mut layout = match known {
        Some(l) => l,
        None => layout_from_column_count(tokens.len())?,
    };

    // charge vs molecular when style unknown and col count is 6 or 9.
    if !style_known && (tokens.len() == 6 || tokens.len() == 9) && tokens.len() >= 3 {
        let token2 = tokens[2];
        let looks_like_charge = is_noninteger_float_token(token2) || !is_int_token(token2);
        let looks_like_molecular = is_int_token(tokens[1]) && is_int_token(token2);
        if looks_like_molecular && !looks_like_charge {
            layout = layout_for_atom_style("molecular").unwrap();
        } else if looks_like_charge {
            layout = layout_for_atom_style("charge").unwrap();
        }
    }
    Ok(layout)
}

fn push_atom_line(
    cols: &mut AtomColumns,
    tokens: &[&str],
    layout: AtomStyleLayout,
) -> std::io::Result<()> {
    let n = tokens.len();
    let min = layout.min_cols();
    if n < min {
        return Err(invalid_data(format!(
            "Invalid Atoms line: expected at least {min} columns, got {n}"
        )));
    }

    let image = if n == min + 3 {
        Some([
            parse_i(tokens[n - 3])?,
            parse_i(tokens[n - 2])?,
            parse_i(tokens[n - 1])?,
        ])
    } else if n == min {
        None
    } else if layout.flexible_tail && n > min {
        if n >= min + 3
            && is_int_token(tokens[n - 3])
            && is_int_token(tokens[n - 2])
            && is_int_token(tokens[n - 1])
        {
            Some([
                parse_i(tokens[n - 3])?,
                parse_i(tokens[n - 2])?,
                parse_i(tokens[n - 1])?,
            ])
        } else {
            None
        }
    } else {
        return Err(invalid_data(format!(
            "Invalid Atoms line: got {n} columns, layout expects {min} or {} \
             (with image flags)",
            min + 3
        )));
    };

    // Required xyz placeholders — filled by field walk.
    let mut got_id = false;
    let mut got_type = false;
    let mut got_x = false;

    for (i, field) in layout.fields.iter().enumerate() {
        let tok = tokens[i];
        match field {
            DataField::Id => {
                cols.id.push(parse_i(tok)?);
                got_id = true;
            }
            DataField::Type => {
                cols.type_refs.push(TypeRef::parse(tok));
                got_type = true;
            }
            DataField::Mol => cols.mol.push(parse_i(tok)?),
            DataField::Charge => cols.charge.push(parse_f(tok)?),
            DataField::X => {
                cols.x.push(parse_f(tok)?);
                got_x = true;
            }
            DataField::Y => cols.y.push(parse_f(tok)?),
            DataField::Z => cols.z.push(parse_f(tok)?),
            DataField::Bodyflag => cols.bodyflag.push(parse_i(tok)?),
            DataField::Mass => cols.mass.push(parse_f(tok)?),
            DataField::Diameter => cols.diameter.push(parse_f(tok)?),
            DataField::Density => cols.density.push(parse_f(tok)?),
            DataField::Volume => cols.volume.push(parse_f(tok)?),
            DataField::ShapeFlag => cols.shape_flag.push(parse_i(tok)?),
            DataField::Mux => cols.mux.push(parse_f(tok)?),
            DataField::Muy => cols.muy.push(parse_f(tok)?),
            DataField::Muz => cols.muz.push(parse_f(tok)?),
            DataField::Spx => cols.spx.push(parse_f(tok)?),
            DataField::Spy => cols.spy.push(parse_f(tok)?),
            DataField::Spz => cols.spz.push(parse_f(tok)?),
            DataField::Sp => cols.sp.push(parse_f(tok)?),
            DataField::Rho => cols.rho.push(parse_f(tok)?),
            DataField::Esph => cols.esph.push(parse_f(tok)?),
            DataField::Cv => cols.cv.push(parse_f(tok)?),
            DataField::Theta => cols.theta.push(parse_f(tok)?),
            DataField::Espin => cols.espin.push(parse_i(tok)?),
            DataField::Eradius => cols.eradius.push(parse_f(tok)?),
            DataField::Status => cols.status.push(parse_i(tok)?),
            DataField::Energy => cols.energy.push(parse_f(tok)?),
            DataField::TemplateIndex => cols.template_index.push(parse_i(tok)?),
            DataField::TemplateAtom => cols.template_atom.push(parse_i(tok)?),
            DataField::EdpdTemp => cols.edpd_temp.push(parse_f(tok)?),
            DataField::EdpdCv => cols.edpd_cv.push(parse_f(tok)?),
            DataField::SmdVolume => cols.smd_volume.push(parse_f(tok)?),
            DataField::SmdMass => cols.smd_mass.push(parse_f(tok)?),
            DataField::SmdKradius => cols.smd_kradius.push(parse_f(tok)?),
            DataField::SmdCradius => cols.smd_cradius.push(parse_f(tok)?),
            DataField::SmdX0 => cols.smd_x0.push(parse_f(tok)?),
            DataField::SmdY0 => cols.smd_y0.push(parse_f(tok)?),
            DataField::SmdZ0 => cols.smd_z0.push(parse_f(tok)?),
            DataField::Area => cols.area.push(parse_f(tok)?),
            DataField::Ed => cols.ed.push(parse_f(tok)?),
            DataField::Em => cols.em.push(parse_f(tok)?),
            DataField::Epsilon => cols.epsilon.push(parse_f(tok)?),
            DataField::Curvature => cols.curvature.push(parse_f(tok)?),
        }
    }

    if !got_id || !got_type || !got_x {
        return Err(invalid_data(
            "Invalid atom style layout: missing id, type, or x field",
        ));
    }
    // y/z must match x length
    if cols.y.len() != cols.x.len() || cols.z.len() != cols.x.len() {
        return Err(invalid_data("Invalid Atoms line: incomplete xyz"));
    }

    match image {
        Some([a, b, c]) => {
            cols.ix.push(a);
            cols.iy.push(b);
            cols.iz.push(c);
        }
        None => {
            // leave unset; pad_defaults fills zeros without marking present
        }
    }

    cols.push_defaults_for_absent_optionals();
    Ok(())
}

// ============================================================================
// Section parsers
// ============================================================================

/// Read the header up to the first section header, which is returned.
///
/// As in LAMMPS `read_data`, every header line is a keyword line; a line that
/// is neither a keyword this reader knows nor a section header, and a keyword
/// whose count does not parse, is refused with an `InvalidData` error naming
/// it. The general-triclinic keywords (`avec`, `bvec`, `cvec`, `abc origin`)
/// are not read, so they are refused rather than leaving the box unset.
fn parse_header_with_first_section<R: BufRead>(
    reader: &mut R,
    skipped: &HashSet<String>,
) -> std::io::Result<(LammpsDataHeader, Option<SectionHeader>)> {
    let mut header = LammpsDataHeader::default();
    let mut line = String::new();

    // Title line. `write_data` ends it with `…, units = <style>`; LAMMPS
    // itself ignores the line, so any other text is accepted as-is.
    reader.read_line(&mut line)?;
    header.units = line.split(',').find_map(|field| {
        let value = field.trim().strip_prefix("units")?.trim_start();
        let style = value.strip_prefix('=')?.split_whitespace().next()?;
        Some(style.to_owned())
    });

    loop {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Ok((header, None));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(section) = SectionHeader::parse(&line, skipped) {
            return Ok((header, Some(section)));
        }
        let tokens = tokenize(trimmed);
        if tokens.is_empty() {
            continue;
        }
        let bad_count = |e: std::num::ParseIntError| {
            invalid_data(format!("Invalid LAMMPS data header line `{trimmed}`: {e}"))
        };
        match tokens.as_slice() {
            [n, "atoms", ..] => header.n_atoms = n.parse().map_err(bad_count)?,
            [n, "bonds", ..] => header.n_bonds = n.parse().map_err(bad_count)?,
            [n, "angles", ..] => header.n_angles = n.parse().map_err(bad_count)?,
            [n, "dihedrals", ..] => header.n_dihedrals = n.parse().map_err(bad_count)?,
            [n, "impropers", ..] => header.n_impropers = n.parse().map_err(bad_count)?,
            // `fix cmap`'s header line, in either of the two spellings it reads.
            [n, "crossterms"] | [n, "cmap", "crossterms"] => {
                header.n_crossterms = n.parse().map_err(bad_count)?
            }
            [
                n,
                kind @ ("atom" | "bond" | "angle" | "dihedral" | "improper"),
                "types",
                ..,
            ] => {
                let n = n.parse().map_err(bad_count)?;
                match *kind {
                    "atom" => header.n_atom_types = n,
                    "bond" => header.n_bond_types = n,
                    "angle" => header.n_angle_types = n,
                    "dihedral" => header.n_dihedral_types = n,
                    _ => header.n_improper_types = n,
                }
            }
            // Counts this reader has no use for: the body sections they size
            // are refused or skipped by name, and the per-atom extras are
            // LAMMPS memory hints.
            [n, "ellipsoids" | "lines" | "triangles" | "bodies", ..]
            | [
                n,
                "extra",
                "bond" | "angle" | "dihedral" | "improper" | "special",
                "per",
                "atom",
                ..,
            ] => {
                n.parse::<usize>().map_err(bad_count)?;
            }
            [lo, hi, "xlo", "xhi", ..] => {
                header.bounds.xlo = lo.parse().map_err(invalid_data)?;
                header.bounds.xhi = hi.parse().map_err(invalid_data)?;
                header.bounds.has_x = true;
            }
            [lo, hi, "ylo", "yhi", ..] => {
                header.bounds.ylo = lo.parse().map_err(invalid_data)?;
                header.bounds.yhi = hi.parse().map_err(invalid_data)?;
                header.bounds.has_y = true;
            }
            [lo, hi, "zlo", "zhi", ..] => {
                header.bounds.zlo = lo.parse().map_err(invalid_data)?;
                header.bounds.zhi = hi.parse().map_err(invalid_data)?;
                header.bounds.has_z = true;
            }
            [xy, xz, yz, "xy", "xz", "yz", ..] => {
                header.bounds.xy = Some(xy.parse().map_err(invalid_data)?);
                header.bounds.xz = Some(xz.parse().map_err(invalid_data)?);
                header.bounds.yz = Some(yz.parse().map_err(invalid_data)?);
            }
            _ => {
                let name = SectionHeader::name_of(trimmed);
                return Err(invalid_data(format!(
                    "LAMMPS data header line `{trimmed}` is neither a header keyword \
                     this reader reads nor a section; if it opens a section, {}",
                    skip_hint(&name)
                )));
            }
        }
    }
}

/// The `read_data` section headers this reader knows by name (docs.lammps.org/
/// read_data.html). Every `* Coeffs` header is one too; see [`SectionHeader`].
const SECTION_NAMES: &[&str] = &[
    "Atoms",
    "Velocities",
    "Masses",
    "Ellipsoids",
    "Lines",
    "Triangles",
    "Bodies",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
    // `fix cmap`'s section (`read_data … fix cmap crossterm CMAP`).
    "CMAP",
    "Atom Type Labels",
    "Bond Type Labels",
    "Angle Type Labels",
    "Dihedral Type Labels",
    "Improper Type Labels",
    "Charges",
];

/// A section header line and the section name it opens.
struct SectionHeader {
    /// The header's words before any `# style` comment, single-spaced.
    name: String,
    /// The header line as read, `# style` comment included.
    line: String,
}

impl SectionHeader {
    /// The header `line` opens, or `None` when it is not a section header.
    ///
    /// A header is a line whose name is in the closed `read_data` vocabulary,
    /// ends in `Coeffs`, or is one the caller asked to skip. A label-led row
    /// such as `CA 12.011` inside `Masses` or a `Coeffs` section is a row of
    /// that section, not a header.
    fn parse(line: &str, skipped: &HashSet<String>) -> Option<Self> {
        let header = Self {
            name: Self::name_of(line),
            line: line.to_string(),
        };
        let known = SECTION_NAMES.contains(&header.name.as_str());
        (known || header.is_coeffs() || skipped.contains(&header.name)).then_some(header)
    }

    /// Whether this is a `* Coeffs` section (`Pair Coeffs`, `PairIJ Coeffs`,
    /// the class2 cross terms, …).
    fn is_coeffs(&self) -> bool {
        self.name.ends_with(" Coeffs")
    }

    /// The section name `line` would open: its words before any `#`
    /// comment, single-spaced.
    fn name_of(line: &str) -> String {
        tokenize(line).join(" ")
    }
}

/// How to get past the section `name` that this reader does not read.
fn skip_hint(name: &str) -> String {
    format!("skip it with LammpsDataReader::with_skipped_section(\"{name}\")")
}

/// The refusal of `row`, found inside `section`, that is not a row of it. It is
/// most likely the header of a section this reader does not read.
fn not_a_row(section: &str, row: &str) -> std::io::Error {
    let name = SectionHeader::name_of(row);
    invalid_data(format!(
        "LAMMPS data line `{name}` in {section} is neither a {section} row nor a \
         section this reader reads; if it opens a section, {}",
        skip_hint(&name)
    ))
}

/// Read the body of the current section up to the next section header (which
/// is returned) or EOF, handing every non-blank, non-comment line to `row`.
fn for_each_row<R: BufRead>(
    reader: &mut R,
    skipped: &HashSet<String>,
    mut row: impl FnMut(&str) -> std::io::Result<()>,
) -> std::io::Result<Option<SectionHeader>> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(header) = SectionHeader::parse(&line, skipped) {
            return Ok(Some(header));
        }
        row(trimmed)?;
    }
}

/// A `* Type Labels` section: `type-id label` per row → map type-id → label.
/// A row that does not start with a type id is refused as a section this
/// reader does not read.
fn parse_type_labels<R: BufRead>(
    reader: &mut R,
    section: &str,
    skipped: &HashSet<String>,
) -> std::io::Result<(HashMap<String, String>, Option<SectionHeader>)> {
    let mut labels = HashMap::new();
    let next = for_each_row(reader, skipped, |row| {
        let tokens = tokenize(row);
        if tokens.first().is_none_or(|t| t.parse::<I>().is_err()) {
            return Err(not_a_row(section, row));
        }
        if tokens.len() < 2 {
            return Err(invalid_data(format!(
                "Invalid {section} line `{row}`: expected `type-id label`"
            )));
        }
        labels.insert(tokens[0].to_string(), tokens[1].to_string());
        Ok(())
    })?;
    Ok((labels, next))
}

/// Masses section: `type mass` per row → map type-id → mass. `type` is a
/// numeric type id or a label from the `Atom Type Labels` read before it.
fn parse_masses<R: BufRead>(
    reader: &mut R,
    atom_type_labels: &HashMap<String, String>,
    skipped: &HashSet<String>,
) -> std::io::Result<(HashMap<I, F>, Option<SectionHeader>)> {
    let label_to_id = invert_type_labels(atom_type_labels);
    let mut masses = HashMap::new();
    let next = for_each_row(reader, skipped, |row| {
        let tokens = tokenize(row);
        if tokens.len() < 2 {
            return Err(invalid_data(format!(
                "Invalid Masses line `{row}`: expected `type mass`"
            )));
        }
        let type_id = match tokens[0].parse::<I>() {
            Ok(tid) => tid,
            Err(_) => *label_to_id.get(tokens[0]).ok_or_else(|| {
                invalid_data(format!(
                    "Invalid Masses line `{row}`: `{}` is neither a type id nor an \
                     Atom Type Labels label",
                    tokens[0]
                ))
            })?,
        };
        let mass = tokens[1]
            .parse::<F>()
            .map_err(|e| invalid_data(format!("Invalid Masses line `{row}`: {e}")))?;
        masses.insert(type_id, mass);
        Ok(())
    })?;
    Ok((masses, next))
}

fn parse_atoms_streamed<R: BufRead>(
    reader: &mut R,
    n_atoms: usize,
    style_hint: Option<&str>,
) -> std::io::Result<AtomColumns> {
    let mut cols = AtomColumns::with_capacity(n_atoms);
    let mut line = String::new();
    let known = style_hint.and_then(layout_for_atom_style);
    let style_known = known.is_some();

    while cols.len() < n_atoms {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let tokens = tokenize(trimmed);
        let layout = resolve_layout(&tokens, known, style_known)?;
        push_atom_line(&mut cols, &tokens, layout)?;
    }
    if cols.len() > 0 && cols.len() < n_atoms {
        return Err(invalid_data(format!(
            "Atoms section has {} rows, the header declares {n_atoms} atoms",
            cols.len()
        )));
    }
    Ok(cols)
}

fn parse_topology_section<R: BufRead>(
    reader: &mut R,
    count: usize,
    n_members: usize,
    section: &str,
) -> std::io::Result<Vec<TopologyTerm>> {
    let min_cols = 2 + n_members;
    let mut terms = Vec::with_capacity(count);
    let mut line = String::new();
    while terms.len() < count {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let tokens = tokenize(trimmed);
        if tokens.len() < min_cols {
            return Err(invalid_data(format!(
                "Invalid {section} line: expected {min_cols} columns, got {}",
                tokens.len()
            )));
        }
        let mut members = [0 as I; 5];
        for (i, slot) in members.iter_mut().enumerate().take(n_members) {
            *slot = parse_i(tokens[2 + i])?;
        }
        terms.push(TopologyTerm {
            type_ref: TypeRef::parse(tokens[1]),
            members,
            n_members: n_members as u8,
        });
    }
    if terms.len() < count {
        return Err(invalid_data(format!(
            "{section} section has {} rows, the header declares {count}",
            terms.len()
        )));
    }
    Ok(terms)
}

/// Rows of a per-atom section (`Velocities`, `Charges`): `id v1 … vK` once for
/// every atom, in any order. Returns the K values aligned to `ids` (the atom
/// ids in row order) and the next section header.
///
/// An id that is not an atom, an id given twice, and fewer rows than atoms are
/// each refused with an `InvalidData` error naming `section`.
fn parse_per_atom_rows<const K: usize, R: BufRead>(
    reader: &mut R,
    ids: &[I],
    section: &str,
    skipped: &HashSet<String>,
) -> std::io::Result<(Vec<[F; K]>, Option<SectionHeader>)> {
    let id_to_idx: HashMap<I, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut values = vec![[0.0 as F; K]; ids.len()];
    let mut seen = vec![false; ids.len()];
    let mut filled = 0usize;
    let next = for_each_row(reader, skipped, |row| {
        let invalid = |why: String| invalid_data(format!("Invalid {section} line `{row}`: {why}"));
        let tokens = tokenize(row);
        if tokens.len() < K + 1 {
            return Err(invalid(format!(
                "expected {} columns, got {}",
                K + 1,
                tokens.len()
            )));
        }
        let id = tokens[0].parse::<I>().map_err(|e| invalid(e.to_string()))?;
        let &idx = id_to_idx
            .get(&id)
            .ok_or_else(|| invalid(format!("atom ID {id} is not in Atoms")))?;
        if std::mem::replace(&mut seen[idx], true) {
            return Err(invalid(format!("atom ID {id} appears twice")));
        }
        for (slot, tok) in values[idx].iter_mut().zip(&tokens[1..]) {
            *slot = tok.parse::<F>().map_err(|e| invalid(e.to_string()))?;
        }
        filled += 1;
        Ok(())
    })?;
    if filled < ids.len() {
        return Err(invalid_data(format!(
            "{section} section has {filled} rows for {} atoms",
            ids.len()
        )));
    }
    Ok((values, next))
}

// ============================================================================
// Frame assembly
// ============================================================================

fn insert_topology_block(
    frame: &mut Frame,
    block_name: &str,
    kind: &str,
    terms: &[TopologyTerm],
    atom_keys: &[&str],
    atom_id_map: &HashMap<I, Idx>,
    label_to_id: &HashMap<String, I>,
) -> std::io::Result<()> {
    if terms.is_empty() {
        return Ok(());
    }
    let n = terms.len();
    let n_members = atom_keys.len();
    let mut member_cols: Vec<Vec<Idx>> = (0..n_members).map(|_| Vec::with_capacity(n)).collect();
    let mut types = Vec::with_capacity(n);

    for term in terms {
        for (i, &atom_id) in term.members().iter().enumerate() {
            let idx = atom_id_map.get(&atom_id).copied().ok_or_else(|| {
                invalid_data(format!("{kind} references unknown atom ID: {atom_id}"))
            })?;
            member_cols[i].push(idx);
        }
        types.push(term.type_ref.resolve(label_to_id));
    }

    let mut block = Block::new();
    for (key, col) in atom_keys.iter().zip(member_cols) {
        insert_column_of_length(&mut block, key, col, n)?;
    }
    insert_column_of_length(
        &mut block,
        keys::TYPE_ID,
        types.iter().map(|&v| v as Idx).collect(),
        n,
    )?;
    frame.insert(block_name, block);
    Ok(())
}

/// Add the string `type` column of `block` from its numeric `type_id`: the
/// label `labels` (id → label, as the `* Type Labels` section declared it)
/// gives that id, else the id itself.
fn insert_type_labels(block: &mut Block, labels: &HashMap<String, String>) -> std::io::Result<()> {
    let Some(ids) = block.get(keys::TYPE_ID).and_then(|c| c.as_uint()) else {
        return Ok(());
    };
    let by_id: HashMap<Idx, &str> = labels
        .iter()
        .filter_map(|(id, label)| Some((id.trim().parse::<Idx>().ok()?, label.as_str())))
        .collect();
    let names: Vec<String> = ids
        .iter()
        .map(|id| {
            by_id
                .get(id)
                .map_or_else(|| id.to_string(), |l| (*l).to_owned())
        })
        .collect();
    let n = names.len();
    insert_column_of_length(block, keys::TYPE, names, n)
}

struct ParsedDataFile {
    header: LammpsDataHeader,
    atoms: AtomColumns,
    bonds: Vec<TopologyTerm>,
    angles: Vec<TopologyTerm>,
    dihedrals: Vec<TopologyTerm>,
    impropers: Vec<TopologyTerm>,
    cmaps: Vec<TopologyTerm>,
    type_masses: HashMap<I, F>,
    atom_type_labels: HashMap<String, String>,
    bond_type_labels: HashMap<String, String>,
    angle_type_labels: HashMap<String, String>,
    dihedral_type_labels: HashMap<String, String>,
    improper_type_labels: HashMap<String, String>,
    /// Captured `* Coeffs` section bodies for force-field I/O (no style lines).
    coeffs_text: String,
}

fn build_frame(mut parsed: ParsedDataFile) -> std::io::Result<Frame> {
    let mut frame = Frame::new();

    // Apply per-type Masses when no per-atom mass column was set.
    if !parsed.atoms.mass.present && !parsed.type_masses.is_empty() {
        let label_to_id = invert_type_labels(&parsed.atom_type_labels);
        let n = parsed.atoms.len();
        parsed.atoms.mass.values.resize(n, 0.0);
        for (i, tref) in parsed.atoms.type_refs.iter().enumerate() {
            let tid = tref.resolve(&label_to_id);
            parsed.atoms.mass.values[i] = parsed.type_masses.get(&tid).copied().unwrap_or(0.0);
        }
        parsed.atoms.mass.present = true;
    }

    // atom id → row index for topology remapping
    let atom_id_map: HashMap<I, Idx> = parsed
        .atoms
        .id
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as Idx))
        .collect();

    if parsed.atoms.len() > 0 {
        let atom_block = parsed.atoms.into_block(&parsed.atom_type_labels)?;
        frame.insert("atoms", atom_block);

        insert_topology_block(
            &mut frame,
            "bonds",
            "Bond",
            &parsed.bonds,
            &[keys::ATOMI, keys::ATOMJ],
            &atom_id_map,
            &invert_type_labels(&parsed.bond_type_labels),
        )?;
        insert_topology_block(
            &mut frame,
            "angles",
            "Angle",
            &parsed.angles,
            &[keys::ATOMI, keys::ATOMJ, keys::ATOMK],
            &atom_id_map,
            &invert_type_labels(&parsed.angle_type_labels),
        )?;
        insert_topology_block(
            &mut frame,
            "dihedrals",
            "Dihedral",
            &parsed.dihedrals,
            &[keys::ATOMI, keys::ATOMJ, keys::ATOMK, keys::ATOML],
            &atom_id_map,
            &invert_type_labels(&parsed.dihedral_type_labels),
        )?;
        insert_topology_block(
            &mut frame,
            "impropers",
            "Improper",
            &parsed.impropers,
            &[keys::ATOMI, keys::ATOMJ, keys::ATOMK, keys::ATOML],
            &atom_id_map,
            &invert_type_labels(&parsed.improper_type_labels),
        )?;
        // A CMAP type is an index into the `fix cmap` file; LAMMPS has no
        // labels for it.
        insert_topology_block(
            &mut frame,
            "cmaps",
            "CMAP",
            &parsed.cmaps,
            &[
                keys::ATOMI,
                keys::ATOMJ,
                keys::ATOMK,
                keys::ATOML,
                keys::ATOMM,
            ],
            &atom_id_map,
            &HashMap::new(),
        )?;

        // Every typed block also carries its string `type`: the file's type
        // label, or the numeric id spelled as a label when the file declares
        // none (the name a force field read from the same file's `* Coeffs`
        // gives that type).
        for (block_name, labels) in [
            ("atoms", &parsed.atom_type_labels),
            ("bonds", &parsed.bond_type_labels),
            ("angles", &parsed.angle_type_labels),
            ("dihedrals", &parsed.dihedral_type_labels),
            ("impropers", &parsed.improper_type_labels),
        ] {
            if let Some(block) = frame.get_mut(block_name) {
                insert_type_labels(block, labels)?;
            }
        }
    }

    let pbc: Pbc3 = [true, true, true];
    if let Some(sb) = simbox_from_bounds(&parsed.header.bounds, pbc)? {
        frame.simbox = Some(sb);
    }

    for (key, labels) in [
        (keys::ATOM_TYPE_LABELS, &parsed.atom_type_labels),
        (keys::BOND_TYPE_LABELS, &parsed.bond_type_labels),
        (keys::ANGLE_TYPE_LABELS, &parsed.angle_type_labels),
        (keys::DIHEDRAL_TYPE_LABELS, &parsed.dihedral_type_labels),
        (keys::IMPROPER_TYPE_LABELS, &parsed.improper_type_labels),
    ] {
        if let Some(s) = labels_to_meta(labels) {
            frame.meta.insert(key.to_string(), s);
        }
    }

    // Header counts (may exceed body rows when types are unused).
    let h = &parsed.header;
    frame.meta.insert(
        "lammps_counts".to_string(),
        format!(
            "atoms={},bonds={},angles={},dihedrals={},impropers={},crossterms={},\
             atom_types={},bond_types={},angle_types={},dihedral_types={},\
             improper_types={}",
            h.n_atoms,
            h.n_bonds,
            h.n_angles,
            h.n_dihedrals,
            h.n_impropers,
            h.n_crossterms,
            h.n_atom_types,
            h.n_bond_types,
            h.n_angle_types,
            h.n_dihedral_types,
            h.n_improper_types,
        ),
    );
    // Unit style from the `write_data` title line; absent when not stated.
    if let Some(units) = &h.units {
        frame.meta.insert(LAMMPS_UNITS.to_string(), units.clone());
    }
    // Which box axes appeared in the header (zero-volume boxes still set has_*).
    frame.meta.insert(
        "lammps_box_axes".to_string(),
        format!(
            "x={},y={},z={}",
            h.bounds.has_x as u8, h.bounds.has_y as u8, h.bounds.has_z as u8
        ),
    );

    // Force-field coefficient sections (Pair/Bond/… Coeffs), read into a force
    // field by `LammpsForcefieldReader::read_data_coeffs(&frame, …)`.
    if !parsed.coeffs_text.is_empty() {
        frame
            .meta
            .insert(LAMMPS_COEFFS_TEXT.to_string(), parsed.coeffs_text);
    }

    Ok(frame)
}

// ============================================================================
// Section dispatch
// ============================================================================

/// Parse the section `header` opens; returns the next section header when the
/// section reader consumed it, `None` when it stopped at a row count.
fn dispatch_section<R: BufRead>(
    header: &SectionHeader,
    reader: &mut R,
    parsed: &mut ParsedDataFile,
    skipped: &HashSet<String>,
    atom_style: Option<&str>,
) -> std::io::Result<Option<SectionHeader>> {
    let name = header.name.as_str();
    if skipped.contains(name) {
        return for_each_row(reader, skipped, |_| Ok(()));
    }
    match name {
        "Atom Type Labels" => {
            let (labels, next) = parse_type_labels(reader, name, skipped)?;
            parsed.atom_type_labels = labels;
            Ok(next)
        }
        "Bond Type Labels" => {
            let (labels, next) = parse_type_labels(reader, name, skipped)?;
            parsed.bond_type_labels = labels;
            Ok(next)
        }
        "Angle Type Labels" => {
            let (labels, next) = parse_type_labels(reader, name, skipped)?;
            parsed.angle_type_labels = labels;
            Ok(next)
        }
        "Dihedral Type Labels" => {
            let (labels, next) = parse_type_labels(reader, name, skipped)?;
            parsed.dihedral_type_labels = labels;
            Ok(next)
        }
        "Improper Type Labels" => {
            let (labels, next) = parse_type_labels(reader, name, skipped)?;
            parsed.improper_type_labels = labels;
            Ok(next)
        }
        "Masses" => {
            let (masses, next) = parse_masses(reader, &parsed.atom_type_labels, skipped)?;
            parsed.type_masses = masses;
            Ok(next)
        }
        "Atoms" => {
            // The caller's atom style is LAMMPS's own `atom_style` command,
            // which governs `read_data`; the section's `# style` comment is
            // only a hint for when no style was given.
            let hint = match atom_style {
                Some(style) => Some(style.to_owned()),
                None => parse_atoms_style_hint(header.line.trim()),
            };
            parsed.atoms = parse_atoms_streamed(reader, parsed.header.n_atoms, hint.as_deref())?;
            Ok(None)
        }
        "Velocities" => {
            let (rows, next) =
                parse_per_atom_rows::<3, _>(reader, &parsed.atoms.id, name, skipped)?;
            let atoms = &mut parsed.atoms;
            for (k, col) in [&mut atoms.vx, &mut atoms.vy, &mut atoms.vz]
                .into_iter()
                .enumerate()
            {
                col.values = rows.iter().map(|r| r[k]).collect();
                col.present = true;
            }
            Ok(next)
        }
        "Charges" => {
            let (rows, next) =
                parse_per_atom_rows::<1, _>(reader, &parsed.atoms.id, name, skipped)?;
            parsed.atoms.charge.values = rows.iter().map(|[q]| *q).collect();
            parsed.atoms.charge.present = true;
            Ok(next)
        }
        "Bonds" => {
            parsed.bonds = parse_topology_section(reader, parsed.header.n_bonds, 2, "Bonds")?;
            Ok(None)
        }
        "Angles" => {
            parsed.angles = parse_topology_section(reader, parsed.header.n_angles, 3, "Angles")?;
            Ok(None)
        }
        "Dihedrals" => {
            parsed.dihedrals =
                parse_topology_section(reader, parsed.header.n_dihedrals, 4, "Dihedrals")?;
            Ok(None)
        }
        "Impropers" => {
            parsed.impropers =
                parse_topology_section(reader, parsed.header.n_impropers, 4, "Impropers")?;
            Ok(None)
        }
        "CMAP" => {
            if parsed.header.n_crossterms == 0 {
                return Err(invalid_data(
                    "LAMMPS data section `CMAP` without a `N crossterms` header line",
                ));
            }
            parsed.cmaps = parse_topology_section(reader, parsed.header.n_crossterms, 5, "CMAP")?;
            Ok(None)
        }
        // Force-field coefficient blocks — the header line (`# style` comment
        // included) and its rows, captured verbatim for the FF reader. A row
        // must start with a type id or a declared type label; anything else is
        // the header of a section this reader does not read.
        _ if header.is_coeffs() => {
            let declared: HashSet<&str> = [
                &parsed.atom_type_labels,
                &parsed.bond_type_labels,
                &parsed.angle_type_labels,
                &parsed.dihedral_type_labels,
                &parsed.improper_type_labels,
            ]
            .into_iter()
            .flat_map(|labels| labels.values().map(String::as_str))
            .collect();
            let text = &mut parsed.coeffs_text;
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(header.line.trim());
            text.push('\n');
            for_each_row(reader, skipped, |row| {
                let first = tokenize(row).first().copied().unwrap_or(row);
                if first.parse::<I>().is_err() && !declared.contains(first) {
                    return Err(not_a_row(name, row));
                }
                text.push_str(row);
                text.push('\n');
                Ok(())
            })
        }
        _ => Err(invalid_data(format!(
            "LAMMPS data section `{name}` is not read by this reader; {}",
            skip_hint(name)
        ))),
    }
}

// ============================================================================
// Reader
// ============================================================================

/// Reads one LAMMPS data file into one [`Frame`].
///
/// No section is lost silently. The sections this reader parses are the five
/// `* Type Labels` sections, `Masses`, `Atoms`, `Velocities`, `Charges`,
/// `Bonds`, `Angles`, `Dihedrals`, `Impropers` and `fix cmap`'s `CMAP`
/// (`index type a1 … a5` per row, sized by the `N crossterms` header line,
/// into a `cmaps` block of `atomi` … `atomm` and the numeric `type_id`, the
/// map's index in the `fix cmap` file); every `* Coeffs` section
/// (`PairIJ` and the class2 cross terms included) is kept verbatim in the
/// frame's [`LAMMPS_COEFFS_TEXT`] meta. Any other section — `Ellipsoids`,
/// `Lines`, `Triangles`, `Bodies`, or a fix-defined one such as `BiTorsions` — is
/// refused with an `InvalidData` error naming it, unless it was named in
/// [`with_skipped_section`], in which case its body is read past and
/// discarded. A line inside a `* Coeffs` or `* Type Labels` section that does
/// not start like a row of it (a type id, or for `Coeffs` a declared type
/// label) is refused the same way.
///
/// A parsed section given twice is refused; `* Coeffs` sections accumulate.
/// A header line that is neither a `read_data` keyword this reader reads nor
/// a section, or whose count does not parse, is refused, as LAMMPS does.
///
/// `Velocities` and `Charges` must give exactly one row per atom: an unknown
/// atom id, a repeated id, or a missing atom is refused. `Charges` overrides
/// any charge the atom style supplied.
///
/// [`with_skipped_section`]: LammpsDataReader::with_skipped_section
///
/// Specs: <https://docs.lammps.org/read_data.html>,
/// <https://docs.lammps.org/atom_style.html>. Atoms are streamed straight into
/// typed column buffers (no intermediate per-atom struct), which cuts peak
/// memory on large systems.
pub struct LammpsDataReader<R: BufRead + Seek> {
    reader: R,
    frame: OnceLock<Option<Frame>>,
    returned: bool,
    skipped: HashSet<String>,
    atom_style: Option<String>,
}

impl<R: BufRead + Seek> LammpsDataReader<R> {
    /// A reader that skips no section.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            frame: OnceLock::new(),
            returned: false,
            skipped: HashSet::new(),
            atom_style: None,
        }
    }

    /// Read the `Atoms` section in `style`'s column layout (`"full"`,
    /// `"atomic"`, `"charge"`, …; accelerator suffixes such as `/kk` are
    /// dropped), as LAMMPS reads a data file under the input script's
    /// `atom_style`. Without it the section's `# style` comment decides, and
    /// without that the column count.
    ///
    /// # Errors
    ///
    /// `InvalidInput` when `style` is not an atom style this reader knows.
    pub fn with_atom_style(mut self, style: &str) -> std::io::Result<Self> {
        let style = normalize_atom_style(style);
        if layout_for_atom_style(&style).is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unknown LAMMPS atom style {style:?}"),
            ));
        }
        self.atom_style = Some(style);
        Ok(self)
    }

    /// Read past the section `header` instead of refusing it, discarding its
    /// body. `header` is the section name without its `# style` comment
    /// (`"Ellipsoids"`, `"Bodies"`, or a fix-defined `"BiTorsions"` the `read_data`
    /// vocabulary does not name — a line reading `header` then opens a
    /// section); it is case-sensitive, as in LAMMPS.
    pub fn with_skipped_section(mut self, header: &str) -> Self {
        self.skipped.insert(SectionHeader::name_of(header));
        self
    }

    fn parse_file(&mut self) -> std::io::Result<Option<Frame>> {
        self.reader.seek(SeekFrom::Start(0))?;
        let (header, first) = parse_header_with_first_section(&mut self.reader, &self.skipped)?;
        let mut parsed = ParsedDataFile {
            header,
            atoms: AtomColumns::with_capacity(0),
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
            impropers: Vec::new(),
            cmaps: Vec::new(),
            type_masses: HashMap::new(),
            atom_type_labels: HashMap::new(),
            bond_type_labels: HashMap::new(),
            angle_type_labels: HashMap::new(),
            dihedral_type_labels: HashMap::new(),
            improper_type_labels: HashMap::new(),
            coeffs_text: String::new(),
        };

        let mut pending = first;
        // Parsed sections read so far: a second one would replace the first.
        let mut read_sections: HashSet<String> = HashSet::new();
        let mut line = String::new();
        loop {
            while let Some(section) = pending.take() {
                let is_parsed = !section.is_coeffs() && !self.skipped.contains(&section.name);
                if is_parsed && !read_sections.insert(section.name.clone()) {
                    return Err(invalid_data(format!(
                        "LAMMPS data section `{}` appears twice",
                        section.name
                    )));
                }
                pending = dispatch_section(
                    &section,
                    &mut self.reader,
                    &mut parsed,
                    &self.skipped,
                    self.atom_style.as_deref(),
                )?;
            }
            // A section that stops at its row count leaves the reader between
            // sections: scan to the next header.
            line.clear();
            if self.reader.read_line(&mut line)? == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let Some(section) = SectionHeader::parse(&line, &self.skipped) else {
                return Err(invalid_data(format!(
                    "LAMMPS data line `{trimmed}` is outside any section; if it opens \
                     a section this reader does not read, {}",
                    skip_hint(&SectionHeader::name_of(trimmed))
                )));
            };
            pending = Some(section);
        }

        if parsed.atoms.len() == 0 && parsed.header.n_atoms > 0 {
            return Err(invalid_data("No atoms found in file"));
        }
        Ok(Some(build_frame(parsed)?))
    }
}

impl<R: BufRead + Seek> Reader for LammpsDataReader<R> {
    type R = R;
    fn new(reader: R) -> Self {
        Self::new(reader)
    }
}

impl<R: BufRead + Seek> FrameReader for LammpsDataReader<R> {
    fn read(&mut self) -> std::io::Result<Option<Frame>> {
        if self.returned {
            return Ok(None);
        }
        if self.frame.get().is_none() {
            let frame = self.parse_file()?;
            let _ = self.frame.set(frame);
        }
        self.returned = true;
        // Validate on the way out: a frame that violates the vocabulary is a
        // malformed file or a reader bug, not a result to return.
        crate::io::reader::check_read_frame(self.frame.get().unwrap().clone())
    }
}

// ============================================================================
// Writer
// ============================================================================

pub struct LammpsDataWriter<W: Write> {
    writer: W,
    options: LammpsDataWriteOptions,
}

impl<W: Write> crate::io::writer::Writer for LammpsDataWriter<W> {
    type W = W;
    fn new(writer: W) -> Self {
        Self::new(writer)
    }
}

impl<W: Write> LammpsDataWriter<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            options: LammpsDataWriteOptions::default(),
        }
    }

    /// Write in the `atom_style` and `units` `options` name.
    pub fn with_options(mut self, options: LammpsDataWriteOptions) -> Self {
        self.options = options;
        self
    }
}

impl<W: Write> FrameWriter for LammpsDataWriter<W> {
    fn write(&mut self, frame: &Frame) -> std::io::Result<()> {
        // Refuse to emit a frame that violates the vocabulary: a bad file
        // looks fine and is found wrong later, by whatever reads it.
        crate::io::writer::check_write_frame(frame)?;
        write_lammps_data_frame_with(&mut self.writer, frame, &self.options, &HashMap::new())
    }
}

fn write_type_label_section<W: Write>(
    writer: &mut W,
    section: &str,
    labels: Option<&[String]>,
) -> std::io::Result<()> {
    let Some(labs) = labels else {
        return Ok(());
    };
    if labs.is_empty() {
        return Ok(());
    }
    writeln!(writer, "{section}")?;
    writeln!(writer)?;
    for (i, lab) in labs.iter().enumerate() {
        writeln!(writer, "{} {}", i + 1, lab)?;
    }
    writeln!(writer)?;
    Ok(())
}

/// Per-row atom IDs: existing ``id`` column, else 1..N (file artifact).
fn resolve_atom_ids(frame: &impl FrameAccess, n: usize) -> Vec<Idx> {
    if let Some(col) = frame.column("atoms", keys::ID).and_then(|c| c.as_uint()) {
        return (0..n).map(|i| col[[i]]).collect();
    }
    if let Some(col) = frame.column("atoms", keys::ID).and_then(|c| c.as_int()) {
        return (0..n).map(|i| col[[i]] as Idx).collect();
    }
    (1..=n as Idx).collect()
}

/// Per-row masses: the stored `mass` column when there is one, else the
/// element's periodic-table mass, else 1.0 (an unknown symbol such as a
/// Drude shell's `"D"`).
fn resolve_row_masses(frame: &impl FrameAccess, n: usize) -> Vec<F> {
    // A stated mass wins: a Drude core is lighter than its element by the
    // shell's mass, a united atom heavier.
    if let Some(col) = frame.column("atoms", keys::MASS).and_then(|c| c.as_float()) {
        return (0..n).map(|i| col[[i]]).collect();
    }
    let mut masses: Vec<F> = vec![1.0; n];
    if let Some(el) = frame
        .column("atoms", keys::ELEMENT)
        .and_then(|c| c.as_string())
    {
        // One periodic-table lookup per distinct symbol, not per row.
        let mut memo: HashMap<&str, Option<F>> = HashMap::new();
        for (i, mass) in masses.iter_mut().enumerate() {
            let sym = el[[i]].as_str();
            let known = *memo.entry(sym).or_insert_with(|| {
                crate::core::Element::by_symbol(sym).map(|e| F::from(e.atomic_mass()))
            });
            if let Some(m) = known {
                *mass = m;
            }
        }
    }
    masses
}

/// Does the atoms block have an int or float column for this data-file field?
pub(crate) fn frame_has_atom_field(frame: &impl FrameAccess, field: DataField) -> bool {
    let key = field_column_key(field);
    if field == DataField::Mol {
        return frame
            .column("atoms", keys::MOL_ID)
            .and_then(|c| c.as_uint())
            .is_some();
    }
    // Mass may be resolved from element without a mass column.
    if field == DataField::Mass {
        return frame
            .column("atoms", keys::MASS)
            .and_then(|c| c.as_float())
            .is_some()
            || frame
                .column("atoms", keys::ELEMENT)
                .and_then(|c| c.as_string())
                .is_some();
    }
    // Type is always present when write proceeds (resolved separately).
    if field == DataField::Type {
        return frame
            .column("atoms", keys::TYPE_ID)
            .and_then(|c| c.as_uint())
            .is_some()
            || frame
                .column("atoms", keys::TYPE_ID)
                .and_then(|c| c.as_int())
                .is_some()
            || frame
                .column("atoms", keys::TYPE)
                .and_then(|c| c.as_string())
                .is_some()
            || frame
                .column("atoms", keys::TYPE)
                .and_then(|c| c.as_uint())
                .is_some()
            || frame
                .column("atoms", keys::TYPE)
                .and_then(|c| c.as_int())
                .is_some();
    }
    if field == DataField::Id {
        // Always writable (1..N if missing).
        return true;
    }
    frame
        .column("atoms", key)
        .and_then(|c| c.as_int())
        .is_some()
        || frame
            .column("atoms", key)
            .and_then(|c| c.as_uint())
            .is_some()
        || frame
            .column("atoms", key)
            .and_then(|c| c.as_float())
            .is_some()
}

/// One ``Atoms`` column, resolved once before the row loop.
enum AtomColumn<'a> {
    Uint(ArrayViewD<'a, Idx>),
    Int(ArrayViewD<'a, I>),
    /// A float column and the factor into the file's units.
    Float(ArrayViewD<'a, F>, F),
    /// Already-resolved per-row values (type ids).
    IdxRows(&'a [Idx]),
    /// Already-resolved per-row values (masses), and the factor into the
    /// file's units.
    FloatRows(&'a [F], F),
}

impl<'a> AtomColumn<'a> {
    /// Resolve the column backing one non-``Id`` data-file field.
    fn resolve(
        frame: &'a impl FrameAccess,
        field: DataField,
        type_ids: &'a [Idx],
        row_masses: &'a [F],
        units: &DataUnits,
    ) -> std::io::Result<Self> {
        let key = field_column_key(field);
        let factor = units.field_factor(field)?;
        let col = match field {
            DataField::Type => Self::IdxRows(type_ids),
            DataField::Mass => Self::FloatRows(row_masses, factor),
            DataField::Bodyflag
            | DataField::ShapeFlag
            | DataField::Espin
            | DataField::Status
            | DataField::TemplateIndex
            | DataField::TemplateAtom => match frame.column("atoms", key).and_then(|c| c.as_uint())
            {
                Some(col) => Self::Uint(col),
                None => Self::Int(
                    frame
                        .column("atoms", key)
                        .and_then(|c| c.as_int())
                        .ok_or_else(|| invalid_data(format!("Missing integer column '{key}'")))?,
                ),
            },
            DataField::Mol => Self::Uint(
                frame
                    .column("atoms", keys::MOL_ID)
                    .and_then(|c| c.as_uint())
                    .ok_or_else(|| invalid_data("Missing mol_id column"))?,
            ),
            _ => Self::Float(
                frame
                    .column("atoms", key)
                    .and_then(|c| c.as_float())
                    .ok_or_else(|| invalid_data(format!("Missing float column '{key}'")))?,
                factor,
            ),
        };
        Ok(col)
    }

    fn write_cell<W: Write>(&self, writer: &mut W, i: usize) -> std::io::Result<()> {
        match self {
            Self::Uint(col) => write!(writer, " {}", col[[i]]),
            Self::Int(col) => write!(writer, " {}", col[[i]]),
            Self::Float(col, factor) => write!(writer, " {}", col[[i]] * factor),
            Self::IdxRows(rows) => write!(writer, " {}", rows[i]),
            Self::FloatRows(rows, factor) => write!(writer, " {}", rows[i] * factor),
        }
    }
}

/// ``Atoms`` section: columns and image flags resolved before the row loop.
fn write_atoms_section<W: Write>(
    writer: &mut W,
    frame: &impl FrameAccess,
    style_name: &str,
    atom_ids: &[Idx],
    columns: &[AtomColumn<'_>],
) -> std::io::Result<()> {
    let images = match (
        frame.column("atoms", keys::IX).and_then(|c| c.as_int()),
        frame.column("atoms", keys::IY).and_then(|c| c.as_int()),
        frame.column("atoms", keys::IZ).and_then(|c| c.as_int()),
    ) {
        (Some(ix), Some(iy), Some(iz)) => Some([ix, iy, iz]),
        _ => None,
    };

    writeln!(writer, "Atoms # {style_name}")?;
    writeln!(writer)?;
    for (i, id) in atom_ids.iter().enumerate() {
        write!(writer, "{id}")?;
        for col in columns {
            col.write_cell(writer, i)?;
        }
        if let Some([ix, iy, iz]) = &images {
            write!(writer, " {} {} {}", ix[[i]], iy[[i]], iz[[i]])?;
        }
        writeln!(writer)?;
    }
    writeln!(writer)?;
    Ok(())
}

/// One topology section: the rows of `blocks`, in order, numbered on from 1
/// across them (a `Bonds` section holds the `bonds` rows, then the `drudes`
/// rows). `type_ids` lists every block's rows in that order.
fn write_topology_section<W: Write>(
    writer: &mut W,
    frame: &impl FrameAccess,
    section: &str,
    blocks: &[&str],
    n_members: usize,
    atom_ids: &[Idx],
    type_ids: &[Idx],
) -> std::io::Result<()> {
    let keys_ep = &keys::ENDPOINTS[..n_members];
    let mut parts = Vec::with_capacity(blocks.len());
    for &block in blocks {
        let n = frame
            .visit_block(block, |b| b.n_rows().unwrap_or(0))
            .unwrap_or(0);
        if n == 0 {
            continue;
        }
        let mut cols = Vec::with_capacity(n_members);
        for k in keys_ep {
            cols.push(
                frame
                    .column(block, k)
                    .and_then(|c| c.as_uint())
                    .ok_or_else(|| invalid_data(format!("Missing '{block}.{k}'")))?,
            );
        }
        parts.push((block, n, cols));
    }
    let n: usize = parts.iter().map(|(_, n, _)| n).sum();
    if n == 0 {
        return Ok(());
    }
    if type_ids.len() != n {
        return Err(invalid_data(format!(
            "internal: {blocks:?} type_ids length {} != nrows {n}",
            type_ids.len()
        )));
    }

    writeln!(writer, "{section}")?;
    writeln!(writer)?;
    let mut row = 0;
    for (block, n_rows, cols) in &parts {
        for i in 0..*n_rows {
            write!(writer, "{} {}", row + 1, type_ids[row])?;
            for col in cols {
                let idx = col[[i]] as usize;
                let atom_id = atom_ids.get(idx).ok_or_else(|| {
                    invalid_data(format!(
                        "{block} endpoint index {idx} out of range for {n} atoms",
                        n = atom_ids.len()
                    ))
                })?;
                write!(writer, " {atom_id}")?;
            }
            writeln!(writer)?;
            row += 1;
        }
    }
    writeln!(writer)?;
    Ok(())
}

/// How far a box-less frame's box reaches past its outermost atom on every
/// side, in the frame's length unit.
pub(crate) const BOXLESS_MARGIN: F = 1.0;

/// The box a frame without one (or with a free box, which has no cell) is
/// written in: the axis-aligned bounds of its coordinates widened by
/// [`BOXLESS_MARGIN`] on every side, so every atom is strictly inside and a
/// planar or one-atom system still has a volume.
///
/// A box-less frame is a non-periodic system, and the box is only the
/// envelope `read_data` needs: read the file with `boundary s s s`, which
/// shrink-wraps it to the atoms (`boundary f f f` keeps it fixed, and loses
/// an atom that leaves it). Under `boundary p p p` the margin would be the
/// distance to the next periodic image — a frame meant to be periodic carries
/// its box.
fn enclosing_bounds(coords: &[Vec<F>]) -> std::io::Result<([F; 3], [F; 3])> {
    if coords.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid_data(
            "frame has no box and a non-finite coordinate: a box-less frame is \
             written inside the bounds of its coordinates",
        ));
    }
    let free = SimBox::free_from_xyz(&coords[0], &coords[1], &coords[2], BOXLESS_MARGIN)
        .map_err(|e| invalid_data(format!("the box of a box-less frame: {e:?}")))?;
    let (o, l) = (free.origin_view(), free.lengths());
    Ok(([o[0], o[1], o[2]], [l[0], l[1], l[2]]))
}

/// [`write_lammps_data_frame_with`] with the default options (the tests'
/// door into the writer).
#[cfg(test)]
fn write_lammps_data_frame<W: Write>(
    writer: &mut W,
    frame: &impl FrameAccess,
) -> std::io::Result<()> {
    write_lammps_data_frame_with(
        writer,
        frame,
        &LammpsDataWriteOptions::default(),
        &HashMap::new(),
    )
}

/// Write `frame` as a data file in the `atom_style` and `units` `options`
/// name, with the mass of an atom type no row uses (a declared inventory
/// label) taken from `label_masses` instead of the placeholder `1`.
fn write_lammps_data_frame_with<W: Write>(
    writer: &mut W,
    frame: &impl FrameAccess,
    options: &LammpsDataWriteOptions,
    label_masses: &HashMap<String, F>,
) -> std::io::Result<()> {
    let n_atoms = frame
        .visit_block("atoms", |b| b.n_rows().unwrap_or(0))
        .unwrap_or(0);
    if n_atoms == 0 {
        return Err(invalid_data("Frame has no atoms to write"));
    }
    // Both decisions the input script shares with the file, before a byte
    // of it is written.
    let units = resolve_units(frame, options.units.as_deref())?;
    let (style_name, layout) = resolve_write_style(frame, options.atom_style.as_deref())?;

    // `write_data` ends its title with the unit style; the reader takes it
    // back into `lammps_units`.
    match units.title {
        Some(style) => writeln!(
            writer,
            "# LAMMPS data file generated by molrs, units = {style}"
        )?,
        None => writeln!(writer, "# LAMMPS data file generated by molrs")?,
    }
    writeln!(writer)?;

    // Core coords are required for every write style, and bound a box-less
    // frame's box.
    let mut coords = Vec::with_capacity(3);
    for key in [keys::X, keys::Y, keys::Z] {
        let col = frame
            .column("atoms", key)
            .and_then(|c| c.as_float())
            .ok_or_else(|| invalid_data(format!("Missing '{key}' column")))?;
        coords.push(col.iter().copied().collect::<Vec<F>>());
    }

    // A per-pair 1-4 override has no LAMMPS form: LAMMPS prices a close pair
    // by special_bonds, lj/charmm's epsilon14/sigma14 and dihedral charmm's
    // w, never per pair. Writing the file without it would run a different
    // force field.
    if let Some(present) = frame.visit_block("pairs", |b| {
        crate::core::schema::PAIR_OVERRIDE_COLUMNS
            .iter()
            .copied()
            .filter(|k| b.contains_key(k))
            .collect::<Vec<_>>()
    }) && !present.is_empty()
    {
        return Err(invalid_data(format!(
            "frame['pairs'] carries the per-pair override columns {present:?}, which a \
             LAMMPS data file cannot express (LAMMPS has no per-pair 1-4 exception)"
        )));
    }

    refuse_virtual_sites(frame)?;

    // Bonded sections need a molecular atom style, and every molecular style
    // carries a molecule ID. The writer never invents one: which atoms form a
    // molecule is the caller's call.
    if !frame_has_atom_field(frame, DataField::Mol) {
        for block in [
            "bonds",
            SPRINGS.0,
            "angles",
            "dihedrals",
            "impropers",
            "cmaps",
        ] {
            let n = frame
                .visit_block(block, |b| b.n_rows().unwrap_or(0))
                .unwrap_or(0);
            if n > 0 {
                return Err(invalid_data(format!(
                    "frame['{block}'] has {n} rows but frame['atoms'] has no 'mol_id' \
                     column; a bonded LAMMPS data file needs a molecule ID per atom \
                     (e.g. the bond graph's connected components, \
                     Topology::from_frame(frame).connected_components())"
                )));
            }
        }
    }

    // A Drude spring (`drudes` row) is a LAMMPS bond: one bond-type id space.
    let type_labels = lammps_type_labels(frame).map_err(invalid_data)?;
    let atom_rt = type_labels.block("atoms").ok_or_else(|| {
        invalid_data(
            "frame['atoms'] has neither 'type' nor 'type_id'; \
             assign a 'type' or 'type_id' column before write",
        )
    })?;
    let bond_rt = type_labels.block("bonds");
    let angle_rt = type_labels.block("angles");
    let dihedral_rt = type_labels.block("dihedrals");
    let improper_rt = type_labels.block("impropers");
    let cmap_rt = type_labels.block("cmaps");

    let atom_ids = resolve_atom_ids(frame, n_atoms);
    let row_masses = resolve_row_masses(frame, n_atoms);

    let n_bonds = [SPRINGS.1, SPRINGS.0]
        .iter()
        .map(|block| {
            frame
                .visit_block(block, |b| b.n_rows().unwrap_or(0))
                .unwrap_or(0)
        })
        .sum::<usize>();
    let n_angles = frame
        .visit_block("angles", |b| b.n_rows().unwrap_or(0))
        .unwrap_or(0);
    let n_dihedrals = frame
        .visit_block("dihedrals", |b| b.n_rows().unwrap_or(0))
        .unwrap_or(0);
    let n_impropers = frame
        .visit_block("impropers", |b| b.n_rows().unwrap_or(0))
        .unwrap_or(0);
    let n_crossterms = frame
        .visit_block("cmaps", |b| b.n_rows().unwrap_or(0))
        .unwrap_or(0);

    let n_atom_types = atom_rt.n_types().max(1);
    let n_bond_types = bond_rt.map(|r| r.n_types()).unwrap_or(0);
    let n_angle_types = angle_rt.map(|r| r.n_types()).unwrap_or(0);
    let n_dihedral_types = dihedral_rt.map(|r| r.n_types()).unwrap_or(0);
    let n_improper_types = improper_rt.map(|r| r.n_types()).unwrap_or(0);

    if let Some(flags) = drude_flags(frame, atom_rt.type_ids(), n_atom_types) {
        // LAMMPS skips comment lines in the header; the DRUDE package's
        // `fix drude` wants exactly this list, in atom-type order.
        writeln!(writer, "# fix drude flags (atom-type order): {flags}")?;
    }
    writeln!(writer, "{n_atoms} atoms")?;
    if n_bonds > 0 {
        writeln!(writer, "{n_bonds} bonds")?;
    }
    if n_angles > 0 {
        writeln!(writer, "{n_angles} angles")?;
    }
    if n_dihedrals > 0 {
        writeln!(writer, "{n_dihedrals} dihedrals")?;
    }
    if n_impropers > 0 {
        writeln!(writer, "{n_impropers} impropers")?;
    }
    // `fix cmap`'s header line; the file is read with
    // `read_data <file> fix <id> crossterm CMAP`.
    if n_crossterms > 0 {
        writeln!(writer, "{n_crossterms} crossterms")?;
    }
    writeln!(writer, "{n_atom_types} atom types")?;
    if n_bond_types > 0 {
        writeln!(writer, "{n_bond_types} bond types")?;
    }
    if n_angle_types > 0 {
        writeln!(writer, "{n_angle_types} angle types")?;
    }
    if n_dihedral_types > 0 {
        writeln!(writer, "{n_dihedral_types} dihedral types")?;
    }
    if n_improper_types > 0 {
        writeln!(writer, "{n_improper_types} improper types")?;
    }
    writeln!(writer)?;

    // A free box (no cell, `Box()`) is no box: its identity placeholder is not
    // a region of space.
    let simbox = frame.simbox_ref().filter(|sb| sb.is_cell_defined());
    let (box_origin, box_lengths, tilts) = if let Some(sb) = simbox {
        let o = sb.origin_view();
        let l = sb.lengths();
        let t = sb.tilts();
        ([o[0], o[1], o[2]], [l[0], l[1], l[2]], [t[0], t[1], t[2]])
    } else {
        let (origin, lengths) = enclosing_bounds(&coords)?;
        (origin, lengths, [0.0; 3])
    };
    let box_origin = box_origin.map(|v| v * units.length);
    let box_lengths = box_lengths.map(|v| v * units.length);
    let tilts = tilts.map(|v| v * units.length);
    writeln!(
        writer,
        "{} {} xlo xhi",
        box_origin[0],
        box_origin[0] + box_lengths[0]
    )?;
    writeln!(
        writer,
        "{} {} ylo yhi",
        box_origin[1],
        box_origin[1] + box_lengths[1]
    )?;
    writeln!(
        writer,
        "{} {} zlo zhi",
        box_origin[2],
        box_origin[2] + box_lengths[2]
    )?;
    if tilts.iter().any(|&t| t != 0.0) {
        writeln!(writer, "{} {} {} xy xz yz", tilts[0], tilts[1], tilts[2])?;
    }
    writeln!(writer)?;

    write_type_label_section(writer, "Atom Type Labels", atom_rt.labels())?;
    write_type_label_section(writer, "Bond Type Labels", bond_rt.and_then(|r| r.labels()))?;
    write_type_label_section(
        writer,
        "Angle Type Labels",
        angle_rt.and_then(|r| r.labels()),
    )?;
    write_type_label_section(
        writer,
        "Dihedral Type Labels",
        dihedral_rt.and_then(|r| r.labels()),
    )?;
    write_type_label_section(
        writer,
        "Improper Type Labels",
        improper_rt.and_then(|r| r.labels()),
    )?;

    // Masses: always emit for non-body styles. First-seen mass per type;
    // unused inventory slots get mass 1.0.
    let body_style = style_name == "body";
    if !body_style {
        writeln!(writer, "Masses")?;
        writeln!(writer)?;
        let mut type_mass = vec![1.0_f64; n_atom_types + 1];
        let mut seen = vec![false; n_atom_types + 1];
        for (i, &mass) in row_masses.iter().enumerate().take(n_atoms) {
            let t = atom_rt.type_ids()[i] as usize;
            if t > 0 && t <= n_atom_types && !seen[t] {
                seen[t] = true;
                type_mass[t] = mass;
            }
        }
        if let Some(labels) = atom_rt.labels() {
            for (k, label) in labels.iter().enumerate() {
                let t = k + 1;
                if t <= n_atom_types
                    && !seen[t]
                    && let Some(&m) = label_masses.get(label)
                {
                    type_mass[t] = m;
                }
            }
        }
        for (t, m) in type_mass.iter().enumerate().take(n_atom_types + 1).skip(1) {
            writeln!(writer, "{t} {}", m * units.mass)?;
        }
        writeln!(writer)?;
    }

    let columns = layout
        .fields
        .iter()
        .filter(|&&f| f != DataField::Id)
        .map(|&f| AtomColumn::resolve(frame, f, atom_rt.type_ids(), &row_masses, &units))
        .collect::<std::io::Result<Vec<_>>>()?;
    write_atoms_section(writer, frame, &style_name, &atom_ids, &columns)?;

    // Velocities section when all three components exist.
    if let (Some(vx), Some(vy), Some(vz)) = (
        frame.column("atoms", keys::VX).and_then(|c| c.as_float()),
        frame.column("atoms", keys::VY).and_then(|c| c.as_float()),
        frame.column("atoms", keys::VZ).and_then(|c| c.as_float()),
    ) {
        writeln!(writer, "Velocities")?;
        writeln!(writer)?;
        for i in 0..n_atoms {
            writeln!(
                writer,
                "{} {} {} {}",
                atom_ids[i],
                vx[[i]] * units.velocity,
                vy[[i]] * units.velocity,
                vz[[i]] * units.velocity
            )?;
        }
        writeln!(writer)?;
    }

    // The bonds, then the Drude springs, in the order `lammps_type_labels`
    // lists their type ids.
    write_topology_section(
        writer,
        frame,
        "Bonds",
        &[SPRINGS.1, SPRINGS.0],
        2,
        &atom_ids,
        bond_rt.map(|r| r.type_ids()).unwrap_or(&[]),
    )?;
    write_topology_section(
        writer,
        frame,
        "Angles",
        &["angles"],
        3,
        &atom_ids,
        angle_rt.map(|r| r.type_ids()).unwrap_or(&[]),
    )?;
    write_topology_section(
        writer,
        frame,
        "Dihedrals",
        &["dihedrals"],
        4,
        &atom_ids,
        dihedral_rt.map(|r| r.type_ids()).unwrap_or(&[]),
    )?;
    write_topology_section(
        writer,
        frame,
        "Impropers",
        &["impropers"],
        4,
        &atom_ids,
        improper_rt.map(|r| r.type_ids()).unwrap_or(&[]),
    )?;
    // The type is the map's index in the `fix cmap` file, which
    // `LammpsForcefieldWriter::write_cmap_str` writes in this same label id order.
    write_topology_section(
        writer,
        frame,
        "CMAP",
        &["cmaps"],
        5,
        &atom_ids,
        cmap_rt.map(|r| r.type_ids()).unwrap_or(&[]),
    )?;

    Ok(())
}

// ============================================================================
// Public API + streaming index
// ============================================================================

pub fn read_lammps_data<P: AsRef<Path>>(path: P) -> std::io::Result<Frame> {
    let file = File::open(path)?;
    let mut reader = LammpsDataReader::new(BufReader::new(file));
    reader
        .read()?
        .ok_or_else(|| invalid_data("No frame found in LAMMPS data file"))
}

/// Write `frame` as a LAMMPS data file, in the inferred `atom_style` with the
/// frame's numbers unchanged.
///
/// Atoms, `Masses`, `Velocities`, the topology sections and the `* Type
/// Labels` are written. A `drudes` row (a Drude core–shell spring) is a
/// `Bonds` row after the bonds, typed as a bond of its own label
/// ([`lammps_type_labels`](crate::io::lammps::lammps_type_labels)), and a
/// Drude system's `fix drude` C/D/N flags head the file as a comment
/// (the DRUDE package reads the springs as bonds). A `virtual_sites` row is
/// refused: a data file builds no virtual site.
pub fn write_lammps_data<P: AsRef<Path>>(path: P, frame: &impl FrameAccess) -> std::io::Result<()> {
    write_lammps_data_with(path, frame, &LammpsDataWriteOptions::default())
}

/// [`write_lammps_data`] in the `atom_style` and `units` `options` name
/// ([`LammpsDataWriteOptions`]): a named style is written exactly or refused
/// when the frame lacks a column it needs; a named `units` converts the
/// geometry, masses, charges and velocities from the frame's units.
pub fn write_lammps_data_with<P: AsRef<Path>>(
    path: P,
    frame: &impl FrameAccess,
    options: &LammpsDataWriteOptions,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    write_lammps_data_frame_with(&mut writer, frame, options, &HashMap::new())?;
    writer.flush()
}

/// Read LAMMPS data `text` — [`read_lammps_data`] on text in memory.
pub fn read_lammps_data_str(text: &str) -> std::io::Result<Frame> {
    LammpsDataReader::new(Cursor::new(text.as_bytes()))
        .read()?
        .ok_or_else(|| invalid_data("No frame found in LAMMPS data text"))
}

/// Write a frame as LAMMPS data text — [`write_lammps_data`] into memory.
pub fn write_lammps_data_str(frame: &impl FrameAccess) -> std::io::Result<String> {
    write_lammps_data_str_with(frame, &LammpsDataWriteOptions::default())
}

/// [`write_lammps_data_with`] into memory.
pub fn write_lammps_data_str_with(
    frame: &impl FrameAccess,
    options: &LammpsDataWriteOptions,
) -> std::io::Result<String> {
    let mut buf = Vec::new();
    write_lammps_data_frame_with(&mut buf, frame, options, &HashMap::new())?;
    String::from_utf8(buf).map_err(invalid_data)
}

/// [`write_lammps_data`], giving each declared atom type no row uses the mass
/// `label_masses` holds for its label (instead of the placeholder `1`). The
/// `fix bond/react` file set needs it: a type only a template uses is
/// declared in the data file, and LAMMPS reads its mass from there.
#[cfg(feature = "ff")]
pub(crate) fn write_lammps_data_with_masses<P: AsRef<Path>>(
    path: P,
    frame: &impl FrameAccess,
    label_masses: &HashMap<String, F>,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    write_lammps_data_frame_with(
        &mut writer,
        frame,
        &LammpsDataWriteOptions::default(),
        label_masses,
    )?;
    writer.flush()
}

pub fn read_lammps_data_bytes(bytes: &[u8]) -> std::io::Result<Frame> {
    let mut reader = LammpsDataReader::new(Cursor::new(bytes));
    reader
        .read()?
        .ok_or_else(|| invalid_data("No frame found in LAMMPS data slice"))
}

pub struct LammpsDataIndexBuilder {
    bytes_seen: u64,
}

impl Default for LammpsDataIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LammpsDataIndexBuilder {
    pub fn new() -> Self {
        Self { bytes_seen: 0 }
    }
}

impl FrameIndexBuilder for LammpsDataIndexBuilder {
    fn feed(&mut self, chunk: &[u8], global_offset: u64) {
        self.bytes_seen = global_offset.saturating_add(chunk.len() as u64);
    }
    fn drain(&mut self) -> Vec<FrameOffset> {
        Vec::new()
    }
    fn finish(self: Box<Self>) -> std::io::Result<Vec<FrameOffset>> {
        if self.bytes_seen == 0 {
            return Ok(Vec::new());
        }
        if self.bytes_seen > u32::MAX as u64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "LAMMPS data file exceeds 4 GiB",
            ));
        }
        Ok(vec![FrameOffset {
            byte_offset: 0,
            byte_len: self.bytes_seen as u32,
        }])
    }
    fn bytes_seen(&self) -> u64 {
        self.bytes_seen
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod streaming_tests {
    use super::*;

    const TINY_DATA: &str = concat!(
        "LAMMPS data file\n\n2 atoms\n1 atom types\n\n",
        "0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n\n",
        "Atoms\n\n1 1 0.0 0.0 0.0\n2 1 1.0 0.0 0.0\n",
    );

    fn build_chunked(bytes: &[u8], cs: usize) -> Vec<FrameOffset> {
        let mut b = Box::new(LammpsDataIndexBuilder::new());
        let mut off: u64 = 0;
        let mut out = Vec::new();
        for piece in bytes.chunks(cs.max(1)) {
            b.feed(piece, off);
            off += piece.len() as u64;
            out.extend(b.drain());
        }
        out.extend(b.finish().expect("finish"));
        out
    }

    #[test]
    fn lammps_data_streaming_single_frame() {
        let bytes = TINY_DATA.as_bytes();
        let one = build_chunked(bytes, bytes.len());
        for cs in [1usize, 7, 31, 64, bytes.len()] {
            assert_eq!(one, build_chunked(bytes, cs));
        }
        let frame = read_lammps_data_bytes(bytes).expect("parse");
        assert_eq!(frame.get("atoms").unwrap().n_rows().unwrap(), 2);
    }
}

#[cfg(test)]
mod atom_style_tests {
    use super::*;
    use molrs::core::FrameAccess;

    fn parse_text(text: &str) -> Frame {
        read_lammps_data_bytes(text.as_bytes()).expect("parse")
    }

    fn xyz(frame: &Frame, i: usize) -> (f64, f64, f64) {
        (
            frame
                .column("atoms", keys::X)
                .and_then(|c| c.as_float())
                .unwrap()[i],
            frame
                .column("atoms", keys::Y)
                .and_then(|c| c.as_float())
                .unwrap()[i],
            frame
                .column("atoms", keys::Z)
                .and_then(|c| c.as_float())
                .unwrap()[i],
        )
    }

    #[test]
    fn normalize_strips_accelerator_suffixes() {
        use crate::io::lammps::atom_style::normalize_atom_style;
        assert_eq!(normalize_atom_style("angle/kk"), "angle");
        assert_eq!(normalize_atom_style("bpm/sphere"), "bpm/sphere");
        assert_eq!(normalize_atom_style("ANGLE/KK"), "angle");
    }

    #[test]
    fn style_hint_parses_angle_kk() {
        assert_eq!(
            parse_atoms_style_hint("Atoms # angle/kk"),
            Some("angle".into())
        );
        assert_eq!(
            parse_atoms_style_hint("Atoms # hybrid charge bond"),
            Some("hybrid".into())
        );
    }

    /// `write_data` records the unit style on its title line; it lands in
    /// `lammps_units`, and a title without one sets no key.
    #[test]
    fn title_units_land_in_meta() {
        let body = "\n1 atoms\n1 atom types\n\n\
                    0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n\
                    Atoms # atomic\n\n1 1 0.1 0.2 0.3\n";
        let titled = format!(
            "LAMMPS data file via write_data, version 4 Jul 2026, \
             timestep = 50000000, units = lj\n{body}"
        );
        let frame = parse_text(&titled);
        assert_eq!(
            frame.meta.get(LAMMPS_UNITS).and_then(|v| v.as_str()),
            Some("lj")
        );
        let bare = parse_text(&format!("LAMMPS data file\n{body}"));
        assert!(!bare.meta.contains_key(LAMMPS_UNITS));
    }

    #[test]
    fn angle_kk_with_image_flags() {
        let text = concat!(
            "LAMMPS data file\n\n2 atoms\n1 atom types\n\n",
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
            "Atoms # angle/kk\n\n",
            "1 42 1 1.5 2.5 3.5 0 0 1\n",
            "2 0 2 4.0 5.0 6.0 1 -1 0\n",
        );
        let frame = parse_text(text);
        assert_eq!(
            frame
                .column("atoms", keys::MOL_ID)
                .and_then(|c| c.as_uint())
                .unwrap()[0],
            42
        );
        assert_eq!(xyz(&frame, 0), (1.5, 2.5, 3.5));
        assert_eq!(
            frame
                .column("atoms", keys::IZ)
                .and_then(|c| c.as_int())
                .unwrap()[0],
            1
        );
        assert!(
            frame
                .column("atoms", keys::CHARGE)
                .and_then(|c| c.as_float())
                .is_none()
        );
    }

    #[test]
    fn bond_and_molecular_styles() {
        for style in ["bond", "molecular", "angle"] {
            let text = format!(
                "LAMMPS data file\n\n1 atoms\n1 atom types\n\n\
                 0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n\
                 Atoms # {style}\n\n7 3 2 0.1 0.2 0.3\n"
            );
            let frame = parse_text(&text);
            assert_eq!(
                frame
                    .column("atoms", keys::MOL_ID)
                    .and_then(|c| c.as_uint())
                    .unwrap()[0],
                3
            );
            assert_eq!(xyz(&frame, 0), (0.1, 0.2, 0.3));
        }
    }

    #[test]
    fn full_charge_atomic_sphere_body_dipole() {
        let full = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms # full\n\n1 2 3 -0.8 1.0 2.0 3.0 1 0 -1\n",
        );
        let f = parse_text(full);
        assert!(
            (f.column("atoms", keys::CHARGE)
                .and_then(|c| c.as_float())
                .unwrap()[0]
                + 0.8)
                .abs()
                < 1e-12
        );
        assert_eq!(xyz(&f, 0), (1.0, 2.0, 3.0));
        assert_eq!(
            f.column("atoms", "ix").and_then(|c| c.as_int()).unwrap()[0],
            1
        );

        let charge = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms # charge\n\n5 1 -0.5 9.0 8.0 7.0\n",
        );
        let f = parse_text(charge);
        assert_eq!(xyz(&f, 0), (9.0, 8.0, 7.0));

        let sphere = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms # sphere\n\n1 1 1.0 2.5 3.0 4.0 5.0\n",
        );
        let f = parse_text(sphere);
        assert_eq!(xyz(&f, 0), (3.0, 4.0, 5.0));
        assert!(
            (f.column("atoms", "diameter")
                .and_then(|c| c.as_float())
                .unwrap()[0]
                - 1.0)
                .abs()
                < 1e-12
        );

        let body = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms # body\n\n1 1 1 6.0 -1.5 -2.5 0.0 1 2 0\n",
        );
        let f = parse_text(body);
        assert!(
            (f.column("atoms", keys::MASS)
                .and_then(|c| c.as_float())
                .unwrap()[0]
                - 6.0)
                .abs()
                < 1e-12
        );
        assert_eq!(xyz(&f, 0), (-1.5, -2.5, 0.0));

        let dipole = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms # dipole\n\n1 1 0.5 1.0 2.0 3.0 0.1 0.2 0.3\n",
        );
        let f = parse_text(dipole);
        assert!(
            (f.column("atoms", keys::MUX)
                .and_then(|c| c.as_float())
                .unwrap()[0]
                - 0.1)
                .abs()
                < 1e-12
        );
        assert_eq!(xyz(&f, 0), (1.0, 2.0, 3.0));
    }

    #[test]
    fn pe_angle_kk_sample_and_masses() {
        let text = concat!(
            "LAMMPS data file\n\n2 atoms\n2 atom types\n\n",
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
            "Masses\n\n1 12.0\n2 1.0\n\n",
            "Atoms # angle\n\n",
            "182153 45539 1 1.9 1.8 1.6 0 0 1\n",
            "10 0 2 0.5 0.5 0.5 0 0 0\n",
        );
        let frame = parse_text(text);
        assert_eq!(
            frame
                .column("atoms", keys::MOL_ID)
                .and_then(|c| c.as_uint())
                .unwrap()[0],
            45539
        );
        assert!(
            (frame
                .column("atoms", keys::MASS)
                .and_then(|c| c.as_float())
                .unwrap()[0]
                - 12.0)
                .abs()
                < 1e-12
        );
        assert!(
            (frame
                .column("atoms", keys::MASS)
                .and_then(|c| c.as_float())
                .unwrap()[1]
                - 1.0)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn topology_bonds_angles() {
        let text = concat!(
            "LAMMPS data file\n\n3 atoms\n2 bonds\n1 angles\n1 atom types\n",
            "1 bond types\n1 angle types\n\n",
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
            "Atoms # molecular\n\n",
            "1 1 1 0 0 0\n2 1 1 1 0 0\n3 1 1 2 0 0\n\n",
            "Bonds\n\n1 1 1 2\n2 1 2 3\n\n",
            "Angles\n\n1 1 1 2 3\n",
        );
        let frame = parse_text(text);
        assert_eq!(frame.get("bonds").unwrap().n_rows().unwrap(), 2);
        assert_eq!(
            (
                frame
                    .column("bonds", keys::ATOMI)
                    .and_then(|c| c.as_uint())
                    .unwrap()[0],
                frame
                    .column("bonds", keys::ATOMJ)
                    .and_then(|c| c.as_uint())
                    .unwrap()[0]
            ),
            (0, 1)
        );
    }

    #[test]
    fn nine_columns_without_hint_is_molecular() {
        let text = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms\n\n1 42 1 1.5 2.5 3.5 0 0 1\n",
        );
        let frame = parse_text(text);
        assert_eq!(
            frame
                .column("atoms", keys::MOL_ID)
                .and_then(|c| c.as_uint())
                .unwrap()[0],
            42
        );
        assert_eq!(xyz(&frame, 0), (1.5, 2.5, 3.5));
    }

    #[test]
    fn write_round_trip_full_with_image_and_velocities() {
        let text = concat!(
            "LAMMPS data file\n\n2 atoms\n1 atom types\n\n",
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
            "Masses\n\n1 12.0\n\n",
            "Atoms # full\n\n",
            "1 1 1 -0.5 1.0 2.0 3.0 0 0 1\n",
            "2 1 1 0.5 4.0 5.0 6.0 1 -1 0\n",
            "\nVelocities\n\n",
            "1 0.1 0.2 0.3\n",
            "2 -0.1 0.0 0.5\n",
        );
        let frame = parse_text(text);
        let mut buf = Vec::new();
        write_lammps_data_frame(&mut buf, &frame).expect("write");
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("Atoms # full"), "{out}");
        assert!(out.contains("Velocities"), "{out}");
        assert!(out.contains("Masses"), "{out}");
        // Image flags preserved
        assert!(out.contains("0 0 1") || out.contains(" 0 0 1\n"), "{out}");

        let frame2 = read_lammps_data_bytes(out.as_bytes()).expect("re-read");
        assert_eq!(frame2.get("atoms").unwrap().n_rows().unwrap(), 2);
        assert!(
            (frame2
                .column("atoms", keys::CHARGE)
                .and_then(|c| c.as_float())
                .unwrap()[0]
                + 0.5)
                .abs()
                < 1e-12
        );
        assert!(
            (frame2
                .column("atoms", keys::VX)
                .and_then(|c| c.as_float())
                .unwrap()[0]
                - 0.1)
                .abs()
                < 1e-12
        );
        assert_eq!(
            frame2
                .column("atoms", "iz")
                .and_then(|c| c.as_int())
                .unwrap()[0],
            1
        );
    }

    #[test]
    fn write_round_trip_sphere_and_dipole() {
        let sphere = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms # sphere\n\n1 1 1.0 2.5 3.0 4.0 5.0\n",
        );
        let frame = parse_text(sphere);
        let mut buf = Vec::new();
        write_lammps_data_frame(&mut buf, &frame).expect("write");
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("Atoms # sphere"), "{out}");
        let f2 = read_lammps_data_bytes(out.as_bytes()).unwrap();
        assert!(
            (f2.column("atoms", "diameter")
                .and_then(|c| c.as_float())
                .unwrap()[0]
                - 1.0)
                .abs()
                < 1e-12
        );
        assert_eq!(
            (
                f2.column("atoms", keys::X)
                    .and_then(|c| c.as_float())
                    .unwrap()[0],
                f2.column("atoms", keys::Y)
                    .and_then(|c| c.as_float())
                    .unwrap()[0],
                f2.column("atoms", keys::Z)
                    .and_then(|c| c.as_float())
                    .unwrap()[0],
            ),
            (3.0, 4.0, 5.0)
        );

        let dipole = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n",
            "Atoms # dipole\n\n1 1 0.5 1.0 2.0 3.0 0.1 0.2 0.3\n",
        );
        let frame = parse_text(dipole);
        let mut buf = Vec::new();
        write_lammps_data_frame(&mut buf, &frame).expect("write");
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("Atoms # dipole"), "{out}");
        let f2 = read_lammps_data_bytes(out.as_bytes()).unwrap();
        assert!(
            (f2.column("atoms", keys::MUX)
                .and_then(|c| c.as_float())
                .unwrap()[0]
                - 0.1)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn write_round_trip_topology_angles() {
        let text = concat!(
            "LAMMPS data file\n\n3 atoms\n2 bonds\n1 angles\n1 atom types\n",
            "1 bond types\n1 angle types\n\n",
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
            "Atoms # molecular\n\n",
            "1 1 1 0 0 0\n2 1 1 1 0 0\n3 1 1 2 0 0\n\n",
            "Bonds\n\n1 1 1 2\n2 1 2 3\n\n",
            "Angles\n\n1 1 1 2 3\n",
        );
        let frame = parse_text(text);
        let mut buf = Vec::new();
        write_lammps_data_frame(&mut buf, &frame).expect("write");
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("2 bonds"), "{out}");
        assert!(out.contains("1 angles"), "{out}");
        assert!(out.contains("Atoms # molecular"), "{out}");
        let f2 = read_lammps_data_bytes(out.as_bytes()).unwrap();
        assert_eq!(f2.get("bonds").unwrap().n_rows().unwrap(), 2);
        assert_eq!(f2.get("angles").unwrap().n_rows().unwrap(), 1);
    }

    /// A box-less frame is written inside the bounds of its coordinates,
    /// [`BOXLESS_MARGIN`] past the outermost atom on every side — a planar
    /// molecule included — never in a placeholder `0 1` box that LAMMPS would
    /// wrap (`p`) or drop (`f`, `s`) the atoms out of.
    #[test]
    fn write_boxless_frame_inside_the_bounds_of_its_atoms() {
        use crate::core::Block;
        use crate::core::Frame as CoreFrame;
        use ndarray::ArrayD;

        let column = |v: Vec<f64>| ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), v).unwrap();
        let mut atoms = Block::new();
        atoms
            .insert(
                keys::TYPE,
                ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec!["c".to_string(); 3]).unwrap(),
            )
            .unwrap();
        atoms.insert(keys::X, column(vec![-2.0, 3.5, 0.0])).unwrap();
        atoms
            .insert(keys::Y, column(vec![10.0, 12.0, 11.0]))
            .unwrap();
        atoms.insert(keys::Z, column(vec![0.0, 0.0, 0.0])).unwrap();
        let mut frame = CoreFrame::new();
        frame.insert("atoms", atoms);
        let mut buf = Vec::new();
        write_lammps_data_frame(&mut buf, &frame).expect("write");
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("\n-3 4.5 xlo xhi\n"), "{out}");
        assert!(out.contains("\n9 13 ylo yhi\n"), "{out}");
        assert!(out.contains("\n-1 1 zlo zhi\n"), "{out}");

        // A free box has no cell: its identity placeholder is no box either.
        let mut free = frame.clone();
        free.simbox = Some(
            SimBox::new_cell(
                ndarray::Array2::eye(3),
                ndarray::array![0.0, 0.0, 0.0],
                [false; 3],
                false,
            )
            .unwrap(),
        );
        let mut buf = Vec::new();
        write_lammps_data_frame(&mut buf, &free).expect("write");
        assert_eq!(String::from_utf8(buf).unwrap(), out);

        let mut nan = frame.clone();
        let atoms = nan.get_mut("atoms").unwrap();
        atoms
            .insert(keys::Z, column(vec![0.0, f64::NAN, 0.0]))
            .unwrap();
        let err = write_lammps_data_frame(&mut Vec::new(), &nan).unwrap_err();
        assert!(err.to_string().contains("no box"), "{err}");
    }

    #[test]
    fn write_refuses_bonds_without_mol_id() {
        use crate::core::Block;
        use crate::core::Frame as CoreFrame;
        use ndarray::ArrayD;

        let mut frame = CoreFrame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                keys::TYPE,
                ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec!["c".to_string(); 2]).unwrap(),
            )
            .unwrap();
        for key in [keys::X, keys::Y, keys::Z] {
            atoms
                .insert(
                    key,
                    ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.0_f64, 1.0]).unwrap(),
                )
                .unwrap();
        }
        frame.insert("atoms", atoms);
        let mut bonds = Block::new();
        for (key, v) in [(keys::ATOMI, 0_u32), (keys::ATOMJ, 1_u32)] {
            bonds
                .insert(
                    key,
                    ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![v]).unwrap(),
                )
                .unwrap();
        }
        frame.insert("bonds", bonds);

        let err = write_lammps_data_frame(&mut Vec::new(), &frame).unwrap_err();
        assert!(err.to_string().contains("mol_id"), "{err}");
    }

    /// A type label is a type name, matched exactly: `c3-c3-h1` and
    /// `h1-c3-c3` are two angle types, both written.
    #[test]
    fn write_keeps_reverse_angle_type_labels_as_two_types() {
        use crate::core::Block;
        use crate::core::Frame as CoreFrame;
        use ndarray::ArrayD;

        let mut frame = CoreFrame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                keys::TYPE,
                ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[3]),
                    vec!["c3".to_string(), "c3".to_string(), "h1".to_string()],
                )
                .unwrap(),
            )
            .unwrap();
        for (key, vals) in [
            (keys::X, vec![0.0_f64, 1.0, 2.0]),
            (keys::Y, vec![0.0_f64, 0.0, 0.0]),
            (keys::Z, vec![0.0_f64, 0.0, 0.0]),
        ] {
            atoms
                .insert(
                    key,
                    ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vals).unwrap(),
                )
                .unwrap();
        }
        atoms
            .insert(
                keys::MOL_ID,
                ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1_u32; 3]).unwrap(),
            )
            .unwrap();
        frame.insert("atoms", atoms);

        let mut angles = Block::new();
        angles
            .insert(
                keys::TYPE,
                ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[2]),
                    vec!["c3-c3-h1".to_string(), "h1-c3-c3".to_string()],
                )
                .unwrap(),
            )
            .unwrap();
        for (key, vals) in [
            (keys::ATOMI, vec![0_u32, 2]),
            (keys::ATOMJ, vec![1_u32, 1]),
            (keys::ATOMK, vec![2_u32, 0]),
        ] {
            angles
                .insert(
                    key,
                    ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vals).unwrap(),
                )
                .unwrap();
        }
        frame.insert("angles", angles);

        let mut buf = Vec::new();
        write_lammps_data_frame(&mut buf, &frame).expect("write");
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("2 angle types"), "{out}");
        assert!(out.contains("c3-c3-h1"), "{out}");
        assert!(out.contains("h1-c3-c3"), "{out}");
    }

    #[test]
    fn write_resolves_string_types_ids_and_masses_without_prepare() {
        // Frame carries only string `type` + coords + element — no type_id,
        // no id, no mass column. Writer must still emit a complete data file.
        use crate::core::Block;
        use crate::core::Frame as CoreFrame;
        use ndarray::ArrayD;

        let mut frame = CoreFrame::new();
        let mut atoms = Block::new();
        let n = 3;
        atoms
            .insert(
                keys::TYPE,
                ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[n]),
                    vec!["O".to_string(), "C".to_string(), "H".to_string()],
                )
                .unwrap(),
            )
            .unwrap();
        atoms
            .insert(
                keys::ELEMENT,
                ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[n]),
                    vec!["O".to_string(), "C".to_string(), "H".to_string()],
                )
                .unwrap(),
            )
            .unwrap();
        for (key, vals) in [
            (keys::X, vec![0.0_f64, 1.0, 0.0]),
            (keys::Y, vec![0.0_f64, 0.0, 1.0]),
            (keys::Z, vec![0.0_f64, 0.0, 0.0]),
        ] {
            atoms
                .insert(
                    key,
                    ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), vals).unwrap(),
                )
                .unwrap();
        }
        frame.insert("atoms", atoms);

        let mut buf = Vec::new();
        write_lammps_data_frame(&mut buf, &frame).expect("write");
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("3 atoms"), "{out}");
        assert!(out.contains("3 atom types"), "{out}");
        assert!(out.contains("Atom Type Labels"), "{out}");
        // Sorted label order: C, H, O
        assert!(out.contains("1 C"), "{out}");
        assert!(out.contains("2 H"), "{out}");
        assert!(out.contains("3 O"), "{out}");
        assert!(out.contains("Masses"), "{out}");
        // Auto-numbered atom ids 1..N
        assert!(out.contains("\n1 "), "{out}");
        assert!(out.contains("\n2 "), "{out}");
        assert!(out.contains("\n3 "), "{out}");
    }

    #[test]
    fn dump_aliases_q_and_mol() {
        use crate::io::lammps::columns::{canonical_dump_column, native_dump_column};
        assert_eq!(canonical_dump_column("q"), keys::CHARGE);
        assert_eq!(canonical_dump_column("mol"), keys::MOL_ID);
        assert_eq!(native_dump_column(keys::CHARGE), "q");
        assert_eq!(native_dump_column(keys::MOL_ID), "mol");
        assert_eq!(canonical_dump_column("spin"), "espin");
    }

    // ------------------------------------------------------------------
    // Sections: per-atom rows, unknown sections, header recognition
    // ------------------------------------------------------------------

    /// Three `bond`-style atoms (ids 1..3, no q column) followed by `body`.
    fn three_bond_atoms_then(body: &str) -> String {
        format!(
            "LAMMPS data file\n\n3 atoms\n1 atom types\n\n\
             0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n\
             Atoms # bond\n\n\
             1 1 1 0.0 0.0 0.0\n2 1 1 1.0 0.0 0.0\n3 1 1 2.0 0.0 0.0\n\n\
             {body}"
        )
    }

    fn refusal(text: &str) -> std::io::Error {
        match read_lammps_data_bytes(text.as_bytes()) {
            Ok(_) => panic!("expected the reader to refuse:\n{text}"),
            Err(e) => e,
        }
    }

    fn assert_invalid_data_naming(err: &std::io::Error, section: &str) {
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{err}");
        assert!(err.to_string().contains(section), "{err}");
    }

    #[test]
    fn charges_section_sets_charge_by_atom_id() {
        let text = three_bond_atoms_then("Charges\n\n3 0.5\n1 -0.25\n2 -0.25\n");
        let frame = parse_text(&text);
        let q = frame
            .column("atoms", keys::CHARGE)
            .and_then(|c| c.as_float())
            .expect("Charges must create the charge column");
        let expected = [-0.25, -0.25, 0.5];
        for (i, want) in expected.iter().enumerate() {
            assert!((q[i] - want).abs() < 1e-12, "row {i}: {} != {want}", q[i]);
        }
    }

    #[test]
    fn charges_section_overrides_atom_style_charge() {
        let text = concat!(
            "LAMMPS data file\n\n3 atoms\n1 atom types\n\n",
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
            "Atoms # full\n\n",
            "1 1 1 1.0 0.0 0.0 0.0\n2 1 1 1.0 1.0 0.0 0.0\n3 1 1 1.0 2.0 0.0 0.0\n\n",
            "Charges\n\n1 0.4\n2 -0.8\n3 0.4\n",
        );
        let frame = parse_text(text);
        let q = frame
            .column("atoms", keys::CHARGE)
            .and_then(|c| c.as_float())
            .unwrap();
        let expected = [0.4, -0.8, 0.4];
        for (i, want) in expected.iter().enumerate() {
            assert!((q[i] - want).abs() < 1e-12, "row {i}: {} != {want}", q[i]);
        }
    }

    #[test]
    fn charges_unknown_atom_id_is_refused() {
        let text = three_bond_atoms_then("Charges\n\n1 0.1\n7 0.2\n2 0.3\n3 0.4\n");
        assert_invalid_data_naming(&refusal(&text), "Charges");
    }

    #[test]
    fn charges_duplicate_atom_id_is_refused() {
        let text = three_bond_atoms_then("Charges\n\n1 0.1\n1 0.2\n2 0.3\n3 0.4\n");
        assert_invalid_data_naming(&refusal(&text), "Charges");
    }

    #[test]
    fn charges_fewer_rows_than_atoms_is_refused() {
        let text = three_bond_atoms_then("Charges\n\n1 0.1\n2 0.3\n");
        assert_invalid_data_naming(&refusal(&text), "Charges");
    }

    #[test]
    fn velocities_unknown_atom_id_is_refused() {
        let text = three_bond_atoms_then(
            "Velocities\n\n1 0.1 0.0 0.0\n7 0.2 0.0 0.0\n2 0.3 0.0 0.0\n3 0.4 0.0 0.0\n",
        );
        assert_invalid_data_naming(&refusal(&text), "Velocities");
    }

    #[test]
    fn velocities_duplicate_atom_id_is_refused() {
        let text = three_bond_atoms_then(
            "Velocities\n\n1 0.1 0.0 0.0\n1 0.2 0.0 0.0\n2 0.3 0.0 0.0\n3 0.4 0.0 0.0\n",
        );
        assert_invalid_data_naming(&refusal(&text), "Velocities");
    }

    #[test]
    fn velocities_fewer_rows_than_atoms_is_refused() {
        let text = three_bond_atoms_then("Velocities\n\n1 0.1 0.0 0.0\n2 0.3 0.0 0.0\n");
        assert_invalid_data_naming(&refusal(&text), "Velocities");
    }

    /// Three `molecular` atoms, an `Ellipsoids` section, then two bonds.
    const WITH_ELLIPSOIDS: &str = concat!(
        "LAMMPS data file\n\n3 atoms\n2 bonds\n1 ellipsoids\n1 atom types\n1 bond types\n\n",
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
        "Atoms # molecular\n\n",
        "1 1 1 0 0 0\n2 1 1 1 0 0\n3 1 1 2 0 0\n\n",
        "Ellipsoids\n\n1 1.0 1.0 1.0 1.0 0.0 0.0 0.0\n\n",
        "Bonds\n\n1 1 1 2\n2 1 2 3\n",
    );

    #[test]
    fn unknown_section_is_refused_by_name() {
        let err = refusal(WITH_ELLIPSOIDS);
        assert_invalid_data_naming(&err, "Ellipsoids");
        assert!(err.to_string().contains("with_skipped_section"), "{err}");
    }

    #[test]
    fn with_skipped_section_discards_that_section_only() {
        let frame = LammpsDataReader::new(Cursor::new(WITH_ELLIPSOIDS.as_bytes()))
            .with_skipped_section("Ellipsoids")
            .read()
            .expect("skipped section must not refuse the file")
            .expect("one frame");
        assert_eq!(frame.get("atoms").unwrap().n_rows().unwrap(), 3);
        assert_eq!(frame.get("bonds").unwrap().n_rows().unwrap(), 2);
    }

    #[test]
    fn type_labelled_masses_row_is_a_row_not_a_header() {
        let text = concat!(
            "LAMMPS data file\n\n3 atoms\n1 atom types\n\n",
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
            "Atom Type Labels\n\n1 CA\n\n",
            "Masses\n\nCA 12.011\n\n",
            "Atoms # bond\n\n",
            "1 1 1 0.0 0.0 0.0\n2 1 1 1.0 0.0 0.0\n3 1 1 2.0 0.0 0.0\n",
        );
        let frame = parse_text(text);
        let mass = frame
            .column("atoms", keys::MASS)
            .and_then(|c| c.as_float())
            .expect("the CA mass row must yield a mass column");
        for i in 0..3 {
            assert!((mass[i] - 12.011).abs() < 1e-12, "row {i}: {}", mass[i]);
        }
    }

    #[test]
    fn pairij_coeffs_section_is_captured() {
        let text = three_bond_atoms_then("PairIJ Coeffs\n\n1 1 0.1 3.0\n");
        let frame = parse_text(&text);
        let coeffs = frame
            .meta
            .get(LAMMPS_COEFFS_TEXT)
            .and_then(|value| value.as_str())
            .expect("PairIJ Coeffs must land in lammps_coeffs_text");
        assert!(coeffs.contains("PairIJ Coeffs"), "{coeffs}");
        assert!(coeffs.contains("1 1 0.1 3.0"), "{coeffs}");
    }

    /// Three `bond` atoms, a `Pair Coeffs` section, then a fix-defined
    /// `BiTorsions` section (`fix amoeba/bitorsion`) this reader has no name
    /// for.
    fn pair_coeffs_then_bitorsions() -> String {
        three_bond_atoms_then("Pair Coeffs\n\n1 0.1 3.0\n\nBiTorsions\n\n1 1 1 2 3 1 2\n")
    }

    #[test]
    fn unknown_section_after_coeffs_is_refused_by_name() {
        let err = refusal(&pair_coeffs_then_bitorsions());
        assert_invalid_data_naming(&err, "BiTorsions");
        assert!(err.to_string().contains("with_skipped_section"), "{err}");
    }

    #[test]
    fn with_skipped_section_skips_a_name_outside_the_vocabulary() {
        let text = pair_coeffs_then_bitorsions();
        let frame = LammpsDataReader::new(Cursor::new(text.as_bytes()))
            .with_skipped_section("BiTorsions")
            .read()
            .expect("a skipped BiTorsions section must not refuse the file")
            .expect("one frame");
        let coeffs = frame
            .meta
            .get(LAMMPS_COEFFS_TEXT)
            .and_then(|value| value.as_str())
            .expect("Pair Coeffs must land in lammps_coeffs_text");
        assert!(coeffs.contains("1 0.1 3.0"), "{coeffs}");
        assert!(!coeffs.contains("BiTorsions"), "{coeffs}");
        assert!(!coeffs.contains("1 1 1 2 3 1 2"), "{coeffs}");
    }

    #[test]
    fn unparseable_header_count_line_is_refused_by_name() {
        let text = concat!(
            "LAMMPS data file\n\n1 atoms\nmany bonds\n1 atom types\n\n",
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
            "Atoms # bond\n\n1 1 1 0.0 0.0 0.0\n",
        );
        assert_invalid_data_naming(&refusal(text), "many bonds");
    }

    #[test]
    fn repeated_masses_section_is_refused_by_name() {
        let text = concat!(
            "LAMMPS data file\n\n1 atoms\n1 atom types\n\n",
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n",
            "Masses\n\n1 12.0\n\n",
            "Masses\n\n1 14.0\n\n",
            "Atoms # bond\n\n1 1 1 0.0 0.0 0.0\n",
        );
        assert_invalid_data_naming(&refusal(text), "Masses");
    }

    /// A frame's `cmaps` block goes out as `fix cmap`'s `N crossterms` header
    /// line and `CMAP` section (types by label id) and comes back as a
    /// `cmaps` block of `atomi` … `atomm` and `type_id`; `N cmap crossterms`
    /// reads too, and a `CMAP` section without its count is refused.
    #[test]
    fn a_cmaps_block_round_trips_through_the_cmap_section() {
        let text = three_bond_atoms_then("");
        let mut frame = parse_text(&text.replace("3 atoms", "6 atoms").replace(
            "3 1 1 2.0 0.0 0.0\n",
            "3 1 1 2.0 0.0 0.0\n4 1 1 3.0 0.0 0.0\n5 1 1 4.0 0.0 0.0\n6 1 1 5.0 0.0 0.0\n",
        ));
        let mut cmaps = Block::new();
        for (p, key) in [
            keys::ATOMI,
            keys::ATOMJ,
            keys::ATOMK,
            keys::ATOML,
            keys::ATOMM,
        ]
        .into_iter()
        .enumerate()
        {
            let col: Vec<Idx> = vec![p as Idx, p as Idx + 1];
            cmaps
                .insert(key, ndarray::Array1::from_vec(col).into_dyn())
                .unwrap();
        }
        cmaps
            .insert(
                keys::TYPE,
                ndarray::Array1::from_vec(vec!["psi".to_owned(), "phi".to_owned()]).into_dyn(),
            )
            .unwrap();
        frame.insert("cmaps", cmaps);
        let mut out = Vec::new();
        LammpsDataWriter::new(&mut out).write(&frame).unwrap();
        let written = String::from_utf8(out).unwrap();
        assert!(written.contains("\n2 crossterms\n"), "{written}");
        // Labels sorted: `phi` = 1, `psi` = 2.
        assert!(
            written.contains("CMAP\n\n1 2 1 2 3 4 5\n2 1 2 3 4 5 6\n"),
            "{written}"
        );

        let back = parse_text(&written);
        let block = &back["cmaps"];
        let col = |k: &str| {
            block
                .get(k)
                .unwrap()
                .as_uint()
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>()
        };
        assert_eq!(col(keys::ATOMI), [0, 1]);
        assert_eq!(col(keys::ATOMM), [4, 5]);
        assert_eq!(col(keys::TYPE_ID), [2, 1]);

        let spelled = parse_text(&written.replace("2 crossterms", "2 cmap crossterms"));
        assert_eq!(spelled["cmaps"].n_rows(), Some(2));
        let err = refusal(&written.replace("2 crossterms\n", ""));
        assert_invalid_data_naming(&err, "crossterms");
    }
}

#[cfg(test)]
mod label_and_drude_tests {
    use super::*;

    const LABELLED: &str = "\
test

3 atoms
1 bonds
2 atom types
1 bond types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Atom Type Labels

1 OW
2 HW

Bond Type Labels

1 OW-HW

Atoms # full

1 1 1 -0.8 0 0 0
2 1 2 0.4 1 0 0
3 1 2 0.4 0 1 0

Bonds

1 1 1 2
";

    fn strings(frame: &Frame, block: &str) -> Vec<String> {
        frame[block]
            .get("type")
            .and_then(|c| c.as_string())
            .unwrap()
            .iter()
            .cloned()
            .collect()
    }

    #[test]
    fn typed_blocks_carry_their_labels_as_type() {
        let frame = read_lammps_data_bytes(LABELLED.as_bytes()).unwrap();
        assert_eq!(strings(&frame, "atoms"), ["OW", "HW", "HW"]);
        assert_eq!(strings(&frame, "bonds"), ["OW-HW"]);
        let ids: Vec<Idx> = frame["atoms"]
            .get("type_id")
            .and_then(|c| c.as_uint())
            .unwrap()
            .iter()
            .copied()
            .collect();
        assert_eq!(ids, [1, 2, 2]);
    }

    #[test]
    fn an_unlabelled_file_spells_ids_as_labels() {
        let text = LABELLED
            .replace("Atom Type Labels\n\n1 OW\n2 HW\n\n", "")
            .replace("Bond Type Labels\n\n1 OW-HW\n\n", "");
        let frame = read_lammps_data_bytes(text.as_bytes()).unwrap();
        assert_eq!(strings(&frame, "atoms"), ["1", "2", "2"]);
        assert_eq!(strings(&frame, "bonds"), ["1"]);
    }

    #[test]
    fn an_explicit_atom_style_overrides_the_section_comment() {
        // Five columns per row: `atomic`, whatever the comment says.
        let text = "test\n\n2 atoms\n1 atom types\n\n0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n\
                    Atoms # full\n\n1 1 0.1 0.2 0.3\n2 1 0.4 0.5 0.6\n";
        let frame = LammpsDataReader::new(Cursor::new(text.as_bytes()))
            .with_atom_style("atomic")
            .unwrap()
            .read()
            .unwrap()
            .unwrap();
        let atoms = &frame["atoms"];
        assert!(!atoms.contains_key("charge") && !atoms.contains_key("mol_id"));
        assert_eq!(atoms.get("z").and_then(|c| c.as_float()).unwrap()[[1]], 0.6);
        assert!(
            LammpsDataReader::new(Cursor::new(text.as_bytes()))
                .with_atom_style("nonsense")
                .is_err()
        );
    }

    #[test]
    fn a_drude_system_writes_fix_drude_flags_in_type_order() {
        // types (sorted labels): CT=1 (core), DCT=2 (shell), HC=3 (plain).
        let mut frame = read_lammps_data_bytes(LABELLED.as_bytes()).unwrap();
        let atoms = frame.get_mut("atoms").unwrap();
        atoms
            .insert(
                "type",
                ndarray::Array1::from_vec(vec!["CT".to_string(), "DCT".into(), "HC".into()])
                    .into_dyn(),
            )
            .unwrap();
        frame.meta.remove(keys::ATOM_TYPE_LABELS);
        frame.insert("drudes", drude_rows(&[(0, 1)], &["CT-DCT"]));
        let mut out = Vec::new();
        write_lammps_data_frame(&mut out, &frame).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.contains("# fix drude flags (atom-type order): C D N\n"),
            "{text}"
        );
        // The comment sits in the header, where LAMMPS and this reader skip it.
        read_lammps_data_bytes(text.as_bytes()).unwrap();
    }

    /// `drudes` rows (`atomi` the core, `atomj` the shell) typed `types`.
    fn drude_rows(pairs: &[(Idx, Idx)], types: &[&str]) -> Block {
        let n = pairs.len();
        let mut drudes = Block::new();
        let (i, j): (Vec<Idx>, Vec<Idx>) = pairs.iter().copied().unzip();
        insert_column_of_length(&mut drudes, keys::ATOMI, i, n).unwrap();
        insert_column_of_length(&mut drudes, keys::ATOMJ, j, n).unwrap();
        let types: Vec<String> = types.iter().map(|t| t.to_string()).collect();
        insert_column_of_length(&mut drudes, keys::TYPE, types, n).unwrap();
        drudes
    }

    /// A polarisable methane-like fragment: core `CT` (1) with shell `DCT`
    /// (2) and two `HC` (3, 4), bonded `CT-HC`.
    const DRUDE_HOST: &str = "\
test

4 atoms
2 bonds
3 atom types
1 bond types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Atom Type Labels

1 CT
2 DCT
3 HC

Bond Type Labels

1 CT-HC

Atoms # full

1 1 1 0.5 0 0 0
2 1 2 -0.5 0 0 0
3 1 3 0 1 0 0
4 1 3 0 -1 0 0

Bonds

1 1 1 3
2 1 1 4
";

    fn drude_host() -> Frame {
        let mut frame = read_lammps_data_bytes(DRUDE_HOST.as_bytes()).unwrap();
        frame.insert("drudes", drude_rows(&[(0, 1)], &["CT-DCT"]));
        frame
    }

    fn written(frame: &Frame) -> std::io::Result<String> {
        let mut out = Vec::new();
        write_lammps_data_frame(&mut out, frame)?;
        Ok(String::from_utf8(out).unwrap())
    }

    /// LAMMPS's DRUDE package reads the core-shell springs as bonds: each
    /// `drudes` row is a `Bonds` row after the bonds, its type a bond type
    /// of its own label.
    #[test]
    fn drude_springs_are_written_as_bonds_of_their_own_type() {
        let text = written(&drude_host()).unwrap();
        assert!(text.contains("\n3 bonds\n"), "{text}");
        assert!(text.contains("\n2 bond types\n"), "{text}");
        assert!(
            text.contains("Bond Type Labels\n\n1 CT-DCT\n2 CT-HC\n"),
            "{text}"
        );
        assert!(
            text.contains("Bonds\n\n1 2 1 3\n2 2 1 4\n3 1 1 2\n"),
            "{text}"
        );
        assert!(text.contains("# fix drude flags (atom-type order): C D N\n"));
        // Read back, a spring is a bond: the data file has no other form.
        let back = read_lammps_data_bytes(text.as_bytes()).unwrap();
        assert_eq!(strings(&back, "bonds"), ["CT-HC", "CT-HC", "CT-DCT"]);
    }

    #[test]
    fn a_drude_row_without_a_type_is_refused_naming_the_block() {
        let mut frame = drude_host();
        let mut untyped = Block::new();
        insert_column_of_length(&mut untyped, keys::ATOMI, vec![0 as Idx], 1).unwrap();
        insert_column_of_length(&mut untyped, keys::ATOMJ, vec![1 as Idx], 1).unwrap();
        frame.insert("drudes", untyped);
        let err = written(&frame).unwrap_err().to_string();
        assert!(err.contains("drudes") && err.contains("type"), "{err}");
    }

    /// A spring is bonded topology: it needs a molecule ID per atom and a
    /// style that holds bonds.
    #[test]
    fn drude_springs_need_a_molecular_style() {
        let mut frame = drude_host();
        frame.remove("bonds");
        frame.get_mut("atoms").unwrap().remove("mol_id");
        let err = written(&frame).unwrap_err().to_string();
        assert!(err.contains("drudes") && err.contains("mol_id"), "{err}");

        let options = LammpsDataWriteOptions {
            atom_style: Some("charge".into()),
            units: None,
        };
        let mut out = Vec::new();
        let err = write_lammps_data_frame_with(&mut out, &frame, &options, &HashMap::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("drudes"), "{err}");
    }

    /// A data file builds no virtual site: a `virtual_sites` row is refused
    /// by name, not written as a free atom.
    #[test]
    fn virtual_sites_are_refused_naming_the_tip4p_pair_style() {
        let mut frame = read_lammps_data_bytes(LABELLED.as_bytes()).unwrap();
        let mut sites = Block::new();
        for (key, v) in [
            ("atomi", 0 as Idx),
            ("atomj", 1),
            ("atomk", 2),
            ("atoml", 1),
        ] {
            insert_column_of_length(&mut sites, key, vec![v], 1).unwrap();
        }
        frame.insert("virtual_sites", sites);
        let err = written(&frame).unwrap_err().to_string();
        assert!(
            err.contains("virtual_sites") && err.contains("tip4p"),
            "{err}"
        );
    }

    #[test]
    fn a_frame_without_drude_particles_writes_no_flags() {
        let frame = read_lammps_data_bytes(LABELLED.as_bytes()).unwrap();
        let mut out = Vec::new();
        write_lammps_data_frame(&mut out, &frame).unwrap();
        assert!(!String::from_utf8(out).unwrap().contains("fix drude"));
    }
}
