//! LAMMPS force-field coefficient writer: the `*.ff` include next to a data file, and the data file's `* Coeffs` sections.
use crate::io::Sink;

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ff::forcefield::{ForceField, Style, TypeDef};
use crate::ff::ir::{
    Engine, LammpsCodec, LammpsCoeffs, ParamCombination, StyleSpec, Token, UnitScale,
};
use crate::ff::style_registry::Registry;
use crate::io::lammps::drude::{drude_note, drude_springs_as_bonds};
use crate::io::lammps::forcefield_reader::{CROSS_TERM_SECTIONS, LAMMPS_CMAP_DIM, LAMMPS_CMAP_MAX};
use crate::io::lammps::units::{LammpsUnitConverter, parse_lammps_units_style};
use crate::io::writer::{ForceFieldWriteError, ForceFieldWriter, put_text};
use molrs::core::{TypeLabels, TypeName};
use ndarray::ArrayD;

/// The fix id the include gives `fix cmap`.
const CMAP_FIX_ID: &str = "cmap";

/// The conversion of `ff`'s parameters, per dimension, into the file's
/// units `to`: the identity when the two are the same unit style. A force
/// field in a unit system LAMMPS has no `units` style for is refused only
/// when it would need converting.
fn units_of(ff: &ForceField, to: &'static str) -> Result<UnitScale, String> {
    let units = ff.required_units()?;
    let from = match parse_lammps_units_style(units) {
        Ok(from) => from,
        Err(_) if units == to => return Ok(UnitScale::IDENTITY),
        Err(e) => return Err(format!("force field units: {e}")),
    };
    LammpsUnitConverter::canonical()
        .map_err(|e| format!("lammps unit system: {e}"))?
        .scale(from, to)
}

/// The spec and LAMMPS codec of `category` `style` in `reg`, or the
/// [`IrError::NoEngineForm`](crate::ff::ir::IrError::NoEngineForm) naming
/// why it has none.
pub(crate) fn codec_of<'r>(
    reg: &'r Registry,
    category: &str,
    style: &str,
) -> Result<(&'r StyleSpec, &'r dyn LammpsCodec), ForceFieldWriteError> {
    let (spec, _) = reg.style(category, style).ok_or_else(|| {
        Engine::Lammps.refuse(
            category,
            style,
            "the style is not registered (molpy.ff.style_registry.GlobalRegistry().register_style), so nothing states \
             its parameters' order and dimensions",
        )
    })?;
    let codec = spec.lammps.require(spec)?;
    Ok((spec, codec))
}

/// The LAMMPS style name of a spec with a codec.
fn lammps_name(spec: &StyleSpec) -> String {
    spec.lammps
        .lammps_name(spec)
        .expect("a style with a LAMMPS codec has a LAMMPS name")
}

/// Render one type's molrs params as the numbers of its LAMMPS coefficient
/// line, through the style's codec in the process-wide registry — the
/// inverse of
/// [`lammps_coeff_params`](crate::io::lammps::forcefield_reader::lammps_coeff_params)
/// and what every `*_coeff` line and `* Coeffs` row this writer emits is.
///
/// The result is the coefficients **after** the type field(s) of the main
/// line (a `class2` style's cross-term lines are not in it). `params` and
/// the result are both in the LAMMPS `units` style `units` (`real`,
/// `metal`, `lj`): the map is the identity on values. `style` is the molrs
/// style name (`dihedral periodic`, written as LAMMPS's `fourier`).
///
/// # Errors
///
/// A style without a LAMMPS form ([`IrError::NoEngineForm`]), an unknown
/// `units` keyword, a missing param (named), a param the line has no place
/// for, or a non-integral multiplicity.
///
/// [`IrError::NoEngineForm`]: crate::ff::ir::IrError::NoEngineForm
///
/// ```
/// use molrs::ff::ir::Params;
/// use molrs::io::lammps::forcefield_writer::lammps_coeff_values;
///
/// let p = Params::from_iter([("k", 450.0), ("r0", 0.9572)]);
/// assert_eq!(lammps_coeff_values("bond", "harmonic", &p, "real").unwrap(), [450.0, 0.9572]);
/// assert!(lammps_coeff_values("bond", "fene", &p, "real").is_err());
/// ```
#[cfg(test)]
pub(crate) fn lammps_coeff_values(
    category: &str,
    style: &str,
    params: &crate::ff::ir::Params,
    units: &str,
) -> Result<Vec<f64>, ForceFieldWriteError> {
    parse_lammps_units_style(units)?;
    crate::ff::style_registry::GlobalRegistry.with(|reg| {
        let (spec, codec) = codec_of(reg, category, style)?;
        Ok(codec
            .write(spec, params, &UnitScale::IDENTITY)?
            .values
            .into_iter()
            .map(Token::value)
            .collect())
    })
}

/// `tokens` joined as the tail of a `*_coeff` line or `* Coeffs` row.
fn render(tokens: &[Token], precision: usize) -> String {
    tokens
        .iter()
        .map(|t| t.render(precision))
        .collect::<Vec<_>>()
        .join(" ")
}

/// A bonded type category the writer emits coefficients for.
#[derive(Clone, Copy)]
struct Bonded {
    /// Style category, and the LAMMPS command prefix (`bond_style`, `bond_coeff`).
    category: &'static str,
    /// The [`TypeLabels`] block holding this category's labels.
    block: &'static str,
    /// Data-file section heading.
    heading: &'static str,
}

const BOND: Bonded = Bonded {
    category: "bond",
    block: "bonds",
    heading: "Bond Coeffs",
};
const ANGLE: Bonded = Bonded {
    category: "angle",
    block: "angles",
    heading: "Angle Coeffs",
};
const DIHEDRAL: Bonded = Bonded {
    category: "dihedral",
    block: "dihedrals",
    heading: "Dihedral Coeffs",
};
const IMPROPER: Bonded = Bonded {
    category: "improper",
    block: "impropers",
    heading: "Improper Coeffs",
};
const CMAP: Bonded = Bonded {
    category: "cmap",
    block: "cmaps",
    heading: "CMAP",
};

/// A bonded label resolved to the style and type that define it.
struct Resolved<'f> {
    /// The category the label is of.
    kind: Bonded,
    /// 1-based type id (label order).
    id: usize,
    /// The label as written: the type's name.
    label: String,
    style: &'f Style,
    ty: &'f TypeDef,
}

impl Resolved<'_> {
    /// The LAMMPS style name and the coefficient lines of the resolved type;
    /// an error names the label.
    fn coeffs(
        &self,
        reg: &Registry,
        units: &UnitScale,
    ) -> Result<(String, LammpsCoeffs), ForceFieldWriteError> {
        let at = |e: ForceFieldWriteError| {
            e.context(format_args!("{} label `{}`", self.kind.block, self.label))
        };
        let (spec, codec) = codec_of(reg, self.kind.category, self.style.name()).map_err(at)?;
        codec
            .check_style(spec, self.style.params())
            .map_err(|e| at(e.into()))?;
        let coeffs = codec
            .write(spec, &self.ty.params, units)
            .map_err(|e| at(e.into()))?;
        Ok((lammps_name(spec), coeffs))
    }
}

/// A pair coefficient row: the ids of its two atom labels and the labels
/// (lower id first: LAMMPS sets nothing for `pair_coeff I J` with `I > J`),
/// and the pair type that defines it.
struct PairRow<'f> {
    ids: (usize, usize),
    labels: (&'f str, &'f str),
    style: &'f Style,
    ty: &'f TypeDef,
}

impl PairRow<'_> {
    /// The pair coefficients in file units, through the style's codec. A
    /// style without a LAMMPS form is an error naming the style and type:
    /// skipping it would leave the `pair_style` line without its
    /// `pair_coeff` rows. Type-less styles (`coul/cut`) produce no rows and
    /// never reach here.
    fn coeffs(
        &self,
        reg: &Registry,
        units: &UnitScale,
        precision: usize,
    ) -> Result<String, ForceFieldWriteError> {
        let at = |e: ForceFieldWriteError| e.context(format_args!("pair type `{}`", self.ty.name));
        let (spec, codec) = codec_of(reg, "pair", self.style.name()).map_err(at)?;
        let coeffs = codec
            .write(spec, &self.ty.params, units)
            .map_err(|e| at(e.into()))?;
        Ok(render(&coeffs.values, precision))
    }
}

/// Label-driven LAMMPS coefficient writer (AMBER/GAFF flavour).
///
/// Holds the system's [`TypeLabels`]. [`ForceFieldWriter::write`] writes the `*.ff` include,
/// [`LammpsForcefieldWriter::write_data_coeffs`] the data-file `* Coeffs`
/// sections; both write the same labels with the same numbers, each style
/// through its LAMMPS codec in the force field's registry
/// ([`ForceField::registry`](crate::ff::forcefield::ForceField::registry)).
///
/// # Coefficient writing, not whole-FF serialization
///
/// molrs has two kinds of force-field writer:
///
/// - **Coefficient writing** (this writer, LAMMPS only) answers "which
///   coefficients does this system's data file need". It is keyed by the
///   system's type labels ([`TypeLabels`]): one coefficient per label, in label
///   id order. A label the [`ForceField`] does not define is an error naming the
///   block and the label; a `ForceField` type no label uses is not written
///   (assembly retyping legitimately leaves stale types behind).
/// - **Whole-FF serialization** ([`crate::io::gromacs`], [`crate::io::openmm_xml`]) writes
///   every type the `ForceField` holds, as a force-field file, and takes no
///   labels.
///
/// The data-file writer and this writer are composed by the caller; neither
/// calls the other.
///
/// # Label matching
///
/// - `atoms` labels select pair coefficients: the self pair named by each label
///   (missing → error), plus explicit cross pairs whose two atom types are both
///   labels.
/// - `bonds`, `angles`, `dihedrals` and `impropers` labels match a
///   `ForceField` type name exactly: a label is the name of the type it
///   stands for, and `h1-c3` does not find a type named `c3-h1`.
/// - A block whose types carry no labels (pure-integer types) is matched by
///   its ids, `"1"`, `"2"`, ….
/// - A style is written only when it holds a used type; an unsupported style
///   is an error only then. Type-less pair styles (`coul/cut`) apply to every
///   atom and are always in play.
///
/// # Coefficients through the styles' codecs
///
/// Inverse of [`LammpsForcefieldReader`](crate::io::lammps::LammpsForcefieldReader). The
/// force-field IR follows the LAMMPS standard — every style's expression,
/// factors and parameter units, with angle-valued parameters in degrees — so
/// a coefficient is written as it is stored, by the style's
/// [`LammpsForm`](crate::ff::ir::LammpsForm) in the registry
/// (`ff-ir-02-protocol` §8): a positional style writes its spec's `params`
/// in order, a style whose line is not positional (`fourier`, `nharmonic`,
/// `lj/charmm`, the `class2` cross-term lines, …) its own codec, and a style
/// registered at run time with a LAMMPS form is written with nothing else
/// added. A style without one is refused by name
/// ([`IrError::NoEngineForm`](crate::ff::ir::IrError::NoEngineForm)):
///
/// ```text
/// pair_style lj/cut/coul/cut 10.0 10.0
/// pair_coeff c3 c3 0.107800 3.397710          # epsilon sigma
/// bond_style harmonic
/// bond_coeff c3-c3 228.890000 1.535400        # k r0
/// angle_style harmonic
/// angle_coeff c3-c3-oh 76.790000 109.660000   # k theta0(deg)
/// dihedral_style fourier
/// dihedral_coeff c3-c3-oh-ho 1 0.060000 3 0.0 # m  k1 periodicity1 phase1 ...
/// ```
///
/// The file is written in the writer's units
/// ([`with_units`](LammpsForcefieldWriter::with_units), default `real`).
/// A force field declared in those units ([`ForceField::units`]) is written
/// number for number; one declared in another LAMMPS unit style (`real`,
/// `metal`, `lj`) has every parameter converted by its dimension
/// ([`ParamDimension`](crate::ff::ir::ParamDimension), [`UnitScale`]) through
/// [`LammpsUnitConverter`] (`from → lj hub → to`) — never ad-hoc eV/kcal factors, never per style.
/// Angle values need no conversion in any unit style.
///
/// Two styles are written under another LAMMPS name: molrs's `dihedral
/// periodic` is LAMMPS's `fourier`, term for term, and AMBER's `improper
/// periodic` with one term at phase 0° or 180° is LAMMPS's `cvff`
/// (`d = cos phase`) — the atom order needs no change, because both price the
/// dihedral I-J-K-L of the stored order (see `improper::periodic`).
///
/// A category whose used types span several LAMMPS styles (say `angle
/// harmonic` and `angle charmm`) is written as one `angle_style hybrid
/// harmonic charmm` line, each `angle_coeff` naming its sub-style; a data
/// file's section is `Angle Coeffs # hybrid` with the sub-style on each row.
/// Every data-file section names its style in the header comment, as LAMMPS's
/// `write_data` does.
///
/// # Drude oscillators (DRUDE package)
///
/// A `drude harmonic` type (molrs's `k·r²` core–shell spring) is written as a
/// `bond harmonic` type of the same name with `K = k`, `r0 = 0` — LAMMPS's
/// harmonic bond is the same unhalved `K (r − r0)²` — so its label, which
/// [`lammps_type_labels`](crate::io::lammps::lammps_type_labels) numbers among the bonds as the data writer does,
/// gets its `bond_coeff` / `Bond Coeffs` row. The springs' `alpha` and
/// `thole` are not LAMMPS bond coefficients: the include lists them in a
/// comment after the bond section, for the input script's `pair_style
/// thole` and `fix drude`, which this writer does not write (no LAMMPS form is
/// registered for `pair thole`). Build the writer's labels with
/// [`lammps_type_labels`](crate::io::lammps::lammps_type_labels); [`TypeLabels::from_frame`] leaves the Drude
/// springs out of the bond types.
///
/// # CMAP crossterms (`fix cmap`)
///
/// A `cmaps` block's labels select `cmap charmm` rows the same way, in id
/// order: [`LammpsForcefieldWriter::write_cmap`] writes their grids as the
/// `fix cmap` file (map `t` is crossterm type `t`, the id the data writer
/// gives the `CMAP` section), and the include names that file
/// ([`with_cmap_file`](LammpsForcefieldWriter::with_cmap_file)) on a
/// `fix cmap all cmap <file>` line, with `fix_modify cmap energy yes` so the crossterms count in `pe`. LAMMPS
/// reads the crossterms with the data file, so the fix must precede
/// `read_data <data> fix cmap crossterm CMAP` (LAMMPS takes it before the box
/// exists); the include writes it first, beside `units`.
///
/// # Pair style layout
///
/// The reader splits a combined `lj/cut/coul/*` kernel into `lj/cut` +
/// `coul/cut` styles. This writer recombines that pair into one
/// `pair_style lj/cut/coul/cut` line so LAMMPS keeps geometric mixing on LJ (writing them as `hybrid` with a
/// `pair_coeff * * coul/cut` wildcard marks every cross pair as explicit and
/// defeats mixing). A force field that already holds a single combined-style
/// name, or only one of the two halves, is written as-is. `lj/charmm` +
/// `coul/charmm` is `pair_style lj/charmm/coul/charmm`, its only LAMMPS
/// spelling, `pair_coeff i j epsilon sigma epsilon14 sigma14`.
///
/// # Per-pair 1-4 overrides
///
/// LAMMPS has no per-pair exception. A frame whose `pairs` block carries an
/// override column ([`PAIR_OVERRIDE_COLUMNS`](molrs::core::schema::PAIR_OVERRIDE_COLUMNS))
/// is refused by name — by the data-file writer, and by
/// [`refuse_inexpressible_pairs`] for a caller writing a force field for it.
#[derive(Debug, Clone)]
pub struct LammpsForcefieldWriter<'a> {
    labels: &'a TypeLabels,
    options: Settings,
}

/// What [`LammpsForcefieldWriter`]'s setters set.
#[derive(Debug, Clone)]
struct Settings {
    precision: usize,
    skip_pair_style: bool,
    skip_special_bonds: bool,
    skip_units: bool,
    units: &'static str,
    cmap_file: Option<String>,
}

impl<'a> LammpsForcefieldWriter<'a> {
    /// Writer for `labels`: 6 decimal places, `real` units, every line of
    /// the include written, no `fix cmap` file.
    pub fn new(labels: &'a TypeLabels) -> Self {
        Self {
            labels,
            options: Settings {
                precision: 6,
                skip_pair_style: false,
                skip_special_bonds: false,
                skip_units: false,
                units: "real",
                cmap_file: None,
            },
        }
    }

    /// Decimal places for floating-point coefficients (default 6).
    pub fn with_precision(mut self, precision: usize) -> Self {
        self.options.precision = precision;
        self
    }

    /// The LAMMPS `units` style the file is written in (default `"real"`),
    /// as [`parse_lammps_units_style`] names it.
    pub fn with_units(mut self, units: &'static str) -> Self {
        self.options.units = units;
        self
    }

    /// Omit the `pair_style` line (default `false`): the input script sets
    /// its own (a relaxation's `lj/cut/coul/cut 10.0`), before the include.
    /// Only the line is the caller's; what the force field says about its
    /// pairs stays in the include — the `pair_coeff`s, `special_bonds`
    /// (unless [`with_skip_special_bonds`](Self::with_skip_special_bonds))
    /// and `pair_modify mix` / `shift`, without which LAMMPS would mix unlike
    /// pairs by its own default (`geometric` for `lj/cut`) and weight 1-4
    /// pairs by its own (`0 0 0`). `pair_modify` needs a pair style, so such
    /// an include is read after the caller's `pair_style`.
    pub fn with_skip_pair_style(mut self, skip: bool) -> Self {
        self.options.skip_pair_style = skip;
        self
    }

    /// Omit `special_bonds` (default `false`): the input script states its
    /// own 1-4 weights. An include that still wrote them would override a
    /// `special_bonds` read before it (Amber's coul 1-4 = 1/1.2 over the
    /// input's 0.5) — so a caller that writes its own says so here.
    pub fn with_skip_special_bonds(mut self, skip: bool) -> Self {
        self.options.skip_special_bonds = skip;
        self
    }

    /// Omit the `units` line (default `false`), so the include can follow
    /// `units` / `atom_style` / `pair_style` in the input (LAMMPS rejects
    /// `units` after the box exists, and a second `units` is redundant).
    pub fn with_skip_units(mut self, skip: bool) -> Self {
        self.options.skip_units = skip;
        self
    }

    /// The `fix cmap` file the include names, as the input script will find
    /// it — where [`write_cmap`](Self::write_cmap) writes. Needed exactly
    /// when the system has CMAP crossterms (default: none).
    pub fn with_cmap_file(mut self, file: impl Into<String>) -> Self {
        self.options.cmap_file = Some(file.into());
        self
    }

    /// Write data-file `* Coeffs` sections only (no `units` / `*_style`
    /// lines) into `sink` — a fragment of a data file, for the caller to
    /// splice in; nothing is written when it is refused.
    ///
    /// Rows are the labels' 1-based ids in [`TypeLabels`] order, with the same
    /// numbers, codecs and units ([`with_units`](Self::with_units)) as
    /// [`ForceFieldWriter::write`]. `Pair Coeffs` holds self pairs only, so
    /// a used explicit cross pair is an error here (write it through the
    /// include). A style's cross-term lines (`class2`'s `bb`, `mbt`, …) are
    /// their sections (`BondBond Coeffs`, `MiddleBondTorsion Coeffs`, …).
    ///
    /// # Errors
    ///
    /// What the include refuses, a used explicit cross pair, and the sink's
    /// OS error.
    pub fn write_data_coeffs<'s>(
        &self,
        sink: impl Into<Sink<'s>>,
        ff: &ForceField,
    ) -> Result<(), ForceFieldWriteError> {
        put_text(sink.into(), &self.data_coeffs_text(ff)?)
    }

    /// The sections [`write_data_coeffs`](Self::write_data_coeffs) writes.
    fn data_coeffs_text(&self, ff: &ForceField) -> Result<String, ForceFieldWriteError> {
        let units = units_of(ff, self.options.units)?;
        ff.registry().with(|reg| {
            let ff = &*drude_springs_as_bonds(ff)?;
            refuse_other_categories(ff, reg)?;
            let mut lines: Vec<String> = Vec::new();
            self.write_data_pair_coeffs(&mut lines, ff, reg, &units)?;
            self.write_data_section(&mut lines, BOND, ff, reg, &units)?;
            self.write_data_section(&mut lines, ANGLE, ff, reg, &units)?;
            self.write_data_section(&mut lines, DIHEDRAL, ff, reg, &units)?;
            self.write_data_section(&mut lines, IMPROPER, ff, reg, &units)?;
            Ok(lines.concat())
        })
    }

    /// Write the LAMMPS `fix cmap` file of the `cmaps` labels into `sink`:
    /// the grid of the `cmap` row each label names, in label id order, so map
    /// `t` is the crossterm type `t` the data writer gives the `CMAP` section;
    /// the inverse of [`LammpsForcefieldReader::read_cmap`](crate::io::lammps::LammpsForcefieldReader::read_cmap).
    /// The layout is CHARMM's: a first line with the `UNITS:` tag LAMMPS
    /// checks, then per map a `# <label>, type <t>` comment and, per φ row, a
    /// `# <φ>` comment over its values, five a line, each with
    /// [`with_precision`](Self::with_precision) decimals as CHARMM's `%13.6f`
    /// is. The grid is converted to the writer's units
    /// ([`with_units`](Self::with_units)) by its dimension, as every
    /// coefficient is.
    ///
    /// # Errors
    ///
    /// No `cmaps` label, a label with no cmap type, a style whose LAMMPS form
    /// is not `fix cmap`, a row without a `grid`, a grid that is not 24×24,
    /// or more than six maps (LAMMPS's `CMAPDIM`, `CMAPMAX`); the sink's OS
    /// error.
    pub fn write_cmap<'s>(
        &self,
        sink: impl Into<Sink<'s>>,
        ff: &ForceField,
    ) -> Result<(), ForceFieldWriteError> {
        put_text(sink.into(), &self.cmap_text(ff)?)
    }

    /// The file [`write_cmap`](Self::write_cmap) writes.
    fn cmap_text(&self, ff: &ForceField) -> Result<String, ForceFieldWriteError> {
        let units = units_of(ff, self.options.units)?;
        let rows = self.resolve(ff, CMAP)?;
        if rows.is_empty() {
            return Err("cmaps: the system has no CMAP crossterm labels".into());
        }
        if rows.len() > LAMMPS_CMAP_MAX {
            return Err(format!(
                "cmaps: {} CMAP types, fix cmap reads at most {LAMMPS_CMAP_MAX}",
                rows.len()
            )
            .into());
        }
        let mut maps = Vec::with_capacity(rows.len());
        for r in &rows {
            let what = || format!("cmaps label `{}`", r.label);
            let dim = ff.registry().with(|reg| {
                let (spec, _) =
                    codec_of(reg, "cmap", r.style.name()).map_err(|e| e.context(what()))?;
                if lammps_name(spec) != "cmap" {
                    return Err(format!(
                        "{}: cmap style `{}` has no fix cmap form",
                        what(),
                        r.style.name()
                    ));
                }
                spec.param("grid")
                    .map(|p| p.dim)
                    .ok_or_else(|| format!("{}: its spec declares no `grid`", what()))
            })?;
            let grid =
                r.ty.params
                    .get("grid")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| format!("{}: no `grid`", what()))?;
            if grid.shape() != [LAMMPS_CMAP_DIM, LAMMPS_CMAP_DIM] {
                return Err(format!(
                    "{}: a {:?} grid; fix cmap reads {LAMMPS_CMAP_DIM}×{LAMMPS_CMAP_DIM}",
                    what(),
                    grid.shape()
                )
                .into());
            }
            let converted = grid.mapv(|v| units.apply(v, dim));
            maps.push((r.label.clone(), converted));
        }
        let titled: Vec<(&str, &ArrayD<f64>)> =
            maps.iter().map(|(label, g)| (label.as_str(), g)).collect();
        cmap_file_text(&titled, self.options.units, self.options.precision)
    }

    /// The include's `fix cmap` lines when the system has CMAP crossterms,
    /// nothing otherwise.
    fn write_cmap_fix(
        &self,
        lines: &mut Vec<String>,
        ff: &ForceField,
    ) -> Result<(), ForceFieldWriteError> {
        if self.resolve(ff, CMAP)?.is_empty() {
            return Ok(());
        }
        let file = self.options.cmap_file.as_deref().ok_or(
            "the system has CMAP crossterms: name the file LammpsForcefieldWriter::write_cmap \
             writes with LammpsForcefieldWriter::with_cmap_file",
        )?;
        if file.is_empty() || file.chars().any(char::is_whitespace) {
            return Err(
                format!("cmap_file {file:?}: LAMMPS reads one word as the fix cmap file").into(),
            );
        }
        lines.push(
            "# CMAP crossterms: this fix must precede `read_data <data> fix cmap crossterm CMAP`\n"
                .to_owned(),
        );
        lines.push(format!("fix {CMAP_FIX_ID} all cmap {file}\n"));
        lines.push(format!("fix_modify {CMAP_FIX_ID} energy yes\n"));
        lines.push("\n".to_owned());
        Ok(())
    }

    /// Labels of `block` in id order; the ids themselves (`"1"`, `"2"`, …)
    /// when the block's types carry no labels.
    fn block_labels(&self, block: &str) -> Vec<String> {
        match self.labels.block(block) {
            None => Vec::new(),
            Some(types) => match types.labels() {
                Some(labels) => labels.to_vec(),
                None => (1..=types.n_types()).map(|id| id.to_string()).collect(),
            },
        }
    }

    /// Every label of `kind`'s block resolved to its type, in id order. The
    /// first style (in force-field order) defining a name wins.
    fn resolve<'f>(
        &self,
        ff: &'f ForceField,
        kind: Bonded,
    ) -> Result<Vec<Resolved<'f>>, ForceFieldWriteError> {
        let labels = self.block_labels(kind.block);
        if labels.is_empty() {
            return Ok(Vec::new());
        }
        let mut by_name: HashMap<&'f str, (&'f Style, &'f TypeDef)> = HashMap::new();
        for style in ff.get_styles(kind.category) {
            for ty in style.defs().types() {
                by_name.entry(ty.name.as_str()).or_insert((style, ty));
            }
        }
        labels
            .iter()
            .enumerate()
            .map(|(i, label)| {
                let &(style, ty) = by_name.get(label.as_str()).ok_or_else(|| {
                    let mut message = format!(
                        "{}: type label `{label}` has no {} type in the force field",
                        kind.block, kind.category
                    );
                    // A label matches a name exactly, orientation included; the
                    // likeliest cause is the other spelling, so name it when it
                    // is the one defined.
                    let other_way = by_name.values().find(|(_, ty)| {
                        let reversed: Vec<&str> =
                            ty.endpoints.iter().rev().map(String::as_str).collect();
                        TypeName::join(&reversed).is_ok_and(|n| n.as_str() == label)
                    });
                    if let Some((_, defined)) = other_way {
                        message.push_str(&format!(
                            " (`{}` is defined on the same atom types read the other way; a \
                             label matches a type name exactly, orientation included)",
                            defined.name
                        ));
                    }
                    message
                })?;
                Ok(Resolved {
                    kind,
                    id: i + 1,
                    label: label.clone(),
                    style,
                    ty,
                })
            })
            .collect()
    }

    /// Pair rows for the `atoms` labels, ordered by id pair: every label's
    /// self pair (missing → error) and every explicit cross pair of two used
    /// labels. The first style (in force-field order) defining a pair wins.
    ///
    /// A force field with no typed pair style (bonded-only, or only type-less
    /// styles such as Coulomb) has no pair rows to write, so it yields none
    /// rather than demanding a self pair per label.
    fn resolve_pairs<'f>(
        &self,
        ff: &'f ForceField,
    ) -> Result<Vec<PairRow<'f>>, ForceFieldWriteError> {
        if !ff.get_styles("pair").iter().any(|s| typed(s)) {
            return Ok(Vec::new());
        }
        let labels = self.block_labels("atoms");
        let ids: HashMap<&str, usize> = labels
            .iter()
            .enumerate()
            .map(|(i, label)| (label.as_str(), i + 1))
            .collect();
        let mut rows: BTreeMap<(usize, usize), PairRow<'f>> = BTreeMap::new();
        for style in ff.get_styles("pair") {
            for ty in style.defs().types() {
                let (Some(&i), Some(&j)) = (
                    ids.get(ty.endpoints[0].as_str()),
                    ids.get(ty.endpoints[1].as_str()),
                ) else {
                    continue;
                };
                let (key, labels) = if i <= j {
                    ((i, j), (ty.endpoints[0].as_str(), ty.endpoints[1].as_str()))
                } else {
                    ((j, i), (ty.endpoints[1].as_str(), ty.endpoints[0].as_str()))
                };
                rows.entry(key).or_insert(PairRow {
                    ids: key,
                    labels,
                    style,
                    ty,
                });
            }
        }
        if let Some((_, label)) = labels
            .iter()
            .enumerate()
            .find(|(i, _)| !rows.contains_key(&(i + 1, i + 1)))
        {
            return Err(
                format!("atoms: type label `{label}` has no pair type in the force field").into(),
            );
        }
        Ok(rows.into_values().collect())
    }

    /// The `pair_style` (and `pair_modify`) lines and the `pair_coeff`
    /// lines of the styles in play, each through its codec.
    ///
    /// LAMMPS's combined styles are written as such: `lj/charmm` +
    /// `coul/charmm` is `lj/charmm/coul/charmm` (its only spelling), and
    /// `lj/cut` + `coul/cut` (`coul/long/pme`) is `lj/cut/coul/cut`
    /// (`lj/cut/coul/long`), which keeps LAMMPS's mixing on the LJ rows (a
    /// `hybrid` with a `pair_coeff * * coul/cut` wildcard marks every cross
    /// pair as explicit). One style is `pair_style <name> <cutoff>`; one
    /// typed style beside a Coulomb style is a `hybrid/overlay` of the two,
    /// the Coulomb one on `* *` — refused when the typed style would mix an
    /// unlike pair, which the overlay leaves unmixed; several typed styles
    /// are a `hybrid`.
    fn write_pair_section(
        &self,
        lines: &mut Vec<String>,
        ff: &ForceField,
        reg: &Registry,
        units: &UnitScale,
    ) -> Result<(), ForceFieldWriteError> {
        let rows = self.resolve_pairs(ff)?;
        if rows.is_empty() {
            return Ok(());
        }
        // Styles in play: those defining a used pair, plus type-less ones
        // (Coulomb), which apply to every atom.
        let styles: Vec<&Style> = ff
            .get_styles("pair")
            .into_iter()
            .filter(|s| !typed(s) || rows.iter().any(|r| std::ptr::eq(r.style, *s)))
            .collect();
        let mut codecs: Vec<(&StyleSpec, &dyn LammpsCodec)> = Vec::new();
        for style in &styles {
            let (spec, codec) = codec_of(reg, "pair", style.name())?;
            codec.check_style(spec, style.params())?;
            codecs.push((spec, codec));
        }
        let opts = &self.options;
        let args = |i: usize| -> Result<Vec<Token>, ForceFieldWriteError> {
            let (spec, codec) = codecs[i];
            Ok(codec.style_args(spec, styles[i].params(), units)?)
        };
        let modify = |i: usize| -> Result<Option<String>, ForceFieldWriteError> {
            let (spec, codec) = codecs[i];
            let keys = codec.pair_modify(spec, styles[i].params())?;
            Ok((!keys.is_empty()).then(|| format!("pair_modify {}\n", keys.join(" "))))
        };
        let names: HashSet<&str> = styles.iter().map(|s| s.name()).collect();
        let find = |name: &str| styles.iter().position(|s| s.name() == name);

        // lj/charmm + coul/charmm is LAMMPS's one `lj/charmm/coul/charmm`; it
        // has no other spelling (no `hybrid` sub-style is either half).
        if names.contains("lj/charmm") || names.contains("coul/charmm") {
            if names != HashSet::from(["lj/charmm", "coul/charmm"]) {
                let mut names: Vec<&str> = names.into_iter().collect();
                names.sort_unstable();
                return Err(format!(
                    "pair styles {names:?}: LAMMPS has lj/charmm only as \
                     `lj/charmm/coul/charmm`, the pair lj/charmm + coul/charmm and nothing \
                     beside it"
                )
                .into());
            }
            let (lj, coul) = (find("lj/charmm").unwrap(), find("coul/charmm").unwrap());
            let style_line = if opts.skip_pair_style {
                None
            } else {
                let (lj_cuts, coul_cuts) = (args(lj)?, args(coul)?);
                let mut cuts = lj_cuts.clone();
                if coul_cuts != lj_cuts {
                    cuts.extend(coul_cuts);
                }
                Some(format!(
                    "pair_style {} {}\n",
                    lammps_name(codecs[lj].0),
                    render(&cuts, opts.precision)
                ))
            };
            push_pair_header(lines, style_line, modify(lj)?);
            return self.push_pair_coeffs(lines, reg, &rows, false, units);
        }

        // lj/cut + its Coulomb: LAMMPS's combined `lj/cut/coul/<cut|long>`.
        if let (2, Some(lj), Some(coul)) = (
            styles.len(),
            find("lj/cut"),
            find("coul/cut").or_else(|| find("coul/long/pme")),
        ) {
            let style_line = if opts.skip_pair_style {
                None
            } else {
                let mut cuts = args(lj)?;
                cuts.extend(args(coul)?);
                Some(format!(
                    "pair_style lj/cut/{} {}\n",
                    lammps_name(codecs[coul].0),
                    render(&cuts, opts.precision)
                ))
            };
            push_pair_header(lines, style_line, modify(lj)?);
            return self.push_pair_coeffs(lines, reg, &rows, false, units);
        }

        if styles.len() == 1 {
            let style_line = if opts.skip_pair_style {
                None
            } else {
                Some(format!(
                    "pair_style {}\n",
                    [lammps_name(codecs[0].0), render(&args(0)?, opts.precision)]
                        .join(" ")
                        .trim_end()
                ))
            };
            push_pair_header(lines, style_line, modify(0)?);
            return self.push_pair_coeffs(lines, reg, &rows, false, units);
        }

        let sub_style = |i: usize| -> Result<String, ForceFieldWriteError> {
            let cuts = args(i)?;
            let name = lammps_name(codecs[i].0);
            Ok(if cuts.is_empty() {
                name
            } else {
                format!("{name} {}", render(&cuts, opts.precision))
            })
        };
        let coulombs: Vec<usize> = (0..styles.len()).filter(|&i| !typed(styles[i])).collect();
        match coulombs.as_slice() {
            // Genuinely independent sub-styles → hybrid with per-substyle
            // cutoffs, each mixing by its own rule (`pair_modify pair <sub>`).
            [] => {
                let style_line = if opts.skip_pair_style {
                    None
                } else {
                    let subs = (0..styles.len())
                        .map(sub_style)
                        .collect::<Result<Vec<_>, _>>()?;
                    Some(format!("pair_style hybrid {}\n", subs.join(" ")))
                };
                let mut modifies = String::new();
                for i in 0..styles.len() {
                    let keys = codecs[i].1.pair_modify(codecs[i].0, styles[i].params())?;
                    if !keys.is_empty() {
                        modifies += &format!(
                            "pair_modify pair {} {}\n",
                            lammps_name(codecs[i].0),
                            keys.join(" ")
                        );
                    }
                }
                push_pair_header(
                    lines,
                    style_line,
                    (!modifies.is_empty()).then_some(modifies),
                );
                self.push_pair_coeffs(lines, reg, &rows, true, units)?;
            }
            // One typed style and a Coulomb style on every pair.
            &[c] if styles.len() == 2 => {
                let t = 1 - c;
                let (spec, _) = codecs[t];
                let mixes = spec.style_param("mixing").is_some()
                    || spec.params.iter().any(|p| p.mix != ParamCombination::None);
                let n = self.block_labels("atoms").len();
                let unlike = rows.iter().filter(|r| r.ids.0 != r.ids.1).count();
                if mixes && unlike != n * (n - 1) / 2 {
                    return Err(Engine::Lammps
                        .refuse(
                            "pair",
                            styles[t].name(),
                            format!(
                                "beside `{}` it is a hybrid/overlay sub-style, which LAMMPS \
                                 does not mix; state every unlike pair as a cross row",
                                styles[c].name()
                            ),
                        )
                        .into());
                }
                if !opts.skip_pair_style {
                    lines.push(format!(
                        "pair_style hybrid/overlay {} {}\n\n",
                        sub_style(t)?,
                        sub_style(c)?
                    ));
                }
                self.push_pair_coeffs(lines, reg, &rows, true, units)?;
                lines.push(format!("pair_coeff * * {}\n", lammps_name(codecs[c].0)));
            }
            _ => {
                let mut names: Vec<&str> = names.into_iter().collect();
                names.sort_unstable();
                return Err(format!(
                    "pair styles {names:?}: a Coulomb style beside several typed pair styles \
                     has no LAMMPS form this writer writes"
                )
                .into());
            }
        }
        lines.push("\n".to_owned());
        Ok(())
    }

    /// `pair_coeff` lines for the rows, naming the sub-style when `hybrid`. A
    /// single-style block ends with a blank line when non-empty.
    fn push_pair_coeffs(
        &self,
        lines: &mut Vec<String>,
        reg: &Registry,
        rows: &[PairRow<'_>],
        hybrid: bool,
        units: &UnitScale,
    ) -> Result<(), ForceFieldWriteError> {
        for row in rows {
            let nums = row.coeffs(reg, units, self.options.precision)?;
            let (i, j) = row.labels;
            if hybrid {
                let (spec, _) = codec_of(reg, "pair", row.style.name())?;
                lines.push(format!("pair_coeff {i} {j} {} {nums}\n", lammps_name(spec)));
            } else {
                lines.push(format!("pair_coeff {i} {j} {nums}\n"));
            }
        }
        if !rows.is_empty() && !hybrid {
            lines.push("\n".to_owned());
        }
        Ok(())
    }

    /// The `T_style` line and `T_coeff` lines for the used labels, in label id
    /// order. When the labels' types belong to more than one LAMMPS style the
    /// line is `T_style hybrid <sub-style>…` and each coefficient line names
    /// its sub-style, as LAMMPS reads them. A style's cross-term lines follow
    /// its type's line.
    fn write_section(
        &self,
        lines: &mut Vec<String>,
        kind: Bonded,
        ff: &ForceField,
        reg: &Registry,
        units: &UnitScale,
    ) -> Result<(), ForceFieldWriteError> {
        let used = self.resolve(ff, kind)?;
        if used.is_empty() {
            return Ok(());
        }
        let coeffs = used
            .iter()
            .map(|r| r.coeffs(reg, units))
            .collect::<Result<Vec<_>, _>>()?;
        let subs = distinct(coeffs.iter().map(|(name, _)| name.as_str()));
        let hybrid = subs.len() > 1;
        let p = self.options.precision;
        lines.push(format!("{}_style {}\n", kind.category, style_line(&subs)));
        for (r, (name, c)) in used.iter().zip(&coeffs) {
            let sub = if hybrid {
                format!("{name} ")
            } else {
                String::new()
            };
            lines.push(format!(
                "{}_coeff {} {sub}{}\n",
                kind.category,
                r.label,
                render(&c.values, p)
            ));
            for (keyword, values) in &c.extra {
                lines.push(format!(
                    "{}_coeff {} {sub}{keyword} {}\n",
                    kind.category,
                    r.label,
                    render(values, p)
                ));
            }
        }
        lines.push("\n".to_owned());
        Ok(())
    }

    fn write_data_pair_coeffs(
        &self,
        lines: &mut Vec<String>,
        ff: &ForceField,
        reg: &Registry,
        units: &UnitScale,
    ) -> Result<(), ForceFieldWriteError> {
        let mut section = Vec::new();
        for row in self.resolve_pairs(ff)? {
            if row.ids.0 != row.ids.1 {
                return Err(format!(
                    "pair type `{}` is an explicit cross pair; a data-file Pair Coeffs \
                     section holds self pairs only (write it through the *.ff include)",
                    row.ty.name
                )
                .into());
            }
            let nums = row.coeffs(reg, units, self.options.precision)?;
            section.push(format!("{} {nums}\n", row.ids.0));
        }
        push_data_section(lines, "Pair Coeffs", section);
        Ok(())
    }

    fn write_data_section(
        &self,
        lines: &mut Vec<String>,
        kind: Bonded,
        ff: &ForceField,
        reg: &Registry,
        units: &UnitScale,
    ) -> Result<(), ForceFieldWriteError> {
        let used = self.resolve(ff, kind)?;
        let coeffs = used
            .iter()
            .map(|r| r.coeffs(reg, units))
            .collect::<Result<Vec<_>, _>>()?;
        let subs = distinct(coeffs.iter().map(|(name, _)| name.as_str()));
        let hybrid = subs.len() > 1;
        let p = self.options.precision;
        let mut section = Vec::new();
        let mut cross: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        for (r, (name, c)) in used.iter().zip(&coeffs) {
            let sub = if hybrid {
                format!("{name} ")
            } else {
                String::new()
            };
            section.push(format!("{} {sub}{}\n", r.id, render(&c.values, p)));
            for (keyword, values) in &c.extra {
                if hybrid {
                    return Err(format!(
                        "{} label `{}`: its `{keyword}` cross-term line in a hybrid data-file \
                         section has no form this writer writes (write the *.ff include)",
                        kind.block, r.label
                    )
                    .into());
                }
                cross
                    .entry(keyword)
                    .or_default()
                    .push(format!("{} {}\n", r.id, render(values, p)));
            }
        }
        // The `# style` comment is how `read_data_coeffs` (and LAMMPS's own
        // `write_data`) knows which style the numbers are; without it a reader
        // falls back to `harmonic`.
        let heading = format!("{} # {}", kind.heading, style_hint(&subs));
        push_data_section(lines, &heading, section);
        for (heading, category, keyword) in CROSS_TERM_SECTIONS {
            if category == kind.category
                && let Some(rows) = cross.remove(keyword)
            {
                push_data_section(lines, heading, rows);
            }
        }
        Ok(())
    }
}

/// A pair section's header: the `pair_style` line (absent when the caller
/// writes its own) and the `pair_modify` lines, then the blank line that ends
/// it when it has any.
fn push_pair_header(lines: &mut Vec<String>, style_line: Option<String>, modify: Option<String>) {
    let any = style_line.is_some() || modify.is_some();
    lines.extend(style_line);
    lines.extend(modify);
    if any {
        lines.push("\n".to_owned());
    }
}

impl LammpsForcefieldWriter<'_> {
    /// The file's text: what [`write`](ForceFieldWriter::write) puts into its sink.
    fn text(&self, ff: &ForceField) -> Result<String, ForceFieldWriteError> {
        let units = units_of(ff, self.options.units)?;
        let mut lines: Vec<String> = Vec::new();
        lines.push("# LAMMPS force field generated by molrs\n".to_owned());
        if !self.options.skip_units {
            lines.push(format!("units {}\n", self.options.units));
            lines.push("\n".to_owned());
        }
        self.write_cmap_fix(&mut lines, ff)?;
        // The force field's 1-4 weights, unless the input script states its
        // own (`skip_special_bonds`): emitted over an input's 0.5, Amber's
        // 1/SCEE is what made PEO-Tg jobs run coul 1-4 = 0.8333. Skipping the
        // `pair_style` line does not skip them — LAMMPS's default is `0 0 0`.
        // A field with no pair style has no pair weights to state.
        let weights = if ff.get_styles("pair").is_empty() {
            ff.special_bonds()
        } else {
            Some(ff.required_special_bonds()?)
        };
        if let (false, Some(sb)) = (self.options.skip_special_bonds, weights) {
            let p = self.options.precision;
            lines.push(format!(
                "special_bonds lj {} {} {} coul {} {} {}\n",
                fmt_num(sb.lj[0], p),
                fmt_num(sb.lj[1], p),
                fmt_num(sb.lj[2], p),
                fmt_num(sb.coul[0], p),
                fmt_num(sb.coul[1], p),
                fmt_num(sb.coul[2], p),
            ));
            lines.push("\n".to_owned());
        }

        ff.registry().with(|reg| {
            let springs = drude_springs_as_bonds(ff)?;
            refuse_other_categories(&springs, reg)?;
            self.write_pair_section(&mut lines, &springs, reg, &units)?;
            self.write_section(&mut lines, BOND, &springs, reg, &units)?;
            lines.extend(drude_note(
                ff,
                &self.block_labels(BOND.block),
                reg,
                &units,
                self.options.precision,
            )?);
            let ff = &*springs;
            self.write_section(&mut lines, ANGLE, ff, reg, &units)?;
            self.write_section(&mut lines, DIHEDRAL, ff, reg, &units)?;
            self.write_section(&mut lines, IMPROPER, ff, reg, &units)
        })?;
        Ok(lines.concat())
    }
}

impl ForceFieldWriter for LammpsForcefieldWriter<'_> {
    fn write<'a>(
        &self,
        sink: impl Into<Sink<'a>>,
        ff: &ForceField,
    ) -> Result<(), ForceFieldWriteError> {
        put_text(sink.into(), &self.text(ff)?)
    }
}

/// Refuse a style with types in a category that prices energy and has no
/// LAMMPS `*_style` command (a run-time category): writing the include
/// without it would drop its energy. A `drude harmonic` spring has been
/// moved to the bonds by then ([`drude_springs_as_bonds`]). A category that
/// prices none (`constraint`, `virtual_site`) is the input script's (`fix
/// shake`, a `*/tip4p/*` pair style, …).
fn refuse_other_categories(ff: &ForceField, reg: &Registry) -> Result<(), ForceFieldWriteError> {
    const WRITTEN: [&str; 7] = [
        "atom", "pair", "bond", "angle", "dihedral", "improper", "cmap",
    ];
    for style in ff.styles() {
        let category = style.category();
        if WRITTEN.contains(&category) || style.type_rows().is_empty() {
            continue;
        }
        if reg.category(category).is_some_and(|c| !c.prices_energy()) {
            continue;
        }
        return Err(Engine::Lammps
            .refuse(
                category,
                style.name(),
                format!("LAMMPS has no `{category}_style` command"),
            )
            .into());
    }
    Ok(())
}

/// Whether a pair style holds types (a Coulomb style holds none).
fn typed(style: &Style) -> bool {
    !style.defs().types().is_empty()
}

/// The distinct names, in first-use order. Two molrs styles LAMMPS spells
/// alike (`improper cvff` and `improper periodic`, both `cvff`) are one
/// LAMMPS style with one coefficient form.
fn distinct<'s>(names: impl Iterator<Item = &'s str>) -> Vec<&'s str> {
    let mut subs: Vec<&str> = Vec::new();
    for name in names {
        if !subs.contains(&name) {
            subs.push(name);
        }
    }
    subs
}

/// The argument of a `*_style` line for `subs`: the one style, or `hybrid`
/// and every sub-style.
fn style_line(subs: &[&str]) -> String {
    match subs {
        [one] => (*one).to_owned(),
        _ => format!("hybrid {}", subs.join(" ")),
    }
}

/// The style a data-file `* Coeffs` header names: the one style, or `hybrid`.
fn style_hint<'s>(subs: &[&'s str]) -> &'s str {
    match subs {
        [one] => one,
        _ => "hybrid",
    }
}

/// `heading`, a blank line, the rows and a blank line; nothing when empty.
fn push_data_section(lines: &mut Vec<String>, heading: &str, rows: Vec<String>) {
    if rows.is_empty() {
        return;
    }
    lines.push(format!("{heading}\n\n"));
    lines.extend(rows);
    lines.push("\n".to_owned());
}

/// Refuse a frame whose special pairs LAMMPS cannot state: LAMMPS derives
/// them from the bonds and weighs them by `special_bonds` alone (see the
/// conventions guide, "1-4 interactions"), so the frame's `pairs` must be
/// exactly the 1-2, 1-3 and 1-4 pairs of its bonds, with no per-pair override
/// cell. The data-file writer refuses the same frames, and so does the OpenMM
/// force-field writer, whose `createSystem` derives them the same way; a
/// caller writing a force field for a frame checks the frame here.
pub fn refuse_inexpressible_pairs(frame: &impl molrs::core::FrameAccess) -> Result<(), String> {
    use molrs::core::ColumnView;
    if !frame.contains_block("pairs") {
        return Ok(());
    }
    let present: Vec<&str> = molrs::core::schema::PAIR_OVERRIDE_COLUMNS
        .iter()
        .copied()
        .filter(|k| frame.column("pairs", k).is_some())
        .collect();
    if !present.is_empty() {
        return Err(format!(
            "pairs: the per-pair override columns {present:?} have no LAMMPS form — LAMMPS \
             prices a 1-4 pair by special_bonds, lj/charmm's epsilon14/sigma14 and dihedral \
             charmm's w, never per pair"
        ));
    }
    let indices = |block: &str, key: &str| -> Vec<usize> {
        match frame.column(block, key) {
            Some(ColumnView::Uint(v)) => v.iter().map(|&x| x as usize).collect(),
            _ => Vec::new(),
        }
    };
    let n_atoms = frame
        .visit_block("atoms", |b| b.n_rows())
        .flatten()
        .unwrap_or(0);
    let mut topology = molrs::core::Topology::with_atoms(n_atoms);
    let edges: Vec<[usize; 2]> = indices("bonds", "atomi")
        .into_iter()
        .zip(indices("bonds", "atomj"))
        .map(|(a, b)| [a, b])
        .collect();
    topology.add_bonds(&edges);
    let close = topology.close_pairs(3);
    let (pi, pj, hops) = (
        indices("pairs", "atomi"),
        indices("pairs", "atomj"),
        indices("pairs", "n_bonds"),
    );
    let derivable = pi.len() == close.len()
        && hops.len() == pi.len()
        && (0..pi.len())
            .all(|r| close.get(&(pi[r].min(pj[r]), pi[r].max(pj[r]))) == Some(&hops[r]));
    if !derivable {
        return Err(format!(
            "pairs: {} rows that are not the {} pairs within three bonds of the frame's \
             bonds — LAMMPS (and an OpenMM force field) derives the special pairs from the \
             bonds and states no other",
            pi.len(),
            close.len()
        ));
    }
    Ok(())
}

pub(crate) fn cmap_file_text(
    maps: &[(&str, &ArrayD<f64>)],
    units: &str,
    precision: usize,
) -> Result<String, ForceFieldWriteError> {
    let width = precision + 7;
    let mut out = format!("# UNITS: {units} CMAP correction maps written by molrs\n");
    for (t, (title, grid)) in maps.iter().enumerate() {
        let n = match grid.shape() {
            [a, b] if a == b => *a,
            shape => return Err(format!("map `{title}`: a {shape:?} grid is not square").into()),
        };
        if title.contains(['\n', '\r']) {
            return Err(format!("map {}: its title holds a line break", t + 1).into());
        }
        out.push_str(&format!("\n# {title}, type {}\n", t + 1));
        let values: Vec<f64> = grid.iter().copied().collect();
        for (i, row) in values.chunks(n).enumerate() {
            let phi = -180.0 + 360.0 * i as f64 / n as f64;
            out.push_str(&format!("\n# {phi:.1}\n"));
            for line in row.chunks(5) {
                for (k, v) in line.iter().enumerate() {
                    if k > 0 {
                        out.push(' ');
                    }
                    out.push_str(&format!("{v:>width$.precision$}"));
                }
                out.push('\n');
            }
        }
    }
    Ok(out)
}

fn fmt_num(v: f64, precision: usize) -> String {
    format!("{v:.precision$}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::ir::Params;
    use crate::io::writer::WrittenText;

    /// What `writer` writes as a data file's `* Coeffs` sections.
    fn data_coeffs(
        writer: &LammpsForcefieldWriter<'_>,
        ff: &ForceField,
    ) -> Result<String, ForceFieldWriteError> {
        let mut out = Vec::new();
        writer.write_data_coeffs(&mut out, ff)?;
        Ok(String::from_utf8(out).expect("the writer writes UTF-8"))
    }

    /// What `writer` writes as the `fix cmap` file.
    fn cmap_file(
        writer: &LammpsForcefieldWriter<'_>,
        ff: &ForceField,
    ) -> Result<String, ForceFieldWriteError> {
        let mut out = Vec::new();
        writer.write_cmap(&mut out, ff)?;
        Ok(String::from_utf8(out).expect("the writer writes UTF-8"))
    }
    use crate::io::{lammps::LammpsForcefieldReader, reader::ForceFieldReader};

    /// Same GAFF2-shaped mini include the reader tests pin.
    const MINI: &str = r#"
# LAMMPS force field generated by molrs
special_bonds amber
pair_style lj/cut/coul/cut 10.0 10.0
pair_coeff c3 c3 0.107800 3.397710
pair_coeff oh oh 0.093000 3.242871
pair_coeff c3 c3 0.107800 3.397710

bond_style harmonic
bond_coeff c3-c3 228.890000 1.535400

angle_style harmonic
angle_coeff c3-c3-oh 76.790000 109.660000

dihedral_style fourier
dihedral_coeff c3-c3-oh-ho 1 0.060000 3 0.000000
"#;

    /// Labels of a system using every type in [`MINI`].
    fn mini_labels() -> TypeLabels {
        labels_of(&[
            ("atoms", &["c3", "c3", "oh"]),
            ("bonds", &["c3-c3"]),
            ("angles", &["c3-c3-oh"]),
            ("dihedrals", &["c3-c3-oh-ho"]),
        ])
    }

    /// A bonded-only field has no pair rows, and the writer demands no self
    /// pair for its atom labels.
    #[test]
    fn a_bonded_only_field_writes_without_pair_rows() {
        let mut ff = ForceField::new("bonded");
        ff.set_units("real");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "c3-c3",
                &["c3", "c3"],
                Params::from_iter([("k", 457.78), ("r0", 1.5354)]),
            )
            .unwrap();
        let labels = labels_of(&[("atoms", &["c3", "c3"]), ("bonds", &["c3-c3"])]);
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert!(text.contains("bond_coeff c3-c3"), "{text}");
        assert!(!text.contains("pair_coeff"), "{text}");
    }

    /// A pair style stating `cutoff = inf` (every pair priced) is refused by
    /// name: LAMMPS truncates every pair style.
    #[test]
    fn an_infinite_cutoff_is_refused() {
        let mut ff = split_pair_ff();
        ff.get_style_mut("pair", "lj/cut")
            .unwrap()
            .set_param("cutoff", f64::INFINITY);
        let labels = labels_of(&[("atoms", &["c3", "hc"])]);
        let err = LammpsForcefieldWriter::new(&labels)
            .written(&ff)
            .unwrap_err()
            .to_string();
        assert!(err.contains("lj/cut") && err.contains("finite"), "{err}");
    }

    #[test]
    fn writes_lammps_units_inverse_of_reader() {
        let ff = LammpsForcefieldReader::new().read(MINI.as_bytes()).unwrap();
        let labels = mini_labels();
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();

        // Combined pair style (not hybrid) so mixing stays intact.
        assert!(
            text.contains("pair_style lj/cut/coul/cut 10.000000 10.000000"),
            "combined pair_style:\n{text}"
        );
        assert!(!text.contains("hybrid"), "must not emit hybrid:\n{text}");
        assert!(
            text.contains("pair_coeff c3 c3 0.107800 3.397710"),
            "pair eps/sigma:\n{text}"
        );

        // The identity: the reader stored K = 228.89, the writer writes it.
        assert!(
            text.contains("bond_coeff c3-c3 228.890000 1.535400"),
            "bond K:\n{text}"
        );
        // angle: K=76.79, theta0 in degrees
        assert!(
            text.contains("angle_coeff c3-c3-oh 76.790000 109.660000"),
            "angle K + deg:\n{text}"
        );
        // fourier: m K n phase_deg. `n` is the cos(n*phi) multiplicity and
        // LAMMPS reads it with `inumeric()`, so it is written as an integer —
        // `3.000000` would be rejected by LAMMPS at parse time. It must also
        // stay in the `n` slot rather than sliding into the phase.
        assert!(
            text.contains("dihedral_coeff c3-c3-oh-ho 1 0.060000 3 0.000000"),
            "dihedral fourier:\n{text}"
        );
    }

    #[test]
    fn round_trip_preserves_molrs_params() {
        let ff = LammpsForcefieldReader::new().read(MINI.as_bytes()).unwrap();
        let labels = mini_labels();
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        let ff2 = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();

        let bt = ff2
            .get_style("bond", "harmonic")
            .unwrap()
            .get_bondtype("c3", "c3")
            .unwrap();
        assert_eq!(bt.params.get("k").and_then(|v| v.as_num()), Some(228.89));
        assert!((bt.params.get("r0").and_then(|v| v.as_num()).unwrap() - 1.5354).abs() < 1e-9);

        let angle = ff2.get_style("angle", "harmonic").unwrap();
        let atypes = angle.defs().types();
        let at = &atypes[0];
        assert_eq!(at.params.get("k").and_then(|v| v.as_num()), Some(76.79));
        assert_eq!(
            at.params.get("theta0").and_then(|v| v.as_num()),
            Some(109.66)
        );

        let dih = ff2.get_style("dihedral", "periodic").unwrap();
        let dtypes = dih.defs().types();
        let dt = &dtypes[0];
        assert!((dt.params.get("k1").and_then(|v| v.as_num()).unwrap() - 0.06).abs() < 1e-12);
        assert!(
            (dt.params
                .get("periodicity1")
                .and_then(|v| v.as_num())
                .unwrap()
                - 3.0)
                .abs()
                < 1e-12
        );
        assert!((dt.params.get("phase1").and_then(|v| v.as_num()).unwrap() - 0.0).abs() < 1e-12);

        let lj = ff2.get_style("pair", "lj/cut").unwrap();
        let pt = lj.get_pairtype("c3", None).unwrap();
        assert!((pt.params.get("epsilon").and_then(|v| v.as_num()).unwrap() - 0.1078).abs() < 1e-9);
        assert!((pt.params.get("sigma").and_then(|v| v.as_num()).unwrap() - 3.39771).abs() < 1e-9);
        assert!(
            (lj.params()
                .get("cutoff")
                .and_then(|v| v.as_num())
                .unwrap_or(0.0)
                - 10.0)
                .abs()
                < 1e-12
        );
        let coul = ff2.get_style("pair", "coul/cut").unwrap();
        assert!(
            (coul
                .params()
                .get("cutoff")
                .and_then(|v| v.as_num())
                .unwrap_or(0.0)
                - 10.0)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn skip_pair_style_omits_header() {
        let ff = LammpsForcefieldReader::new().read(MINI.as_bytes()).unwrap();
        let labels = mini_labels();
        let text = LammpsForcefieldWriter::new(&labels)
            .with_skip_pair_style(true)
            .written(&ff)
            .unwrap();
        assert!(!text.contains("pair_style"), "no pair_style:\n{text}");
        // The caller owns the `pair_style` line only: the force field's 1-4
        // weights and mixing rule stay (the reader declared LAMMPS's default,
        // geometric, for an include without `pair_modify mix`).
        assert!(
            text.contains(
                "special_bonds lj 0.000000 0.000000 0.500000 coul 0.000000 0.000000 0.833333\n"
            ),
            "1-4 weights remain:\n{text}"
        );
        assert!(text.contains("pair_modify mix geometric\n"), "{text}");
        assert!(text.contains("pair_coeff c3 c3"), "coeffs remain:\n{text}");
    }

    /// `skip_special_bonds` is the caller stating its own 1-4 weights: the
    /// include must not override them (Amber 1/1.2 over an input's 0.5).
    #[test]
    fn skip_special_bonds_omits_only_special_bonds() {
        let ff = LammpsForcefieldReader::new().read(MINI.as_bytes()).unwrap();
        let labels = mini_labels();
        for skip_pair_style in [false, true] {
            let text = LammpsForcefieldWriter::new(&labels)
                .with_skip_pair_style(skip_pair_style)
                .with_skip_special_bonds(true)
                .written(&ff)
                .unwrap();
            assert!(!text.contains("special_bonds"), "{text}");
            assert_eq!(text.contains("pair_style"), !skip_pair_style, "{text}");
            assert!(text.contains("pair_coeff c3 c3"), "{text}");
        }
    }

    #[test]
    fn default_write_keeps_special_bonds() {
        let ff = LammpsForcefieldReader::new().read(MINI.as_bytes()).unwrap();
        let labels = mini_labels();
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert!(
            text.contains("special_bonds lj"),
            "full include declares 1-4:\n{text}"
        );
    }

    #[test]
    fn skip_units_omits_units_line() {
        let ff = LammpsForcefieldReader::new().read(MINI.as_bytes()).unwrap();
        let labels = mini_labels();
        let text = LammpsForcefieldWriter::new(&labels)
            .with_skip_units(true)
            .with_skip_pair_style(true)
            .written(&ff)
            .unwrap();
        assert!(
            !text.contains("units "),
            "include after input units:\n{text}"
        );
        assert!(text.contains("bond_style harmonic"), "{text}");
        assert!(text.contains("bond_coeff c3-c3"), "{text}");
    }

    /// A dihedral label is its type's name: the reader defines `h1-c3-os-c3`
    /// and `c3-os-c3-h1` as two types, and the writer emits one row per label
    /// used, each under its own name.
    #[test]
    fn write_ff_writes_each_dihedral_label_under_its_own_name() {
        const SRC: &str = r#"
special_bonds amber
pair_style lj/cut 10.0
pair_coeff c3 c3 0.107800 3.397710
dihedral_style fourier
dihedral_coeff h1-c3-os-c3 1 0.337000 3 0.000000
dihedral_coeff c3-os-c3-h1 1 0.337000 3 0.000000
"#;
        let ff = LammpsForcefieldReader::new().read(SRC.as_bytes()).unwrap();
        let labels = labels_of(&[
            ("atoms", &["c3"]),
            ("dihedrals", &["h1-c3-os-c3", "c3-os-c3-h1"]),
        ]);
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "dihedral_coeff"),
            vec![
                "dihedral_coeff c3-os-c3-h1 1 0.337000 3 0.000000",
                "dihedral_coeff h1-c3-os-c3 1 0.337000 3 0.000000",
            ],
            "{text}"
        );
    }

    #[test]
    fn writes_units_real_line_by_default() {
        let ff = LammpsForcefieldReader::new().read(MINI.as_bytes()).unwrap();
        let labels = mini_labels();
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert!(text.contains("units real\n"), "default units real:\n{text}");
    }

    #[test]
    fn write_data_coeffs_matches_command_form_numbers() {
        let ff = LammpsForcefieldReader::new().read(MINI.as_bytes()).unwrap();
        let labels = mini_labels();
        let writer = LammpsForcefieldWriter::new(&labels);
        let data = data_coeffs(&writer, &ff).unwrap();
        assert!(
            !data.contains("pair_style") && !data.contains("units "),
            "sections only:\n{data}"
        );
        assert!(data.contains("Pair Coeffs\n"), "{data}");
        assert!(data.contains("Bond Coeffs # harmonic\n"), "{data}");
        assert!(
            data.contains("1 0.107800 3.397710") || data.contains("1 0.107800"),
            "pair row:\n{data}"
        );
        assert!(
            data.contains("1 228.890000 1.535400"),
            "bond K=k/2:\n{data}"
        );

        let cmd = writer.written(&ff).unwrap();
        // Same bond K appears in both layouts.
        assert!(cmd.contains("bond_coeff c3-c3 228.890000 1.535400"));
        assert!(data.contains("228.890000 1.535400"));
    }

    fn coeff_ids(text: &str, heading: &str) -> Vec<u32> {
        let Some(rest) = text.split(heading).nth(1) else {
            return vec![];
        };
        // Past the rest of the header line (its `# style` comment).
        rest.lines()
            .skip(1)
            .skip_while(|l| l.trim().is_empty())
            .take_while(|l| !l.trim().is_empty())
            .filter_map(|l| l.split_whitespace().next()?.parse().ok())
            .collect()
    }

    /// A label is its type's name, so labels in both orientations are two
    /// types: each block gets two ids and one coeff row per id (never two rows
    /// under one id — LAMMPS would read the extra line as an unknown
    /// identifier).
    #[test]
    fn write_data_coeffs_writes_one_row_per_label_in_either_orientation() {
        const SRC: &str = r#"
special_bonds amber
pair_style lj/cut 10.0
pair_coeff c3 c3 0.107800 3.397710

bond_style harmonic
bond_coeff c3-h1 340.000000 1.090000
bond_coeff h1-c3 340.000000 1.090000

angle_style harmonic
angle_coeff c3-c3-h1 37.500000 110.000000
angle_coeff h1-c3-c3 37.500000 110.000000

dihedral_style fourier
dihedral_coeff h1-c3-c3-os 2 0.250000 1 0.000000 0.000000 3 0.000000
dihedral_coeff os-c3-c3-h1 2 0.250000 1 0.000000 0.000000 3 0.000000
"#;
        let ff = LammpsForcefieldReader::new().read(SRC.as_bytes()).unwrap();
        let labels = labels_of(&[
            ("atoms", &["c3"]),
            ("bonds", &["c3-h1", "h1-c3"]),
            ("angles", &["c3-c3-h1", "h1-c3-c3"]),
            ("dihedrals", &["h1-c3-c3-os", "os-c3-c3-h1"]),
        ]);
        let data = data_coeffs(&LammpsForcefieldWriter::new(&labels), &ff).unwrap();
        assert_eq!(coeff_ids(&data, "Bond Coeffs"), vec![1, 2], "{data}");
        assert_eq!(coeff_ids(&data, "Angle Coeffs"), vec![1, 2], "{data}");
        assert_eq!(coeff_ids(&data, "Dihedral Coeffs"), vec![1, 2], "{data}");
    }

    #[test]
    fn write_data_coeffs_does_not_steal_atom_type_id_from_hyphen_head() {
        const SRC: &str = r#"
special_bonds amber
pair_style lj/cut 10.0
pair_coeff c3 c3 0.107800 3.397710
"#;
        // The dihedral label's head `c3` is an atom label with a pair type;
        // the dihedral must still resolve against dihedral types only.
        let ff = LammpsForcefieldReader::new().read(SRC.as_bytes()).unwrap();
        let labels = labels_of(&[("atoms", &["c3"]), ("dihedrals", &["c3-os-c3-h1"])]);
        let err = data_coeffs(&LammpsForcefieldWriter::new(&labels), &ff).unwrap_err();
        assert!(
            err.contains("c3-os-c3-h1"),
            "unmapped dihedral must not resolve as atom type c3: {err}"
        );
    }

    /// A `real` force field written as `metal` has its energies converted
    /// through the lj hub; reading the `metal` file back gives a `metal`
    /// force field, and writing that back as `metal` is the identity.
    #[test]
    fn metal_write_converts_energy_via_lj_hub() {
        let ff = LammpsForcefieldReader::new().read(MINI.as_bytes()).unwrap();
        assert_eq!(ff.units(), Some("real"));
        let labels = mini_labels();
        let text = LammpsForcefieldWriter::new(&labels)
            .with_units("metal")
            .written(&ff)
            .unwrap();
        assert!(text.contains("units metal\n"), "metal header:\n{text}");

        // 0.1078 kcal/mol → eV through lj hub
        let sys = crate::io::lammps::units::LammpsUnitConverter::canonical().unwrap();
        let eps_ev = sys.energy(0.1078, "real", "metal").unwrap();
        let expected = format!("pair_coeff c3 c3 {:.6}", eps_ev);
        assert!(text.contains(&expected), "expected {expected} in:\n{text}");

        // Length unchanged (Å in both real and metal).
        assert!(
            text.contains(&format!("{:.6}", 3.39771)),
            "sigma stays Å:\n{text}"
        );

        // The metal file reads as a metal force field, number for number, and
        // writes back unchanged.
        let ff2 = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
        assert_eq!(ff2.units(), Some("metal"));
        let again = LammpsForcefieldWriter::new(&labels)
            .with_units("metal")
            .written(&ff2)
            .unwrap();
        assert_eq!(again, text);
    }

    #[test]
    fn atom_type_filter_restricts_pair_coeffs() {
        let ff = LammpsForcefieldReader::new().read(MINI.as_bytes()).unwrap();
        // Only `c3` is used; `oh` is a ForceField type nobody labels.
        let labels = labels_of(&[("atoms", &["c3"])]);
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert!(text.contains("pair_coeff c3 c3"));
        assert!(
            !text.contains("pair_coeff oh oh"),
            "oh filtered out:\n{text}"
        );
    }

    // ------------------------------------------------------------------
    // Label-driven writing (system-forcefield-06): a hand-built ForceField
    // plus TypeLabels from a hand-written Frame, one stage per test.
    // ------------------------------------------------------------------

    use crate::core::Block;
    use crate::core::Frame;
    use crate::core::TypeLabels;
    use crate::core::keys;
    use ndarray::{ArrayD, IxDyn};

    /// A block whose only column is the string `type` label per row.
    fn label_type_block(types: &[&str]) -> Block {
        let mut block = Block::new();
        block
            .insert(
                keys::TYPE,
                ArrayD::from_shape_vec(
                    IxDyn(&[types.len()]),
                    types.iter().map(|t| t.to_string()).collect::<Vec<String>>(),
                )
                .unwrap(),
                None,
            )
            .unwrap();
        block
    }

    /// `TypeLabels` of a Frame holding one labelled block per `(name, types)`.
    fn labels_of(blocks: &[(&str, &[&str])]) -> TypeLabels {
        let mut frame = Frame::new();
        for (name, types) in blocks {
            frame.insert(*name, label_type_block(types));
        }
        TypeLabels::from_frame(&frame).unwrap()
    }

    fn lj(eps: f64, sigma: f64) -> Params {
        Params::from_iter([("epsilon", eps), ("sigma", sigma)])
    }

    /// Harmonic bond: molrs's `k` is LAMMPS's `K`.
    fn bond(k: f64, r0: f64) -> Params {
        Params::from_iter([("k", k), ("r0", r0)])
    }

    /// Split `lj/cut` (cutoff 9) + `coul/cut` (cutoff 10) with hand-written
    /// ε/σ for `c3`, `hc` and `oh`, and harmonic bonds `c3-hc`, `c3-oh`.
    fn split_pair_ff() -> ForceField {
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
        ff.def_style("pair", "coul/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap();
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("c3-hc", &["c3", "hc"], bond(340.0, 1.09))
            .unwrap()
            .def_type("c3-oh", &["c3", "oh"], bond(320.0, 1.41))
            .unwrap();
        ff
    }

    /// Rows of a data-file `heading` section (between its blank lines), past
    /// the header line and its `# style` comment.
    fn data_section_rows(text: &str, heading: &str) -> Vec<String> {
        let Some(rest) = text.split(heading).nth(1) else {
            return vec![];
        };
        rest.lines()
            .skip(1)
            .skip_while(|l| l.trim().is_empty())
            .take_while(|l| !l.trim().is_empty())
            .map(str::to_owned)
            .collect()
    }

    fn lines_starting_with(text: &str, prefix: &str) -> Vec<String> {
        text.lines()
            .filter(|l| l.starts_with(prefix))
            .map(str::to_owned)
            .collect()
    }

    /// `lj/cut` (cutoff 9) with `c3` / `hc` self rows and, when given, the
    /// style-level string param `mixing`.
    fn lj_only_ff(mixing: Option<&str>) -> ForceField {
        let mut params = Params::from_iter([("cutoff", 9.0)]);
        if let Some(rule) = mixing {
            params.set("mixing", rule);
        }
        let mut ff = ForceField::new("hand");
        ff.set_units("real");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "lj/cut", params)
            .unwrap()
            .def_type("c3", &["c3"], lj(0.1078, 3.39771))
            .unwrap()
            .def_type("hc", &["hc"], lj(0.0157, 2.64953))
            .unwrap();
        ff
    }

    /// [`lj_only_ff`] plus a type-less `coul/cut` (cutoff 10): the split pair
    /// the writer recombines into `lj/cut/coul/cut`.
    fn split_pair_ff_mixing(rule: &str) -> ForceField {
        let mut ff = lj_only_ff(Some(rule));
        ff.def_style("pair", "coul/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap();
        ff
    }

    /// Combined branch (`lj/cut` + `coul/cut` → `lj/cut/coul/cut`): an
    /// `lj/cut` declaring no rule is evaluated arithmetically by the kernel,
    /// so the export must say so — LAMMPS' own default is geometric.
    #[test]
    fn combined_undeclared_lj_cut_writes_pair_modify_mix_arithmetic() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["c3", "hc"])]);
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "pair_modify"),
            vec!["pair_modify mix arithmetic".to_owned()],
            "{text}"
        );
    }

    /// Single-style branch: the same rule for a lone undeclared `lj/cut`.
    #[test]
    fn single_undeclared_lj_cut_writes_pair_modify_mix_arithmetic() {
        let ff = lj_only_ff(None);
        let labels = labels_of(&[("atoms", &["c3", "hc"])]);
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "pair_modify"),
            vec!["pair_modify mix arithmetic".to_owned()],
            "{text}"
        );
    }

    /// A pair `hybrid`: each sub-style whose spec declares `mixing` states
    /// its rule on its own `pair_modify pair <sub>` line (LAMMPS's default
    /// for `lj/cut` is geometric); one without (`buck`) writes none.
    #[test]
    fn a_pair_hybrid_writes_each_sub_style_mixing_rule() {
        let mut ff = lj_only_ff(None);
        ff.def_style("pair", "buck", Params::from_iter([("cutoff", 9.0)]))
            .unwrap()
            .def_type(
                "oh",
                &["oh"],
                Params::from_iter([("a", 1000.0), ("rho", 0.3), ("c", 10.0)]),
            )
            .unwrap();
        let labels = labels_of(&[("atoms", &["c3", "oh"])]);
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert!(text.contains("pair_style hybrid lj/cut"), "{text}");
        assert_eq!(
            lines_starting_with(&text, "pair_modify"),
            vec!["pair_modify pair lj/cut mix arithmetic".to_owned()],
            "{text}"
        );
    }

    /// A declared rule is written as declared, in both branches.
    #[test]
    fn declared_geometric_lj_cut_writes_pair_modify_mix_geometric() {
        let labels = labels_of(&[("atoms", &["c3", "hc"])]);
        for ff in [
            split_pair_ff_mixing("geometric"),
            lj_only_ff(Some("geometric")),
        ] {
            let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
            assert_eq!(
                lines_starting_with(&text, "pair_modify"),
                vec!["pair_modify mix geometric".to_owned()],
                "{text}"
            );
        }
    }

    #[test]
    fn label_writer_skips_unused_forcefield_types_in_include() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["c3", "hc", "hc"]), ("bonds", &["c3-hc"])]);
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert!(!text.contains("pair_coeff oh"), "unused oh:\n{text}");
        assert!(!text.contains("c3-oh"), "unused c3-oh:\n{text}");
        assert!(text.contains("pair_coeff c3 c3"), "{text}");
        assert!(text.contains("bond_coeff c3-hc"), "{text}");
    }

    #[test]
    fn label_writer_skips_unused_forcefield_types_in_data_coeffs() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["c3", "hc", "hc"]), ("bonds", &["c3-hc"])]);
        let data = data_coeffs(&LammpsForcefieldWriter::new(&labels), &ff).unwrap();
        assert_eq!(
            data_section_rows(&data, "Pair Coeffs"),
            vec!["1 0.107800 3.397710", "2 0.015700 2.649530"],
            "{data}"
        );
        assert_eq!(
            data_section_rows(&data, "Bond Coeffs"),
            vec!["1 340.000000 1.090000"],
            "{data}"
        );
    }

    #[test]
    fn label_writer_missing_bond_label_is_err_naming_block_and_label() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["c3", "hc"]), ("bonds", &["c3-n"])]);
        let err = LammpsForcefieldWriter::new(&labels)
            .written(&ff)
            .unwrap_err();
        assert!(err.contains("bonds"), "names the block: {err}");
        assert!(err.contains("c3-n"), "names the label: {err}");
    }

    #[test]
    fn label_writer_missing_atom_label_in_data_coeffs_is_err_naming_block_and_label() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["c3", "os"])]);
        let err = data_coeffs(&LammpsForcefieldWriter::new(&labels), &ff).unwrap_err();
        assert!(err.contains("atoms"), "names the block: {err}");
        assert!(err.contains("os"), "names the label: {err}");
    }

    /// Improper `k` of the kernel `k·(χ − χ₀)²` (LAMMPS's `K`), `chi0` in
    /// degrees.
    fn improper_ff() -> ForceField {
        let mut ff = ForceField::new("hand");
        ff.set_units("real");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("improper", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "c3-n-c-o",
                &["c3", "n", "c", "o"],
                Params::from_iter([("k", 2.2), ("chi0", 180.0)]),
            )
            .unwrap();
        ff
    }

    #[test]
    fn label_writer_matches_improper_label_exactly() {
        let ff = improper_ff();
        let labels = labels_of(&[("impropers", &["c3-n-c-o"])]);
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "improper_coeff"),
            vec!["improper_coeff c3-n-c-o 2.200000 180.000000"],
            "{text}"
        );
    }

    #[test]
    fn label_writer_does_not_resolve_reversed_improper_label() {
        let ff = improper_ff();
        let labels = labels_of(&[("impropers", &["o-c-n-c3"])]);
        let err = LammpsForcefieldWriter::new(&labels)
            .written(&ff)
            .unwrap_err();
        assert!(err.contains("impropers"), "names the block: {err}");
        assert!(err.contains("o-c-n-c3"), "names the label: {err}");
    }

    /// AMBER `improper periodic`: `k` kcal/mol, integer `periodicity`, `phase` deg.
    fn periodic_improper_ff(phase: f64) -> ForceField {
        let mut ff = ForceField::new("hand");
        ff.set_units("real");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("improper", "periodic", Params::new())
            .unwrap()
            .def_type(
                "c3-o-c-os",
                &["c3", "o", "c", "os"],
                Params::from_iter([("k", 1.1), ("periodicity", 2.0), ("phase", phase)]),
            )
            .unwrap();
        ff
    }

    #[test]
    fn periodic_improper_is_written_as_cvff() {
        // AMBER stores π rounded to 3.1416 in a prmtop, 7.3e-6 rad above the
        // true value, which a prmtop reader stores as 180.0004°.
        let ff = periodic_improper_ff((std::f64::consts::PI + 7.3e-6).to_degrees());
        let labels = labels_of(&[("impropers", &["c3-o-c-os"])]);
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "improper_style"),
            vec!["improper_style cvff"],
            "{text}"
        );
        assert_eq!(
            lines_starting_with(&text, "improper_coeff"),
            vec!["improper_coeff c3-o-c-os 1.100000 -1 2"],
            "{text}"
        );
    }

    #[test]
    fn periodic_improper_phase_sets_the_cvff_sign() {
        let zero = Params::from_iter([("k", 1.1), ("periodicity", 2.0), ("phase", 0.0)]);
        assert_eq!(
            lammps_coeff_values("improper", "periodic", &zero, "real").unwrap(),
            [1.1, 1.0, 2.0]
        );
        let pi = Params::from_iter([("k", 1.1), ("periodicity", 2.0), ("phase", 180.0)]);
        assert_eq!(
            lammps_coeff_values("improper", "periodic", &pi, "real").unwrap(),
            [1.1, -1.0, 2.0]
        );
    }

    #[test]
    fn periodic_improper_with_other_phase_is_refused() {
        let p = Params::from_iter([("k", 1.1), ("periodicity", 2.0), ("phase", 57.3)]);
        let err = lammps_coeff_values("improper", "periodic", &p, "real").unwrap_err();
        assert!(err.contains("cvff"), "{err}");
    }

    #[test]
    fn label_data_coeff_ids_follow_type_labels_not_forcefield_order() {
        // ForceField order: oh, hc, c3 / c3-oh, c3-hc. Label order (sorted):
        // c3=1, hc=2, oh=3 / c3-hc=1, c3-oh=2.
        let mut ff = ForceField::new("hand");
        ff.set_units("real");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 9.0)]))
            .unwrap()
            .def_type("oh", &["oh"], lj(0.093, 3.242871))
            .unwrap()
            .def_type("hc", &["hc"], lj(0.0157, 2.64953))
            .unwrap()
            .def_type("c3", &["c3"], lj(0.1078, 3.39771))
            .unwrap();
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("c3-oh", &["c3", "oh"], bond(320.0, 1.41))
            .unwrap()
            .def_type("c3-hc", &["c3", "hc"], bond(340.0, 1.09))
            .unwrap();
        let labels = labels_of(&[
            ("atoms", &["hc", "oh", "c3", "hc"]),
            ("bonds", &["c3-oh", "c3-hc"]),
        ]);
        let data = data_coeffs(&LammpsForcefieldWriter::new(&labels), &ff).unwrap();
        assert_eq!(
            data_section_rows(&data, "Pair Coeffs"),
            vec![
                "1 0.107800 3.397710",
                "2 0.015700 2.649530",
                "3 0.093000 3.242871"
            ],
            "{data}"
        );
        assert_eq!(
            data_section_rows(&data, "Bond Coeffs"),
            vec!["1 340.000000 1.090000", "2 320.000000 1.410000"],
            "{data}"
        );
    }

    #[test]
    fn label_writer_combines_split_lj_coul_into_one_pair_style_line() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["c3", "hc"])]);
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "pair_style"),
            vec!["pair_style lj/cut/coul/cut 9.000000 10.000000"],
            "{text}"
        );
    }

    #[test]
    fn a_pair_style_without_a_cutoff_is_refused_not_defaulted() {
        let mut ff = ForceField::new("hand");
        ff.set_units("real");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "lj/cut", Params::new())
            .unwrap()
            .def_type("c3", &["c3"], lj(0.1078, 3.39771))
            .unwrap();
        ff.def_style("pair", "coul/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap();
        let labels = labels_of(&[("atoms", &["c3"])]);
        let err = LammpsForcefieldWriter::new(&labels)
            .written(&ff)
            .unwrap_err();
        assert!(err.contains("'lj/cut' has no cutoff"), "{err}");
    }

    #[test]
    fn label_writer_pair_coeff_lines_follow_hand_written_eps_sigma() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["hc", "c3"])]);
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "pair_coeff"),
            vec![
                "pair_coeff c3 c3 0.107800 3.397710",
                "pair_coeff hc hc 0.015700 2.649530"
            ],
            "{text}"
        );
    }

    #[test]
    fn label_writer_writes_cross_pair_only_when_both_endpoints_used() {
        let mut ff = split_pair_ff();
        ff.get_style_mut("pair", "lj/cut")
            .unwrap()
            .def_type("c3-hc", &["c3", "hc"], lj(0.05, 3.0))
            .unwrap()
            .def_type("c3-oh", &["c3", "oh"], lj(0.07, 3.3))
            .unwrap();
        let labels = labels_of(&[("atoms", &["c3", "hc"])]);
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert!(
            text.contains("pair_coeff c3 hc 0.050000 3.000000\n"),
            "{text}"
        );
        assert!(!text.contains("pair_coeff c3 oh"), "{text}");
    }

    #[test]
    fn label_writer_skip_pair_style_omits_the_pair_style_line_only() {
        let ff = split_pair_ff();
        let labels = labels_of(&[("atoms", &["c3", "hc"])]);
        let text = LammpsForcefieldWriter::new(&labels)
            .with_skip_pair_style(true)
            .written(&ff)
            .unwrap();
        assert!(!text.contains("pair_style"), "{text}");
        assert!(
            text.contains(
                "special_bonds lj 0.000000 0.000000 1.000000 coul 0.000000 0.000000 1.000000\n"
            ),
            "{text}"
        );
        assert!(text.contains("pair_modify mix arithmetic\n"), "{text}");
        assert!(
            text.contains("pair_coeff c3 c3 0.107800 3.397710"),
            "{text}"
        );
    }

    /// Harmonic `c3-hc` plus an unsupported `fene` style holding the bond
    /// `c3-oh` on `c3`, `oh`.
    fn ff_with_fene() -> ForceField {
        let mut ff = ForceField::new("hand");
        ff.set_units("real");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("c3-hc", &["c3", "hc"], bond(340.0, 1.09))
            .unwrap();
        ff.def_style("bond", "fene", Params::new())
            .unwrap()
            .def_type(
                "c3-oh",
                &["c3", "oh"],
                Params::from_iter([("k", 30.0), ("r0", 1.5), ("epsilon", 1.0), ("sigma", 1.0)]),
            )
            .unwrap();
        ff
    }

    #[test]
    fn label_writer_tolerates_unsupported_style_holding_only_unused_types() {
        let ff = ff_with_fene();
        let labels = labels_of(&[("bonds", &["c3-hc"])]);
        let writer = LammpsForcefieldWriter::new(&labels);
        let text = writer.written(&ff).unwrap();
        assert!(!text.contains("fene"), "{text}");
        assert!(text.contains("bond_coeff c3-hc"), "{text}");
        let data = data_coeffs(&writer, &ff).unwrap();
        assert_eq!(
            data_section_rows(&data, "Bond Coeffs"),
            vec!["1 340.000000 1.090000"],
            "{data}"
        );
    }

    /// An orientation mismatch reads differently from a missing type: the
    /// error names the spelling that is defined.
    #[test]
    fn label_writer_names_the_reversed_spelling_that_is_defined() {
        let ff = ff_with_fene();
        let labels = labels_of(&[("bonds", &["hc-c3"])]);
        let err = LammpsForcefieldWriter::new(&labels)
            .written(&ff)
            .unwrap_err();
        assert!(err.contains("`hc-c3` has no bond type"), "{err}");
        assert!(err.contains("`c3-hc` is defined"), "{err}");
    }

    #[test]
    fn label_writer_rejects_unsupported_style_holding_a_used_type() {
        let ff = ff_with_fene();
        let labels = labels_of(&[("bonds", &["c3-hc", "c3-oh"])]);
        let writer = LammpsForcefieldWriter::new(&labels);
        let err = writer.written(&ff).unwrap_err();
        assert!(err.contains("fene"), "names the style: {err}");
        let err = data_coeffs(&writer, &ff).unwrap_err();
        assert!(err.contains("fene"), "names the style: {err}");
    }

    /// A label is matched to a type name exactly. The force field defines the
    /// bond `c3-h1` (K = 340) and the qualified angle `C_3-C_R-O_2@1_1.5_2`
    /// (K = 60, theta0 = 120 degrees);
    /// labels spelled the same are written, and the reversed spellings
    /// `h1-c3` / `O_2-C_R-C_3@1.5_1_2` find no type — an error naming the
    /// block and the label.
    #[test]
    fn label_writer_matches_labels_to_type_names_exactly() {
        let mut ff = ForceField::new("hand");
        ff.set_units("real");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("c3-h1", &["c3", "h1"], bond(340.0, 1.09))
            .unwrap();
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "C_3-C_R-O_2@1_1.5_2",
                &["C_3", "C_R", "O_2"],
                Params::from_iter([("k", 60.0), ("theta0", 120.0)]),
            )
            .unwrap();

        let exact = labels_of(&[("bonds", &["c3-h1"]), ("angles", &["C_3-C_R-O_2@1_1.5_2"])]);
        let text = LammpsForcefieldWriter::new(&exact)
            .written(&ff)
            .expect("exact labels resolve");
        assert_eq!(
            lines_starting_with(&text, "bond_coeff"),
            vec!["bond_coeff c3-h1 340.000000 1.090000"],
            "{text}"
        );
        assert_eq!(
            lines_starting_with(&text, "angle_coeff"),
            vec!["angle_coeff C_3-C_R-O_2@1_1.5_2 60.000000 120.000000"],
            "{text}"
        );

        let reversed_bond = labels_of(&[("bonds", &["h1-c3"])]);
        let err = LammpsForcefieldWriter::new(&reversed_bond)
            .written(&ff)
            .unwrap_err();
        assert!(err.contains("bonds") && err.contains("h1-c3"), "{err}");

        let reversed_angle = labels_of(&[("angles", &["O_2-C_R-C_3@1.5_1_2"])]);
        let err = LammpsForcefieldWriter::new(&reversed_angle)
            .written(&ff)
            .unwrap_err();
        assert!(
            err.contains("angles") && err.contains("O_2-C_R-C_3@1.5_1_2"),
            "{err}"
        );
    }

    /// MMFF's buffered Coulomb (`delta = 0.05`) has no LAMMPS style: refused.
    #[test]
    fn a_buffered_coulomb_is_refused() {
        let mut ff = split_pair_ff();
        ff.get_style_mut("pair", "coul/cut")
            .unwrap()
            .set_param("delta", 0.05);
        let labels = labels_of(&[("atoms", &["c3", "hc"])]);
        let err = LammpsForcefieldWriter::new(&labels)
            .written(&ff)
            .unwrap_err();
        assert!(err.contains("delta"), "{err}");
    }

    /// A pair style whose types carry no ε/σ (`thole`: per-type `charge`,
    /// `alpha`, `damp`) has no `pair_coeff` form here. Writing the
    /// `pair_style` line without its coefficients is an incomplete include, so
    /// both writers refuse and name the category and the style.
    #[test]
    fn pair_style_without_writable_coeffs_is_err_naming_style() {
        let mut ff = ForceField::new("hand");
        ff.set_units("real");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "thole", Params::from_iter([("cutoff", 12.0)]))
            .unwrap()
            .def_type(
                "c3",
                &["c3"],
                Params::from_iter([("charge", -0.2), ("alpha", 1.1), ("damp", 2.6)]),
            )
            .unwrap();
        let labels = labels_of(&[("atoms", &["c3"])]);
        let writer = LammpsForcefieldWriter::new(&labels);

        for err in [
            writer
                .written(&ff)
                .expect_err("thole has no pair_coeff form"),
            data_coeffs(&writer, &ff).expect_err("thole has no Pair Coeffs form"),
        ] {
            assert!(err.contains("pair") && err.contains("thole"), "{err}");
        }
    }

    fn assert_values(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{got:?} != {want:?}");
        for (g, w) in got.iter().zip(want) {
            let tol = 1e-12 * w.abs().max(1.0);
            assert!((g - w).abs() <= tol, "{got:?} != {want:?}");
        }
    }

    /// `(category, style, stored params, expected LAMMPS values)`.
    type ValuesCase<'a> = (&'a str, &'a str, &'a [(&'a str, f64)], &'a [f64]);

    /// Hand-derived goldens for every kernel, in `real`: the identity on
    /// values, slot for slot — LAMMPS's `K`, degrees, energies as stored.
    #[test]
    fn lammps_coeff_values_renders_each_kernel() {
        let cases: &[ValuesCase<'_>] = &[
            (
                "bond",
                "harmonic",
                &[("k", 450.0), ("r0", 0.9572)],
                &[450.0, 0.9572],
            ),
            (
                "bond",
                "morse",
                &[("d0", 95.6), ("alpha", 2.0), ("r0", 1.53)],
                &[95.6, 2.0, 1.53],
            ),
            (
                "angle",
                "harmonic",
                &[("k", 55.0), ("theta0", 104.52)],
                &[55.0, 104.52],
            ),
            (
                "improper",
                "harmonic",
                &[("k", 10.0), ("chi0", 180.0)],
                &[10.0, 180.0],
            ),
            (
                "improper",
                "cvff",
                &[("k", 1.1), ("sign", -1.0), ("periodicity", 2.0)],
                &[1.1, -1.0, 2.0],
            ),
            (
                "dihedral",
                "opls",
                &[("k1", 1.0), ("k2", 2.0), ("k3", 3.0), ("k4", 4.0)],
                &[1.0, 2.0, 3.0, 4.0],
            ),
            (
                "dihedral",
                "harmonic",
                &[("k", 2.0), ("sign", -1.0), ("periodicity", 3.0)],
                &[2.0, -1.0, 3.0],
            ),
            (
                "dihedral",
                "periodic",
                &[
                    ("k1", 0.5),
                    ("periodicity1", 1.0),
                    ("phase1", 180.0),
                    ("k2", 0.25),
                    ("periodicity2", 3.0),
                    ("phase2", 0.0),
                ],
                &[2.0, 0.5, 1.0, 180.0, 0.25, 3.0, 0.0],
            ),
            (
                "dihedral",
                "charmm",
                &[
                    ("k", 0.2),
                    ("periodicity", 3.0),
                    ("phase", 180.0),
                    ("w", 0.5),
                ],
                &[0.2, 3.0, 180.0, 0.5],
            ),
            (
                "dihedral",
                "multi/harmonic",
                &[
                    ("a1", 1.0),
                    ("a2", 2.0),
                    ("a3", 3.0),
                    ("a4", 4.0),
                    ("a5", 5.0),
                ],
                &[1.0, 2.0, 3.0, 4.0, 5.0],
            ),
            (
                "pair",
                "lj/cut",
                &[("epsilon", 0.066), ("sigma", 3.5)],
                &[0.066, 3.5],
            ),
        ];
        for (category, style, params, want) in cases {
            let got = lammps_coeff_values(
                category,
                style,
                &Params::from_iter(params.iter().copied()),
                "real",
            )
            .unwrap_or_else(|e| panic!("{category} {style}: {e}"));
            assert_values(&got, want);
        }
    }

    #[test]
    fn lammps_coeff_values_rejects_unsupported_kernel_and_missing_param() {
        let err = lammps_coeff_values("bond", "fene", &Params::from_iter([("k", 1.0)]), "real")
            .unwrap_err();
        assert!(err.contains("bond") && err.contains("fene"), "{err}");

        let err = lammps_coeff_values(
            "bond",
            "harmonic",
            &Params::from_iter([("k", 450.0)]),
            "real",
        )
        .unwrap_err();
        assert!(err.contains("r0"), "{err}");

        let bond = Params::from_iter([("k", 450.0), ("r0", 0.9572)]);
        let err = lammps_coeff_values("bond", "harmonic", &bond, "si").unwrap_err();
        assert!(err.contains("si"), "{err}");
    }

    /// `dihedral periodic` is molrs's name for LAMMPS `dihedral_style
    /// fourier`, term for term; the unindexed one-term spelling the kernel
    /// accepts is written as one fourier term.
    #[test]
    fn periodic_dihedral_is_written_as_fourier() {
        let single = Params::from_iter([("k", 0.3), ("periodicity", 2.0), ("phase", 180.0)]);
        assert_eq!(
            lammps_coeff_values("dihedral", "periodic", &single, "real").unwrap(),
            vec![1.0, 0.3, 2.0, 180.0]
        );
    }

    /// Writer and reader are inverse through their two public homes: the
    /// values rendered for a kernel read back, as tokens, to the same params,
    /// and the tokens come back as written.
    #[test]
    fn lammps_coeff_values_round_trips_through_lammps_coeff_params() {
        use crate::io::lammps::forcefield_reader::lammps_coeff_params;
        // (category, LAMMPS style, molrs style, tokens)
        let cases: &[(&str, &str, &str, &[&str])] = &[
            ("bond", "harmonic", "harmonic", &["450", "0.9572"]),
            ("bond", "morse", "morse", &["95.6", "2", "1.53"]),
            ("angle", "harmonic", "harmonic", &["55", "104.52"]),
            (
                "angle",
                "charmm",
                "charmm",
                &["33.43", "110.1", "22.53", "2.179"],
            ),
            ("improper", "harmonic", "harmonic", &["10", "180"]),
            ("improper", "cvff", "cvff", &["1.1", "-1", "2"]),
            ("dihedral", "opls", "opls", &["1", "2", "3", "4"]),
            ("dihedral", "harmonic", "harmonic", &["2", "-1", "3"]),
            (
                "dihedral",
                "fourier",
                "periodic",
                &["2", "0.5", "1", "180", "0.25", "3", "0"],
            ),
            ("dihedral", "charmm", "charmm", &["0.2", "3", "180", "0.5"]),
            (
                "dihedral",
                "multi/harmonic",
                "multi/harmonic",
                &["1", "2", "3", "4", "5"],
            ),
            (
                "dihedral",
                "nharmonic",
                "nharmonic",
                &["3", "1", "-2", "0.5"],
            ),
            ("pair", "lj/cut", "lj/cut", &["0.066", "3.5"]),
        ];
        for (category, lammps, molrs, tokens) in cases {
            let params = lammps_coeff_params(category, lammps, tokens, "real").unwrap();
            let values = lammps_coeff_values(category, molrs, &params, "real")
                .unwrap_or_else(|e| panic!("{category} {molrs}: {e}"));
            let strings: Vec<String> = values.iter().map(f64::to_string).collect();
            let strs: Vec<&str> = strings.iter().map(String::as_str).collect();
            assert_eq!(&strs, tokens, "{category} {lammps}: the identity on tokens");
            let back = lammps_coeff_params(category, lammps, &strs, "real").unwrap();
            assert_eq!(back, params, "{category} {lammps}: {strs:?}");
        }
    }

    /// A CHARMM include with Urey–Bradley angles beside plain ones: one
    /// `angle_style hybrid`, each `angle_coeff` naming its sub-style.
    const UB_HYBRID: &str = "\
# LAMMPS force field generated by molrs
units real

special_bonds lj 0.000000 0.000000 0.000000 coul 0.000000 0.000000 0.000000

angle_style hybrid harmonic charmm
angle_coeff CT-CT-CT harmonic 58.350000 113.600000
angle_coeff HA-CT-CT charmm 33.430000 110.100000 22.530000 2.179000
angle_coeff HA-CT-HA charmm 35.500000 108.400000 5.400000 1.802000

";

    /// `angle charmm` reads `K theta0 K_ub r_ub` as written and writes it back
    /// the same: the LAMMPS read → write identity, hybrid included. (The
    /// writer orders rows by label id, and `TypeLabels` sorts labels.)
    #[test]
    fn angle_charmm_hybrid_reads_and_writes_back_identically() {
        let ff = LammpsForcefieldReader::new()
            .read(UB_HYBRID.as_bytes())
            .unwrap();
        let ub = ff.get_style("angle", "charmm").unwrap();
        let types = ub.defs().types();
        let t = types.iter().find(|t| t.name == "HA-CT-CT").unwrap();
        for (key, want) in [
            ("k", 33.43),
            ("theta0", 110.1),
            ("k_ub", 22.53),
            ("r_ub", 2.179),
        ] {
            assert_eq!(
                t.params.get(key).and_then(|v| v.as_num()),
                Some(want),
                "{key}"
            );
        }
        assert!(ff.get_style("angle", "harmonic").is_some());

        let labels = labels_of(&[("angles", &["CT-CT-CT", "HA-CT-CT", "HA-CT-HA"])]);
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert_eq!(text, UB_HYBRID);
        let again = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
        assert_eq!(
            LammpsForcefieldWriter::new(&labels)
                .written(&again)
                .unwrap(),
            text
        );
    }

    /// One style needs no `hybrid`; the data-file section names it.
    #[test]
    fn angle_charmm_alone_is_a_plain_style_and_its_data_section_says_so() {
        let ff = LammpsForcefieldReader::new()
            .read(UB_HYBRID.as_bytes())
            .unwrap();
        let labels = labels_of(&[("angles", &["HA-CT-CT"])]);
        let writer = LammpsForcefieldWriter::new(&labels);
        let text = writer.written(&ff).unwrap();
        assert_eq!(
            lines_starting_with(&text, "angle_"),
            vec![
                "angle_style charmm",
                "angle_coeff HA-CT-CT 33.430000 110.100000 22.530000 2.179000",
            ],
            "{text}"
        );
        let data = data_coeffs(&writer, &ff).unwrap();
        assert!(data.contains("Angle Coeffs # charmm\n"), "{data}");
        assert_eq!(
            data_section_rows(&data, "Angle Coeffs"),
            vec!["1 33.430000 110.100000 22.530000 2.179000"]
        );
    }

    /// A data file's hybrid section carries the sub-style on each row, and
    /// `read_data_coeffs` reads it back to the same force field.
    #[test]
    fn angle_charmm_hybrid_data_coeffs_round_trip() {
        use crate::io::lammps::forcefield_reader::LammpsTypeLabelMaps;
        let ff = LammpsForcefieldReader::new()
            .read(UB_HYBRID.as_bytes())
            .unwrap();
        let names = ["CT-CT-CT", "HA-CT-CT", "HA-CT-HA"];
        let labels = labels_of(&[("angles", &names)]);
        let data = data_coeffs(&LammpsForcefieldWriter::new(&labels), &ff).unwrap();
        assert!(data.contains("Angle Coeffs # hybrid\n"), "{data}");
        assert_eq!(
            data_section_rows(&data, "Angle Coeffs"),
            vec![
                "1 harmonic 58.350000 113.600000",
                "2 charmm 33.430000 110.100000 22.530000 2.179000",
                "3 charmm 35.500000 108.400000 5.400000 1.802000",
            ]
        );
        let maps = LammpsTypeLabelMaps {
            angle: (1..).zip(names.iter().map(|n| n.to_string())).collect(),
            ..Default::default()
        };
        let back = LammpsForcefieldReader::new()
            .read_data_sections(&data, &maps, "real")
            .unwrap();
        let angles = |ff: &ForceField, style: &str| -> Vec<TypeDef> {
            ff.get_style("angle", style)
                .unwrap()
                .defs()
                .types()
                .to_vec()
        };
        for style in ["charmm", "harmonic"] {
            assert_eq!(angles(&back, style), angles(&ff, style), "{style}");
        }
    }

    /// Under `angle_style harmonic`, a `K theta0 K_ub r_ub` line is refused —
    /// read as harmonic it would silently drop its Urey–Bradley term — and a
    /// hybrid line must name a declared sub-style.
    #[test]
    fn a_urey_bradley_line_under_the_wrong_style_is_refused() {
        let err = LammpsForcefieldReader::new()
            .read(
                "special_bonds charmm\nangle_style harmonic\n\
                 angle_coeff A-B-A 33.43 110.1 22.53 2.179\n"
                    .as_bytes(),
            )
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("takes 2 coefficients (k theta0), got 4"),
            "{err}"
        );
        let err = LammpsForcefieldReader::new()
            .read(
                "special_bonds charmm\nangle_style hybrid harmonic\n\
                 angle_coeff A-B-A charmm 33.43 110.1 22.53 2.179\n"
                    .as_bytes(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("not one of"), "{err}");
    }

    // ── fix cmap ────────────────────────────────────────────────────────────

    const ALANINE: &str = include_str!("../../ff/potential/cmap/testdata/charmm36_alanine.cmap");

    /// The numbers of a CHARMM cmap file, as its lines: comments dropped.
    fn number_lines(text: &str) -> Vec<&str> {
        text.lines()
            .map(str::trim_end)
            .filter(|l| !l.is_empty() && !l.trim_start().starts_with('#'))
            .collect()
    }

    fn cmap_ff(grids: &[(&str, ArrayD<f64>)]) -> ForceField {
        let mut ff = ForceField::new("charmm");
        ff.set_units("real");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        let style = ff.def_style("cmap", "charmm", Params::new()).unwrap();
        for (name, grid) in grids {
            let mut params = Params::new();
            params.set("grid", grid.clone());
            style.def_type(name, &[*name; 5], params).unwrap();
        }
        ff
    }

    /// Read CHARMM's file, write it: every number line comes back as the very
    /// line it was read from, and the text reads back to the same bits.
    #[test]
    fn a_charmm_cmap_file_is_written_back_line_for_line() {
        use crate::io::lammps::forcefield_reader::parse_cmap_file;
        let map = parse_cmap_file(ALANINE).unwrap().maps.remove(0);
        let ff = cmap_ff(&[("ala", map.clone())]);
        let labels = labels_of(&[("cmaps", &["ala", "ala"])]);
        let text = cmap_file(&LammpsForcefieldWriter::new(&labels), &ff).unwrap();
        assert_eq!(number_lines(&text), number_lines(ALANINE));
        assert!(text.starts_with("# UNITS: real "), "{text}");
        assert!(text.contains("\n# ala, type 1\n"), "{text}");
        let back = parse_cmap_file(&text).unwrap();
        assert_eq!(back.units.as_deref(), Some("real"));
        assert_eq!(back.maps, vec![map.clone()]);

        // Any value survives once the decimals reach its 17th digit.
        let odd = map.mapv(|v| v / 3.0);
        let text = cmap_file_text(&[("odd", &odd)], "real", 25).unwrap();
        assert_eq!(parse_cmap_file(&text).unwrap().maps, vec![odd]);
    }

    /// Maps go out in the `cmaps` labels' id order (the data file's crossterm
    /// types), converted to the file's units; the include names the file on
    /// a `fix cmap` line beside `units`, and reads back.
    #[test]
    fn the_cmap_file_and_fix_line_follow_the_labels() {
        let a = ArrayD::from_shape_fn(vec![24, 24], |ix| (ix[0] * 24 + ix[1]) as f64 / 100.0);
        let b = a.mapv(|v| -v);
        let ff = cmap_ff(&[("b", b.clone()), ("a", a.clone())]);
        let labels = labels_of(&[("cmaps", &["b", "a", "b"])]);
        let writer = LammpsForcefieldWriter::new(&labels)
            .with_cmap_file("sys.cmap")
            .with_skip_pair_style(true);
        let text = cmap_file(&writer, &ff).unwrap();
        let maps = crate::io::lammps::forcefield_reader::parse_cmap_file(&text)
            .unwrap()
            .maps;
        assert_eq!(maps, vec![a.clone(), b]);

        let include = writer.written(&ff).unwrap();
        let fix = include.find("fix cmap all cmap sys.cmap\nfix_modify cmap energy yes\n");
        let units = include.find("units real");
        assert!(fix.is_some() && units < fix, "{include}");

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("sys.cmap"), &text).unwrap();
        let path = dir.path().join("sys.ff");
        std::fs::write(&path, include.replace("\n\n", "\nspecial_bonds charmm\n\n")).unwrap();
        let back = LammpsForcefieldReader::new().read(path.as_path()).unwrap();
        let rows = back.get_types("cmap");
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0].params.get("grid").and_then(|v| v.as_array()),
            Some(&a)
        );

        // In metal units the energies are converted.
        let metal = LammpsForcefieldWriter::new(&labels)
            .with_units("metal")
            .with_precision(12);
        let text = cmap_file(&metal, &ff).unwrap();
        let file = crate::io::lammps::forcefield_reader::parse_cmap_file(&text).unwrap();
        assert_eq!(file.units.as_deref(), Some("metal"));
        let ev = file.maps[0][[0, 1]];
        assert!((ev - 0.01 * 0.0433641).abs() < 1e-8, "{ev}");
    }

    #[test]
    fn what_fix_cmap_cannot_read_is_refused() {
        let grid = ArrayD::zeros(vec![24, 24]);
        let labels = labels_of(&[("cmaps", &["a"])]);
        // The include of a system with crossterms needs the file name.
        let err = LammpsForcefieldWriter::new(&labels)
            .written(&cmap_ff(&[("a", grid.clone())]))
            .unwrap_err();
        assert!(err.contains("cmap_file"), "{err}");
        // fix cmap reads 24×24 maps, at most six.
        let err = cmap_file(
            &LammpsForcefieldWriter::new(&labels),
            &cmap_ff(&[("a", ArrayD::zeros(vec![12, 12]))]),
        )
        .unwrap_err();
        assert!(err.contains("24×24"), "{err}");
        let names = ["a", "b", "c", "d", "e", "f", "g"];
        let many: Vec<(&str, ArrayD<f64>)> = names.iter().map(|n| (*n, grid.clone())).collect();
        let err = cmap_file(
            &LammpsForcefieldWriter::new(&labels_of(&[("cmaps", &names)])),
            &cmap_ff(&many),
        )
        .unwrap_err();
        assert!(err.contains("at most 6"), "{err}");
        // A label without a row.
        let zz = labels_of(&[("cmaps", &["zz"])]);
        let err =
            cmap_file(&LammpsForcefieldWriter::new(&zz), &cmap_ff(&[("a", grid)])).unwrap_err();
        assert!(err.contains("`zz`"), "{err}");
    }

    /// The pair settings LAMMPS can hold are written, the others refused by
    /// name: `shift` is `pair_modify shift yes`; a Mie `n`/`m` is refused;
    /// `coul/long/pme` without Ewald parameters is the real-space
    /// `lj/cut/coul/long` (and reads back so), with them refused; a type-less
    /// style LAMMPS has no form for here (`coul/tt`) is refused.
    #[test]
    fn pair_settings_are_written_or_refused_by_name() {
        let labels = labels_of(&[("atoms", &["c3"])]);
        let read = |text: &str| LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
        let write = |ff: &ForceField| LammpsForcefieldWriter::new(&labels).written(ff);

        let shifted = read(
            "special_bonds amber\npair_style lj/cut 10.0\npair_modify mix arithmetic shift yes\npair_coeff c3 c3 0.1 3.4\n",
        );
        let text = write(&shifted).unwrap();
        assert!(
            text.contains("pair_modify mix arithmetic shift yes"),
            "{text}"
        );
        assert_eq!(
            read(&text)
                .get_style("pair", "lj/cut")
                .unwrap()
                .params()
                .get("shift")
                .and_then(|v| v.as_num()),
            Some(1.0)
        );

        let mut mie = shifted.clone();
        mie.get_style_mut("pair", "lj/cut")
            .unwrap()
            .set_param("n", 9.0);
        assert!(write(&mie).unwrap_err().contains("n = 9"));

        let long = read(
            "special_bonds amber\npair_style lj/cut/coul/long 10.0 12.0\npair_coeff c3 c3 0.1 3.4\n",
        );
        let text = write(&long).unwrap();
        assert!(text.contains("pair_style lj/cut/coul/long 10"), "{text}");
        assert!(read(&text).get_style("pair", "coul/long/pme").is_some());
        let mut ewald = long.clone();
        ewald
            .get_style_mut("pair", "coul/long/pme")
            .unwrap()
            .set_param("alpha", 0.3);
        assert!(write(&ewald).unwrap_err().contains("alpha"));

        let mut tt =
            read("special_bonds amber\npair_style lj/cut 10.0\npair_coeff c3 c3 0.1 3.4\n");
        tt.def_style("pair", "coul/tt", Params::from_iter([("cutoff", 10.0)]))
            .unwrap();
        assert!(write(&tt).unwrap_err().contains("coul/tt"));
    }

    /// `pair_modify mix sixthpower` reads as the style's `mixing` and is
    /// written back.
    #[test]
    fn sixthpower_mixing_is_read_and_written_back() {
        let labels = labels_of(&[("atoms", &["c3"])]);
        let ff = LammpsForcefieldReader::new()
            .read(
                "special_bonds amber\npair_style lj/cut 10.0\npair_modify mix sixthpower\n\
                 pair_coeff c3 c3 0.1 3.4\n"
                    .as_bytes(),
            )
            .unwrap();
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(
            lj.params().get("mixing").and_then(|v| v.as_text()),
            Some("sixthpower")
        );
        let text = LammpsForcefieldWriter::new(&labels).written(&ff).unwrap();
        assert!(text.contains("pair_modify mix sixthpower"), "{text}");
    }

    // ── Drude oscillators ───────────────────────────────────────────────────

    /// `CT` cores with `DCT` shells: two `CT-HC` bonds and one `CT-DCT`
    /// Drude spring.
    fn drude_frame() -> Frame {
        let mut frame = Frame::new();
        frame.insert("atoms", label_type_block(&["CT", "DCT", "HC", "HC"]));
        frame
            .get_mut("atoms")
            .unwrap()
            .insert(
                "mass",
                ndarray::array![11.6, 0.4, 1.008, 1.008].into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("bonds", label_type_block(&["CT-HC", "CT-HC"]));
        frame.insert("drudes", label_type_block(&["CT-DCT"]));
        frame
    }

    fn drude_ff() -> ForceField {
        let mut ff = ForceField::new("clpol");
        ff.set_units("real");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("CT-HC", &["CT", "HC"], bond(340.0, 1.09))
            .unwrap();
        ff.def_style("drude", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-DCT",
                &["CT", "DCT"],
                Params::from_iter(
                    [("k", 500.0), ("alpha", 1.2), ("thole", 2.6)]
                        .iter()
                        .copied(),
                ),
            )
            .unwrap();
        ff
    }

    /// A Drude spring (`drude harmonic`, `k*r^2`) is a LAMMPS harmonic bond
    /// (`K (r - r0)^2`) with `K = k` and `r0 = 0`, numbered among the bonds
    /// as the data writer numbers its `drudes` rows; `alpha` and `thole`
    /// ride in a comment.
    #[test]
    fn a_drude_spring_is_written_as_a_harmonic_bond_coeff() {
        let text = crate::io::writer::capture(|sink| {
            LammpsForcefieldWriter::new(
                &crate::io::lammps::lammps_type_labels(&drude_frame()).unwrap(),
            )
            .with_skip_special_bonds(true)
            .write(sink, &drude_ff())
        })
        .expect("the springs are bonds");
        assert!(
            text.contains(
                "bond_style harmonic\n\
                 bond_coeff CT-DCT 500.000000 0.000000\n\
                 bond_coeff CT-HC 340.000000 1.090000\n"
            ),
            "{text}"
        );
        assert!(
            text.contains("#   CT-DCT (CT DCT): alpha 1.200000 thole 2.600000\n"),
            "{text}"
        );
        assert!(!text.contains("drude_style"), "{text}");

        let coeffs = crate::io::writer::capture(|sink| {
            LammpsForcefieldWriter::new(
                &crate::io::lammps::lammps_type_labels(&drude_frame()).unwrap(),
            )
            .with_skip_special_bonds(true)
            .write_data_coeffs(sink, &drude_ff())
        })
        .unwrap();
        assert_eq!(
            data_section_rows(&coeffs, "Bond Coeffs # harmonic"),
            ["1 500.000000 0.000000", "2 340.000000 1.090000"]
        );
    }

    /// The bond ids of the coefficients are the data file's.
    #[test]
    fn drude_bond_ids_agree_with_the_data_file() {
        let mut frame = drude_frame();
        let atoms = frame.get_mut("atoms").unwrap();
        for (key, values) in [
            ("x", vec![0.0, 0.0, 1.0, -1.0]),
            ("y", vec![0.0; 4]),
            ("z", vec![0.0; 4]),
        ] {
            atoms
                .insert(key, ndarray::Array1::from_vec(values).into_dyn(), None)
                .unwrap();
        }
        atoms
            .insert(
                "mol_id",
                ndarray::Array1::from_vec(vec![1_u64; 4]).into_dyn(),
                None,
            )
            .unwrap();
        for (block, pairs) in [
            ("bonds", [(0_u64, 2_u64), (0, 3)].as_slice()),
            ("drudes", &[(0, 1)]),
        ] {
            let b = frame.get_mut(block).unwrap();
            let (i, j): (Vec<u64>, Vec<u64>) = pairs.iter().copied().unzip();
            b.insert("atomi", ndarray::Array1::from_vec(i).into_dyn(), None)
                .unwrap();
            b.insert("atomj", ndarray::Array1::from_vec(j).into_dyn(), None)
                .unwrap();
        }
        let data =
            crate::io::writer::capture(|sink| crate::io::write_lammps_data(sink, &frame)).unwrap();
        assert!(
            data.contains("Bond Type Labels\n\n1 CT-DCT\n2 CT-HC\n"),
            "{data}"
        );
        assert!(
            data.contains("Bonds\n\n1 2 1 3\n2 2 1 4\n3 1 1 2\n"),
            "{data}"
        );
    }

    /// Without the Drude rows in the frame, the spring type is an unused
    /// force-field type: not written, and no comment.
    #[test]
    fn an_unused_drude_type_writes_nothing() {
        let mut frame = drude_frame();
        frame.remove("drudes");
        let text = crate::io::writer::capture(|sink| {
            LammpsForcefieldWriter::new(&crate::io::lammps::lammps_type_labels(&frame).unwrap())
                .with_skip_special_bonds(true)
                .write(sink, &drude_ff())
        })
        .unwrap();
        assert!(!text.contains("CT-DCT"), "{text}");
        assert!(!text.contains("DRUDE"), "{text}");
    }
}
