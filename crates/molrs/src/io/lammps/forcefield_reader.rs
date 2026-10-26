//! LAMMPS force-field reader (the `*.ff` include next to a data file).

use crate::core::constants::VACUUM_DIELECTRIC;
use crate::core::constants::{AMBER_SCEE, AMBER_SCNB};
use crate::ff::forcefield::ForceField;
use crate::ff::ir::CombiningRule;
use crate::ff::ir::{LammpsCodec, StyleSpec};
use crate::ff::ir::{Params, SpecialBonds};
use crate::ff::style_registry::{Registry, RegistryRef};
use crate::io::lammps::units::parse_lammps_units_style;
use crate::io::reader::ForceFieldReader;
use crate::io::{Source, invalid_data};
use molrs::core::FrameAccess;
use molrs::core::constants::{COULOMB_METAL, COULOMB_REAL};
use molrs::core::{TypeLabels, TypeName};
use ndarray::ArrayD;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// Id→label maps of a data file's `* Type Labels` sections.
#[derive(Debug, Clone, Default)]
pub(crate) struct LammpsTypeLabelMaps {
    pub(crate) atom: BTreeMap<u32, String>,
    pub(crate) bond: BTreeMap<u32, String>,
    pub(crate) angle: BTreeMap<u32, String>,
    pub(crate) dihedral: BTreeMap<u32, String>,
    pub(crate) improper: BTreeMap<u32, String>,
}

impl LammpsTypeLabelMaps {
    /// The maps `frame`'s type-label inventories declare, ids as written —
    /// what the LAMMPS data reader stored from the `* Type Labels` sections.
    fn from_frame(frame: &impl FrameAccess) -> Result<Self, String> {
        let map = |block: &str| -> Result<BTreeMap<u32, String>, String> {
            TypeLabels::declared_ids(frame, block)?
                .into_iter()
                .map(|(id, label)| {
                    u32::try_from(id)
                        .map(|id| (id, label))
                        .map_err(|_| format!("{block}: type id {id} is out of range"))
                })
                .collect()
        };
        Ok(Self {
            atom: map("atoms")?,
            bond: map("bonds")?,
            angle: map("angles")?,
            dihedral: map("dihedrals")?,
            improper: map("impropers")?,
        })
    }

    /// The id a coefficient row's leading token names in `category`: the
    /// integer as written, or the id the category's `* Type Labels` section
    /// declared for the label (`pair` and `pairij` rows name atom types).
    fn type_id(&self, category: &str, token: &str) -> Option<u32> {
        if let Ok(id) = token.parse::<u32>() {
            return Some(id);
        }
        let map = match category {
            "pair" | "pairij" => &self.atom,
            "bond" => &self.bond,
            "angle" => &self.angle,
            "dihedral" => &self.dihedral,
            "improper" => &self.improper,
            _ => return None,
        };
        map.iter()
            .find(|(_, label)| label.as_str() == token)
            .map(|(id, _)| *id)
    }
}

/// Reader for a LAMMPS force-field include (`*.ff`), AMBER/GAFF flavour.
///
/// Parses a LAMMPS force-field include — `pair_style`/`pair_coeff` and the
/// `*_style`/`*_coeff` lines of every bonded category, a `hybrid` of styles
/// in any of them — with **type-label** coefficients into a molrs
/// [`ForceField`]. A LAMMPS style reads as the registered style whose
/// [`LammpsForm`](crate::ff::ir::LammpsForm) writes its name
/// ([`Registry::lammps_style`]), through that form's codec
/// (`ff-ir-02-protocol` §8): a positional style's coefficients are its
/// spec's `params` in order, so a style registered at run time with a LAMMPS
/// form reads with nothing else written. Inverse of
/// [`LammpsForcefieldWriter`](crate::io::lammps::LammpsForcefieldWriter), e.g.:
///
/// ```text
/// pair_style lj/cut/coul/cut 10.0 10.0
/// pair_coeff c3 c3 0.107800 3.397710          # epsilon sigma
/// bond_style harmonic
/// bond_coeff c3-c3 228.890000 1.535400        # K r0
/// angle_style harmonic
/// angle_coeff c3-c3-oh 76.790000 109.660000   # K theta0(deg)
/// dihedral_style fourier
/// dihedral_coeff c3-c3-oh-ho 1 0.060000 3 0.0 # m  K1 n1 d1(deg) [K2 n2 d2 ...]
/// ```
///
/// # The identity on coefficients
///
/// molrs's convention **is** LAMMPS's (the manual, "Force-field
/// conventions"): every style's energy expression, factors and parameter
/// units are the LAMMPS style's, with angle-valued parameters in degrees. So
/// this reader converts nothing: each coefficient is stored as written, under
/// the name of its slot, and the force field declares the file's `units`
/// (`real` when the file has no `units` line). One token → params
/// map does that for every reader. The only renames are of style names molrs spells
/// differently: `dihedral_style fourier` is molrs's `dihedral periodic`.
///
/// A `class2` style's cross-term lines (`angle_coeff t bb …`, `dihedral_coeff
/// t mbt …`, a data file's `BondBond Coeffs`, …) read when their force
/// constants are zero, and are refused otherwise: the force-field IR has no
/// cross terms.
///
/// A coefficient line carries exactly its style's coefficients: an extra
/// token is an error, as it is in LAMMPS, never a number dropped (an
/// `angle_coeff t K theta0 K_ub r_ub` line under `angle_style harmonic` would
/// otherwise lose its Urey–Bradley term).
///
/// # Hybrid bonded styles
///
/// `angle_style hybrid harmonic charmm` declares one molrs style per
/// sub-style, and each `angle_coeff t <sub-style> <coeffs…>` line is a type of
/// the sub-style it names (which must be one of the declared ones). A data
/// file's `Angle Coeffs # hybrid` section is the same with the sub-styles
/// declared by its rows.
///
/// # Pair rows
///
/// `pair_coeff i i ε σ` is atom type `i`'s own `lj/cut` row. A cross
/// `pair_coeff i j ε σ` (`i ≠ j`; CHARMM NBFIX and the like) is an explicit
/// pair type between `i` and `j`, which the `lj/cut` kernel uses for that pair
/// in place of the mixing rule. A later line for the same pair, in either
/// order, replaces an earlier one, as in LAMMPS. A cross line with a wildcard
/// (`pair_coeff c3 * …`) is refused rather than expanded.
///
/// # Pair styles
///
/// `pair_style lj/cut` is `lj/cut` alone (LAMMPS prices no charge under it);
/// `lj/cut/coul/cut` is `lj/cut` with `coul/cut`; `lj/cut/coul/long` is
/// `lj/cut` with `coul/long/pme` at its cutoff and LAMMPS's Coulomb
/// constant. Any other pair style reads through its codec (`buck`, `morse`,
/// `lj/class2`, a style registered with a LAMMPS form), alone, as a `hybrid`
/// of such styles, or as a `hybrid/overlay` with `coul/cut` / `coul/long` on
/// `* *`; its `mixing` is `pair_modify mix`, else LAMMPS's `geometric`. The
/// Ewald parameters of the last are the input script's
/// `kspace_style` accuracy, not an `alpha`, so they are not read and the style
/// prices nothing until a caller states them. A `hybrid` / `hybrid/overlay`
/// of `lj/cut` with `coul/cut` or `coul/long` reads the same way; accelerator
/// suffixes (`/omp`, `/kk`, …) are dropped. Any other Coulomb (`coul/debye`,
/// `coul/dsf`, `coul/wolf`, …) is refused. `pair_modify mix <rule>` is the
/// `mixing` and `pair_modify shift yes` the `shift` of `lj/cut` (refused
/// under the switched CHARMM style).
///
/// # CHARMM pair style
///
/// `pair_style lj/charmm/coul/charmm inner outer [inner2 outer2]` is molrs's
/// `lj/charmm` (per type `epsilon sigma epsilon14 sigma14`; a two-number
/// `pair_coeff` stores its 1-4 pair equal to the regular one, as LAMMPS reads
/// it) plus `coul/charmm`, each with its `inner` / `cutoff`; mixing is
/// LAMMPS's `arithmetic` unless `pair_modify mix` says otherwise.
/// `lj/charmm/coul/long` is refused (an Ewald real-space Coulomb).
///
/// # Charges and masses
///
/// Per-atom charge and mass live in the LAMMPS **data** file, not this include,
/// so they are not read here: the `coul/cut` style draws charges from the
/// [`Frame`](molrs::core::Frame) at evaluation time, with LAMMPS's own
/// Coulomb constant (`qqr2e`) for the file's units.
///
/// # CMAP crossterms (`fix cmap`)
///
/// A `fix <id> <group> cmap <file>` line reads `<file>` (relative to the
/// include's directory when the include is read from a path, else to the
/// working directory) as [`read_cmap`](LammpsForcefieldReader::read_cmap)
/// does into a `cmap charmm` style: map `t` of the file is
/// the row named `"t"` — the crossterm type a data file's `CMAP` section
/// gives — with the synthetic endpoints `t-t-t-t-t`. A `fix_modify` line for
/// that fix is accepted and changes nothing; any other fix style is refused.
///
/// 1-4 weights are **declared** on a `special_bonds` line and stored on
/// [`ForceField::special_bonds`](crate::ff::forcefield::ForceField::special_bonds)
/// (dimensionless `[1-2, 1-3, 1-4]`). An include that omits the line is an
/// error — LAMMPS's own default (`0 0 0`) is not AMBER's weights, so this
/// reader will not invent either. Data-file `read_data_coeffs` synthesizes
/// an explicit AMBER-like line so those reads keep the 0.5 / 5/6 they have
/// always produced.
#[derive(Debug, Clone)]
pub struct LammpsForcefieldReader {
    /// Used when the file has no `units` line. Molecular includes default to
    /// **real** (LAMMPS bare-script default is `lj` — pass `default_units: Lj`
    /// or write an explicit `units` line when that matters).
    pub default_units: &'static str,
    /// The registry whose styles' LAMMPS forms the file reads through: the
    /// process-wide one unless [`with_registry`](Self::with_registry) gives
    /// another.
    pub registry: RegistryRef,
}

impl Default for LammpsForcefieldReader {
    fn default() -> Self {
        Self {
            default_units: "real",
            registry: RegistryRef::Global,
        }
    }
}

impl LammpsForcefieldReader {
    /// The reader: `real` units for a file without a `units` line, the
    /// process-wide registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Read each style through its codec in `registry` instead of the
    /// process-wide one; the force field read carries it
    /// ([`ForceField::registry`]).
    pub fn with_registry(mut self, registry: Arc<Registry>) -> Self {
        self.registry = RegistryRef::Own(registry);
        self
    }

    /// The force field a LAMMPS data file's `* Coeffs` sections define, from
    /// the frame the data reader returned
    /// ([`read_lammps_data`](crate::io::read_lammps_data)) — the file is read
    /// once, for the system and its coefficients alike.
    ///
    /// The sections are the frame's
    /// [`LAMMPS_COEFFS_TEXT`](crate::core::keys::LAMMPS_COEFFS_TEXT)
    /// meta; a row's numeric type id is named by the label the file's
    /// `* Type Labels` section gave it (the frame's type-label inventories,
    /// ids as written), else by the id itself. `units` is the unit style the
    /// coefficients are in: the one the file's title line stated
    /// ([`LAMMPS_UNITS`](crate::core::keys::LAMMPS_UNITS)) when `None`,
    /// else [`default_units`](Self::default_units). A `PairIJ Coeffs` row
    /// `i j ε σ` is the `pair_coeff i j ε σ` line: with `i ≠ j` an explicit
    /// cross pair that replaces the mixing rule for that type pair.
    ///
    /// # Styles
    ///
    /// A data file has no `*_style` lines; `write_data` records each style as
    /// the section header's comment, e.g. `Bond Coeffs # harmonic/kk`. That
    /// hint selects the category's style, with an accelerator suffix (`/kk`,
    /// `/gpu`, `/omp`, `/intel`, `/opt`) removed. A hinted style no registered
    /// style reads (`cosine`, …) is an error naming the section and the
    /// style — the numbers are never read under another kernel. The
    /// `class2` cross-term sections (`BondBond Coeffs`, …) read as their
    /// `*_coeff` lines.
    ///
    /// A section **without** a hint, and a category with no section, fall back
    /// to `harmonic` (bond, angle, dihedral, improper) and `lj/cut` (pair).
    /// The pair cutoff is always 10.0 in the file's units: a data file does not
    /// carry one. The coefficients are stored as written, in `units`.
    ///
    /// # Errors
    ///
    /// [`InvalidData`](std::io::ErrorKind::InvalidData), as the frame doors
    /// fail: a frame with no `* Coeffs` sections, a `units` that disagrees
    /// with the one the file stated, a malformed type-label inventory, an
    /// unsupported hinted style, plus every error of the `*_coeff` parse.
    pub fn read_data_coeffs(
        &self,
        frame: &impl FrameAccess,
        units: Option<&str>,
    ) -> std::io::Result<ForceField> {
        self.data_coeffs(frame, units).map_err(invalid_data)
    }

    /// [`read_data_coeffs`](Self::read_data_coeffs), its error the message.
    fn data_coeffs(
        &self,
        frame: &impl FrameAccess,
        units: Option<&str>,
    ) -> Result<ForceField, String> {
        use crate::core::keys::{LAMMPS_COEFFS_TEXT, LAMMPS_UNITS};
        let meta = frame.meta_ref();
        let text = meta
            .get(LAMMPS_COEFFS_TEXT)
            .ok_or_else(|| {
                format!(
                    "the frame has no `* Coeffs` sections (meta {LAMMPS_COEFFS_TEXT:?}): \
                     read it with the LAMMPS data reader from a file that has them"
                )
            })?
            .as_str()
            .ok_or_else(|| format!("meta {LAMMPS_COEFFS_TEXT:?} must be a string"))?;
        let stated = match meta.get(LAMMPS_UNITS) {
            None => None,
            Some(value) => {
                Some(parse_lammps_units_style(value.as_str().ok_or_else(
                    || format!("meta {LAMMPS_UNITS:?} must be a string"),
                )?)?)
            }
        };
        let units = match (units.map(parse_lammps_units_style).transpose()?, stated) {
            (Some(given), Some(stated)) if given != stated => {
                return Err(format!(
                    "units {given:?} disagree with the data file's own `units = {stated}`"
                ));
            }
            (Some(units), _) | (None, Some(units)) => units,
            (None, None) => self.default_units,
        };
        let labels = LammpsTypeLabelMaps::from_frame(frame)?;
        self.read_data_sections(text, &labels, units)
    }

    /// [`read_data_coeffs`](Self::read_data_coeffs) of the `* Coeffs` text
    /// itself, with the `* Type Labels` maps and the unit style.
    pub(crate) fn read_data_sections(
        &self,
        coeffs_text: &str,
        labels: &LammpsTypeLabelMaps,
        units: &str,
    ) -> Result<ForceField, String> {
        // Synthesize style lines so the shared dispatcher can run, then parse
        // section-form coeff lines rewritten as command-form.
        let mut synthetic = String::new();
        synthetic.push_str(&format!("units {units}\n"));
        // Data files carry no special_bonds; keep the AMBER weights this path
        // has always assumed, as an explicit declaration. Built from the same
        // constants as the `amber` preset so the two cannot drift apart.
        synthetic.push_str(&format!(
            "special_bonds lj 0.0 0.0 {} coul 0.0 0.0 {}\n",
            1.0 / AMBER_SCNB,
            1.0 / AMBER_SCEE
        ));
        let (hints, commands) = data_sections_to_commands(coeffs_text, labels)?;
        for category in ["pair", "bond", "angle", "dihedral", "improper"] {
            let style = match hints.get(category) {
                Some(hint) => {
                    self.registry
                        .with(|reg| hint.require_supported(reg, category))?;
                    hint.style.as_str()
                }
                // A data file without a hint: the AMBER-style default above,
                // charges priced by a plain Coulomb.
                None if category == "pair" => "lj/cut/coul/cut",
                None => "harmonic",
            };
            if category == "pair" {
                synthetic.push_str(&format!("pair_style {style} {DATA_PAIR_CUTOFF}\n"));
            } else {
                synthetic.push_str(&format!("{category}_style {style}\n"));
            }
        }
        synthetic.push_str(&commands);
        self.read_str_with_labels(&synthetic, labels, None)
    }

    /// Read the LAMMPS `fix cmap` file (CHARMM layout) `src` holds — a path
    /// or a whole file's bytes — into a force field of one `cmap charmm`
    /// style: map `t` (1-based, the crossterm type of a data file's `CMAP`
    /// section) is the row named `"t"`, with the synthetic endpoints
    /// `t-t-t-t-t` (the file names no atom types), its `grid` the map as
    /// written. The inverse of
    /// [`LammpsForcefieldWriter::write_cmap`](crate::io::lammps::LammpsForcefieldWriter::write_cmap).
    ///
    /// The file is read as `FixCMAP::read_grid_map` reads it: every `#`
    /// starts a comment to the end of its line; what is left is
    /// whitespace-separated numbers, 576 (24 × 24) per map, maps one after
    /// the other in φ-major order. The force field is in the units of the
    /// first line's `UNITS: <style>` tag, or
    /// [`default_units`](Self::default_units) when it has none.
    ///
    /// ```
    /// use molrs::io::lammps::LammpsForcefieldReader;
    ///
    /// let text = format!("# UNITS: real\n# map 1\n{}", "0.5\n".repeat(576));
    /// let ff = LammpsForcefieldReader::new().read_cmap(text.as_bytes()).unwrap();
    /// assert_eq!(ff.units(), Some("real"));
    /// assert!(ff.get_style("cmap", "charmm").is_some());
    /// ```
    ///
    /// # Errors
    ///
    /// The OS error of a path that cannot be read;
    /// [`InvalidData`](std::io::ErrorKind::InvalidData) where LAMMPS is
    /// lenient and reading is total — a trailing incomplete map, a line whose
    /// values run past the end of a map, a seventh map — and for a
    /// non-numeric token, a file with no map, and a `UNITS:` tag that is no
    /// LAMMPS unit style.
    pub fn read_cmap<'a>(&self, src: impl Into<Source<'a>>) -> std::io::Result<ForceField> {
        self.cmap(&src.into().text()?).map_err(invalid_data)
    }

    /// The force field of the `fix cmap` file `text` ([`read_cmap`](Self::read_cmap)).
    fn cmap(&self, text: &str) -> Result<ForceField, String> {
        let file = parse_cmap_file(text)?;
        let units = match &file.units {
            Some(units) => parse_lammps_units_style(units)?,
            None => self.default_units,
        };
        let mut ff = ForceField::new("LAMMPS");
        ff.registry = self.registry.clone();
        add_cmap_rows(&mut ff, &file.maps)?;
        ff.set_units(units);
        Ok(ff)
    }

    fn read_str_with_labels(
        &self,
        text: &str,
        labels: &LammpsTypeLabelMaps,
        dir: Option<&Path>,
    ) -> Result<ForceField, String> {
        self.registry
            .with(|reg| self.read_with(reg, text, labels, dir))
    }

    fn read_with(
        &self,
        reg: &Registry,
        text: &str,
        labels: &LammpsTypeLabelMaps,
        dir: Option<&Path>,
    ) -> Result<ForceField, String> {
        let mut file_units = self.default_units;
        let mut ff = ForceField::new("LAMMPS");
        ff.registry = self.registry.clone();
        let mut pair_rows: Vec<PairRow> = Vec::new();
        let mut pair_line = PairLine::LjCut {
            lj: None,
            coul: Coul::Cut(None),
        };
        let mut pair_mix: Option<String> = None;
        let mut pair_shift = false;
        // The LAMMPS style each category's coefficient lines are read under.
        let mut styles: BTreeMap<&'static str, BondedStyle> = BTreeMap::new();
        let mut saw_special_bonds = false;
        // The `fix cmap` read so far: its fix id and its file's `UNITS:` tag.
        let mut cmap_fix: Option<(String, Option<String>)> = None;

        for (lineno, raw) in text.lines().enumerate() {
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            let mut tok = line.split_whitespace();
            let kw = tok.next().unwrap();
            let rest: Vec<&str> = tok.collect();
            let where_ = || format!("line {}", lineno + 1);

            match kw {
                "units" => {
                    let name = rest
                        .first()
                        .ok_or_else(|| format!("{}: units missing style name", where_()))?;
                    file_units =
                        parse_lammps_units_style(name).map_err(|e| format!("{}: {e}", where_()))?;
                }
                "pair_style" => pair_line = require_pair_style(reg, &rest, &where_)?,
                "bond_style" | "angle_style" | "dihedral_style" | "improper_style" => {
                    let category = kw.trim_end_matches("_style");
                    let category = BONDED.iter().copied().find(|c| *c == category).unwrap();
                    let name = rest
                        .first()
                        .ok_or_else(|| format!("{}: {kw} missing name", where_()))?;
                    let declared = if *name == "hybrid" {
                        let subs: Vec<String> = rest[1..].iter().map(|s| (*s).to_owned()).collect();
                        for sub in &subs {
                            def_bonded_style(&mut ff, reg, category, sub, &[], &where_)?;
                        }
                        BondedStyle::Hybrid(subs)
                    } else {
                        def_bonded_style(&mut ff, reg, category, name, &rest[1..], &where_)?;
                        BondedStyle::Single((*name).to_owned())
                    };
                    styles.insert(category, declared);
                }
                "pair_coeff" => {
                    collect_pair(reg, &rest, &pair_line, &mut pair_rows, &where_, labels)?
                }
                "bond_coeff" | "angle_coeff" | "dihedral_coeff" | "improper_coeff" => {
                    let category = kw.trim_end_matches("_coeff");
                    let category = BONDED.iter().copied().find(|c| *c == category).unwrap();
                    let declared = styles.get(category).ok_or_else(|| {
                        format!("{}: coeff before its `{category}_style`", where_())
                    })?;
                    add_bonded(&mut ff, reg, category, declared, &rest, &where_, labels)?;
                }
                "special_bonds" => {
                    ff.set_special_bonds(parse_special_bonds(&rest, &where_)?);
                    saw_special_bonds = true;
                }
                "pair_modify" => {
                    if let Some(i) = rest.iter().position(|t| *t == "mix") {
                        let rule = rest.get(i + 1).ok_or_else(|| {
                            format!("{}: pair_modify mix missing a rule", where_())
                        })?;
                        CombiningRule::parse(rule).map_err(|e| format!("{}: {e}", where_()))?;
                        pair_mix = Some((*rule).to_owned());
                    }
                    if let Some(i) = rest.iter().position(|t| *t == "shift") {
                        pair_shift = match rest.get(i + 1) {
                            Some(&"yes") => true,
                            Some(&"no") => false,
                            other => {
                                return Err(format!(
                                    "{}: pair_modify shift takes yes or no, got {other:?}",
                                    where_()
                                ));
                            }
                        };
                    }
                }
                "fix" => {
                    let [id, _group, style, args @ ..] = rest.as_slice() else {
                        return Err(format!(
                            "{}: fix needs an id, a group and a style",
                            where_()
                        ));
                    };
                    if *style != "cmap" {
                        return Err(format!(
                            "{}: unsupported fix style `{style}` (a force-field include \
                             holds only `fix cmap`)",
                            where_()
                        ));
                    }
                    let [file] = args else {
                        return Err(format!("{}: fix cmap takes one file name", where_()));
                    };
                    if cmap_fix.is_some() {
                        return Err(format!("{}: a second fix cmap", where_()));
                    }
                    let path = dir.map_or_else(|| Path::new(file).to_path_buf(), |d| d.join(file));
                    let text = std::fs::read_to_string(&path).map_err(|e| {
                        format!("{}: fix cmap file {}: {e}", where_(), path.display())
                    })?;
                    let maps = parse_cmap_file(&text)
                        .map_err(|e| format!("{}: {}: {e}", where_(), path.display()))?;
                    add_cmap_rows(&mut ff, &maps.maps)?;
                    cmap_fix = Some(((*id).to_owned(), maps.units));
                }
                "fix_modify" => {
                    let id = rest.first().copied().unwrap_or_default();
                    if cmap_fix.as_ref().is_none_or(|(cmap, _)| cmap != id) {
                        return Err(format!(
                            "{}: fix_modify of `{id}`, which is no fix cmap read before it",
                            where_()
                        ));
                    }
                }
                "atom_style" | "kspace_style" => {}
                other => return Err(format!("{}: unknown LAMMPS keyword `{other}`", where_())),
            }
        }
        if !saw_special_bonds {
            return Err(
                "special_bonds declaration is missing; 1-4 weights cannot be invented".into(),
            );
        }
        if let Some((_, Some(tag))) = &cmap_fix
            && tag != file_units
        {
            return Err(format!(
                "the fix cmap file states UNITS: {tag}, the include is in {file_units} \
                 (LAMMPS refuses the file)"
            ));
        }

        build_pairs(
            &mut ff,
            reg,
            &pair_rows,
            pair_line,
            pair_mix.as_deref(),
            pair_shift,
            coulomb_constant(file_units),
        )?;
        // Every number is as the file wrote it, so the force field is in the
        // file's units.
        ff.set_units(file_units);
        Ok(ff)
    }
}

impl ForceFieldReader for LammpsForcefieldReader {
    /// Read the include `src` holds; a `fix cmap` file it names is found
    /// relative to the include's directory when `src` is a path.
    fn read<'a>(&self, src: impl Into<Source<'a>>) -> std::io::Result<ForceField> {
        let src = src.into();
        let text = src.text()?;
        let dir = src.path().and_then(Path::parent);
        self.read_str_with_labels(&text, &LammpsTypeLabelMaps::default(), dir)
            .map_err(invalid_data)
    }
}

// ── fix cmap ────────────────────────────────────────────────────────────────

/// The map size LAMMPS `fix cmap` reads: 24×24, 15° spacing (`CMAPDIM`).
pub const LAMMPS_CMAP_DIM: usize = 24;

/// The most maps one `fix cmap` file holds (`CMAPMAX`).
pub const LAMMPS_CMAP_MAX: usize = 6;

/// The maps of a LAMMPS `fix cmap` file (CHARMM format), as
/// [`parse_cmap_file`] reads them; the door,
/// [`LammpsForcefieldReader::read_cmap`], returns the force field they are.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LammpsCmapFile {
    /// The unit style of the first line's `UNITS: <style>` tag, as LAMMPS
    /// reads it (the word after a `UNITS:` token); `None` without one —
    /// LAMMPS's own `charmm36.cmap` spells it `UNITS:real`, which LAMMPS does
    /// not read as a tag either.
    pub units: Option<String>,
    /// The maps in file order — map `t` is crossterm type `t + 1` — each
    /// [`LAMMPS_CMAP_DIM`]² energies, φ-major (`[i][j]` at φ = −180° + 15°·i,
    /// ψ = −180° + 15°·j).
    pub maps: Vec<ArrayD<f64>>,
}

/// Parse a LAMMPS `fix cmap` file, as `FixCMAP::read_grid_map` reads it.
///
/// Every `#` starts a comment to the end of its line; what is left is
/// whitespace-separated numbers, 576 (24 × 24) per map, maps one after the
/// other in φ-major order (CHARMM writes each φ row under a `# <φ>` comment,
/// five values a line).
///
/// Reading is total where LAMMPS is lenient: a trailing incomplete map, a
/// line whose values run past the end of a map (LAMMPS drops the rest of the
/// line), and a seventh map (LAMMPS reads six and ignores the rest) are
/// refused, as are a non-numeric token and a file with no map.
pub(crate) fn parse_cmap_file(text: &str) -> Result<LammpsCmapFile, String> {
    const PER_MAP: usize = LAMMPS_CMAP_DIM * LAMMPS_CMAP_DIM;
    let units = text.lines().next().and_then(|first| {
        let mut words = first.split_whitespace();
        words.find(|w| *w == "UNITS:")?;
        words.next().map(str::to_owned)
    });
    let mut maps: Vec<ArrayD<f64>> = Vec::new();
    let mut values: Vec<f64> = Vec::with_capacity(PER_MAP);
    for (lineno, raw) in text.lines().enumerate() {
        let line = strip_comment(raw);
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        if maps.len() == LAMMPS_CMAP_MAX {
            return Err(format!(
                "line {}: a map past the {LAMMPS_CMAP_MAX} fix cmap reads",
                lineno + 1
            ));
        }
        for (k, token) in tokens.iter().enumerate() {
            let value = token
                .parse::<f64>()
                .map_err(|_| format!("line {}: {token:?} is not a number", lineno + 1))?;
            values.push(value);
            if values.len() == PER_MAP {
                if k + 1 < tokens.len() {
                    return Err(format!(
                        "line {}: values past the end of map {}, which fix cmap discards",
                        lineno + 1,
                        maps.len() + 1
                    ));
                }
                let grid = std::mem::replace(&mut values, Vec::with_capacity(PER_MAP));
                maps.push(
                    ArrayD::from_shape_vec(vec![LAMMPS_CMAP_DIM, LAMMPS_CMAP_DIM], grid)
                        .map_err(|e| e.to_string())?,
                );
            }
        }
    }
    if !values.is_empty() {
        return Err(format!(
            "map {} is incomplete: {}/{PER_MAP} values",
            maps.len() + 1,
            values.len()
        ));
    }
    if maps.is_empty() {
        return Err("no CMAP map in the file".into());
    }
    Ok(LammpsCmapFile { units, maps })
}

/// Define map `t` (0-based) of `maps` as the `cmap charmm` row `"t+1"`.
fn add_cmap_rows(ff: &mut ForceField, maps: &[ArrayD<f64>]) -> Result<(), String> {
    let style = ff
        .def_style("cmap", "charmm", Params::new())
        .map_err(|e| e.to_string())?;
    for (t, grid) in maps.iter().enumerate() {
        let name = (t + 1).to_string();
        let mut params = Params::new();
        params.set("grid", grid.clone());
        let ends = [name.as_str(); 5];
        style
            .def_type(&name, &ends, params)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Pair cutoff `read_data_coeffs` declares: a data file carries none.
const DATA_PAIR_CUTOFF: f64 = 10.0;

/// The bonded categories, in the order a `*_style` / `*_coeff` keyword names
/// them.
const BONDED: [&str; 4] = ["bond", "angle", "dihedral", "improper"];

/// LAMMPS's Coulomb constant `qqr2e` for a `units` style, which the
/// `coul/cut` style states.
fn coulomb_constant(units: &str) -> f64 {
    match units {
        "metal" => COULOMB_METAL,
        "lj" => 1.0,
        _ => COULOMB_REAL,
    }
}

/// The data-file sections of the `class2` cross-term lines: heading,
/// category, and the `*_coeff` keyword the section's rows are.
pub(crate) const CROSS_TERM_SECTIONS: [(&str, &str, &str); 7] = [
    ("BondBond Coeffs", "angle", "bb"),
    ("BondAngle Coeffs", "angle", "ba"),
    ("MiddleBondTorsion Coeffs", "dihedral", "mbt"),
    ("EndBondTorsion Coeffs", "dihedral", "ebt"),
    ("AngleTorsion Coeffs", "dihedral", "at"),
    ("AngleAngleTorsion Coeffs", "dihedral", "aat"),
    ("BondBond13 Coeffs", "dihedral", "bb13"),
];

/// The registered style the LAMMPS style `name` of `category` reads as,
/// with its codec, or an error naming it.
fn lammps_style<'r>(
    reg: &'r Registry,
    category: &str,
    name: &str,
    where_: &dyn Fn() -> String,
) -> Result<(&'r StyleSpec, &'r dyn LammpsCodec), String> {
    reg.lammps_style(category, name).ok_or_else(|| {
        format!(
            "{}: unsupported {category}_style `{name}`: no registered style has this LAMMPS form \
             (molpy.ff.style_registry.GlobalRegistry().register_style with a LAMMPS form registers one)",
            where_()
        )
    })
}

/// A data-file `* Coeffs` section header's `# <style>` comment.
#[derive(Debug)]
struct SectionStyleHint {
    /// The header line as written, e.g. `Bond Coeffs # fene/kk`.
    header: String,
    /// The style with any accelerator suffix removed, e.g. `fene`.
    style: String,
}

impl SectionStyleHint {
    /// The hint on `header`, or `None` when it has no `# <style>` comment.
    fn parse(header: &str) -> Option<Self> {
        let (_, comment) = header.split_once('#')?;
        let raw = comment.split_whitespace().next()?;
        let style = ["/kk", "/gpu", "/omp", "/intel", "/opt"]
            .iter()
            .find_map(|suffix| raw.strip_suffix(suffix))
            .unwrap_or(raw);
        Some(Self {
            header: header.trim().to_owned(),
            style: style.to_owned(),
        })
    }

    /// Refuse a style the `category`'s `*_style` directive would refuse, with
    /// the error naming this section.
    fn require_supported(&self, reg: &Registry, category: &str) -> Result<(), String> {
        let where_ = || format!("`{}` section", self.header);
        let style = self.style.as_str();
        match category {
            "pair" => require_pair_style(reg, &[style, &DATA_PAIR_CUTOFF.to_string()], &where_)
                .map(|_| ()),
            // Each row names its sub-style, checked as the row is read.
            _ if style == "hybrid" => Ok(()),
            _ => lammps_style(reg, category, style, &where_).map(|_| ()),
        }
    }
}

/// Rewrite data-file section blocks into `*_coeff` command lines, and collect
/// each section header's `# <style>` hint by category.
fn data_sections_to_commands(
    text: &str,
    labels: &LammpsTypeLabelMaps,
) -> Result<(BTreeMap<&'static str, SectionStyleHint>, String), String> {
    let mut out = String::new();
    let mut hints = BTreeMap::new();
    let mut section: Option<&str> = None;
    // The `*_coeff` keyword of a cross-term section's rows.
    let mut cross_keyword: Option<&str> = None;
    for (lineno, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        let cross = CROSS_TERM_SECTIONS
            .iter()
            .find(|(heading, ..)| lower.starts_with(&heading.to_ascii_lowercase()));
        if let Some(&(_, category, keyword)) = cross {
            section = Some(category);
            cross_keyword = Some(keyword);
            continue;
        }
        let opened = [
            ("pair coeffs", "pair"),
            ("pairij coeffs", "pairij"),
            ("bond coeffs", "bond"),
            ("angle coeffs", "angle"),
            ("dihedral coeffs", "dihedral"),
            ("improper coeffs", "improper"),
        ]
        .into_iter()
        .find(|(name, _)| lower.starts_with(name));
        if let Some((_, kind)) = opened {
            section = Some(kind);
            cross_keyword = None;
            // `PairIJ Coeffs` is the pair category: one hint for both forms.
            let category = if kind == "pairij" { "pair" } else { kind };
            match SectionStyleHint::parse(raw) {
                Some(hint) => hints.insert(category, hint),
                None => hints.remove(category),
            };
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        // A coefficient row leads with a type id or a declared type label
        // (LAMMPS ≥ 2022) and carries at least one number; a line with no
        // number at all is the header of a section this pass does not read.
        let is_row =
            parts[0].parse::<u32>().is_ok() || parts[1..].iter().any(|p| p.parse::<f64>().is_ok());
        if !is_row {
            section = None;
            continue;
        }
        let Some(kind) = section else {
            continue;
        };
        let id = labels.type_id(kind, parts[0]).ok_or_else(|| {
            format!(
                "line {}: `{}` is neither a type id nor a declared {kind} type label",
                lineno + 1,
                parts[0]
            )
        })?;
        let atom_label = |id: u32| {
            labels
                .atom
                .get(&id)
                .cloned()
                .unwrap_or_else(|| id.to_string())
        };
        let type_tok = match kind {
            "pair" | "pairij" => atom_label(id),
            "bond" => labels
                .bond
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("{id}-{id}")),
            "angle" => labels
                .angle
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("{id}-{id}-{id}")),
            "dihedral" => labels
                .dihedral
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("{id}-{id}-{id}-{id}")),
            "improper" => labels
                .improper
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("{id}-{id}-{id}-{id}")),
            _ => unreachable!(),
        };
        match kind {
            "pair" => {
                // Pair Coeffs: id values…  →  pair_coeff T T values…
                if parts.len() < 2 {
                    return Err(format!(
                        "line {}: Pair Coeffs needs `id` and its coefficients",
                        lineno + 1
                    ));
                }
                out.push_str(&format!(
                    "pair_coeff {type_tok} {type_tok} {}\n",
                    parts[1..].join(" ")
                ));
            }
            "pairij" => {
                // PairIJ Coeffs: i j values…  →  pair_coeff Ti Tj values…. A
                // row with i ≠ j is an explicit cross pair, kept as a pair
                // type of its own.
                if parts.len() < 3 {
                    return Err(format!(
                        "line {}: PairIJ Coeffs needs `i j` and its coefficients",
                        lineno + 1
                    ));
                }
                let j = labels.type_id("pair", parts[1]).ok_or_else(|| {
                    format!(
                        "line {}: `{}` is neither a type id nor a declared atom type label \
                         (PairIJ Coeffs)",
                        lineno + 1,
                        parts[1]
                    )
                })?;
                out.push_str(&format!(
                    "pair_coeff {type_tok} {} {}\n",
                    atom_label(j),
                    parts[2..].join(" ")
                ));
            }
            other => {
                // bond/angle/dihedral/improper: id params… → *_coeff TYPE params…
                // (a cross-term section's: *_coeff TYPE <keyword> params…)
                out.push_str(&format!("{other}_coeff {type_tok}"));
                if let Some(keyword) = cross_keyword {
                    out.push(' ');
                    out.push_str(keyword);
                }
                for p in &parts[1..] {
                    out.push(' ');
                    out.push_str(p);
                }
                out.push('\n');
            }
        }
    }
    Ok((hints, out))
}

// ── pair ──────────────────────────────────────────────────────────────────────

/// The pair kernel a `pair_style` line declares, with its cutoffs in the
/// file's length unit.
#[derive(Clone, Debug, PartialEq)]
enum PairLine {
    /// An `lj/cut…` style: molrs's `lj/cut` and its Coulomb half.
    LjCut { lj: Option<f64>, coul: Coul },
    /// `lj/charmm/coul/charmm inner outer [inner2 outer2]`: molrs's
    /// `lj/charmm` + `coul/charmm`, each with its `(inner, cutoff)`.
    Charmm { lj: (f64, f64), coul: (f64, f64) },
    /// Any other style through its codec, alone or as the sub-styles of a
    /// `hybrid` (`hybrid/overlay`, with a Coulomb style on `* *`).
    Styles { subs: Vec<PairSub>, coul: Coul },
}

/// One pair style read through its codec: its LAMMPS name, the molrs style
/// it reads as, and the style parameters its `pair_style` arguments state.
#[derive(Clone, Debug, PartialEq)]
struct PairSub {
    lammps: String,
    style: String,
    params: Params,
}

/// The Coulomb half of an `lj/cut…` pair line, with its cutoff.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Coul {
    /// None: `pair_style lj/cut` prices no charge.
    None,
    /// `coul/cut`: molrs's `coul/cut`.
    Cut(Option<f64>),
    /// `coul/long`: the real-space half of an Ewald sum, molrs's
    /// `coul/long/pme`. Its Ewald parameters are the input script's
    /// `kspace_style` (an accuracy, not an `alpha`), so they are not read, and
    /// the style prices nothing until a caller states them.
    Long(Option<f64>),
}

impl Coul {
    fn with_cutoff(self, cut: Option<f64>) -> Coul {
        match self {
            Coul::None => Coul::None,
            Coul::Cut(c) => Coul::Cut(c.or(cut)),
            Coul::Long(c) => Coul::Long(c.or(cut)),
        }
    }
}

/// A LAMMPS style name without its accelerator suffix (`/omp`, `/kk`, …).
fn without_accelerator(name: &str) -> &str {
    for suffix in ["/omp", "/opt", "/kk", "/gpu", "/intel"] {
        if let Some(base) = name.strip_suffix(suffix) {
            return base;
        }
    }
    name
}

/// The pair style `name` read through its codec, `args` its `pair_style`
/// arguments.
fn pair_sub(
    reg: &Registry,
    name: &str,
    args: &[&str],
    where_: &dyn Fn() -> String,
) -> Result<PairSub, String> {
    let (spec, codec) = lammps_style(reg, "pair", name, where_)?;
    let params = codec
        .parse_style_args(spec, args)
        .map_err(|e| format!("{}: {e}", where_()))?;
    Ok(PairSub {
        lammps: name.to_owned(),
        style: spec.name.to_string(),
        params,
    })
}

/// Validate the pair kernel and return what it declares.
///
/// Three spellings map to the reader's lj/cut + coul/cut pair:
///
/// - the combined kernel — `pair_style lj/cut/coul/cut 10.0 [12.0]`, Coulomb
///   cutoff defaulting to the LJ one when omitted;
/// - `hybrid lj/cut 10.0 coul/cut 10.0`, one pair per sub-style;
/// - `hybrid/overlay lj/cut 10.0 coul/cut 10.0`, both on every pair — what this
///   reader's own force field means, so its writer emits it.
///
/// `pair_style lj/cut` alone is `lj/cut` with no Coulomb style (LAMMPS prices
/// no charge under it); `lj/cut/coul/long` (and a hybrid `coul/long`) is
/// `lj/cut` + `coul/long/pme` with the Coulomb cutoff and constant, its Ewald
/// parameters left for the caller (the `kspace_style` states an accuracy).
/// Any other `lj/cut/coul/…` Coulomb (`debye`, `dsf`, `wolf`, …) is refused.
///
/// `lj/charmm/coul/charmm inner outer [inner2 outer2]` maps to `lj/charmm` +
/// `coul/charmm` with LAMMPS's switching cutoffs (the Coulomb pair the LJ
/// one when only two are given). Its `coul/long` sibling is refused: its
/// Coulomb is switched in real space, which no molrs kernel is.
///
/// Every other name is the registered style whose LAMMPS form writes it, its
/// arguments read by its codec (`pair_style buck 10.0`); a `hybrid` of such
/// styles reads each sub-style so.
///
/// The cutoffs are part of the force field, not a rendering detail: a reader
/// that keeps only the kernel name cannot write a runnable input back out.
fn require_pair_style(
    reg: &Registry,
    rest: &[&str],
    where_: &dyn Fn() -> String,
) -> Result<PairLine, String> {
    let name = rest
        .first()
        .ok_or_else(|| format!("{}: pair_style missing kernel name", where_()))?;
    if *name == "hybrid" || *name == "hybrid/overlay" {
        return hybrid_pair(reg, &rest[1..], where_);
    }
    let name = without_accelerator(name);
    let numbers = || {
        rest[1..]
            .iter()
            .map(|t| parse_f64(t, "pair_style cutoff", where_))
            .collect::<Result<Vec<_>, _>>()
    };
    if name == "lj/charmm/coul/charmm" {
        return match *numbers()?.as_slice() {
            [a, b] => Ok(PairLine::Charmm {
                lj: (a, b),
                coul: (a, b),
            }),
            [a, b, c, d] => Ok(PairLine::Charmm {
                lj: (a, b),
                coul: (c, d),
            }),
            _ => Err(format!(
                "{}: pair_style lj/charmm/coul/charmm takes `inner outer [inner2 outer2]` \
                 (a data file's section hint carries no switching cutoffs; declare the \
                 style in an include)",
                where_()
            )),
        };
    }
    if name.starts_with("lj/charmm") {
        return Err(format!(
            "{}: unsupported pair_style `{name}`: of the CHARMM pair styles molrs reads \
             `lj/charmm/coul/charmm` (its Coulomb is plain and switched; a `coul/long` \
             Coulomb is the real-space half of an Ewald sum)",
            where_()
        ));
    }
    if name == "lj/cut" || name.starts_with("lj/cut/coul/") {
        let nums = numbers()?;
        let lj = nums.first().copied();
        let cut = nums.get(1).copied().or(lj);
        let coul = match name {
            "lj/cut" => Coul::None,
            "lj/cut/coul/cut" => Coul::Cut(cut),
            "lj/cut/coul/long" => Coul::Long(cut),
            _ => {
                return Err(format!(
                    "{}: unsupported pair_style `{name}` (of the lj/cut Coulomb styles molrs \
                     reads `lj/cut/coul/cut` and `lj/cut/coul/long`)",
                    where_()
                ));
            }
        };
        return Ok(PairLine::LjCut { lj, coul });
    }
    if matches!(name, "coul/cut" | "coul/long" | "coul/charmm") {
        return Err(format!(
            "{}: unsupported pair_style `{name}` alone (a Coulomb style is read beside the \
             van der Waals one: `lj/cut/{name}`, or a `hybrid/overlay`)",
            where_()
        ));
    }
    Ok(PairLine::Styles {
        subs: vec![pair_sub(reg, name, &rest[1..], where_)?],
        coul: Coul::None,
    })
}

/// A `hybrid` / `hybrid/overlay` pair line, e.g. `lj/cut 10.0 coul/cut
/// 10.0`: each sub-style and the numbers after it. `lj/cut` with `coul/cut`
/// or `coul/long` is the [`PairLine::LjCut`] pair (a sub-style's cutoff is
/// then optional, LAMMPS's global default); any other sub-style reads
/// through its codec.
fn hybrid_pair(
    reg: &Registry,
    rest: &[&str],
    where_: &dyn Fn() -> String,
) -> Result<PairLine, String> {
    let mut subs: Vec<(&str, Vec<&str>)> = Vec::new();
    for tok in rest {
        if tok.parse::<f64>().is_ok() {
            let Some((_, args)) = subs.last_mut() else {
                return Err(format!(
                    "{}: pair_style hybrid: a number `{tok}` before any sub-style",
                    where_()
                ));
            };
            args.push(tok);
        } else {
            subs.push((without_accelerator(tok), Vec::new()));
        }
    }
    let number = |args: &[&str]| args.first().and_then(|t| t.parse::<f64>().ok());
    let mut coul = Coul::None;
    let mut typed: Vec<(&str, Vec<&str>)> = Vec::new();
    for (name, args) in subs {
        match name {
            "coul/cut" => coul = Coul::Cut(number(&args)),
            "coul/long" => coul = Coul::Long(number(&args)),
            _ => typed.push((name, args)),
        }
    }
    if let [("lj/cut", args)] = typed.as_slice() {
        let lj = number(args);
        return Ok(PairLine::LjCut {
            lj,
            coul: coul.with_cutoff(lj),
        });
    }
    let subs = typed
        .iter()
        .map(|(name, args)| pair_sub(reg, name, args, where_))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PairLine::Styles { subs, coul })
}

/// One `pair_coeff` row: the two atom types (equal for a self pair), the
/// molrs style it is a type of, and its parameters.
type PairRow = (String, String, String, Params);

fn collect_pair(
    reg: &Registry,
    rest: &[&str],
    line: &PairLine,
    rows: &mut Vec<PairRow>,
    where_: &dyn Fn() -> String,
    labels: &LammpsTypeLabelMaps,
) -> Result<(), String> {
    // pair_coeff <i> <j> [sub-style] <values…>. A self pair i==j is an atom
    // type's own row; a cross pair i!=j is an explicit override (NBFIX) that
    // the kernel uses in place of the combining rule. A later line for the same
    // pair replaces an earlier one, as in LAMMPS.
    if rest.len() < 2 {
        return Err(format!("{}: pair_coeff needs `<i> <j> ...`", where_()));
    }
    let ti = resolve_atom_type(rest[0], labels, where_)?;
    let tj = resolve_atom_type(rest[1], labels, where_)?;
    let mut args = &rest[2..];
    // A hybrid line names its sub-style before the numbers: `c3 c3 lj/cut …`.
    // The `* * coul/cut` wildcard (charges come from the frame) has nothing
    // to transcribe. The sub-style must be one the line declared, not any
    // non-number (`notanumber` would otherwise be skipped silently).
    let mut sub: Option<&str> = None;
    if let Some(&first) = args.first()
        && first.parse::<f64>().is_err()
    {
        if matches!(first, "coul/cut" | "coul/long") {
            return Ok(());
        }
        sub = Some(first);
        args = &args[1..];
    }
    let unexpected = |tok: &str| {
        format!(
            "{}: pair_coeff unexpected token `{tok}` (not a sub-style of the pair_style line)",
            where_()
        )
    };
    let style = match line {
        PairLine::LjCut { .. } => match sub {
            None | Some("lj/cut") => "lj/cut".to_owned(),
            Some(other) => return Err(unexpected(other)),
        },
        PairLine::Charmm { .. } => match sub {
            None => "lj/charmm".to_owned(),
            Some(other) => return Err(unexpected(other)),
        },
        PairLine::Styles { subs, .. } => match (sub, subs.as_slice()) {
            (None, [one]) => one.style.clone(),
            (None, _) => {
                return Err(format!(
                    "{}: pair_coeff under a hybrid pair_style names no sub-style",
                    where_()
                ));
            }
            (Some(name), _) => subs
                .iter()
                .find(|s| s.lammps == name)
                .ok_or_else(|| unexpected(name))?
                .style
                .clone(),
        },
    };
    if ti != tj && (ti.contains('*') || tj.contains('*')) {
        return Err(format!(
            "{}: pair_coeff `{ti} {tj}` is a wildcard cross pair; expand it to explicit \
             type pairs",
            where_()
        ));
    }
    let (spec, _) = reg
        .style("pair", &style)
        .ok_or_else(|| format!("{}: pair style `{style}` is not registered", where_()))?;
    let codec = spec.lammps.require(spec).map_err(|e| e.to_string())?;
    let params = codec
        .read(spec, args)
        .map_err(|e| format!("{}: {e}", where_()))?;
    let same =
        |(a, b, s, _): &PairRow| s == &style && ((a == &ti && b == &tj) || (a == &tj && b == &ti));
    match rows.iter_mut().find(|row| same(row)) {
        Some(row) => *row = (ti, tj, style, params),
        None => rows.push((ti, tj, style, params)),
    }
    Ok(())
}

/// Emit the collected pair rows as their styles, plus the Coulomb style the
/// line declares (charges resolved from the frame).
///
/// `coul/cut` is the **buffered** Coulomb `E = k·qᵢqⱼ/(D·(r + δ))`; a LAMMPS
/// force field is the unbuffered case (δ = 0) in vacuum with LAMMPS's `k`
/// (`qqr2e`) for its units. The force field states the constant explicitly:
/// the kernel has no default, because MMFF's `k` is a different number and
/// both are correct.
fn build_pairs(
    ff: &mut ForceField,
    reg: &Registry,
    rows: &[PairRow],
    line: PairLine,
    mix: Option<&str>,
    shift: bool,
    coulomb: f64,
) -> Result<(), String> {
    if rows.is_empty() {
        return Ok(());
    }
    // 1-4 scaling lives on the ForceField's `special_bonds` (set in `read_str`),
    // not on the pair styles — `PotentialCompiler::compile` projects it into the kernels.
    let coul_style = |coul: Coul| -> Option<(&'static str, Params)> {
        let mut p = vec![("coulomb", coulomb), ("dielectric", VACUUM_DIELECTRIC)];
        let name = match coul {
            Coul::None => return None,
            Coul::Cut(c) => {
                p.extend(c.map(|c| ("cutoff", c)));
                "coul/cut"
            }
            Coul::Long(c) => {
                p = vec![("coulomb", coulomb)];
                p.extend(c.map(|c| ("cutoff", c)));
                "coul/long/pme"
            }
        };
        Some((name, Params::from_iter(p.iter().copied())))
    };
    // LAMMPS mixes **geometrically** unless `pair_modify mix` says otherwise,
    // and the CHARMM styles **arithmetically**; record it explicitly rather
    // than inherit a kernel default.
    let mut styles: Vec<(String, Params)> = Vec::new();
    let coulomb_style = match line {
        PairLine::LjCut { lj, coul } => {
            let mut p = Params::from_iter(
                lj.map(|c| vec![("cutoff", c)])
                    .unwrap_or_default()
                    .iter()
                    .copied(),
            );
            if shift {
                p.set("shift", 1.0);
            }
            p.set("mixing", mix.unwrap_or("geometric"));
            styles.push(("lj/cut".into(), p));
            coul_style(coul)
        }
        PairLine::Charmm { lj, coul } => {
            if shift {
                return Err(
                    "pair_modify shift yes under lj/charmm/coul/charmm: its switch already \
                     takes the energy to zero at the cutoff"
                        .into(),
                );
            }
            let mut p = Params::from_iter([("inner", lj.0), ("cutoff", lj.1)]);
            p.set("mixing", mix.unwrap_or("arithmetic"));
            styles.push(("lj/charmm".into(), p));
            Some((
                "coul/charmm",
                Params::from_iter([
                    ("coulomb", coulomb),
                    ("dielectric", VACUUM_DIELECTRIC),
                    ("inner", coul.0),
                    ("cutoff", coul.1),
                ]),
            ))
        }
        PairLine::Styles { subs, coul } => {
            for sub in subs {
                let (spec, codec) = reg
                    .lammps_style("pair", &sub.lammps)
                    .expect("a sub-style read through its codec");
                let mut p = sub.params;
                if spec.style_param("mixing").is_some() {
                    p.set(
                        "mixing",
                        codec.fixed_mixing().or(mix).unwrap_or("geometric"),
                    );
                }
                if shift {
                    if spec.style_param("shift").is_none() {
                        return Err(format!(
                            "pair_modify shift yes under `{}`: molrs's style has no shift",
                            sub.lammps
                        ));
                    }
                    p.set("shift", 1.0);
                }
                styles.push((sub.style, p));
            }
            coul_style(coul)
        }
    };

    for (name, params) in styles {
        let style = ff
            .def_style("pair", &name, params)
            .map_err(|e| e.to_string())?;
        for (ti, tj, _, params) in rows.iter().filter(|r| r.2 == name) {
            if ti == tj {
                style.def_type(ti, &[ti], params.clone())
            } else {
                let type_name = TypeName::pair(ti, tj)?;
                style.def_type(type_name.as_str(), &[ti, tj], params.clone())
            }
            .map_err(|e| e.to_string())?;
        }
    }
    if let Some((name, params)) = coulomb_style {
        ff.def_style("pair", name, params)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ── bonded ──────────────────────────────────────────────────────────────────

/// The LAMMPS style a bonded category's coefficient lines are read under.
enum BondedStyle {
    /// `<category>_style <name>`.
    Single(String),
    /// `<category>_style hybrid <sub-style>…`: each coefficient line names
    /// its sub-style. Empty for a data file's `# hybrid` section, whose rows
    /// declare them.
    Hybrid(Vec<String>),
}

/// Declare the style the LAMMPS bonded style `name` reads as (`args` its
/// `*_style` arguments), refusing one no registered style reads.
fn def_bonded_style(
    ff: &mut ForceField,
    reg: &Registry,
    category: &str,
    name: &str,
    args: &[&str],
    where_: &dyn Fn() -> String,
) -> Result<(), String> {
    let (spec, codec) = lammps_style(reg, category, name, where_)?;
    let params = codec
        .parse_style_args(spec, args)
        .map_err(|e| format!("{}: {e}", where_()))?;
    ff.def_style(category, &spec.name, params)
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// One `<category>_coeff <type> [<sub-style>] [<keyword>] <values…>` line
/// under its LAMMPS style: a type's coefficients, or (after a keyword the
/// style's codec writes, `class2`'s `bb`, `mbt`, …) a cross-term line it
/// checks.
fn add_bonded(
    ff: &mut ForceField,
    reg: &Registry,
    category: &'static str,
    declared: &BondedStyle,
    rest: &[&str],
    where_: &dyn Fn() -> String,
    labels: &LammpsTypeLabelMaps,
) -> Result<(), String> {
    let (name, endpoints) = match category {
        "bond" => {
            let (name, e) = label_type::<2>(rest.first(), category, where_, Some(&labels.bond))?;
            (name, e.to_vec())
        }
        "angle" => {
            let (name, e) = label_type::<3>(rest.first(), category, where_, Some(&labels.angle))?;
            (name, e.to_vec())
        }
        "dihedral" => {
            let map = Some(&labels.dihedral);
            let (name, e) = label_type::<4>(rest.first(), category, where_, map)?;
            (name, e.to_vec())
        }
        _ => {
            let map = Some(&labels.improper);
            let (name, e) = label_type::<4>(rest.first(), category, where_, map)?;
            (name, e.to_vec())
        }
    };
    let (lammps_name, values) = match declared {
        BondedStyle::Single(name) => (name.as_str(), &rest[1..]),
        BondedStyle::Hybrid(subs) => {
            let sub = *rest.get(1).ok_or_else(|| {
                format!(
                    "{}: {category}_coeff under `{category}_style hybrid` names no sub-style",
                    where_()
                )
            })?;
            if subs.is_empty() {
                if lammps_style(reg, category, sub, where_)
                    .is_ok_and(|(spec, _)| ff.get_style(category, &spec.name).is_none())
                {
                    def_bonded_style(ff, reg, category, sub, &[], where_)?;
                }
            } else if !subs.iter().any(|s| s == sub) {
                return Err(format!(
                    "{}: {category}_coeff sub-style `{sub}` is not one of the \
                     `{category}_style hybrid` sub-styles ({})",
                    where_(),
                    subs.join(", ")
                ));
            }
            (sub, &rest[2..])
        }
    };
    let (spec, codec) = lammps_style(reg, category, lammps_name, where_)?;
    let at = |e: String| format!("{}: {e}", where_());
    if let Some(&keyword) = values
        .first()
        .filter(|k| codec.extra_keywords().contains(k))
    {
        let mut params = Params::new();
        return codec
            .parse_extra(spec, keyword, &values[1..], &mut params)
            .map_err(at);
    }
    let params = codec.read(spec, values).map_err(at)?;
    let ends: Vec<&str> = endpoints.iter().map(String::as_str).collect();
    let directive = format!("{category}_style {lammps_name}");
    style_mut(ff, category, &spec.name, &directive, where_)?
        .def_type(&name, &ends, params)
        .map_err(|e| e.to_string())?;
    Ok(())
}

// ── coefficient conversion ──────────────────────────────────────────────────

/// One LAMMPS coefficient line as the molrs params the reader stores,
/// through the codec of the style the LAMMPS style reads as in the
/// process-wide registry.
///
/// `values` are the coefficient tokens **after** the type field(s) — for
/// `bond_coeff c3-c3 228.89 1.5354` that is `["228.89", "1.5354"]`, for
/// `pair_coeff c3 c3 0.1078 3.3977` it is `["0.1078", "3.3977"]`. `style` is
/// the LAMMPS style name (`fourier`, `lj/cut/coul/long`: every `lj/cut…`
/// style carries `lj/cut`'s `epsilon sigma`). `units` is the LAMMPS `units`
/// keyword the numbers are written in (`real`, `metal`, `lj`), which the
/// params are in too: the force-field IR follows the LAMMPS standard, so every
/// value is stored as written — the ½ inside LAMMPS's `K`, degrees for every
/// angle-valued slot — under the name of its slot, the spec's `params` in
/// order for a positional style:
///
/// | category / style        | LAMMPS tokens        | stored params |
/// |-------------------------|----------------------|---------------|
/// | `bond harmonic`         | `K r0`               | `k`, `r0` |
/// | `bond morse`            | `D0 alpha r0`        | `d0`, `alpha`, `r0` |
/// | `bond class2`           | `r0 K2 K3 K4`        | `r0`, `k2`, `k3`, `k4` |
/// | `angle harmonic`        | `K theta0`           | `k`, `theta0` (deg) |
/// | `angle charmm`          | `K theta0 K_ub r_ub` | `k`, `theta0` (deg), `k_ub`, `r_ub` |
/// | `angle class2`          | `theta0 K2 K3 K4`    | `theta0` (deg), `k2`, `k3`, `k4` |
/// | `improper harmonic`     | `K chi0`             | `k`, `chi0` (deg) |
/// | `improper cvff`         | `K d n`              | `k`, `sign = d` (±1), `periodicity = n` |
/// | `dihedral opls`         | `K1 K2 K3 K4`        | `k1..k4` |
/// | `dihedral harmonic`     | `K d n`              | `k`, `sign = d` (±1), `periodicity = n` |
/// | `dihedral fourier`      | `m K1 n1 d1 …`       | molrs `periodic`: `k<i>`, `periodicity<i>`, `phase<i>` (deg) |
/// | `dihedral charmm`       | `K n d w`            | `k`, `periodicity`, `phase` (deg), `w` |
/// | `dihedral multi/harmonic` | `A1 A2 A3 A4 A5`   | `a1..a5` |
/// | `dihedral nharmonic`    | `N A1 … AN`          | `a1..aN` |
/// | `dihedral class2`       | `K1 phi1 K2 phi2 K3 phi3` | `k1`, `phi1` (deg), … |
/// | `pair lj/cut…`, `lj/class2` | `epsilon sigma`  | `epsilon`, `sigma` |
/// | `pair buck`             | `A rho C`            | `a`, `rho`, `c` |
/// | `pair morse`            | `D0 alpha r0`        | `d0`, `alpha`, `r0` |
/// | `pair lj/charmm/coul/charmm` | `epsilon sigma [epsilon14 sigma14]` | `epsilon`, `sigma`, `epsilon14`, `sigma14` (absent → `epsilon`, `sigma`) |
///
/// LAMMPS's single-letter `n` and `d` take molrs's descriptive names because
/// LAMMPS spells two different things `d`: a phase (`charmm`, `fourier`) and a
/// sign (`harmonic`, `cvff`).
///
/// # Errors
///
/// A LAMMPS style no registered style reads, an unknown `units` keyword, a
/// missing coefficient, an extra one, or a non-numeric token.
///
/// ```
/// use molrs::io::lammps::forcefield_reader::lammps_coeff_params;
///
/// let p = lammps_coeff_params("bond", "harmonic", &["450", "0.9572"], "real").unwrap();
/// assert_eq!(p.get("k"), Some(450.0));
/// let p = lammps_coeff_params("angle", "harmonic", &["55", "104.52"], "real").unwrap();
/// assert_eq!(p.get("theta0"), Some(104.52));
/// assert!(lammps_coeff_params("bond", "fene", &["1", "2", "3", "4"], "real").is_err());
/// ```
#[cfg(test)]
pub(crate) fn lammps_coeff_params(
    category: &str,
    style: &str,
    values: &[&str],
    units: &str,
) -> Result<Params, String> {
    parse_lammps_units_style(units)?;
    let style = match style {
        s if category == "pair" && s.starts_with("lj/cut") => "lj/cut",
        s => s,
    };
    let where_ = || format!("{category} {style}");
    crate::ff::style_registry::GlobalRegistry.with(|reg| {
        let (spec, codec) = lammps_style(reg, category, style, &where_)?;
        codec.read(spec, values)
    })
}

// ── helpers ─────────────────────────────────────────────────────────────────

/// Fetch the style created by the matching `*_style` directive; a coeff line
/// before its style is an error, not a silently-dropped parameter.
fn style_mut<'a>(
    ff: &'a mut ForceField,
    category: &str,
    name: &str,
    directive: &str,
    where_: &dyn Fn() -> String,
) -> Result<&'a mut crate::ff::forcefield::Style, String> {
    ff.get_style_mut(category, name)
        .ok_or_else(|| format!("{}: coeff before its `{directive}` declaration", where_()))
}

/// Drop a trailing `#` comment from a LAMMPS line.
fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}

/// Resolve one atom-type token: numeric id via label map, or bare label / `*`.
fn resolve_atom_type(
    raw: &str,
    labels: &LammpsTypeLabelMaps,
    where_: &dyn Fn() -> String,
) -> Result<String, String> {
    if raw == "*" {
        return Ok("*".into());
    }
    if let Ok(id) = raw.parse::<u32>() {
        return Ok(labels
            .atom
            .get(&id)
            .cloned()
            .unwrap_or_else(|| id.to_string()));
    }
    let _ = where_;
    Ok(raw.to_owned())
}

/// A label-only type key: the type name, and the `N` endpoints it means.
///
/// LAMMPS coefficient lines carry a type label and nothing else, so this is
/// the one place molrs infers endpoints from a name
/// ([`TypeName::infer_endpoints`]). The name is the label verbatim; a numeric
/// id is first expanded through `label_map`, or to the synthetic `id-id-…`.
fn label_type<const N: usize>(
    label: Option<&&str>,
    kind: &str,
    where_: impl Fn() -> String,
    label_map: Option<&BTreeMap<u32, String>>,
) -> Result<(String, [String; N]), String> {
    let raw = label.ok_or_else(|| format!("{}: {kind}_coeff missing type label", where_()))?;
    let name = match raw.parse::<u32>() {
        Ok(id) => label_map
            .and_then(|m| m.get(&id).cloned())
            .unwrap_or_else(|| {
                std::iter::repeat_n(id.to_string(), N)
                    .collect::<Vec<_>>()
                    .join("-")
            }),
        Err(_) => (*raw).to_owned(),
    };
    let parts = TypeName::infer_endpoints(&name, N)
        .map_err(|e| format!("{}: {kind} type: {e}", where_()))?;
    let endpoints = std::array::from_fn(|i| parts[i].to_owned());
    Ok((name, endpoints))
}

fn get<'a>(
    rest: &'a [&'a str],
    idx: usize,
    what: &str,
    where_: &dyn Fn() -> String,
) -> Result<&'a str, String> {
    rest.get(idx)
        .copied()
        .ok_or_else(|| format!("{}: missing {what}", where_()))
}

fn parse_f64(raw: &str, what: &str, where_: &dyn Fn() -> String) -> Result<f64, String> {
    raw.parse::<f64>()
        .map_err(|_| format!("{}: {what} is not a number: {raw:?}", where_()))
}

fn parse_triple(
    toks: &[&str],
    start: usize,
    what: &str,
    where_: &dyn Fn() -> String,
) -> Result<[f64; 3], String> {
    Ok([
        parse_f64(get(toks, start, what, where_)?, what, where_)?,
        parse_f64(get(toks, start + 1, what, where_)?, what, where_)?,
        parse_f64(get(toks, start + 2, what, where_)?, what, where_)?,
    ])
}

fn parse_special_bonds(rest: &[&str], where_: &dyn Fn() -> String) -> Result<SpecialBonds, String> {
    let mut toks: Vec<&str> = rest.to_vec();
    while toks.len() >= 2 {
        let kind = toks[toks.len() - 2];
        let val = toks[toks.len() - 1];
        if (kind == "angle" || kind == "dihedral") && (val == "yes" || val == "no") {
            toks.truncate(toks.len() - 2);
            continue;
        }
        break;
    }
    if toks.is_empty() {
        return Err(format!("{}: special_bonds missing weights", where_()));
    }
    match toks[0] {
        // The preset is AMBER's own pair of divisors, spelled as the weights
        // LAMMPS wants: 1/SCNB and 1/SCEE.
        "amber" if toks.len() == 1 => Ok(SpecialBonds {
            lj: [0.0, 0.0, 1.0 / AMBER_SCNB],
            coul: [0.0, 0.0, 1.0 / AMBER_SCEE],
        }),
        "charmm" if toks.len() == 1 => Ok(SpecialBonds {
            lj: [0.0, 0.0, 0.0],
            coul: [0.0, 0.0, 0.0],
        }),
        "dreiding" if toks.len() == 1 => Ok(SpecialBonds {
            lj: [0.0, 0.0, 1.0],
            coul: [0.0, 0.0, 1.0],
        }),
        "fene" if toks.len() == 1 => Ok(SpecialBonds {
            lj: [0.0, 1.0, 1.0],
            coul: [0.0, 1.0, 1.0],
        }),
        "lj" | "coul" => {
            let mut lj = [0.0, 0.0, 0.0];
            let mut coul = [0.0, 0.0, 0.0];
            let mut i = 0;
            while i < toks.len() {
                match toks[i] {
                    "lj" => {
                        lj = parse_triple(&toks, i + 1, "lj weight", where_)?;
                        i += 4;
                    }
                    "coul" => {
                        coul = parse_triple(&toks, i + 1, "coul weight", where_)?;
                        i += 4;
                    }
                    other => {
                        return Err(format!(
                            "{}: unknown special_bonds token `{other}`",
                            where_()
                        ));
                    }
                }
            }
            Ok(SpecialBonds { lj, coul })
        }
        first if first.parse::<f64>().is_ok() => {
            if toks.len() != 3 {
                return Err(format!(
                    "{}: special_bonds bare weights need three numbers",
                    where_()
                ));
            }
            let w = parse_triple(&toks, 0, "weight", where_)?;
            Ok(SpecialBonds { lj: w, coul: w })
        }
        other => Err(format!(
            "{}: unknown special_bonds token `{other}`",
            where_()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    /// A LAMMPS include covering every style, with values copied from a real
    /// GAFF2 PEO `.ff` (figure5).
    const MINI: &str = r#"
# LAMMPS force field generated by molrs
special_bonds amber
pair_style lj/cut/coul/cut 10.0 10.0
pair_coeff c3 c3 0.107800 3.397710
pair_coeff oh oh 0.093000 3.242871
pair_coeff c3 c3 0.107800 3.397710   # duplicate; the last one is used

bond_style harmonic
bond_coeff c3-c3 228.890000 1.535400

angle_style harmonic
angle_coeff c3-c3-oh 76.790000 109.660000

dihedral_style fourier
dihedral_coeff c3-c3-oh-ho 1 0.060000 3 0.000000
"#;

    #[test]
    fn reads_lammps_units() {
        let ff = LammpsForcefieldReader::new().read(MINI.as_bytes()).unwrap();

        // Every coefficient is stored as written: bond K, r0.
        let bond = ff.get_style("bond", "harmonic").unwrap();
        let bt = bond.get_bondtype("c3", "c3").unwrap();
        assert_eq!(
            bt.params.get("k").and_then(|v| v.as_num()),
            Some(228.89),
            "k"
        );
        assert_eq!(
            bt.params.get("r0").and_then(|v| v.as_num()),
            Some(1.5354),
            "r0"
        );

        // angle: K, theta0 in degrees.
        let angle = ff.get_style("angle", "harmonic").unwrap();
        let at = &angle.defs().types()[0];
        assert_eq!(
            at.params.get("k").and_then(|v| v.as_num()),
            Some(76.79),
            "ak"
        );
        assert_eq!(
            at.params.get("theta0").and_then(|v| v.as_num()),
            Some(109.66),
            "theta0"
        );

        // dihedral fourier is molrs's `periodic`: k1/periodicity1/phase1 (deg).
        assert!(ff.get_style("dihedral", "fourier").is_none());
        let dih = ff.get_style("dihedral", "periodic").unwrap();
        let dt = &dih.defs().types()[0];
        assert!(
            (dt.params.get("k1").and_then(|v| v.as_num()).unwrap() - 0.06).abs() < 1e-12,
            "k1"
        );
        assert!(
            (dt.params
                .get("periodicity1")
                .and_then(|v| v.as_num())
                .unwrap()
                - 3.0)
                .abs()
                < 1e-12,
            "periodicity1"
        );
        assert!(
            (dt.params.get("phase1").and_then(|v| v.as_num()).unwrap() - 0.0).abs() < 1e-12,
            "phase1"
        );

        // pair: ε/σ pass through; the duplicate c3 row is ignored.
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let pt = lj.get_pairtype("c3", None).unwrap();
        assert!(
            (pt.params.get("epsilon").and_then(|v| v.as_num()).unwrap() - 0.1078).abs() < 1e-9,
            "eps"
        );
        assert!(
            (pt.params.get("sigma").and_then(|v| v.as_num()).unwrap() - 3.39771).abs() < 1e-9,
            "sig"
        );
        assert!(ff.get_style("pair", "coul/cut").is_some(), "coul style");

        // The cutoffs on the `pair_style` line belong to the force field: without
        // them a written-back include is not a runnable LAMMPS input.
        assert!(
            (lj.params()
                .get("cutoff")
                .and_then(|v| v.as_num())
                .unwrap_or(0.0)
                - 10.0)
                .abs()
                < 1e-12,
            "lj cutoff"
        );
        let coul = ff.get_style("pair", "coul/cut").unwrap();
        assert!(
            (coul
                .params()
                .get("cutoff")
                .and_then(|v| v.as_num())
                .unwrap_or(0.0)
                - 10.0)
                .abs()
                < 1e-12,
            "coulomb cutoff"
        );

        // AMBER/GAFF 1-4 scaling is recorded on the ForceField's special_bonds
        // (1-2/1-3 excluded), the source the pair kernels consume.
        let sb = ff.special_bonds().unwrap();
        assert_eq!(sb.lj, [0.0, 0.0, 0.5]);
        assert!((sb.coul_14() - 1.0 / 1.2).abs() < 1e-12);
        assert_eq!(sb.coul[0], 0.0);
        assert_eq!(sb.coul[1], 0.0);
    }

    /// A LAMMPS `improper_style harmonic` term evaluates at the energy LAMMPS
    /// gives it, `K·(χ − χ₀)²` with `χ₀` in degrees: the reader stores `k = K`,
    /// not the `k = 2K` of the bond/angle form map, which would price every
    /// LAMMPS improper twice too high.
    #[test]
    fn a_lammps_improper_evaluates_at_the_lammps_energy() {
        use crate::ff::compile::PotentialCompiler;
        use molrs::core::Block;
        use molrs::core::Frame;
        use molrs::op::Idx;
        use ndarray::Array1;

        let text = "special_bonds amber\n\
                    improper_style harmonic\n\
                    improper_coeff a-b-c-d 10.0 10.0\n";
        let ff = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();

        let mut impropers = Block::new();
        for (key, atom) in [("atomi", 0), ("atomj", 1), ("atomk", 2), ("atoml", 3)] {
            impropers
                .insert(key, Array1::from_vec(vec![atom as Idx]).into_dyn(), None)
                .unwrap();
        }
        impropers
            .insert(
                "type",
                Array1::from_vec(vec!["a-b-c-d".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("impropers", impropers);
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();

        // i = (0,1,0), j = origin, k = (1,0,0), l = (1, cos χ, sin χ): the
        // I-J-K-L dihedral is χ = 30°.
        let chi = 30.0_f64.to_radians();
        let (sin, cos) = chi.sin_cos();
        let coords = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, cos, sin];
        // LAMMPS: E = K·(χ − χ₀)² = 10 · (20°)² kcal/mol, in radians².
        let lammps = 10.0 * 20.0_f64.to_radians().powi(2);
        let e = pots.energy_at(&coords);
        assert!((e - lammps).abs() < 1e-10, "E = {e}, LAMMPS gives {lammps}");
    }

    #[test]
    fn special_bonds_presets_and_absent_line() {
        let amber = LammpsForcefieldReader::new()
            .read(
                "special_bonds amber\npair_style lj/cut 10.0\npair_coeff c3 c3 0.1 3.4\n"
                    .as_bytes(),
            )
            .unwrap();
        assert!((amber.special_bonds().unwrap().coul_14() - 5.0 / 6.0).abs() < 1e-12);
        assert_eq!(amber.special_bonds().unwrap().lj_14(), 0.5);

        let err = LammpsForcefieldReader::new()
            .read("pair_style lj/cut 10.0\npair_coeff c3 c3 0.1 3.4\n".as_bytes())
            .unwrap_err();
        assert!(err.to_string().contains("special_bonds"), "{err}");
    }

    /// Reduced units stay reduced: an `lj` include declares `lj`, it is not
    /// relabelled as the store's `real`.
    #[test]
    fn units_lj_include_declares_lj_units() {
        let ff = LammpsForcefieldReader::new()
            .read(
                "units lj\nspecial_bonds lj 0.0 0.0 0.0 coul 0.0 0.0 0.0\n\
                 pair_style lj/cut 2.5\npair_coeff A A 1.0 1.0\n"
                    .as_bytes(),
            )
            .unwrap();
        assert_eq!(ff.units(), Some("lj"));
        assert_eq!(ff.units(), Some("lj"));
    }

    #[test]
    fn include_without_a_units_line_reads_as_real() {
        let ff = LammpsForcefieldReader::new()
            .read(
                "special_bonds amber\npair_style lj/cut 10.0\npair_coeff c3 c3 0.1 3.4\n"
                    .as_bytes(),
            )
            .unwrap();
        assert_eq!(ff.units(), Some("real"));
    }

    #[test]
    fn unknown_keyword_errors() {
        let err = LammpsForcefieldReader::new()
            .read("mystery_style foo\n".as_bytes())
            .unwrap_err();
        assert!(
            err.to_string().contains("unknown LAMMPS keyword"),
            "err: {err}"
        );
    }

    #[test]
    fn coeff_before_style_errors() {
        let err = LammpsForcefieldReader::new()
            .read("bond_coeff c3-c3 1.0 1.5\n".as_bytes())
            .unwrap_err();
        assert!(err.to_string().contains("before its"), "err: {err}");
    }

    #[test]
    fn wrong_arity_type_label_errors() {
        let err = LammpsForcefieldReader::new()
            .read("bond_style harmonic\nbond_coeff c3-c3-oh 1.0 1.5\n".as_bytes())
            .unwrap_err();
        assert!(err.to_string().contains("expected 2"), "err: {err}");
    }

    /// A cross `pair_coeff i j` (NBFIX) is a pair type of its own. The reader used
    /// to drop it silently, so the pair was priced by the mixing rule instead.
    #[test]
    fn a_cross_pair_coeff_is_kept_as_a_pair_type() {
        let text = "\
special_bonds amber
pair_style lj/cut 10.0
pair_coeff c3 c3 0.1078 3.39771
pair_coeff oh oh 0.0930 3.24287
pair_coeff c3 oh 0.2500 3.10000
";
        let ff = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let pt = lj.get_pairtype("c3", Some("oh")).expect("cross row c3-oh");
        assert!((pt.params.get("epsilon").and_then(|v| v.as_num()).unwrap() - 0.25).abs() < 1e-12);
        assert!((pt.params.get("sigma").and_then(|v| v.as_num()).unwrap() - 3.1).abs() < 1e-12);
        assert!(lj.get_pairtype("c3", None).is_some());
    }

    /// LAMMPS uses the last `pair_coeff` given for a pair.
    #[test]
    fn a_repeated_pair_coeff_replaces_the_earlier_one() {
        let text = "\
special_bonds amber
pair_style lj/cut 10.0
pair_coeff c3 c3 0.1078 3.39771
pair_coeff c3 c3 0.2000 3.00000
pair_coeff c3 oh 0.2500 3.10000
pair_coeff oh c3 0.3000 3.20000
";
        let ff = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let own = lj.get_pairtype("c3", None).unwrap();
        assert!((own.params.get("epsilon").and_then(|v| v.as_num()).unwrap() - 0.2).abs() < 1e-12);
        let cross = lj.get_pairtype("c3", Some("oh")).expect("cross row");
        assert!(
            (cross
                .params
                .get("epsilon")
                .and_then(|v| v.as_num())
                .unwrap()
                - 0.3)
                .abs()
                < 1e-12
        );
    }

    /// `pair_coeff 1 *` names a cross pair against every type; expanding it is
    /// not implemented, so it is refused rather than dropped.
    #[test]
    fn a_wildcard_cross_pair_coeff_is_an_error() {
        let text = "\
special_bonds amber
pair_style lj/cut 10.0
pair_coeff c3 * 0.1078 3.39771
";
        assert!(LammpsForcefieldReader::new().read(text.as_bytes()).is_err());
    }

    /// The `hybrid/overlay` pair form round-trips: both cutoffs come back and the
    /// wildcard `* * coul/cut` line is skipped rather than mistaken for LJ coefficients.
    #[test]
    fn reads_hybrid_overlay_pair_style() {
        let text = "\
special_bonds amber
pair_style hybrid/overlay lj/cut 10.0 coul/cut 12.0
pair_coeff * * coul/cut
pair_coeff c3 c3 lj/cut 0.1078 3.39771
";
        let ff = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        assert!(
            (lj.params()
                .get("cutoff")
                .and_then(|v| v.as_num())
                .unwrap_or(0.0)
                - 10.0)
                .abs()
                < 1e-12
        );
        let pt = lj.get_pairtype("c3", None).unwrap();
        assert!((pt.params.get("epsilon").and_then(|v| v.as_num()).unwrap() - 0.1078).abs() < 1e-9);
        assert!((pt.params.get("sigma").and_then(|v| v.as_num()).unwrap() - 3.39771).abs() < 1e-9);
        let coul = ff.get_style("pair", "coul/cut").unwrap();
        assert!(
            (coul
                .params()
                .get("cutoff")
                .and_then(|v| v.as_num())
                .unwrap_or(0.0)
                - 12.0)
                .abs()
                < 1e-12
        );
    }

    /// A hybrid line whose sub-styles carry no cutoff (`hybrid lj/cut coul/cut`)
    /// must not read the following sub-style name as a cutoff number — both fall
    /// back with no recorded cutoff.
    /// `lj/charmm/coul/charmm` is molrs's `lj/charmm` + `coul/charmm`: the
    /// switching cutoffs as style params (two numbers: one pair for both),
    /// `pair_coeff` with two or four numbers, mixing LAMMPS's `arithmetic`.
    #[test]
    fn reads_lj_charmm_coul_charmm() {
        let text = "special_bonds charmm
pair_style lj/charmm/coul/charmm 8.0 10.0
pair_coeff CT CT 0.055 3.875 0.01 3.385
pair_coeff OH OH 0.1521 3.1508
";
        let ff = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
        let lj = ff.get_style("pair", "lj/charmm").unwrap();
        assert_eq!(lj.params().get("inner").and_then(|v| v.as_num()), Some(8.0));
        assert_eq!(
            lj.params().get("cutoff").and_then(|v| v.as_num()),
            Some(10.0)
        );
        assert_eq!(
            lj.params().get("mixing").and_then(|v| v.as_text()),
            Some("arithmetic")
        );
        let ct = &lj.get_pairtype("CT", None).unwrap().params;
        assert_eq!(
            (
                ct.get("epsilon14").and_then(|v| v.as_num()),
                ct.get("sigma14").and_then(|v| v.as_num())
            ),
            (Some(0.01), Some(3.385))
        );
        let oh = &lj.get_pairtype("OH", None).unwrap().params;
        assert_eq!(
            (
                oh.get("epsilon14").and_then(|v| v.as_num()),
                oh.get("sigma14").and_then(|v| v.as_num())
            ),
            (Some(0.1521), Some(3.1508))
        );
        let coul = ff.get_style("pair", "coul/charmm").unwrap();
        assert_eq!(
            coul.params().get("inner").and_then(|v| v.as_num()),
            Some(8.0)
        );
        assert_eq!(
            coul.params().get("cutoff").and_then(|v| v.as_num()),
            Some(10.0)
        );

        let four = text.replace("8.0 10.0", "8.0 10.0 9.0 12.0");
        let ff = LammpsForcefieldReader::new().read(four.as_bytes()).unwrap();
        let coul = ff.get_style("pair", "coul/charmm").unwrap();
        assert_eq!(
            coul.params().get("inner").and_then(|v| v.as_num()),
            Some(9.0)
        );
        assert_eq!(
            coul.params().get("cutoff").and_then(|v| v.as_num()),
            Some(12.0)
        );

        for bad in [
            text.replace("8.0 10.0", "10.0"),
            text.replace("0.01 3.385", "0.01"),
            text.replace("lj/charmm/coul/charmm", "lj/charmm/coul/long"),
        ] {
            assert!(
                LammpsForcefieldReader::new().read(bad.as_bytes()).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn reads_hybrid_pair_style_without_cutoffs() {
        let text = "special_bonds amber
pair_style hybrid lj/cut coul/cut
pair_coeff c3 c3 lj/cut 0.1078 3.39771
";
        let ff = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        assert!(
            lj.params().get("cutoff").and_then(|v| v.as_num()).is_none(),
            "no lj cutoff recorded"
        );
        assert!(
            (lj.get_pairtype("c3", None)
                .unwrap()
                .params
                .get("sigma")
                .and_then(|v| v.as_num())
                .unwrap()
                - 3.39771)
                .abs()
                < 1e-9
        );
    }

    /// A `metal` file is read in `metal`: the numbers are not converted, the
    /// force field declares `metal`, and its Coulomb constant is LAMMPS's
    /// `metal` `qqr2e`.
    #[test]
    fn metal_units_are_kept_and_declared() {
        let text = "\
units metal
special_bonds amber
pair_style lj/cut/coul/cut 10.0
pair_coeff c3 c3 1.0 3.4
bond_style harmonic
bond_coeff c3-c3 1.0 1.5
";
        let ff = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
        assert_eq!(ff.units(), Some("metal"));
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(
            lj.get_pairtype("c3", None)
                .unwrap()
                .params
                .get("epsilon")
                .and_then(|v| v.as_num()),
            Some(1.0)
        );
        assert_eq!(
            lj.params().get("cutoff").and_then(|v| v.as_num()),
            Some(10.0)
        );
        let bt = ff
            .get_style("bond", "harmonic")
            .unwrap()
            .get_bondtype("c3", "c3")
            .unwrap();
        assert_eq!(bt.params.get("k").and_then(|v| v.as_num()), Some(1.0));
        assert_eq!(bt.params.get("r0").and_then(|v| v.as_num()), Some(1.5));
        let coul = ff.get_style("pair", "coul/cut").unwrap();
        assert_eq!(
            coul.params().get("coulomb").and_then(|v| v.as_num()),
            Some(COULOMB_METAL)
        );
    }

    #[test]
    fn numeric_bond_id_with_type_labels_map() {
        let mut labels = LammpsTypeLabelMaps::default();
        labels.bond.insert(1, "CT-HC".into());
        labels.atom.insert(1, "CT".into());
        labels.atom.insert(2, "HC".into());
        let text = "\
units real
special_bonds amber
bond_style harmonic
bond_coeff 1 100.0 1.09
pair_style lj/cut 10.0
pair_coeff 1 1 0.1 3.5
";
        let ff = LammpsForcefieldReader::new()
            .read_str_with_labels(text, &labels, None)
            .unwrap();
        let bond = ff.get_style("bond", "harmonic").unwrap();
        assert!(bond.get_bondtype("CT", "HC").is_some());
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        assert!(lj.get_pairtype("CT", None).is_some());
    }

    #[test]
    fn data_coeffs_section_with_labels() {
        let mut labels = LammpsTypeLabelMaps::default();
        labels.bond.insert(1, "OW-HW".into());
        labels.atom.insert(1, "OW".into());
        let coeffs = "\
Bond Coeffs

1 450.0 0.9572

Pair Coeffs

1 0.1521 3.1507
";
        let ff = LammpsForcefieldReader::new()
            .read_data_sections(coeffs, &labels, "real")
            .unwrap();
        let bond = ff.get_style("bond", "harmonic").unwrap();
        let bt = bond.get_bondtype("OW", "HW").unwrap();
        assert_eq!(bt.params.get("k").and_then(|v| v.as_num()), Some(450.0));
        assert!((bt.params.get("r0").and_then(|v| v.as_num()).unwrap() - 0.9572).abs() < 1e-9);
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let pt = lj.get_pairtype("OW", None).unwrap();
        assert!((pt.params.get("epsilon").and_then(|v| v.as_num()).unwrap() - 0.1521).abs() < 1e-9);
    }

    /// A data file whose `* Type Labels` number the types out of sorted order
    /// (`1 hc`, `2 c3`), with its `* Coeffs` rows by id.
    const LABELLED_DATA: &str = "\
LAMMPS data file via write_data, version 4 Jul 2026, timestep = 0, units = real

3 atoms
2 atom types
2 bonds
1 bond types
1 angles
1 angle types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Atom Type Labels

1 hc
2 c3

Bond Type Labels

1 c3-hc

Angle Type Labels

1 hc-c3-hc

Masses

1 1.008
2 12.011

Pair Coeffs # lj/cut/coul/cut

1 0.0157 2.6495
2 0.1094 3.3997

Bond Coeffs # harmonic

1 340.0 1.09

Angle Coeffs # harmonic

1 35.0 109.5

Atoms # full

1 1 2 -0.2 5.0 5.0 5.0
2 1 1 0.1 6.09 5.0 5.0
3 1 1 0.1 4.64 6.03 5.0

Bonds

1 1 1 2
2 1 1 3

Angles

1 1 2 1 3
";

    fn data_frame(text: &str) -> crate::core::Frame {
        crate::io::read_lammps_data(text.as_bytes()).unwrap()
    }

    /// The frame the data reader returned is the whole input: the rows are
    /// named by the file's own label ids (`1` is `hc`, though `c3` sorts
    /// first), in the units its title line states.
    #[test]
    fn data_coeffs_read_from_the_frame_name_rows_by_the_files_labels() {
        let ff = LammpsForcefieldReader::new()
            .read_data_coeffs(&data_frame(LABELLED_DATA), None)
            .unwrap();
        assert_eq!(ff.units(), Some("real"));
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let eps = |t: &str| {
            lj.get_pairtype(t, None)
                .unwrap()
                .params
                .get("epsilon")
                .and_then(|v| v.as_num())
        };
        assert_eq!(eps("hc"), Some(0.0157));
        assert_eq!(eps("c3"), Some(0.1094));
        let bond = ff.get_style("bond", "harmonic").unwrap();
        assert_eq!(
            bond.get_bondtype("c3", "hc")
                .unwrap()
                .params
                .get("k")
                .and_then(|v| v.as_num()),
            Some(340.0)
        );
        assert!(ff.get_style("angle", "harmonic").is_some());
    }

    /// `read_data_coeffs` reads the frame the data door returned, whether
    /// that read a path or bytes: the same force field.
    #[test]
    fn data_coeffs_read_the_frame_of_a_path_or_of_bytes() {
        use crate::io::writer::WrittenText;
        let include = |ff: &ForceField| {
            let frame = data_frame(LABELLED_DATA);
            crate::io::lammps::forcefield_writer::LammpsForcefieldWriter::new(
                &TypeLabels::from_frame(&frame).unwrap(),
            )
            .written(ff)
            .unwrap()
        };
        let reader = LammpsForcefieldReader::new();
        let by_bytes = reader
            .read_data_coeffs(&data_frame(LABELLED_DATA), None)
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labelled.data");
        std::fs::write(&path, LABELLED_DATA).unwrap();
        let frame = crate::io::read_lammps_data(path.as_path()).unwrap();
        let by_path = reader.read_data_coeffs(&frame, None).unwrap();
        assert_eq!(include(&by_path), include(&by_bytes));
    }

    /// LAMMPS data -> force field -> LAMMPS data (the frame plus the force
    /// field's `* Coeffs`) -> force field gives the same force field.
    #[test]
    fn data_coeffs_round_trip_through_a_written_data_file() {
        use crate::io::lammps::forcefield_writer::LammpsForcefieldWriter;
        use crate::io::writer::FrameWriter;
        use crate::io::writer::WrittenText;
        let frame = data_frame(LABELLED_DATA);
        let ff = LammpsForcefieldReader::new()
            .read_data_coeffs(&frame, None)
            .unwrap();
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let mut coeffs = Vec::new();
        LammpsForcefieldWriter::new(&labels)
            .write_data_coeffs(&mut coeffs, &ff)
            .unwrap();
        let coeffs = String::from_utf8(coeffs).unwrap();
        let mut written = Vec::new();
        crate::io::lammps::data::LammpsDataWriter::new(&mut written)
            .write(&frame)
            .unwrap();
        let text = format!("{}\n{coeffs}", String::from_utf8(written).unwrap());
        let again_frame = data_frame(&text);
        let again = LammpsForcefieldReader::new()
            .read_data_coeffs(&again_frame, Some("real"))
            .unwrap();
        let include = |ff: &ForceField, frame: &crate::core::Frame| {
            LammpsForcefieldWriter::new(&TypeLabels::from_frame(frame).unwrap())
                .written(ff)
                .unwrap()
        };
        assert_eq!(include(&again, &again_frame), include(&ff, &frame));
        assert!(include(&ff, &frame).contains("bond_coeff c3-hc 340"));
    }

    /// `units` that disagree with the file's title line, and a frame with no
    /// `* Coeffs`, are refused.
    #[test]
    fn data_coeffs_refuse_other_units_and_a_frame_without_coeffs() {
        let frame = data_frame(LABELLED_DATA);
        let err = LammpsForcefieldReader::new()
            .read_data_coeffs(&frame, Some("metal"))
            .unwrap_err();
        assert!(
            err.to_string().contains("metal") && err.to_string().contains("real"),
            "{err}"
        );
        let err = LammpsForcefieldReader::new()
            .read_data_coeffs(&crate::core::Frame::new(), None)
            .unwrap_err();
        assert!(err.to_string().contains("lammps_coeffs_text"), "{err}");
    }

    /// `PairIJ Coeffs` rows are pair_coeff lines: the self rows are the types'
    /// own, `1 2` an explicit cross pair.
    #[test]
    fn data_coeffs_pairij_rows_keep_their_cross_pairs() {
        let mut labels = LammpsTypeLabelMaps::default();
        labels.atom.insert(1, "c3".into());
        labels.atom.insert(2, "oh".into());
        let coeffs = "\
PairIJ Coeffs # lj/cut

1 1 0.1078 3.39771
1 2 0.2500 3.10000
2 2 0.0930 3.24287
";
        let ff = LammpsForcefieldReader::new()
            .read_data_sections(coeffs, &labels, "real")
            .unwrap();
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let cross = lj.get_pairtype("c3", Some("oh")).expect("cross row c3-oh");
        assert!(
            (cross
                .params
                .get("epsilon")
                .and_then(|v| v.as_num())
                .unwrap()
                - 0.25)
                .abs()
                < 1e-12
        );
        assert!((cross.params.get("sigma").and_then(|v| v.as_num()).unwrap() - 3.1).abs() < 1e-12);
        let own = lj.get_pairtype("oh", None).expect("self row oh");
        assert!(
            (own.params.get("epsilon").and_then(|v| v.as_num()).unwrap() - 0.093).abs() < 1e-12
        );
        assert_eq!(lj.type_rows().len(), 3);
    }

    #[test]
    fn data_coeffs_pairij_row_needs_both_ids() {
        let err = read_data("PairIJ Coeffs\n\n1 0.1 3.0\n").unwrap_err();
        assert!(err.contains("PairIJ Coeffs"), "{err}");
    }

    fn read_data(coeffs: &str) -> Result<ForceField, String> {
        LammpsForcefieldReader::new().read_data_sections(
            coeffs,
            &LammpsTypeLabelMaps::default(),
            "real",
        )
    }

    /// A coefficient row led by a declared type label (LAMMPS ≥ 2022, what
    /// `write_lammps_data` writes beside `* Type Labels`) is that type's row;
    /// an undeclared label is refused by name, never dropped as a section.
    #[test]
    fn data_coeffs_rows_may_lead_with_a_declared_label() {
        let mut labels = LammpsTypeLabelMaps::default();
        labels.atom.insert(1, "CT".into());
        labels.bond.insert(1, "CT-CT".into());
        let ff = LammpsForcefieldReader::new()
            .read_data_sections(
                "Pair Coeffs # lj/cut\n\nCT 0.1 3.5\n\nBond Coeffs # harmonic\n\nCT-CT 300.0 1.5\n",
                &labels,
                "real",
            )
            .unwrap();
        assert!(ff.get_style("pair", "lj/cut").is_some(), "{ff:?}");
        assert!(ff.get_style("bond", "harmonic").is_some(), "{ff:?}");
        let err = LammpsForcefieldReader::new()
            .read_data_sections("Pair Coeffs # lj/cut\n\nXX 0.1 3.5\n", &labels, "real")
            .unwrap_err();
        assert!(err.contains("XX"), "{err}");
    }

    /// `Bond Coeffs # fene/kk` must not be read as harmonic: the reader has no
    /// FENE kernel, so it refuses, naming the section and the style.
    #[test]
    fn data_coeffs_unsupported_bond_hint_is_an_error() {
        let err = read_data("Bond Coeffs # fene/kk\n\n1 30 1.5 1 1\n").unwrap_err();
        assert!(err.contains("Bond Coeffs"), "{err}");
        assert!(
            err.contains("`fene`"),
            "stripped style must be named: {err}"
        );
    }

    #[test]
    fn data_coeffs_unsupported_angle_hint_is_an_error() {
        let err = read_data("Angle Coeffs # cosine/kk\n\n1 2.156\n").unwrap_err();
        assert!(err.contains("cosine"), "{err}");
        assert!(err.contains("Angle Coeffs"), "{err}");
    }

    #[test]
    fn data_coeffs_unsupported_pair_hint_is_an_error() {
        let err = read_data("Pair Coeffs # born\n\n1 1.0 2.0 3.0 4.0 5.0\n").unwrap_err();
        assert!(err.contains("born"), "{err}");
        assert!(err.contains("Pair Coeffs"), "{err}");
    }

    /// A pair style with a positional LAMMPS form reads from its data-file
    /// hint through its codec, its cutoff the data path's 10.0.
    #[test]
    fn data_coeffs_pair_morse_hint_reads_through_its_codec() {
        let ff = read_data("Pair Coeffs # morse\n\n1 1.0 2.0 3.0\n").unwrap();
        let morse = ff.get_style("pair", "morse").unwrap();
        assert_eq!(
            morse.params().get("cutoff").and_then(|v| v.as_num()),
            Some(DATA_PAIR_CUTOFF)
        );
        let p = morse.type_params("1").unwrap();
        assert_eq!(
            (
                p.get("d0").and_then(|v| v.as_num()),
                p.get("alpha").and_then(|v| v.as_num()),
                p.get("r0").and_then(|v| v.as_num())
            ),
            (Some(1.0), Some(2.0), Some(3.0))
        );
    }

    #[test]
    fn data_coeffs_angle_harmonic_hint_reads() {
        let ff = read_data("Angle Coeffs # harmonic\n\n1 50.0 109.5\n").unwrap();
        let a = ff.get_style("angle", "harmonic").unwrap();
        let at = &a.defs().types()[0];
        assert_eq!(at.params.get("k").and_then(|v| v.as_num()), Some(50.0));
        assert_eq!(
            at.params.get("theta0").and_then(|v| v.as_num()),
            Some(109.5)
        );
    }

    #[test]
    fn data_coeffs_accelerator_suffix_is_stripped() {
        let ff = read_data("Bond Coeffs # harmonic/kk\n\n1 450.0 0.9572\n").unwrap();
        let bt = ff
            .get_style("bond", "harmonic")
            .unwrap()
            .get_bondtype("1", "1")
            .unwrap();
        assert_eq!(bt.params.get("k").and_then(|v| v.as_num()), Some(450.0));
        assert!((bt.params.get("r0").and_then(|v| v.as_num()).unwrap() - 0.9572).abs() < 1e-12);
    }

    #[test]
    fn data_coeffs_pair_lj_cut_coul_long_hint_reads() {
        let ff = read_data("Pair Coeffs # lj/cut/coul/long/kk\n\n1 0.1521 3.1507\n").unwrap();
        let pt = ff
            .get_style("pair", "lj/cut")
            .unwrap()
            .get_pairtype("1", None)
            .unwrap();
        assert!(
            (pt.params.get("epsilon").and_then(|v| v.as_num()).unwrap() - 0.1521).abs() < 1e-12
        );
        assert!((pt.params.get("sigma").and_then(|v| v.as_num()).unwrap() - 3.1507).abs() < 1e-12);
    }

    /// The hint selects the dihedral kernel and its coefficient layout: four
    /// OPLS coefficients, not the default harmonic `K d n`.
    #[test]
    fn data_coeffs_dihedral_hint_selects_kernel() {
        let ff = read_data("Dihedral Coeffs # opls/omp\n\n1 1.0 2.0 3.0 4.0\n").unwrap();
        let d = ff.get_style("dihedral", "opls").unwrap();
        let dt = &d.defs().types()[0];
        assert!((dt.params.get("k4").and_then(|v| v.as_num()).unwrap() - 4.0).abs() < 1e-12);
    }

    /// A section without a `# style` hint keeps the documented default.
    #[test]
    fn data_coeffs_without_hint_defaults_to_harmonic() {
        let ff = read_data("Bond Coeffs\n\n1 450.0 0.9572\n").unwrap();
        let bt = ff
            .get_style("bond", "harmonic")
            .unwrap()
            .get_bondtype("1", "1")
            .unwrap();
        assert_eq!(bt.params.get("k").and_then(|v| v.as_num()), Some(450.0));
    }

    #[test]
    fn dihedral_opls_four_coeffs() {
        let text = "\
special_bonds amber
dihedral_style opls
dihedral_coeff CT-CT-CT-CT 1.0 2.0 3.0 4.0
";
        let ff = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
        let d = ff.get_style("dihedral", "opls").unwrap();
        let dt = &d.defs().types()[0];
        assert!((dt.params.get("k1").and_then(|v| v.as_num()).unwrap() - 1.0).abs() < 1e-12);
        assert!((dt.params.get("k4").and_then(|v| v.as_num()).unwrap() - 4.0).abs() < 1e-12);
    }

    /// `lammps_coeff_params` returns the params the reader stores, for each
    /// kernel: every value as written, under its molrs slot name.
    #[test]
    fn lammps_coeff_params_is_the_identity_on_values() {
        let close = |p: &Params, key: &str, want: f64| {
            let got = p
                .get(key)
                .and_then(|v| v.as_num())
                .unwrap_or_else(|| panic!("missing `{key}`"));
            assert!((got - want).abs() < 1e-12, "{key}: {got} != {want}");
        };
        let keys = |p: &Params| {
            let mut k: Vec<String> = p
                .iter()
                .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
                .map(|(k, _)| k.to_owned())
                .collect();
            k.sort();
            k
        };

        let p = lammps_coeff_params("bond", "harmonic", &["450", "0.9572"], "real").unwrap();
        close(&p, "k", 450.0);
        close(&p, "r0", 0.9572);
        assert_eq!(keys(&p), ["k", "r0"]);

        let p = lammps_coeff_params("bond", "morse", &["95.6", "2.0", "1.53"], "metal").unwrap();
        close(&p, "d0", 95.6);
        close(&p, "alpha", 2.0);
        close(&p, "r0", 1.53);
        assert_eq!(keys(&p), ["alpha", "d0", "r0"]);

        let p = lammps_coeff_params("angle", "harmonic", &["55", "104.52"], "real").unwrap();
        close(&p, "k", 55.0);
        close(&p, "theta0", 104.52);
        assert_eq!(keys(&p), ["k", "theta0"]);

        let p = lammps_coeff_params("improper", "harmonic", &["10", "180"], "real").unwrap();
        close(&p, "k", 10.0);
        close(&p, "chi0", 180.0);
        assert_eq!(keys(&p), ["chi0", "k"]);

        let p = lammps_coeff_params("improper", "cvff", &["1.1", "-1", "2"], "real").unwrap();
        close(&p, "k", 1.1);
        close(&p, "sign", -1.0);
        close(&p, "periodicity", 2.0);
        assert_eq!(keys(&p), ["k", "periodicity", "sign"]);

        let p = lammps_coeff_params("dihedral", "opls", &["1", "2", "3", "4"], "real").unwrap();
        for (key, want) in [("k1", 1.0), ("k2", 2.0), ("k3", 3.0), ("k4", 4.0)] {
            close(&p, key, want);
        }
        assert_eq!(keys(&p), ["k1", "k2", "k3", "k4"]);

        let p = lammps_coeff_params("dihedral", "harmonic", &["2", "-1", "3"], "real").unwrap();
        close(&p, "k", 2.0);
        close(&p, "sign", -1.0);
        close(&p, "periodicity", 3.0);
        assert_eq!(keys(&p), ["k", "periodicity", "sign"]);

        // fourier: m, then (K n d) per term; d stays degrees.
        let p = lammps_coeff_params(
            "dihedral",
            "fourier",
            &["2", "0.5", "1", "180", "0.25", "3", "0"],
            "real",
        )
        .unwrap();
        close(&p, "k1", 0.5);
        close(&p, "periodicity1", 1.0);
        close(&p, "phase1", 180.0);
        close(&p, "k2", 0.25);
        close(&p, "periodicity2", 3.0);
        close(&p, "phase2", 0.0);

        // charmm: K n d w — its own layout, not fourier's `m K n d`.
        let p =
            lammps_coeff_params("dihedral", "charmm", &["0.2", "3", "180", "0.5"], "real").unwrap();
        close(&p, "k", 0.2);
        close(&p, "periodicity", 3.0);
        close(&p, "phase", 180.0);
        close(&p, "w", 0.5);
        assert_eq!(keys(&p), ["k", "periodicity", "phase", "w"]);

        // multi/harmonic: A1..A5, energies.
        let p = lammps_coeff_params(
            "dihedral",
            "multi/harmonic",
            &["1", "2", "3", "4", "5"],
            "real",
        )
        .unwrap();
        for (key, want) in [
            ("a1", 1.0),
            ("a2", 2.0),
            ("a3", 3.0),
            ("a4", 4.0),
            ("a5", 5.0),
        ] {
            close(&p, key, want);
        }

        let p = lammps_coeff_params("pair", "lj/cut", &["0.066", "3.5"], "real").unwrap();
        close(&p, "epsilon", 0.066);
        close(&p, "sigma", 3.5);
        assert_eq!(keys(&p), ["epsilon", "sigma"]);
    }

    #[test]
    fn lammps_coeff_params_rejects_unsupported_kernel_and_bad_tokens() {
        let err = lammps_coeff_params("bond", "fene", &["1", "2", "3", "4"], "real").unwrap_err();
        assert!(err.contains("bond") && err.contains("fene"), "{err}");

        let err = lammps_coeff_params("bond", "harmonic", &["450"], "real").unwrap_err();
        assert!(err.contains("r0"), "{err}");

        let err = lammps_coeff_params("bond", "harmonic", &["x", "1"], "real").unwrap_err();
        assert!(err.contains("not a number"), "{err}");

        let err = lammps_coeff_params("bond", "harmonic", &["1", "1"], "si").unwrap_err();
        assert!(err.contains("si"), "{err}");
    }

    /// The reader stores `dihedral_style charmm` under its own kernel with the
    /// `K n d w` layout, not as a fourier term.
    #[test]
    fn dihedral_charmm_reads_its_own_layout() {
        let text = "\
special_bonds charmm
dihedral_style charmm
dihedral_coeff A-B-C-D 0.2 3 180 1.0
";
        let ff = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
        let d = ff.get_style("dihedral", "charmm").unwrap();
        let dt = &d.defs().types()[0];
        assert!((dt.params.get("k").and_then(|v| v.as_num()).unwrap() - 0.2).abs() < 1e-12);
        assert!(
            (dt.params
                .get("periodicity")
                .and_then(|v| v.as_num())
                .unwrap()
                - 3.0)
                .abs()
                < 1e-12
        );
        assert_eq!(dt.params.get("phase").and_then(|v| v.as_num()), Some(180.0));
        assert!(ff.get_style("dihedral", "periodic").is_none());
    }

    #[test]
    fn numeric_pair_types_stable_with_bonded_coeffs() {
        // Regression: bonded types whose endpoints look like "10" must not
        // scramble pair self-types named "10".
        let text = "\
units real
special_bonds amber
pair_style lj/cut 10.0
pair_coeff 1 1 0.11 3.5
pair_coeff 2 2 0.08 3.6
pair_coeff 10 10 0.046 0.4
bond_style harmonic
bond_coeff 1-1 100.0 1.5
bond_coeff 10-10 200.0 1.2
dihedral_style harmonic
dihedral_coeff 10-10-10-10 0.2 1.0 180.0
";
        let ff = LammpsForcefieldReader::new().read(text.as_bytes()).unwrap();
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        let p1 = lj.get_pairtype("1", None).unwrap();
        let p2 = lj.get_pairtype("2", None).unwrap();
        let p10 = lj.get_pairtype("10", None).unwrap();
        assert!((p1.params.get("epsilon").and_then(|v| v.as_num()).unwrap() - 0.11).abs() < 1e-12);
        assert!((p2.params.get("epsilon").and_then(|v| v.as_num()).unwrap() - 0.08).abs() < 1e-12);
        assert!(
            (p10.params.get("epsilon").and_then(|v| v.as_num()).unwrap() - 0.046).abs() < 1e-12
        );
        assert!((p10.params.get("sigma").and_then(|v| v.as_num()).unwrap() - 0.4).abs() < 1e-12);
    }

    // ── fix cmap ────────────────────────────────────────────────────────────

    const ALANINE: &str = include_str!("../../ff/potential/cmap/testdata/charmm36_alanine.cmap");

    /// CHARMM's own file: the `# <φ>` comments are skipped, the 576 numbers
    /// fill the map φ-major, and `UNITS:real` (one word) is no tag, as for
    /// LAMMPS.
    #[test]
    fn a_charmm_cmap_file_reads_phi_major() {
        let file = parse_cmap_file(ALANINE).unwrap();
        assert_eq!(file.units, None);
        assert_eq!(file.maps.len(), 1);
        let map = &file.maps[0];
        assert_eq!(map.shape(), &[24, 24]);
        // `# -180.0` row: first and last value; `# -165.0` row: first value.
        assert_eq!(map[[0, 0]], 0.126790);
        assert_eq!(map[[0, 23]], -0.036650);
        assert_eq!(map[[1, 0]], -0.127133);

        let ff = LammpsForcefieldReader::new()
            .read_cmap(ALANINE.as_bytes())
            .unwrap();
        assert_eq!(ff.units(), Some("real"));
        let rows = ff.get_types("cmap");
        assert_eq!(
            (rows[0].name.as_str(), rows[0].endpoints[4].as_str()),
            ("1", "1")
        );
        assert_eq!(
            rows[0].params.get("grid").and_then(|v| v.as_array()),
            Some(map)
        );
    }

    #[test]
    fn a_cmap_file_fix_cmap_would_misread_is_refused() {
        let map = "0.5\n".repeat(576);
        let tagged = format!("# UNITS: metal\n{map}{map}");
        let file = parse_cmap_file(&tagged).unwrap();
        assert_eq!((file.units.as_deref(), file.maps.len()), (Some("metal"), 2));

        for (text, why) in [
            (format!("{map}0.5 0.5\n"), "incomplete"),
            (map.repeat(7), "past the 6"),
            (
                format!("{}0.5 0.5\n{}", "0.5\n".repeat(575), map),
                "past the end of map 1",
            ),
            ("# nothing\n".to_owned(), "no CMAP map"),
            (format!("{}x\n", "0.5\n".repeat(575)), "not a number"),
        ] {
            let err = parse_cmap_file(&text).unwrap_err();
            assert!(err.contains(why), "{why}: {err}");
        }
    }

    /// An include's `fix cmap` line reads its file relative to the include,
    /// `fix_modify` of it is accepted, and a file in other units is refused.
    #[test]
    fn an_include_reads_its_fix_cmap_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("c.cmap"), ALANINE).unwrap();
        let include = "units real\nfix cm all cmap c.cmap\nfix_modify cm energy yes\n\
                       special_bonds charmm\n";
        let path = dir.path().join("sys.ff");
        std::fs::write(&path, include).unwrap();
        let ff = LammpsForcefieldReader::new().read(path.as_path()).unwrap();
        assert_eq!(ff.get_types("cmap").len(), 1);

        for (text, why) in [
            (
                "fix cm all nve\nspecial_bonds charmm\n",
                "unsupported fix style",
            ),
            (
                "fix_modify cm energy yes\nspecial_bonds charmm\n",
                "no fix cmap",
            ),
        ] {
            std::fs::write(&path, text).unwrap();
            let err = LammpsForcefieldReader::new()
                .read(path.as_path())
                .unwrap_err();
            assert!(err.to_string().contains(why), "{err}");
        }
        std::fs::write(
            dir.path().join("c.cmap"),
            format!("# UNITS: metal\n{}", "0.5\n".repeat(576)),
        )
        .unwrap();
        std::fs::write(&path, include).unwrap();
        let err = LammpsForcefieldReader::new()
            .read(path.as_path())
            .unwrap_err();
        assert!(err.to_string().contains("UNITS: metal"), "{err}");
    }

    /// `pair_style lj/cut` prices no charge in LAMMPS: no Coulomb style.
    #[test]
    fn a_bare_lj_cut_has_no_coulomb_style() {
        let ff = LammpsForcefieldReader::new()
            .read(
                "special_bonds amber\npair_style lj/cut 10.0\npair_coeff c3 c3 0.1 3.4\n"
                    .as_bytes(),
            )
            .unwrap();
        assert!(ff.get_style("pair", "lj/cut").is_some());
        assert!(ff.get_style("pair", "coul/cut").is_none());
        assert_eq!(ff.get_styles("pair").len(), 1);
    }

    /// `lj/cut/coul/long` is `coul/long/pme` with its cutoff and constant; its
    /// Ewald parameters are the script's `kspace_style`, not read, so pricing
    /// refuses until they are stated.
    #[test]
    fn lj_cut_coul_long_is_coul_long_pme_without_its_ewald_parameters() {
        for line in [
            "pair_style lj/cut/coul/long 10.0 12.0",
            "pair_style lj/cut/coul/long/omp 10.0 12.0",
            "pair_style hybrid/overlay lj/cut 10.0 coul/long 12.0",
        ] {
            let ff = LammpsForcefieldReader::new()
                .read(format!("special_bonds amber\n{line}\npair_coeff c3 c3 0.1 3.4\n").as_bytes())
                .unwrap();
            assert!(ff.get_style("pair", "coul/cut").is_none(), "{line}");
            let coul = ff.get_style("pair", "coul/long/pme").expect(line);
            assert_eq!(
                coul.params().get("cutoff").and_then(|v| v.as_num()),
                Some(12.0),
                "{line}"
            );
            assert_eq!(
                coul.params().get("coulomb").and_then(|v| v.as_num()),
                Some(COULOMB_REAL),
                "{line}"
            );
            assert_eq!(
                coul.params().get("alpha").and_then(|v| v.as_num()),
                None,
                "{line}"
            );
        }
    }

    /// Another LAMMPS Coulomb is refused by name, not read as plain.
    #[test]
    fn another_lj_cut_coulomb_is_refused() {
        for style in ["lj/cut/coul/debye 10.0", "lj/cut/coul/dsf 0.2 10.0"] {
            let err = LammpsForcefieldReader::new()
                .read(
                    format!("special_bonds amber\npair_style {style}\npair_coeff c3 c3 0.1 3.4\n")
                        .as_bytes(),
                )
                .unwrap_err();
            assert!(
                err.to_string().contains(style.split(' ').next().unwrap()),
                "{err}"
            );
        }
    }

    /// `pair_modify shift yes` is `lj/cut`'s `shift`; the switched CHARMM
    /// style refuses it.
    #[test]
    fn pair_modify_shift_is_the_lj_cut_shift() {
        let ff = LammpsForcefieldReader::new()
            .read("special_bonds amber\npair_style lj/cut 10.0\npair_modify mix arithmetic shift yes\n\
                 pair_coeff c3 c3 0.1 3.4\n".as_bytes())
            .unwrap();
        let lj = ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(lj.params().get("shift").and_then(|v| v.as_num()), Some(1.0));
        assert_eq!(
            lj.params().get("mixing").and_then(|v| v.as_text()),
            Some("arithmetic")
        );
        let err = LammpsForcefieldReader::new()
            .read(
                "special_bonds charmm\npair_style lj/charmm/coul/charmm 8.0 10.0\n\
                 pair_modify shift yes\n\
                 pair_coeff c3 c3 0.1 3.4\n"
                    .as_bytes(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("shift"), "{err}");
    }
}
