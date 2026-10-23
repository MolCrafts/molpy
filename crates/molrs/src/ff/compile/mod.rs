//! Compiling a force field: a [`ForceField`]'s declarations bound, through
//! the style registry ([`crate::ff::style_registry`]), to the kernels of
//! [`crate::ff::potential`] on a typed [`Frame`].
//!
//! * [`PotentialCompiler`] compiles a force field against a frame;
//! * [`ExplicitTerms`] compiles one style over explicit instances;
//! * the 1-4 exceptions (`exceptions`, `one_four`) decide which close pairs
//!   the [`PairExceptions`](crate::ff::potential::pair::PairExceptions)
//!   kernel prices, and at what parameters.

pub(crate) mod exceptions;
mod explicit_terms;
pub(crate) mod one_four;

pub use explicit_terms::ExplicitTerms;

use crate::ff::style_registry;
use std::borrow::Cow;

use crate::ff::forcefield::{ForceField, Style};
use crate::ff::ir::{self, CategorySpec, Coordinate, EndpointOrder};
use crate::ff::ir::{ParamSource, SpecialClass};
use crate::ff::ir::{Params, SpecialBonds};
use crate::ff::potential::{
    CompileError, ForceTerm, PairWeights, Potentials, ScaledTerm, WeightedTerm,
};
use crate::ff::style_registry::RegisteredStyle;
use crate::ff::style_registry::RowSource;
use crate::ff::style_registry::{Registry, RegistryRef};
use molrs::core::Frame;
use molrs::core::schema::PAIR_OVERRIDE_COLUMNS;
use molrs::core::schema::block_names::{ATOMS, PAIRS};

/// Compiles one [`ForceField`] against typed [`Frame`]s.
///
/// The force field declares; the compiler reads those declarations and builds
/// kernels. It borrows the force field, so it is cheap to make where the frame
/// is at hand:
///
/// ```
/// use molrs::ff::forcefield::ForceField;
/// use molrs::ff::ir::Params;
/// use molrs::ff::compile::PotentialCompiler;
/// use molrs::core::Frame;
///
/// let mut ff = ForceField::new("example");
/// ff.def_style("bond", "harmonic", Params::new())
///     .unwrap()
///     .def_type("A-B", &["A", "B"], Params::from_iter([("k", 300.0), ("r0", 1.5)]))
///     .unwrap();
///
/// // A frame without a `bonds` block has nothing for the bond style to read.
/// let pots = PotentialCompiler::new(&ff).compile(&Frame::new()).unwrap();
/// assert!(pots.members().is_empty());
/// ```
///
/// Each style's category and kernel come from the force-field IR registry
/// ([`crate::ff::style_registry`]): the force field's own
/// ([`ForceField::with_registry`]), else the process-wide one.
///
/// [`PotentialCompiler`] reads one force field and has two doors; the
/// difference between them is what carries the non-bonded pair list:
///
/// * [`PotentialCompiler::compile`] resolves every pair style against the
///   frame's `pairs` block — a fixed list, finite by construction. Right for
///   a molecule in free space, and what the geometry optimizer and the
///   conformer pipeline use.
/// * [`PotentialCompiler::compile_typed`] resolves them against the **atoms**,
///   so the kernels answer for whatever pairs a neighbour search turns up.
///   Right for a periodic box, and what MD uses.
///
/// Both doors price a pair exactly as LAMMPS does: only at `r < cutoff`, the
/// style's own `cutoff` (and between `inner` and `cutoff` through the
/// switch of a style that defines one, `lj/charmm`, `coul/charmm`) — the
/// pairs a list holds beyond it price nothing, 1-4 pairs scaled by
/// `special_bonds` included. A style that states no cutoff has cutoff ∞
/// (the spec's default of `lj/cut`, `coul/cut`, …), so a field read from an
/// engine priced untruncated (OpenMM `NoCutoff`, a prmtop) prices every
/// listed pair. The 1-4 exceptions kernel (a `dihedral charmm` `w`, per-pair
/// overrides) is no pair style and is never truncated, as LAMMPS's
/// `dihedral_style charmm` prices its 1-4 pair at any distance.
///
/// The direction is one-way: a force field **declares** styles, types and
/// constants, and the compiler reads them to build kernels. Nothing here is
/// imported back by [`crate::ff::forcefield`].
#[derive(Debug, Clone, Copy)]
pub struct PotentialCompiler<'a> {
    ff: &'a ForceField,
}

impl<'a> PotentialCompiler<'a> {
    /// A compiler that reads `ff`, against `ff`'s registry
    /// ([`ForceField::registry`]).
    pub fn new(ff: &'a ForceField) -> Self {
        Self { ff }
    }

    /// The registry this compile reads.
    ///
    /// The process-wide one is copied out rather than read under its lock:
    /// a kernel constructor is arbitrary code (a third party's, a Python
    /// callable's), and one that registered a style while the compile held
    /// the lock would deadlock.
    fn registry(&self) -> Cow<'a, Registry> {
        match self.ff.registry() {
            RegistryRef::Own(r) => Cow::Borrowed(r.as_ref()),
            RegistryRef::Global => {
                Cow::Owned(style_registry::with_global_registry(Registry::clone))
            }
        }
    }

    /// Build evaluable [`Potentials`] by expanding every style against a
    /// typed [`Frame`].
    ///
    /// Each style resolves its string type labels to per-element parameter
    /// arrays, so the resulting potentials are **molecule-bound**: they retain
    /// no Frame and evaluate from coordinates alone. Styles with no kernel
    /// (atom styles) are skipped.
    ///
    /// Every pair style prices a `pairs` row only inside its `cutoff`
    /// (`r < cutoff`, with its switch where it defines one), as
    /// [`compile_typed`](Self::compile_typed) and LAMMPS do; ∞ when the style
    /// states none.
    ///
    /// The `pairs` block carries the 1-2 / 1-3 weights by whether a row is
    /// there, so a force field that *scales* those classes rather than
    /// excluding them is an [`Err`] here — see
    /// [`SpecialBonds::compiled_inclusion`](crate::ff::ir::SpecialBonds::compiled_inclusion).
    /// [`compile_typed`](Self::compile_typed) carries a per-pair weight and
    /// takes every force field.
    ///
    /// # 1-4 exceptions
    ///
    /// The pairs a `dihedral charmm` `w` prices and the `pairs` rows carrying
    /// per-pair override columns are one more member, the exceptions kernel
    /// ([`PairExceptions`](crate::ff::potential::pair::PairExceptions)); the
    /// pair styles see the `pairs` list without the override rows. Both doors
    /// build it the same way. Precedence per pair: an override cell, else the
    /// dihedral's `w`, else `special_bonds`. The exceptions kernel is not
    /// truncated: LAMMPS's `dihedral_style charmm` prices its 1-4 pair at any
    /// distance, and an OpenMM exception is outside the cutoff method too.
    pub fn compile(&self, frame: &Frame) -> Result<Potentials, CompileError> {
        // The `pairs` block need not have come from `intramolecular_pairs` — a
        // GROMACS `[ pairs ]` section is read straight off a file — so the
        // weights are checked here too, at the door that decides the physics,
        // and not only where the list happens to be built.
        let reg = self.registry();
        let special = self.pair_weights(&reg)?;
        if let Some(sb) = special {
            sb.compiled_inclusion()?;
        }
        // The 1-4 exceptions (dihedral charmm `w`, per-pair overrides) are one
        // kernel; the regular pair kernels see the `pairs` list without the
        // override rows, which is their weight 0.
        let exceptions = exceptions::plan(self.ff, frame)?;
        let regular = regular_pairs(frame, &exceptions.override_rows)?;
        let mut pots = Potentials::new();
        for style in self.ff.styles() {
            let frame = if category_of(&reg, style)?.is_pair_driven() {
                &*regular
            } else {
                frame
            };
            // `member` skips a style whose topology block is absent or empty; a
            // *present* block with an unknown type label is a real error and
            // propagates from the kernel constructor.
            if let Some(pot) = self.member(&reg, style, frame, special)? {
                pots.push(pot);
            }
        }
        if let Some(kernel) = exceptions.kernel {
            pots.push(ForceTerm::indexed(kernel));
        }
        // Record the atom count so callers (e.g. the geometry optimizer's batch
        // path) can validate coordinate shapes against this topology.
        pots.set_n_atoms(frame.get(ATOMS).and_then(|b| b.n_rows()).unwrap_or(0));
        Ok(pots)
    }

    /// Build the members of a **neighbour-driven** force evaluation, each with
    /// the bond-distance weights its non-bonded term takes.
    ///
    /// The counterpart of [`compile`](Self::compile), and what periodic MD
    /// needs. That one resolves every pair style against the frame's `pairs`
    /// block — a fixed list, finite by construction, which is right for a
    /// free-boundary molecule and wrong for a periodic system. This one
    /// resolves them against the **atoms**, so the kernels can answer for
    /// whatever pairs a neighbour search turns up; a pair style needs a
    /// finite `cutoff` here. Over the same pairs, the two doors price the
    /// same energy: both truncate at the style's `cutoff`. Of
    /// the `pairs` block it reads only the rows carrying per-pair overrides:
    /// those pairs go to the exceptions kernel (an indexed member, weight
    /// `None`), and every pair member's [`PairWeights`] weights them 0.
    ///
    /// The weights come back per member rather than once, because a force field
    /// may scale close van-der-Waals and electrostatic neighbours differently —
    /// Amber uses `1/2` and `1/1.2` — and in molrs those are separate kernels.
    /// A bonded member takes `None`: it *is* the bonded interaction, not a
    /// scaled copy of it.
    ///
    /// # Why the weights matter here and not there
    ///
    /// A compiled list carries the exclusions by leaving the excluded rows out
    /// and baking the 1-4 factor into the parameters. A neighbour table has no
    /// such memory — it finds every pair inside the cutoff, bonded or not — so
    /// without the weights a bonded pair is counted twice: once by the bond
    /// term and once at full non-bonded strength, at bond length.
    pub fn compile_typed(&self, frame: &Frame) -> Result<Vec<WeightedTerm>, CompileError> {
        let reg = self.registry();
        let special = self.pair_weights(&reg)?;
        let exceptions = exceptions::plan(self.ff, frame)?;
        let mut out = Vec::new();
        for style in self.ff.styles() {
            // A bonded style contributes nothing when the molecule carries no
            // topology of its kind (`member` skips it). A pair style is never
            // skipped: which pairs exist is the neighbour search's answer, not
            // the frame's.
            if let Some((pot, class)) = self.typed_member(&reg, style, frame)? {
                let weights = match (class, special) {
                    (None, _) => None,
                    (Some(c), Some(sb)) => {
                        let by_distance = match c {
                            SpecialClass::Vdw => sb.lj_weights(),
                            SpecialClass::Coulomb => sb.coul_weights(),
                        };
                        Some(PairWeights::new(by_distance, exceptions.replaced.clone()))
                    }
                    (Some(_), None) => {
                        return Err(self.ff.required_special_bonds().unwrap_err().into());
                    }
                };
                out.push((pot, weights));
            }
        }
        if let Some(kernel) = exceptions.kernel {
            out.push((ForceTerm::indexed(kernel), None));
        }
        Ok(out)
    }

    /// The force field's declared pair weights — required, and an error when
    /// absent, as soon as one of its styles prices pairs; a field of bonded
    /// styles alone needs none.
    fn pair_weights(&self, reg: &Registry) -> Result<Option<&'a SpecialBonds>, CompileError> {
        for style in self.ff.styles() {
            if category_of(reg, style)?.is_pair_driven() {
                return Ok(Some(self.ff.required_special_bonds()?));
            }
        }
        Ok(self.ff.special_bonds())
    }

    /// Build `style`'s kernel for a **neighbour-driven** evaluation, and say
    /// which special-bonds weights scale it.
    ///
    /// The counterpart of [`member`](Self::member). A bonded style is built
    /// identically — it reads indices, and a neighbour table does not concern
    /// it. A pair style is built in its typed form, which reads no `pairs`
    /// block: there is none to read when the list is rebuilt every few steps,
    /// and a kernel whose parameters were resolved against an older one would
    /// be naming different atoms.
    ///
    /// A pair style with no typed form is an [`Err`], not a fallback to the
    /// compiled one. Falling back would hand back a kernel whose parameters
    /// belong to a pair list nobody is evaluating, and it would answer.
    fn typed_member(
        &self,
        reg: &Registry,
        style: &Style,
        frame: &Frame,
    ) -> Result<Option<ScaledTerm>, CompileError> {
        let category = category_of(reg, style)?;
        let category = &*category;
        if !category.prices_energy() {
            return Ok(None);
        }
        if !category.is_pair_driven() {
            // Bonded styles are unchanged, and take no special-bonds weight:
            // the term *is* the bonded interaction, not a scaled copy of it.
            return Ok(self.member(reg, style, frame, None)?.map(|p| (p, None)));
        }
        // No `pairs` gate: a typed pair kernel is built from the atoms, and a
        // frame with atoms always has those.
        let spec = category;
        let category = style.category();
        let entry = reg.registered_style(category, style.name());
        let type_params = style.defs().kernel_type_params()?;
        param_source_of(entry, style, &type_params)?;
        let type_refs: Vec<(&str, &Params)> = type_params
            .iter()
            .map(|(name, params)| (name.as_str(), params))
            .collect();
        let entry = with_fallback(reg, spec, style, entry, &type_refs)?;
        let (pot, special) = entry
            .typed(
                spec,
                style.params(),
                &type_refs,
                frame,
                reg.expression_compiler(),
            )?
            .ok_or_else(|| {
                format!(
                    "pair style '{}' has no neighbour-driven form, so it cannot be \
                     evaluated over a neighbour list; its compiled form answers only \
                     for the pair list it was built from",
                    style.name()
                )
            })?;
        Ok(Some((pot, Some(special))))
    }

    /// Build `style`'s molecule-bound [`ForceTerm`] by **expanding** its type
    /// parameters against `frame`'s topology — each bond/angle/… row's string
    /// type label is resolved to its parameters and stored as per-element
    /// arrays, so the resulting potential evaluates from coordinates alone.
    ///
    /// Returns `Ok(None)` for a style that carries no pairwise kernel (an atom
    /// style — types/charges only), `Err` for an unknown `(category, name)`.
    ///
    /// The `(category, name)` → kernel mapping, and the category's block and
    /// gating, live in the force-field IR [`Registry`]; a new potential or a
    /// new category is added by registering it, not by editing this dispatch.
    fn member(
        &self,
        reg: &Registry,
        style: &Style,
        frame: &Frame,
        special_bonds: Option<&SpecialBonds>,
    ) -> Result<Option<ForceTerm>, CompileError> {
        let spec = category_of(reg, style)?;
        let spec = &*spec;
        if !spec.prices_energy() {
            return Ok(None);
        }
        let category = style.category();
        let entry = reg.registered_style(category, style.name());
        // A style contributes nothing when the molecule carries no topology of its
        // kind: a bonded style with no bonds/angles/dihedrals/impropers, or a pair
        // style when the neighbour list is empty (e.g. methane, whose every atom
        // pair is 1-2 or 1-3 excluded). Skip it rather than letting the kernel ctor
        // fault on the absent/empty block.
        // Which block gates a style is the *kernel's* property, not the
        // category's: PME is a `pair` style that reads charges and
        // `exclusions` and never looks at `pairs`, so gating it on `pairs`
        // deleted a system's whole long-range electrostatics whenever the
        // caller had not built a pair list.
        let gated =
            entry.map_or(RowSource::CategoryBlock, |e| e.row_source()) == RowSource::CategoryBlock;
        // A pair style's compiled terms are the frame's `pairs` list; every
        // other category's, its own block.
        let block: &str = if spec.is_pair_driven() {
            PAIRS
        } else {
            &spec.block
        };
        let topo_block = gated.then_some(block);
        if let Some(block_name) = topo_block {
            let rows = frame.get(block_name).and_then(|b| b.n_rows()).unwrap_or(0);
            if rows == 0 {
                return Ok(None);
            }
        }
        let type_params = style.defs().kernel_type_params()?;
        // A style whose kernel resolves its parameters from type rows
        // (`ParamSource::TypeRows`) can resolve nothing without them — so no rows
        // is an error, not a silently-zero potential. A `PerInstance` style
        // (MMFF's bonded terms, `coul/cut`, `pme`) reads its numbers from Frame
        // columns the typifier baked and ignores `tp` entirely, so zero rows is
        // its *normal* state. The registry says which one this is; a blanket
        // `category != "pair"` test would let MMFF register as table-driven and
        // then be fed 4,065 rows of XML no code reads.
        //
        // The registry is the authority because it is where the kernel is declared;
        // an unregistered style falls through to `TypeRows` here and then fails on
        // the kernel lookup below with a more specific message.
        let param_source = param_source_of(entry, style, &type_params)?;
        let type_refs: Vec<(&str, &Params)> = type_params
            .iter()
            .map(|(name, params)| (name.as_str(), params))
            .collect();
        let entry = with_fallback(reg, spec, style, entry, &type_refs)?;
        // Project the ForceField's `special_bonds` 1-4 weights into the params the
        // pair kernel reads (`lj14scale` / `coulomb14scale`), so the kernel scales
        // 1-4-flagged pairs without the registry signature carrying special_bonds.
        // Bonded kernels see their params unchanged.
        let params: Cow<Params> = if spec.is_pair_driven() {
            let special_bonds = match special_bonds {
                Some(sb) => sb,
                None => self.ff.required_special_bonds()?,
            };
            let mut p = style.params().clone();
            p.set(crate::ff::ir::LJ_14_SCALE, special_bonds.lj_14());
            p.set(crate::ff::ir::COULOMB_14_SCALE, special_bonds.coul_14());
            Cow::Owned(p)
        } else {
            Cow::Borrowed(style.params())
        };
        let frame = self.own_rows(reg, style, topo_block, param_source, frame)?;
        let pot = entry.compiled(spec, &params, &type_refs, &frame, reg.expression_compiler())?;
        Ok(Some(pot))
    }

    /// `frame` with `style`'s topology block cut to the rows whose type is one
    /// of `style`'s — when another table-driven style of the same category
    /// owns the rest (a LAMMPS `angle_style hybrid harmonic charmm`).
    ///
    /// A kernel resolves every row of its block against its own types, so
    /// without this a second style's rows are "unknown types" to the first.
    /// A row whose type no style of the category defines is still an error.
    /// Per-instance styles (MMFF's bend and stretch-bend) each price every
    /// row and are never cut; a category with one table-driven style is
    /// passed through unchanged.
    fn own_rows<'f>(
        &self,
        reg: &Registry,
        style: &Style,
        block_name: Option<&str>,
        source: ParamSource,
        frame: &'f Frame,
    ) -> Result<Cow<'f, Frame>, CompileError> {
        let table_driven = |s: &&Style| {
            reg.param_source(s.category(), s.name())
                .unwrap_or(ParamSource::TypeRows)
                == ParamSource::TypeRows
        };
        let siblings: Vec<&Style> = self
            .ff
            .get_styles(style.category())
            .into_iter()
            .filter(table_driven)
            .collect();
        let (Some(block_name), ParamSource::TypeRows, true) =
            (block_name, source, siblings.len() > 1)
        else {
            return Ok(Cow::Borrowed(frame));
        };
        let Some(block) = frame.get(block_name) else {
            return Ok(Cow::Borrowed(frame));
        };
        let types = block
            .get("type")
            .and_then(|c| c.get::<String>())
            .ok_or_else(|| format!("{block_name} block missing \"type\" column"))?;
        let names = |s: &Style| -> Result<Vec<String>, String> {
            Ok(s.defs()
                .kernel_type_params()?
                .into_iter()
                .map(|(name, _)| name)
                .collect())
        };
        let own = names(style)?;
        let all: Vec<Vec<String>> = siblings
            .iter()
            .map(|s| names(s))
            .collect::<Result<_, _>>()?;
        let mut keep = Vec::new();
        for (row, label) in types.iter().enumerate() {
            if own.contains(label) {
                keep.push(row);
            } else if !all.iter().flatten().any(|n| n == label) {
                return Err(format!(
                    "{block_name} row {row}: type '{label}' is defined by no {} style",
                    style.category()
                )
                .into());
            }
        }
        let mut cut = frame.clone();
        cut.insert(
            block_name,
            block.select_rows(&keep).map_err(|e| e.to_string())?,
        );
        Ok(Cow::Owned(cut))
    }
}

/// Where `style`'s kernel reads its numbers: the registered spec's source,
/// else — for an unregistered style priced by its `expression` — the
/// instances. A style read from type rows that has none is an error, never a
/// silently-zero potential.
fn param_source_of(
    entry: Option<&RegisteredStyle>,
    style: &Style,
    type_params: &[(String, Params)],
) -> Result<ParamSource, CompileError> {
    let source = entry.map_or_else(
        || {
            if style
                .params()
                .get("expression")
                .and_then(|v| v.as_text())
                .is_some()
            {
                ParamSource::PerInstance
            } else {
                ParamSource::TypeRows
            }
        },
        |e| e.spec().source,
    );
    if type_params.is_empty() && source == ParamSource::TypeRows {
        return Err(format!(
            "Style '{}' ({}) has no type definitions",
            style.name(),
            style.category()
        )
        .into());
    }
    Ok(source)
}

/// `style`'s params and kernel rows gathered under its spec in `ff`'s
/// registry ([`StyleSpec::gather`](ir::StyleSpec::gather)): what
/// a reader of a built-in style's numbers outside a kernel (the 1-4
/// exceptions, `materialize_one_four`) reads, so a declared default is the
/// same number there as in the kernel. An unregistered style is as stated.
pub(crate) fn gathered(
    ff: &ForceField,
    style: &Style,
) -> Result<(Params, Vec<(String, Params)>), CompileError> {
    let rows = style.defs().kernel_type_params()?;
    let spec = ff.registry().with(|r| {
        r.style(style.category(), style.name())
            .map(|(spec, _)| spec.clone())
    });
    let Some(spec) = spec else {
        return Ok((style.params().clone(), rows));
    };
    let refs: Vec<(&str, &Params)> = rows.iter().map(|(l, p)| (l.as_str(), p)).collect();
    Ok(spec.gather(style.params(), &refs)?)
}

/// The registered entry of `style` (checking an `expression` of its own
/// that differs from the registry's against the registered kernel, D16),
/// else — when it carries an `expression` style param — the entry its
/// expression prices it under (the compile fallback, protocol §4), else
/// [`ir::IrError::NoKernel`].
fn with_fallback<'r>(
    reg: &'r Registry,
    category: &CategorySpec,
    style: &Style,
    entry: Option<&'r RegisteredStyle>,
    tp: &[(&str, &Params)],
) -> Result<Cow<'r, RegisteredStyle>, CompileError> {
    if let Some(e) = entry {
        return Ok(e.with_instance_expression(
            category,
            style.params().get("expression").and_then(|v| v.as_text()),
            reg.expression_compiler(),
        )?);
    }
    match RegisteredStyle::fallback(
        category,
        style.name(),
        style.params(),
        tp,
        reg.expression_compiler(),
    ) {
        Some(built) => Ok(Cow::Owned(built?)),
        None => Err(no_kernel(style)),
    }
}

/// The registered category of `style`.
///
/// A category no registry declares that the force field holds anyway (one
/// read from a record, its arity stated) behaves as
/// [`CategorySpec::custom`] of its arity, a compound category: priced from
/// its block `<name>s` by its style's `expression`, refused by name
/// ([`ir::IrError::NoKernel`]) when it has rows and nothing prices it, and
/// never priced below two endpoints (molrec: no energy). One of the IR's
/// [built-in categories](ir::builtin_categories) missing from the registry
/// is an error naming it.
pub(crate) fn category_of<'r>(
    reg: &'r Registry,
    style: &Style,
) -> Result<Cow<'r, CategorySpec>, CompileError> {
    let category = style.category();
    if let Some(spec) = reg.category(category) {
        return Ok(Cow::Borrowed(spec));
    }
    if ir::builtin_categories().iter().any(|c| c.name == category) {
        return Err(ir::IrError::UnknownCategory {
            category: category.to_owned(),
        }
        .into());
    }
    let arity = u8::try_from(style.arity()).expect("a type names at most five endpoints");
    let coordinate = if arity >= 2 {
        Coordinate::Compound
    } else {
        Coordinate::None
    };
    Ok(Cow::Owned(CategorySpec::custom(
        category.to_owned(),
        arity,
        coordinate,
        EndpointOrder::Reversible,
    )))
}

fn no_kernel(style: &Style) -> CompileError {
    ir::IrError::NoKernel {
        category: style.category().to_owned(),
        style: style.name().to_owned(),
    }
    .into()
}

/// `frame` as the regular pair kernels see it: without the `pairs` rows a
/// per-pair override hands to the exceptions kernel, and without the override
/// columns.
fn regular_pairs<'f>(
    frame: &'f Frame,
    override_rows: &[usize],
) -> Result<Cow<'f, Frame>, CompileError> {
    if override_rows.is_empty() {
        return Ok(Cow::Borrowed(frame));
    }
    let Some(block) = frame.get(PAIRS) else {
        return Ok(Cow::Borrowed(frame));
    };
    let n = block.n_rows().unwrap_or(0);
    let keep: Vec<usize> = (0..n)
        .filter(|r| override_rows.binary_search(r).is_err())
        .collect();
    let mut kept = block.select_rows(&keep).map_err(|e| e.to_string())?;
    for key in PAIR_OVERRIDE_COLUMNS {
        kept.remove(key);
    }
    let mut out = frame.clone();
    out.insert(PAIRS, kept);
    Ok(Cow::Owned(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::core::Block;
    use molrs::op::{F, Idx};
    use ndarray::Array1;

    /// A field that prices pairs states its 1-2 / 1-3 / 1-4 weights: both
    /// doors refuse one that declares none, and a field of bonded styles
    /// alone needs none.
    #[test]
    fn a_pair_style_without_special_bonds_is_refused_by_both_doors() {
        let mut ff = ForceField::new("undeclared");
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_iter([("epsilon", 0.1), ("sigma", 3.0)]),
            )
            .unwrap();
        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        let compiler = PotentialCompiler::new(&ff);
        for err in [
            compiler.compile(&frame).map(|_| ()).unwrap_err(),
            compiler.compile_typed(&frame).map(|_| ()).unwrap_err(),
        ] {
            assert!(
                err.to_string().contains("declares no special_bonds"),
                "{err}"
            );
        }

        let mut bonded = ForceField::new("bonded");
        bonded
            .def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-A",
                &["A", "A"],
                Params::from_iter([("k", 1.0), ("r0", 1.0)]),
            )
            .unwrap();
        assert!(PotentialCompiler::new(&bonded).compile(&frame).is_ok());
    }

    fn two_atoms() -> Frame {
        let mut atoms = Block::new();
        for (key, v) in [("x", [0.0, 1.6]), ("y", [0.0, 0.0]), ("z", [0.0, 0.0])] {
            atoms
                .insert(key, Array1::from_vec(v.to_vec()).into_dyn(), None)
                .unwrap();
        }
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame
    }

    fn with_bond(mut frame: Frame, label: &str) -> Frame {
        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn(), None)
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn(), None)
            .unwrap();
        bonds
            .insert(
                "type",
                Array1::from_vec(vec![label.to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("bonds", bonds);
        frame
    }

    fn bond_ff() -> ForceField {
        let mut ff = ForceField::new("t");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_iter([("k", 300.0), ("r0", 1.5)]),
            )
            .unwrap();
        ff
    }

    #[test]
    fn an_atom_style_carries_no_kernel() {
        let mut ff = ForceField::new("t");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], Params::from_iter([("mass", 12.0)]))
            .unwrap();
        let frame = two_atoms();
        let compiler = PotentialCompiler::new(&ff);
        assert!(compiler.compile(&frame).unwrap().members().is_empty());
        assert!(compiler.compile_typed(&frame).unwrap().is_empty());
    }

    #[test]
    fn a_bonded_style_is_skipped_without_its_block_and_built_with_it() {
        let ff = bond_ff();
        let compiler = PotentialCompiler::new(&ff);
        let bare = two_atoms();
        assert!(compiler.compile(&bare).unwrap().members().is_empty());
        assert!(compiler.compile_typed(&bare).unwrap().is_empty());

        let bonded = with_bond(two_atoms(), "CT-CT");
        let pots = compiler.compile(&bonded).unwrap();
        assert_eq!(pots.members().len(), 1);
        // k·(r − r0)² = 300·(1.6 − 1.5)² = 3 kcal/mol (LAMMPS bond harmonic)
        let e = pots.energy_at(&[0.0, 0.0, 0.0, 1.6, 0.0, 0.0]);
        assert!((e - 3.0).abs() < 1e-10, "{e}");
        assert_eq!(compiler.compile_typed(&bonded).unwrap().len(), 1);
    }

    /// A cmap style is gated on the `cmaps` block like any bonded style, and
    /// a present block is built by both doors into one member (here a
    /// degenerate crossterm, which `fix cmap` and molrs price at zero).
    #[test]
    fn a_cmap_style_is_gated_on_cmaps_and_built_by_both_doors() {
        let mut params = Params::new();
        params.set("grid", ndarray::ArrayD::zeros(vec![24, 24]));
        let mut ff = ForceField::new("t");
        ff.def_style("cmap", "charmm", Params::new())
            .unwrap()
            .def_type("c", &["C", "N", "CA", "C", "N"], params)
            .unwrap();
        let compiler = PotentialCompiler::new(&ff);
        assert!(compiler.compile(&two_atoms()).unwrap().members().is_empty());
        assert!(compiler.compile_typed(&two_atoms()).unwrap().is_empty());

        let mut frame = two_atoms();
        let mut cmaps = Block::new();
        for key in ["atomi", "atomj", "atomk", "atoml", "atomm"] {
            cmaps
                .insert(key, Array1::from_vec(vec![0 as Idx]).into_dyn(), None)
                .unwrap();
        }
        cmaps
            .insert(
                "type",
                Array1::from_vec(vec!["c".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("cmaps", cmaps);
        let pots = compiler.compile(&frame).unwrap();
        assert_eq!(pots.members().len(), 1);
        assert_eq!(pots.energy_at(&[0.0, 0.0, 0.0, 1.6, 0.0, 0.0]), 0.0);
        let typed = compiler.compile_typed(&frame).unwrap();
        assert_eq!(typed.len(), 1);
        assert!(
            typed[0].1.is_none(),
            "a crossterm takes no special-bonds weight"
        );
    }

    #[test]
    fn an_unknown_type_label_in_a_present_block_is_an_error() {
        let ff = bond_ff();
        let frame = with_bond(two_atoms(), "XX-XX");
        assert!(PotentialCompiler::new(&ff).compile(&frame).is_err());
    }

    #[test]
    fn compile_typed_refuses_a_pair_style_without_a_neighbour_driven_form() {
        let mut ff = ForceField::new("t");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style(
            "pair",
            "coul/long/pme",
            Params::from_iter([("coulomb", 332.06371), ("cutoff", 9.0)]),
        )
        .unwrap();
        let err = PotentialCompiler::new(&ff)
            .compile_typed(&two_atoms())
            .unwrap_err();
        assert!(err.to_string().contains("neighbour-driven"), "{err}");
    }

    /// The pairs of [`straddling`], and which of them are flagged 1-4.
    const LINKS: [(usize, usize); 4] = [(0, 1), (0, 2), (0, 3), (1, 3)];
    const IS_14: [bool; 4] = [true, false, true, false];

    /// Four atoms on x at 0, 2.0, 3.1 and 7.0 Å, charged, and the list of
    /// four of their pairs ([`LINKS`]): (0, 1) 2.0 Å and (0, 3) 7.0 Å flagged
    /// 1-4, (0, 2) 3.1 Å and (1, 3) 5.0 Å regular — each kind on each side of
    /// a 4 Å cutoff.
    fn straddling() -> (Frame, Vec<F>) {
        let x = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.1, 0.0, 0.0, 7.0, 0.0, 0.0];
        let (links, is_14) = (LINKS, IS_14);
        let mut atoms = Block::new();
        for (d, key) in ["x", "y", "z"].iter().enumerate() {
            let col: Vec<F> = x.iter().skip(d).step_by(3).copied().collect();
            atoms
                .insert(*key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(); 4]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "charge",
                Array1::from_vec(vec![0.5, -0.4, 0.3, 0.2]).into_dyn(),
                None,
            )
            .unwrap();
        let mut pairs = Block::new();
        let col = |k: usize| -> Vec<Idx> { links.iter().map(|l| [l.0, l.1][k] as Idx).collect() };
        pairs
            .insert("atomi", Array1::from_vec(col(0)).into_dyn(), None)
            .unwrap();
        pairs
            .insert("atomj", Array1::from_vec(col(1)).into_dyn(), None)
            .unwrap();
        pairs
            .insert("is_14", Array1::from_vec(is_14.to_vec()).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("pairs", pairs);
        (frame, x)
    }

    /// The `pairs`-list door truncates every pair style at its `cutoff`
    /// (`r < cutoff`), 1-4 pairs included, and shifts a shifted `lj/cut`
    /// there — LAMMPS's `lj/cut` with `pair_modify shift yes` and
    /// `coul/cut`, by hand — and prices what the neighbour-driven door
    /// prices over the same pairs with the same weights.
    #[test]
    fn compile_truncates_every_pair_at_its_cutoff_as_lammps_and_compile_typed() {
        use crate::ff::potential::pair::fixtures::table_over;
        let (frame, x) = straddling();
        let (links, is_14) = (LINKS, IS_14);
        let (rc, w14) = (4.0, 0.5);
        let special = SpecialBonds {
            lj: [0.0, 0.0, w14],
            coul: [0.0, 0.0, w14],
        };
        let r = |(i, j): (usize, usize)| (x[3 * j] - x[3 * i]).abs();
        let weight = |k: usize| if is_14[k] { w14 } else { 1.0 };
        // (style, its row, its style params besides `cutoff`)
        type Case<'a> = (&'a str, Vec<(&'a str, F)>, Vec<(&'a str, F)>);
        let cases: [Case; 6] = [
            (
                "lj/cut",
                vec![("epsilon", 0.2), ("sigma", 3.0)],
                vec![("shift", 1.0)],
            ),
            ("lj/cut", vec![("epsilon", 0.2), ("sigma", 3.0)], vec![]),
            ("lj/class2", vec![("epsilon", 0.2), ("sigma", 3.0)], vec![]),
            (
                "buck",
                vec![("a", 1000.0), ("rho", 0.3), ("c", 50.0)],
                vec![],
            ),
            (
                "morse",
                vec![("d0", 0.5), ("alpha", 1.2), ("r0", 3.0)],
                vec![],
            ),
            (
                "coul/cut",
                vec![],
                vec![("coulomb", 332.06371), ("dielectric", 1.0)],
            ),
        ];
        let lj = |r: F| 0.8 * ((3.0 / r).powi(12) - (3.0 / r).powi(6));
        for (name, row, style) in cases {
            let mut ff = ForceField::new("t");
            ff.set_special_bonds(special);
            let mut sp = Params::from_iter(style.iter().copied());
            sp.set("cutoff", rc);
            let s = ff.def_style("pair", name, sp).unwrap();
            if !row.is_empty() {
                s.def_type("A", &["A"], Params::from_iter(row.iter().copied()))
                    .unwrap();
            }
            let compiler = PotentialCompiler::new(&ff);
            let (e, f) = compiler.compile(&frame).unwrap().energy_forces_at(&x);
            let shifted = style.iter().any(|(k, _)| *k == "shift");
            let by_hand: F = links
                .iter()
                .enumerate()
                .filter(|&(_, &l)| r(l) < rc)
                .map(|(k, &l)| {
                    let q = [0.5, -0.4, 0.3, 0.2];
                    weight(k)
                        * match name {
                            "lj/cut" if shifted => lj(r(l)) - lj(rc),
                            "lj/cut" => lj(r(l)),
                            "coul/cut" => 332.06371 * q[l.0] * q[l.1] / r(l),
                            _ => 0.0,
                        }
                })
                .sum();
            if matches!(name, "lj/cut" | "coul/cut") {
                assert!(
                    (e - by_hand).abs() <= 1e-12 * by_hand.abs(),
                    "{name} shift={shifted}: compile {e}, LAMMPS by hand {by_hand}"
                );
            }
            let table = table_over(&x, &links);
            let factor: Vec<F> = (0..links.len()).map(weight).collect();
            let mut ft = vec![0.0; x.len()];
            let mut acc = crate::ff::potential::Accumulator::adding_into(&mut ft);
            for (member, _) in compiler.compile_typed(&frame).unwrap() {
                let ForceTerm::Pair(p) = &member else {
                    panic!("a pair member")
                };
                p.evaluate_pairs(&x, &table, &factor, &mut acc);
            }
            let et = acc.energy();
            let scale = f.iter().fold(e.abs(), |m, v| m.max(v.abs()));
            assert!(scale > 1e-6, "{name}: the pairs inside must contribute");
            assert!(
                (e - et).abs() <= 1e-12 * scale,
                "{name} shift={shifted}: compile {e}, compile_typed {et}"
            );
            for (a, b) in f.iter().zip(&ft) {
                assert!((a - b).abs() <= 1e-12 * scale, "{name}: force {a} vs {b}");
            }
        }
    }

    #[test]
    fn compile_refuses_a_1_2_or_1_3_weight_other_than_0_or_1() {
        let mut ff = bond_ff();
        // A fraction on 1-3: a compiled pairs list carries that class by row
        // presence, which cannot say "half".
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.5, 0.5],
            coul: [0.0, 0.5, 0.5],
        });
        let frame = with_bond(two_atoms(), "CT-CT");
        let err = PotentialCompiler::new(&ff).compile(&frame).unwrap_err();
        assert!(err.to_string().contains("1-3"), "{err}");

        // A fraction on 1-2 is refused the same way.
        let mut ff = bond_ff();
        ff.set_special_bonds(SpecialBonds {
            lj: [0.5, 0.0, 0.5],
            coul: [0.5, 0.0, 0.5],
        });
        let err = PotentialCompiler::new(&ff).compile(&frame).unwrap_err();
        assert!(err.to_string().contains("1-2"), "{err}");
    }
}
