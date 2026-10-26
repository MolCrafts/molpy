//! Compiling a force field: a [`ForceField`]'s declarations bound, through
//! the style registry ([`crate::ff::style_registry`]), to the kernels of
//! [`crate::ff::potential`] on a typed [`Frame`].
//!
//! * [`PotentialCompiler`] compiles a force field against a frame;
//! * [`ExplicitTerms`] compiles one style over explicit instances;
//! * the frame's special pairs (its `pairs` block) weight the pairs the pair
//!   styles price, and the 1-4 exceptions (`exceptions`) decide which of them
//!   the [`PairExceptions`] kernel prices instead, and at what parameters.

pub(crate) mod exceptions;
mod explicit_terms;

pub use explicit_terms::ExplicitTerms;

use crate::ff::style_registry;
use std::borrow::Cow;

use crate::ff::forcefield::{ForceField, Style};
use crate::ff::ir::Params;
use crate::ff::ir::{self, CategorySpec, Coordinate, EndpointOrder};
use crate::ff::ir::{ParamSource, SpecialClass};
use crate::ff::potential::pair::{CandidatePairs, PairExceptions};
use crate::ff::potential::{CompileError, ForceTerm, Potentials, ScaledTerm, SpecialWeights};
use crate::ff::special_pairs::SpecialPairs;
use crate::ff::style_registry::RowSource;
use crate::ff::style_registry::{PairContext, RegisteredStyle};
use crate::ff::style_registry::{Registry, RegistryRef};
use molrs::core::schema::block_names::ATOMS;
use molrs::core::{Frame, Neighbors};

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
/// [`PotentialCompiler`] reads one force field and has two doors; they
/// differ in where the candidate non-bonded pairs come from, and weigh them
/// alike, from the frame's `pairs` block (its special pairs):
///
/// * [`PotentialCompiler::compile`] prices every pair of atoms — right for a
///   molecule in free space, and what the geometry optimizer and the
///   conformer pipeline use; [`compile_over`](PotentialCompiler::compile_over)
///   bounds those candidates by a neighbour table.
/// * [`PotentialCompiler::compile_typed`] resolves the pair styles against
///   the **atoms**, so the kernels answer for whatever pairs a neighbour
///   search turns up. Right for a periodic box, and what MD uses.
///
/// Both doors price a pair exactly as LAMMPS does: only at `r < cutoff`, the
/// style's own `cutoff` (and between `inner` and `cutoff` through the
/// switch of a style that defines one, `lj/charmm`, `coul/charmm`). A pair
/// the `pairs` block names takes its weight (its override cell, else the
/// force field's `special_bonds` for its `n_bonds`); every other pair is at
/// full strength. The 1-4 exceptions kernel (a `dihedral charmm` `w`, a row's
/// parameter cells, `lj/charmm`'s `epsilon14`) is no pair style and is never
/// truncated, as LAMMPS's `dihedral_style charmm` prices its 1-4 pair at any
/// distance.
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
            RegistryRef::Global => Cow::Owned(style_registry::GlobalRegistry.with(Registry::clone)),
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
    /// The pair styles price every pair of atoms at its weight: a pair the
    /// frame's `pairs` block names at its row's (its override cell, else the
    /// force field's `special_bonds` for its `n_bonds`), every other pair at
    /// full strength, each inside the style's `cutoff`. `pair coul/long/pme`
    /// prices the same weights in its own minimum-image loop.
    ///
    /// # Errors
    ///
    /// Beside a kernel's own refusals: a pair style on a frame whose `bonds`
    /// name a pair no `pairs` row does (the special pairs were never
    /// generated, or a bond was added after them); a list-priced pair style
    /// on a frame with a periodic box, whose candidate pairs are priced at raw
    /// distances (evaluate it through [`compile_typed`](Self::compile_typed));
    /// and a frame of more than [`MAX_ATOMS_FOR_ALL_PAIRS`] atoms.
    pub fn compile(&self, frame: &Frame) -> Result<Potentials, CompileError> {
        self.compile_listed(frame, None)
    }

    /// [`compile`](Self::compile) with the candidate pairs taken from a
    /// cutoff-bounded neighbour table instead of every pair of atoms.
    ///
    /// For a molecule too large for every pair, the table bounds the work;
    /// the frame's special pairs still weigh what it holds. The frame is not
    /// changed.
    pub fn compile_over(
        &self,
        frame: &Frame,
        neighbors: &Neighbors,
    ) -> Result<Potentials, CompileError> {
        self.compile_listed(frame, Some(neighbors))
    }

    fn compile_listed(
        &self,
        frame: &Frame,
        neighbors: Option<&Neighbors>,
    ) -> Result<Potentials, CompileError> {
        let reg = self.registry();
        let pairing = self.pairing(&reg, frame)?;
        let candidates = match &pairing {
            Some(p) if p.list_priced => Some(p.candidates(frame, neighbors)?),
            _ => None,
        };
        self.compile_members(&reg, frame, pairing, candidates)
    }

    /// The members of `frame` with the pair styles pricing exactly
    /// `candidates` at their weights, and no special pair: the terms a caller
    /// listed by hand ([`ExplicitTerms`]).
    pub(crate) fn compile_candidates(
        &self,
        frame: &Frame,
        candidates: CandidatePairs,
    ) -> Result<Potentials, CompileError> {
        let reg = self.registry();
        let pairing = Pairing {
            lj: SpecialWeights::default(),
            coul: SpecialWeights::default(),
            exceptions: None,
            list_priced: true,
            no_candidates: CandidatePairs::default(),
        };
        self.compile_members(&reg, frame, Some(pairing), Some(candidates))
    }

    fn compile_members(
        &self,
        reg: &Registry,
        frame: &Frame,
        pairing: Option<Pairing>,
        candidates: Option<CandidatePairs>,
    ) -> Result<Potentials, CompileError> {
        let context = pairing.as_ref().map(|p| PairContext {
            candidates: candidates.as_ref().unwrap_or(&p.no_candidates),
            coulomb: &p.coul,
        });
        let mut pots = Potentials::new();
        for style in self.ff.styles() {
            // `member` skips a style whose topology block is absent or empty; a
            // *present* block with an unknown type label is a real error and
            // propagates from the kernel constructor.
            if let Some(pot) = self.member(reg, style, frame, context.as_ref())? {
                pots.push(pot);
            }
        }
        if let Some(kernel) = pairing.and_then(|p| p.exceptions) {
            pots.push(ForceTerm::indexed(kernel));
        }
        // Record the atom count so callers (e.g. the geometry optimizer's batch
        // path) can validate coordinate shapes against this topology.
        pots.set_n_atoms(n_atoms_of(frame));
        Ok(pots)
    }

    /// The weights of the frame's pairs under this force field, and the
    /// exceptions kernel; `None` when the force field has no pair style.
    fn pairing(&self, reg: &Registry, frame: &Frame) -> Result<Option<Pairing>, CompileError> {
        let mut any = false;
        let mut list_priced = false;
        for style in self.ff.styles() {
            let category = category_of(reg, style)?;
            if !category.is_pair_driven() || !category.prices_energy() {
                continue;
            }
            any = true;
            list_priced |= row_source(reg, style) == RowSource::CategoryBlock;
        }
        if !any {
            // A `dihedral charmm` `w > 0` prices a 1-4 pair through pair
            // styles this field does not have: LAMMPS's refusal, kept here.
            exceptions::dihedral_weights(self.ff, frame)?;
            return Ok(None);
        }
        let special = self.ff.required_special_bonds()?;
        let pairs = SpecialPairs::of(frame)?.unwrap_or_default();
        pairs.require_bonds_covered(frame)?;
        let exceptions = exceptions::plan(self.ff, frame, &pairs, special)?;
        let n = n_atoms_of(frame);
        Ok(Some(Pairing {
            lj: pairs.weights(SpecialClass::Vdw, special, &exceptions.replaced, n),
            coul: pairs.weights(SpecialClass::Coulomb, special, &exceptions.replaced, n),
            exceptions: exceptions.kernel,
            list_priced,
            no_candidates: CandidatePairs::default(),
        }))
    }

    /// Build the members of a **neighbour-driven** force evaluation, each with
    /// the weights its non-bonded term takes.
    ///
    /// The counterpart of [`compile`](Self::compile), and what periodic MD
    /// needs: the pair styles are resolved against the **atoms**, so the
    /// kernels can answer for whatever pairs a neighbour search turns up; a
    /// pair style needs a finite `cutoff` here. Over the same pairs, the two
    /// doors price the same energy. `pair coul/long/pme`'s real-space erfc
    /// follows the neighbour table; its reciprocal sum reads the periodic box
    /// and corrects the special pairs by their weights.
    ///
    /// The weights come back per member, because a force field may scale
    /// close van-der-Waals and electrostatic neighbours differently — Amber
    /// uses `1/2` and `1/1.2` — and in molrs those are separate kernels. A
    /// bonded member, and the exceptions kernel, take the empty table: they
    /// *are* their interactions, not a scaled copy of them. A neighbour table
    /// finds every pair inside the cutoff, bonded or not, so without the
    /// weights a bonded pair would be counted twice: once by the bond term and
    /// once at full non-bonded strength, at bond length.
    pub fn compile_typed(
        &self,
        frame: &Frame,
    ) -> Result<Vec<(ForceTerm, SpecialWeights)>, CompileError> {
        let reg = self.registry();
        let pairing = self.pairing(&reg, frame)?;
        let none = SpecialWeights::default();
        let coul = pairing.as_ref().map_or(&none, |p| &p.coul);
        let mut out = Vec::new();
        for style in self.ff.styles() {
            // A bonded style contributes nothing when the molecule carries no
            // topology of its kind (`member` skips it). A pair style is never
            // skipped: which pairs exist is the neighbour search's answer, not
            // the frame's.
            if let Some((pot, class)) = self.typed_member(&reg, style, frame, coul)? {
                let weights = match (class, &pairing) {
                    (Some(SpecialClass::Vdw), Some(p)) => p.lj.clone(),
                    (Some(SpecialClass::Coulomb), Some(p)) => p.coul.clone(),
                    _ => SpecialWeights::default(),
                };
                out.push((pot, weights));
            }
        }
        if let Some(kernel) = pairing.and_then(|p| p.exceptions) {
            out.push((ForceTerm::indexed(kernel), SpecialWeights::default()));
        }
        Ok(out)
    }

    /// Build `style`'s kernel for a **neighbour-driven** evaluation, and say
    /// which special-bonds weights scale it.
    ///
    /// The counterpart of [`member`](Self::member). A bonded style is built
    /// identically — it reads indices, and a neighbour table does not concern
    /// it. A pair style is built in its typed form, a type-pair table that
    /// prices whatever pairs the neighbour table holds: the table is rebuilt
    /// every few steps, and a kernel whose parameters were resolved against
    /// an older one would be naming different atoms.
    ///
    /// A pair style with no typed form is an [`Err`], not a fallback to the
    /// compiled one. Falling back would hand back a kernel whose parameters
    /// belong to a pair list nobody is evaluating, and it would answer.
    fn typed_member(
        &self,
        reg: &Registry,
        style: &Style,
        frame: &Frame,
        coulomb: &SpecialWeights,
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
                coulomb,
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
        pairs: Option<&PairContext<'_>>,
    ) -> Result<Option<ForceTerm>, CompileError> {
        let spec = category_of(reg, style)?;
        let spec = &*spec;
        if !spec.prices_energy() {
            return Ok(None);
        }
        let category = style.category();
        let entry = reg.registered_style(category, style.name());
        // A style contributes nothing when the molecule carries no topology of
        // its kind: a bonded style with no bonds/angles/dihedrals/impropers, or
        // a list-priced pair style with no pair to price (methane: every pair
        // is 1-2 or 1-3, weighted 0). Skip it rather than letting the kernel
        // fault on the absent block. Which rows gate a style is the *kernel's*
        // property, not the category's: PME is a `pair` style whose row source
        // is the atoms, and prices the frame whatever its pairs.
        let gated =
            entry.map_or(RowSource::CategoryBlock, |e| e.row_source()) == RowSource::CategoryBlock;
        let topo_block = (gated && !spec.is_pair_driven()).then_some(spec.block.as_ref());
        if let Some(block_name) = topo_block {
            let rows = frame.get(block_name).and_then(|b| b.n_rows()).unwrap_or(0);
            if rows == 0 {
                return Ok(None);
            }
        }
        if gated && spec.is_pair_driven() && pairs.is_none_or(|p| p.candidates.len() == 0) {
            return Ok(None);
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
        let frame = self.own_rows(reg, style, topo_block, param_source, frame)?;
        let pot = entry.compiled(
            spec,
            style.params(),
            &type_refs,
            &frame,
            reg.expression_compiler(),
            pairs.filter(|_| spec.is_pair_driven()),
        )?;
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

/// The most atoms whose every pair [`PotentialCompiler::compile`] prices:
/// `N(N−1)/2` candidates with no cutoff is the shape of that list, and past a
/// certain `N` its failure mode is the OOM killer. 20 000 atoms is about 2·10⁸
/// pairs; a larger molecule wants
/// [`compile_over`](PotentialCompiler::compile_over) or, periodic,
/// [`compile_typed`](PotentialCompiler::compile_typed).
pub const MAX_ATOMS_FOR_ALL_PAIRS: usize = 20_000;

/// The weights a compile prices the frame's pairs at, and its exceptions.
struct Pairing {
    lj: SpecialWeights,
    coul: SpecialWeights,
    exceptions: Option<PairExceptions>,
    /// Whether a pair style prices a candidate list (all but PME).
    list_priced: bool,
    /// The candidates of a compile that prices no list.
    no_candidates: CandidatePairs,
}

impl Pairing {
    /// The candidate pairs, each with its weight in both classes: every pair
    /// of atoms, or the pairs of `neighbors`; a pair both classes weigh 0 is
    /// left out.
    fn candidates(
        &self,
        frame: &Frame,
        neighbors: Option<&Neighbors>,
    ) -> Result<CandidatePairs, CompileError> {
        let n = n_atoms_of(frame);
        let mut out = CandidatePairs::default();
        let mut add = |i: usize, j: usize| {
            let (lj, coul) = (self.lj.weight(i, j), self.coul.weight(i, j));
            if lj != 0.0 || coul != 0.0 {
                out.push(i, j, lj, coul);
            }
        };
        let Some(neighbors) = neighbors else {
            if frame
                .simbox
                .as_ref()
                .is_some_and(|b| b.pbc().iter().any(|&p| p))
            {
                return Err(CompileError::Invalid(
                    "compile: the frame has a periodic box, and compile prices its pairs at \
                     raw distances with no periodic images. Evaluate a periodic system \
                     through compile_typed"
                        .into(),
                ));
            }
            if n > MAX_ATOMS_FOR_ALL_PAIRS {
                return Err(CompileError::Invalid(format!(
                    "compile: {n} atoms would price {} pairs with no cutoff; above \
                     {MAX_ATOMS_FOR_ALL_PAIRS} atoms bound them with compile_over, or \
                     evaluate through compile_typed",
                    n * (n - 1) / 2
                )));
            }
            for i in 0..n {
                for j in (i + 1)..n {
                    add(i, j);
                }
            }
            return Ok(out);
        };
        let mut seen = std::collections::HashSet::new();
        for (&a, &b) in neighbors
            .query_point_indices()
            .iter()
            .zip(neighbors.point_indices())
        {
            let (lo, hi) = (a.min(b) as usize, a.max(b) as usize);
            if lo != hi && seen.insert((lo, hi)) {
                add(lo, hi);
            }
        }
        Ok(out)
    }
}

/// The frame's atom count.
fn n_atoms_of(frame: &Frame) -> usize {
    frame.get(ATOMS).and_then(|b| b.n_rows()).unwrap_or(0)
}

/// Where `style`'s compiled terms come from: its category's block (a pair
/// style's: its candidate pairs), or the atoms (PME). An unregistered style
/// is block-priced.
fn row_source(reg: &Registry, style: &Style) -> RowSource {
    reg.row_source(style.category(), style.name())
        .unwrap_or(RowSource::CategoryBlock)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::ir::SpecialBonds;
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

    /// A chain 0-1-2-3 along x with its special pairs; `scale` is the
    /// (0, 3) row's `lj_scale` cell, when it has one.
    fn chain_with_pairs(scale: Option<F>) -> Frame {
        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(); 4]).into_dyn(),
                None,
            )
            .unwrap();
        for (key, v) in [
            ("x", [0.0, 1.5, 3.0, 4.5]),
            ("y", [0.0; 4]),
            ("z", [0.0; 4]),
        ] {
            atoms
                .insert(key, Array1::from_vec(v.to_vec()).into_dyn(), None)
                .unwrap();
        }
        let mut bonds = Block::new();
        bonds
            .insert(
                "atomi",
                Array1::from_vec(vec![0_u64, 1, 2]).into_dyn(),
                None,
            )
            .unwrap();
        bonds
            .insert(
                "atomj",
                Array1::from_vec(vec![1_u64, 2, 3]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);
        let mut pairs = molrs::core::Topology::from_frame(&frame)
            .unwrap()
            .pairs_block();
        if let Some(scale) = scale {
            let i = pairs.get("atomi").unwrap().get::<u64>().unwrap().to_owned();
            let j = pairs.get("atomj").unwrap().get::<u64>().unwrap().to_owned();
            let valid: Vec<bool> = (0..i.len()).map(|r| (i[[r]], j[[r]]) == (0, 3)).collect();
            let values: Vec<F> = valid.iter().map(|&v| if v { scale } else { 0.0 }).collect();
            pairs
                .insert("lj_scale", Array1::from_vec(values).into_dyn(), Some(valid))
                .unwrap();
        }
        frame.insert("pairs", pairs);
        frame
    }

    fn lj_a(special: SpecialBonds) -> ForceField {
        let mut ff = ForceField::new("lj");
        ff.set_special_bonds(special);
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_iter([("epsilon", 0.1), ("sigma", 3.0)]),
            )
            .unwrap();
        ff
    }

    /// A row's `lj_scale` cell is that pair's weight, in place of its
    /// class's: both doors price the 1-4 pair (0, 3) of the chain at it.
    #[test]
    fn a_row_scale_is_its_pairs_weight_at_both_doors() {
        let ff = lj_a(SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 0.5],
        });
        let coords: Vec<F> = (0..4).flat_map(|k| [1.5 * k as F, 0.0, 0.0]).collect();
        let lj03 = {
            let s6 = (3.0_f64 / 4.5).powi(6);
            0.4 * (s6 * s6 - s6)
        };
        for (scale, want) in [(None, 0.5), (Some(0.25), 0.25)] {
            let frame = chain_with_pairs(scale);
            let e = PotentialCompiler::new(&ff)
                .compile(&frame)
                .unwrap()
                .energy_at(&coords);
            assert!((e - want * lj03).abs() < 1e-12, "{e} vs {}", want * lj03);
            let terms = PotentialCompiler::new(&ff).compile_typed(&frame).unwrap();
            let (_, weights) = terms.iter().find(|(_, w)| !w.is_empty()).unwrap();
            assert_eq!(weights.weight(0, 3), want);
            assert_eq!(weights.weight(0, 1), 0.0);
        }
    }

    /// A fractional 1-2 or 1-3 weight is a weight like any other: the bond
    /// pair of two atoms is priced at half strength.
    #[test]
    fn a_fractional_close_weight_is_priced() {
        let mut frame = two_atoms();
        let mut atoms = frame.get("atoms").unwrap().clone();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(); 2]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("atoms", atoms);
        let frame = with_bond(frame, "A-A");
        let mut frame = frame;
        frame.insert(
            "pairs",
            molrs::core::Topology::from_frame(&frame)
                .unwrap()
                .pairs_block(),
        );
        let ff = lj_a(SpecialBonds {
            lj: [0.5, 0.0, 0.5],
            coul: [0.5, 0.0, 0.5],
        });
        let e = PotentialCompiler::new(&ff)
            .compile(&frame)
            .unwrap()
            .energy_at(&[0.0, 0.0, 0.0, 1.6, 0.0, 0.0]);
        let s6 = (3.0_f64 / 1.6).powi(6);
        let want = 0.5 * 0.4 * (s6 * s6 - s6);
        assert!((e - want).abs() < 1e-9 * want.abs(), "{e} vs {want}");
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
            typed[0].1.is_empty(),
            "a crossterm takes no special-bonds weight"
        );
    }

    #[test]
    fn an_unknown_type_label_in_a_present_block_is_an_error() {
        let ff = bond_ff();
        let frame = with_bond(two_atoms(), "XX-XX");
        assert!(PotentialCompiler::new(&ff).compile(&frame).is_err());
    }

    /// The pairs of [`straddling`] the styles price, and which of them are
    /// 1-4 pairs.
    const LINKS: [(usize, usize); 4] = [(0, 1), (0, 2), (0, 3), (1, 3)];
    const IS_14: [bool; 4] = [true, false, true, false];

    /// Four atoms on x at 0, 2.0, 3.1 and 7.0 Å, charged, whose special pairs
    /// leave four pairs priced ([`LINKS`]): (0, 1) 2.0 Å and (0, 3) 7.0 Å
    /// 1-4 pairs, (0, 2) 3.1 Å and (1, 3) 5.0 Å at full strength — each kind
    /// on each side of a 4 Å cutoff — and (1, 2), (2, 3) excluded.
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
        // (pair, n_bonds, excluded): the 1-4 pairs by their class, the two
        // pairs no style prices by a 0 weight.
        let rows: [((Idx, Idx), Idx, bool); 4] = [
            ((0, 1), 3, false),
            ((0, 3), 3, false),
            ((1, 2), 0, true),
            ((2, 3), 0, true),
        ];
        debug_assert_eq!(
            links.iter().zip(is_14).filter(|(_, f)| *f).count(),
            rows.iter().filter(|r| r.1 == 3).count()
        );
        let mut pairs = Block::new();
        for (key, col) in [
            ("atomi", rows.iter().map(|r| r.0.0).collect::<Vec<Idx>>()),
            ("atomj", rows.iter().map(|r| r.0.1).collect()),
            ("n_bonds", rows.iter().map(|r| r.1).collect()),
        ] {
            pairs
                .insert(key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        let excluded: Vec<bool> = rows.iter().map(|r| r.2).collect();
        for key in ["lj_scale", "coul_scale"] {
            pairs
                .insert(
                    key,
                    Array1::from_vec(vec![0.0; rows.len()]).into_dyn(),
                    Some(excluded.clone()),
                )
                .unwrap();
        }
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("pairs", pairs);
        (frame, x)
    }

    /// The `compile` door truncates every pair style at its `cutoff`
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
}
