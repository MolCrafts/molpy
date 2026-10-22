//! molrs's own Tier-3 kernel constructors: `(category, style_name)` →
//! [`KernelConstructor`], with the neighbour-driven form and the row source
//! of the styles that have one.
//!
//! A crate-private table, not a registry anyone extends: the force-field IR
//! registry ([`crate::ff::style_registry::Registry`], the one registry) seeds its sealed
//! built-ins from [`BuiltinKernels::builtin`], each beside its spec, and
//! `PotentialCompiler` resolves every style through that registry. A third
//! party registers through
//! [`register_style`](crate::ff::style_registry::register_style).
//!
//! Where a kernel's numbers come from ([`ParamSource`](crate::ff::ir::ParamSource))
//! is its spec's declaration ([`StyleSpec::source`](crate::ff::ir::StyleSpec::source)),
//! not this table's: MMFF's and UFF's bonded styles, `pair/coul/cut`,
//! `coul/charmm` and `coul/long/pme` read per-instance Frame columns the
//! typifier bakes (or per-atom charges) and ignore the type rows, and their
//! specs say [`ParamSource::PerInstance`](crate::ff::ir::ParamSource::PerInstance).

use std::collections::HashMap;

use crate::ff::ir::SpecialClass;
use crate::ff::style_registry::{KernelConstructor, RowSource};

use crate::ff::potential::{angle, bond, cmap, dihedral, improper, kspace, pair};

/// Maps `(category, style_name)` to the constructor that builds its kernel.
#[derive(Default)]
pub(crate) struct BuiltinKernels {
    constructors: HashMap<(String, String), Registration>,
}

/// What is known about one `(category, style_name)`.
struct Registration {
    constructor: KernelConstructor,
    /// Which block's emptiness means this style contributes nothing. Reset to
    /// the default by an override, for the reason `typed` is — a re-registered
    /// style is a different force law and inherits none of the old one's
    /// declarations.
    rows: RowSource,
    /// The neighbour-driven form of a pair style, when it has one, and which
    /// special-bonds weights scale it.
    ///
    /// The registered constructor resolves its parameters against a **pair
    /// list** and can only answer for that list. A neighbour table is a
    /// different list every rebuild, so an evaluation driven by one needs a
    /// kernel that finds its parameters from the atoms instead. Same
    /// signature, different question — so it is a second registration rather
    /// than a flag on the first.
    typed: Option<(KernelConstructor, SpecialClass)>,
}

impl BuiltinKernels {
    /// Every registered `(category, style)`, sorted.
    #[cfg(test)]
    pub(crate) fn styles(&self) -> Vec<(&str, &str)> {
        let mut out: Vec<(&str, &str)> = self
            .constructors
            .keys()
            .map(|(c, s)| (c.as_str(), s.as_str()))
            .collect();
        out.sort_unstable();
        out
    }

    /// Register (or override) the kernel for `(category, name)`.
    fn register(&mut self, category: &str, name: &str, constructor: KernelConstructor) {
        // A registration is one unit. Keeping the previous `typed` across an
        // override would leave `compile` and `compile_typed`
        // evaluating *different force fields* for the same style name, with
        // nothing to say so — so an override clears it and the caller
        // re-registers both.
        self.constructors.insert(
            (category.to_owned(), name.to_owned()),
            Registration {
                constructor,
                rows: RowSource::CategoryBlock,
                typed: None,
            },
        );
    }

    /// Declare where a registered style's rows come from.
    ///
    /// Only needed to say [`RowSource::Atoms`]; the default is the category's
    /// topology block. Like [`register_typed`](Self::register_typed) this is a
    /// second statement about a style that must already exist, and an override
    /// resets it — a re-registered style is a different force law.
    ///
    /// # Panics
    ///
    /// If `(category, name)` has no compiled registration.
    fn declare_rows(&mut self, category: &str, name: &str, rows: RowSource) {
        let r = self
            .constructors
            .get_mut(&(category.to_owned(), name.to_owned()))
            .unwrap_or_else(|| {
                panic!(
                    "declare_rows('{category}', '{name}') before the style is registered: \
                     where a kernel's rows come from is a statement about that kernel"
                )
            });
        r.rows = rows;
    }

    /// Where this style's rows come from, or `None` if it is not registered.
    #[cfg(test)]
    fn row_source(&self, category: &str, name: &str) -> Option<RowSource> {
        self.constructors
            .get(&(category.to_owned(), name.to_owned()))
            .map(|r| r.rows)
    }

    /// Register the neighbour-driven form of an already-registered pair style.
    ///
    /// A style without one cannot be evaluated over a neighbour table at all,
    /// and [`PotentialCompiler::compile_typed`](crate::ff::compile::PotentialCompiler::compile_typed)
    /// says so rather than quietly falling back to the compiled form, whose
    /// parameters would belong to a pair list nobody is evaluating.
    ///
    /// # Panics
    ///
    /// If `(category, name)` has no compiled registration. A neighbour-driven
    /// form is an *alternative* way to build a style that already exists, so a
    /// silent no-op here would leave the caller believing their style works
    /// under MD when `PotentialCompiler::compile_typed` will refuse it.
    fn register_typed(
        &mut self,
        category: &str,
        name: &str,
        constructor: KernelConstructor,
        special: SpecialClass,
    ) {
        let r = self
            .constructors
            .get_mut(&(category.to_owned(), name.to_owned()))
            .unwrap_or_else(|| {
                panic!(
                    "register_typed('{category}', '{name}') before the style is registered: \
                     a neighbour-driven form is an alternative to a compiled one, not a \
                     registration of its own"
                )
            });
        r.typed = Some((constructor, special));
    }

    /// The constructor registered for `(category, name)`, if any.
    #[cfg(test)]
    fn get(&self, category: &str, name: &str) -> Option<KernelConstructor> {
        self.constructors
            .get(&(category.to_owned(), name.to_owned()))
            .map(|r| r.constructor)
    }

    /// The neighbour-driven constructor for `(category, name)` and the weight
    /// set that scales it, if it has one.
    #[cfg(test)]
    fn get_typed(&self, category: &str, name: &str) -> Option<(KernelConstructor, SpecialClass)> {
        self.constructors
            .get(&(category.to_owned(), name.to_owned()))
            .and_then(|r| r.typed)
    }

    /// `(category, name)` as a Tier-3 [`Kernel`](crate::ff::style_registry::Kernel) of
    /// the force-field IR registry, which seeds its built-ins from here.
    pub(crate) fn kernel(
        &self,
        category: &str,
        name: &str,
    ) -> Option<crate::ff::style_registry::Kernel> {
        self.constructors
            .get(&(category.to_owned(), name.to_owned()))
            .map(|r| crate::ff::style_registry::Kernel::Constructor {
                compiled: r.constructor,
                typed: r.typed,
                rows: r.rows,
            })
    }

    /// A registry seeded with every built-in kernel.
    pub(crate) fn builtin() -> Self {
        let mut r = Self::default();
        // bonded
        r.register(
            "bond",
            "harmonic",
            bond::harmonic::bond_harmonic_constructor,
        );
        r.register("bond", "class2", bond::class2::bond_class2_constructor);
        r.register("bond", "morse", bond::morse::bond_morse_constructor);
        r.register(
            "angle",
            "harmonic",
            angle::harmonic::angle_harmonic_constructor,
        );
        r.register("angle", "class2", angle::class2::angle_class2_constructor);
        r.register(
            "dihedral",
            "opls",
            dihedral::opls::dihedral_opls_constructor,
        );
        r.register(
            "dihedral",
            "charmm",
            dihedral::charmm::dihedral_charmm_constructor,
        );
        r.register(
            "dihedral",
            "multi/harmonic",
            dihedral::multi_harmonic::dihedral_multi_harmonic_constructor,
        );
        r.register(
            "dihedral",
            "periodic",
            dihedral::periodic::dihedral_periodic_constructor,
        );
        r.register(
            "dihedral",
            "harmonic",
            dihedral::harmonic::dihedral_harmonic_constructor,
        );
        r.register(
            "dihedral",
            "class2",
            dihedral::class2::dihedral_class2_constructor,
        );
        r.register(
            "dihedral",
            "nharmonic",
            dihedral::multi_harmonic::dihedral_nharmonic_constructor,
        );
        // pair / nonbonded
        r.register("pair", "lj/cut", pair::lj_cut::pair_lj_cut_constructor);
        r.register(
            "pair",
            "lj/class2",
            pair::lj_class2::pair_lj_class2_constructor,
        );
        r.register("pair", "buck", pair::buck::pair_buck_constructor);
        r.register("pair", "morse", pair::morse::pair_morse_constructor);
        r.register("pair", "thole", pair::thole::pair_thole_constructor);
        r.register(
            "pair",
            "coul/tt",
            pair::tang_toennies::pair_tang_toennies_constructor,
        );
        // `coul/cut` reads per-atom `charge` off the Frame — there is no charge
        // type-row to read, and its ctor binds `_type_params`.
        //
        // It is the BUFFERED Coulomb, `E = k·qᵢqⱼ/(D·(r + δ))`, with k / D / δ all
        // read from the style. `δ = 0` is the textbook Coulomb (OPLS, LAMMPS);
        // `k = 332.0716, D = 1.0, δ = 0.05` is MMFF's electrostatics. MMFF owns no
        // electrostatic kernel of its own — it configures this one.
        r.register(
            "pair",
            "coul/cut",
            pair::coul_cut::pair_coul_cut_constructor,
        );
        // MMFF94 — five per-instance BONDED styles. Their kernels read the columns
        // the typifier bakes (`kb`/`r0`, `ka`/`theta0`, `kba_*`, `v1`/`v2`/`v3`,
        // `koop`), never a type row: MMFF's context rules (aromaticity, ring size,
        // equivalence degradation, empirical fallbacks) are not a
        // `(type_i, type_j, …) → params` table and cannot be made into one.
        r.register("bond", "mmff_bond", bond::mmff::bond_mmff_constructor);
        r.register("angle", "mmff_angle", angle::mmff::angle_mmff_constructor);
        r.register(
            "angle",
            "mmff_stbn",
            angle::mmff::angle_mmff_stretch_bend_constructor,
        );
        r.register(
            "dihedral",
            "mmff_torsion",
            dihedral::mmff::dihedral_mmff_constructor,
        );
        r.register(
            "improper",
            "mmff_oop",
            improper::mmff::improper_mmff_constructor,
        );
        // UFF — per-instance bonded + LJ (typifier bakes kb/r0, ka/order/c*, V/order)
        r.register("bond", "uff_bond", bond::uff::bond_uff_constructor);
        r.register("angle", "uff_angle", angle::uff::angle_uff_constructor);
        r.register(
            "dihedral",
            "uff_torsion",
            dihedral::uff::dihedral_uff_constructor,
        );
        r.register("pair", "uff_lj", pair::uff::pair_uff_vdw_constructor);
        r.register(
            "improper",
            "uff_inversion",
            improper::uff::improper_uff_constructor,
        );
        r.register(
            "improper",
            "harmonic",
            improper::harmonic::improper_harmonic_constructor,
        );
        r.register(
            "improper",
            "cvff",
            improper::cvff::improper_cvff_constructor,
        );
        r.register(
            "improper",
            "periodic",
            improper::periodic::improper_periodic_constructor,
        );
        // vdW is the one MMFF style that genuinely IS a per-atom-type table:
        // 95 types, 95 rows, and `pair_mmff_vdw_constructor` opens by indexing `tp`.
        r.register("pair", "mmff_vdw", pair::mmff::pair_mmff_vdw_constructor);
        r.register(
            "pair",
            "coul/long/pme",
            kspace::pme::pair_coul_long_pme_constructor,
        );
        // PME sums in reciprocal space over the atoms' charges and subtracts
        // `exclusions`; it reads no `pairs` block, so the pair category's gate
        // must not delete it when there is none.
        r.declare_rows("pair", "coul/long/pme", RowSource::Atoms);
        // The neighbour-driven counterparts. Same styles, parameters keyed on
        // the atoms rather than on a `pairs` block, which is what an evaluation
        // over a rebuilt neighbour table needs.
        r.register_typed(
            "pair",
            "lj/cut",
            pair::lj_cut::pair_lj_cut_typed_constructor,
            SpecialClass::Vdw,
        );
        r.register_typed(
            "pair",
            "lj/class2",
            pair::lj_class2::pair_lj_class2_typed_constructor,
            SpecialClass::Vdw,
        );
        r.register_typed(
            "pair",
            "buck",
            pair::buck::pair_buck_typed_constructor,
            SpecialClass::Vdw,
        );
        r.register_typed(
            "pair",
            "morse",
            pair::morse::pair_morse_typed_constructor,
            SpecialClass::Vdw,
        );
        r.register_typed(
            "pair",
            "uff_lj",
            pair::uff::pair_uff_vdw_typed_constructor,
            SpecialClass::Vdw,
        );
        r.register_typed(
            "pair",
            "mmff_vdw",
            pair::mmff::pair_mmff_vdw_typed_constructor,
            SpecialClass::Vdw,
        );
        r.register_typed(
            "pair",
            "coul/cut",
            pair::coul_cut::pair_coul_cut_typed_constructor,
            SpecialClass::Coulomb,
        );
        r.register_typed(
            "pair",
            "coul/tt",
            pair::tang_toennies::pair_tang_toennies_typed_constructor,
            SpecialClass::Coulomb,
        );
        r.register_typed(
            "pair",
            "thole",
            pair::thole::pair_thole_typed_constructor,
            SpecialClass::Coulomb,
        );
        // CHARMM angle + Urey–Bradley (LAMMPS `angle_style charmm`).
        r.register("angle", "charmm", angle::charmm::angle_charmm_constructor);
        // CMAP: a five-atom crossterm over the `cmaps` block (LAMMPS `fix cmap`).
        r.register("cmap", "charmm", cmap::charmm::cmap_charmm_constructor);
        // LAMMPS `lj/charmm/coul/charmm`, as its two halves. `coul/charmm`
        // reads per-atom charges, like `coul/cut`.
        r.register(
            "pair",
            "lj/charmm",
            pair::charmm::pair_lj_charmm_constructor,
        );
        r.register_typed(
            "pair",
            "lj/charmm",
            pair::charmm::pair_lj_charmm_typed_constructor,
            SpecialClass::Vdw,
        );
        r.register(
            "pair",
            "coul/charmm",
            pair::charmm::pair_coul_charmm_constructor,
        );
        r.register_typed(
            "pair",
            "coul/charmm",
            pair::charmm::pair_coul_charmm_typed_constructor,
            SpecialClass::Coulomb,
        );

        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_has_core_kernels() {
        let r = BuiltinKernels::builtin();
        assert!(r.get("bond", "harmonic").is_some());
        assert!(r.get("pair", "lj/cut").is_some());
        assert!(r.get("pair", "buck").is_some());
        assert!(r.get("pair", "coul/long/pme").is_some());
        assert!(r.get("kspace", "pme").is_none());
        assert!(r.get("bond", "does-not-exist").is_none());
    }

    #[test]
    fn register_overrides_and_adds() {
        let mut r = BuiltinKernels::default();
        assert!(r.constructors.is_empty());
        r.register("pair", "lj/cut", pair::lj_cut::pair_lj_cut_constructor);
        assert_eq!(r.constructors.len(), 1);
        assert!(r.get("pair", "lj/cut").is_some());
        // re-registering the same key overrides, not duplicates
        r.register("pair", "lj/cut", pair::buck::pair_buck_constructor);
        assert_eq!(r.constructors.len(), 1);
    }

    /// Every neighbour-driven form registered by [`BuiltinKernels::builtin`] is
    /// still there when `builtin` returns.
    ///
    /// Re-registering a style clears its typed entry on purpose — an override
    /// replaces the force law, and a stale neighbour-driven form would make
    /// `compile` and `compile_typed` evaluate different physics for
    /// the same name. That makes registration **order-sensitive**: a
    /// `register_typed` followed later by a `register` for the same key drops
    /// the typed form silently, and the only symptom is `compile_typed`
    /// reporting a style it was told about as unknown. This pins the order.
    #[test]
    fn every_typed_registration_survives_builtin() {
        let r = BuiltinKernels::builtin();
        for name in [
            "lj/cut",
            "lj/class2",
            "buck",
            "morse",
            "uff_lj",
            "mmff_vdw",
            "coul/cut",
            "coul/tt",
            "thole",
            "lj/charmm",
            "coul/charmm",
        ] {
            assert!(
                r.get_typed("pair", name).is_some(),
                "pair '{name}' lost its neighbour-driven form: a later \
                 register for the same key must come *before* \
                 its register_typed"
            );
        }
        // The same ordering trap, for the other per-style declaration.
        assert_eq!(
            r.row_source("pair", "coul/long/pme"),
            Some(RowSource::Atoms),
            "PME lost its row-source declaration: a later register \
             for the same key must come *before* its declare_rows"
        );
    }

    /// PME survives a frame with no `pairs` block.
    ///
    /// It is registered under `pair` because that is where an electrostatic
    /// style belongs, and `PotentialCompiler::compile` skips a pair style whose
    /// `pairs` block is absent or empty — a rule that is right for every other
    /// pair kernel and deleted PME outright. The symptom was a system with
    /// zero long-range electrostatics and no error: the style was declared,
    /// accepted, and dropped.
    #[test]
    fn pme_is_not_gated_on_a_pairs_block() {
        let r = BuiltinKernels::builtin();
        assert_eq!(
            r.row_source("pair", "coul/long/pme"),
            Some(RowSource::Atoms),
            "PME reads charges and exclusions, never `pairs`"
        );
        for gated in ["lj/cut", "coul/cut", "buck", "thole"] {
            assert_eq!(
                r.row_source("pair", gated),
                Some(RowSource::CategoryBlock),
                "'{gated}' does read `pairs`, so an empty one still means no work"
            );
        }
    }

    /// An override drops the neighbour-driven form rather than keeping a form
    /// built for the force law that was just replaced.
    #[test]
    fn re_registering_a_style_clears_its_typed_form() {
        let mut r = BuiltinKernels::default();
        r.register("pair", "lj/cut", pair::lj_cut::pair_lj_cut_constructor);
        r.register_typed(
            "pair",
            "lj/cut",
            pair::lj_cut::pair_lj_cut_typed_constructor,
            SpecialClass::Vdw,
        );
        assert!(r.get_typed("pair", "lj/cut").is_some());
        r.register("pair", "lj/cut", pair::buck::pair_buck_constructor);
        assert!(
            r.get_typed("pair", "lj/cut").is_none(),
            "the typed form was built for lj/cut's parameters, not buck's"
        );
    }
}
