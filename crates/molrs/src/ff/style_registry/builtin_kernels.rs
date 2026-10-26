//! molrs's own kernels: `(category, style_name)` → the [`Kernel`] that
//! prices it.
//!
//! A crate-private table, not a registry anyone extends: the force-field IR
//! registry ([`crate::ff::style_registry::Registry`], the one registry) seeds
//! its sealed built-ins from [`BuiltinKernels::builtin`], each beside its
//! spec, and `PotentialCompiler` resolves every style through that registry.
//! A third party registers through
//! [`GlobalRegistry::register_style`](crate::ff::style_registry::GlobalRegistry::register_style).
//!
//! Every built-in whose energy is a function of one coordinate or of its
//! atoms' positions is a **form** — [`Kernel::Scalar`] or
//! [`Kernel::Compound`], exactly the tiers a third party registers — and the
//! form kernels ([`crate::ff::potential::form_kernel`]) do everything else:
//! the column fetch, the type lookup, the pair table and mixing, the 1-4
//! weight, the cutoff, the neighbour-driven door and the virial. Each is
//! registered as its `dyn` form and built monomorphized on its type
//! ([`Native`]): the form kernel inlines its force law, and a pair style
//! evaluates through its [`pair_law`](crate::ff::potential::form_kernel::ScalarForm::pair_law)
//! in one fused pass, as a hand-written kernel would. Two
//! built-ins stay whole-kernel constructors ([`Kernel::Constructor`]):
//!
//! * `pair coul/long/pme` — the reciprocal-space sum, the self energy and
//!   the exclusion correction. The real-space erfc term is the pair form
//!   [`CoulLong`](crate::ff::potential::pair::CoulLong) beside that sum. A
//!   compiled frame with no `pairs` block still prices the real-space term
//!   inside the constructor, minimum-imaged, because there is no list to
//!   hand the form. The neighbour-driven door always prices it over the
//!   neighbour table.
//! * `cmap charmm` — its parameter is an energy grid whose bicubic-spline
//!   coefficients are derived once per type at build; a compound form is
//!   handed the raw grid at every evaluation and would re-derive them each
//!   time.
//!
//! Where a kernel's numbers come from ([`ParamSource`](crate::ff::ir::ParamSource))
//! is its spec's declaration ([`StyleSpec::source`](crate::ff::ir::StyleSpec::source)),
//! not this table's: MMFF's and UFF's bonded styles and `uff_lj` read
//! per-instance Frame columns the typifier bakes, `coul/cut`, `coul/charmm`
//! and `coul/long/pme` per-atom charges, and their specs say
//! [`ParamSource::PerInstance`](crate::ff::ir::ParamSource::PerInstance).

use std::collections::HashMap;
use std::sync::Arc;

use crate::ff::ir::SpecialClass;
use crate::ff::potential::form_kernel::{CompoundForm, ScalarForm};
use crate::ff::potential::{angle, bond, cmap, dihedral, improper, kspace, pair};
use crate::ff::style_registry::{Kernel, Native, RowSource};

/// Maps `(category, style_name)` to the kernel that prices it.
#[derive(Default)]
pub(crate) struct BuiltinKernels {
    kernels: HashMap<(String, String), (Kernel, Option<Native>)>,
}

impl BuiltinKernels {
    /// Every registered `(category, style)`, sorted.
    #[cfg(test)]
    pub(crate) fn styles(&self) -> Vec<(&str, &str)> {
        let mut out: Vec<(&str, &str)> = self
            .kernels
            .keys()
            .map(|(c, s)| (c.as_str(), s.as_str()))
            .collect();
        out.sort_unstable();
        out
    }

    fn insert(&mut self, category: &str, name: &str, kernel: Kernel, native: Option<Native>) {
        let previous = self
            .kernels
            .insert((category.to_owned(), name.to_owned()), (kernel, native));
        assert!(
            previous.is_none(),
            "built-in {category} `{name}` registered twice"
        );
    }

    /// `(category, name)` priced by the scalar form `Fm`: registered as its
    /// `dyn` form, built monomorphized on its type.
    fn scalar<Fm: ScalarForm + Default>(&mut self, category: &str, name: &str, form: Fm) {
        self.insert(
            category,
            name,
            Kernel::Scalar(Arc::new(form)),
            Some(Native::scalar::<Fm>()),
        );
    }

    /// `(category, name)` priced by the compound form `Fm`.
    fn compound<Fm: CompoundForm + Default>(&mut self, category: &str, name: &str, form: Fm) {
        self.insert(
            category,
            name,
            Kernel::Compound(Arc::new(form)),
            Some(Native::compound::<Fm>()),
        );
    }

    /// The kernel of `(category, name)`, if it is a built-in.
    pub(crate) fn kernel(&self, category: &str, name: &str) -> Option<Kernel> {
        self.kernels
            .get(&(category.to_owned(), name.to_owned()))
            .map(|(kernel, _)| kernel.clone())
    }

    /// The monomorphized build of the built-in form `(category, name)`.
    pub(crate) fn native(&self, category: &str, name: &str) -> Option<Native> {
        self.kernels
            .get(&(category.to_owned(), name.to_owned()))
            .and_then(|(_, native)| *native)
    }

    /// Every built-in kernel.
    pub(crate) fn builtin() -> Self {
        let mut r = Self::default();
        // bond
        r.scalar("bond", "harmonic", bond::Harmonic);
        r.scalar("bond", "class2", bond::Class2);
        r.scalar("bond", "morse", bond::Morse);
        r.scalar("bond", "mmff_bond", bond::Mmff);
        r.scalar("bond", "uff_bond", bond::Uff);
        // angle
        r.scalar("angle", "harmonic", angle::Harmonic);
        r.scalar("angle", "class2", angle::Class2);
        // CHARMM angle + Urey–Bradley (LAMMPS `angle_style charmm`): the 1-3
        // distance makes it a function of the three positions.
        r.compound("angle", "charmm", angle::Charmm);
        r.scalar("angle", "mmff_angle", angle::Mmff);
        r.compound("angle", "mmff_stbn", angle::MmffStretchBend);
        r.scalar("angle", "uff_angle", angle::Uff);
        // dihedral
        r.scalar("dihedral", "opls", dihedral::Opls);
        r.scalar("dihedral", "charmm", dihedral::Charmm);
        r.scalar("dihedral", "multi/harmonic", dihedral::MultiHarmonic);
        r.scalar("dihedral", "nharmonic", dihedral::MultiHarmonic);
        r.scalar("dihedral", "periodic", dihedral::Periodic);
        // LAMMPS `dihedral_style harmonic`, `K[1 + d cos(nφ)]`: the function
        // of the dihedral `improper cvff` is.
        r.scalar("dihedral", "harmonic", improper::Cvff);
        r.scalar("dihedral", "class2", dihedral::Class2);
        r.scalar("dihedral", "mmff_torsion", dihedral::Mmff);
        r.scalar("dihedral", "uff_torsion", dihedral::Uff);
        // improper
        r.scalar("improper", "harmonic", improper::Harmonic);
        r.scalar("improper", "cvff", improper::Cvff);
        r.scalar("improper", "periodic", improper::Periodic);
        r.compound("improper", "mmff_oop", improper::Mmff);
        r.compound("improper", "uff_inversion", improper::Uff);
        // pair. `coul/cut` is the BUFFERED Coulomb, `E = k·qᵢqⱼ/(D·(r + δ))`:
        // `δ = 0` is the textbook Coulomb (OPLS, LAMMPS), `k = 332.0716, D =
        // 1.0, δ = 0.05` MMFF's electrostatics, which owns no form of its own.
        r.scalar("pair", "lj/cut", pair::LjCut);
        r.scalar("pair", "lj/class2", pair::LjClass2);
        r.scalar("pair", "buck", pair::Buck);
        r.scalar("pair", "morse", pair::Morse);
        r.scalar("pair", "coul/cut", pair::CoulCut);
        r.scalar("pair", "lj/charmm", pair::LjCharmm);
        r.scalar("pair", "coul/charmm", pair::CoulCharmm);
        r.scalar("pair", "thole", pair::Thole);
        r.scalar("pair", "coul/tt", pair::TangToennies);
        r.scalar("pair", "uff_lj", pair::UffLj);
        r.scalar("pair", "mmff_vdw", pair::MmffVdw);
        // Reciprocal sum, self energy and exclusion correction. The
        // real-space erfc term is `CoulLong`. With no `pairs` block the
        // constructor still prices real space (minimum image), so the pair
        // gate must not delete the style. The neighbour door is that pair
        // form plus the reciprocal sum.
        r.insert(
            "pair",
            "coul/long/pme",
            Kernel::Constructor {
                compiled: kspace::pme::pair_coul_long_pme_constructor,
                typed: Some((kspace::pme::pair_coul_long_pme_typed, SpecialClass::Coulomb)),
                rows: RowSource::Atoms,
            },
            None,
        );
        // CMAP: a five-atom crossterm over the `cmaps` block (LAMMPS `fix cmap`).
        r.insert(
            "cmap",
            "charmm",
            Kernel::constructor(cmap::charmm::cmap_charmm_constructor),
            None,
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
        assert!(matches!(
            r.kernel("bond", "harmonic"),
            Some(Kernel::Scalar(_))
        ));
        assert!(matches!(
            r.kernel("pair", "lj/cut"),
            Some(Kernel::Scalar(_))
        ));
        assert!(matches!(
            r.kernel("angle", "charmm"),
            Some(Kernel::Compound(_))
        ));
        assert!(matches!(
            r.kernel("pair", "coul/long/pme"),
            Some(Kernel::Constructor { .. })
        ));
        assert!(r.kernel("kspace", "pme").is_none());
        assert!(r.kernel("bond", "does-not-exist").is_none());
    }

    /// Every built-in pair style but PME is a scalar form with both doors.
    /// PME's real-space term is the pair form `CoulLong`, tested with the
    /// reciprocal sum in `kspace::pme`; the style stays a constructor
    /// because the reciprocal sum is not a function of one distance.
    #[test]
    fn every_pair_form_has_both_doors() {
        let r = BuiltinKernels::builtin();
        for (category, name) in r.styles() {
            if category != "pair" || name == "coul/long/pme" {
                continue;
            }
            assert!(
                matches!(r.kernel(category, name), Some(Kernel::Scalar(_))),
                "pair `{name}` is no scalar form"
            );
        }
    }

    /// PME survives a frame with no `pairs` block.
    ///
    /// It is registered under `pair` because that is where an electrostatic
    /// style belongs, and `PotentialCompiler::compile` skips a pair style whose
    /// `pairs` block is absent or empty — a rule that is right for every other
    /// pair kernel and deleted PME outright. The symptom was a system with
    /// zero long-range electrostatics and no error: the style was declared,
    /// accepted, and dropped. The row source stays the atoms. A present
    /// `pairs` block prices the real-space term as the pair form; it does
    /// not decide whether the style is built.
    #[test]
    fn pme_is_not_gated_on_a_pairs_block() {
        let r = crate::ff::style_registry::Registry::builtin();
        assert_eq!(
            r.row_source("pair", "coul/long/pme"),
            Some(RowSource::Atoms),
            "an absent pairs block still builds PME"
        );
        for gated in ["lj/cut", "coul/cut", "buck", "thole"] {
            assert_eq!(
                r.row_source("pair", gated),
                Some(RowSource::CategoryBlock),
                "'{gated}' does read `pairs`, so an empty one still means no work"
            );
        }
    }
}
