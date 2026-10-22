//! [`SpecialBonds`]: LAMMPS `special_bonds`, the 1-2 / 1-3 / 1-4 weights of
//! the non-bonded kernels.

use molrs::core::BondDistanceWeights;

/// Per-nonbonded-kind 1-2 / 1-3 / 1-4 interaction scale weights — LAMMPS
/// `special_bonds` semantics, owned by the [`ForceField`](crate::ff::forcefield::ForceField).
///
/// The always-on geometric table is [`crate::core::BondDistanceWeights`]: one
/// arbitrary-length vector with an explicit 1-N tail. A LAMMPS triple is not
/// a transcription (`charmm 0 0 0` is `[0, 0, 0, 1]` there). There is no
/// `From` / `Into` between the two types.
///
/// A weight of `0.0` fully excludes that neighbour class; `1.0` leaves it at
/// full strength.
///
/// # Two doors, two expressive powers
///
/// A **compiled** pair list (`intramolecular_pairs` → `PotentialCompiler::compile`) carries
/// the 1-2 / 1-3 weights by *presence*: the row is there or it is not. That is
/// one bit, so it expresses `0.0` and `1.0` and nothing between, and it
/// expresses only weights the van-der-Waals and Coulomb kernels **share** —
/// one list feeds both. [`compiled_inclusion`](Self::compiled_inclusion) is
/// that judgement, and both doors on that path call it rather than assume.
///
/// A **neighbour-driven** evaluation (`PotentialCompiler::compile_typed`) carries them as a
/// per-pair factor ([`lj_weights`](Self::lj_weights) /
/// [`coul_weights`](Self::coul_weights)), so it expresses every weight, and
/// the two kernels independently.
///
/// The 1-4 weight `[2]` is not part of this: both doors scale it inside the
/// kernel, so a fraction is fine there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecialBonds {
    /// LJ / van-der-Waals `[1-2, 1-3, 1-4]` scale weights.
    pub lj: [f64; 3],
    /// Coulomb `[1-2, 1-3, 1-4]` scale weights.
    pub coul: [f64; 3],
}

impl Default for SpecialBonds {
    /// Exclude 1-2 and 1-3 neighbours; leave 1-4 unscaled. Force-field readers
    /// override the 1-4 weights (Amber: lj `0.5`, coul `0.8333`).
    fn default() -> Self {
        DEFAULT_SPECIAL_BONDS
    }
}

impl SpecialBonds {
    /// The LJ 1-4 scale weight (the `[2]` entry of [`Self::lj`]).
    pub fn lj_14(&self) -> f64 {
        self.lj[2]
    }

    /// The Coulomb 1-4 scale weight (the `[2]` entry of [`Self::coul`]).
    pub fn coul_14(&self) -> f64 {
        self.coul[2]
    }

    /// The LJ weights as a bond-distance table, full strength past 1-4.
    ///
    /// What a neighbour-driven evaluation needs. A compiled intramolecular
    /// list carried these by *omitting* the excluded rows and baking the 1-4
    /// factor into the parameters, so only `[2]` was ever read; a neighbour
    /// table finds every pair inside the cutoff and needs all three.
    pub fn lj_weights(&self) -> BondDistanceWeights {
        Self::table(self.lj)
    }

    /// The Coulomb weights as a bond-distance table, full strength past 1-4.
    ///
    /// Separate from [`lj_weights`](Self::lj_weights) because a force field may
    /// scale the two differently — Amber uses `1/2` for van der Waals and
    /// `1/1.2` for electrostatics — and in molrs they are separate kernels.
    pub fn coul_weights(&self) -> BondDistanceWeights {
        Self::table(self.coul)
    }

    /// Whether a compiled `pairs` list can carry the 1-2 and 1-3 weights, and
    /// if so whether each class belongs *in* the list.
    ///
    /// `Ok([keep_12, keep_13])` — `false` means omit those rows (the class is
    /// excluded), `true` means emit them unflagged (full strength). `Err` means
    /// the weights are outside what a presence/absence list can say, and the
    /// caller must use the neighbour-driven door instead of quietly rounding.
    ///
    /// Two ways to fall outside:
    ///
    /// * a **fraction** — `lj[1] == 0.5` scales 1-3 pairs to half strength, and
    ///   a row that is merely present cannot say "half";
    /// * a **split** — `lj[1] == 1.0` with `coul[1] == 0.0` wants the row for
    ///   one kernel and not for the other, and there is one list for both.
    ///
    /// LAMMPS's own presets exercise both the accepted values: `amber`,
    /// `charmm` and `dreiding` exclude 1-3 (`false`), `fene` keeps it
    /// (`[0, 1, 1]` → `true`).
    pub fn compiled_inclusion(&self) -> Result<[bool; 2], String> {
        let mut keep = [false; 2];
        for (k, slot) in keep.iter_mut().enumerate() {
            let class = if k == 0 { "1-2" } else { "1-3" };
            let (lj, coul) = (self.lj[k], self.coul[k]);
            if lj != coul {
                return Err(format!(
                    "special_bonds {class}: lj {lj} and coul {coul} differ, and a \
                     compiled pairs list is shared by both kernels — it can include \
                     the row or omit it, not do one for van der Waals and the other \
                     for Coulomb. Use PotentialCompiler::compile_typed, which carries \
                     a per-pair weight per kernel."
                ));
            }
            *slot = if lj == 0.0 {
                false
            } else if lj == 1.0 {
                true
            } else {
                return Err(format!(
                    "special_bonds {class} weight {lj}: a compiled pairs list carries \
                     this class by whether the row is present, so it expresses 0 or 1 \
                     and nothing between. Use PotentialCompiler::compile_typed, which \
                     carries a per-pair weight."
                ));
            };
        }
        Ok(keep)
    }

    fn table(w: [f64; 3]) -> BondDistanceWeights {
        BondDistanceWeights::new(vec![w[0], w[1], w[2], 1.0])
            .expect("a four-entry weight table is always well formed")
    }
}

/// The weights of a force field that declares none ([`SpecialBonds::default`]).
pub(crate) const DEFAULT_SPECIAL_BONDS: SpecialBonds = SpecialBonds {
    lj: [0.0, 0.0, 1.0],
    coul: [0.0, 0.0, 1.0],
};
