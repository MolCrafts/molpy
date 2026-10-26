//! How a force field says per-atom-type Lennard-Jones parameters combine.
//!
//! A combining rule is a **force-field** declaration, not a kernel constant —
//! it is what `<ForceField combining_rule>` and LAMMPS's `pair_modify mix` set,
//! and reading an OPLS pack under Lorentz-Berthelot silently shifts every σ.
//! It is IR vocabulary: the force field declares it and the pair kernels read
//! it, so it lives beneath both.

use molrs::op::F;

/// How per-atom-type ε/σ combine into a pair's ε/σ.
///
/// The style-level **string** param `mixing` selects it. LAMMPS spells the same
/// choice `pair_modify mix <rule>`; OpenMM / foyer XML spells it
/// `combining_rule` on `<ForceField>`. It is a force-field property, not a
/// kernel constant: OPLS-AA is `geometric` on both ε and σ, AMBER / GAFF and
/// CHARMM are `arithmetic` (Lorentz-Berthelot), COMPASS / class2 is
/// `sixthpower`. Reading an OPLS pack with Lorentz-Berthelot silently shifts
/// every σ — hence the explicit knob.
/// The combining rules' canonical spellings (LAMMPS's `pair_modify mix`
/// names), in [`CombiningRule`] variant order: what a style's `mixing` param, and a
/// record's `params.mixing`, may name.
pub const COMBINING_RULES: [&str; 3] = ["arithmetic", "geometric", "sixthpower"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombiningRule {
    /// `ε = √(εᵢεⱼ)`, `σ = ½(σᵢ + σⱼ)` — Lorentz-Berthelot (AMBER, CHARMM).
    Arithmetic,
    /// `ε = √(εᵢεⱼ)`, `σ = √(σᵢσⱼ)` — OPLS-AA.
    Geometric,
    /// `σ⁶ = ½(σᵢ⁶ + σⱼ⁶)`, `ε = 2√(εᵢεⱼ)·σᵢ³σⱼ³/(σᵢ⁶ + σⱼ⁶)` — COMPASS / class2.
    SixthPower,
}

impl CombiningRule {
    /// The rule an `lj/cut` style that declares no `mixing` is evaluated
    /// under: Lorentz-Berthelot, the AMBER-family rule. Every reader whose
    /// format has a rule states it (AMBER prmtop: `arithmetic`).
    pub(crate) const UNDECLARED: CombiningRule = CombiningRule::Arithmetic;

    /// The canonical spelling, the one [`CombiningRule::parse`] maps back to `self`.
    pub(crate) fn name(self) -> &'static str {
        COMBINING_RULES[self as usize]
    }

    /// Parse the canonical spelling (LAMMPS's `pair_modify mix` names); a
    /// reader translates its own synonyms (foyer's `lorentz`) at its door.
    pub fn parse(name: &str) -> Result<Self, String> {
        const RULES: [CombiningRule; 3] = [
            CombiningRule::Arithmetic,
            CombiningRule::Geometric,
            CombiningRule::SixthPower,
        ];
        COMBINING_RULES
            .iter()
            .position(|rule| *rule == name)
            .map(|i| RULES[i])
            .ok_or_else(|| {
                format!(
                    "unknown mixing rule '{name}' (expected {})",
                    COMBINING_RULES.join(" | ")
                )
            })
    }

    /// Combine one pair's per-type `(ε, σ)` into the pair's `(ε, σ)`.
    pub fn combine(self, (eps_i, sig_i): (F, F), (eps_j, sig_j): (F, F)) -> (F, F) {
        match self {
            Self::Arithmetic => ((eps_i * eps_j).sqrt(), 0.5 * (sig_i + sig_j)),
            Self::Geometric => ((eps_i * eps_j).sqrt(), (sig_i * sig_j).sqrt()),
            Self::SixthPower => {
                let (si3, sj3) = (sig_i.powi(3), sig_j.powi(3));
                let (si6, sj6) = (si3 * si3, sj3 * sj3);
                let denom = si6 + sj6;
                if denom <= 0.0 {
                    return (0.0, 0.0);
                }
                let eps = 2.0 * (eps_i * eps_j).sqrt() * si3 * sj3 / denom;
                (eps, (0.5 * denom).powf(1.0 / 6.0))
            }
        }
    }
}

/// Relative tolerance under which two Lennard-Jones `(ε, σ)` pairs are one.
const SAME_LJ: F = 1e-12;

/// Whether two Lennard-Jones `(ε, σ)` pairs are the same pair, to a relative
/// [`SAME_LJ`] on each of ε and σ — the test that decides whether an explicit
/// pair row says anything the [`CombiningRule`] (or the regular pair, for a
/// 1-4 row) does not already say.
pub(crate) fn same_lj(a: (F, F), b: (F, F)) -> bool {
    let close = |x: F, y: F| (x - y).abs() <= SAME_LJ * x.abs().max(y.abs());
    close(a.0, b.0) && close(a.1, b.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [CombiningRule; 3] = [
        CombiningRule::Arithmetic,
        CombiningRule::Geometric,
        CombiningRule::SixthPower,
    ];

    #[test]
    fn name_is_the_canonical_spelling_of_every_rule() {
        assert_eq!(CombiningRule::Arithmetic.name(), "arithmetic");
        assert_eq!(CombiningRule::Geometric.name(), "geometric");
        assert_eq!(CombiningRule::SixthPower.name(), "sixthpower");
    }

    #[test]
    fn parse_inverts_name() {
        for m in ALL {
            assert_eq!(CombiningRule::parse(m.name()), Ok(m), "{m:?}");
        }
    }

    /// An `lj/cut` style that declares no rule is evaluated Lorentz-Berthelot.
    #[test]
    fn undeclared_rule_is_arithmetic() {
        assert_eq!(CombiningRule::UNDECLARED, CombiningRule::Arithmetic);
    }

    /// The refusal names the rule and lists the choices without a run of
    /// stray spaces from a line continuation.
    #[test]
    fn unknown_rule_message_names_the_rule_without_stray_spaces() {
        let err = CombiningRule::parse("bogus").expect_err("unknown rule");
        assert!(err.contains("'bogus'"), "{err}");
        assert!(err.contains("arithmetic | geometric | sixthpower"), "{err}");
        assert!(!err.contains("  "), "run of spaces in: {err:?}");
    }

    /// LAMMPS's `sixthpower` (Waldman–Hagler): σᵢⱼ = ((σᵢ⁶ + σⱼ⁶)/2)^⅙,
    /// εᵢⱼ = 2√(εᵢεⱼ) σᵢ³σⱼ³/(σᵢ⁶ + σⱼ⁶) (`pair.cpp`, `mix_energy`/`mix_distance`).
    #[test]
    fn sixthpower_is_the_waldman_hagler_rule() {
        let (eps, sigma) = CombiningRule::SixthPower.combine((0.2, 3.0), (0.05, 4.0));
        let want_sigma = ((3.0f64.powi(6) + 4.0f64.powi(6)) / 2.0).powf(1.0 / 6.0);
        let want_eps =
            2.0 * (0.2f64 * 0.05).sqrt() * 27.0 * 64.0 / (3.0f64.powi(6) + 4.0f64.powi(6));
        assert!((sigma - want_sigma).abs() < 1e-14);
        assert!((eps - want_eps).abs() < 1e-15);
        // A type with itself is itself.
        let (e, s) = CombiningRule::SixthPower.combine((0.2, 3.0), (0.2, 3.0));
        assert!((e - 0.2).abs() < 1e-15 && (s - 3.0).abs() < 1e-14);
    }
}
