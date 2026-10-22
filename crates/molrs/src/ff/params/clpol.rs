//! CL&Pol parameters: the scaleLJ fragment table and the per-atom-type
//! Drude polarisation table.

/// `(name, q, mu, alpha, polarizable)` rows used by CL&Pol scaleLJ.
///
/// Source: paduagroup/clandpol (Goloviznina, Canongia Lopes, Costa Gomes &
/// Pádua, *J. Chem. Theory Comput.* **15** (2019) 5858; *PCCP* **20** (2018)
/// 10992) — `fragment.ff` (`q`, `mu`) and `alpha.ff` (per-atom
/// polarisabilities, also summed per fragment for [`CLPOL_FRAGMENTS`]).
pub const CLPOL_FRAGMENTS: &[(&str, f64, f64, f64, bool)] = &[
    ("c2c1im", 1.0, 1.1558, 12.383, false),
    ("bf4", -1.0, 0.0, 3.078, false),
    ("pf6", -1.0, 0.0, 4.987, false),
    ("ntf2", -1.0, 4.0070, 15.162, false),
    ("dca", -1.0, 0.8874, 8.268, false),
];

/// One `alpha.ff` row: the Drude oscillator CL&Pol attaches to an atom type.
///
/// Units as in the file: `m_d` in u, `q_d_sign` the sign of the Drude charge
/// (its magnitude follows from `k_d` and `alpha`), `k_d` in kJ·mol⁻¹·Å⁻² for
/// a spring energy written `k_d/2 · r_D²` (`alpha.ff`'s "k/2 r_D²" form),
/// `alpha` in Å³, `a_thole` dimensionless. A type with `k_d == 0` carries no
/// Drude particle (its polarisability is folded into its bonded heavy atom).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClpolPolarizability {
    /// CL&P atom type the row applies to.
    pub type_name: &'static str,
    /// Drude particle mass, u.
    pub m_d: f64,
    /// Sign of the Drude charge (`-1.0`, `1.0`, or `0.0` when unpolarised).
    pub q_d_sign: f64,
    /// Drude spring constant in the `k/2 r²` convention, kJ·mol⁻¹·Å⁻².
    pub k_d: f64,
    /// Atomic polarisability, Å³.
    pub alpha: f64,
    /// Thole damping parameter (dimensionless).
    pub a_thole: f64,
}

impl ClpolPolarizability {
    /// Whether the type carries a Drude particle (`k_d > 0`).
    pub fn is_polarizable(&self) -> bool {
        self.k_d > 0.0
    }
}

/// `alpha.ff` (version 2024/06/05), one row per atom type, in file order.
///
/// [`CLPOL_POLARIZABILITY`] is `alpha.ff` version 2024/06/05 transcribed row
/// for row; a caller's own `alpha.ff` is read by
/// [`crate::io::read_clpol_alpha`].
#[rustfmt::skip]
pub const CLPOL_POLARIZABILITY: &[ClpolPolarizability] = &[
    // metal cations CHARMM polarizable
    ClpolPolarizability { type_name: "Li", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 0.032, a_thole: 2.6 },
    ClpolPolarizability { type_name: "Na", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 0.157, a_thole: 2.6 },
    // alkyl groups
    ClpolPolarizability { type_name: "CS", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CT2", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CT3", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "C3H", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CT", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "HC", m_d: 0.0, q_d_sign: 0.0, k_d: 0.0, alpha: 0.323, a_thole: 0.0 },
    // aromatic
    ClpolPolarizability { type_name: "CA", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.122, a_thole: 2.6 },
    ClpolPolarizability { type_name: "HA", m_d: 0.0, q_d_sign: 0.0, k_d: 0.0, alpha: 0.323, a_thole: 0.0 },
    ClpolPolarizability { type_name: "CAO", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.122, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CAM", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.122, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CAP", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.122, a_thole: 2.6 },
    // alcohols
    ClpolPolarizability { type_name: "OH", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.144, a_thole: 2.6 },
    ClpolPolarizability { type_name: "HO", m_d: 0.0, q_d_sign: 0.0, k_d: 0.0, alpha: 0.323, a_thole: 0.0 },
    ClpolPolarizability { type_name: "CTO", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    // ethers
    ClpolPolarizability { type_name: "OS", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.144, a_thole: 2.6 },
    ClpolPolarizability { type_name: "HCO", m_d: 0.0, q_d_sign: 0.0, k_d: 0.0, alpha: 0.323, a_thole: 0.0 },
    ClpolPolarizability { type_name: "C2P", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "C3P", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "COM", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "OY", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.144, a_thole: 2.6 },
    ClpolPolarizability { type_name: "C3O", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    // imidazolium PCCP 20(2018)10992
    ClpolPolarizability { type_name: "CR", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.122, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CW", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.122, a_thole: 2.6 },
    ClpolPolarizability { type_name: "NA", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.208, a_thole: 2.6 },
    ClpolPolarizability { type_name: "C1", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "C1A", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "C2", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CE", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "H1", m_d: 0.0, q_d_sign: 0.0, k_d: 0.0, alpha: 0.323, a_thole: 0.0 },
    ClpolPolarizability { type_name: "HCR", m_d: 0.0, q_d_sign: 0.0, k_d: 0.0, alpha: 0.323, a_thole: 0.0 },
    ClpolPolarizability { type_name: "HCW", m_d: 0.0, q_d_sign: 0.0, k_d: 0.0, alpha: 0.323, a_thole: 0.0 },
    // pyridinium PCCP 20(2018)10992
    ClpolPolarizability { type_name: "NAP", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.208, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CAPO", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.122, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CAPM", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.122, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CAPP", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.122, a_thole: 2.6 },
    ClpolPolarizability { type_name: "HAP", m_d: 0.0, q_d_sign: 0.0, k_d: 0.0, alpha: 0.323, a_thole: 0.0 },
    // ammonium, pyrrolidinium PCCP 20(2018)10992
    ClpolPolarizability { type_name: "N4", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.208, a_thole: 2.6 },
    ClpolPolarizability { type_name: "N3", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.208, a_thole: 2.6 },
    ClpolPolarizability { type_name: "H3", m_d: 0.0, q_d_sign: 0.0, k_d: 0.0, alpha: 0.323, a_thole: 0.0 },
    // phosphonium PCCP 20(2018)10992
    ClpolPolarizability { type_name: "P3", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.237, a_thole: 2.6 },
    ClpolPolarizability { type_name: "C1P", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "C2I", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    // isoalkylphosphonium
    ClpolPolarizability { type_name: "CPI4", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CI4", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "HI4", m_d: 0.0, q_d_sign: 0.0, k_d: 0.0, alpha: 0.323, a_thole: 0.0 },
    // bistriflamide PCCP 20(2018)10992
    ClpolPolarizability { type_name: "CBT", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "NBT", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.698, a_thole: 2.6 },
    ClpolPolarizability { type_name: "OBT", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.144, a_thole: 2.6 },
    ClpolPolarizability { type_name: "SBT", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.553, a_thole: 2.6 },
    ClpolPolarizability { type_name: "F1", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 0.625, a_thole: 2.6 },
    ClpolPolarizability { type_name: "FSI", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 0.625, a_thole: 2.6 },
    // dicyanamide PCCP 20(2018)10992
    ClpolPolarizability { type_name: "N3A", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.698, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CZA", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.587, a_thole: 2.6 },
    ClpolPolarizability { type_name: "NZA", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.698, a_thole: 2.6 },
    // tetrafluoroborate PCCP 20(2018)10992
    ClpolPolarizability { type_name: "B", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 0.578, a_thole: 2.6 },
    ClpolPolarizability { type_name: "FBF", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 0.625, a_thole: 2.6 },
    // triflate PCCP 20(2018)10992
    ClpolPolarizability { type_name: "OTF", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.144, a_thole: 2.6 },
    // acetate PCCP 20(2018)10992
    ClpolPolarizability { type_name: "O2", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.144, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CO2", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.432, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CTA", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    // isovalerate
    ClpolPolarizability { type_name: "C2V", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "C3V", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CTV", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    // cyclohexanoate
    ClpolPolarizability { type_name: "C3C", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    // phenylacetate
    ClpolPolarizability { type_name: "C2C", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CPH", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.122, a_thole: 2.6 },
    // p-methylbenzoate
    ClpolPolarizability { type_name: "CBI", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.122, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CBP", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.122, a_thole: 2.6 },
    // hexafluorophosphate PCCP 20(2018)10992
    ClpolPolarizability { type_name: "P", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.237, a_thole: 2.6 },
    ClpolPolarizability { type_name: "FP", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 0.625, a_thole: 2.6 },
    // trifluorocetate PCCP 20(2018)10992
    ClpolPolarizability { type_name: "O2F", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.144, a_thole: 2.6 },
    ClpolPolarizability { type_name: "CFA", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 1.016, a_thole: 2.6 },
    ClpolPolarizability { type_name: "FFA", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 0.625, a_thole: 2.6 },
    // halides Chem. Phys 149(2018) 044302
    ClpolPolarizability { type_name: "Cl", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 4.4, a_thole: 2.6 },
    ClpolPolarizability { type_name: "Br", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 5.8, a_thole: 2.6 },
    ClpolPolarizability { type_name: "I", m_d: 0.4, q_d_sign: -1.0, k_d: 4184.0, alpha: 8.81, a_thole: 2.6 },
];

/// The [`CLPOL_POLARIZABILITY`] row of `type_name`, if it has one.
pub fn clpol_polarizability(type_name: &str) -> Option<&'static ClpolPolarizability> {
    CLPOL_POLARIZABILITY
        .iter()
        .find(|row| row.type_name == type_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_holds_every_alpha_ff_row_once() {
        assert_eq!(CLPOL_POLARIZABILITY.len(), 78);
        let mut names: Vec<&str> = CLPOL_POLARIZABILITY.iter().map(|r| r.type_name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), CLPOL_POLARIZABILITY.len());
    }

    #[test]
    fn rows_read_as_alpha_ff_states_them() {
        let nbt = clpol_polarizability("NBT").unwrap();
        assert_eq!(
            (nbt.m_d, nbt.q_d_sign, nbt.k_d, nbt.alpha, nbt.a_thole),
            (0.4, -1.0, 4184.0, 1.698, 2.6)
        );
        assert!(nbt.is_polarizable());
        let hc = clpol_polarizability("HC").unwrap();
        assert!(!hc.is_polarizable());
        assert_eq!(hc.alpha, 0.323);
        assert_eq!(clpol_polarizability("I").unwrap().alpha, 8.81);
        assert!(clpol_polarizability("XX").is_none());
    }
}
