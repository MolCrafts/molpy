//! Virtual-site augmentation: Drude shells (CL&Pol) and TIP4P M-sites.
//!
//! Each builder returns a new molecule and never edits its input. Sites are
//! written in molrs's vocabulary: a site is an atom whose
//! [`keys::VSITE`] field names its kind (it carries
//! no `element`, which is IUPAC-only), a Drude shell is tied to its core by a
//! `drudes` row (`drude/harmonic`: `k`, `alpha`, `thole`) and a constructed
//! site to its frame atoms by a `virtual_sites` row (`virtual_site/average3`:
//! `w1`, `w2`, `w3`).

use std::collections::HashMap;

use molrs::core::keys;
use molrs::core::{Atomistic, MolRsError, NeighborKinds, NodeId, SimBox, UnitPreset, UnitRegistry};
use molrs::op::F;

use crate::props;

/// TIP4P/2005's O–M distance, Å.
pub const TIP4P_OM: F = 0.1546;

/// One CL&Pol `alpha.ff` row, in its own units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Polarizability {
    /// Drude particle mass, u.
    pub m_d: F,
    /// Sign of the Drude charge.
    pub q_d_sign: F,
    /// Drude spring constant in the `½ k r²` form, kJ·mol⁻¹·Å⁻².
    pub k_d: F,
    /// Polarizability, Å³.
    pub alpha: F,
    /// Thole damping parameter.
    pub a_thole: F,
}

/// The CL&Pol polarizer: a Drude shell on every atom whose type has `k_D > 0`
/// in its table.
///
/// The shell takes charge `q_D = ±√(4πε₀ · k_D · α)` and mass `m_D`, sits on
/// its core, and is typed `<prefix><core type>`; the core's charge and mass
/// lose the shell's, so the atom's net charge is conserved. The `drudes` row
/// (`atomi` the core, `atomj` the shell, `type` `"<core>-<shell>"`, style
/// `harmonic`) carries `k = k_D / 2` in kcal·mol⁻¹·Å⁻² — molrs's `k r²` form
/// against `alpha.ff`'s `½ k_D r²` in kJ — `alpha` and `thole = a_thole`.
#[derive(Clone, Debug)]
pub struct DrudeBuilder {
    table: HashMap<String, Polarizability>,
    prefix: String,
}

impl DrudeBuilder {
    /// A polarizer over `table` (type name → row), shells typed with the
    /// prefix `D`.
    pub fn new(table: HashMap<String, Polarizability>) -> Self {
        Self {
            table,
            prefix: "D".to_owned(),
        }
    }

    /// The polarizer over molrs's shipped `alpha.ff`
    /// ([`CLPOL_POLARIZABILITY`](crate::ff::params::CLPOL_POLARIZABILITY)).
    pub fn clpol() -> Self {
        Self::new(
            crate::ff::params::CLPOL_POLARIZABILITY
                .iter()
                .map(|row| {
                    (
                        row.type_name.to_owned(),
                        Polarizability {
                            m_d: row.m_d,
                            q_d_sign: row.q_d_sign,
                            k_d: row.k_d,
                            alpha: row.alpha,
                            a_thole: row.a_thole,
                        },
                    )
                })
                .collect(),
        )
    }

    /// Type each shell `<prefix><core type>`.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// `mol` with a Drude shell on every polarizable atom.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] naming the atoms when an atom that is no
    /// site has no `type` (its polarizability cannot be looked up: type the
    /// molecule first), or a polarizable atom has no `charge` to split.
    pub fn apply(&self, mol: &Atomistic) -> Result<Atomistic, MolRsError> {
        let kcal_per_kj = kj_to_kcal()?;
        // 4πε₀ in e²/(kJ·mol⁻¹·Å): the inverse of `real`'s Coulomb constant,
        // in kJ.
        let four_pi_eps0 = kcal_per_kj / UnitPreset::real().coulomb();
        let mut untyped = Vec::new();
        let mut hosts: Vec<(NodeId, String, Polarizability)> = Vec::new();
        for (row, (id, atom)) in mol.atoms().enumerate() {
            if atom.get_str(keys::VSITE).is_some() {
                continue;
            }
            let Some(type_name) = atom.get_str(keys::TYPE) else {
                untyped.push(row);
                continue;
            };
            if let Some(&params) = self.table.get(type_name).filter(|p| p.k_d > 0.0) {
                hosts.push((id, type_name.to_owned(), params));
            }
        }
        if !untyped.is_empty() {
            return Err(MolRsError::validation(format!(
                "DrudeBuilder needs a typed graph: atoms {untyped:?} have no 'type', so \
                 their CL&Pol polarizability cannot be looked up; typify the molecule first"
            )));
        }
        let uncharged: Vec<NodeId> = hosts
            .iter()
            .filter(|(id, ..)| charge_of(mol, *id).is_none())
            .map(|(id, ..)| *id)
            .collect();
        if !uncharged.is_empty() {
            return Err(MolRsError::validation(format!(
                "DrudeBuilder needs charged hosts: {} polarizable atom(s) have no 'charge' \
                 to split with their Drude shell",
                uncharged.len()
            )));
        }

        let mut out = mol.clone();
        for (host, core_type, params) in hosts {
            let sign = if params.q_d_sign >= 0.0 { 1.0 } else { -1.0 };
            let q_d = sign * (four_pi_eps0 * params.k_d * params.alpha).sqrt();
            let shell_type = format!("{}{core_type}", self.prefix);
            let core = out.get_atom(host)?;
            let mut shell_props = props! {
                keys::VSITE => "drude",
                keys::TYPE => shell_type.as_str(),
                keys::ATOMIC_NUMBER => 0_u8,
                keys::CHARGE => q_d,
                keys::MASS => params.m_d,
            };
            for axis in [keys::X, keys::Y, keys::Z] {
                if let Some(v) = core.get_f64(axis) {
                    shell_props.set(axis, v);
                }
            }
            let shell = out.add_atom(shell_props)?;
            let charge = charge_of(&out, host).expect("checked above");
            out.set_atom(host, keys::CHARGE, charge - q_d)?;
            if let Some(mass) = core.get_f64(keys::MASS) {
                out.set_atom(host, keys::MASS, mass - params.m_d)?;
            }
            out.add_term(
                "drudes",
                &[host, shell],
                props! {
                    keys::TYPE => format!("{core_type}-{shell_type}"),
                    "style" => "harmonic",
                    "k" => params.k_d / 2.0 * kcal_per_kj,
                    "alpha" => params.alpha,
                    "thole" => params.a_thole,
                },
            )?;
        }
        Ok(out)
    }
}

/// A TIP4P M-site on each water's HOH bisector.
///
/// The massless M-site sits at `d_om` from O along the bisector of the
/// minimum-image O→H vectors (plain differences without a box); O's charge
/// moves onto it and O is left neutral. The construction is a
/// `virtual_sites` row of style `average3` (`atomi` M, then O, H, H) with
/// `w1 = 1 − 2a`, `w2 = w3 = a`, `a = d_om / |r_OH1 + r_OH2|`, so an engine
/// rebuilds M from the moving atoms. A water is an oxygen bonded to exactly
/// two hydrogens.
#[derive(Clone, Debug)]
pub struct Tip4pBuilder {
    d_om: F,
    simbox: Option<SimBox>,
}

impl Tip4pBuilder {
    /// M-sites at `d_om` Å from O, O→H vectors as plain differences.
    pub fn new(d_om: F) -> Self {
        Self { d_om, simbox: None }
    }

    /// Take O→H vectors as minimum images in `simbox`.
    pub fn with_box(mut self, simbox: SimBox) -> Self {
        self.simbox = Some(simbox);
        self
    }

    /// `mol` with an M-site on every water.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] for a water atom without coordinates or
    /// charge, or a linear water (its bisector is undefined).
    pub fn apply(&self, mol: &Atomistic) -> Result<Atomistic, MolRsError> {
        let waters: Vec<(NodeId, [NodeId; 2])> = mol
            .atoms()
            .filter(|(_, a)| a.atomic_number() == Some(8))
            .filter_map(|(o, _)| {
                let hs: Vec<NodeId> = mol
                    .neighbors(o, NeighborKinds::BONDS)
                    .map(|(n, _)| n)
                    .filter(|&n| mol.get_atom(n).ok().and_then(|a| a.atomic_number()) == Some(1))
                    .collect();
                <[NodeId; 2]>::try_from(hs).ok().map(|hs| (o, hs))
            })
            .collect();
        let mut out = mol.clone();
        for (o, hs) in waters {
            let x_o = position(mol, o)?;
            let mut bisector = [0.0; 3];
            for h in hs {
                let x_h = position(mol, h)?;
                let raw = [x_h[0] - x_o[0], x_h[1] - x_o[1], x_h[2] - x_o[2]];
                let d = match &self.simbox {
                    Some(b) => b.mic().apply(raw),
                    None => raw,
                };
                for k in 0..3 {
                    bisector[k] += d[k];
                }
            }
            let norm = bisector.iter().map(|v| v * v).sum::<F>().sqrt();
            if norm == 0.0 {
                return Err(MolRsError::validation(
                    "TIP4P water is linear: its HOH bisector is undefined",
                ));
            }
            let a = self.d_om / norm;
            let charge = charge_of(mol, o).ok_or_else(|| {
                MolRsError::validation("TIP4P water oxygen has no 'charge' to move onto M")
            })?;
            let m = out.add_atom(props! {
                keys::VSITE => "massless",
                keys::ATOMIC_NUMBER => 0_u8,
                keys::CHARGE => charge,
                keys::X => x_o[0] + a * bisector[0],
                keys::Y => x_o[1] + a * bisector[1],
                keys::Z => x_o[2] + a * bisector[2],
            })?;
            out.set_atom(o, keys::CHARGE, 0.0)?;
            out.add_term(
                "virtual_sites",
                &[m, o, hs[0], hs[1]],
                props! {
                    "style" => "average3",
                    "w1" => 1.0 - 2.0 * a,
                    "w2" => a,
                    "w3" => a,
                },
            )?;
        }
        Ok(out)
    }
}

/// kJ → kcal.
fn kj_to_kcal() -> Result<F, MolRsError> {
    UnitRegistry::global()
        .factor("kJ", "kcal")
        .map_err(|e| MolRsError::validation(format!("kJ to kcal: {e}")))
}

fn charge_of(mol: &Atomistic, id: NodeId) -> Option<F> {
    mol.get_atom(id).ok()?.get_f64(keys::CHARGE)
}

fn position(mol: &Atomistic, id: NodeId) -> Result<[F; 3], MolRsError> {
    let atom = mol.get_atom(id)?;
    let axis = |key: &str| {
        atom.get_f64(key)
            .ok_or_else(|| MolRsError::validation(format!("TIP4P water atom has no '{key}'")))
    };
    Ok([axis(keys::X)?, axis(keys::Y)?, axis(keys::Z)?])
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Element;

    /// A charged, typed C–Cl pair: Cl polarizable in the shipped table.
    fn chloride() -> Atomistic {
        let mut m = Atomistic::new();
        let c = m
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C, keys::TYPE => "HC",
                keys::CHARGE => 0.1, keys::MASS => 12.011,
                keys::X => 0.0, keys::Y => 0.0, keys::Z => 0.0,
            })
            .unwrap();
        let cl = m
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::Cl, keys::TYPE => "Cl",
                keys::CHARGE => -1.1, keys::MASS => 35.45,
                keys::X => 1.8, keys::Y => 0.0, keys::Z => 0.0,
            })
            .unwrap();
        m.add_bond(c, cl, props! {}).unwrap();
        m
    }

    /// The shell recovers α from `q_D² / (4πε₀ k_D)`, the net charge and mass
    /// are conserved, and the spring is `k_D / 2` in kcal.
    #[test]
    fn a_drude_shell_splits_charge_and_mass_and_recovers_alpha() {
        let mol = chloride();
        let out = DrudeBuilder::clpol().apply(&mol).unwrap();
        assert_eq!(mol.n_atoms(), 2, "the input is untouched");
        assert_eq!(out.n_atoms(), 3);
        let total =
            |m: &Atomistic, key: &str| -> F { m.atoms().filter_map(|(_, a)| a.get_f64(key)).sum() };
        assert!((total(&out, keys::CHARGE) - total(&mol, keys::CHARGE)).abs() < 1e-12);
        assert!((total(&out, keys::MASS) - total(&mol, keys::MASS)).abs() < 1e-12);

        let (shell, atom) = out
            .atoms()
            .find(|(_, a)| a.get_str(keys::VSITE) == Some("drude"))
            .unwrap();
        assert_eq!(atom.get_str(keys::TYPE), Some("DCl"));
        let q_d = atom.get_f64(keys::CHARGE).unwrap();
        let kcal = kj_to_kcal().unwrap();
        let id = out.term_ids("drudes").unwrap().next().unwrap();
        let row = out.get_term("drudes", id).unwrap();
        assert_eq!(row.nodes[1], shell);
        let k = row.props.get_f64("k").unwrap();
        assert!((k - 4184.0 / 2.0 * kcal).abs() < 1e-9, "{k}");
        let four_pi_eps0 = kcal / UnitPreset::real().coulomb();
        let alpha = q_d * q_d / (four_pi_eps0 * 2.0 * k / kcal);
        assert!((alpha - 4.4).abs() < 1e-9, "{alpha}");
        assert!(q_d < 0.0);
    }

    #[test]
    fn an_untyped_atom_or_uncharged_host_is_refused() {
        let mut mol = chloride();
        let first = mol.atoms().next().unwrap().0;
        mol.clear_atom(first, keys::TYPE).unwrap();
        assert!(DrudeBuilder::clpol().apply(&mol).is_err());
        let mut mol = chloride();
        let cl = mol.atoms().nth(1).unwrap().0;
        mol.clear_atom(cl, keys::CHARGE).unwrap();
        let err = DrudeBuilder::clpol().apply(&mol).unwrap_err();
        assert!(err.to_string().contains("charge"), "{err}");
    }

    /// The M-site sits `d_om` along the bisector, carries O's charge, and the
    /// `average3` weights rebuild it.
    #[test]
    fn a_tip4p_site_sits_on_the_bisector_and_rebuilds_from_its_weights() {
        let mut w = Atomistic::new();
        let o = w
            .add_atom(
                props! { keys::ATOMIC_NUMBER => Element::O, keys::CHARGE => -0.8,
                keys::X => 0.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        let mut hs = Vec::new();
        for x in [0.757, -0.757] {
            let h = w
                .add_atom(
                    props! { keys::ATOMIC_NUMBER => Element::H, keys::CHARGE => 0.4,
                    keys::X => x, keys::Y => 0.586, keys::Z => 0.0 },
                )
                .unwrap();
            w.add_bond(o, h, props! {}).unwrap();
            hs.push(h);
        }
        let out = Tip4pBuilder::new(TIP4P_OM).apply(&w).unwrap();
        let (_, m) = out
            .atoms()
            .find(|(_, a)| a.get_str(keys::VSITE) == Some("massless"))
            .unwrap();
        assert_eq!(m.get_f64(keys::CHARGE), Some(-0.8));
        assert!((m.get_f64(keys::Y).unwrap() - TIP4P_OM).abs() < 1e-12);
        assert_eq!(out.get_atom(o).unwrap().get_f64(keys::CHARGE), Some(0.0));
        let id = out.term_ids("virtual_sites").unwrap().next().unwrap();
        let row = out.get_term("virtual_sites", id).unwrap();
        let wt = |k: &str| row.props.get_f64(k).unwrap();
        let rebuilt = wt("w2") * 0.586 + wt("w3") * 0.586;
        assert!((rebuilt - TIP4P_OM).abs() < 1e-12, "{rebuilt}");
        assert!((wt("w1") + wt("w2") + wt("w3") - 1.0).abs() < 1e-12);
    }
}
