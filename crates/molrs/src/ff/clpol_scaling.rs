//! CL&Pol SAPT-derived Lennard-Jones scaling: a force field in, its `pair`
//! epsilon (and optionally sigma) scaled per fragment pair out.

use std::collections::HashMap;
use std::fmt;

use super::forcefield::{ForceField, StyleDefs};
use crate::op::centroid;

const C0: f64 = 0.254_952;
const C1: f64 = 0.106_906;
const SIGMA_SCALE: f64 = 0.985;

/// Scaling properties for one molecular fragment.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentScaling {
    pub name: String,
    pub q: f64,
    pub mu: f64,
    pub alpha: f64,
    pub polarizable: bool,
}

/// CL&Pol's fragment scaling table (paduagroup/clandpol `fragment.ff`,
/// [`CLPOL_FRAGMENTS`](crate::ff::params::CLPOL_FRAGMENTS)): each fragment's
/// charge, dipole and polarizability, by fragment name, as [`scale_lj`] reads
/// it.
pub fn fragment_table() -> HashMap<String, FragmentScaling> {
    crate::ff::params::CLPOL_FRAGMENTS
        .iter()
        .copied()
        .map(|(name, q, mu, alpha, polarizable)| {
            (
                name.to_string(),
                FragmentScaling {
                    name: name.to_string(),
                    q,
                    mu,
                    alpha,
                    polarizable,
                },
            )
        })
        .collect()
}

/// Atom data needed to assign force-field types and calculate a fragment COM.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentAtoms {
    pub name: String,
    pub atom_types: Vec<String>,
    pub coords: Vec<[f64; 3]>,
    /// Per-atom masses weighting the centre of mass. A non-positive finite
    /// mass counts as weight 1; a non-finite mass, or a total that overflows,
    /// is [`ScaleLjError::InvalidMass`]. An empty fragment's centre is the
    /// origin.
    pub masses: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScaleLjError {
    InvalidAlpha(String),
    InvalidMass(String),
    MissingFragment(String),
    Shape(String),
}

impl fmt::Display for ScaleLjError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlpha(name) => write!(f, "fragment '{name}' alpha must be positive"),
            Self::InvalidMass(name) => {
                write!(f, "fragment '{name}' has a non-finite mass or total mass")
            }
            Self::MissingFragment(name) => write!(f, "no scaling data for fragment '{name}'"),
            Self::Shape(name) => write!(
                f,
                "fragment '{name}' types, coordinates, and masses must have equal length"
            ),
        }
    }
}

impl std::error::Error for ScaleLjError {}

/// SAPT epsilon-scaling factor for a fragment pair at COM distance `r` (Å).
pub fn compute_k_ij(
    fr_i: &FragmentScaling,
    fr_j: &FragmentScaling,
    r: f64,
) -> Result<f64, ScaleLjError> {
    if fr_i.alpha <= 0.0 {
        return Err(ScaleLjError::InvalidAlpha(fr_i.name.clone()));
    }
    if fr_j.alpha <= 0.0 {
        return Err(ScaleLjError::InvalidAlpha(fr_j.name.clone()));
    }
    let mut denominator = 1.0;
    if !fr_i.polarizable {
        denominator += C0 * r * r * fr_j.q.powi(2) / fr_j.alpha + C1 * fr_j.mu.powi(2) / fr_j.alpha;
    }
    if !fr_j.polarizable {
        denominator += C0 * r * r * fr_i.q.powi(2) / fr_i.alpha + C1 * fr_i.mu.powi(2) / fr_i.alpha;
    }
    Ok(1.0 / denominator)
}

/// Mass-weighted centre of `fragment`. A non-positive finite mass counts as
/// weight 1; an empty fragment's centre is the origin; a non-finite mass or an
/// overflowing total mass is [`ScaleLjError::InvalidMass`].
fn center_of_mass(fragment: &FragmentAtoms) -> Result<[f64; 3], ScaleLjError> {
    if fragment.atom_types.len() != fragment.coords.len()
        || fragment.coords.len() != fragment.masses.len()
    {
        return Err(ScaleLjError::Shape(fragment.name.clone()));
    }
    // A non-finite mass has no centre of mass to give.
    if fragment.masses.iter().any(|mass| !mass.is_finite()) {
        return Err(ScaleLjError::InvalidMass(fragment.name.clone()));
    }
    // An empty fragment's centre is the origin.
    if fragment.masses.is_empty() {
        return Ok([0.0; 3]);
    }
    // A non-positive finite mass counts as weight 1.
    let weights: Vec<f64> = fragment
        .masses
        .iter()
        .map(|&mass| if mass > 0.0 { mass } else { 1.0 })
        .collect();
    // Every weight is positive and finite, so `None` means the total mass
    // overflowed `f64` — a non-finite fragment mass.
    centroid(&fragment.coords, &weights)
        .ok_or_else(|| ScaleLjError::InvalidMass(fragment.name.clone()))
}

/// Clone and scale cross-fragment LJ pair parameters without mutating `ff`.
pub fn scale_lj(
    ff: &ForceField,
    fragments: &[FragmentAtoms],
    scaling: &HashMap<String, FragmentScaling>,
    scale_sigma: bool,
) -> Result<ForceField, ScaleLjError> {
    let mut type_to_fragment = HashMap::new();
    let mut centers = HashMap::new();
    for fragment in fragments {
        if !scaling.contains_key(&fragment.name) {
            return Err(ScaleLjError::MissingFragment(fragment.name.clone()));
        }
        centers.insert(fragment.name.clone(), center_of_mass(fragment)?);
        for atom_type in &fragment.atom_types {
            type_to_fragment.insert(atom_type.clone(), fragment.name.clone());
        }
    }

    // Edit a clone: every piece of declared state (name, units, special_bonds, style
    // params such as `mixing`) is kept by construction.
    let mut output = ff.clone();
    // Read pass: the scaled values of each cross-fragment pair type.
    struct PairEdit {
        style: String,
        pair: String,
        epsilon: Option<f64>,
        sigma: Option<f64>,
    }
    let mut edits: Vec<PairEdit> = Vec::new();
    for style in output.get_styles("pair") {
        let StyleDefs::Pair(types) = style.defs() else {
            continue;
        };
        for pair in types {
            let (Some(fi), Some(fj)) = (
                type_to_fragment.get(&pair.itom),
                type_to_fragment.get(&pair.jtom),
            ) else {
                continue;
            };
            if fi == fj {
                continue;
            }
            let ci = centers[fi];
            let cj = centers[fj];
            let distance =
                ((ci[0] - cj[0]).powi(2) + (ci[1] - cj[1]).powi(2) + (ci[2] - cj[2]).powi(2))
                    .sqrt();
            let factor = compute_k_ij(&scaling[fi], &scaling[fj], distance)?;
            // `set_type_param` edits the first row of a name, so only the first
            // row of a duplicated name is read (02 makes duplicates impossible).
            if edits
                .iter()
                .any(|e| e.style == style.name() && e.pair == pair.name)
            {
                continue;
            }
            edits.push(PairEdit {
                style: style.name().to_owned(),
                pair: pair.name.clone(),
                epsilon: pair
                    .params
                    .get("epsilon")
                    .and_then(|v| v.as_num())
                    .map(|eps| eps * factor),
                sigma: pair
                    .params
                    .get("sigma")
                    .and_then(|v| v.as_num())
                    .filter(|_| scale_sigma)
                    .map(|sigma| sigma * SIGMA_SCALE),
            });
        }
    }
    // Write pass: edit each through `set_type_param` (an edit, not a definition).
    for edit in edits {
        let style = output
            .get_style_mut("pair", &edit.style)
            .expect("the style was read from this force field above");
        if let Some(epsilon) = edit.epsilon {
            style.set_type_param(&edit.pair, "epsilon", epsilon);
        }
        if let Some(sigma) = edit.sigma {
            style.set_type_param(&edit.pair, "sigma", sigma);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::ir::{Params, SpecialBonds};

    /// Fragment `A` holds types `A1`, `A2` with its centre of mass at the
    /// origin; fragment `B` holds `B1` at (3, 4, 0), so the COM distance is 5.
    fn fragments() -> Vec<FragmentAtoms> {
        vec![
            FragmentAtoms {
                name: "A".into(),
                atom_types: vec!["A1".into(), "A2".into()],
                coords: vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                masses: vec![1.0, 1.0],
            },
            FragmentAtoms {
                name: "B".into(),
                atom_types: vec!["B1".into()],
                coords: vec![[3.0, 4.0, 0.0]],
                masses: vec![1.0],
            },
        ]
    }

    fn scaling() -> HashMap<String, FragmentScaling> {
        HashMap::from([
            (
                "A".to_string(),
                FragmentScaling {
                    name: "A".into(),
                    q: 1.0,
                    mu: 0.0,
                    alpha: 2.0,
                    polarizable: false,
                },
            ),
            (
                "B".to_string(),
                FragmentScaling {
                    name: "B".into(),
                    q: 0.0,
                    mu: 1.0,
                    alpha: 4.0,
                    polarizable: false,
                },
            ),
        ])
    }

    const INPUT_SPECIAL_BONDS: SpecialBonds = SpecialBonds {
        lj: [0.0, 0.0, 0.5],
        coul: [0.0, 0.0, 0.8333],
    };

    /// One `lj/cut` style (`mixing = geometric`) with a cross-fragment pair
    /// `A1-B1` and a same-fragment pair `A1-A2`.
    fn input_ff() -> ForceField {
        let mut style_params = Params::from_iter([("cutoff", 10.0)]);
        style_params.set("mixing", "geometric");
        let mut ff = ForceField::new("clpol");
        ff.set_special_bonds(INPUT_SPECIAL_BONDS);
        ff.def_style("pair", "lj/cut", style_params)
            .unwrap()
            .def_type(
                "A1-B1",
                &["A1", "B1"],
                Params::from_iter([("epsilon", 0.2), ("sigma", 3.0)]),
            )
            .unwrap()
            .def_type(
                "A1-A2",
                &["A1", "A2"],
                Params::from_iter([("epsilon", 0.3), ("sigma", 3.2)]),
            )
            .unwrap();
        ff
    }

    fn pair_param(ff: &ForceField, itom: &str, jtom: &str, key: &str) -> Option<f64> {
        ff.get_style("pair", "lj/cut")?
            .get_pairtype(itom, Some(jtom))?
            .params
            .get(key)
            .and_then(|v| v.as_num())
    }

    #[test]
    fn cross_fragment_epsilon_is_input_times_k_ij() {
        let out = scale_lj(&input_ff(), &fragments(), &scaling(), false).unwrap();
        let s = scaling();
        let expected = 0.2 * compute_k_ij(&s["A"], &s["B"], 5.0).unwrap();
        let got = pair_param(&out, "A1", "B1", "epsilon").unwrap();
        assert!((got - expected).abs() < 1e-12, "{got} != {expected}");
    }

    #[test]
    fn scale_sigma_multiplies_cross_fragment_sigma_by_0_985() {
        let out = scale_lj(&input_ff(), &fragments(), &scaling(), true).unwrap();
        let got = pair_param(&out, "A1", "B1", "sigma").unwrap();
        assert!((got - 3.0 * 0.985).abs() < 1e-12, "{got}");
    }

    #[test]
    fn same_fragment_pair_is_untouched() {
        let out = scale_lj(&input_ff(), &fragments(), &scaling(), true).unwrap();
        assert_eq!(pair_param(&out, "A1", "A2", "epsilon"), Some(0.3));
        assert_eq!(pair_param(&out, "A1", "A2", "sigma"), Some(3.2));
    }

    #[test]
    fn output_keeps_the_input_special_bonds() {
        let out = scale_lj(&input_ff(), &fragments(), &scaling(), false).unwrap();
        assert_eq!(*out.special_bonds(), INPUT_SPECIAL_BONDS);
    }

    /// Declared state is carried as declared, not re-derived from the
    /// defaulting getters.
    #[test]
    fn output_keeps_the_declared_units_and_special_bonds() {
        let mut ff = input_ff();
        ff.set_units("metal");

        let out = scale_lj(&ff, &fragments(), &scaling(), false).unwrap();

        assert_eq!(out.declared_units(), ff.declared_units());
        assert_eq!(out.declared_units(), Some("metal"));
        assert_eq!(out.declared_special_bonds(), ff.declared_special_bonds());
        assert_eq!(out.declared_special_bonds(), Some(&INPUT_SPECIAL_BONDS));
    }

    #[test]
    fn output_keeps_the_pair_style_mixing_param() {
        let out = scale_lj(&input_ff(), &fragments(), &scaling(), false).unwrap();
        let style = out.get_style("pair", "lj/cut").unwrap();
        assert_eq!(
            style.params().get("mixing").and_then(|v| v.as_text()),
            Some("geometric")
        );
    }

    #[test]
    fn infinite_mass_is_refused_not_placed_at_the_origin() {
        let mut frags = fragments();
        frags[1].masses = vec![f64::INFINITY];
        let err = scale_lj(&input_ff(), &frags, &scaling(), false).unwrap_err();
        assert_eq!(err, ScaleLjError::InvalidMass("B".into()));
    }

    #[test]
    fn closed_form_keeps_mu_term_independent_of_distance() {
        let a = FragmentScaling {
            name: "a".into(),
            q: -1.0,
            mu: 0.0,
            alpha: 3.0,
            polarizable: false,
        };
        let b = FragmentScaling {
            name: "b".into(),
            q: 1.0,
            mu: 2.0,
            alpha: 8.0,
            polarizable: true,
        };
        let inv1 = 1.0 / compute_k_ij(&a, &b, 3.0).unwrap();
        let inv2 = 1.0 / compute_k_ij(&a, &b, 6.0).unwrap();
        let charge_only = C0 * (3.0_f64.powi(2) - 6.0_f64.powi(2)) / b.alpha;
        assert!((inv1 - inv2 - charge_only).abs() < 1e-12);
    }
}
