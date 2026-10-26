//! CL&Pol SAPT-derived Lennard-Jones scaling ([`ClpolScaling`]): a force
//! field in, its `pair` epsilon (and optionally sigma) scaled per fragment
//! pair out.

use std::collections::HashMap;
use std::fmt;

use super::forcefield::ForceField;
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

impl FragmentScaling {
    /// The SAPT epsilon-scaling factor of this fragment and `other` at
    /// centre-of-mass distance `r` (Å).
    ///
    /// # Errors
    ///
    /// [`ClpolScalingError::InvalidAlpha`] for a fragment whose
    /// polarizability is not positive.
    pub fn k_ij(&self, other: &FragmentScaling, r: f64) -> Result<f64, ClpolScalingError> {
        for fragment in [self, other] {
            if fragment.alpha <= 0.0 {
                return Err(ClpolScalingError::InvalidAlpha(fragment.name.clone()));
            }
        }
        let mut denominator = 1.0;
        if !self.polarizable {
            denominator +=
                C0 * r * r * other.q.powi(2) / other.alpha + C1 * other.mu.powi(2) / other.alpha;
        }
        if !other.polarizable {
            denominator +=
                C0 * r * r * self.q.powi(2) / self.alpha + C1 * self.mu.powi(2) / self.alpha;
        }
        Ok(1.0 / denominator)
    }
}

/// CL&Pol's Lennard-Jones scaling: the fragment table it reads (each
/// fragment's charge, dipole and polarizability, by name) and whether it also
/// scales σ.
#[derive(Debug, Clone, PartialEq)]
pub struct ClpolScaling {
    pub fragments: HashMap<String, FragmentScaling>,
    /// Scale each cross-fragment σ by 0.985 as well as ε.
    pub scale_sigma: bool,
}

impl Default for ClpolScaling {
    /// CL&Pol's own fragment table (paduagroup/clandpol `fragment.ff`,
    /// [`CLPOL_FRAGMENTS`](crate::ff::params::CLPOL_FRAGMENTS)), ε alone.
    fn default() -> Self {
        Self {
            fragments: clpol_fragments(),
            scale_sigma: false,
        }
    }
}

/// [`CLPOL_FRAGMENTS`](crate::ff::params::CLPOL_FRAGMENTS) by fragment name.
fn clpol_fragments() -> HashMap<String, FragmentScaling> {
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
    /// is [`ClpolScalingError::InvalidMass`]. An empty fragment's centre is the
    /// origin.
    pub masses: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClpolScalingError {
    InvalidAlpha(String),
    InvalidMass(String),
    MissingFragment(String),
    Shape(String),
}

impl fmt::Display for ClpolScalingError {
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

impl std::error::Error for ClpolScalingError {}

/// Mass-weighted centre of `fragment`. A non-positive finite mass counts as
/// weight 1; an empty fragment's centre is the origin; a non-finite mass or an
/// overflowing total mass is [`ClpolScalingError::InvalidMass`].
fn center_of_mass(fragment: &FragmentAtoms) -> Result<[f64; 3], ClpolScalingError> {
    if fragment.atom_types.len() != fragment.coords.len()
        || fragment.coords.len() != fragment.masses.len()
    {
        return Err(ClpolScalingError::Shape(fragment.name.clone()));
    }
    // A non-finite mass has no centre of mass to give.
    if fragment.masses.iter().any(|mass| !mass.is_finite()) {
        return Err(ClpolScalingError::InvalidMass(fragment.name.clone()));
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
        .ok_or_else(|| ClpolScalingError::InvalidMass(fragment.name.clone()))
}

impl ClpolScaling {
    /// A copy of `ff` whose cross-fragment `pair` rows have ε (and, with
    /// [`scale_sigma`](Self::scale_sigma), σ) scaled by the SAPT factor of
    /// their two fragments ([`FragmentScaling::k_ij`]) at the distance of the
    /// fragments' centres of mass. `ff` is not changed.
    ///
    /// # Errors
    ///
    /// A fragment the table has no row for, a fragment whose types,
    /// coordinates and masses differ in length, a non-finite mass, or a
    /// non-positive polarizability.
    pub fn apply(
        &self,
        ff: &ForceField,
        fragments: &[FragmentAtoms],
    ) -> Result<ForceField, ClpolScalingError> {
        let (scaling, scale_sigma) = (&self.fragments, self.scale_sigma);
        let mut type_to_fragment = HashMap::new();
        let mut centers = HashMap::new();
        for fragment in fragments {
            if !scaling.contains_key(&fragment.name) {
                return Err(ClpolScalingError::MissingFragment(fragment.name.clone()));
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
            for pair in style.defs().types() {
                let (Some(fi), Some(fj)) = (
                    type_to_fragment.get(&pair.endpoints[0]),
                    type_to_fragment.get(&pair.endpoints[1]),
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
                let factor = scaling[fi].k_ij(&scaling[fj], distance)?;
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
        let out = ClpolScaling {
            fragments: scaling(),
            scale_sigma: false,
        }
        .apply(&input_ff(), &fragments())
        .unwrap();
        let s = scaling();
        let expected = 0.2 * s["A"].k_ij(&s["B"], 5.0).unwrap();
        let got = pair_param(&out, "A1", "B1", "epsilon").unwrap();
        assert!((got - expected).abs() < 1e-12, "{got} != {expected}");
    }

    #[test]
    fn scale_sigma_multiplies_cross_fragment_sigma_by_0_985() {
        let out = ClpolScaling {
            fragments: scaling(),
            scale_sigma: true,
        }
        .apply(&input_ff(), &fragments())
        .unwrap();
        let got = pair_param(&out, "A1", "B1", "sigma").unwrap();
        assert!((got - 3.0 * 0.985).abs() < 1e-12, "{got}");
    }

    #[test]
    fn same_fragment_pair_is_untouched() {
        let out = ClpolScaling {
            fragments: scaling(),
            scale_sigma: true,
        }
        .apply(&input_ff(), &fragments())
        .unwrap();
        assert_eq!(pair_param(&out, "A1", "A2", "epsilon"), Some(0.3));
        assert_eq!(pair_param(&out, "A1", "A2", "sigma"), Some(3.2));
    }

    #[test]
    fn output_keeps_the_input_special_bonds() {
        let out = ClpolScaling {
            fragments: scaling(),
            scale_sigma: false,
        }
        .apply(&input_ff(), &fragments())
        .unwrap();
        assert_eq!(*out.special_bonds().unwrap(), INPUT_SPECIAL_BONDS);
    }

    /// Declared state is carried as declared, not re-derived from the
    /// defaulting getters.
    #[test]
    fn output_keeps_the_declared_units_and_special_bonds() {
        let mut ff = input_ff();
        ff.set_units("metal");

        let out = ClpolScaling {
            fragments: scaling(),
            scale_sigma: false,
        }
        .apply(&ff, &fragments())
        .unwrap();

        assert_eq!(out.units(), ff.units());
        assert_eq!(out.units(), Some("metal"));
        assert_eq!(out.special_bonds(), ff.special_bonds());
        assert_eq!(out.special_bonds(), Some(&INPUT_SPECIAL_BONDS));
    }

    #[test]
    fn output_keeps_the_pair_style_mixing_param() {
        let out = ClpolScaling {
            fragments: scaling(),
            scale_sigma: false,
        }
        .apply(&input_ff(), &fragments())
        .unwrap();
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
        let err = ClpolScaling {
            fragments: scaling(),
            scale_sigma: false,
        }
        .apply(&input_ff(), &frags)
        .unwrap_err();
        assert_eq!(err, ClpolScalingError::InvalidMass("B".into()));
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
        let inv1 = 1.0 / a.k_ij(&b, 3.0).unwrap();
        let inv2 = 1.0 / a.k_ij(&b, 6.0).unwrap();
        let charge_only = C0 * (3.0_f64.powi(2) - 6.0_f64.powi(2)) / b.alpha;
        assert!((inv1 - inv2 - charge_only).abs() < 1e-12);
    }
}
