//! Space-group symmetry: affine operators on fractional coordinates.
//!
//! A crystal structure is usually published as an *asymmetric unit* plus a
//! space group; the unit cell is recovered by applying every operator to each
//! site. [`SpaceGroup`] holds the operators — from an explicit list of
//! Jones-faithful triplets (`"-y+1/2, x, z+1/4"`, the form CIFs use), from a
//! CIF frame's [`keys::SYMOPS`](crate::core::keys::SYMOPS), or closed from a
//! handful of generators — and expands a fractional site into its symmetry
//! images, de-duplicated under the periodic boundary.
//!
//! Operators compose in floating point, so a translation such as
//! `1/3 + 2/3` comes out as `0.9999999999999999`; every translation is
//! wrapped into `[0, 1)` with a tolerance ([`WRAP_TOL`]), and operators are
//! compared on keys rounded to [`KEY_DECIMALS`] places.

use std::collections::HashSet;

use crate::core::{Frame, MetaValue, keys};
use crate::op::F;

/// A wrapped coordinate this close below 1 is the lattice translation 0.
pub const WRAP_TOL: F = 1e-9;
/// Decimal places an operator is rounded to when two are compared.
pub const KEY_DECIMALS: i32 = 6;
/// The largest group [`SpaceGroup::from_generators`] closes before it gives
/// up on a set that is not crystallographic.
pub const MAX_ORDER: usize = 1536;

/// One affine operator: `x' = R x + t` on fractional coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SymOp {
    /// The rotation / reflection, row-major.
    pub rotation: [[F; 3]; 3],
    /// The translation, each component in `[0, 1)`.
    pub translation: [F; 3],
}

impl SymOp {
    /// The identity.
    pub const IDENTITY: Self = Self {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
    };

    /// Parse a Jones-faithful triplet (`"x,y,z"`, `"-y+1/2, x+1/2, z"`).
    ///
    /// # Errors
    ///
    /// A triplet without exactly three components, an empty component, or a
    /// term that is neither a coordinate (`x`, `-2y`, `1/2z`) nor a number
    /// (`1/4`, `0.5`), named.
    pub fn parse(triplet: &str) -> Result<Self, String> {
        let comps: Vec<&str> = triplet.split(',').collect();
        let [a, b, c] = comps.as_slice() else {
            return Err(format!(
                "expected 3 components, got {}: {triplet:?}",
                comps.len()
            ));
        };
        let mut op = SymOp {
            rotation: [[0.0; 3]; 3],
            translation: [0.0; 3],
        };
        for (axis, comp) in [a, b, c].into_iter().enumerate() {
            let (row, t) =
                parse_component(comp).map_err(|e| format!("{e} in triplet {triplet:?}"))?;
            op.rotation[axis] = row;
            op.translation[axis] = t;
        }
        Ok(op)
    }

    /// `self` applied after `first`: `R = R₂R₁`, `t = R₂t₁ + t₂` wrapped.
    fn after(&self, first: &SymOp) -> SymOp {
        let rotation = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..3)
                    .map(|k| self.rotation[i][k] * first.rotation[k][j])
                    .sum()
            })
        });
        let translation: [F; 3] = std::array::from_fn(|i| {
            (0..3)
                .map(|k| self.rotation[i][k] * first.translation[k])
                .sum::<F>()
                + self.translation[i]
        });
        SymOp {
            rotation,
            translation: translation.map(|v| wrap(v, WRAP_TOL)),
        }
    }

    /// The image of fractional point `p`, wrapped into `[0, 1)`.
    pub fn apply(&self, p: [F; 3], tol: F) -> [F; 3] {
        std::array::from_fn(|i| {
            let v = (0..3).map(|k| self.rotation[i][k] * p[k]).sum::<F>() + self.translation[i];
            wrap(v, tol)
        })
    }

    /// A key equal for operators that differ only by rounding: the
    /// translation is rounded before it is reduced mod 1, so `0.9999…` and
    /// `0.0` key alike.
    fn key(&self) -> [i64; 12] {
        let scale = 10f64.powi(KEY_DECIMALS);
        let round = |v: F| (v * scale).round() as i64;
        let modulus = scale as i64;
        let mut key = [0; 12];
        for i in 0..3 {
            for j in 0..3 {
                key[3 * i + j] = round(self.rotation[i][j]);
            }
            key[9 + i] = round(self.translation[i]).rem_euclid(modulus);
        }
        key
    }
}

/// `v` reduced into `[0, 1)`; a value within `tol` below 1 is 0.
fn wrap(v: F, tol: F) -> F {
    let w = v.rem_euclid(1.0);
    if w >= 1.0 - tol { 0.0 } else { w }
}

/// One component (`"1/4-y"`) as its `[cx, cy, cz]` row and its translation,
/// reduced into `[0, 1)`.
fn parse_component(comp: &str) -> Result<([F; 3], F), String> {
    let compact: String = comp.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.is_empty() {
        return Err("empty coordinate component".to_owned());
    }
    let mut row = [0.0; 3];
    let mut translation = 0.0;
    // Split before every sign so each term carries its own.
    let mut terms = Vec::new();
    let mut start = 0;
    for (i, ch) in compact.char_indices() {
        if (ch == '+' || ch == '-') && i > start {
            terms.push(&compact[start..i]);
            start = i;
        }
    }
    terms.push(&compact[start..]);
    for term in terms {
        let (sign, body) = match term.strip_prefix('-') {
            Some(rest) => (-1.0, rest),
            None => (1.0, term.strip_prefix('+').unwrap_or(term)),
        };
        match body.chars().last() {
            Some(axis @ ('x' | 'y' | 'z' | 'X' | 'Y' | 'Z')) => {
                let coeff = &body[..body.len() - 1];
                let value = if coeff.is_empty() {
                    1.0
                } else {
                    number(coeff)?
                };
                row["xyz".find(axis.to_ascii_lowercase()).expect("an axis")] += sign * value;
            }
            _ => translation += sign * number(body)?,
        }
    }
    Ok((row, translation.rem_euclid(1.0)))
}

/// A number written `a/b` or as a decimal.
fn number(text: &str) -> Result<F, String> {
    let parse = |s: &str| {
        s.parse::<F>()
            .map_err(|_| format!("{text:?} is not a number or a coordinate"))
    };
    match text.split_once('/') {
        Some((num, den)) => Ok(parse(num)? / parse(den)?),
        None => parse(text),
    }
}

/// A space group as its list of affine operators, identity included.
#[derive(Clone, Debug, PartialEq)]
pub struct SpaceGroup {
    operators: Vec<SymOp>,
}

impl SpaceGroup {
    /// The group of an explicit, complete operator list, taken verbatim.
    ///
    /// # Errors
    ///
    /// The first triplet that does not parse ([`SymOp::parse`]).
    pub fn from_triplets<'a>(triplets: impl IntoIterator<Item = &'a str>) -> Result<Self, String> {
        let operators = triplets
            .into_iter()
            .map(SymOp::parse)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { operators })
    }

    /// The group a CIF frame's operator loop states
    /// ([`keys::SYMOPS`], as the CIF reader keeps it).
    ///
    /// # Errors
    ///
    /// A frame without `symops` (a CIF with no operator loop is not taken to
    /// be P1: state `x,y,z` in the file), a `symops` that is no list of
    /// strings, or a triplet that does not parse.
    pub fn from_frame(frame: &Frame) -> Result<Self, String> {
        let symops = frame.meta.get(keys::SYMOPS).ok_or_else(|| {
            format!(
                "frame.meta has no '{}' (the CIF has no _space_group_symop_operation_xyz \
                 or _symmetry_equiv_pos_as_xyz loop); refusing to assume P1",
                keys::SYMOPS
            )
        })?;
        let MetaValue::Json(serde_json::Value::Array(items)) = symops else {
            return Err(format!("meta '{}' is not a list of triplets", keys::SYMOPS));
        };
        let triplets = items
            .iter()
            .map(|v| {
                v.as_str()
                    .ok_or_else(|| format!("meta '{}' holds a non-string {v}", keys::SYMOPS))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_triplets(triplets)
    }

    /// The group closed from `generators`: compose operators until the set
    /// is closed under multiplication; the identity is added when absent.
    ///
    /// # Errors
    ///
    /// A triplet that does not parse, or a set that has not closed by
    /// `MAX_ORDER` (1536) operators (it is not crystallographic).
    pub fn from_generators<'a>(
        generators: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self, String> {
        let mut operators = Self::from_triplets(generators)?.operators;
        let identity = SymOp::IDENTITY.key();
        if !operators.iter().any(|op| op.key() == identity) {
            operators.insert(0, SymOp::IDENTITY);
        }
        let mut seen: HashSet<[i64; 12]> = operators.iter().map(SymOp::key).collect();
        let mut i = 0;
        while i < operators.len() {
            let first = operators[i];
            for j in 0..operators.len() {
                let op = operators[j].after(&first);
                if seen.insert(op.key()) {
                    operators.push(op);
                    if operators.len() > MAX_ORDER {
                        return Err(format!(
                            "generator set did not close (> {MAX_ORDER} operators)"
                        ));
                    }
                }
            }
            i += 1;
        }
        Ok(Self { operators })
    }

    /// The operators, in group order.
    pub fn operators(&self) -> &[SymOp] {
        &self.operators
    }

    /// Number of operators.
    pub fn order(&self) -> usize {
        self.operators.len()
    }

    /// Every symmetry image of fractional site `frac` in `[0, 1)`, an image
    /// within `symprec` (per fractional axis, under the periodic boundary)
    /// of an earlier one dropped — the site's multiplicity is the length.
    pub fn equivalent_positions(&self, frac: [F; 3], symprec: F) -> Vec<[F; 3]> {
        let mut out: Vec<[F; 3]> = Vec::new();
        for op in &self.operators {
            let p = op.apply(frac, symprec);
            let same = |q: &[F; 3]| {
                (0..3).all(|k| {
                    let d = p[k] - q[k];
                    (d - d.round()).abs() < symprec
                })
            };
            if !out.iter().any(same) {
                out.push(p);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_triplet_parses_its_rotation_and_wrapped_translation() {
        let op = SymOp::parse("-y+1/2, x, z-1/4").unwrap();
        assert_eq!(
            op.rotation,
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        );
        assert_eq!(op.translation, [0.5, 0.0, 0.75]);
        assert_eq!(
            SymOp::parse("1/2-x, 0.5+Y, 2z").unwrap().rotation[2],
            [0.0, 0.0, 2.0]
        );
        for bad in ["x,y", "x,,z", "x,y,q"] {
            assert!(SymOp::parse(bad).is_err(), "{bad}");
        }
    }

    /// Inversion alone closes into P-1 (order 2); adding the two
    /// face-centring translations makes the F-centred group four times that.
    #[test]
    fn generators_close_into_the_group() {
        let p1bar = SpaceGroup::from_generators(["-x,-y,-z"]).unwrap();
        assert_eq!(p1bar.order(), 2);
        let f_centred =
            SpaceGroup::from_generators(["-x,-y,-z", "x,y+1/2,z+1/2", "x+1/2,y,z+1/2"]).unwrap();
        assert_eq!(f_centred.order(), 8);
        assert!(SpaceGroup::from_generators(["x+1/7,y,z"]).is_ok());
    }

    /// A general position has the group's order of images; a special one,
    /// fewer.
    #[test]
    fn equivalent_positions_collapse_on_special_positions() {
        let group = SpaceGroup::from_generators(["-x,-y,-z", "x+1/2,y+1/2,z"]).unwrap();
        assert_eq!(group.order(), 4);
        assert_eq!(group.equivalent_positions([0.1, 0.2, 0.3], 1e-5).len(), 4);
        assert_eq!(group.equivalent_positions([0.0, 0.0, 0.0], 1e-5).len(), 2);
    }

    #[test]
    fn a_cif_frame_states_its_operators_or_is_refused() {
        let mut frame = Frame::new();
        assert!(SpaceGroup::from_frame(&frame).unwrap_err().contains("P1"));
        frame.meta.insert(
            keys::SYMOPS,
            MetaValue::Json(serde_json::json!(["x,y,z", "-x,-y,-z"])),
        );
        assert_eq!(SpaceGroup::from_frame(&frame).unwrap().order(), 2);
    }
}
