//! [`Params`]: the key-value parameters of a style or a type, and the key
//! ([`pair_key`]) a pair kernel finds a type's row under.

use std::collections::HashMap;

use crate::ff::ir::ParamValue;

/// Key-value parameter bag for type definitions: one [`ParamValue`] per key.
///
/// A number is what energy kernels read (`k`, `r0`, the numeric type `id`,
/// …); a string carries I/O metadata across the boundary (`element`, or any
/// string metadata carried by convention as a keyword param); an array is an
/// N-D `f64` parameter (a CMAP correction's `grid`). Read one kind with
/// [`ParamValue::as_num`], [`ParamValue::as_text`] or
/// [`ParamValue::as_array`] — `None` when the value is another kind.
///
/// Equality is exact (the same keys, `f64` values equal under `==` with no
/// tolerance, equal strings, arrays of one shape with every element equal
/// under `==`): it decides whether a re-definition is the same definition,
/// which is a question of identity, not closeness.
///
/// ```
/// use molrs::ff::ir::{ParamValue, Params};
///
/// let mut p: Params = [("k", 300.0), ("r0", 1.5)].into_iter().collect();
/// p.set("element", "C");
/// assert_eq!(p.get("k").and_then(ParamValue::as_num), Some(300.0));
/// assert_eq!(p.get("element").and_then(ParamValue::as_text), Some("C"));
/// assert_eq!(p.get("element").and_then(ParamValue::as_num), None);
/// ```
#[derive(Clone, Default, PartialEq)]
pub struct Params {
    values: HashMap<String, ParamValue>,
}

/// Keys in order: two equal `Params` print the same text, whatever order
/// their map iterates in (a reader keys its own types on the text).
impl std::fmt::Debug for Params {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(
                self.values
                    .iter()
                    .collect::<std::collections::BTreeMap<_, _>>(),
            )
            .finish()
    }
}

impl<K: Into<String>, V: Into<ParamValue>> FromIterator<(K, V)> for Params {
    fn from_iter<It: IntoIterator<Item = (K, V)>>(iter: It) -> Self {
        Params {
            values: iter
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }
}

impl Params {
    /// An empty bag.
    pub fn new() -> Self {
        Self::default()
    }

    /// The value of `key`, or `None`.
    pub fn get(&self, key: &str) -> Option<&ParamValue> {
        self.values.get(key)
    }

    /// Set (or replace) `key`.
    pub fn set(&mut self, key: &str, value: impl Into<ParamValue>) {
        self.values.insert(key.to_owned(), value.into());
    }

    /// Every `(key, value)`, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ParamValue)> + '_ {
        self.values.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Numeric energy parameters, excluding annotations and estimate provenance.
    pub(crate) fn iter_parameters(&self) -> impl Iterator<Item = (&str, f64)> + '_ {
        self.iter().filter_map(|(key, value)| {
            crate::ff::ir::is_parameter_column(key)
                .then(|| value.as_num().map(|number| (key, number)))
                .flatten()
        })
    }

    /// Whether `self` and `other` price alike: equal on every key that is a
    /// parameter (not an annotation column of the force-field section), a
    /// key one carries and the other lacks being a difference. The
    /// annotation keys (`desc`, `doi`, `smarts`, …) take no part. Exact, like
    /// `==`.
    pub fn same_parameters(&self, other: &Params) -> bool {
        use crate::ff::ir::is_parameter_column;
        use std::collections::BTreeMap;
        fn parameters(params: &Params) -> BTreeMap<&str, &ParamValue> {
            params
                .iter()
                .filter(|(key, _)| is_parameter_column(key))
                .collect()
        }
        parameters(self) == parameters(other)
    }

    /// The keys that are parameters: `self` without its annotation columns
    /// (`desc`, `doi`, `smarts`, …) and an estimate's provenance
    /// (`estimated`, `estimate_penalty`, …).
    pub fn parameters(&self) -> Params {
        use crate::ff::ir::is_parameter_column;
        self.iter()
            .filter(|(key, _)| is_parameter_column(key))
            .map(|(key, value)| (key, value.clone()))
            .collect()
    }

    /// Whether `self` and `other` hold the same keys on every side and the
    /// same values within `rel_tol`: equal strings, and numbers (array
    /// elements, of arrays of one shape) that are equal or differ by at most
    /// `rel_tol` of the larger magnitude.
    ///
    /// Not `==`: a re-definition is the same definition only when it is
    /// identical ([`Params`]). This is for comparing two **sources** of one
    /// parameter, which store it at their own precision — an AMBER prmtop
    /// keeps eight significant digits of a value `gaff.dat` writes in four.
    pub fn approx_eq(&self, other: &Params, rel_tol: f64) -> bool {
        let close = |a: f64, b: f64| {
            a == b
                || (a.is_finite()
                    && b.is_finite()
                    && (a - b).abs() <= rel_tol * a.abs().max(b.abs()))
        };
        self.values.len() == other.values.len()
            && self
                .values
                .iter()
                .all(|(key, value)| match (value, other.values.get(key)) {
                    (ParamValue::Num(a), Some(ParamValue::Num(b))) => close(*a, *b),
                    (ParamValue::Text(a), Some(ParamValue::Text(b))) => a == b,
                    (ParamValue::Array(a), Some(ParamValue::Array(b))) => {
                        a.shape() == b.shape() && a.iter().zip(b.iter()).all(|(&x, &y)| close(x, y))
                    }
                    _ => false,
                })
    }
}

/// The key a pair kernel finds the row of atom types `a` and `b` under in
/// [`StyleDefs::kernel_type_params`](crate::ff::forcefield::StyleDefs::kernel_type_params): [`TypeName::pair`] of the two in byte
/// order, so `(a, b)` and `(b, a)` share it (a self pair is `a` itself).
///
/// [`TypeName::pair`]: molrs::core::TypeName::pair
pub fn pair_key(a: &str, b: &str) -> Result<String, String> {
    let (i, j) = if a <= b { (a, b) } else { (b, a) };
    Ok(molrs::core::TypeName::pair(i, j)?.as_str().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerance_never_equates_a_finite_parameter_with_infinity() {
        for (a, b, equal) in [
            (10.0, f64::INFINITY, false),
            (f64::NEG_INFINITY, f64::INFINITY, false),
            (f64::INFINITY, f64::INFINITY, true),
            (f64::NAN, f64::NAN, false),
            (10.0, 10.000001, true),
        ] {
            let scalar = Params::from_iter([("cutoff", a)]);
            let other = Params::from_iter([("cutoff", b)]);
            assert_eq!(scalar.approx_eq(&other, 1e-6), equal, "{a} versus {b}");
            let array = Params::from_iter([("grid", ndarray::array![a].into_dyn())]);
            let other = Params::from_iter([("grid", ndarray::array![b].into_dyn())]);
            assert_eq!(array.approx_eq(&other, 1e-6), equal, "array {a} versus {b}");
        }
    }
}
