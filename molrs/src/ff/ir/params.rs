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
