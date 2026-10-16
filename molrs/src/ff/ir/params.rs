//! [`Params`]: the key-value parameters of a style or a type, and the key
//! ([`pair_key`]) a pair kernel finds a type's row under.

use std::collections::HashMap;

use ndarray::ArrayD;

/// Key-value parameter bag for type definitions.
///
/// Holds numeric params (`k`, `r0`, the numeric type `id`, …), string params
/// (`element`, or any string metadata carried by convention as a keyword
/// param) and array params (an N-dimensional `f64` array, e.g. a CMAP
/// correction's `grid`), each on its own side. Energy kernels read the numeric
/// and array sides; the string side preserves I/O metadata across the
/// boundary.
///
/// Equality is exact on every side (the same keys, `f64` values equal under
/// `==` with no tolerance, equal strings, arrays of one shape with every
/// element equal under `==`): it decides whether a re-definition is the same
/// definition, which is a question of identity, not closeness.
#[derive(Clone, Default, PartialEq)]
pub struct Params {
    inner: HashMap<String, f64>,
    strings: HashMap<String, String>,
    arrays: HashMap<String, ArrayD<f64>>,
}

/// Keys in order: two equal `Params` print the same text, whatever order
/// their maps iterate in (a reader keys its own types on the text).
impl std::fmt::Debug for Params {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Params")
            .field(
                "inner",
                &self
                    .inner
                    .iter()
                    .collect::<std::collections::BTreeMap<_, _>>(),
            )
            .field(
                "strings",
                &self
                    .strings
                    .iter()
                    .collect::<std::collections::BTreeMap<_, _>>(),
            )
            .field(
                "arrays",
                &self
                    .arrays
                    .iter()
                    .collect::<std::collections::BTreeMap<_, _>>(),
            )
            .finish()
    }
}

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_pairs(pairs: &[(&str, f64)]) -> Self {
        let mut out = Self::new();
        for &(k, v) in pairs {
            out.set(k, v);
        }
        out
    }

    pub fn get(&self, key: &str) -> Option<f64> {
        self.inner.get(key).copied()
    }

    pub fn set(&mut self, key: &str, value: f64) {
        self.inner.insert(key.to_owned(), value);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, f64)> + '_ {
        self.inner.iter().map(|(k, v)| (k.as_str(), *v))
    }

    // -- string params (element, and other string metadata by convention) --

    pub fn set_str(&mut self, key: &str, value: &str) {
        self.strings.insert(key.to_owned(), value.to_owned());
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.strings.get(key).map(String::as_str)
    }

    pub fn iter_strings(&self) -> impl Iterator<Item = (&str, &str)> + '_ {
        self.strings.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    // -- array params (a CMAP `grid`, and any other N-D parameter) --

    /// Set (or replace) the array param `key`.
    pub fn set_array(&mut self, key: &str, value: ArrayD<f64>) {
        self.arrays.insert(key.to_owned(), value);
    }

    /// The array param `key`, or `None`.
    pub fn get_array(&self, key: &str) -> Option<&ArrayD<f64>> {
        self.arrays.get(key)
    }

    pub fn iter_arrays(&self) -> impl Iterator<Item = (&str, &ArrayD<f64>)> + '_ {
        self.arrays.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Whether `self` and `other` price alike: equal on every key that is a
    /// parameter (not an annotation column of the force-field section),
    /// numeric, string and array, a key one carries and the other lacks being
    /// a difference. The annotation keys (`desc`, `doi`, `smarts`, …) take no
    /// part. Exact, like `==`.
    pub fn same_parameters(&self, other: &Params) -> bool {
        use crate::ff::ir::is_parameter_column;
        use std::collections::BTreeMap;
        fn parameters<V>(map: &HashMap<String, V>) -> BTreeMap<&str, &V> {
            map.iter()
                .filter(|(key, _)| is_parameter_column(key))
                .map(|(key, value)| (key.as_str(), value))
                .collect()
        }
        parameters(&self.inner) == parameters(&other.inner)
            && parameters(&self.strings) == parameters(&other.strings)
            && parameters(&self.arrays) == parameters(&other.arrays)
    }

    /// The keys that are parameters: `self` without its annotation columns
    /// (`desc`, `doi`, `smarts`, …).
    pub fn parameters(&self) -> Params {
        use crate::ff::ir::is_parameter_column;
        fn kept<V: Clone>(map: &HashMap<String, V>) -> HashMap<String, V> {
            map.iter()
                .filter(|(key, _)| is_parameter_column(key))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        }
        Params {
            inner: kept(&self.inner),
            strings: kept(&self.strings),
            arrays: kept(&self.arrays),
        }
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
        let close = |a: f64, b: f64| a == b || (a - b).abs() <= rel_tol * a.abs().max(b.abs());
        fn same_keys<V>(a: &HashMap<String, V>, b: &HashMap<String, V>) -> bool {
            a.len() == b.len() && a.keys().all(|k| b.contains_key(k))
        }
        same_keys(&self.inner, &other.inner)
            && self.inner.iter().all(|(k, &v)| close(v, other.inner[k]))
            && self.strings == other.strings
            && same_keys(&self.arrays, &other.arrays)
            && self.arrays.iter().all(|(k, a)| {
                let b = &other.arrays[k];
                a.shape() == b.shape() && a.iter().zip(b.iter()).all(|(&x, &y)| close(x, y))
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
