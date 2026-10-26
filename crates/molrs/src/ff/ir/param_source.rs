//! [`ParamSource`]: where a style's kernel reads its parameters from, a
//! declaration of its [`StyleSpec`](crate::ff::ir::StyleSpec).

/// Where a kernel's parameters come from — the question the empty-type-params
/// guard must ask before it rejects a style with no type rows.
///
/// A kernel constructor that binds its type-params as `_tp` (i.e. resolves
/// nothing from them) **is not a table-driven style**, and must say so by being
/// registered [`PerInstance`](ParamSource::PerInstance).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamSource {
    /// Parameters come from the style's type-definition rows (the `tp` slice).
    /// A style with no rows resolves nothing, and is an error.
    TypeRows,
    /// Parameters are resolved per interaction by the typifier and baked into
    /// [`Frame`](molrs::core::Frame) columns; `tp` is ignored and may legitimately be empty.
    PerInstance,
}
