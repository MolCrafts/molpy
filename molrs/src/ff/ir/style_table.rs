//! The columns of a style's table: one row per force-field type, named by
//! `name`, keyed by its endpoint columns, annotated by string columns, and
//! carrying one column per parameter (molrec `docs/spec/forcefield.md`).
//!
//! The record's `forcefield` section ([`crate::io::mrec::ForceFieldSection`])
//! validates its tables against this vocabulary; the force-field model reads
//! its rows by it.

/// The endpoint columns of a style table, in position order.
pub const ENDPOINT_COLUMNS: [&str; 5] = ["itom", "jtom", "ktom", "ltom", "mtom"];

/// The annotation columns of a style table: `string`, nullable.
pub const ANNOTATION_COLUMNS: [&str; 7] = [
    "class",
    "element",
    "smarts",
    "smirks",
    "overrides",
    "desc",
    "doi",
];

/// The provenance columns of an estimated term (the GAFF and OPLS
/// estimators' [`Provenance`](crate::ff::typifier::Provenance)): `estimated`
/// (the flag, `1.0`), `estimate_penalty` (number), `estimate_method` and
/// `estimate_analog` (text). They say how a row's numbers were found, not
/// what they are: like an annotation, they price nothing.
pub const PROVENANCE_COLUMNS: [&str; 4] = [
    "estimated",
    "estimate_penalty",
    "estimate_method",
    "estimate_analog",
];

/// The array parameter whose shape the chapter pins: a `cmap` row's
/// correction table, `f64[T, N, N]` — an `N × N` grid over the two
/// dihedrals, every row of the table sharing `N`. Any other parameter may be
/// an array too (`f64[T, S…]`).
pub const CMAP_GRID: &str = "grid";

/// Whether a style-table column prices nothing: an annotation
/// ([`ANNOTATION_COLUMNS`]) or an estimate's provenance
/// ([`PROVENANCE_COLUMNS`]).
///
/// The one rule every consumer of a row applies to these keys: an engine
/// writer passes them over (a format with a place for them, the frcmod
/// row's comment, writes them there), and [`ForceField::merge`] compares
/// rows without them. A key that is neither, and that the writer's format
/// has no column for, is still refused.
///
/// [`ForceField::merge`]: crate::ff::forcefield::ForceField::merge
pub fn is_unpriced_column(column: &str) -> bool {
    ANNOTATION_COLUMNS.contains(&column) || PROVENANCE_COLUMNS.contains(&column)
}

/// Whether a style-table column is a parameter: not `name`, an endpoint, or
/// a column that prices nothing ([`is_unpriced_column`]). Rows restating one
/// pair are compared on these alone.
pub fn is_parameter_column(column: &str) -> bool {
    column != "name" && !ENDPOINT_COLUMNS.contains(&column) && !is_unpriced_column(column)
}
