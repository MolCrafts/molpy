//! The vocabulary as an inspectable, serializable document: [`SchemaDocument`].

use super::{EndpointTarget, SCHEMA_BLOCKS, SCHEMA_COLUMNS};
use crate::core::UnitPreset;
use serde::{Deserialize, Serialize};

/// A canonical column, as data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ColumnDoc {
    /// Canonical key.
    pub key: String,
    /// Constant name (`ATOMI`) exported by the language bindings.
    pub const_name: String,
    /// Storage dtype, as [`DType::name`](crate::core::DType::name)
    /// spells it (`float`, `i64`, `uint`, `bool`, `string`, …).
    pub dtype: String,
    /// `scalar`, or `vec(n)`.
    pub shape: String,
    /// Physical dimension as lower-case preset names joined by `" * "`
    /// (`"length"`, `"charge * length"`), `"dimensionless"`, or empty for a
    /// column that is not a physical quantity.
    pub dimension: String,
    /// The dimension's unit in the `real` preset — the convention molrs's
    /// readers normalise to. Empty for a dimensionless column and for one
    /// that is not a physical quantity; [`dimension`](Self::dimension) tells
    /// the two apart.
    pub unit: String,
    /// One-line meaning.
    pub doc: String,
}

/// A canonical block, as data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlockDoc {
    /// Canonical block name.
    pub name: String,
    /// `node`, `relation(k)`, or `grid`.
    pub row_kind: String,
    /// Block the endpoints index into by default, for relation blocks.
    pub endpoint_target: Option<String>,
    /// Endpoint column keys, in position order.
    pub endpoint_columns: Vec<String>,
    /// The endpoint columns whose target is declared per block (`targets`)
    /// rather than defaulted (`members.atom`).
    pub declared_endpoints: Vec<String>,
    /// Columns that must be present.
    pub required: Vec<String>,
    /// Conventional but optional columns.
    pub optional: Vec<String>,
    /// Whether columns outside the vocabulary are admissible here.
    pub open: bool,
    /// One-line meaning.
    pub doc: String,
}

/// One constant projected for another language: a column, a block, or a meta key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NamedValue {
    /// Rust constant name (`"ATOMI"`, `"ATOMS"`, `"UNITS"`).
    pub const_name: String,
    /// The string that constant holds.
    pub value: String,
}

/// An ordered group of key strings (`COORDS`, `TOPOLOGY`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NamedGroup {
    /// Rust constant name (`"COORDS"`).
    pub const_name: String,
    /// Member keys, in group order.
    pub values: Vec<String>,
}

/// Every name the bindings export, projected from the compile-time tables.
///
/// Columns, groups, block names, block groups, and frame-meta keys. This is
/// the document `keysDocument()` hands to JavaScript; it is not a second
/// vocabulary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KeysDocument {
    /// Scalar column constants, vocabulary order.
    pub columns: Vec<NamedValue>,
    /// Column groups (`COORDS`, `ENDPOINTS`, …).
    pub groups: Vec<NamedGroup>,
    /// Scalar block-name constants.
    pub blocks: Vec<NamedValue>,
    /// Block groups (`TOPOLOGY`). Not themselves blocks.
    pub block_groups: Vec<NamedGroup>,
    /// Frame-meta keys. Not columns.
    pub meta: Vec<NamedValue>,
}

/// The whole vocabulary, owned and serializable.
///
/// The compile-time tables are the source of truth; this is an owned projection
/// of them. It exists so the schema is something a user can *look at* — print
/// it, publish it, diff two releases of it — from every binding, rather than a
/// rule that only exists inside the Rust type system.
///
/// Two runs produce byte-identical JSON — the tables are sorted and the
/// document preserves that order — so `diff`ing the artifact across releases
/// shows exactly what changed about the contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SchemaDocument {
    /// Stable identity of this schema.
    pub id: String,
    /// Every canonical column.
    pub columns: Vec<ColumnDoc>,
    /// Every canonical block.
    pub blocks: Vec<BlockDoc>,
}

/// Borrow the compile-time tables into an owned document.
pub fn document() -> SchemaDocument {
    let real = UnitPreset::real();
    SchemaDocument {
        id: "https://molcrafts.org/schema/frame".to_string(),
        columns: SCHEMA_COLUMNS
            .iter()
            .map(|c| ColumnDoc {
                key: c.key.to_string(),
                const_name: c.const_name.to_string(),
                dtype: c.dtype.name().to_string(),
                shape: c.shape.to_string(),
                dimension: c.dimension.to_string(),
                unit: c.dimension.unit_in(&real).unwrap_or_default(),
                doc: c.doc.to_string(),
            })
            .collect(),
        blocks: SCHEMA_BLOCKS
            .iter()
            .map(|b| BlockDoc {
                name: b.name.to_string(),
                row_kind: b.row_kind.to_string(),
                endpoint_target: b.endpoints.and_then(|e| {
                    e.columns.iter().find_map(|(_, target)| match target {
                        EndpointTarget::Block(block) => Some(block.to_string()),
                        EndpointTarget::Declared => None,
                    })
                }),
                endpoint_columns: b.endpoint_columns().iter().map(|s| s.to_string()).collect(),
                declared_endpoints: b
                    .endpoints
                    .map(|e| {
                        e.columns
                            .iter()
                            .filter(|(_, target)| *target == EndpointTarget::Declared)
                            .map(|(column, _)| column.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
                required: b.required.iter().map(|s| s.to_string()).collect(),
                optional: b.optional.iter().map(|s| s.to_string()).collect(),
                open: b.open,
                doc: b.doc.to_string(),
            })
            .collect(),
    }
}

impl SchemaDocument {
    /// Canonical JSON — stable across runs, so two releases can be diffed.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("schema document is always serializable")
    }

    /// The published Markdown tables.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "# Frame schema\n\n`{}`\n\n## Columns\n\n",
            self.id
        ));
        out.push_str(
            "| key | dtype | shape | dimension | unit (real) | meaning |\n\
             |---|---|---|---|---|---|\n",
        );
        for c in &self.columns {
            let dimension = if c.dimension.is_empty() {
                "—"
            } else {
                &c.dimension
            };
            let unit = if c.unit.is_empty() { "—" } else { &c.unit };
            out.push_str(&format!(
                "| `{}` | {} | {} | {} | {} | {} |\n",
                c.key, c.dtype, c.shape, dimension, unit, c.doc
            ));
        }
        out.push_str("\n## Blocks\n\n");
        out.push_str(
            "| block | rows | endpoints → | required | meaning |\n|---|---|---|---|---|\n",
        );
        for b in &self.blocks {
            let defaulted: Vec<&str> = b
                .endpoint_columns
                .iter()
                .filter(|c| !b.declared_endpoints.contains(c))
                .map(String::as_str)
                .collect();
            let mut ep = match &b.endpoint_target {
                Some(t) => format!("`{}` → `{}`", defaulted.join("`, `"), t),
                None => "—".to_string(),
            };
            if !b.declared_endpoints.is_empty() {
                ep.push_str(&format!(
                    "; `{}` → declared",
                    b.declared_endpoints.join("`, `")
                ));
            }
            let req = if b.required.is_empty() {
                "—".to_string()
            } else {
                format!("`{}`", b.required.join("`, `"))
            };
            out.push_str(&format!(
                "| `{}` | {} | {} | {} | {} |\n",
                b.name, b.row_kind, ep, req, b.doc
            ));
        }
        out
    }
}

impl std::fmt::Display for SchemaDocument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_markdown())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_covers_every_table_entry() {
        let d = document();
        assert_eq!(d.columns.len(), SCHEMA_COLUMNS.len());
        assert_eq!(d.blocks.len(), SCHEMA_BLOCKS.len());
    }

    #[test]
    fn json_round_trips_losslessly() {
        // The published artifact is what downstream tooling and the other
        // bindings read; if it cannot come back unchanged it is not a contract.
        let d = document();
        let back: SchemaDocument = serde_json::from_str(&d.to_json()).expect("valid json");
        assert_eq!(d, back);
    }

    #[test]
    fn json_is_stable_across_runs() {
        assert_eq!(document().to_json(), document().to_json());
    }

    fn column_doc(key: &str) -> ColumnDoc {
        document()
            .columns
            .into_iter()
            .find(|c| c.key == key)
            .unwrap_or_else(|| panic!("`{key}` missing from the document"))
    }

    #[test]
    fn x_has_dimension_length_and_real_unit_angstrom() {
        let x = column_doc("x");
        assert_eq!(x.dimension, "length");
        assert_eq!(x.unit, "angstrom");
    }

    #[test]
    fn mux_has_dimension_charge_times_length_and_its_real_unit() {
        let mux = column_doc("mux");
        assert_eq!(mux.dimension, "charge * length");
        assert_eq!(mux.unit, "elementary_charge * angstrom");
    }

    #[test]
    fn markdown_header_names_dimension_and_real_unit() {
        assert!(
            document()
                .to_markdown()
                .contains("| key | dtype | shape | dimension | unit (real) | meaning |")
        );
    }

    #[test]
    fn markdown_names_every_column_and_block() {
        let md = document().to_markdown();
        for c in SCHEMA_COLUMNS {
            assert!(md.contains(&format!("`{}`", c.key)), "{} missing", c.key);
        }
        for b in SCHEMA_BLOCKS {
            assert!(md.contains(&format!("`{}`", b.name)), "{} missing", b.name);
        }
    }
}
