//! The fields of a LAMMPS data / dump line: tokens, numbers, and type
//! references (a numeric type or a type label) with their label maps.

use crate::io::invalid_data;
use molrs::op::{F, I};
use std::collections::HashMap;

/// Split a line on whitespace. Token count is small (≤ ~20); float parsing
/// dominates cost on large files, not the token `Vec`.
///
/// Inline comments (``# …``) are stripped first — LAMMPS data files from
/// VMD/TopoTools routinely write ``# element RES`` after atom rows and
/// ``# type-name`` after Masses; those must not pollute the column layout.
pub(crate) fn tokenize(line: &str) -> Vec<&str> {
    let code = line.split('#').next().unwrap_or(line);
    code.split_whitespace().collect()
}

pub(crate) fn parse_i(token: &str) -> std::io::Result<I> {
    token.parse::<I>().map_err(invalid_data)
}

pub(crate) fn parse_f(token: &str) -> std::io::Result<F> {
    token.parse::<F>().map_err(invalid_data)
}

/// Atom / bond / angle / … type as written in a data file.
#[derive(Debug, Clone)]
pub(crate) enum TypeRef {
    Id(I),
    Label(String),
}

impl TypeRef {
    pub(crate) fn parse(token: &str) -> Self {
        match token.parse::<I>() {
            Ok(id) => TypeRef::Id(id),
            Err(_) => TypeRef::Label(token.to_string()),
        }
    }

    pub(crate) fn resolve(&self, label_to_id: &HashMap<String, I>) -> I {
        match self {
            TypeRef::Id(id) => *id,
            TypeRef::Label(label) => label_to_id.get(label).copied().unwrap_or(1),
        }
    }
}

pub(crate) fn invert_type_labels(id_to_label: &HashMap<String, String>) -> HashMap<String, I> {
    let mut out = HashMap::with_capacity(id_to_label.len());
    for (id_str, label) in id_to_label {
        if let Ok(id) = id_str.parse::<I>() {
            out.insert(label.clone(), id);
        }
    }
    out
}

pub(crate) fn labels_to_meta(id_to_label: &HashMap<String, String>) -> Option<String> {
    if id_to_label.is_empty() {
        return None;
    }
    Some(
        id_to_label
            .iter()
            .map(|(id, label)| format!("{id}:{label}"))
            .collect::<Vec<_>>()
            .join(","),
    )
}
