//! [`ForceField`] ↔ [`ForceFieldSection`]: a force field as the `forcefield`
//! section of a record (molrec `docs/spec/forcefield.md`).
//!
//! | molrs | section |
//! |---|---|
//! | [`ForceField::name`] | `document.name` |
//! | [`ForceField::units`] (declared, or the `real` default) | `document.units`: `{preset, length, energy, angle, charge, mass}` from the preset's table (`lj`: `{preset}` alone) |
//! | [`ForceField::declared_special_bonds`] | `document.special_bonds`; absent ⇔ `None` |
//! | [`ForceField::styles`], in order | `document.styles`, in order |
//! | [`Style::category`], [`Style::name`] | `category`, `style` |
//! | a category no registry declares ([`StyleDefs::Relation`]), its arity | the same `category`, its arity the endpoint columns' count |
//! | [`Style::params`] numeric / string | `params` numbers / strings |
//! | the string style params `expression`, `endpoint_key` | the entry fields of those names, `expression` byte for byte |
//! | a style the force-field IR registry holds as a **custom** style, with no `expression` of its own | the registry's `expression`, so a process that registered nothing prices it (a built-in style writes none) |
//! | [`Style::type_rows`], in definition order | the rows of the table at [`style_block_name`] |
//! | row name, endpoints (a pair always two) | `name`, `itom`…`mtom` |
//! | a pair self row / an explicit cross row (NBFIX) | a pair row with `itom == jtom` / `itom != jtom` |
//! | row [`Params`] numeric / string | `f64` / `string` columns, after `name` and the endpoints, keys sorted bytewise |
//! | a row array param of shape `S` (every row carrying the key at one shape) | an `f64[T, S…]` column, null where a row lacks it ([`ForceFieldSection::validate`]: finite, every axis ≥ 1; a `cmap` row's `grid` is `S = [N, N]`, `N ≥ 2`) |
//! | a numeric param under a canonical non-`f64` key (`atomic_number`, `id`, …) | a column of that key's dtype; [`ForceFieldSection::to_forcefield`] reads it back as `f64` |
//! | a key a row does not carry | null in that row |
//!
//! `from_forcefield(ff).to_forcefield()` is `ff` with its units declared,
//! and `from_forcefield(s.to_forcefield())` is `s` for every section
//! `from_forcefield` produced. Units are never converted: a section in a unit system that is no
//! molrs preset is refused, not rescaled.
//!
//! [`Style::category`]: crate::ff::forcefield::Style::category
//! [`Style::name`]: crate::ff::forcefield::Style::name
//! [`Style::params`]: crate::ff::forcefield::Style::params
//! [`Style::type_rows`]: crate::ff::forcefield::Style::type_rows
//! [`StyleDefs::Relation`]: crate::ff::forcefield::StyleDefs::Relation

use crate::ff::forcefield::Category;
use std::collections::{BTreeMap, BTreeSet};

use indexmap::IndexMap;
use ndarray::{ArrayD, Axis};
use serde_json::{Map as JsonMap, Value as JsonValue, json};

use super::forcefield_section::{
    EndpointKey, ForceFieldSection, SECTION_PRESETS, UNIT_QUANTITIES, style_block_name, unit_preset,
};
use crate::core::{Block, Column, DType};
use crate::ff::forcefield::{ForceField, Style};
use crate::ff::ir::ENDPOINT_COLUMNS;
use crate::ff::ir::{ParamValue, Params, SpecialBonds};
use crate::ff::style_registry::Registry;

/// The string style params that are entry fields of `document.styles`.
const ENTRY_FIELDS: [&str; 2] = ["expression", "endpoint_key"];

/// The quantities [`ForceFieldSection::from_forcefield`] states beside a preset.
const STATED_QUANTITIES: [&str; 5] = ["length", "energy", "angle", "charge", "mass"];

/// One value of a row's or a style's params.
#[derive(Debug, Clone, PartialEq)]
enum Value<'a> {
    Number(f64),
    Text(&'a str),
    Array(&'a ArrayD<f64>),
}

impl Value<'_> {
    fn kind(&self) -> &'static str {
        match self {
            Value::Number(_) => "a number",
            Value::Text(_) => "a string",
            Value::Array(_) => "an array",
        }
    }
}

/// The params of `params` as one sorted map.
fn merged(params: &Params) -> BTreeMap<&str, Value<'_>> {
    params
        .iter()
        .map(|(key, value)| {
            let value = match value {
                ParamValue::Num(n) => Value::Number(*n),
                ParamValue::Text(t) => Value::Text(t),
                ParamValue::Array(a) => Value::Array(a),
            };
            (key, value)
        })
        .collect()
}

fn units_document(units: &str) -> Result<JsonValue, String> {
    let table = unit_preset(units).ok_or_else(|| {
        format!("units {units:?} is no unit preset the forcefield section can state")
    })?;
    let mut out = JsonMap::new();
    out.insert("preset".into(), units.into());
    if units != "lj" {
        for (quantity, unit) in UNIT_QUANTITIES.iter().zip(table) {
            if let Some(unit) = unit
                && STATED_QUANTITIES.contains(quantity)
            {
                out.insert((*quantity).into(), unit.into());
            }
        }
    }
    Ok(JsonValue::Object(out))
}

/// The `expression` `style` persists with: its own, else — a custom style
/// `registry` holds (not a sealed built-in) — the registry's, so a process
/// that registered nothing can price what it reads (protocol D16).
fn persisted_expression<'a>(style: &'a Style, registry: &'a Registry) -> Option<&'a str> {
    style
        .params()
        .get("expression")
        .and_then(|v| v.as_text())
        .or_else(|| {
            if registry.is_sealed(style.category(), style.name()) {
                return None;
            }
            registry
                .style(style.category(), style.name())
                .and_then(|(spec, _)| spec.expression.as_deref())
        })
}

fn section_style(style: &Style, registry: &Registry) -> Result<JsonValue, String> {
    let what = || format!("{}/{} style params", style.category(), style.name());
    let mut entry = JsonMap::new();
    entry.insert("category".into(), style.category().into());
    entry.insert("style".into(), style.name().into());
    let mut params = JsonMap::new();
    let mut fields = JsonMap::new();
    for (key, value) in merged(style.params()) {
        match value {
            Value::Text(text) if ENTRY_FIELDS.contains(&key) => {
                fields.insert(key.into(), text.into());
            }
            Value::Number(_) if ENTRY_FIELDS.contains(&key) => {
                return Err(format!("{}: {key:?} is a string", what()));
            }
            Value::Number(n) => {
                let number = serde_json::Number::from_f64(n)
                    .ok_or_else(|| format!("{}: {key:?} = {n} is not finite", what()))?;
                params.insert(key.into(), JsonValue::Number(number));
            }
            Value::Text(text) => {
                params.insert(key.into(), text.into());
            }
            Value::Array(_) => {
                return Err(format!(
                    "{}: {key:?} is an array; a style param is a number or a string",
                    what()
                ));
            }
        }
    }
    if !params.is_empty() {
        entry.insert("params".into(), JsonValue::Object(params));
    }
    fields.remove("expression");
    if let Some(expression) = persisted_expression(style, registry) {
        entry.insert("expression".into(), expression.into());
    }
    if let Some(key) = fields.remove("endpoint_key")
        && key != "type"
    {
        entry.insert("endpoint_key".into(), key);
    }
    Ok(JsonValue::Object(entry))
}

fn strings(values: Vec<String>) -> Column {
    let n = values.len();
    Column::from(ArrayD::from_shape_vec(vec![n], values).expect("one value per row"))
}

/// The `f64[T, S…]` column of an array param: every row's array at one shape
/// `S`, a row without one filled with zeros (and null).
fn array_column(
    key: &str,
    cells: &[Option<Value<'_>>],
    what: &dyn Fn() -> String,
) -> Result<Column, String> {
    let arrays: Vec<Option<&ArrayD<f64>>> = cells
        .iter()
        .map(|c| match c {
            Some(Value::Array(a)) => Some(*a),
            _ => None,
        })
        .collect();
    let shape = arrays
        .iter()
        .flatten()
        .next()
        .expect("an array column has an array")
        .shape()
        .to_vec();
    if shape.is_empty() {
        // An `f64[T]` column reads back as numbers, not as 0-d arrays.
        return Err(format!(
            "{}: param {key:?} is a 0-d array; a section holds a scalar as a number",
            what()
        ));
    }
    if let Some(other) = arrays.iter().flatten().find(|a| a.shape() != shape) {
        return Err(format!(
            "{}: param {key:?} is an array of shape {shape:?} in one row and {:?} in \
             another; a column holds one shape",
            what(),
            other.shape()
        ));
    }
    let size: usize = shape.iter().product();
    let mut values = Vec::with_capacity(arrays.len() * size);
    for array in &arrays {
        match array {
            Some(a) => values.extend(a.iter().copied()),
            None => values.extend(std::iter::repeat_n(0.0, size)),
        }
    }
    let mut full = vec![arrays.len()];
    full.extend(&shape);
    Ok(Column::from(
        ArrayD::from_shape_vec(full, values).expect("one array per row"),
    ))
}

/// The column of one param across the rows, at the dtype it is stored at.
fn param_column(
    key: &str,
    cells: &[Option<Value<'_>>],
    what: &dyn Fn() -> String,
) -> Result<(Column, Vec<bool>), String> {
    let validity: Vec<bool> = cells.iter().map(Option::is_some).collect();
    let mut kinds: Vec<&str> = cells.iter().flatten().map(Value::kind).collect();
    kinds.sort_unstable();
    kinds.dedup();
    if let [first, second, ..] = kinds[..] {
        return Err(format!(
            "{}: param {key:?} is {first} in one row and {second} in another",
            what()
        ));
    }
    let text = kinds == ["a string"];
    let canonical = molrs::core::schema::column(key).map(|spec| spec.dtype);
    let column = if kinds == ["an array"] {
        if canonical.is_some() {
            return Err(format!(
                "{}: param {key:?} is an array, and the canonical key {key:?} is a scalar",
                what()
            ));
        }
        array_column(key, cells, what)?
    } else if text {
        if canonical.is_some_and(|dtype| dtype != DType::String) {
            return Err(format!(
                "{}: param {key:?} is a string, and the canonical key {key:?} is not",
                what()
            ));
        }
        strings(
            cells
                .iter()
                .map(|c| match c {
                    Some(Value::Text(t)) => (*t).to_owned(),
                    _ => String::new(),
                })
                .collect(),
        )
    } else {
        let numbers: Vec<f64> = cells
            .iter()
            .map(|c| match c {
                Some(Value::Number(n)) => *n,
                _ => 0.0,
            })
            .collect();
        let shape = vec![numbers.len()];
        match canonical {
            None | Some(DType::Float) => {
                Column::from(ArrayD::from_shape_vec(shape, numbers).expect("one value per row"))
            }
            Some(DType::Uint) => {
                let mut values = Vec::with_capacity(numbers.len());
                for (n, valid) in numbers.iter().zip(&validity) {
                    if *valid && !(n.fract() == 0.0 && *n >= 0.0 && *n < u64::MAX as f64) {
                        return Err(format!(
                            "{}: param {key:?} = {n} is no non-negative integer, and the \
                             canonical key {key:?} is u64",
                            what()
                        ));
                    }
                    values.push(*n as u64);
                }
                Column::from(ArrayD::from_shape_vec(shape, values).expect("one per row"))
            }
            Some(other) => {
                return Err(format!(
                    "{}: param {key:?} is a number, and the canonical key {key:?} is {}",
                    what(),
                    other.name()
                ));
            }
        }
    };
    Ok((column, validity))
}

fn style_table(style: &Style) -> Result<Block, String> {
    let what = || format!("{}/{}", style.category(), style.name());
    let rows = style.type_rows();
    let arity = style.arity();
    let mut block = Block::new();
    let column_err = |e: molrs::core::BlockError| format!("{}: {e}", what());
    block
        .insert(
            "name",
            strings(rows.iter().map(|(name, _, _)| (*name).to_owned()).collect()),
            None,
        )
        .map_err(column_err)?;
    for (position, endpoint) in ENDPOINT_COLUMNS[..arity].iter().enumerate() {
        let values = rows
            .iter()
            .map(|(_, ends, _)| ends[position].to_owned())
            .collect();
        block
            .insert(*endpoint, strings(values), None)
            .map_err(column_err)?;
    }

    let mut per_row = Vec::with_capacity(rows.len());
    let mut keys = BTreeSet::new();
    for (_, _, params) in &rows {
        let row = merged(params);
        keys.extend(row.keys().copied());
        per_row.push(row);
    }
    for key in keys {
        if key == "name" || ENDPOINT_COLUMNS.contains(&key) {
            return Err(format!(
                "{}: a param named {key:?} would shadow the table's own column",
                what()
            ));
        }
        let cells: Vec<Option<Value<'_>>> = per_row.iter().map(|r| r.get(key).cloned()).collect();
        let (column, validity) = param_column(key, &cells, &what)?;
        block.insert(key, column, None).map_err(column_err)?;
        block.set_validity(key, validity).map_err(column_err)?;
    }
    Ok(block)
}

/// The preset a section's `units` name, whether by `preset` or by stating
/// exactly one preset's length and energy (and nothing that disagrees).
fn molrs_units(units: &JsonMap<String, JsonValue>) -> Result<String, String> {
    if let Some(preset) = units.get("preset").and_then(JsonValue::as_str) {
        return Ok(preset.to_owned());
    }
    let stated: Vec<(usize, &str)> = UNIT_QUANTITIES
        .iter()
        .enumerate()
        .filter_map(|(i, q)| units.get(*q).and_then(JsonValue::as_str).map(|u| (i, u)))
        .collect();
    let states = |q: &str| stated.iter().any(|(i, _)| UNIT_QUANTITIES[*i] == q);
    if states("length") && states("energy") {
        let matching: Vec<&str> = SECTION_PRESETS
            .into_iter()
            .filter(|&name| name != "lj")
            .filter(|name| {
                let table = unit_preset(name).expect("a preset");
                stated.iter().all(|(i, unit)| table[*i] == Some(*unit))
            })
            .collect();
        if let [only] = matching[..] {
            return Ok(only.to_owned());
        }
    }
    Err(format!(
        "units {} are no unit preset; molrs does not convert a force field's units",
        JsonValue::Object(units.clone())
    ))
}

fn special_bonds(value: &JsonValue) -> SpecialBonds {
    let weights = |key: &str| {
        let w = value[key].as_array().expect("validated: three weights");
        [0, 1, 2].map(|i| w[i].as_f64().expect("validated: finite weights"))
    };
    SpecialBonds {
        lj: weights("lj"),
        coul: weights("coul"),
    }
}

/// The params of row `row` of `table`: every column but `name` and the
/// endpoints, where not null; a column with trailing axes is an array param.
fn row_params(table: &Block, row: usize) -> Params {
    let mut params = Params::new();
    for (key, column) in table.iter() {
        if key == "name" || ENDPOINT_COLUMNS.contains(&key) {
            continue;
        }
        if table.validity(key).is_some_and(|mask| !mask[row]) {
            continue;
        }
        if let Some(values) = column.get::<String>() {
            params.set(key, values[[row]].as_str());
        } else if let Some(values) = column.get::<f64>() {
            if values.ndim() > 1 {
                params.set(key, values.index_axis(Axis(0), row).to_owned());
            } else {
                params.set(key, values[[row]]);
            }
        } else if let Some(values) = column.get::<u64>() {
            params.set(key, values[[row]] as f64);
        } else if let Some(values) = column.get::<i32>() {
            params.set(key, f64::from(values[[row]]));
        } else if let Some(values) = column.get::<i64>() {
            params.set(key, values[[row]] as f64);
        }
    }
    params
}

impl ForceFieldSection {
    /// `ff` as a record's `forcefield` section (see the module docs for the
    /// mapping).
    ///
    /// A style's own `expression` is written byte for byte. A custom style
    /// the process-wide force-field IR registry holds, with none of its own,
    /// is written with the registry's `expression`, so a process that
    /// registered nothing reads a style it can price; a built-in style is
    /// written with none.
    ///
    /// # Errors
    ///
    /// An `Err` naming the style and key when the force field has no section
    /// form: units that are no preset; a non-finite or array style param; a
    /// param key of one kind (number, string, array) in one row and another
    /// in another, on two sides of one [`Params`], or `name` / an endpoint
    /// column; an array param at two shapes; a param under a canonical key at
    /// another dtype (a non-integral `atomic_number`, a numeric `element`, an
    /// array); or anything [`ForceFieldSection::validate`] refuses — a
    /// non-finite array value, or a `cmap` `grid` that is not square, among
    /// it.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::ff::forcefield::ForceField;
    /// use molrs::ff::ir::Params;
    /// use molrs::io::mrec::ForceFieldSection;
    ///
    /// let mut ff = ForceField::new("example");
    /// ff.def_style("bond", "harmonic", Params::new())
    ///     .unwrap()
    ///     .def_type("A-B", &["A", "B"], Params::from_iter([("k", 300.0), ("r0", 1.5)]))
    ///     .unwrap();
    /// let section = ForceFieldSection::from_forcefield(&ff).unwrap();
    /// assert!(section.table("bond", "harmonic").is_some());
    /// let back = section.to_forcefield().unwrap();
    /// assert_eq!(back.get_bondtypes()[0].params.get("k").and_then(|v| v.as_num()), Some(300.0));
    /// ```
    pub fn from_forcefield(ff: &ForceField) -> Result<ForceFieldSection, String> {
        ff.read_registry(|registry| Self::section_of(ff, registry))
    }

    /// The section of `ff`, a style without its own expression written with
    /// `registry`'s.
    fn section_of(ff: &ForceField, registry: &Registry) -> Result<ForceFieldSection, String> {
        let mut document = JsonMap::new();
        document.insert("name".into(), ff.name.clone().into());
        document.insert("units".into(), units_document(ff.units())?);
        if let Some(sb) = ff.declared_special_bonds() {
            document.insert(
                "special_bonds".into(),
                json!({"lj": sb.lj.to_vec(), "coul": sb.coul.to_vec()}),
            );
        }
        let mut entries = Vec::with_capacity(ff.styles().len());
        let mut tables = IndexMap::with_capacity(ff.styles().len());
        for style in ff.styles() {
            entries.push(section_style(style, registry)?);
            tables.insert(
                style_block_name(style.category(), style.name()),
                style_table(style)?,
            );
        }
        document.insert("styles".into(), JsonValue::Array(entries));
        let section = ForceFieldSection { document, tables };
        section.validate().map_err(|e| e.to_string())?;
        Ok(section)
    }

    /// The force field this section describes (see the module docs for the
    /// mapping).
    ///
    /// The section's units become the force field's declared units; they are
    /// not converted. A table no style names and a document key this build
    /// does not know are unknown content the force field has no place for:
    /// they stay with the section.
    ///
    /// A style's `expression` is kept byte for byte, whether or not anything
    /// registered the style. A category beyond molrs's seven is kept as a
    /// [`StyleDefs::Relation`](crate::ff::forcefield::StyleDefs::Relation): with the arity
    /// the force-field IR registry declares for it, or — a category nothing
    /// declares — the count of its table's endpoint columns. Reading never
    /// evaluates an `expression`; compiling a style nothing can price is
    /// where it is refused.
    ///
    /// # Errors
    ///
    /// An `Err` when the section fails [`ForceFieldSection::validate`], or
    /// has no [`ForceField`] form: a registered category whose table names
    /// another number of endpoints; units that are no preset (stated by
    /// `preset`, or by one preset's own length and energy); a `smirks`-keyed
    /// style; a `class`-keyed style whose endpoints are not all atom-type
    /// names.
    pub fn to_forcefield(&self) -> Result<ForceField, String> {
        let section = self;
        section.validate().map_err(|e| e.to_string())?;
        let doc = &section.document;
        let mut ff = ForceField::new(section.name().expect("validated: a name"));
        let units = doc["units"].as_object().expect("validated: units");
        ff.set_units(&molrs_units(units)?);
        if let Some(sb) = doc.get("special_bonds") {
            ff.set_special_bonds(special_bonds(sb));
        }

        let styles = section.styles().map_err(|e| e.to_string())?;
        let atom_names: BTreeSet<&str> = styles
            .iter()
            .filter(|s| s.category == "atom")
            .filter_map(|s| section.tables.get(&s.block_name()))
            .filter_map(|t| t.get("name").and_then(Column::get::<String>))
            .flat_map(|names| names.iter().map(String::as_str))
            .collect();
        for entry in &styles {
            let what = format!("{}/{}", entry.category, entry.style);
            let mut params = Params::new();
            for (key, value) in entry.params.into_iter().flatten() {
                match value {
                    JsonValue::String(text) => params.set(key, text.as_str()),
                    number => params.set(key, number.as_f64().expect("validated: a number")),
                }
            }
            if let Some(expression) = entry.expression {
                params.set("expression", expression);
            }
            let table = &section.tables[&entry.block_name()];
            match entry.endpoint_key {
                EndpointKey::Type => {}
                EndpointKey::Smirks => {
                    return Err(format!(
                        "{what} is smirks-keyed; a molrs force field keys rows by atom type"
                    ));
                }
                EndpointKey::Class => {
                    for endpoint in ENDPOINT_COLUMNS {
                        let Some(values) = table.get(endpoint).and_then(Column::get::<String>)
                        else {
                            continue;
                        };
                        if let Some(class) = values
                            .iter()
                            .find(|v| !v.is_empty() && !atom_names.contains(v.as_str()))
                        {
                            return Err(format!(
                                "{what} is class-keyed and its endpoint {class:?} is no atom \
                                 type; a molrs force field keys rows by atom type"
                            ));
                        }
                    }
                    params.set("endpoint_key", EndpointKey::Class.as_str());
                }
            }
            let endpoint_columns: Vec<&ArrayD<String>> = ENDPOINT_COLUMNS
                .iter()
                .filter_map(|c| table.get(c).and_then(Column::get::<String>))
                .collect();
            let style = ff
                .def_style(
                    Category::new(entry.category).arity(endpoint_columns.len()),
                    entry.style,
                    params,
                )
                .map_err(|e| format!("{what}: {e}"))?;
            let names = table
                .get("name")
                .and_then(Column::get::<String>)
                .expect("validated: a name column");
            for (row, name) in names.iter().enumerate() {
                let endpoints: Vec<&str> = endpoint_columns
                    .iter()
                    .map(|column| column[[row]].as_str())
                    .collect();
                style
                    .def_type(name, &endpoints, row_params(table, row))
                    .map_err(|e| format!("{what}: {e}"))?;
            }
        }
        Ok(ff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::forcefield::Category;
    use crate::ff::forcefield::tests::assert_same_definitions;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::typifier::Typifier;
    use crate::io::reader::ForceFieldReader;

    /// `ff` with its units declared: what `from_forcefield(ff).to_forcefield()`
    /// gives back.
    fn declared(ff: &ForceField) -> ForceField {
        let mut out = ff.clone();
        out.set_units(ff.units());
        out
    }

    fn same_section(a: &ForceFieldSection, b: &ForceFieldSection, what: &str) {
        assert_eq!(a.document, b.document, "{what}: document");
        assert_eq!(
            a.tables.keys().collect::<Vec<_>>(),
            b.tables.keys().collect::<Vec<_>>(),
            "{what}: tables"
        );
        for (name, x) in &a.tables {
            let y = &b.tables[name];
            assert_eq!(format!("{x:?}"), format!("{y:?}"), "{what}: {name} layout");
            for (key, column) in x.iter() {
                let other = y.get(key).unwrap();
                assert_eq!(x.validity(key), y.validity(key), "{what}: {name}.{key}");
                match (column.get::<f64>(), other.get::<f64>()) {
                    (Some(p), Some(q)) => {
                        let bits =
                            |a: &ArrayD<f64>| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                        assert_eq!(bits(p), bits(q), "{what}: {name}.{key}");
                    }
                    _ => assert_eq!(
                        (column.get::<String>(), column.get::<u64>()),
                        (other.get::<String>(), other.get::<u64>()),
                        "{what}: {name}.{key}"
                    ),
                }
            }
        }
    }

    /// The round-trip property, both ways: `to_forcefield ∘ from_forcefield` is
    /// the identity up to declared units, and `from_forcefield ∘ to_forcefield` the
    /// identity on the section. Through a `*.mrec` store too, where it can be
    /// written.
    fn round_trips(ff: &ForceField, what: &str) {
        let section = ForceFieldSection::from_forcefield(ff)
            .unwrap_or_else(|e| panic!("{what}: from_forcefield: {e}"));
        let back = section
            .to_forcefield()
            .unwrap_or_else(|e| panic!("{what}: to_forcefield: {e}"));
        assert_same_definitions(&declared(ff), &back);
        let again = ForceFieldSection::from_forcefield(&back).unwrap();
        same_section(&section, &again, what);

        #[cfg(feature = "filesystem")]
        {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("ff.mrec");
            let mut record = crate::io::mrec::MolRec::new();
            record.forcefield = Some(section.clone());
            molrs::io::write_mrec(&path, &record).unwrap();
            let stored = crate::io::mrec::MrecFile::open(&path)
                .and_then(|f| f.forcefield())
                .unwrap()
                .unwrap();
            assert_eq!(stored.document, section.document, "{what}: stored document");
            let from_store = stored
                .to_forcefield()
                .unwrap_or_else(|e| panic!("{what}: from the store: {e}"));
            assert_same_definitions(&declared(ff), &from_store);
        }
    }

    #[test]
    fn the_built_in_libraries_round_trip() {
        use crate::ff::typifier::mmff::{Mmff94Typifier, Mmff94sTypifier};
        use crate::ff::typifier::{
            AtdParameterSet, AtdTypifier, ElementTypifier, GaffParameterSet, GaffTypifier,
            OplsAaTypifier, UffTypifier,
        };
        round_trips(OplsAaTypifier::oplsaa().source_forcefield(), "OPLS-AA");
        round_trips(
            GaffTypifier::new(GaffParameterSet::Gaff).source_forcefield(),
            "GAFF",
        );
        round_trips(
            GaffTypifier::new(GaffParameterSet::Gaff2).source_forcefield(),
            "GAFF2",
        );
        for set in [
            AtdParameterSet::Bcc,
            AtdParameterSet::Abcg2,
            AtdParameterSet::Gas,
            AtdParameterSet::Gff,
        ] {
            round_trips(
                AtdTypifier::new(set).source_forcefield(),
                &format!("ATD {set:?}"),
            );
        }
        round_trips(Mmff94Typifier::new().source_forcefield(), "MMFF94");
        round_trips(Mmff94sTypifier::new().source_forcefield(), "MMFF94s");
        round_trips(UffTypifier::new().source_forcefield(), "UFF");
        round_trips(ElementTypifier::new().source_forcefield(), "element");
    }

    #[test]
    fn a_lammps_read_force_field_round_trips() {
        let text = "special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.8333\n\
                    pair_style lj/cut/coul/long 10.0 10.0\n\
                    pair_modify mix arithmetic\n\
                    pair_coeff c3 c3 0.107800 3.397710\n\
                    pair_coeff oh oh 0.093000 3.242871\n\
                    pair_coeff c3 oh 0.250000 3.100000\n\
                    bond_style harmonic\n\
                    bond_coeff c3-oh 314.1 1.426\n\
                    angle_style harmonic\n\
                    angle_coeff c3-c3-oh 76.79 109.66\n\
                    dihedral_style fourier\n\
                    dihedral_coeff c3-c3-oh-ho 2 0.16 3 0.0 0.25 1 0.0\n\
                    improper_style harmonic\n\
                    improper_coeff c3-oh-c3-c3 1.1 180.0\n";
        let ff = crate::io::lammps::forcefield_reader::LammpsForcefieldReader::new()
            .read(text.as_bytes())
            .unwrap();
        // The cross pair_coeff (NBFIX) is a pair row with itom != jtom.
        let section = ForceFieldSection::from_forcefield(&ff).unwrap();
        let pairs = section.table("pair", "lj/cut").unwrap();
        let ends = |c: &str| {
            pairs
                .get(c)
                .and_then(Column::get::<String>)
                .unwrap()
                .to_owned()
        };
        let (itom, jtom) = (ends("itom"), ends("jtom"));
        let rows: Vec<(&str, &str)> = itom
            .iter()
            .zip(jtom.iter())
            .map(|(i, j)| (i.as_str(), j.as_str()))
            .collect();
        assert_eq!(rows, [("c3", "c3"), ("oh", "oh"), ("c3", "oh")]);
        round_trips(&ff, "LAMMPS");
    }

    #[test]
    fn a_gromacs_read_force_field_round_trips() {
        let text = "[ defaults ]\n1 2 yes 0.5 0.8333\n\
                    [ atomtypes ]\n\
                    CT CT 6 12.011 -0.18 A 0.35 0.276144\n\
                    HC HC 1 1.008 0.06 A 0.25 0.12552\n\
                    [ bondtypes ]\nCT HC 1 0.109 284512.0\n\
                    [ angletypes ]\nHC CT HC 1 107.8 276.144\n\
                    [ dihedraltypes ]\nX CT CT X 1 0.0 0.6276 3\n";
        let ff = crate::io::gromacs::top_reader::GromacsTopForcefieldReader::new()
            .read(text.as_bytes())
            .unwrap();
        // The GROMACS bond_type is the atom table's `class`.
        let section = ForceFieldSection::from_forcefield(&ff).unwrap();
        let atoms = section.table("atom", "full").unwrap();
        assert_eq!(atoms.dtype("class"), Some(DType::String));
        assert_eq!(atoms.dtype("atomic_number"), Some(DType::Uint));
        // `X` is the empty-string wildcard.
        let dihedrals = section.table("dihedral", "periodic").unwrap();
        let itom = dihedrals
            .get("itom")
            .and_then(Column::get::<String>)
            .unwrap();
        assert_eq!(itom[[0]], "");
        round_trips(&ff, "GROMACS");
    }

    #[test]
    fn an_openmm_read_force_field_round_trips_with_its_string_params() {
        let xml = r#"<ForceField name="mini" combining_rule="geometric">
  <AtomTypes>
    <Type name="opls_135" class="CT" element="C" mass="12.011" def="[C;X4](C)(H)(H)H" desc="alkane CH3"/>
    <Type name="opls_140" class="HC" element="H" mass="1.008" def="H[C;X4]" overrides="opls_135"/>
  </AtomTypes>
  <HarmonicBondForce>
    <Bond class1="CT" class2="HC" length="0.109" k="284512.0"/>
  </HarmonicBondForce>
  <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
    <Atom type="opls_135" charge="-0.18" sigma="0.35" epsilon="0.276144"/>
    <Atom type="opls_140" charge="0.06" sigma="0.25" epsilon="0.12552"/>
  </NonbondedForce>
</ForceField>"#;
        let ff = crate::io::openmm_xml::reader::OpenmmXmlReader::new()
            .read(xml.as_bytes())
            .unwrap();
        let section = ForceFieldSection::from_forcefield(&ff).unwrap();
        let atoms = section.table("atom", "full").unwrap();
        for annotation in ["class", "element", "smarts"] {
            assert_eq!(atoms.dtype(annotation), Some(DType::String), "{annotation}");
        }
        assert!(atoms.validity("overrides").is_some(), "absent on opls_135");
        round_trips(&ff, "OpenMM XML");
    }

    /// Explicit cross rows from every reader that makes them survive the
    /// section and the store, and still price their pair after the trip.
    #[test]
    fn explicit_cross_rows_round_trip_and_still_override_mixing() {
        use crate::ff::compile::PotentialCompiler;
        use molrs::core::Frame;
        use molrs::op::Idx;
        use ndarray::Array1;

        let gromacs = "[ defaults ]\n1 3 yes 0.5 0.5\n\
                       [ atomtypes ]\n\
                       A 12.0 0.0 A 0.30 0.4184\n\
                       B 12.0 0.0 A 0.36 1.6736\n\
                       [ nonbond_params ]\nA B 1 0.20 3.7656\n";
        let ff = crate::io::gromacs::top_reader::GromacsTopForcefieldReader::new()
            .read(gromacs.as_bytes())
            .unwrap();
        round_trips(&ff, "GROMACS nonbond_params");
        let back = ForceFieldSection::from_forcefield(&ff)
            .unwrap()
            .to_forcefield()
            .unwrap();

        // One A–B pair at 2.5 Å: priced by the cross row (ε = 0.9, σ = 2), not
        // by geometric mixing.
        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(), "B".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut pairs = Block::new();
        pairs
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn(), None)
            .unwrap();
        pairs
            .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn(), None)
            .unwrap();
        atoms
            .insert("charge", Array1::from_vec(vec![0.0, 0.0]).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("pairs", pairs);
        let r: f64 = 2.5;
        let coords = [0.0, 0.0, 0.0, r, 0.0, 0.0];
        let s6 = (2.0 / r).powi(6);
        let want = 4.0 * 0.9 * (s6 * s6 - s6);
        let e = PotentialCompiler::new(&back)
            .compile(&frame)
            .unwrap()
            .energy_at(&coords);
        assert!((e - want).abs() < 1e-9, "E = {e}, cross row gives {want}");
    }

    /// `lj/charmm`'s `one_four` round-trips through the section, and a value
    /// other than `regular` / `epsilon14` is refused both ways.
    #[test]
    fn lj_charmm_one_four_round_trips_and_an_unknown_value_is_refused() {
        let mut ff = ForceField::new("x");
        let mut sp = Params::from_iter([("inner", 8.0), ("cutoff", 10.0)]);
        sp.set("one_four", "epsilon14");
        ff.def_style("pair", "lj/charmm", sp)
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_iter([
                    ("epsilon", 0.1),
                    ("sigma", 3.0),
                    ("epsilon14", 0.04),
                    ("sigma14", 2.0),
                ]),
            )
            .unwrap();
        let section = ForceFieldSection::from_forcefield(&ff).unwrap();
        let back = section.to_forcefield().unwrap();
        assert_eq!(
            back.get_style("pair", "lj/charmm")
                .unwrap()
                .params()
                .get("one_four")
                .and_then(|v| v.as_text()),
            Some("epsilon14")
        );
        round_trips(&ff, "lj/charmm one_four");

        let mut bad = ff.clone();
        bad.get_style_mut("pair", "lj/charmm")
            .unwrap()
            .set_param("one_four", "both");
        assert!(
            ForceFieldSection::from_forcefield(&bad)
                .unwrap_err()
                .contains("one_four")
        );
        let mut doc = section.clone();
        doc.document["styles"][0]["params"]["one_four"] = json!("both");
        let err = doc.to_forcefield().unwrap_err();
        assert!(err.contains("one_four"), "{err}");
    }

    #[test]
    fn style_level_strings_become_entry_fields_and_mixing_a_param() {
        let mut ff = ForceField::new("custom");
        let mut lj = Params::from_iter([("cutoff", 10.0)]);
        lj.set("mixing", "geometric");
        ff.def_style("pair", "lj/cut", lj).unwrap();
        let mut fene = Params::new();
        fene.set("expression", "-0.5*K*R0^2*log(1-(r/R0)^2)");
        ff.def_style("bond", "fene", fene).unwrap();
        let section = ForceFieldSection::from_forcefield(&ff).unwrap();
        let styles = &section.document["styles"];
        assert_eq!(
            styles[0]["params"],
            json!({"cutoff": 10.0, "mixing": "geometric"})
        );
        assert_eq!(
            styles[1]["expression"],
            json!("-0.5*K*R0^2*log(1-(r/R0)^2)")
        );
        assert!(styles[1].get("params").is_none());
        assert_eq!(
            section.document["units"],
            json!({"preset": "real", "length": "angstrom", "energy": "kcal/mol",
                   "angle": "degree", "charge": "e", "mass": "dalton"})
        );
        round_trips(&ff, "custom");
    }

    /// A section stating `angle: radian` beside a preset disagrees with its
    /// own preset (the presets state degrees), and is refused rather than read
    /// in a unit it does not hold.
    #[test]
    fn a_section_stating_radians_beside_a_preset_is_refused() {
        let mut ff = ForceField::new("t");
        ff.def_style("bond", "harmonic", Params::new()).unwrap();
        let mut section = ForceFieldSection::from_forcefield(&ff).unwrap();
        section.document["units"] = json!({"preset": "real", "length": "angstrom",
            "energy": "kcal/mol", "angle": "radian", "charge": "e", "mass": "dalton"});
        assert!(section.to_forcefield().is_err());
    }

    #[test]
    fn a_section_molrs_cannot_hold_is_refused_not_approximated() {
        let mut base = ForceField::new("t");
        base.def_style("bond", "harmonic", Params::new()).unwrap();
        let section = ForceFieldSection::from_forcefield(&base).unwrap();

        let mut nm = section.clone();
        nm.document["units"] = json!({"length": "nm", "energy": "kcal/mol"});
        assert!(nm.to_forcefield().unwrap_err().contains("units"));

        let mut openmm = section.clone();
        openmm.document["units"] = json!({"length": "nm", "energy": "kJ/mol"});
        assert_eq!(openmm.to_forcefield().unwrap().units(), "openmm");

        let mut stated = section.clone();
        stated.document["units"] = json!({"length": "angstrom", "energy": "eV"});
        assert_eq!(stated.to_forcefield().unwrap().units(), "metal");

        let mut smirks = section;
        smirks.document["styles"][0]["endpoint_key"] = json!("smirks");
        let mut rows = Block::new();
        rows.insert("name", strings(vec!["b1".into()]), None)
            .unwrap();
        rows.insert("smirks", strings(vec!["[#6:1]-[#1:2]".into()]), None)
            .unwrap();
        smirks.tables.insert("bond.harmonic".into(), rows);
        assert!(smirks.to_forcefield().unwrap_err().contains("smirks"));
    }

    fn grid(n: usize, scale: f64) -> ArrayD<f64> {
        ArrayD::from_shape_fn(vec![n, n], |ix| scale * (ix[0] * n + ix[1]) as f64)
    }

    fn cmap_ff(grids: &[Option<ArrayD<f64>>]) -> ForceField {
        let mut ff = ForceField::new("charmm");
        let style = ff.def_style("cmap", "charmm", Params::new()).unwrap();
        for (i, g) in grids.iter().enumerate() {
            let mut params = Params::from_iter([("id", i as f64)]);
            if let Some(g) = g {
                params.set("grid", g.clone());
            }
            style
                .def_type(&format!("c{i}"), &["C", "NH1", "CT1", "C", "NH1"], params)
                .unwrap();
        }
        ff
    }

    /// A cmap style's grids become one `f64[T, N, N]` column, a row without a
    /// grid a null row of it, and come back bit for bit — through the
    /// section, and through a `*.mrec` store.
    #[test]
    fn a_cmap_style_with_a_grid_round_trips() {
        let ff = cmap_ff(&[Some(grid(24, 0.125)), None, Some(grid(24, -1.0 / 3.0))]);
        let section = ForceFieldSection::from_forcefield(&ff).unwrap();
        let table = section.table("cmap", "charmm").unwrap();
        assert_eq!(table.get("grid").unwrap().shape(), &[3, 24, 24]);
        assert_eq!(table.validity("grid"), Some(&[true, false, true][..]));
        assert_eq!(
            table.get("mtom").and_then(Column::get::<String>).unwrap()[[0]],
            "NH1"
        );
        round_trips(&ff, "cmap");
        let back = section.to_forcefield().unwrap();
        let rows = back.get_cmaptypes();
        assert_eq!(
            rows[2].params.get("grid").and_then(|v| v.as_array()),
            Some(&grid(24, -1.0 / 3.0))
        );
        assert_eq!(rows[1].params.get("grid").and_then(|v| v.as_array()), None);
    }

    /// A grid the section cannot hold is refused by `from_forcefield`, not
    /// reshaped: two sizes in one table, a non-square grid, an array style
    /// param.
    #[test]
    fn an_array_the_section_cannot_hold_is_refused() {
        let err = ForceFieldSection::from_forcefield(&cmap_ff(&[
            Some(grid(24, 1.0)),
            Some(grid(12, 1.0)),
        ]))
        .unwrap_err();
        assert!(err.contains("\"grid\"") && err.contains("shape"), "{err}");

        let rect = ArrayD::from_elem(vec![2, 3], 0.0);
        let err = ForceFieldSection::from_forcefield(&cmap_ff(&[Some(rect)])).unwrap_err();
        assert!(err.contains("cmap grid"), "{err}");

        let mut params = Params::new();
        params.set("grid", grid(2, 1.0));
        let mut ff = ForceField::new("t");
        ff.def_style("cmap", "charmm", params).unwrap();
        let err = ForceFieldSection::from_forcefield(&ff).unwrap_err();
        assert!(err.contains("style param"), "{err}");

        let mut scalar = Params::new();
        scalar.set("grid", ArrayD::from_elem(vec![], 1.0));
        let mut ff = ForceField::new("t");
        ff.def_style("cmap", "charmm", Params::new())
            .unwrap()
            .def_type("c", &["A", "B", "C", "D", "E"], scalar)
            .unwrap();
        let err = ForceFieldSection::from_forcefield(&ff).unwrap_err();
        assert!(err.contains("0-d"), "{err}");
    }

    /// Any type param may be an array (molrec: `f64[T, S…]`, one shape per
    /// column): a rank-1 table and a rank-2 grid on a dihedral style, a row
    /// without one a null row, back bit for bit.
    #[test]
    fn any_type_param_may_be_an_array() {
        let mut ff = ForceField::new("t");
        let style = ff
            .def_style("dihedral", "table/linear", Params::new())
            .unwrap();
        let table = |s: f64| ArrayD::from_shape_fn(vec![12], |ix| s * ix[0] as f64 - 0.1);
        let mut a = Params::from_iter([("n", 12.0)]);
        a.set("table", table(0.25));
        a.set("grid", grid(3, 1.0 / 7.0));
        let mut b = Params::from_iter([("n", 12.0)]);
        b.set("table", table(-1.0 / 3.0));
        style
            .def_type("a", &["A", "B", "B", "A"], a)
            .unwrap()
            .def_type("b", &["A", "B", "B", "C"], b)
            .unwrap();
        let section = ForceFieldSection::from_forcefield(&ff).unwrap();
        let t = section.table("dihedral", "table/linear").unwrap();
        assert_eq!(t.get("table").unwrap().shape(), &[2, 12]);
        assert_eq!(t.get("grid").unwrap().shape(), &[2, 3, 3]);
        assert_eq!(t.validity("grid"), Some(&[true, false][..]));
        round_trips(&ff, "array params");
    }

    /// A category beyond molrs's seven round-trips: one the registry
    /// declares (`virtual_site`, no endpoints) and one nothing declares,
    /// whose arity is its endpoint columns'.
    #[test]
    fn a_category_beyond_the_seven_round_trips() {
        let mut ff = ForceField::new("t");
        ff.def_style("virtual_site", "tip4p", Params::new())
            .unwrap()
            .def_type("M", &[], Params::from_iter([("d", 0.15)]))
            .unwrap();
        ff.def_style(Category::new("bespoke").arity(4), "x", Params::new())
            .unwrap()
            .def_type("q", &["A", "B", "C", "D"], Params::from_iter([("k", 1.5)]))
            .unwrap();
        let section = ForceFieldSection::from_forcefield(&ff).unwrap();
        let bespoke = section.table("bespoke", "x").unwrap();
        assert!(bespoke.contains_key("ltom") && !bespoke.contains_key("mtom"));
        round_trips(&ff, "beyond the seven");
        let back = section.to_forcefield().unwrap();
        assert_eq!(back.get_style("bespoke", "x").unwrap().arity(), 4);
        assert_eq!(back.get_style("virtual_site", "tip4p").unwrap().arity(), 0);
        assert_eq!(back.get_relationtypes("bespoke")[0].endpoints.len(), 4);
    }

    #[test]
    fn a_force_field_without_a_section_form_is_refused_by_name() {
        let mut ff = ForceField::new("t");
        let mut odd = Params::new();
        odd.set("w", "x");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("A-B", &["A", "B"], Params::from_iter([("w", 1.0)]))
            .unwrap()
            .def_type("A-C", &["A", "C"], odd)
            .unwrap();
        assert!(
            ForceFieldSection::from_forcefield(&ff)
                .unwrap_err()
                .contains("\"w\"")
        );

        let mut ff = ForceField::new("t");
        ff.set_units("furlongs");
        assert!(
            ForceFieldSection::from_forcefield(&ff)
                .unwrap_err()
                .contains("furlongs")
        );
    }

    /// A cmap row holding CHARMM's alanine map round-trips through the
    /// section and a `*.mrec` store, and the force field read back prices a
    /// crossterm to the same bits.
    #[test]
    fn a_populated_cmap_round_trips_and_prices_the_same() {
        use crate::ff::compile::PotentialCompiler;
        use crate::ff::potential::cmap::charmm::tests::{alanine, chain};
        use molrs::core::Block;
        use ndarray::Array1;

        let mut ff = ForceField::new("charmm");
        let mut params = Params::new();
        params.set("grid", alanine());
        ff.def_style("cmap", "charmm", Params::new())
            .unwrap()
            .def_type("ala", &["C", "NH1", "CT1", "C", "NH1"], params)
            .unwrap();
        round_trips(&ff, "charmm cmap");
        let back = ForceFieldSection::from_forcefield(&ff)
            .unwrap()
            .to_forcefield()
            .unwrap();

        let mut cmaps = Block::new();
        for (p, key) in ["atomi", "atomj", "atomk", "atoml", "atomm"]
            .into_iter()
            .enumerate()
        {
            cmaps
                .insert(key, Array1::from_vec(vec![p as u64]).into_dyn(), None)
                .unwrap();
        }
        cmaps
            .insert(
                "type",
                Array1::from_vec(vec!["ala".to_owned()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = molrs::core::Frame::new();
        frame.insert("cmaps", cmaps);
        let x = chain(-63.0, -41.0);
        let e = |ff: &ForceField| {
            PotentialCompiler::new(ff)
                .compile(&frame)
                .unwrap()
                .energy_at(&x)
        };
        assert_eq!(e(&ff).to_bits(), e(&back).to_bits());
        assert!(e(&ff) != 0.0);
    }
}
