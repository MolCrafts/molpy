//! The force-field XML: a [`ForceField`] written whole, one element per style.
//!
//! The document keeps OpenMM's outline — a `<ForceField>` root holding
//! `<Type>` rows whose endpoints are `class1` … attributes — but every
//! potential is a style of the force-field IR, spelled the same way whatever
//! its kernel: `<BondStyle name="harmonic">` and `<PairStyle name="mmff_vdw">`
//! are the same kind of element. Parameter names and conventions are the IR's,
//! and every value is in the units the root declares, as the force field holds
//! it: nothing is converted, so a write followed by a read is the identity.
//!
//! ```xml
//! <ForceField name="TIP3P" units="real">
//!   <SpecialBonds lj="0.0 0.0 0.5" coul="0.0 0.0 0.8333" />
//!   <AtomStyle name="full">
//!     <Type name="OW" charge="-0.834" mass="15.999" />
//!   </AtomStyle>
//!   <BondStyle name="harmonic">
//!     <Type name="OW-HW" class1="OW" class2="HW" k="450.0" r0="0.9572" />
//!   </BondStyle>
//!   <PairStyle name="lj/cut" cutoff="10.0" mixing="geometric">
//!     <Type name="OW" class1="OW" class2="OW" epsilon="0.1521" sigma="3.1507" />
//!   </PairStyle>
//! </ForceField>
//! ```
//!
//! | element | holds |
//! |---|---|
//! | `<ForceField name units>` | [`ForceField::name`], [`ForceField::units`] |
//! | `<SpecialBonds lj coul>` | the declared 1-2 / 1-3 / 1-4 weights, three numbers each |
//! | `<{Category}Style name …>` | one style: `AtomStyle`, `BondStyle`, …, `CmapStyle`, or a registered category (`foo_bar` is `<FooBarStyle>`); its other attributes are its parameters |
//! | `<Type name class1 … class5 …>` | one type row: its endpoints, then its parameters |
//! | `<Array name shape>` | an array parameter of the style or row it sits in, its values row-major in the text |
//!
//! An attribute that parses as a number is a numeric parameter; any other is a
//! string parameter. OpenMM's own force-field XML (`<HarmonicBondForce>`, nm,
//! kJ/mol) is a different document, read by
//! [`OpenmmXmlReader`](crate::io::openmm_xml::OpenmmXmlReader).
//!
//! [`XmlForcefieldReader`] reads it and [`XmlForcefieldWriter`] writes it,
//! through [`ForceFieldReader::read`] and [`ForceFieldWriter::write`].

use std::fmt::Write as _;

use ndarray::{ArrayD, IxDyn};

use crate::ff::forcefield::{DefError, ForceField, Style};
use crate::ff::ir::{Params, SpecialBonds};
use crate::io::reader::ForceFieldReader;
use crate::io::writer::{ForceFieldWriteError, ForceFieldWriter, put_text};
use crate::io::xml_attribute::{attr_str, forcefield_name, forcefield_root};
use crate::io::{Sink, Source, invalid_data};

/// The endpoint attributes of a `<Type>`, in order.
const ENDPOINT_ATTRS: [&str; 5] = ["class1", "class2", "class3", "class4", "class5"];

/// Every style element's name ends with this.
const STYLE_SUFFIX: &str = "Style";

/// Reader of the force-field XML: a [`ForceField`] from the document
/// [`XmlForcefieldWriter`] writes, values as written.
///
/// The root's `units` become the force field's declared units; without them
/// the force field declares none (and reads the `real` default).
///
/// [`read`](ForceFieldReader::read) refuses, as
/// [`InvalidData`](std::io::ErrorKind::InvalidData) naming it: a root that is
/// not `<ForceField>` or carries another attribute than `name` and `units`; an
/// element none of the layout's (an OpenMM `<…Force>` names
/// [`OpenmmXmlReader`](crate::io::openmm_xml::OpenmmXmlReader)); a missing or
/// malformed attribute; a style or row that does not fit its category.
///
/// ```
/// use molrs::io::XmlForcefieldReader;
/// use molrs::io::reader::ForceFieldReader;
///
/// let xml = r#"<ForceField name="t" units="real">
///   <BondStyle name="harmonic">
///     <Type name="OW-HW" class1="OW" class2="HW" k="450.0" r0="0.9572" />
///   </BondStyle>
/// </ForceField>"#;
/// let ff = XmlForcefieldReader::new().read(xml.as_bytes()).unwrap();
/// assert_eq!(ff.get_types("bond").len(), 1);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct XmlForcefieldReader;

impl XmlForcefieldReader {
    /// The reader; the format has no settings.
    pub fn new() -> Self {
        Self
    }
}

impl ForceFieldReader for XmlForcefieldReader {
    fn read<'a>(&self, src: impl Into<Source<'a>>) -> std::io::Result<ForceField> {
        parse(&src.into().text()?).map_err(invalid_data)
    }
}

/// Writer of the force-field XML — the inverse of [`XmlForcefieldReader`].
///
/// The root states the force field's units (declared, or the `real`
/// default); values are written as it holds them. Reading the file back gives
/// the force field with its units declared.
///
/// [`write`](ForceFieldWriter::write) refuses what the layout cannot hold, by
/// name: a parameter called `name` (or `class1` … on a row); a string
/// parameter whose value reads back as a number; a category no element name
/// reads back as; a style of a category nothing declares that has no type rows
/// to state its arity.
#[derive(Debug, Clone, Copy, Default)]
pub struct XmlForcefieldWriter;

impl XmlForcefieldWriter {
    /// The writer; the format has no settings.
    pub fn new() -> Self {
        Self
    }
}

impl ForceFieldWriter for XmlForcefieldWriter {
    fn write<'a>(
        &self,
        sink: impl Into<Sink<'a>>,
        ff: &ForceField,
    ) -> Result<(), ForceFieldWriteError> {
        put_text(sink.into(), &render(ff)?)
    }
}

/// The force field the force-field XML text `xml` states.
fn parse(xml: &str) -> Result<ForceField, String> {
    let doc = roxmltree::Document::parse(xml).map_err(|e| format!("XML parse error: {e}"))?;
    let root = forcefield_root(&doc)?;
    if let Some(attr) = root
        .attributes()
        .find(|a| !matches!(a.name(), "name" | "units"))
    {
        return Err(format!(
            "<ForceField {}=…>: the root carries `name` and `units` only",
            attr.name()
        ));
    }
    let mut ff = ForceField::new(forcefield_name(&root));
    if let Some(units) = root.attribute("units") {
        ff.set_units(units);
    }
    for child in root.children().filter(|n| n.is_element()) {
        let tag = child.tag_name().name();
        if tag == "SpecialBonds" {
            ff.set_special_bonds(read_special_bonds(&child)?);
        } else if let Some(category) = style_category(tag) {
            read_style(&mut ff, &category, &child)?;
        } else {
            return Err(foreign_element(tag));
        }
    }
    Ok(ff)
}

/// Why `tag` is not read here, naming the door that reads it when it has one.
fn foreign_element(tag: &str) -> String {
    if tag.ends_with("Force") || matches!(tag, "AtomTypes" | "Residues" | "Script") {
        format!(
            "<{tag}> belongs to OpenMM's force-field XML, not to this one: read it with \
             OpenmmXmlReader"
        )
    } else {
        format!(
            "<{tag}> is not an element of the force-field XML (its elements are \
             <SpecialBonds> and one <…Style> per style)"
        )
    }
}

/// The category a style element names (`<FooBarStyle>` → `foo_bar`), or
/// `None` when `tag` is no style element.
fn style_category(tag: &str) -> Option<String> {
    let stem = tag.strip_suffix(STYLE_SUFFIX)?;
    let first = stem.chars().next()?;
    if !first.is_ascii_uppercase() || !stem.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    let mut category = String::with_capacity(stem.len() + 2);
    for (i, c) in stem.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                category.push('_');
            }
            category.push(c.to_ascii_lowercase());
        } else {
            category.push(c);
        }
    }
    Some(category)
}

/// The element a style of `category` is written as — the inverse of
/// [`style_category`], or an error when no element name reads back as
/// `category` (`a_2` would read back as `a2`).
fn style_tag(category: &str) -> Result<String, String> {
    let mut tag = String::with_capacity(category.len() + STYLE_SUFFIX.len());
    for part in category.split('_') {
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            tag.push(first.to_ascii_uppercase());
            tag.extend(chars);
        }
    }
    tag.push_str(STYLE_SUFFIX);
    if style_category(&tag).as_deref() == Some(category) {
        Ok(tag)
    } else {
        Err(format!(
            "category `{category}` has no element name that reads back as itself"
        ))
    }
}

fn read_special_bonds(node: &roxmltree::Node) -> Result<SpecialBonds, String> {
    let weights = |key: &str| -> Result<[f64; 3], String> {
        let text = attr_str(node, key)?;
        let values = text
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("<SpecialBonds {key}={text:?}>: {e}"))?;
        <[f64; 3]>::try_from(values).map_err(|v| {
            format!(
                "<SpecialBonds {key}={text:?}>: three weights (1-2, 1-3, 1-4), not {}",
                v.len()
            )
        })
    };
    Ok(SpecialBonds {
        lj: weights("lj")?,
        coul: weights("coul")?,
    })
}

/// Define the style `node` states, of `category`, with its rows.
///
/// A category nothing declares (built in or registered) takes its arity from
/// the endpoint count of the style's first row.
fn read_style(ff: &mut ForceField, category: &str, node: &roxmltree::Node) -> Result<(), String> {
    let name = attr_str(node, "name")?;
    let what = format!("<{} name={name:?}>", node.tag_name().name());
    let mut params = attribute_params(node, &["name"]);
    let mut rows = Vec::new();
    for child in node.children().filter(|n| n.is_element()) {
        match child.tag_name().name() {
            "Array" => read_array(&child, &mut params, &what)?,
            "Type" => rows.push(child),
            other => {
                return Err(format!(
                    "{what}: <{other}> is not an element of a style (its children are <Type> \
                     and <Array>)"
                ));
            }
        }
    }
    match ff.def_style(category, name, params.clone()) {
        Ok(_) => {}
        Err(DefError::UnknownCategory(_)) => {
            let Some(first) = rows.first() else {
                return Err(format!(
                    "{what}: category `{category}` is declared nowhere, and a style of it \
                     with no <Type> states no arity"
                ));
            };
            let arity = endpoints(first).map_err(|e| format!("{what}: {e}"))?.len();
            ff.def_style_with_arity(category, arity, name, params.clone())
                .map_err(|e| format!("{what}: {e}"))?;
        }
        Err(e) => return Err(format!("{what}: {e}")),
    }
    // Defined above with these exact params: this returns that style.
    let style = ff
        .def_style(category, name, params)
        .map_err(|e| format!("{what}: {e}"))?;
    for row in &rows {
        read_type(style, row)?;
    }
    Ok(())
}

fn read_type(style: &mut Style, node: &roxmltree::Node) -> Result<(), String> {
    let name = attr_str(node, "name")?;
    let what = format!("<Type name={name:?}>");
    let ends = endpoints(node).map_err(|e| format!("{what}: {e}"))?;
    let mut reserved = vec!["name"];
    reserved.extend(ENDPOINT_ATTRS);
    let mut params = attribute_params(node, &reserved);
    for child in node.children().filter(|n| n.is_element()) {
        match child.tag_name().name() {
            "Array" => read_array(&child, &mut params, &what)?,
            other => {
                return Err(format!(
                    "{what}: <{other}> is not an element of a type row (its children are \
                     <Array>)"
                ));
            }
        }
    }
    style
        .def_type(name, &ends, params)
        .map_err(|e| format!("{what}: {e}"))?;
    Ok(())
}

/// The `class1` … attributes of a `<Type>`, in order; a later one without
/// every earlier one is an error.
fn endpoints<'a>(node: &roxmltree::Node<'a, '_>) -> Result<Vec<&'a str>, String> {
    let present: Vec<&'a str> = ENDPOINT_ATTRS
        .iter()
        .map_while(|attr| node.attribute(*attr))
        .collect();
    if let Some(stray) = ENDPOINT_ATTRS[present.len()..]
        .iter()
        .find(|attr| node.attribute(**attr).is_some())
    {
        return Err(format!("`{stray}` without `class{}`", present.len() + 1));
    }
    Ok(present)
}

/// The attributes of `node` outside `reserved`: a value that parses as a
/// number is a numeric parameter, any other a string parameter.
fn attribute_params(node: &roxmltree::Node, reserved: &[&str]) -> Params {
    let mut params = Params::new();
    for attr in node.attributes() {
        if reserved.contains(&attr.name()) {
            continue;
        }
        match attr.value().parse::<f64>() {
            Ok(value) => params.set(attr.name(), value),
            Err(_) => params.set(attr.name(), attr.value()),
        }
    }
    params
}

/// Add the array parameter an `<Array name shape>` element states to `params`.
fn read_array(node: &roxmltree::Node, params: &mut Params, what: &str) -> Result<(), String> {
    let key = attr_str(node, "name")?;
    let what = format!("{what}: <Array name={key:?}>");
    if params.get(key).and_then(|v| v.as_num()).is_some()
        || params.get(key).and_then(|v| v.as_text()).is_some()
        || params.get(key).and_then(|v| v.as_array()).is_some()
    {
        return Err(format!("{what}: parameter `{key}` is stated twice"));
    }
    let shape = attr_str(node, "shape")?
        .split_whitespace()
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{what}: shape: {e}"))?;
    let values = node
        .text()
        .unwrap_or("")
        .split_whitespace()
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{what}: value: {e}"))?;
    let array =
        ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|e| format!("{what}: {e}"))?;
    params.set(key, array);
    Ok(())
}

/// `ff` as force-field XML text ([`XmlForcefieldWriter`]).
fn render(ff: &ForceField) -> Result<String, String> {
    let mut out = String::new();
    // `units` is written when declared; an undeclared one stays absent.
    let units = ff
        .units()
        .map(|u| format!(" units=\"{}\"", escape(u)))
        .unwrap_or_default();
    writeln!(out, "<ForceField name=\"{}\"{units}>", escape(&ff.name)).unwrap();
    if let Some(sb) = ff.special_bonds() {
        writeln!(
            out,
            "  <SpecialBonds lj=\"{}\" coul=\"{}\" />",
            numbers(&sb.lj),
            numbers(&sb.coul)
        )
        .unwrap();
    }
    for style in ff.styles() {
        write_style(&mut out, style)?;
    }
    out.push_str("</ForceField>\n");
    Ok(out)
}

fn write_style(out: &mut String, style: &Style) -> Result<(), String> {
    let category = style.category();
    let what = format!("{category} style `{}`", style.name());
    let tag = style_tag(category).map_err(|e| format!("{what}: {e}"))?;
    let rows = style.type_rows();
    if rows.is_empty() && !is_declared(category) {
        return Err(format!(
            "{what}: category `{category}` is declared nowhere, and a style of it with no \
             type rows cannot state its arity"
        ));
    }
    let (attrs, arrays) = param_attributes(style.params(), &["name"], &what)?;
    write!(out, "  <{tag} name=\"{}\"{attrs}", escape(style.name())).unwrap();
    if rows.is_empty() && arrays.is_empty() {
        out.push_str(" />\n");
        return Ok(());
    }
    out.push_str(">\n");
    write_arrays(out, &arrays, "    ");
    let mut reserved = vec!["name"];
    reserved.extend(ENDPOINT_ATTRS);
    for (name, endpoints, params) in rows {
        let (attrs, arrays) =
            param_attributes(params, &reserved, &format!("{what}, type `{name}`"))?;
        write!(out, "    <Type name=\"{}\"", escape(name)).unwrap();
        for (attr, endpoint) in ENDPOINT_ATTRS.iter().zip(&endpoints) {
            write!(out, " {attr}=\"{}\"", escape(endpoint)).unwrap();
        }
        out.push_str(&attrs);
        if arrays.is_empty() {
            out.push_str(" />\n");
        } else {
            out.push_str(">\n");
            write_arrays(out, &arrays, "      ");
            out.push_str("    </Type>\n");
        }
    }
    writeln!(out, "  </{tag}>").unwrap();
    Ok(())
}

/// Whether `category` is declared without a force field holding a style of
/// it — built in or registered, so a reader knows its arity.
fn is_declared(category: &str) -> bool {
    !matches!(
        ForceField::new("probe").def_style(category, "probe", Params::new()),
        Err(DefError::UnknownCategory(_))
    )
}

/// The array parameters of a style or row, by key: each an `<Array>` child.
type ArrayParams<'p> = Vec<(&'p str, &'p ArrayD<f64>)>;

/// The numeric and string parameters of `params` as ` key="value"`
/// attributes, sorted by key, and its array parameters, sorted by key.
fn param_attributes<'p>(
    params: &'p Params,
    reserved: &[&str],
    what: &str,
) -> Result<(String, ArrayParams<'p>), String> {
    let mut attrs: Vec<(&str, String)> = Vec::new();
    for (key, value) in params
        .iter()
        .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
    {
        attrs.push((key, format!("{value:?}")));
    }
    for (key, value) in params
        .iter()
        .filter_map(|(k, v)| v.as_text().map(|s| (k, s)))
    {
        if value.parse::<f64>().is_ok() {
            return Err(format!(
                "{what}: string parameter `{key}` = {value:?} would read back as a number"
            ));
        }
        attrs.push((key, escape(value)));
    }
    let mut arrays: ArrayParams<'p> = params
        .iter()
        .filter_map(|(k, v)| v.as_array().map(|a| (k, a)))
        .collect();
    if let Some(key) = attrs
        .iter()
        .map(|(key, _)| *key)
        .chain(arrays.iter().map(|(key, _)| *key))
        .find(|key| reserved.contains(key))
    {
        return Err(format!(
            "{what}: a parameter cannot be called `{key}`, the attribute the layout reserves for \
             the {}",
            if key == "name" { "name" } else { "endpoints" }
        ));
    }
    attrs.sort_by(|a, b| a.0.cmp(b.0));
    arrays.sort_by(|a, b| a.0.cmp(b.0));
    let mut out = String::new();
    for (key, value) in attrs {
        write!(out, " {key}=\"{value}\"").unwrap();
    }
    Ok((out, arrays))
}

fn write_arrays(out: &mut String, arrays: &[(&str, &ArrayD<f64>)], indent: &str) {
    for (key, array) in arrays {
        let shape: Vec<String> = array.shape().iter().map(usize::to_string).collect();
        let values: Vec<f64> = array.iter().copied().collect();
        writeln!(
            out,
            "{indent}<Array name=\"{}\" shape=\"{}\">{}</Array>",
            escape(key),
            shape.join(" "),
            numbers(&values)
        )
        .unwrap();
    }
}

/// `values` space-separated, each in a form that reads back to the same `f64`.
fn numbers(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("{v:?}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `text` with the five XML special characters escaped.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// The force field `xml` states.
    fn read(xml: &str) -> std::io::Result<ForceField> {
        XmlForcefieldReader::new().read(xml.as_bytes())
    }

    /// `ff` written into a buffer, as text.
    fn write(ff: &ForceField) -> Result<String, ForceFieldWriteError> {
        let mut out = Vec::new();
        XmlForcefieldWriter::new().write(&mut out, ff)?;
        Ok(String::from_utf8(out).expect("the writer writes UTF-8"))
    }

    /// Every key of `params`, with its value spelled out: an exact comparison
    /// that, unlike `same_parameters`, also counts annotation keys.
    fn every_param(params: &Params) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        for (key, value) in params
            .iter()
            .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
        {
            out.insert(key.to_owned(), format!("f64 {value:?}"));
        }
        for (key, value) in params
            .iter()
            .filter_map(|(k, v)| v.as_text().map(|s| (k, s)))
        {
            out.insert(key.to_owned(), format!("str {value}"));
        }
        for (key, value) in params
            .iter()
            .filter_map(|(k, v)| v.as_array().map(|a| (k, a)))
        {
            out.insert(key.to_owned(), format!("array {value:?}"));
        }
        out
    }

    /// `a` and `b` hold the same force field: name, units, special bonds,
    /// and every style with every row, in order, every parameter exact.
    fn assert_same(a: &ForceField, b: &ForceField, what: &str) {
        assert_eq!(a.name, b.name, "{what}: name");
        assert_eq!(a.units(), b.units(), "{what}: units");
        assert_eq!(
            a.special_bonds(),
            b.special_bonds(),
            "{what}: special bonds"
        );
        assert_eq!(a.styles().len(), b.styles().len(), "{what}: style count");
        for (sa, sb) in a.styles().iter().zip(b.styles()) {
            let style = format!("{what}: {} style `{}`", sa.category(), sa.name());
            assert_eq!(
                (sa.category(), sa.name()),
                (sb.category(), sb.name()),
                "{style}"
            );
            assert_eq!(
                every_param(sa.params()),
                every_param(sb.params()),
                "{style}: params"
            );
            let (ra, rb) = (sa.type_rows(), sb.type_rows());
            assert_eq!(ra.len(), rb.len(), "{style}: row count");
            for ((na, ea, pa), (nb, eb, pb)) in ra.iter().zip(&rb) {
                assert_eq!((na, ea), (nb, eb), "{style}: row");
                assert_eq!(every_param(pa), every_param(pb), "{style}: row `{na}`");
            }
        }
    }

    /// Write, read back, write again: the same force field, the same text.
    fn round_trips(ff: &ForceField, what: &str) {
        let text = write(ff).unwrap_or_else(|e| panic!("{what}: {e}"));
        let back = read(&text).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_same(ff, &back, what);
        assert_eq!(back.units(), ff.units(), "{what}: units as declared");
        assert_eq!(write(&back).unwrap(), text, "{what}: text");
    }

    /// Every library molrs ships survives the XML exactly — OPLS-AA's string
    /// `mixing`, the atom styles, the special bonds, MMFF's `mmff_vdw` table
    /// and its row-free per-instance styles.
    #[test]
    fn the_built_in_libraries_round_trip() {
        use crate::ff::typifier::Typifier as _;
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

    /// The module example reads as documented.
    #[test]
    fn the_documented_example_reads() {
        let xml = r#"
        <ForceField name="TIP3P" units="real">
          <SpecialBonds lj="0.0 0.0 0.5" coul="0.0 0.0 0.8333" />
          <AtomStyle name="full">
            <Type name="OW" charge="-0.834" mass="15.999" />
          </AtomStyle>
          <BondStyle name="harmonic">
            <Type name="OW-HW" class1="OW" class2="HW" k="450.0" r0="0.9572" />
          </BondStyle>
          <PairStyle name="lj/cut" cutoff="10.0" mixing="geometric">
            <Type name="OW" class1="OW" class2="OW" epsilon="0.1521" sigma="3.1507" />
          </PairStyle>
        </ForceField>
        "#;
        let ff = read(xml).unwrap();
        assert_eq!(ff.units(), Some("real"));
        assert_eq!(ff.special_bonds().unwrap().coul_14(), 0.8333);
        let pair = ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(
            pair.params().get("cutoff").and_then(|v| v.as_num()),
            Some(10.0)
        );
        assert_eq!(
            pair.params().get("mixing").and_then(|v| v.as_text()),
            Some("geometric")
        );
        let bond = ff.get_style("bond", "harmonic").unwrap();
        assert_eq!(
            bond.get_bondtype("HW", "OW")
                .unwrap()
                .params
                .get("k")
                .and_then(|v| v.as_num()),
            Some(450.0)
        );
        assert_eq!(
            ff.get_style("atom", "full")
                .unwrap()
                .type_params("OW")
                .unwrap()
                .get("mass")
                .and_then(|v| v.as_num()),
            Some(15.999)
        );
    }

    /// An array parameter — a CMAP grid — survives as an `<Array>`.
    #[test]
    fn an_array_parameter_round_trips() {
        let mut ff = ForceField::new("cmap");
        let grid =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.5, -1.25, 2.0, 1e-9, 0.0, 3.5]).unwrap();
        let mut params = Params::new();
        params.set("grid", grid);
        ff.def_style("cmap", "charmm", Params::new())
            .unwrap()
            .def_type("C-N-C-C-N", &["C", "N", "C", "C", "N"], params)
            .unwrap();
        let text = write(&ff).unwrap();
        assert!(text.contains("<CmapStyle name=\"charmm\">"), "{text}");
        assert!(
            text.contains("<Array name=\"grid\" shape=\"2 3\">"),
            "{text}"
        );
        round_trips(&ff, "cmap");
    }

    /// A category nothing declares takes its arity from its rows, both ways.
    #[test]
    fn an_undeclared_category_round_trips_by_its_rows() {
        let mut ff = ForceField::new("relation");
        ff.def_style_with_arity("hbond_pair", 2, "custom", Params::new())
            .unwrap()
            .def_type("O-H", &["O", "H"], Params::from_iter([("k", 1.5)]))
            .unwrap();
        let text = write(&ff).unwrap();
        assert!(text.contains("<HbondPairStyle name=\"custom\">"), "{text}");
        round_trips(&ff, "hbond_pair");

        let mut empty = ForceField::new("relation");
        empty
            .def_style_with_arity("hbond_pair", 2, "custom", Params::new())
            .unwrap();
        let err = write(&empty).unwrap_err();
        assert!(err.contains("arity"), "{err}");
    }

    /// An OpenMM element is refused, naming the reader that takes it.
    #[test]
    fn an_openmm_element_names_its_reader() {
        let xml = r#"<ForceField name="t"><HarmonicBondForce /></ForceField>"#;
        let err = read(xml).unwrap_err();
        assert!(
            err.to_string().contains("HarmonicBondForce")
                && err.to_string().contains("OpenmmXmlReader"),
            "{err}"
        );
    }

    /// What the layout cannot hold is refused by name, never written lossily.
    #[test]
    fn the_writer_refuses_what_would_not_read_back() {
        let mut ff = ForceField::new("t");
        let mut params = Params::new();
        params.set("label", "1.5");
        ff.def_style("bond", "harmonic", params).unwrap();
        let err = write(&ff).unwrap_err();
        assert!(err.contains("label") && err.contains("number"), "{err}");

        let mut ff = ForceField::new("t");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("A-B", &["A", "B"], Params::from_iter([("class1", 1.0)]))
            .unwrap();
        let err = write(&ff).unwrap_err();
        assert!(err.contains("class1"), "{err}");
    }

    /// Malformed input is an error naming what is wrong.
    #[test]
    fn malformed_input_is_named() {
        let cases = [
            (r#"<NotForceField />"#, "Root element must be <ForceField>"),
            (r#"<ForceField combining_rule="x" />"#, "combining_rule"),
            (
                r#"<ForceField><BondStyle /></ForceField>"#,
                "missing attribute 'name'",
            ),
            (
                r#"<ForceField><BondStyle name="h"><Type name="A-B" class1="A" class3="B" /></BondStyle></ForceField>"#,
                "class3",
            ),
            (
                r#"<ForceField><BondStyle name="h"><Type name="A-B" k="1" /></BondStyle></ForceField>"#,
                "A-B",
            ),
            (
                r#"<ForceField><SpecialBonds lj="0 0" coul="0 0 0" /></ForceField>"#,
                "three weights",
            ),
            (
                r#"<ForceField><CmapStyle name="c"><Type name="t" class1="A" class2="B" class3="C" class4="D" class5="E"><Array name="grid" shape="2 2">1 2 3</Array></Type></CmapStyle></ForceField>"#,
                "grid",
            ),
            (r#"<ForceField><VdWParams /></ForceField>"#, "VdWParams"),
        ];
        for (xml, needle) in cases {
            let err = read(xml).unwrap_err();
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{xml}");
            assert!(err.to_string().contains(needle), "{xml}: {err}");
        }
    }
}
