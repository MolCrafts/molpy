//! The OPLS-AA typing half of an OpenMM force-field XML: the SMARTS `def`,
//! `overrides`, `priority` and `layer` of each `<AtomTypes><Type>`, the
//! annotations Foyer adds to the OpenMM schema. Not a file format of its own:
//! [`OpenmmXmlReader::read_opls_typifier_str`](super::OpenmmXmlReader::read_opls_typifier_str)
//! reads it beside the force field of the same document.

use crate::ff::typifier::{OplsTypeRow, OplsTypingMetadata};
use crate::io::xml_attribute::{attr_str, forcefield_root};

/// Parse OPLS-AA typing metadata ([`OplsTypingMetadata`]) from an XML string —
/// the typing half of a caller's OPLS-AA XML, beside the force field
/// [`read_openmm_forcefield_str`](crate::io::read_openmm_forcefield_str) reads.
///
/// Reads each `<Type>` of the `<AtomTypes>` section, transcribing the typing
/// attributes the potential reader
/// ([`OpenmmXmlReader`](crate::io::openmm_xml::OpenmmXmlReader))
/// drops: `class`, `def` (SMARTS), `overrides` (comma list), `priority`, and
/// `layer`. This is purely additive — it never touches the potential
/// `ForceField`; the two are read from the same XML but kept separate (as MMFF's
/// typing metadata is).
///
/// Rows with no `def` are still recorded (with `def = None`); they are types
/// excluded from automatic SMARTS typing.
///
/// # Errors
///
/// Returns `Err` if the root element is not `<ForceField>`, a `<Type>` lacks the
/// required `name`/`class`, `priority`/`layer` is present but non-integer, or
/// the table is not valid ([`OplsTypingMetadata::validate`]: an `overrides` naming
/// a type the XML does not declare, or overrides forming a cycle).
pub(crate) fn read_typing_metadata(xml: &str) -> Result<OplsTypingMetadata, String> {
    let doc = roxmltree::Document::parse(xml).map_err(|e| format!("XML parse error: {}", e))?;

    let root = forcefield_root(&doc)?;

    let mut meta = OplsTypingMetadata::new();

    for child in root.children().filter(|n| n.is_element()) {
        if child.tag_name().name() != "AtomTypes" {
            continue;
        }
        for t in child
            .children()
            .filter(|n| n.is_element() && n.tag_name().name() == "Type")
        {
            let name = attr_str(&t, "name")?.to_owned();
            let class = attr_str(&t, "class")?.to_owned();
            let def = t.attribute("def").map(str::to_owned);
            let overrides = t
                .attribute("overrides")
                .map(parse_overrides)
                .unwrap_or_default();
            let priority = match t.attribute("priority") {
                None => None,
                Some(s) => Some(s.parse::<i64>().map_err(|_| {
                    format!(
                        "<Type name={:?}> attribute 'priority' is not an integer: {:?}",
                        name, s
                    )
                })?),
            };
            let layer = match t.attribute("layer") {
                None => 0,
                Some(s) => s.parse::<u32>().map_err(|_| {
                    format!(
                        "<Type name={:?}> attribute 'layer' is not an integer: {:?}",
                        name, s
                    )
                })?,
            };
            meta.insert(
                name,
                OplsTypeRow {
                    class,
                    def,
                    overrides,
                    priority,
                    layer,
                },
            );
        }
    }

    meta.validate()?;
    Ok(meta)
}

/// Split a comma-separated `overrides` attribute into trimmed, non-empty names.
fn parse_overrides(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- typing metadata readers ---------------------------------------

    // --- OPLS typing metadata reader (the inline-fixture parse is
    //     `test_opls_typing_overrides_and_layer_defaults`; the rest are edge
    //     cases) ------------------------------------------------------------

    #[test]
    fn test_opls_typing_overrides_and_layer_defaults() {
        // A row with overrides (of two declared types) + a row with no `def`. (Inline edge fixture: exercises overrides splitting +
        // def=None + default layer.)
        let xml = r#"<ForceField name="OPLS-AA">
          <AtomTypes>
            <Type name="opls_135" class="CT" element="C" mass="12.011" def="[C;X4](C)(H)(H)H"/>
            <Type name="opls_140" class="HC" element="H" mass="1.008" def="[H][C]"/>
            <Type name="opls_144" class="HA" element="H" mass="1.008" def="[H][#6]"/>
            <Type name="opls_146" class="HA" element="H" mass="1.008" def="[H][c]" overrides="opls_144, opls_140"/>
            <Type name="opls_001" class="opls_001" element="C" mass="12.011"/>
          </AtomTypes>
        </ForceField>"#;
        let meta = read_typing_metadata(xml).unwrap();

        let r135 = meta.get("opls_135").unwrap();
        assert_eq!(r135.class, "CT");
        assert_eq!(r135.def.as_deref(), Some("[C;X4](C)(H)(H)H"));
        assert!(r135.overrides.is_empty());
        assert_eq!(r135.layer, 0);
        assert_eq!(r135.priority, None);

        let r146 = meta.get("opls_146").unwrap();
        assert_eq!(
            r146.overrides,
            vec!["opls_144".to_string(), "opls_140".to_string()]
        );

        // Row with no def.
        let r001 = meta.get("opls_001").unwrap();
        assert_eq!(r001.def, None);
    }

    #[test]
    fn test_opls_typing_explicit_priority_and_layer() {
        let xml = r#"<ForceField name="OPLS-AA">
          <AtomTypes>
            <Type name="opls_x" class="CT" def="[C]" priority="7" layer="2"/>
          </AtomTypes>
        </ForceField>"#;
        let meta = read_typing_metadata(xml).unwrap();
        let r = meta.get("opls_x").unwrap();
        assert_eq!(r.priority, Some(7));
        assert_eq!(r.layer, 2);
    }

    #[test]
    fn test_opls_typing_missing_class_errors() {
        let xml = r#"<ForceField name="OPLS-AA">
          <AtomTypes><Type name="opls_135" def="[C]"/></AtomTypes>
        </ForceField>"#;
        let err = read_typing_metadata(xml).unwrap_err();
        assert!(err.contains("class"), "err: {err}");
    }

    #[test]
    fn test_opls_typing_bad_priority_errors() {
        let xml = r#"<ForceField name="OPLS-AA">
          <AtomTypes><Type name="opls_135" class="CT" priority="high"/></AtomTypes>
        </ForceField>"#;
        let err = read_typing_metadata(xml).unwrap_err();
        assert!(err.contains("priority"), "err: {err}");
    }

    #[test]
    fn test_opls_typing_wrong_root_errors() {
        let err = read_typing_metadata(r#"<System name="x"/>"#).unwrap_err();
        assert!(err.contains("ForceField"), "err: {err}");
    }
}
