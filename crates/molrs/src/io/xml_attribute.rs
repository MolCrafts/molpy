//! Attribute reading shared by the XML readers
//! ([`read_xml_forcefield`](crate::io::read_xml_forcefield) and the OPLS-AA
//! typing annotations of an OpenMM file): the `<ForceField>` root and required
//! attributes.

/// The `<ForceField>` root of a parsed document, or why it is not one.
pub(crate) fn forcefield_root<'a, 'i>(
    doc: &'a roxmltree::Document<'i>,
) -> Result<roxmltree::Node<'a, 'i>, String> {
    let root = doc.root_element();
    if root.tag_name().name() != "ForceField" {
        return Err(format!(
            "Root element must be <ForceField>, got <{}>",
            root.tag_name().name()
        ));
    }
    Ok(root)
}

pub(crate) fn attr_str<'a>(node: &'a roxmltree::Node, name: &str) -> Result<&'a str, String> {
    node.attribute(name)
        .ok_or_else(|| format!("<{}> missing attribute '{}'", node.tag_name().name(), name))
}
