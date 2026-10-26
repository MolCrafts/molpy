//! Attribute reading shared by the XML readers
//! ([`read_xml_forcefield`](crate::io::read_xml_forcefield), the OpenMM
//! force-field reader and the OPLS-AA typing annotations of an OpenMM file):
//! the `<ForceField>` root, its name and required attributes.

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

/// The force-field name the root states, or the empty name when it states
/// none — OpenMM's own files carry no `name`, and no stand-in is made up.
pub(crate) fn forcefield_name<'a>(root: &roxmltree::Node<'a, '_>) -> &'a str {
    root.attribute("name").unwrap_or("")
}

/// The attribute `name` of `node`, or why it is missing.
pub(crate) fn attr_str<'a>(node: &'a roxmltree::Node, name: &str) -> Result<&'a str, String> {
    node.attribute(name)
        .ok_or_else(|| format!("<{}> missing attribute '{}'", node.tag_name().name(), name))
}
