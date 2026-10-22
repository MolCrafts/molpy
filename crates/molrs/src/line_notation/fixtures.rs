//! Test-only walkers over the shared [`SmilesIr`] AST.
//!
//! Three test modules (`parser`, `smiles::from_atomistic`, `smiles::write`)
//! each need the same visit-order traversal of an IR; this is the single copy
//! they share. Compiled under `#[cfg(test)]` only, so it never reaches a
//! shipped build.

use super::ast::{AtomNode, BondingDescriptor, Chain, ChainElement, SmilesIr};

/// Every atom node of `ir` in visit order: head, then bonded atoms and branch
/// chains in written order (ring closures carry no atom).
pub(crate) fn atom_nodes(ir: &SmilesIr) -> Vec<&AtomNode> {
    let mut nodes = Vec::new();
    for component in &ir.components {
        collect(component, &mut nodes);
    }
    nodes
}

/// Every bonding descriptor of `ir`, in atom-visit order then per-atom list
/// order.
pub(crate) fn descriptors(ir: &SmilesIr) -> Vec<BondingDescriptor> {
    atom_nodes(ir)
        .into_iter()
        .flat_map(|node| node.descriptors.iter().cloned())
        .collect()
}

fn collect<'a>(chain: &'a Chain, nodes: &mut Vec<&'a AtomNode>) {
    nodes.push(&chain.head);
    for elem in &chain.tail {
        match elem {
            ChainElement::BondedAtom { atom, .. } => nodes.push(atom),
            ChainElement::Branch { chain, .. } => collect(chain, nodes),
            ChainElement::RingClosure { .. } => {}
        }
    }
}
