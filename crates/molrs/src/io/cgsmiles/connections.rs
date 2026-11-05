//! The lowest level's resolved pairs as validated instance/port connections.
//!
//! Both doors out of a complete `CGsmiles` string — direct expansion into one
//! molecule ([`CgSmilesIr::to_atomistic`]) and the explicit assembly graph
//! ([`CgSmilesIr::to_assembly_graph`]) — read the lowest level through
//! [`CgSmilesIr::lowest_assembly`], so a malformed or incomplete pairing is
//! refused the same way by each and never silently capped by one of them.

use std::collections::{BTreeMap, HashSet};

use crate::core::{Atomistic, BondNumber};
use crate::io::cgsmiles::templates::cg_build;
use crate::io::cgsmiles::{CgGraph, CgSmilesIr, PairEnd, ResolvedPair};
use crate::io::smiles::{BondKind, Notation, SmilesError, SmilesErrorKind};

/// The lowest level of an IR and its resolved pairs, as
/// [`CgSmilesIr::lowest_level`] reads them.
pub(super) struct LowestLevel<'ir> {
    /// The lowest level — the IR's last.
    pub(super) level: &'ir CgGraph,
    /// That level's resolved pairs.
    pub(super) pairs: &'ir [ResolvedPair],
}

/// One resolved connection of the lowest level: both ends as
/// `(instance, port ordinal)` in that level's node order, source first, and
/// the bond class the pairing decided.
pub(super) struct ResolvedConnection {
    pub(super) ends: [(usize, usize); 2],
    pub(super) kind: BondKind,
}

/// The lowest level read as an assembly: one definition name per instance
/// and the validated connections between the instances.
pub(super) struct LowestAssembly<'ir> {
    pub(super) names: Vec<&'ir str>,
    pub(super) connections: Vec<ResolvedConnection>,
}

impl CgSmilesIr {
    /// The lowest level with what every reader of it needs: the level itself
    /// and its resolved pairs, under a fragment table that defines its nodes.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::CgBuild`] when there is not one pair list per level
    /// — the reader always builds one, and a hand-built IR that breaks it
    /// would otherwise read as a pairless level — and
    /// [`SmilesErrorKind::CgNotExpandable`] for a base-only string (no
    /// fragment table, or no level), which writes beads and never says what
    /// they are made of. The first check makes the lowest level's pair list
    /// present exactly when the lowest level is.
    pub(super) fn lowest_level(&self) -> Result<LowestLevel<'_>, SmilesError> {
        if self.pairs.len() != self.levels.len() {
            let reason = format!(
                "pairs/levels misaligned: {} pair lists for {} levels",
                self.pairs.len(),
                self.levels.len()
            );
            return Err(cg_build(self.span, reason));
        }
        match (self.fragments.last(), self.levels.last(), self.pairs.last()) {
            (Some(_), Some(level), Some(pairs)) => Ok(LowestLevel { level, pairs }),
            _ => Err(SmilesError::new(
                SmilesErrorKind::CgNotExpandable("base-only string (no fragment table)".to_owned()),
                self.span,
                "",
                Notation::CgSmiles,
            )),
        }
    }

    /// Validate the lowest level's pairing against `templates`, the port
    /// tables its ordinals index (see [`Self::templates`]), and read it as
    /// instance/port connections.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::CgNotExpandable`](crate::io::smiles::SmilesErrorKind::CgNotExpandable)
    /// for base-only input; otherwise
    /// [`SmilesErrorKind::CgBuild`](crate::io::smiles::SmilesErrorKind::CgBuild)
    /// for an instance without a template, a pair end that is not a
    /// last-level port, an instance or port outside the level or its
    /// template, a port used twice, a bond of an edge resolved twice or not
    /// at all, and an unpaired port whose order a hydrogen handle cannot
    /// stand for ([`BondNumber::needs_explicit_cap`]). Only the last is
    /// reachable from [`Self::parse`]'s output; it asks for an explicit cap
    /// fragment.
    pub(super) fn lowest_assembly(
        &self,
        templates: &BTreeMap<String, Atomistic>,
    ) -> Result<LowestAssembly<'_>, SmilesError> {
        let LowestLevel { level, pairs } = self.lowest_level()?;
        let mut orders: BTreeMap<&str, Vec<BondNumber>> = BTreeMap::new();
        let mut names = Vec::with_capacity(level.nodes.len());
        for (instance, node) in level.nodes.iter().enumerate() {
            let name = node.name.as_str();
            if !orders.contains_key(name) {
                let template = templates.get(name).ok_or_else(|| {
                    cg_build(
                        node.span,
                        format!("instance {instance}: no template '{name}'"),
                    )
                })?;
                let ports = template
                    .port_ids()
                    .map(|id| template.get_port(id).map(|port| port.order))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| cg_build(node.span, format!("instance {instance}: {e}")))?;
                orders.insert(name, ports);
            }
            names.push(name);
        }

        let mut consumed = HashSet::new();
        let mut paired = HashSet::new();
        let mut connections = Vec::with_capacity(pairs.len());
        for (index, pair) in pairs.iter().enumerate() {
            let build = |reason| cg_build(self.span, format!("resolved pair {index}: {reason}"));
            let edge = level
                .edges
                .get(pair.edge)
                .ok_or_else(|| build(format!("no coarse edge {}", pair.edge)))?;
            if pair.bond >= usize::from(edge.order.multiplicity())
                || !paired.insert((pair.edge, pair.bond))
            {
                return Err(build(format!(
                    "invalid or repeated bond {} of edge {}",
                    pair.bond, pair.edge
                )));
            }
            let mut ends = [(0, 0); 2];
            for (side, (end, expected)) in [(&pair.src, edge.i), (&pair.dst, edge.j)]
                .into_iter()
                .enumerate()
            {
                let PairEnd::Body { instance, port } = *end else {
                    return Err(build(format!("{end:?} is not a last-level port")));
                };
                if instance != expected {
                    return Err(build(format!(
                        "instance {instance} is not edge endpoint {expected}"
                    )));
                }
                let n_ports = orders[names[instance]].len();
                if port >= n_ports {
                    return Err(build(format!(
                        "instance {instance} ('{}') has {n_ports} ports, no port {port}",
                        names[instance]
                    )));
                }
                if !consumed.insert((instance, port)) {
                    return Err(build(format!(
                        "instance {instance}, port {port} is used twice"
                    )));
                }
                ends[side] = (instance, port);
            }
            connections.push(ResolvedConnection {
                ends,
                kind: pair.kind,
            });
        }
        for (index, edge) in level.edges.iter().enumerate() {
            for bond in 0..usize::from(edge.order.multiplicity()) {
                if !paired.contains(&(index, bond)) {
                    return Err(cg_build(
                        edge.span,
                        format!("edge {index} is missing resolved bond {bond}"),
                    ));
                }
            }
        }
        for (instance, (name, node)) in names.iter().zip(&level.nodes).enumerate() {
            for (port, order) in orders[name].iter().enumerate() {
                if order.needs_explicit_cap() && !consumed.contains(&(instance, port)) {
                    return Err(cg_build(
                        node.span,
                        format!(
                            "instance {instance} unused non-single port {port} requires an explicit cap fragment"
                        ),
                    ));
                }
            }
        }
        Ok(LowestAssembly { names, connections })
    }
}
