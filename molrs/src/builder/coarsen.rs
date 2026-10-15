//! [`Coarsener`] — map disjoint node groups of a source graph onto the sites
//! of a new [`CoarseGrain`].

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::core::CenterError;
use crate::core::CoarseGrain;
use crate::core::keys;
use crate::core::{Graph, GraphAccess, MolGraph, NodeId, Props};

/// Why [`Coarsener::coarsen`] refuses its input. Nothing is built when it
/// returns one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoarsenError {
    /// `groups` and `site_types` differ in length.
    LengthMismatch {
        /// `groups.len()`.
        groups: usize,
        /// `site_types.len()`.
        site_types: usize,
    },
    /// Group `group` lists no node.
    EmptyGroup {
        /// Index of the empty group.
        group: usize,
    },
    /// `node` is listed twice: in group `first`, then again in group `second`
    /// (`first == second` for a repeat within one group).
    Overlap {
        /// The repeated node.
        node: NodeId,
        /// The group of its first listing.
        first: usize,
        /// The group of its second listing.
        second: usize,
    },
    /// The centre of group `group` has no answer.
    Center {
        /// Index of the group.
        group: usize,
        /// Why the centre-of-mass computation (the source type's `center`,
        /// e.g. [`Atomistic::center`](crate::core::Atomistic::center))
        /// refused it.
        source: CenterError,
    },
}

impl fmt::Display for CoarsenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { groups, site_types } => write!(
                f,
                "{groups} groups but {site_types} site types; one type per group"
            ),
            Self::EmptyGroup { group } => write!(f, "group {group} is empty"),
            Self::Overlap {
                node,
                first,
                second,
            } => write!(
                f,
                "node {} is in group {first} and again in group {second}",
                u64::from(*node)
            ),
            Self::Center { group, source } => write!(f, "group {group}: {source}"),
        }
    }
}

impl std::error::Error for CoarsenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Center { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Maps groups of a borrowed source graph onto coarse-grained sites.
///
/// This is the centre-of-mass mapping operator of coarse-grained modelling
/// (Noid, *J. Chem. Phys.* **139**, 090901 (2013), doi:10.1063/1.4818908):
/// site I stands for group G_I and sits at its centre of mass. The groups
/// typically come from [`SubgraphMatcher::find`](crate::perceive::SubgraphMatcher::find)
/// after the caller has made them disjoint.
///
/// The source is borrowed, not copied: any [`Graph`] works (a [`MolGraph`],
/// an [`Atomistic`](crate::core::Atomistic) or a [`CoarseGrain`]). The one
/// verb is [`coarsen`](Self::coarsen).
///
/// # Examples
///
/// ```
/// use molrs::builder::Coarsener;
/// use molrs::core::keys;
/// use molrs::core::CoarseGrain;
/// use molrs::props;
///
/// let mut src = CoarseGrain::new();
/// let a = src.add_bead(props! {
///     keys::BEAD_TYPE => "S",
///     keys::X => 0.0,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// }).unwrap();
/// let b = src.add_bead(props! {
///     keys::BEAD_TYPE => "S",
///     keys::X => 4.0,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// }).unwrap();
/// src.set_bead(a, keys::MASS, 1.0).unwrap();
/// src.set_bead(b, keys::MASS, 3.0).unwrap();
///
/// let cg = Coarsener::new(&src).coarsen(&[vec![a, b]], &["A"]).unwrap();
/// let (_, site) = cg.beads().next().unwrap();
/// // (1·0 + 3·4) / (1 + 3) = 3 Å, M = 4
/// assert_eq!(site.position(), Some([3.0, 0.0, 0.0]));
/// assert_eq!(site.get_f64(keys::MASS), Some(4.0));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Coarsener<'a> {
    source: &'a MolGraph,
}

impl<'a> Coarsener<'a> {
    /// A coarsener over `source` (any graph type), borrowed.
    pub fn new(source: &'a impl Graph) -> Self {
        Self {
            source: source.graph(),
        }
    }

    /// A new [`CoarseGrain`] with one site per group.
    ///
    /// A *coarse-grained* (CG) model replaces a group of atoms by one
    /// particle, a *site* (or bead). Site I stands for the node group
    /// G_I = `groups[I]` and is written in group order, so site I is row I of
    /// the result. The mapping is the centre-of-mass operator (Noid,
    /// *J. Chem. Phys.* **139**, 090901 (2013), doi:10.1063/1.4818908). With
    /// m_i the source `mass` of node i and r_i = (x, y, z) its position in Å:
    ///
    /// - **Mass** M_I = Σ_{i∈G_I} m_i, written as the site's `mass`, in the
    ///   unit the source `mass` column is stored in (nothing is converted).
    /// - **Position** R_I = Σ_{i∈G_I} m_i r_i / M_I, in Å, computed by
    ///   the source's `center` (e.g.
    ///   [`Atomistic::center`](crate::core::Atomistic::center)). Only mass ratios enter R_I, so the mass unit does not
    ///   affect it. No periodic imaging is applied: a group split across a
    ///   periodic box face averages its split coordinates, so unwrap the source
    ///   first when a group straddles the box.
    /// - **Axis** A_I = R_I − r_{first}, the vector (Å) from the first listed
    ///   node of G_I to the site, written as `axis_x/y/z` and read back by
    ///   [`CoarseGrain::axes`]; zero for a one-node group. It fixes the
    ///   site's direction when the group's first node is a distinguished end
    ///   (the head a matched pattern lists first).
    /// - **Type** `site_types[I]`, written as the site's `bead_type`.
    /// - **Members** the handles of G_I, in the listed order, as
    ///   [`CoarseGrain::bead_members`].
    ///
    /// **Bonding rule.** Sites I ≠ J are bonded, once, if and only if some
    /// source relation of the arity-2 kind named `bonds` joins a node of G_I
    /// to a node of G_J. Bonds inside a group, and bonds touching a node in no
    /// group, add nothing. A CG bond has no order. Relations of any other kind
    /// (`ports`, angles, …) are ignored.
    ///
    /// **Disjointness.** No node may be listed twice, across groups or within
    /// one. Making the groups disjoint is the caller's job —
    /// `SubgraphMatcher::find` returns overlapping groups — and it is checked
    /// here, not repaired.
    ///
    /// # Arguments
    ///
    /// * `groups` — disjoint, non-empty node groups of the source.
    /// * `site_types` — one `bead_type` per group.
    ///
    /// # Returns
    ///
    /// The new `CoarseGrain`; empty when `groups` is empty.
    ///
    /// # Errors
    ///
    /// All checks run before the first write, in this order, reporting the
    /// first offender:
    ///
    /// 1. [`CoarsenError::LengthMismatch`] — `groups.len() != site_types.len()`.
    /// 2. [`CoarsenError::EmptyGroup`] — the first empty group.
    /// 3. [`CoarsenError::Overlap`] — the first node listed a second time,
    ///    across groups or within one.
    /// 4. [`CoarsenError::Center`] — per group, in order: a node that is not
    ///    in the source, lacks a finite position, or lacks a finite
    ///    non-negative `mass`, or a group whose total mass is not positive.
    ///
    /// # Complexity
    ///
    /// O(Σ|G_I| + E), with Σ|G_I| the total number of listed nodes and E the
    /// number of source `bonds` relations: every listed node is read once for
    /// its position and mass, and every source bond is visited once.
    ///
    /// # Examples
    ///
    /// Four beads in a chain a–b–c–d, grouped {a, b} and {c, d}: the b–c bond
    /// crosses the groups and becomes the one site bond.
    ///
    /// ```
    /// use molrs::builder::Coarsener;
    /// use molrs::core::keys;
    /// use molrs::core::CoarseGrain;
    /// use molrs::props;
    ///
    /// let mut src = CoarseGrain::new();
    /// let mut beads = Vec::new();
    /// for (x, mass) in [(0.0, 1.0), (4.0, 3.0), (8.0, 2.0), (10.0, 2.0)] {
    ///     let bead = src.add_bead(props! {
    ///         keys::BEAD_TYPE => "S",
    ///         keys::X => x,
    ///         keys::Y => 0.0,
    ///         keys::Z => 0.0,
    ///     }).unwrap();
    ///     src.set_bead(bead, keys::MASS, mass)?;
    ///     beads.push(bead);
    /// }
    /// for pair in beads.windows(2) {
    ///     src.add_bond(pair[0], pair[1], props! {})?;
    /// }
    ///
    /// let groups = [vec![beads[0], beads[1]], vec![beads[2], beads[3]]];
    /// let cg = Coarsener::new(&src).coarsen(&groups, &["A", "B"])?;
    ///
    /// let sites: Vec<_> = cg.bead_ids().collect();
    /// // R_0 = (1·0 + 3·4) / 4 = 3 Å, R_1 = (2·8 + 2·10) / 4 = 9 Å
    /// assert_eq!(cg.positions(&sites)?, vec![[3.0, 0.0, 0.0], [9.0, 0.0, 0.0]]);
    /// assert_eq!(cg.bead_types(&sites)?, ["A", "B"]);
    /// assert_eq!(cg.n_bonds(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn coarsen(
        &self,
        groups: &[Vec<NodeId>],
        site_types: &[&str],
    ) -> Result<CoarseGrain, CoarsenError> {
        if groups.len() != site_types.len() {
            return Err(CoarsenError::LengthMismatch {
                groups: groups.len(),
                site_types: site_types.len(),
            });
        }
        if let Some(group) = groups.iter().position(Vec::is_empty) {
            return Err(CoarsenError::EmptyGroup { group });
        }

        let mut group_of: HashMap<NodeId, usize> = HashMap::new();
        for (second, members) in groups.iter().enumerate() {
            for &node in members {
                if let Some(first) = group_of.insert(node, second) {
                    return Err(CoarsenError::Overlap {
                        node,
                        first,
                        second,
                    });
                }
            }
        }

        let table = self.source.node_table();
        let mut sites: Vec<([f64; 3], f64, [f64; 3])> = Vec::with_capacity(groups.len());
        for (group, members) in groups.iter().enumerate() {
            let refuse = |source| CoarsenError::Center { group, source };
            let r = self.source.center(members).map_err(refuse)?;
            // `center` has just validated every member's mass.
            let mut m = 0.0;
            for &node in members {
                m += table
                    .get::<f64>(node, keys::MASS)
                    .map_err(|_| refuse(CenterError::BadMass { node }))?;
            }
            let first = self
                .source
                .get_node(members[0])
                .ok()
                .and_then(|node| node.position())
                .ok_or(refuse(CenterError::BadPosition { node: members[0] }))?;
            sites.push((r, m, [r[0] - first[0], r[1] - first[1], r[2] - first[2]]));
        }

        let mut cg = CoarseGrain::new();
        let mut ids: Vec<NodeId> = Vec::with_capacity(groups.len());
        for ((&site_type, members), &([x, y, z], m, axis)) in
            site_types.iter().zip(groups).zip(&sites)
        {
            let mut props = Props::new()
                .with(keys::BEAD_TYPE, site_type)
                .with(keys::X, x)
                .with(keys::Y, y)
                .with(keys::Z, z)
                .with(keys::MASS, m);
            for (key, value) in keys::AXIS.into_iter().zip(axis) {
                props.set(key, value);
            }
            let site = cg
                .add_bead(props)
                .expect("a CoarseGrain built here gets a string bead_type and f64 fields");
            cg.set_bead_members(site, members.iter().map(|&n| u64::from(n)).collect())
                .expect("the bead was just added");
            ids.push(site);
        }

        let bonds = self
            .source
            .kind_id("bonds")
            .filter(|&kind| self.source.arity(kind) == 2);
        if let Some(kind) = bonds {
            let mut seen: HashSet<(usize, usize)> = HashSet::new();
            for rid in self.source.relation_ids(kind) {
                let Ok(ends) = self.source.relation_nodes(kind, rid) else {
                    continue;
                };
                let (Some(&gi), Some(&gj)) = (group_of.get(&ends[0]), group_of.get(&ends[1]))
                else {
                    continue;
                };
                if gi != gj && seen.insert((gi.min(gj), gi.max(gj))) {
                    cg.add_bond(ids[gi], ids[gj], Props::new())
                        .expect("both sites were just added to this CoarseGrain");
                }
            }
        }

        Ok(cg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::NodeId;
    use crate::core::keys;
    use crate::props;

    /// A source bead of type "S" at `(x, 0, 0)` carrying `mass`.
    fn bead(src: &mut CoarseGrain, x: f64, mass: f64) -> NodeId {
        let id = src
            .add_bead(
                props! { keys::BEAD_TYPE => "S", keys::X => x, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        src.set_bead(id, keys::MASS, mass).unwrap();
        id
    }

    /// Beads at x = 0, 4, 8, 10 with masses 1, 3, 2, 2; bonds b0-b1, b1-b2,
    /// b2-b3, b0-b3.
    fn square() -> (CoarseGrain, Vec<NodeId>) {
        let mut src = CoarseGrain::new();
        let b: Vec<NodeId> = [(0.0, 1.0), (4.0, 3.0), (8.0, 2.0), (10.0, 2.0)]
            .into_iter()
            .map(|(x, m)| bead(&mut src, x, m))
            .collect();
        for (i, j) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
            src.add_bond(b[i], b[j], props! {}).unwrap();
        }
        (src, b)
    }

    fn position(cg: &CoarseGrain, site: NodeId) -> [f64; 3] {
        cg.get_bead(site).unwrap().position().unwrap()
    }

    #[test]
    fn sites_sit_at_the_mass_weighted_centre_with_the_group_mass() {
        let (src, b) = square();
        let cg = Coarsener::new(&src)
            .coarsen(&[vec![b[0], b[1]], vec![b[2], b[3]]], &["A", "B"])
            .unwrap();
        let sites: Vec<NodeId> = cg.bead_ids().collect();
        assert_eq!(sites.len(), 2);

        // (1·0 + 3·4) / 4 = 3 ; (2·8 + 2·10) / 4 = 9
        for (site, x, bead_type) in [(sites[0], 3.0, "A"), (sites[1], 9.0, "B")] {
            let r = position(&cg, site);
            assert!((r[0] - x).abs() < 1e-12, "{r:?}");
            assert!(r[1].abs() < 1e-12 && r[2].abs() < 1e-12, "{r:?}");
            let bag = cg.get_bead(site).unwrap();
            assert!((bag.get_f64(keys::MASS).unwrap() - 4.0).abs() < 1e-12);
            assert_eq!(bag.get_str(keys::BEAD_TYPE), Some(bead_type));
        }
        assert_eq!(
            cg.bead_members(sites[0]),
            [u64::from(b[0]), u64::from(b[1])]
        );
    }

    #[test]
    fn each_site_axis_runs_from_its_first_member_to_the_site() {
        let (src, b) = square();
        // Group {b1, b0}: R = (4·3 + 0·1) / 4 = 3, first member b1 at 4 → −1.
        // Group {b2}: one member, axis 0.
        let cg = Coarsener::new(&src)
            .coarsen(&[vec![b[1], b[0]], vec![b[2]]], &["A", "B"])
            .unwrap();
        let sites: Vec<NodeId> = cg.bead_ids().collect();

        assert_eq!(
            cg.axes(&sites).unwrap(),
            vec![[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        );
    }

    #[test]
    fn two_crossing_source_bonds_give_one_site_bond() {
        let (src, b) = square();
        let cg = Coarsener::new(&src)
            .coarsen(&[vec![b[0], b[1]], vec![b[2], b[3]]], &["A", "B"])
            .unwrap();
        let sites: Vec<NodeId> = cg.bead_ids().collect();
        assert_eq!(cg.n_bonds(), 1);
        let (_, bond) = cg.bonds().next().unwrap();
        let mut ends = [bond.nodes[0], bond.nodes[1]];
        ends.sort_by_key(|&n| u64::from(n));
        let mut want = [sites[0], sites[1]];
        want.sort_by_key(|&n| u64::from(n));
        assert_eq!(ends, want);
    }

    #[test]
    fn an_ungrouped_bead_is_absent_and_adds_no_bond() {
        // b0 - b1 - b2 with b1 left out: both source bonds touch it.
        let mut src = CoarseGrain::new();
        let b: Vec<NodeId> = [0.0, 1.0, 2.0]
            .into_iter()
            .map(|x| bead(&mut src, x, 1.0))
            .collect();
        src.add_bond(b[0], b[1], props! {}).unwrap();
        src.add_bond(b[1], b[2], props! {}).unwrap();

        let cg = Coarsener::new(&src)
            .coarsen(&[vec![b[0]], vec![b[2]]], &["A", "A"])
            .unwrap();
        assert_eq!(cg.n_beads(), 2);
        assert_eq!(cg.n_bonds(), 0);
        assert!(cg.beads_of_atom(u64::from(b[1])).is_empty());
    }

    #[test]
    fn no_groups_give_an_empty_coarse_grain() {
        let (src, _) = square();
        let cg = Coarsener::new(&src).coarsen(&[], &[]).unwrap();
        assert_eq!(cg.n_beads(), 0);
        assert_eq!(cg.n_bonds(), 0);
    }

    #[test]
    fn a_type_count_differing_from_the_group_count_is_refused() {
        let (src, b) = square();
        assert_eq!(
            Coarsener::new(&src)
                .coarsen(&[vec![b[0]], vec![b[1]]], &["A"])
                .err(),
            Some(CoarsenError::LengthMismatch {
                groups: 2,
                site_types: 1
            })
        );
    }

    #[test]
    fn an_empty_group_is_refused() {
        let (src, b) = square();
        assert_eq!(
            Coarsener::new(&src)
                .coarsen(&[vec![b[0]], vec![]], &["A", "B"])
                .err(),
            Some(CoarsenError::EmptyGroup { group: 1 })
        );
    }

    #[test]
    fn a_node_in_two_groups_is_refused() {
        let (src, b) = square();
        assert_eq!(
            Coarsener::new(&src)
                .coarsen(&[vec![b[0], b[1]], vec![b[1], b[2]]], &["A", "B"])
                .err(),
            Some(CoarsenError::Overlap {
                node: b[1],
                first: 0,
                second: 1
            })
        );
    }

    #[test]
    fn a_node_twice_in_one_group_is_refused() {
        let (src, b) = square();
        assert_eq!(
            Coarsener::new(&src)
                .coarsen(&[vec![b[0], b[0]]], &["A"])
                .err(),
            Some(CoarsenError::Overlap {
                node: b[0],
                first: 0,
                second: 0
            })
        );
    }

    #[test]
    fn a_massless_node_is_refused_through_center() {
        let (mut src, b) = square();
        let bare = src
            .add_bead(
                props! { keys::BEAD_TYPE => "S", keys::X => 1.0, keys::Y => 0.0, keys::Z => 0.0 },
            )
            .unwrap();
        assert_eq!(
            Coarsener::new(&src)
                .coarsen(&[vec![b[0]], vec![bare]], &["A", "B"])
                .err(),
            Some(CoarsenError::Center {
                group: 1,
                source: CenterError::BadMass { node: bare }
            })
        );
    }
}
