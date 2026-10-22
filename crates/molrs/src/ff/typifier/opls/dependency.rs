//! OPLS-AA typing dependency analysis for `%label` layered defs.
//!
//! A SMARTS `def` may reference an atom's *previously-assigned* OPLS type via
//! a `%<type name>` context-label token (e.g. benzene's aromatic-H type
//! `opls_146` is `[#1]-[c;%opls_145]` — "a hydrogen bonded to a carbon already
//! typed `opls_145`"). Such a def therefore **depends** on `opls_145` being
//! resolved first. A label is a dependency iff it names a def-carrying type of
//! the metadata, whatever its spelling: a rule set whose types are not named
//! `opls_*` layers the same way.
//!
//! [`OplsDependencyAnalyzer`] extracts these per-def dependencies, then assigns
//! a topological **level** to every type via Kahn's algorithm (no-dep types →
//! level 0; a type depending only on `< k` → level `k`). Genuine dependency
//! cycles (mutually-referencing types) are detected with Tarjan's
//! strongly-connected-components algorithm and placed together at
//! `max_level + 1` as a *circular group*, to be resolved by fixed-point
//! iteration in the [`LayeredTypingEngine`](super::layered::LayeredTypingEngine).
//!
//! Dependencies are restricted to type names that themselves carry a def (a
//! reference to a no-def type is not a dependency), and an SCC is recorded only
//! when it contains more than one type (a true cycle).

use std::collections::{HashMap, HashSet, VecDeque};

use molrs::perceive::smarts::SmartsPattern;

use super::typing_metadata::OplsTypingMetadata;

/// Topological dependency analysis over the `%label`-referencing OPLS defs.
///
/// Only types carrying a SMARTS `def` are nodes; a dependency edge `A → B`
/// means "A's def references `%B`" (and `B` itself has a def). Levels and
/// circular groups are computed once at construction.
#[derive(Debug, Clone)]
pub struct OplsDependencyAnalyzer {
    /// type → set of types it depends on (each itself a def-carrying type);
    /// kept for the tests, which inspect it.
    #[cfg(test)]
    dependencies: HashMap<String, HashSet<String>>,
    /// type → topological level.
    levels: HashMap<String, usize>,
    /// Circular dependency groups (Tarjan SCCs of size > 1), each placed at
    /// `max_level + 1`.
    circular_groups: Vec<HashSet<String>>,
}

impl OplsDependencyAnalyzer {
    /// Build the analyzer from the typing metadata.
    ///
    /// Each def is parsed once to collect its `%label` context-label
    /// references; unparseable defs are skipped here (they fail-fast later, at
    /// [`LayeredTypingEngine::build`](super::layered::LayeredTypingEngine::build)).
    /// A def with no `def` string is not a node.
    pub fn new(meta: &OplsTypingMetadata) -> Self {
        // Node set: every type carrying a (parseable) def.
        let mut pattern_types: HashSet<String> = HashSet::new();
        let mut raw_deps: HashMap<String, HashSet<String>> = HashMap::new();
        for (name, row) in meta.iter() {
            let Some(def) = row.def.as_deref() else {
                continue;
            };
            let Ok(pat) = SmartsPattern::parse(def) else {
                // Malformed defs are handled (fail-fast) by the engine; for
                // dependency analysis we simply treat them as label-free.
                pattern_types.insert(name.clone());
                raw_deps.insert(name.clone(), HashSet::new());
                continue;
            };
            pattern_types.insert(name.clone());
            // In OPLS-style typing a context label is only ever a type name
            // (the parser already stripped the leading `%`). Whether it names
            // a type of this metadata is decided below, whatever the name
            // looks like.
            let deps: HashSet<String> = pat.context_labels().into_iter().collect();
            raw_deps.insert(name.clone(), deps);
        }

        // A label is a dependency iff it names a type of the metadata that
        // carries a def (a reference to a no-def type is not a dependency).
        let dependencies: HashMap<String, HashSet<String>> = raw_deps
            .into_iter()
            .map(|(name, deps)| {
                let valid = deps
                    .into_iter()
                    .filter(|d| pattern_types.contains(d))
                    .collect();
                (name, valid)
            })
            .collect();

        let (levels, circular_groups) = compute_levels(&dependencies);
        Self {
            #[cfg(test)]
            dependencies,
            levels,
            circular_groups,
        }
    }

    /// The dependency set of a type (the types it references via `%label`).
    #[cfg(test)]
    pub fn dependencies_of(&self, name: &str) -> Option<&HashSet<String>> {
        self.dependencies.get(name)
    }

    /// The topological level assigned to a type, if it is a node.
    pub fn level(&self, name: &str) -> Option<usize> {
        self.levels.get(name).copied()
    }

    /// The maximum level over all nodes, or `None` when there are no nodes.
    pub fn max_level(&self) -> Option<usize> {
        self.levels.values().copied().max()
    }

    /// All type names at a given level (unordered).
    #[cfg(test)]
    pub fn types_at_level(&self, level: usize) -> Vec<String> {
        self.levels
            .iter()
            .filter(|&(_, &l)| l == level)
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// The detected circular-dependency groups (each at `max_level + 1`).
    #[cfg(test)]
    pub fn circular_groups(&self) -> &[HashSet<String>] {
        &self.circular_groups
    }

    /// Whether a type belongs to a circular-dependency group.
    pub fn is_circular(&self, name: &str) -> bool {
        self.circular_groups.iter().any(|g| g.contains(name))
    }
}

/// Kahn topological leveling + Tarjan SCC for the residual cycle, replicating
/// `DependencyAnalyzer._compute_levels` / `_detect_circular_groups`.
fn compute_levels(
    dependencies: &HashMap<String, HashSet<String>>,
) -> (HashMap<String, usize>, Vec<HashSet<String>>) {
    let mut levels: HashMap<String, usize> = HashMap::new();

    // in_degree = number of (unresolved) dependencies of each node.
    let mut in_degree: HashMap<String, usize> = dependencies
        .iter()
        .map(|(name, deps)| (name.clone(), deps.len()))
        .collect();

    // Seed the queue with dependency-free nodes (level 0).
    let mut queue: VecDeque<String> = dependencies
        .iter()
        .filter(|(_, deps)| deps.is_empty())
        .map(|(name, _)| name.clone())
        .collect();

    let mut processed: HashSet<String> = HashSet::new();
    let mut current_level = 0usize;
    while !queue.is_empty() {
        let level_size = queue.len();
        for _ in 0..level_size {
            let name = queue.pop_front().unwrap();
            levels.insert(name.clone(), current_level);
            processed.insert(name.clone());
            // Every node depending on `name` loses one unresolved dependency.
            for (other, deps) in dependencies.iter() {
                if deps.contains(&name)
                    && let Some(d) = in_degree.get_mut(other)
                {
                    *d -= 1;
                    if *d == 0 && !processed.contains(other) {
                        queue.push_back(other.clone());
                    }
                }
            }
        }
        current_level += 1;
    }

    // Unprocessed nodes participate in dependency cycles.
    let unprocessed: HashSet<String> = dependencies
        .keys()
        .filter(|n| !processed.contains(*n))
        .cloned()
        .collect();

    let mut circular_groups = Vec::new();
    if !unprocessed.is_empty() {
        circular_groups = detect_circular_groups(dependencies, &unprocessed);
        let max_level = levels.values().copied().max().map_or(-1i64, |m| m as i64);
        let cycle_level = (max_level + 1) as usize;
        for group in &circular_groups {
            for name in group {
                levels.insert(name.clone(), cycle_level);
            }
        }
    }

    (levels, circular_groups)
}

/// Tarjan's strongly-connected-components over the unprocessed sub-graph;
/// returns only the genuinely circular components (size > 1), mirroring molpy.
fn detect_circular_groups(
    dependencies: &HashMap<String, HashSet<String>>,
    unprocessed: &HashSet<String>,
) -> Vec<HashSet<String>> {
    struct Tarjan<'a> {
        deps: &'a HashMap<String, HashSet<String>>,
        unprocessed: &'a HashSet<String>,
        index_counter: usize,
        index: HashMap<String, usize>,
        lowlink: HashMap<String, usize>,
        stack: Vec<String>,
        on_stack: HashSet<String>,
        groups: Vec<HashSet<String>>,
    }

    impl Tarjan<'_> {
        fn strongconnect(&mut self, node: &str) {
            self.index.insert(node.to_string(), self.index_counter);
            self.lowlink.insert(node.to_string(), self.index_counter);
            self.index_counter += 1;
            self.stack.push(node.to_string());
            self.on_stack.insert(node.to_string());

            if let Some(succs) = self.deps.get(node) {
                // Deterministic order for reproducible grouping.
                let mut succs: Vec<&String> = succs.iter().collect();
                succs.sort();
                for other in succs {
                    if !self.unprocessed.contains(other) {
                        continue;
                    }
                    if !self.index.contains_key(other) {
                        self.strongconnect(other);
                        let lo = self.lowlink[other];
                        let cur = self.lowlink[node];
                        self.lowlink.insert(node.to_string(), cur.min(lo));
                    } else if self.on_stack.contains(other) {
                        let oi = self.index[other];
                        let cur = self.lowlink[node];
                        self.lowlink.insert(node.to_string(), cur.min(oi));
                    }
                }
            }

            if self.lowlink[node] == self.index[node] {
                let mut component = HashSet::new();
                loop {
                    let w = self.stack.pop().unwrap();
                    self.on_stack.remove(&w);
                    let is_root = w == node;
                    component.insert(w);
                    if is_root {
                        break;
                    }
                }
                if component.len() > 1 {
                    self.groups.push(component);
                }
            }
        }
    }

    let mut t = Tarjan {
        deps: dependencies,
        unprocessed,
        index_counter: 0,
        index: HashMap::new(),
        lowlink: HashMap::new(),
        stack: Vec::new(),
        on_stack: HashSet::new(),
        groups: Vec::new(),
    };
    // Deterministic node iteration order.
    let mut nodes: Vec<&String> = unprocessed.iter().collect();
    nodes.sort();
    for node in nodes {
        if !t.index.contains_key(node) {
            t.strongconnect(node);
        }
    }
    t.groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::typifier::OplsTypeRow;

    fn row(def: Option<&str>) -> OplsTypeRow {
        OplsTypeRow {
            class: "X".to_string(),
            def: def.map(str::to_string),
            overrides: Vec::new(),
            priority: None,
            layer: 0,
        }
    }

    fn meta_with(rows: &[(&str, Option<&str>)]) -> OplsTypingMetadata {
        let mut m = OplsTypingMetadata::new();
        for (name, def) in rows {
            m.insert(*name, row(*def));
        }
        m
    }

    #[test]
    fn extracts_percent_dependencies() {
        // opls_146's def references %opls_145 -> dependency {opls_145}.
        let meta = meta_with(&[
            (
                "opls_145",
                Some("[C;X3;r6]1[C;X3;r6][C;X3;r6][C;X3;r6][C;X3;r6][C;X3;r6]1"),
            ),
            ("opls_146", Some("H[C;%opls_145]")),
        ]);
        let a = OplsDependencyAnalyzer::new(&meta);
        assert!(a.dependencies_of("opls_145").unwrap().is_empty());
        let d146 = a.dependencies_of("opls_146").unwrap();
        assert_eq!(d146.len(), 1);
        assert!(d146.contains("opls_145"));
    }

    #[test]
    fn dependency_on_a_type_without_def_is_dropped() {
        // %opls_999 references a type with no def -> not a dependency edge.
        let meta = meta_with(&[
            ("opls_999", None), // no def
            ("opls_500", Some("H[C;%opls_999]")),
        ]);
        let a = OplsDependencyAnalyzer::new(&meta);
        // opls_999 is not a node; opls_500 has no valid deps -> level 0.
        assert!(a.dependencies_of("opls_500").unwrap().is_empty());
        assert_eq!(a.level("opls_500"), Some(0));
        assert_eq!(a.level("opls_999"), None);
    }

    #[test]
    fn kahn_levels_three_chain() {
        // A (no dep) -> B (dep A) -> C (dep B): levels 0, 1, 2.
        let meta = meta_with(&[
            ("opls_a", Some("[C;X4]")),
            ("opls_b", Some("H[C;%opls_a]")),
            ("opls_c", Some("[O;%opls_b]")),
        ]);
        let a = OplsDependencyAnalyzer::new(&meta);
        assert_eq!(a.level("opls_a"), Some(0));
        assert_eq!(a.level("opls_b"), Some(1));
        assert_eq!(a.level("opls_c"), Some(2));
        assert_eq!(a.max_level(), Some(2));
        assert_eq!(a.types_at_level(0), vec!["opls_a".to_string()]);
        assert!(a.circular_groups().is_empty());
    }

    #[test]
    fn tarjan_detects_two_cycle_at_max_level_plus_one() {
        // base (level 0) + a two-cycle A<->B that depends on base.
        //   opls_base: no deps                      -> level 0
        //   opls_x: %opls_base, %opls_y             -> cycle
        //   opls_y: %opls_x                          -> cycle
        let meta = meta_with(&[
            ("opls_base", Some("[C;X4]")),
            ("opls_x", Some("[C;%opls_base][O;%opls_y]")),
            ("opls_y", Some("[N;%opls_x]")),
        ]);
        let a = OplsDependencyAnalyzer::new(&meta);
        assert_eq!(a.level("opls_base"), Some(0));
        // One circular group {opls_x, opls_y} at level max_level+1 = 1.
        assert_eq!(a.circular_groups().len(), 1);
        let grp = &a.circular_groups()[0];
        assert!(grp.contains("opls_x") && grp.contains("opls_y"));
        assert_eq!(grp.len(), 2);
        assert!(a.is_circular("opls_x") && a.is_circular("opls_y"));
        assert_eq!(a.level("opls_x"), Some(1));
        assert_eq!(a.level("opls_y"), Some(1));
        assert!(!a.is_circular("opls_base"));
    }

    #[test]
    fn no_deps_all_level_zero() {
        let meta = meta_with(&[
            ("opls_135", Some("[C;X4](C)(H)(H)H")),
            ("opls_140", Some("H[C;X4]")),
        ]);
        let a = OplsDependencyAnalyzer::new(&meta);
        assert_eq!(a.level("opls_135"), Some(0));
        assert_eq!(a.level("opls_140"), Some(0));
        assert_eq!(a.max_level(), Some(0));
    }

    /// A context label is a dependency iff it names a type in the metadata,
    /// whatever the name looks like: `HX = [#1]-[#6;%CX]` depends on `CX`
    /// (no `opls_` prefix) and so sits at level 1.
    #[test]
    fn any_type_name_label_is_a_dependency() {
        let meta = meta_with(&[("CX", Some("[#6;X4]")), ("HX", Some("[#1]-[#6;%CX]"))]);
        let a = OplsDependencyAnalyzer::new(&meta);
        let deps = a.dependencies_of("HX").expect("HX is a node");
        assert!(deps.contains("CX"), "HX depends on CX: {deps:?}");
        assert_eq!(a.level("CX"), Some(0));
        assert_eq!(a.level("HX"), Some(1));
    }
}
