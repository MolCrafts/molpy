//! 2D depiction — coordinates for drawing a molecular graph on a page.
//!
//! [`Depictor::layout`] gives every atom of an [`Atomistic`] a position in the
//! `z = 0` plane, laid out the way a chemist sketches a structure: rings as
//! regular polygons (fused rings share an edge, spiro rings an atom), chains
//! as a 120° zigzag, sp centres straight, substituents spread into the free
//! angle around their atom, and disconnected species (a cation and its
//! counter-ions) side by side. It is RDKit's `Compute2DCoords` in purpose,
//! not in algorithm: a deterministic constructive layout followed by branch
//! flips that remove atom clashes. The graph itself is untouched — charges,
//! ports, types and every other property travel with the copy — so a 2D
//! viewer (molvis' sketch) can draw formal charges and attachment points from
//! the same frame [`Atomistic::to_frame`] writes.
//!
//! A layout is geometry for the eye, not for a force field: bond lengths are
//! one uniform `bond_length` and angles idealized.
//!
//! # Examples
//!
//! ```
//! # #[cfg(feature = "smiles")]
//! # {
//! use molrs::depict::Depictor;
//! use molrs::io::smiles::SmilesIr;
//!
//! let benzene = SmilesIr::parse("c1ccccc1")?.to_atomistic()?;
//! let flat = Depictor::default().layout(&benzene)?;
//! let points: Vec<[f64; 3]> = flat
//!     .atoms()
//!     .map(|(_, atom)| atom.position().expect("every atom is placed"))
//!     .collect();
//! let (cx, cy) = (
//!     points.iter().map(|p| p[0]).sum::<f64>() / 6.0,
//!     points.iter().map(|p| p[1]).sum::<f64>() / 6.0,
//! );
//! for [x, y, z] in points {
//!     assert_eq!(z, 0.0);
//!     // A regular hexagon of 1.5 Å sides has a 1.5 Å circumradius.
//!     assert!(((x - cx).hypot(y - cy) - 1.5).abs() < 1e-9);
//! }
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;

use crate::core::{Atomistic, BondOrder, MolRsError, NodeId, keys};
use crate::perceive::perceive_rings;

type P = [f64; 2];

/// The bond length a layout draws when none is chosen: RDKit's 1.5.
const DEFAULT_BOND_LENGTH: f64 = 1.5;

/// Two unbonded atoms closer than this fraction of a bond length clash.
const CLASH_FRACTION: f64 = 0.6;

/// Gap between the bounding boxes of two disconnected species, in bond lengths.
const COMPONENT_GAP: f64 = 2.0;

/// Clash-removal sweeps over the flippable bonds.
const FLIP_PASSES: usize = 3;

/// How far a terminal atom may swing from its ideal bond angle to clear a
/// clash (radians), tried in `TERMINAL_STEPS` steps each way.
const TERMINAL_SWING: f64 = PI / 4.0;
const TERMINAL_STEPS: usize = 9;

/// Lays a molecular graph out in 2D for drawing.
///
/// Construct it with the bond length to draw ([`Default`] is 1.5, the unit
/// RDKit draws in and close to a C–C bond in Å), then call
/// [`layout`](Self::layout).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Depictor {
    bond_length: f64,
}

impl Default for Depictor {
    fn default() -> Self {
        Self {
            bond_length: DEFAULT_BOND_LENGTH,
        }
    }
}

impl Depictor {
    /// A depictor drawing every bond `bond_length` long.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `bond_length` is not a finite positive
    /// number.
    pub fn new(bond_length: f64) -> Result<Self, MolRsError> {
        if !(bond_length.is_finite() && bond_length > 0.0) {
            return Err(MolRsError::validation(format!(
                "Depictor: bond_length must be finite and positive, got {bond_length}"
            )));
        }
        Ok(Self { bond_length })
    }

    /// The bond length this depictor draws.
    pub fn bond_length(&self) -> f64 {
        self.bond_length
    }

    /// A copy of `mol` with 2D coordinates: `x` / `y` from the layout and
    /// `z = 0` on every atom, replacing any coordinates it had.
    ///
    /// Hydrogens present as atoms are laid out like any other atom; remove
    /// them first ([`remove_hydrogens`](crate::perceive::remove_hydrogens))
    /// for a skeletal sketch. Every property and relation of `mol` (formal
    /// charges, ports, types, angles, …) is carried across unchanged.
    ///
    /// # Errors
    ///
    /// [`MolRsError`] when a bond names an atom the molecule does not hold or
    /// a coordinate cannot be written — both a broken graph invariant.
    pub fn layout(&self, mol: &Atomistic) -> Result<Atomistic, MolRsError> {
        let graph = Graph2d::of(mol)?;
        let coords = Layout::new(&graph, self.bond_length).run();
        let mut out = mol.clone();
        for (id, [x, y]) in mol.atom_ids().zip(coords) {
            out.set_atom(id, keys::X, x)?;
            out.set_atom(id, keys::Y, y)?;
            out.set_atom(id, keys::Z, 0.0)?;
        }
        Ok(out)
    }
}

/// The molecule in index space: atom `i` is the `i`-th of `atom_ids`.
struct Graph2d {
    adj: Vec<Vec<usize>>,
    /// Atom is an sp centre drawn straight: two neighbours and a triple bond
    /// or two double bonds.
    linear: Vec<bool>,
    heavy: Vec<bool>,
    /// SSSR rings, each an ordered closed path.
    rings: Vec<Vec<usize>>,
}

impl Graph2d {
    fn of(mol: &Atomistic) -> Result<Self, MolRsError> {
        let ids: Vec<NodeId> = mol.atom_ids().collect();
        let index: HashMap<NodeId, usize> =
            ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let n = ids.len();
        let row = |id: &NodeId| {
            index.get(id).copied().ok_or_else(|| {
                MolRsError::validation(format!("Depictor: bond endpoint {id:?} is not an atom"))
            })
        };
        let mut adj = vec![Vec::new(); n];
        let (mut triple, mut double) = (vec![0_u8; n], vec![0_u8; n]);
        for (bid, bond) in mol.bonds() {
            let (i, j) = (row(&bond.nodes[0])?, row(&bond.nodes[1])?);
            if i == j || adj[i].contains(&j) {
                continue;
            }
            adj[i].push(j);
            adj[j].push(i);
            match mol.bond_type(bid)? {
                BondOrder::Triple => {
                    triple[i] += 1;
                    triple[j] += 1;
                }
                BondOrder::Double => {
                    double[i] += 1;
                    double[j] += 1;
                }
                _ => {}
            }
        }
        for list in &mut adj {
            list.sort_unstable();
        }
        let linear = (0..n)
            .map(|i| adj[i].len() == 2 && (triple[i] > 0 || double[i] > 1))
            .collect();
        let heavy = mol
            .atoms()
            .map(|(_, atom)| atom.atomic_number() != Some(1))
            .collect();
        let rings = perceive_rings(mol)
            .rings()
            .iter()
            .map(|ring| ring.iter().map(row).collect::<Result<Vec<_>, _>>())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            adj,
            linear,
            heavy,
            rings,
        })
    }

    fn n(&self) -> usize {
        self.adj.len()
    }

    /// Rings grouped into ring systems (rings sharing an atom), each a list
    /// of ring indices in ascending order; systems ordered by first ring.
    fn ring_systems(&self) -> Vec<Vec<usize>> {
        let m = self.rings.len();
        let mut parent: Vec<usize> = (0..m).collect();
        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        let mut owner: HashMap<usize, usize> = HashMap::new();
        for (r, ring) in self.rings.iter().enumerate() {
            for &atom in ring {
                if let Some(&other) = owner.get(&atom) {
                    let (a, b) = (find(&mut parent, r), find(&mut parent, other));
                    parent[a.max(b)] = a.min(b);
                } else {
                    owner.insert(atom, r);
                }
            }
        }
        let mut systems: Vec<Vec<usize>> = Vec::new();
        let mut slot: HashMap<usize, usize> = HashMap::new();
        for r in 0..m {
            let root = find(&mut parent, r);
            let s = *slot.entry(root).or_insert_with(|| {
                systems.push(Vec::new());
                systems.len() - 1
            });
            systems[s].push(r);
        }
        systems
    }
}

fn add(a: P, b: P) -> P {
    [a[0] + b[0], a[1] + b[1]]
}
fn sub(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1]]
}
fn scale(a: P, s: f64) -> P {
    [a[0] * s, a[1] * s]
}
fn norm(a: P) -> f64 {
    a[0].hypot(a[1])
}
fn angle(a: P) -> f64 {
    a[1].atan2(a[0])
}
fn polar(theta: f64) -> P {
    [theta.cos(), theta.sin()]
}
fn rotate(a: P, theta: f64) -> P {
    let (s, c) = theta.sin_cos();
    [c * a[0] - s * a[1], s * a[0] + c * a[1]]
}
fn centroid(points: impl Iterator<Item = P>) -> Option<P> {
    let (mut sum, mut count) = ([0.0, 0.0], 0.0);
    for p in points {
        sum = add(sum, p);
        count += 1.0;
    }
    (count > 0.0).then(|| scale(sum, 1.0 / count))
}

/// The angles of `r` new bonds placed evenly in the largest free gap between
/// the `occupied` bond angles (radians); with nothing occupied, `r` bonds
/// spread over the full turn from 30°.
fn spread(occupied: &[f64], r: usize) -> Vec<f64> {
    if occupied.is_empty() {
        let step = 2.0 * PI / r as f64;
        return (0..r).map(|j| PI / 6.0 + step * j as f64).collect();
    }
    let mut sorted: Vec<f64> = occupied.iter().map(|a| a.rem_euclid(2.0 * PI)).collect();
    sorted.sort_by(f64::total_cmp);
    let (mut start, mut gap) = (
        sorted[sorted.len() - 1],
        sorted[0] + 2.0 * PI - sorted[sorted.len() - 1],
    );
    for pair in sorted.windows(2) {
        if pair[1] - pair[0] > gap {
            (start, gap) = (pair[0], pair[1] - pair[0]);
        }
    }
    (0..r)
        .map(|j| start + gap * (j + 1) as f64 / (r + 1) as f64)
        .collect()
}

/// The circumradius of a regular `k`-gon with side `side`.
fn circumradius(k: usize, side: f64) -> f64 {
    side / (2.0 * (PI / k as f64).sin())
}

struct Layout<'g> {
    g: &'g Graph2d,
    bond: f64,
    pos: Vec<Option<P>>,
    /// Sign of the turn that placed each chain atom, for the zigzag.
    turn: Vec<f64>,
    system_of: Vec<Option<usize>>,
    systems: Vec<Vec<usize>>,
    /// The atoms of the component being placed.
    component: Vec<usize>,
}

impl<'g> Layout<'g> {
    fn new(g: &'g Graph2d, bond: f64) -> Self {
        let systems = g.ring_systems();
        let mut system_of = vec![None; g.n()];
        for (s, rings) in systems.iter().enumerate() {
            for &r in rings {
                for &atom in &g.rings[r] {
                    system_of[atom] = Some(s);
                }
            }
        }
        Self {
            g,
            bond,
            pos: vec![None; g.n()],
            turn: vec![1.0; g.n()],
            system_of,
            systems,
            component: Vec::new(),
        }
    }

    fn run(mut self) -> Vec<P> {
        let components = self.components();
        let mut placed: Vec<Vec<usize>> = Vec::with_capacity(components.len());
        for atoms in components {
            self.place_component(&atoms);
            self.remove_clashes(&atoms);
            placed.push(atoms);
        }
        let mut coords: Vec<P> = self.pos.iter().map(|p| p.unwrap_or([0.0, 0.0])).collect();
        pack(&mut coords, &placed, COMPONENT_GAP * self.bond);
        coords
    }

    /// Connected components, each in ascending atom order, ordered by their
    /// first atom.
    fn components(&self) -> Vec<Vec<usize>> {
        let n = self.g.n();
        let mut seen = vec![false; n];
        let mut out = Vec::new();
        for start in 0..n {
            if seen[start] {
                continue;
            }
            seen[start] = true;
            let mut atoms = vec![start];
            let mut queue = VecDeque::from([start]);
            while let Some(a) = queue.pop_front() {
                for &b in &self.g.adj[a] {
                    if !seen[b] {
                        seen[b] = true;
                        atoms.push(b);
                        queue.push_back(b);
                    }
                }
            }
            atoms.sort_unstable();
            out.push(atoms);
        }
        out
    }

    fn place_component(&mut self, atoms: &[usize]) {
        self.component = atoms.to_vec();
        let mut queue = VecDeque::new();
        // Root: the largest ring system, else one end of the longest path so
        // a chain starts at its end and zigzags along its length.
        let root_system = atoms
            .iter()
            .filter_map(|&a| self.system_of[a])
            .max_by_key(|&s| (self.system_atoms(s).len(), std::cmp::Reverse(s)));
        match root_system {
            Some(s) => {
                let local = self.layout_system(s);
                let mut members: Vec<usize> = local.keys().copied().collect();
                members.sort_unstable();
                for &a in &members {
                    self.pos[a] = Some(local[&a]);
                    queue.push_back(a);
                }
            }
            None => {
                let end = self.farthest(atoms[0]);
                self.pos[end] = Some([0.0, 0.0]);
                queue.push_back(end);
            }
        }
        while let Some(a) = queue.pop_front() {
            self.expand(a, &mut queue);
        }
    }

    /// The heavy atom farthest (in bonds) from `start` (any atom when the
    /// component has no heavy one); ties go to the lower index.
    fn farthest(&self, start: usize) -> usize {
        let mut dist: HashMap<usize, usize> = HashMap::from([(start, 0)]);
        let mut queue = VecDeque::from([start]);
        let mut best = (0, start);
        while let Some(a) = queue.pop_front() {
            let d = dist[&a];
            // A heavy atom outranks any hydrogen, then distance, then index.
            let key = (self.g.heavy[a], d);
            let held = (self.g.heavy[best.1], best.0);
            if key > held || (key == held && a < best.1) {
                best = (d, a);
            }
            for &b in &self.g.adj[a] {
                if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(b) {
                    e.insert(d + 1);
                    queue.push_back(b);
                }
            }
        }
        best.1
    }

    /// Atoms reachable from `from` through unplaced atoms, `from` included.
    fn unplaced_weight(&self, from: usize) -> usize {
        let mut seen = vec![from];
        let mut queue = VecDeque::from([from]);
        while let Some(a) = queue.pop_front() {
            for &b in &self.g.adj[a] {
                if self.pos[b].is_none() && !seen.contains(&b) {
                    seen.push(b);
                    queue.push_back(b);
                }
            }
        }
        seen.len()
    }

    /// Place the unplaced neighbours of the placed atom `a`.
    fn expand(&mut self, a: usize, queue: &mut VecDeque<usize>) {
        let here = self.pos[a].expect("only placed atoms are expanded");
        let occupied: Vec<f64> = self.g.adj[a]
            .iter()
            .filter_map(|&b| self.pos[b].map(|p| angle(sub(p, here))))
            .collect();
        let mut children: Vec<usize> = self.g.adj[a]
            .iter()
            .copied()
            .filter(|&b| self.pos[b].is_none())
            .collect();
        if children.is_empty() {
            return;
        }
        // The main branch (heavy, largest) takes the chain direction.
        let mut keyed: Vec<(bool, usize, usize)> = children
            .iter()
            .map(|&b| (self.g.heavy[b], self.unplaced_weight(b), b))
            .collect();
        keyed.sort_by(|x, y| y.0.cmp(&x.0).then(y.1.cmp(&x.1)).then(x.2.cmp(&y.2)));
        children = keyed.into_iter().map(|(_, _, b)| b).collect();

        let mut directions: Vec<(usize, f64, f64)> = Vec::with_capacity(children.len());
        let heavy_children = children.iter().filter(|&&b| self.g.heavy[b]).count();
        if occupied.len() == 1 && heavy_children <= 1 && children.len() <= 3 {
            // A chain atom: the main branch zigzags, terminal atoms fill the
            // larger free angle.
            let forward = occupied[0] + PI;
            let sign = -self.turn[a];
            let main = if self.g.linear[a] {
                forward
            } else {
                forward + sign * PI / 3.0
            };
            directions.push((children[0], main, sign));
            let rest = spread(&[occupied[0], main], children.len() - 1);
            directions.extend(children[1..].iter().zip(rest).map(|(&b, t)| (b, t, -sign)));
        } else if occupied.len() == 1 {
            // A branch point: every bond evenly around the atom, the heaviest
            // branches nearest straight ahead.
            let forward = occupied[0] + PI;
            let mut slots = spread(&occupied, children.len());
            let off = |t: f64| ((t - forward + PI).rem_euclid(2.0 * PI) - PI).abs();
            slots.sort_by(|x, y| off(*x).total_cmp(&off(*y)));
            directions.extend(children.iter().zip(slots).map(|(&b, t)| (b, t, 1.0)));
        } else {
            let angles = spread(&occupied, children.len());
            directions.extend(children.iter().zip(angles).map(|(&b, t)| (b, t, 1.0)));
        }
        for (b, nominal, sign) in directions {
            let theta = if self.g.adj[b].len() == 1 {
                self.clear_angle(a, nominal)
            } else {
                nominal
            };
            let target = add(here, scale(polar(theta), self.bond));
            match self.system_of[b] {
                Some(s) => self.place_system_at(s, b, target, theta, queue),
                None => {
                    self.pos[b] = Some(target);
                    self.turn[b] = sign;
                    queue.push_back(b);
                }
            }
        }
    }

    /// The bond angle nearest `nominal` (within ±`TERMINAL_SWING`) at which a
    /// terminal atom bonded to `a` clears every placed atom of the component;
    /// the angle of greatest clearance when none does.
    fn clear_angle(&self, a: usize, nominal: f64) -> f64 {
        let here = self.pos[a].expect("the parent is placed");
        let threshold = CLASH_FRACTION * self.bond;
        let clearance = |theta: f64| {
            let p = add(here, scale(polar(theta), self.bond));
            self.component
                .iter()
                .filter(|&&m| m != a)
                .filter_map(|&m| self.pos[m])
                .map(|q| norm(sub(q, p)))
                .fold(f64::INFINITY, f64::min)
        };
        let mut best = (clearance(nominal), nominal);
        if best.0 >= threshold {
            return nominal;
        }
        for step in 1..=TERMINAL_STEPS {
            let delta = TERMINAL_SWING * step as f64 / TERMINAL_STEPS as f64;
            for theta in [nominal + delta, nominal - delta] {
                let c = clearance(theta);
                if c >= threshold {
                    return theta;
                }
                if c > best.0 {
                    best = (c, theta);
                }
            }
        }
        best.1
    }

    /// Place ring system `s` rigidly so its atom `anchor` lands on `target`
    /// and the system's centroid lies straight ahead along `theta`.
    fn place_system_at(
        &mut self,
        s: usize,
        anchor: usize,
        target: P,
        theta: f64,
        queue: &mut VecDeque<usize>,
    ) {
        let local = self.layout_system(s);
        let origin = local[&anchor];
        let center = centroid(local.values().copied()).unwrap_or(origin);
        let offset = sub(center, origin);
        let turn = if norm(offset) > 0.0 {
            theta - angle(offset)
        } else {
            0.0
        };
        let mut members: Vec<usize> = local.keys().copied().collect();
        members.sort_unstable_by_key(|&m| (m != anchor, m));
        for m in members {
            self.pos[m] = Some(add(target, rotate(sub(local[&m], origin), turn)));
            queue.push_back(m);
        }
    }

    fn system_atoms(&self, s: usize) -> Vec<usize> {
        let mut atoms: Vec<usize> = self.systems[s]
            .iter()
            .flat_map(|&r| self.g.rings[r].iter().copied())
            .collect();
        atoms.sort_unstable();
        atoms.dedup();
        atoms
    }

    /// Ring system `s` in local coordinates: the most fused ring as a regular
    /// polygon at the origin, then each ring sharing the most placed atoms
    /// built outward on its shared edge (fused), atom (spiro) or path
    /// (bridged).
    fn layout_system(&self, s: usize) -> HashMap<usize, P> {
        let rings: Vec<&Vec<usize>> = self.systems[s].iter().map(|&r| &self.g.rings[r]).collect();
        let shared = |ring: &Vec<usize>| {
            ring.iter()
                .filter(|a| rings.iter().filter(|other| other.contains(a)).count() > 1)
                .count()
        };
        let first = (0..rings.len())
            .max_by_key(|&r| (shared(rings[r]), rings[r].len(), std::cmp::Reverse(r)))
            .expect("a ring system holds a ring");
        let mut pos: HashMap<usize, P> = HashMap::new();
        let k = rings[first].len();
        let radius = circumradius(k, self.bond);
        for (t, &atom) in rings[first].iter().enumerate() {
            let theta = PI / 2.0 + PI / k as f64 + 2.0 * PI * t as f64 / k as f64;
            pos.insert(atom, scale(polar(theta), radius));
        }
        let mut done = vec![false; rings.len()];
        done[first] = true;
        while let Some(next) = (0..rings.len())
            .filter(|&r| !done[r])
            .map(|r| (rings[r].iter().filter(|a| pos.contains_key(a)).count(), r))
            .filter(|&(count, _)| count > 0)
            .max_by_key(|&(count, r)| (count, std::cmp::Reverse(r)))
            .map(|(_, r)| r)
        {
            done[next] = true;
            while self.attach_ring(rings[next], &mut pos) {}
        }
        pos
    }

    /// Place one unplaced stretch of `ring` from the placed atoms around it;
    /// `false` once the whole ring is placed.
    fn attach_ring(&self, ring: &[usize], pos: &mut HashMap<usize, P>) -> bool {
        let k = ring.len();
        let placed: Vec<bool> = ring.iter().map(|a| pos.contains_key(a)).collect();
        let Some(last) = (0..k).find(|&i| placed[i] && !placed[(i + 1) % k]) else {
            return false;
        };
        let stretch: Vec<usize> = (1..k)
            .map(|step| (last + step) % k)
            .take_while(|&i| !placed[i])
            .map(|i| ring[i])
            .collect();
        let q = pos[&ring[last]];
        let p = pos[&ring[(last + stretch.len() + 1) % k]];
        let away = centroid(pos.values().copied()).unwrap_or(q);
        let radius = circumradius(k, self.bond);
        if ring[last] == ring[(last + stretch.len() + 1) % k] {
            // Spiro: one shared atom; the new ring sits opposite the placed ones.
            let out = sub(q, away);
            let dir = if norm(out) > 0.0 {
                scale(out, 1.0 / norm(out))
            } else {
                [1.0, 0.0]
            };
            let center = add(q, scale(dir, radius));
            let start = angle(sub(q, center));
            for (t, &atom) in stretch.iter().enumerate() {
                let theta = start + 2.0 * PI * (t + 1) as f64 / k as f64;
                pos.insert(atom, add(center, scale(polar(theta), radius)));
            }
            return true;
        }
        // Fused or bridged: a circle through both placed ends, centred on
        // the side away from the placed atoms; the stretch fills its far arc.
        let chord = sub(p, q);
        let half = norm(chord) / 2.0;
        let r = radius.max(half * 1.000_001);
        let mid = add(q, scale(chord, 0.5));
        let normal = if half > 0.0 {
            [-chord[1] / (2.0 * half), chord[0] / (2.0 * half)]
        } else {
            [0.0, 1.0]
        };
        let depth = (r * r - half * half).max(0.0).sqrt();
        let side = if (away[0] - mid[0]) * normal[0] + (away[1] - mid[1]) * normal[1] > 0.0 {
            -1.0
        } else {
            1.0
        };
        let center = add(mid, scale(normal, side * depth));
        let (aq, ap) = (angle(sub(q, center)), angle(sub(p, center)));
        let ccw = (ap - aq).rem_euclid(2.0 * PI);
        let far = |sweep: f64| {
            let probe = add(center, scale(polar(aq + sweep / 2.0), r));
            norm(sub(probe, away))
        };
        let sweep = if far(ccw) >= far(ccw - 2.0 * PI) {
            ccw
        } else {
            ccw - 2.0 * PI
        };
        let m = stretch.len();
        for (t, &atom) in stretch.iter().enumerate() {
            let theta = aq + sweep * (t + 1) as f64 / (m + 1) as f64;
            pos.insert(atom, add(center, scale(polar(theta), r)));
        }
        true
    }

    /// Mirror branches across acyclic bonds while that lowers the clash
    /// score of the component.
    fn remove_clashes(&mut self, atoms: &[usize]) {
        let threshold = CLASH_FRACTION * self.bond;
        let mut score = self.clash_score(atoms, threshold);
        if score == 0.0 {
            return;
        }
        let ring_bond = |i: usize, j: usize| {
            self.g
                .rings
                .iter()
                .any(|ring| ring.contains(&i) && ring.contains(&j))
        };
        let mut bonds: Vec<(usize, usize)> = Vec::new();
        for &i in atoms {
            for &j in &self.g.adj[i] {
                if i < j && self.g.adj[i].len() > 1 && self.g.adj[j].len() > 1 && !ring_bond(i, j) {
                    bonds.push((i, j));
                }
            }
        }
        for _ in 0..FLIP_PASSES {
            let mut improved = false;
            for &(i, j) in &bonds {
                let side = self.side_of(j, i);
                let saved: Vec<(usize, Option<P>)> =
                    side.iter().map(|&a| (a, self.pos[a])).collect();
                self.mirror(&side, i, j);
                let trial = self.clash_score(atoms, threshold);
                if trial + 1e-12 < score {
                    score = trial;
                    improved = true;
                } else {
                    for (a, p) in saved {
                        self.pos[a] = p;
                    }
                }
                if score == 0.0 {
                    return;
                }
            }
            if !improved {
                return;
            }
        }
    }

    /// Atoms on `j`'s side of the bond `i`–`j` (`j` included).
    fn side_of(&self, j: usize, i: usize) -> Vec<usize> {
        let mut seen = vec![j];
        let mut queue = VecDeque::from([j]);
        while let Some(a) = queue.pop_front() {
            for &b in &self.g.adj[a] {
                if b != i && !seen.contains(&b) {
                    seen.push(b);
                    queue.push_back(b);
                }
            }
        }
        seen
    }

    /// Reflect `atoms` across the line through atoms `i` and `j`.
    fn mirror(&mut self, atoms: &[usize], i: usize, j: usize) {
        let (Some(a), Some(b)) = (self.pos[i], self.pos[j]) else {
            return;
        };
        let axis = sub(b, a);
        let len = norm(axis);
        if len == 0.0 {
            return;
        }
        let u = scale(axis, 1.0 / len);
        for &m in atoms {
            if let Some(p) = self.pos[m] {
                let d = sub(p, a);
                let along = d[0] * u[0] + d[1] * u[1];
                let foot = add(a, scale(u, along));
                self.pos[m] = Some(sub(scale(foot, 2.0), p));
            }
        }
    }

    /// Sum of squared overlaps of unbonded atom pairs closer than
    /// `threshold`, found on a grid of `threshold`-sized cells.
    fn clash_score(&self, atoms: &[usize], threshold: f64) -> f64 {
        let cell = |p: P| {
            (
                (p[0] / threshold).floor() as i64,
                (p[1] / threshold).floor() as i64,
            )
        };
        let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for &a in atoms {
            if let Some(p) = self.pos[a] {
                grid.entry(cell(p)).or_default().push(a);
            }
        }
        let mut score = 0.0;
        for &a in atoms {
            let Some(p) = self.pos[a] else { continue };
            let (cx, cy) = cell(p);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let Some(bucket) = grid.get(&(cx + dx, cy + dy)) else {
                        continue;
                    };
                    for &b in bucket {
                        if b <= a || self.g.adj[a].contains(&b) {
                            continue;
                        }
                        let d = norm(sub(self.pos[b].expect("gridded atoms are placed"), p));
                        if d < threshold {
                            score += (threshold - d) * (threshold - d);
                        }
                    }
                }
            }
        }
        score
    }
}

/// Lay the components out left to right, `gap` apart, each centred on
/// `y = 0`.
fn pack(coords: &mut [P], components: &[Vec<usize>], gap: f64) {
    let mut cursor = 0.0;
    for atoms in components {
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for &a in atoms {
            for d in 0..2 {
                lo[d] = lo[d].min(coords[a][d]);
                hi[d] = hi[d].max(coords[a][d]);
            }
        }
        let shift = [cursor - lo[0], -(lo[1] + hi[1]) / 2.0];
        for &a in atoms {
            coords[a] = add(coords[a], shift);
        }
        cursor += hi[0] - lo[0] + gap;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BondNumber, Element};
    use crate::props;

    const L: f64 = 1.5;

    fn xy(mol: &Atomistic) -> Vec<[f64; 3]> {
        mol.atoms()
            .map(|(_, a)| a.position().expect("placed"))
            .collect()
    }

    fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    fn chain(n: usize) -> Atomistic {
        let mut mol = Atomistic::new();
        let ids: Vec<_> = (0..n)
            .map(|_| {
                mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
                    .unwrap()
            })
            .collect();
        for w in ids.windows(2) {
            mol.add_bond(w[0], w[1], props! {}).unwrap();
        }
        mol
    }

    fn ring(mol: &mut Atomistic, n: usize) -> Vec<NodeId> {
        let ids: Vec<_> = (0..n)
            .map(|_| {
                mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
                    .unwrap()
            })
            .collect();
        for i in 0..n {
            mol.add_bond(ids[i], ids[(i + 1) % n], props! {}).unwrap();
        }
        ids
    }

    fn min_unbonded_distance(mol: &Atomistic) -> f64 {
        let ids: Vec<_> = mol.atom_ids().collect();
        let p = xy(mol);
        let mut best = f64::INFINITY;
        for i in 0..ids.len() {
            for j in i + 1..ids.len() {
                if mol.bond_between(ids[i], ids[j]).is_none() {
                    best = best.min(dist(p[i], p[j]));
                }
            }
        }
        best
    }

    #[test]
    fn every_bond_is_drawn_at_the_bond_length_in_the_plane() {
        let mut mol = Atomistic::new();
        let a = ring(&mut mol, 6);
        let b = ring(&mut mol, 5);
        mol.add_bond(a[0], b[0], props! {}).unwrap();
        let out = Depictor::default().layout(&mol).unwrap();
        let p = xy(&out);
        let ids: Vec<_> = out.atom_ids().collect();
        for (_, bond) in out.bonds() {
            let i = ids.iter().position(|&x| x == bond.nodes[0]).unwrap();
            let j = ids.iter().position(|&x| x == bond.nodes[1]).unwrap();
            assert!((dist(p[i], p[j]) - L).abs() < 1e-9, "bond {i}-{j}");
        }
        assert!(p.iter().all(|q| q[2] == 0.0));
    }

    #[test]
    fn a_chain_zigzags_at_120_degrees() {
        let out = Depictor::default().layout(&chain(5)).unwrap();
        let p = xy(&out);
        for w in p.windows(3) {
            // Law of cosines: two bonds at 120° span sqrt(3) bond lengths.
            assert!((dist(w[0], w[2]) - L * 3f64.sqrt()).abs() < 1e-9);
        }
        // Zigzag, not a coil: the ends of a 5-chain are 4 * L * cos(30°) apart.
        assert!((dist(p[0], p[4]) - 4.0 * L * (PI / 6.0).cos()).abs() < 1e-9);
    }

    #[test]
    fn naphthalene_is_two_regular_hexagons_sharing_an_edge() {
        let mut mol = Atomistic::new();
        let a = ring(&mut mol, 6);
        let mut b = vec![a[0], a[1]];
        for _ in 0..4 {
            b.push(
                mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
                    .unwrap(),
            );
        }
        for i in 1..6 {
            if mol.bond_between(b[i], b[(i + 1) % 6]).is_none() {
                mol.add_bond(b[i], b[(i + 1) % 6], props! {}).unwrap();
            }
        }
        let out = Depictor::default().layout(&mol).unwrap();
        assert!((min_unbonded_distance(&out) - L * 3f64.sqrt()).abs() < 1e-6);
        let p = xy(&out);
        let ids: Vec<_> = out.atom_ids().collect();
        for (_, bond) in out.bonds() {
            let i = ids.iter().position(|&x| x == bond.nodes[0]).unwrap();
            let j = ids.iter().position(|&x| x == bond.nodes[1]).unwrap();
            assert!((dist(p[i], p[j]) - L).abs() < 1e-9);
        }
    }

    #[test]
    fn a_triple_bond_centre_is_drawn_straight() {
        let mut mol = chain(4);
        let ids: Vec<_> = mol.atom_ids().collect();
        let mid = mol.bond_between(ids[1], ids[2]).unwrap();
        mol.set_bond_class(mid, BondOrder::Triple, BondNumber::Triple)
            .unwrap();
        let p = xy(&Depictor::default().layout(&mol).unwrap());
        assert!((dist(p[0], p[3]) - 3.0 * L).abs() < 1e-9, "{p:?}");
    }

    #[test]
    fn disconnected_species_do_not_overlap() {
        let mut mol = chain(3);
        let o = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::O, keys::FORMAL_CHARGE => -1 })
            .unwrap();
        let h = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        mol.add_bond(o, h, props! {}).unwrap();
        let out = Depictor::default().layout(&mol).unwrap();
        let p = xy(&out);
        for carbon in &p[..3] {
            for ion in &p[3..] {
                assert!(dist(*carbon, *ion) >= COMPONENT_GAP * L - 1e-9);
            }
        }
        assert_eq!(
            out.get_atom(o).unwrap().get_int(keys::FORMAL_CHARGE),
            Some(-1)
        );
    }

    #[test]
    fn a_crowded_centre_keeps_its_branches_apart() {
        // Neopentane with explicit hydrogens: 17 atoms around one centre.
        let mut mol = Atomistic::new();
        let c = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        for _ in 0..4 {
            let m = mol
                .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
                .unwrap();
            mol.add_bond(c, m, props! {}).unwrap();
            for _ in 0..3 {
                let h = mol
                    .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
                    .unwrap();
                mol.add_bond(m, h, props! {}).unwrap();
            }
        }
        let out = Depictor::default().layout(&mol).unwrap();
        let closest = min_unbonded_distance(&out);
        assert!(
            closest > CLASH_FRACTION * L,
            "closest unbonded pair {closest} in {:?}",
            xy(&out)
        );
    }

    #[test]
    fn a_bond_length_must_be_positive_and_finite() {
        assert!(Depictor::new(0.0).is_err());
        assert!(Depictor::new(f64::NAN).is_err());
        assert_eq!(Depictor::new(2.0).unwrap().bond_length(), 2.0);
    }
}
