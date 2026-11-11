//! Tests of the charge-equivalence partition, and the literal path walk the
//! histogram engine is checked against.

use super::*;
use crate::core::{Element, Props};
use crate::props;

/// The depth-first walk that scores every simple path out of an atom — the
/// literal port of `scorepath()`, kept as the reference the histogram engine
/// ([`paths`]) is tested against. Exponential in the number of rings in series.
struct PathScorer<'a> {
    flat: &'a Flat,
    geom: &'a [Torsion],
    max_path_length: Option<usize>,
    /// The path currently being walked (`equatom.c`'s `selectelement`).
    path: Vec<usize>,
    /// Membership of `path`, for an O(1) "have I visited this atom?" test.
    on_path: Vec<bool>,
    /// The scores collected for the atom being processed.
    scores: Vec<f64>,
}

impl PathScorer<'_> {
    /// Every path score for one atom, sorted ascending.
    fn scores_for(&mut self, atom: usize) -> &[f64] {
        self.scores.clear();
        self.path.clear();
        self.on_path.iter_mut().for_each(|v| *v = false);
        self.walk(atom);
        // Ascending, as `equatom.c::sort()` leaves them. No NaN can occur: every
        // term is finite and positive.
        self.scores.sort_by(f64::total_cmp);
        &self.scores
    }

    /// Extend the current path by `atom`, recurse into its unvisited neighbours,
    /// and score the path that ends here.
    fn walk(&mut self, atom: usize) {
        self.path.push(atom);
        self.on_path[atom] = true;

        // `-pl`: a path longer than the cap is abandoned *before* it is scored.
        let capped = self
            .max_path_length
            .is_some_and(|max| self.path.len() > max);
        if !capped {
            let degree = self.flat.adj[atom].len();
            for k in 0..degree.min(MAX_CON) {
                let next = self.flat.adj[atom][k];
                if !self.on_path[next] {
                    self.walk(next);
                }
            }
            // The path ending here is scored when the walk reaches a free slot of
            // `con[]` — so an atom with `MAX_CON` neighbours terminates no path.
            if degree < MAX_CON {
                let score = self.score();
                self.scores.push(score);
            }
        }

        self.on_path[atom] = false;
        self.path.pop();
    }

    /// Score the current path: Antechamber Eq. (I), accumulated left to right in
    /// the same order as `scorepath()` accumulates it, so that the exact equality
    /// the classification rests on is the same equality antechamber tests.
    fn score(&self) -> f64 {
        let mut score = 0.0;
        for (j, atom) in self.path.iter().enumerate() {
            let position = (j + 1) as f64 * POSITION_WEIGHT;
            let element = f64::from(self.flat.z[*atom]) * ELEMENT_WEIGHT;
            score += self.coefficient(j) * (position + element);
        }
        score
    }

    /// The E/Z coefficient of one path position (`1.0` unless `-eq 2` found a
    /// restricted torsion spanning it).
    ///
    /// `scorepath()` tests two windows — the four positions *ending* at `j` and the
    /// four *starting* at `j` — each in both directions, and lets the second
    /// override the first. Ported as written.
    fn coefficient(&self, j: usize) -> f64 {
        if self.geom.is_empty() {
            return 1.0;
        }
        let mut coef = 1.0;
        if j >= 3
            && let Some(c) = self.match_torsion(&self.path[j - 3..=j])
        {
            coef = c;
        }
        if self.path.len() - j > 3
            && let Some(c) = self.match_torsion(&self.path[j..=j + 3])
        {
            coef = c;
        }
        coef
    }

    /// The coefficient of the restricted torsion this four-atom window traverses,
    /// in either direction, if any.
    fn match_torsion(&self, window: &[usize]) -> Option<f64> {
        self.geom
            .iter()
            .find(|t| window == t.atoms || window.iter().rev().copied().eq(t.atoms.iter().copied()))
            .map(|t| if t.trans { TRANS_COEF } else { CIS_COEF })
    }
}

/// An atom of element `el` with no coordinates.
fn atom(el: Element) -> Props {
    props! { keys::ATOMIC_NUMBER => el }
}

/// An atom of element `el` at `[x, y, z]`.
fn atom_at(el: Element, [x, y, z]: [f64; 3]) -> Props {
    props! { keys::ATOMIC_NUMBER => el, keys::X => x, keys::Y => y, keys::Z => z }
}

/// Methanol, with the raw `sqm` Mulliken charges antechamber's own run
/// produced (the methyl hydrogens are split 0.053 / 0.098 / 0.053 by the
/// conformer).
fn methanol() -> Atomistic {
    let mut mol = Atomistic::new();
    let c = mol
        .add_atom(atom_at(Element::C, [-0.371, -0.017, -0.004]))
        .unwrap();
    let o = mol
        .add_atom(atom_at(Element::O, [0.893, 0.621, -0.023]))
        .unwrap();
    let h1 = mol
        .add_atom(atom_at(Element::H, [-0.481, -0.582, 0.924]))
        .unwrap();
    let h2 = mol
        .add_atom(atom_at(Element::H, [-1.153, 0.744, -0.061]))
        .unwrap();
    let h3 = mol
        .add_atom(atom_at(Element::H, [-0.455, -0.687, -0.863]))
        .unwrap();
    let ho = mol
        .add_atom(atom_at(Element::H, [1.567, -0.079, 0.028]))
        .unwrap();
    for (a, b) in [(c, o), (c, h1), (c, h2), (c, h3), (o, ho)] {
        mol.add_bond(a, b, props! {}).expect("bond");
    }
    for (a, q) in [
        (c, -0.073),
        (o, -0.326),
        (h1, 0.053),
        (h2, 0.098),
        (h3, 0.053),
        (ho, 0.195),
    ] {
        mol.set_atom(a, keys::CHARGE, q).expect("charge");
    }
    mol
}

#[test]
fn methyl_hydrogens_are_one_class_and_the_hydroxyl_is_not() {
    let mol = methanol();
    let ids: Vec<NodeId> = mol.atoms().map(|(id, _)| id).collect();
    let classes = perceive_equivalence_classes(&mol, EquivalenceOptions::bcc());

    assert_eq!(classes.n_classes(), 4, "C, O, 3×methyl H, hydroxyl H");
    let methyl = classes.class_of(ids[2]).expect("H class");
    assert_eq!(classes.class_of(ids[3]), Some(methyl));
    assert_eq!(classes.class_of(ids[4]), Some(methyl));
    assert_ne!(
        classes.class_of(ids[5]),
        Some(methyl),
        "hydroxyl H is not a methyl H"
    );
}

#[test]
fn off_gives_every_atom_its_own_class() {
    let mol = methanol();
    let classes = perceive_equivalence_classes(&mol, EquivalenceOptions::off());
    assert_eq!(classes.n_classes(), mol.n_atoms());

    assert!(classes.classes().all(|c| c.len() == 1));
}

#[test]
fn path_score_is_blind_to_the_order_of_atoms_along_a_path() {
    // The score depends on a path only through its length and its sum of
    // atomic numbers, which is why automorphism orbits are the wrong engine.
    let mut mol = Atomistic::new();
    let c = mol.add_atom(atom(Element::C)).unwrap();
    let n = mol.add_atom(atom(Element::N)).unwrap();
    let o = mol.add_atom(atom(Element::O)).unwrap();
    mol.add_bond(c, n, props! {}).expect("bond");
    mol.add_bond(n, o, props! {}).expect("bond");
    let flat = Flat::new(&mol);
    let mut scorer = PathScorer {
        flat: &flat,
        geom: &[],
        max_path_length: None,
        path: Vec::new(),
        on_path: vec![false; 3],
        scores: Vec::new(),
    };
    scorer.path = vec![0, 1, 2]; // C–N–O
    let cno = scorer.score();
    scorer.path = vec![0, 2, 1]; // C–O–N (not a real path; the formula only)
    let con = scorer.score();
    assert_eq!(cno.to_bits(), con.to_bits());
}

/// ac-003 — the score comparison is **exact**, and must stay exact.
///
/// Two mathematically-distinct scores are always at least `0.01` apart: a score
/// is `(11·T + 8·S)/100` for integers `T` (positions) and `S` (Σ Z), and
/// `gcd(11, 8) = 1`. So the *only* way two sorted score arrays can differ by
/// less than a tolerance anyone would pick is the floating-point one: the arrays
/// are mathematically **equal** and disagree in the last bit because the paths
/// visit the same atomic numbers in a different order.
///
/// This fixture is exactly that. Two fragments in one graph — antechamber
/// equivalences across a whole file, so a salt's cation and anion are compared
/// too. Both nitrogens see `{O, S}`; both have four paths; the mathematical
/// profiles are identical, the length-3 path being `N–O–Cl` (7 + 8 + 17) in one
/// and `N–S–F` (7 + 16 + 9) in the other — Σ Z = 32 either way. They differ by
/// **4.4e-16**, and antechamber's `!=` keeps them apart.
///
/// The fragments are contrived, deliberately: an exhaustive sweep of every
/// valence-legal fragment up to five heavy atoms finds **no** ordinary molecule
/// where two atoms tie mathematically but not in `f64`. That is a statement
/// about how safe the exact comparison is in practice, not a licence to relax
/// it — a tolerance here would be a silent, unbounded charge-smearing bug, and
/// this test is what stops one being introduced.
#[test]
fn two_atoms_a_tolerance_would_merge_are_kept_apart() {
    let mut mol = Atomistic::new();
    // Fragment 1: Cl–O–N(–S).
    let n1 = mol.add_atom(atom(Element::N)).unwrap();
    let o1 = mol.add_atom(atom(Element::O)).unwrap();
    let cl = mol.add_atom(atom(Element::Cl)).unwrap();
    let s1 = mol.add_atom(atom(Element::S)).unwrap();
    // Fragment 2: F–S–N(–O).
    let n2 = mol.add_atom(atom(Element::N)).unwrap();
    let s2 = mol.add_atom(atom(Element::S)).unwrap();
    let f2 = mol.add_atom(atom(Element::F)).unwrap();
    let o2 = mol.add_atom(atom(Element::O)).unwrap();
    for (a, b) in [(n1, o1), (o1, cl), (n1, s1), (n2, s2), (s2, f2), (n2, o2)] {
        mol.add_bond(a, b, props! {}).expect("bond");
    }

    // The two score arrays are the same length and agree to 4.4e-16 …
    let flat = Flat::new(&mol);
    let mut scorer = PathScorer {
        flat: &flat,
        geom: &[],
        max_path_length: None,
        path: Vec::new(),
        on_path: vec![false; mol.n_atoms()],
        scores: Vec::new(),
    };
    let a: Vec<f64> = scorer.scores_for(0).to_vec();
    let b: Vec<f64> = scorer.scores_for(4).to_vec();
    assert_eq!(a.len(), b.len(), "same number of paths");
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    assert!(worst > 0.0, "the arrays are not identical");
    assert!(
        worst < 1.0e-15,
        "the arrays differ by {worst:e} — below any tolerance one would pick"
    );

    // … and they are still NOT merged.
    let classes = perceive_equivalence_classes(&mol, EquivalenceOptions::bcc());
    assert_ne!(
        classes.class_of(n1),
        classes.class_of(n2),
        "a tolerance-based comparison merged two atoms antechamber keeps apart"
    );
}

#[test]
fn a_path_length_cap_scores_no_longer_path() {
    let mol = methanol();
    let uncapped = perceive_equivalence_classes(&mol, EquivalenceOptions::bcc());
    let capped = perceive_equivalence_classes(
        &mol,
        EquivalenceOptions {
            level: EquivalenceLevel::Paths,
            max_path_length: Some(1),
        },
    );
    // With only the one-atom paths scored, every atom is scored by its element
    // alone: the three methyl H, the hydroxyl H all collapse into one class.
    assert!(capped.n_classes() < uncapped.n_classes());
    assert_eq!(capped.n_classes(), 3, "C | O | every H");
}

/// A flattened graph straight from atomic numbers and bonds, built the way
/// [`Flat::new`] builds it — so it can carry what a molecule rarely does
/// (a self-loop, a doubled bond, a seven-coordinate atom).
fn flat_of(z: &[u8], bonds: &[(usize, usize, BondOrder)], xyz: &[[f64; 3]]) -> Flat {
    let mut adj = vec![Vec::new(); z.len()];
    let mut adj_bond = vec![Vec::new(); z.len()];
    for (k, &(i, j, _)) in bonds.iter().enumerate() {
        adj[i].push(j);
        adj[j].push(i);
        adj_bond[i].push(k);
        adj_bond[j].push(k);
    }
    Flat {
        ids: Vec::new(),
        z: z.to_vec(),
        adj,
        adj_bond,
        xyz: xyz.to_vec(),
        bonds: bonds.to_vec(),
    }
}

/// The reference walk's scores for every atom, as `(bits, count)` histograms.
fn walked(flat: &Flat, geom: &[Torsion], cap: Option<usize>) -> Vec<Vec<(u64, u128)>> {
    let mut scorer = PathScorer {
        flat,
        geom,
        max_path_length: cap,
        path: Vec::new(),
        on_path: vec![false; flat.z.len()],
        scores: Vec::new(),
    };
    (0..flat.z.len())
        .map(|i| {
            let mut hist: Vec<(u64, u128)> = Vec::new();
            for s in scorer.scores_for(i) {
                match hist.last_mut() {
                    Some((bits, n)) if *bits == s.to_bits() => *n += 1,
                    _ => hist.push((s.to_bits(), 1)),
                }
            }
            hist
        })
        .collect()
}

/// The histogram engine's result in the same shape.
fn counted(flat: &Flat, geom: &[Torsion], cap: Option<usize>) -> Vec<Vec<(u64, u128)>> {
    let engine = PathHistograms::new(flat, geom, cap);
    (0..flat.z.len())
        .map(|i| {
            engine
                .histogram(i)
                .into_iter()
                .map(|(bits, n)| (bits, n.as_u128().expect("small counts")))
                .collect()
        })
        .collect()
}

/// A tiny deterministic generator (xorshift64*), so the sweep is reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The histogram engine reproduces the reference walk **bit for bit and count
/// for count** — every atom, every score — on a sweep of random graphs: fused,
/// spiro and bridged ring systems, disconnected fragments, doubled bonds,
/// self-loops, six- and seven-coordinate atoms (the `con[6]` truncation), at
/// `-eq 1` and `-eq 2` (random geometry, random double bonds), with and
/// without a `-pl` cap.
#[test]
fn the_histogram_engine_matches_the_path_walk_exactly() {
    const ELEMENTS: [u8; 9] = [1, 6, 6, 6, 7, 8, 9, 16, 35];
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut with_torsions, mut with_hub, mut with_cycles) = (0, 0, 0);
    for case in 0..1000 {
        let n = 2 + rng.below(9);
        let z: Vec<u8> = (0..n)
            .map(|_| ELEMENTS[rng.below(ELEMENTS.len())])
            .collect();
        let xyz: Vec<[f64; 3]> = (0..n)
            .map(|_| [0, 1, 2].map(|_| rng.below(2000) as f64 / 500.0 - 2.0))
            .collect();
        let order = |rng: &mut Rng| {
            if rng.below(3) == 0 {
                BondOrder::Double
            } else {
                BondOrder::Single
            }
        };
        let mut bonds = Vec::new();
        for i in 1..n {
            // Mostly a tree, sometimes a second fragment.
            if case % 7 != 0 || i != n / 2 {
                let j = rng.below(i);
                bonds.push((j, i, order(&mut rng)));
            }
        }
        for _ in 0..rng.below(5) {
            let (i, j) = (rng.below(n), rng.below(n));
            if i != j || case % 11 == 0 {
                bonds.push((i, j, order(&mut rng)));
            }
        }
        if case % 13 == 0 {
            // A hub bonded to every atom: six or more neighbours exercise the
            // con[] truncation and the no-termination rule.
            for k in 1..n {
                bonds.push((0, k, BondOrder::Single));
            }
        }
        let flat = flat_of(&z, &bonds, &xyz);
        let geom = flat.restricted_torsions();
        with_torsions += usize::from(!geom.is_empty());
        with_hub += usize::from(flat.adj.iter().any(|a| a.len() > MAX_CON));
        with_cycles += usize::from(bonds.len() >= n);
        for cap in [None, Some(0), Some(1), Some(3), Some(5)] {
            for torsions in [&[][..], &geom[..]] {
                assert_eq!(
                    counted(&flat, torsions, cap),
                    walked(&flat, torsions, cap),
                    "case {case}: z={z:?} bonds={bonds:?} cap={cap:?} eq2={}",
                    !torsions.is_empty()
                );
            }
        }
    }
    // The sweep is not vacuous: it reached every branch it claims to.
    assert!(
        with_torsions > 60,
        "{with_torsions} cases had -eq 2 torsions"
    );
    assert!(with_hub > 5, "{with_hub} cases had a 7+-coordinate atom");
    assert!(with_cycles > 100, "{with_cycles} cases had a ring");
}

/// The partition the reference walk's histograms induce, as class-of-row.
fn walked_partition(flat: &Flat, geom: &[Torsion]) -> Vec<usize> {
    let hists = walked(flat, geom, None);
    (0..hists.len())
        .map(|i| (0..=i).find(|&j| hists[j] == hists[i]).unwrap())
        .collect()
}

/// End to end — twin shortcut, block decomposition, tries — on random
/// molecules with methyl-like twin sets, geometry and double bonds:
/// [`perceive_equivalence_classes`] partitions exactly as the walk does, at
/// `-eq 1` and at `-eq 2`.
#[test]
fn perceived_classes_are_the_walk_s_classes() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for case in 0..300 {
        let heavy = 2 + rng.below(6);
        let mut mol = Atomistic::new();
        let mut ids = Vec::new();
        for _ in 0..heavy {
            let el = [Element::C, Element::C, Element::N, Element::O][rng.below(4)];
            let xyz = [0, 1, 2].map(|_| rng.below(2000) as f64 / 500.0 - 2.0);
            ids.push(mol.add_atom(atom_at(el, xyz)).unwrap());
        }
        let mut bonded = std::collections::HashSet::new();
        let mut bond = |mol: &mut Atomistic, i: usize, j: usize, rng: &mut Rng| {
            let key = (i.min(j), i.max(j));
            if i != j && bonded.insert(key) {
                let order = if rng.below(3) == 0 { 2 } else { 1 };
                mol.add_bond(ids[i], ids[j], props! { keys::BOND_TYPE => order })
                    .unwrap();
            }
        };
        for i in 1..heavy {
            let j = rng.below(i);
            bond(&mut mol, i, j, &mut rng);
        }
        for _ in 0..rng.below(3) {
            let (i, j) = (rng.below(heavy), rng.below(heavy));
            bond(&mut mol, i, j, &mut rng);
        }
        // Terminal H / F sets: twins.
        for &carrier in &ids {
            for _ in 0..rng.below(4) {
                let el = if case % 5 == 0 {
                    Element::F
                } else {
                    Element::H
                };
                let xyz = [0, 1, 2].map(|_| rng.below(2000) as f64 / 500.0 - 2.0);
                let h = mol.add_atom(atom_at(el, xyz)).unwrap();
                mol.add_bond(carrier, h, props! {}).unwrap();
            }
        }
        let flat = Flat::new(&mol);
        let all: Vec<NodeId> = mol.atoms().map(|(id, _)| id).collect();
        for level in [EquivalenceLevel::Paths, EquivalenceLevel::PathsAndGeometry] {
            let geom = match level {
                EquivalenceLevel::PathsAndGeometry => flat.restricted_torsions(),
                _ => Vec::new(),
            };
            let classes = perceive_equivalence_classes(
                &mol,
                EquivalenceOptions {
                    level,
                    max_path_length: None,
                },
            );
            let got: Vec<usize> = all
                .iter()
                .map(|id| {
                    let c = classes.class_of(*id).unwrap();
                    all.iter()
                        .position(|other| classes.class_of(*other) == Some(c))
                        .unwrap()
                })
                .collect();
            assert_eq!(got, walked_partition(&flat, &geom), "case {case} {level:?}");
        }
    }
}

/// Poly-p-phenylene: `rings` benzene rings joined para, hydrogens on every
/// free carbon. Atom `6k + j` is carbon `j` of ring `k` (`j = 0` bonds the
/// previous ring, `j = 3` the next); hydrogens follow.
fn polyphenylene(rings: usize) -> Atomistic {
    let mut mol = Atomistic::new();
    let carbons: Vec<NodeId> = (0..6 * rings)
        .map(|_| mol.add_atom(atom(Element::C)).unwrap())
        .collect();
    for k in 0..rings {
        for j in 0..6 {
            let order = if j % 2 == 0 { 2 } else { 1 };
            mol.add_bond(
                carbons[6 * k + j],
                carbons[6 * k + (j + 1) % 6],
                props! { keys::BOND_TYPE => order },
            )
            .unwrap();
        }
        if k + 1 < rings {
            mol.add_bond(carbons[6 * k + 3], carbons[6 * (k + 1)], props! {})
                .unwrap();
        }
    }
    for k in 0..rings {
        for j in 0..6 {
            let joined = (j == 0 && k > 0) || (j == 3 && k + 1 < rings);
            if !joined {
                let h = mol.add_atom(atom(Element::H)).unwrap();
                mol.add_bond(carbons[6 * k + j], h, props! {}).unwrap();
            }
        }
    }
    mol
}

/// On a molecule small enough to walk, the classes are the walk's classes.
#[test]
fn a_short_ring_chain_partitions_as_the_walk_partitions() {
    let mol = polyphenylene(5);
    let flat = Flat::new(&mol);
    assert_eq!(counted(&flat, &[], None), walked(&flat, &[], None));
}

/// Scaling guard: 40 rings in series is `~2^40` simple paths per atom, which
/// the walk could never finish. Counted over the block-cut tree it takes
/// well under a second in release; the bound is loose for debug CI builds.
/// The result must still be chemically right: the chain is mirror-symmetric
/// and each ring's two ortho (and two meta) carbons are equivalent.
#[test]
fn forty_rings_in_series_partition_quickly_and_symmetrically() {
    let rings = 40;
    let mol = polyphenylene(rings);
    let ids: Vec<NodeId> = mol.atoms().map(|(id, _)| id).collect();

    let started = std::time::Instant::now();
    let classes = perceive_equivalence_classes(&mol, EquivalenceOptions::bcc());
    let elapsed = started.elapsed();
    assert!(elapsed.as_secs_f64() < 30.0, "took {elapsed:?}");

    let class = |k: usize, j: usize| classes.class_of(ids[6 * k + j]).unwrap();
    for k in 0..rings {
        assert_eq!(class(k, 1), class(k, 5), "ring {k} ortho");
        assert_eq!(class(k, 2), class(k, 4), "ring {k} meta");
        for j in 0..6 {
            assert_eq!(class(k, j), class(rings - 1 - k, (9 - j) % 6), "mirror");
        }
    }
    assert_ne!(class(0, 0), class(1, 0), "an end ring is not an inner ring");
    assert_ne!(class(1, 0), class(2, 0), "depth separates inner rings");
}

/// A count past `u128` carries into the overflow limbs, exactly.
#[test]
fn path_counts_do_not_overflow() {
    let mut big = counts::PathCount::from_u128(u128::MAX);
    big.add(&counts::PathCount::from_u128(u128::MAX));
    assert_eq!(big.as_u128(), None);
    let mut same = counts::PathCount::from_u128(u128::MAX - 1);
    same.add(&counts::PathCount::from_u128(u128::MAX));
    same.add(&counts::PathCount::from_u128(1));
    assert_eq!(big, same, "2·(2^128 − 1) either way");
    let mut small = counts::PathCount::from_u128(3);
    small.add(&counts::PathCount::from_u128(4));
    assert_eq!(small.as_u128(), Some(7));

    // Scaling is repeated addition, across the u128 boundary too.
    assert_eq!(small.scaled(6).as_u128(), Some(42));
    let mut by_adding = counts::PathCount::from_u128(0);
    for _ in 0..1000 {
        by_adding.add(&big);
    }
    assert_eq!(big.scaled(1000), by_adding);
    assert_eq!(counts::PathCount::from_u128(u128::MAX).scaled(2), big);
}
