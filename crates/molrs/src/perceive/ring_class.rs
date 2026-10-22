//! antechamber's ring perception and ring classes (`ring.c`).

/// One slot of antechamber's ring table (`RING`): the ring's atoms (sorted
/// ascending once purified) and its size, `0` for an empty or removed slot.
#[derive(Debug, Clone, Copy)]
pub struct AntechamberRingMembership {
    /// The ring's atoms; only the first [`num`](Self::num) are meaningful.
    pub atoms: [usize; 12],
    /// The ring size, or `0`.
    pub num: usize,
}

impl Default for AntechamberRingMembership {
    fn default() -> Self {
        Self {
            atoms: [usize::MAX; 12],
            num: 0,
        }
    }
}

impl AntechamberRingMembership {
    /// The ring's atoms.
    pub fn members(&self) -> &[usize] {
        &self.atoms[..self.num]
    }
}

/// One atom's ring facts — antechamber's `AROM`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AntechamberRingSummary {
    /// `rg[0]` — rings the atom is in; `rg[n]` — rings of size `n`.
    pub rg: [i32; 11],
    /// `1` when the atom is in no classified ring.
    pub nr: i32,
    /// `ar1` … `ar5`: how many rings of each class the atom is in.
    pub ar: [i32; 5],
}

/// The ring table and the per-atom facts `ringdetect` leaves behind.
#[derive(Debug, Clone)]
pub struct RingClasses {
    /// Every slot of the ring table, empty and stale ones included — the slots
    /// antechamber's later passes iterate over.
    pub rings: Vec<AntechamberRingMembership>,
    /// Per atom, in index order.
    pub atoms: Vec<AntechamberRingSummary>,
}

/// antechamber's `ringdetect` on a molecule in index space.
///
/// antechamber does not use a smallest set of smallest rings. `ringdetect`
/// walks paths of up to ten ring-capable atoms (C with three or four
/// connections, N, O and S with at least two, P) from every such atom, through
/// the first four neighbours of each, and keeps every cycle of 3 … 10 atoms
/// without a chord (`purify`). Each ring is then classed (`aromatic`) from the
/// connection counts of its atoms and the current bond types:
///
/// * **AR5** — every atom an sp3 carbon;
/// * **AR4** — any sp3 carbon, four-connected P, or three-or-more-connected S;
/// * **AR3** — planar, with a double bond (2 or 8) from a ring atom to an atom
///   in no ring;
/// * **AR1** — a six-ring of six planar atoms whose every N / P carries a
///   double or aromatic bond (2, 8, 10);
/// * **AR2** — any other ring of enough planar atoms;
/// * **AR4** — what is left.
///
/// For AM1-BCC (`bondtype`, and `atomtype` under the BCC and ABCG2 tables) the
/// five-ring of an indole-like fused system is then taken out of AR2: every atom
/// in both a five- and a six-ring and in two AR1 / AR2 rings knocks one off the
/// AR2 count of every atom of each five-ring it is in (the count can go
/// negative, as antechamber's does).
///
/// Selenophene's ring holds an Se, which no path may cross, so it is no ring at
/// all here; cubane's faces are rings, its body diagonals are not.
///
/// # Provenance
///
/// A transcription of AmberTools' `antechamber/ring.c` (`ring_detect_cycle`,
/// `purify`, `ringproperty`, `aromatic`, `ringdetect`), including its ring table
/// bookkeeping: `purify` compacts the table without shrinking its count, so a
/// ring the compaction moved forward can be counted again from its old slot.
/// That is antechamber's arithmetic, and the `RG` / `AR` counts its atom types
/// are matched against.
///
/// # Arguments
///
/// * `z` — atomic numbers.
/// * `con` — every atom's neighbours, in the order its bonds are listed.
/// * `bonds` — `(i, j, antechamber bond type)` per bond, in bond order.
/// * `bcc` — apply the AM1-BCC indole rule (`ringdetect(…, 1)`).
pub fn perceive_ring_classes(
    z: &[u8],
    con: &[Vec<usize>],
    bonds: &[(usize, usize, i32)],
    bcc: bool,
) -> RingClasses {
    let n = z.len();
    let walker = Walker { z, con };

    // The counting pass sizes the table; the storing pass fills it, purifying
    // after every start atom.
    let mut count = 0;
    let mut path = vec![0usize; 12];
    for start in 0..n {
        if walker.ring_capable(start) {
            walker.walk(&mut path, 0, start, &mut count, None);
        }
    }
    let mut rings = vec![AntechamberRingMembership::default(); count];
    let mut used = 0;
    for start in 0..n {
        if walker.ring_capable(start) {
            walker.walk(&mut path, 0, start, &mut used, Some(&mut rings[..]));
            purify(con, &mut rings[..used]);
        }
    }
    purify(con, &mut rings[..used]);

    let mut atoms = vec![
        AntechamberRingSummary {
            nr: 1,
            ..AntechamberRingSummary::default()
        };
        n
    ];
    for ring in &rings {
        for a in ring.members() {
            atoms[*a].rg[0] += 1;
            atoms[*a].rg[ring.num] += 1;
        }
    }
    classify(z, con, bonds, &rings, &mut atoms);

    if bcc {
        for i in 0..n {
            let f = atoms[i];
            if f.rg[5] >= 1 && f.rg[6] >= 1 && f.ar[0] + f.ar[1] >= 2 {
                for ring in rings.iter().filter(|r| r.num == 5) {
                    if ring.members().contains(&i) {
                        for a in ring.members() {
                            atoms[*a].ar[1] -= 1;
                        }
                    }
                }
            }
        }
    }
    RingClasses { rings, atoms }
}

/// The path search of `ring_detect_cycle`.
struct Walker<'a> {
    z: &'a [u8],
    con: &'a [Vec<usize>],
}

impl Walker<'_> {
    /// An atom a ring path may visit.
    fn ring_capable(&self, a: usize) -> bool {
        let degree = self.con[a].len();
        match self.z[a] {
            6 => degree > 2,
            8 | 16 => degree != 1,
            7 | 15 => true,
            _ => false,
        }
    }

    /// Extend the path `path[..len]` by `atom`, recording (when `store` is
    /// given) every ring that closes back onto the path's first atom.
    fn walk(
        &self,
        path: &mut [usize],
        len: usize,
        atom: usize,
        ringnum: &mut usize,
        mut store: Option<&mut [AntechamberRingMembership]>,
    ) {
        path[len] = atom;
        let len = len + 1;
        for i in 0..4 {
            let Some(&next) = self.con[atom].get(i) else {
                return;
            };
            if !self.ring_capable(next) || path[..len].contains(&next) {
                continue;
            }
            if len > 10 {
                return;
            }
            if (2..10).contains(&len) && self.con[path[0]].iter().take(4).any(|c| *c == next) {
                if let Some(rings) = store.as_deref_mut() {
                    let slot = &mut rings[*ringnum];
                    slot.atoms[..len].copy_from_slice(&path[..len]);
                    slot.atoms[len] = next;
                    slot.num = len + 1;
                }
                *ringnum += 1;
            }
            self.walk(path, len, next, ringnum, store.as_deref_mut());
        }
    }
}

/// `purify`: sort each ring's atoms, drop duplicates and rings with a chord
/// (an atom with three ring neighbours), and compact the survivors to the front
/// — leaving the slots behind them as they were.
fn purify(con: &[Vec<usize>], rings: &mut [AntechamberRingMembership]) {
    if rings.is_empty() {
        return;
    }
    for ring in rings.iter_mut() {
        let num = ring.num;
        ring.atoms[..num].sort_unstable();
    }
    for i in 0..rings.len() {
        for j in i + 1..rings.len() {
            if rings[i].num == rings[j].num
                && rings[i].num != 0
                && rings[i].members() == rings[j].members()
            {
                rings[j].num = 0;
            }
        }
    }
    for ring in rings.iter_mut() {
        let members = ring.members();
        let chord = members.iter().any(|a| {
            con[*a]
                .iter()
                .take(6)
                .filter(|c| members.contains(c))
                .count()
                == 3
        });
        if chord {
            ring.num = 0;
        }
    }
    let kept: Vec<AntechamberRingMembership> =
        rings.iter().filter(|r| r.num != 0).copied().collect();
    for (slot, ring) in rings.iter_mut().zip(kept) {
        slot.num = ring.num;
        slot.atoms[..ring.num].copy_from_slice(ring.members());
    }
}

/// `aromatic`: the class of every ring slot, counted onto its atoms.
fn classify(
    z: &[u8],
    con: &[Vec<usize>],
    bonds: &[(usize, usize, i32)],
    rings: &[AntechamberRingMembership],
    atoms: &mut [AntechamberRingSummary],
) {
    // `initarom`: 2 for an atom that can be planar with a π bond, 1 for a lone
    // pair donor, negative for a saturated centre.
    let init: Vec<i32> = (0..z.len())
        .map(|a| {
            let degree = con[a].len();
            match z[a] {
                6 if degree == 3 => 2,
                6 if degree == 4 => -2,
                7 if degree <= 3 => 2,
                8 if degree == 2 => 1,
                15 if degree == 2 => 2,
                15 if degree == 3 => 1,
                15 if degree >= 4 => -1,
                16 if degree == 2 => 1,
                16 if degree >= 3 => -1,
                _ => 0,
            }
        })
        .collect();
    let in_ring: Vec<bool> = atoms.iter().map(|f| f.rg[0] > 0).collect();
    const AR1: usize = 0;
    const AR2: usize = 1;
    const AR3: usize = 2;
    const AR4: usize = 3;
    const AR5: usize = 4;

    for ring in rings {
        let num = ring.num as i32;
        let members = ring.members();
        let total: i32 = members.iter().map(|a| init[*a]).sum();
        let mark = |atoms: &mut [AntechamberRingSummary], class: usize| {
            for a in members {
                atoms[*a].ar[class] += 1;
            }
        };
        if total == -2 * num {
            mark(atoms, AR5);
            continue;
        }
        if members.iter().any(|a| init[*a] < 0) {
            mark(atoms, AR4);
            continue;
        }
        if total >= num && total <= 2 * num {
            let exocyclic_double = bonds.iter().any(|(i, j, t)| {
                let out = usize::from(members.contains(i) && !in_ring[*j])
                    + usize::from(members.contains(j) && !in_ring[*i]);
                out == 1 && matches!(t, 2 | 8)
            });
            if exocyclic_double {
                mark(atoms, AR3);
                continue;
            }
        }
        if total == 12 && num == 6 {
            let pi_bonded = |a: usize| {
                bonds
                    .iter()
                    .any(|(i, j, t)| (*i == a || *j == a) && matches!(t, 2 | 8 | 10))
            };
            let pure = members
                .iter()
                .filter(|a| matches!(z[**a], 7 | 15))
                .all(|a| pi_bonded(*a));
            if pure {
                mark(atoms, AR1);
                continue;
            }
        }
        if total >= num + 3 {
            mark(atoms, AR2);
            continue;
        }
        mark(atoms, AR4);
    }
    for f in atoms.iter_mut() {
        if f.ar.iter().any(|c| *c > 0) {
            f.nr = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Index-space input: neighbours in bond order, every bond typed `1`
    /// unless `types` says otherwise.
    fn classes(z: &[u8], bonds: &[(usize, usize)], types: &[i32], bcc: bool) -> RingClasses {
        let mut con = vec![Vec::new(); z.len()];
        for (i, j) in bonds {
            con[*i].push(*j);
            con[*j].push(*i);
        }
        let typed: Vec<(usize, usize, i32)> = bonds
            .iter()
            .enumerate()
            .map(|(k, (i, j))| (*i, *j, types.get(k).copied().unwrap_or(1)))
            .collect();
        perceive_ring_classes(z, &con, &typed, bcc)
    }

    /// Carbazole's heavy atoms and the hydrogens that make the ring atoms
    /// three-connected, in the GAFF benchmark's mol2 order.
    fn carbazole() -> (Vec<u8>, Vec<(usize, usize)>) {
        let mut z = vec![6u8; 13];
        z[6] = 7;
        z.extend([1u8; 9]);
        let bonds = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (0, 5),
            (4, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (7, 12),
            (3, 12),
            (0, 13),
            (1, 14),
            (2, 15),
            (5, 16),
            (6, 17),
            (8, 18),
            (9, 19),
            (10, 20),
            (11, 21),
        ];
        (z, bonds)
    }

    #[test]
    fn carbazole_has_its_three_rings_and_no_perimeter() {
        let (z, bonds) = carbazole();
        let c = classes(&z, &bonds, &[], false);
        let mut rings: Vec<Vec<usize>> = c
            .rings
            .iter()
            .filter(|r| r.num > 0)
            .map(|r| r.members().to_vec())
            .collect();
        rings.sort();
        assert_eq!(
            rings,
            vec![
                vec![0, 1, 2, 3, 4, 5],
                vec![3, 4, 6, 7, 12],
                vec![7, 8, 9, 10, 11, 12]
            ]
        );
    }

    #[test]
    fn the_bcc_indole_rule_takes_carbazoles_five_ring_out_of_ar2() {
        // `atomtype -p bcc` (AmberTools 26.1): the benzene atoms AR1, the
        // five-ring's AR2 knocked to 0 by the first fused atom it meets.
        let (z, bonds) = carbazole();
        let gaff = classes(&z, &bonds, &[], false);
        let bcc = classes(&z, &bonds, &[], true);
        assert_eq!(gaff.atoms[6].ar, [0, 1, 0, 0, 0], "N: AR2");
        assert_eq!(bcc.atoms[6].ar, [0, 0, 0, 0, 0], "N: not AR2 under BCC");
        assert_eq!(bcc.atoms[3].ar, [1, 0, 0, 0, 0], "fused C: AR1 only");
        assert_eq!(
            (bcc.atoms[3].rg[0], bcc.atoms[3].rg[5], bcc.atoms[3].rg[6]),
            (2, 1, 1)
        );
    }

    #[test]
    fn the_bcc_indole_rule_can_take_ar2_below_zero() {
        // Acenaphthylene's five-ring meets two fused atoms that still qualify,
        // so its CH carbons end at AR2 = -1 — and `apcheck` reads a bare
        // `AR2` as satisfied by exactly -1.
        let mut z = vec![6u8; 12];
        z.extend([1u8; 8]);
        let bonds = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (0, 10),
            (10, 11),
            (2, 11),
            (6, 11),
            (0, 12),
            (1, 13),
            (3, 14),
            (4, 15),
            (5, 16),
            (7, 17),
            (8, 18),
            (9, 19),
        ];
        let c = classes(&z, &bonds, &[], true);
        assert_eq!(c.atoms[0].ar[1], -1);
        assert_eq!(c.atoms[11].ar, [2, -1, 0, 0, 0]);
    }

    #[test]
    fn an_exocyclic_double_bond_makes_a_planar_ring_ar3() {
        // 1,4-benzoquinone with its C=O stated double: AR3. With every bond
        // single (a connectivity-only file) the same ring is AR1.
        let z = [8u8, 6, 6, 6, 6, 8, 6, 6, 1, 1, 1, 1];
        let bonds = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (4, 6),
            (6, 7),
            (1, 7),
            (2, 8),
            (3, 9),
            (6, 10),
            (7, 11),
        ];
        let stated = [2, 1, 2, 1, 2, 1, 2, 1];
        assert_eq!(
            classes(&z, &bonds, &stated, false).atoms[1].ar,
            [0, 0, 1, 0, 0]
        );
        assert_eq!(classes(&z, &bonds, &[], false).atoms[1].ar, [1, 0, 0, 0, 0]);
    }

    #[test]
    fn a_ring_through_selenium_is_no_ring() {
        let z = [6u8, 6, 6, 34, 6, 1, 1, 1, 1];
        let bonds = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 4),
            (0, 5),
            (1, 6),
            (2, 7),
            (4, 8),
        ];
        let c = classes(&z, &bonds, &[], false);
        assert!(c.atoms.iter().all(|f| f.rg[0] == 0 && f.nr == 1));
    }
}
