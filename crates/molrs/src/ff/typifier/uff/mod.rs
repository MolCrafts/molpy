use std::collections::{HashMap, HashSet};

use molrs::core::PropValue;
use molrs::core::TypeName;
use molrs::core::schema::block_names::{ANGLES, BONDS, DIHEDRALS, IMPROPERS};
use molrs::core::{Atomistic, Element, NodeId, Props, keys};

use crate::core::NeighborKinds;
use crate::core::constants::UFF_COULOMB;
use crate::ff::forcefield::{DefError, ForceField};
use crate::ff::ir::{Params, SpecialBonds};
use crate::ff::params::uff::{AtomicParams, LAMBDA, params_for_label};
use crate::ff::typifier::{Annotation, TypeAssignment, Typifier, TypingError};
use crate::line_notation::atom_keys;
use crate::perceive::assign_aromaticity;
use crate::perceive::{Hybridization, perceive_conjugated_atoms, perceive_hybridizations};

/// Universal Force Field typifier (Rappé 1992, RDKit-aligned).
///
/// Universal Force Field typifier.
///
/// Assigns RDKit-style UFF atom labels, generates bond/angle/dihedral topology,
/// and resolves per-instance force constants so
/// [`PotentialCompiler::compile`](crate::ff::compile::PotentialCompiler::compile)
/// can compile `uff_bond` / `uff_angle` / `uff_torsion` / `uff_lj` kernels.
///
/// # Labels
///
/// Every bonded term is one type, named by a [`TypeName`] over the UFF atom
/// labels of its atoms, qualified with the values its parameters depend on
/// that are not a function of those labels (Rappé et al., JACS 114, 10024
/// (1992)); each field is an `f64` in Rust `Display` form:
///
/// - bond `{ti}-{tj}@{bo}` — the effective bond order used (`1`, `1.5`, `2`,
///   `3`; an amide C–N is `1`, as RDKit prices it);
/// - angle `{ti}-{tj}-{tk}@{bo_ij}_{bo_jk}_{code}` — `code` is the RDKit
///   coordination code before the effective map (`0`, `1`, `2`, `3`, `4`,
///   `30`, `35`, `40`, `45`), which fixes `theta0` and the Fourier/order form;
/// - torsion `{ti}-{tj}-{tk}-{tl}@{V}_{n}_{nphi0}` — `V` the barrier after
///   division by the torsion count about the central bond, `n` the
///   periodicity, `nphi0` `0` for `cosTerm = +1` and `180` for `cosTerm = -1`;
/// - inversion `{tj}-{ta}-{tb}-{tc}` in improper node order (centre first,
///   the order of LAMMPS's out-of-plane styles; RDKit lists it second),
///   unqualified: `K, c0, c1, c2` depend only on the centre element and on
///   whether an endpoint is `O_2` / `O_R`.
///
/// A bond, angle or torsion is oriented before it is named: of its forward and
/// reversed endpoint labels (then qualifier fields, an angle's two bond orders
/// swapping with its ends) it takes the smaller, slot by slot, so both
/// orientations of one term share one name and one type. Bond and angle params
/// are evaluated on that orientation: their force constants multiply
/// the two end atoms' `Z*` into a running product, and floating-point products
/// do not reassociate, so evaluating each term in its own node order could give
/// one name two params differing in the last bit. (A torsion's params read only
/// the central pair, symmetrically.) Inversions are **not** oriented:
/// reversing a 4-atom term moves the centre to position 4 and names a different
/// term.
///
/// # Route
///
/// ```ignore
/// let mut typing = Typing::new(UffTypifier::new());
/// let frame = typing.typify(&mol)?.to_frame()?; // terms, special pairs, types
/// let pots = PotentialCompiler::new(typing.forcefield()).compile(&frame)?;
/// ```
///
/// Organic / main-group subset only (see [`crate::ff::params::uff`]). No GFN-FF.
pub struct UffTypifier {
    ff: ForceField,
}

impl Default for UffTypifier {
    fn default() -> Self {
        Self::new()
    }
}

impl UffTypifier {
    /// Infallible: parameters are compile-time constants.
    ///
    /// An input-free constructor over the `ff/params` UFF table: the one
    /// definition result it can meet is its own five literal styles, and
    /// `tests::new_defines_without_conflict` proves it `Ok`.
    pub fn new() -> Self {
        Self::try_new().expect(
            "UFF styles define without conflict — proved by \
             ff::typifier::uff::tests::new_defines_without_conflict",
        )
    }

    /// The fallible body of [`new`](Self::new).
    fn try_new() -> Result<Self, DefError> {
        let mut ff = ForceField::new("UFF");
        // The UFF parameters are kcal/mol and Å.
        ff.set_units("real");
        for (category, name) in [
            ("bond", "uff_bond"),
            ("angle", "uff_angle"),
            ("dihedral", "uff_torsion"),
            ("improper", "uff_inversion"),
            ("pair", "uff_lj"),
        ] {
            ff.def_style(category, name, Params::new())?;
        }
        // UFF has no electrostatics; 1-4 LJ at full strength (RDKit).
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.0, 1.0],
            coul: [0.0, 0.0, 0.0],
        });
        Ok(Self { ff })
    }
}

/// `type` → a type named `name` on `endpoints` under `style`, with `params`.
fn typed(
    style: &str,
    name: &TypeName,
    endpoints: &[&str],
    params: Params,
) -> Vec<(String, Annotation)> {
    vec![(
        "type".to_owned(),
        Annotation::Type {
            style: style.to_owned(),
            name: name.to_string(),
            endpoints: endpoints.iter().map(|e| (*e).to_owned()).collect(),
            params,
        },
    )]
}

/// Whether a term reads in reverse: [`TypeName::reads_reversed`] on its
/// `labels`, and — when they are a palindrome — `reversed_fields` (the
/// qualifier fields as they read from the other end) comparing smaller, slot
/// by slot, than `fields`. A tie keeps the forward orientation.
fn reads_reversed(labels: &[&str], fields: &[f64], reversed_fields: &[f64]) -> bool {
    if labels.iter().ne(labels.iter().rev()) {
        return TypeName::reads_reversed(labels);
    }
    let text = |f: &[f64]| f.iter().map(f64::to_string).collect::<Vec<_>>();
    text(reversed_fields) < text(fields)
}

/// `name` qualified with `fields`, each in `f64` `Display` form.
fn qualified(labels: &[&str], fields: &[f64]) -> Result<TypeName, String> {
    let fields: Vec<String> = fields.iter().map(f64::to_string).collect();
    let fields: Vec<&str> = fields.iter().map(String::as_str).collect();
    TypeName::join(labels)?.with_qualifier(&fields)
}

impl Typifier for UffTypifier {
    /// Label atoms and resolve per-instance UFF parameters.
    ///
    /// Angles, dihedrals (regenerated) and inversions (added) are enumerated
    /// onto `graph`. Atoms get `type` (the UFF label), `x1` and `D1` as plain
    /// values; every bond, angle, dihedral and generated improper gets `type`
    /// — its label (see the [type docs](UffTypifier#labels)) — as a type defined
    /// under `uff_bond` / `uff_angle` / `uff_torsion` / `uff_inversion` with
    /// the params the kernels read. The `uff_lj` rows `{x1, D1}` of the labels
    /// used are pairs; every library style is declared.
    fn assign(&self, graph: &mut Atomistic) -> Result<TypeAssignment, TypingError> {
        graph
            .generate_topology(true, true, false, true, true)
            .map_err(TypingError::malformed)?;

        let atom_ids: Vec<NodeId> = graph.atom_ids().collect();
        let n = atom_ids.len();
        let rows = graph.atom_table();
        const LIVE: &str = "a relation endpoint is a live atom";
        let row = |id: NodeId| rows.row(id).expect(LIVE);

        // Neighbours + bond orders
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        let mut bond_order: HashMap<(usize, usize), f64> = HashMap::new();
        for (i, &aid) in atom_ids.iter().enumerate() {
            for (nid, bid) in graph.neighbors(aid, NeighborKinds::BONDS) {
                if let Some(j) = rows.row(nid) {
                    adj[i].push(j);
                    // UFF's bond-order term is the localized count; an
                    // aromatic bond is conventionally 1.5 *here*, as a UFF
                    // parameter, not as a molrs bond order. A bond with no
                    // order has no UFF term to price.
                    let ord = if graph
                        .bond_type(bid)
                        .map_err(TypingError::malformed)?
                        .is_aromatic()
                    {
                        1.5
                    } else {
                        match graph
                            .bond_number(bid)
                            .map_err(TypingError::malformed)?
                            .count()
                        {
                            0 => {
                                return Err(TypingError::malformed(format!(
                                    "UFF: bond {i}-{j} states no bond order; UFF prices a single, \
                                     double, triple or aromatic bond"
                                )));
                            }
                            n => n as f64,
                        }
                    };
                    bond_order.insert(if i < j { (i, j) } else { (j, i) }, ord);
                }
            }
            adj[i].sort_unstable();
            adj[i].dedup();
        }
        let order_of = |i: usize, j: usize| {
            *bond_order
                .get(&(i.min(j), i.max(j)))
                .expect("order_of is asked about bonded pairs only")
        };

        // RDKit's UFF labels follow RDKit aromaticity: perceived on a private
        // copy (as OPLS does), so the caller's graph keeps its bond classes.
        let mut perceived = graph.clone();
        assign_aromaticity(&mut perceived).map_err(TypingError::malformed)?;
        let aromatic_atom: Vec<bool> = atom_ids
            .iter()
            .map(|&aid| {
                perceived
                    .get_atom(aid)
                    .map(|atom| atom_keys::is_aromatic(&atom))
                    .map_err(TypingError::malformed)
            })
            .collect::<Result<_, _>>()?;

        // Hybridization + labels
        let mut labels: Vec<String> = Vec::with_capacity(n);
        let mut params: Vec<&'static AtomicParams> = Vec::with_capacity(n);
        // RDKit's hybridization and conjugation (`perceive`): what its
        // `getAtomLabel`, angle coordination codes and torsion rules read.
        let hyb = perceive_hybridizations(graph);
        let conjugated = perceive_conjugated_atoms(graph);
        let mut z: Vec<u8> = Vec::with_capacity(n);
        let mut m = TypeAssignment::default();
        let mut lj_used: HashSet<String> = HashSet::new();

        for (i, &aid) in atom_ids.iter().enumerate() {
            let el = element_of(graph, aid).map_err(TypingError::malformed)?;
            z.push(el.z());
            let valence: f64 = adj[i].iter().map(|&j| order_of(i, j)).sum();
            let label = atom_label(el, hyb[i], aromatic_atom[i] || conjugated[i], valence);
            let p = params_for_label(&label).ok_or_else(|| TypingError::MissingAtomType {
                table: "UFF".to_owned(),
                detail: format!("atom {i}: no parameters for label '{label}'"),
            })?;
            if lj_used.insert(label.clone()) {
                m.pairs.push((
                    "uff_lj".to_owned(),
                    label.clone(),
                    vec![label.clone()],
                    Params::from_iter([("x1", p.x1), ("D1", p.d1)]),
                ));
            }
            m.nodes.push(vec![
                (
                    "type".to_owned(),
                    Annotation::Value(PropValue::Str(label.clone())),
                ),
                ("x1".to_owned(), Annotation::Value(PropValue::F64(p.x1))),
                ("D1".to_owned(), Annotation::Value(PropValue::F64(p.d1))),
            ]);
            labels.push(label);
            params.push(p);
        }

        // Bonds — the effective bond order, then the params on the oriented
        // term.
        let bonds: Vec<(usize, usize)> = graph
            .bonds()
            .map(|(_, b)| (row(b.nodes[0]), row(b.nodes[1])))
            .collect();
        for (i, j) in bonds {
            let mut bo = order_of(i, j);
            if aromatic_atom[i] && aromatic_atom[j] && (bo - 1.5).abs() < 0.2 {
                bo = 1.5;
            }
            let (p, q) = if reads_reversed(&[&labels[i], &labels[j]], &[bo], &[bo]) {
                (j, i)
            } else {
                (i, j)
            };
            let ends = [labels[p].as_str(), labels[q].as_str()];
            let name = qualified(&ends, &[bo]).map_err(TypingError::malformed)?;
            let (r0, kb) = bond_rest_and_k(params[p], params[q], bo);
            m.link_mut(BONDS).push(typed(
                "uff_bond",
                &name,
                &ends,
                Params::from_iter([("kb", kb), ("r0", r0)]),
            ));
        }

        // Angles
        let rings = crate::perceive::perceive_rings(graph);
        let in_ring3 = |aid: NodeId| rings.rings_of_size(3).iter().any(|r| r.contains(&aid));
        let in_ring4 = |aid: NodeId| rings.rings_of_size(4).iter().any(|r| r.contains(&aid));
        let angles: Vec<[usize; 3]> = graph
            .angles()
            .map(|(_, a)| std::array::from_fn(|n| row(a.nodes[n])))
            .collect();
        for [i, j, k] in angles {
            let mut order = match hyb[j] {
                Hybridization::Sp => 1u8,
                Hybridization::Sp2 => 3,
                Hybridization::Sp3d2 => 4,
                _ => 0,
            };
            // Ring hacks for sp2 (RDKit Builder) — simplified
            if hyb[j] == Hybridization::Sp2 {
                if in_ring3(atom_ids[j]) {
                    order = if in_ring3(atom_ids[i]) && in_ring3(atom_ids[k]) {
                        35
                    } else {
                        30
                    };
                } else if in_ring4(atom_ids[j]) {
                    order = if in_ring4(atom_ids[i]) && in_ring4(atom_ids[k]) {
                        45
                    } else {
                        40
                    };
                }
            }
            let code = f64::from(order);
            let (p, q) = if reads_reversed(
                &[&labels[i], &labels[j], &labels[k]],
                &[order_of(i, j), order_of(j, k), code],
                &[order_of(j, k), order_of(i, j), code],
            ) {
                (k, i)
            } else {
                (i, k)
            };
            let ends = [labels[p].as_str(), labels[j].as_str(), labels[q].as_str()];
            let name = qualified(&ends, &[order_of(p, j), order_of(j, q), code])
                .map_err(TypingError::malformed)?;
            let (theta0, order_eff) = match order {
                30 => (150.0_f64.to_radians(), 0u8),
                35 => (60.0_f64.to_radians(), 0),
                40 => (135.0_f64.to_radians(), 0),
                45 => (90.0_f64.to_radians(), 0),
                o => (params[j].theta0_rad(), o),
            };
            let ka = angle_force_constant(
                theta0,
                order_of(p, j),
                order_of(j, q),
                params[p],
                params[j],
                params[q],
            );
            let (c0, c1, c2) = if order_eff == 0 {
                fourier_coeffs(theta0)
            } else {
                (0.0, 0.0, 0.0)
            };
            m.link_mut(ANGLES).push(typed(
                "uff_angle",
                &name,
                &ends,
                Params::from_iter([
                    ("ka", ka),
                    ("order", f64::from(order_eff)),
                    ("c0", c0),
                    ("c1", c1),
                    ("c2", c2),
                    ("theta0", theta0.to_degrees()),
                ]),
            ));
        }

        // Dihedrals — scale V by multiplicity about the central bond
        let dihedrals: Vec<[usize; 4]> = graph
            .dihedrals()
            .map(|(_, d)| std::array::from_fn(|n| row(d.nodes[n])))
            .collect();
        let mut bond_counts: HashMap<(usize, usize), usize> = HashMap::new();
        for idx in &dihedrals {
            let key = (idx[1].min(idx[2]), idx[1].max(idx[2]));
            *bond_counts.entry(key).or_insert(0) += 1;
        }
        for [i, j, k, l] in dihedrals {
            // Only SP2/SP3 central atoms (RDKit); any other centre gets a zero
            // barrier — energy 0.
            let (v, order, cos_term) = if !matches!(hyb[j], Hybridization::Sp2 | Hybridization::Sp3)
                || !matches!(hyb[k], Hybridization::Sp2 | Hybridization::Sp3)
            {
                (0.0, 3u8, -1.0)
            } else {
                let end_sp2 = hyb[i] == Hybridization::Sp2 || hyb[l] == Hybridization::Sp2;
                let (v, order, cos_term) = torsion_params(
                    order_of(j, k),
                    z[j],
                    z[k],
                    hyb[j],
                    hyb[k],
                    params[j],
                    params[k],
                    end_sp2,
                );
                let mult = bond_counts
                    .get(&(j.min(k), j.max(k)))
                    .copied()
                    .unwrap_or(1)
                    .max(1) as f64;
                (v / mult, order, cos_term)
            };
            let nphi0 = if cos_term > 0.0 { 0.0 } else { 180.0 };
            let fields = [v, f64::from(order), nphi0];
            let forward = [
                labels[i].as_str(),
                labels[j].as_str(),
                labels[k].as_str(),
                labels[l].as_str(),
            ];
            let ends = if reads_reversed(&forward, &fields, &fields) {
                [forward[3], forward[2], forward[1], forward[0]]
            } else {
                forward
            };
            let name = qualified(&ends, &fields).map_err(TypingError::malformed)?;
            m.link_mut(DIHEDRALS).push(typed(
                "uff_torsion",
                &name,
                &ends,
                Params::from_iter([("V", v), ("order", f64::from(order)), ("cosTerm", cos_term)]),
            ));
        }

        // Inversions (RDKit Tools::addInversions) — three Wilson rows per centre.
        let mut inversions = HashMap::new();
        for j in 0..n {
            if adj[j].len() != 3 {
                continue;
            }
            let zc = z[j];
            let eligible = matches!(zc, 6 | 7 | 8 | 15 | 33 | 51 | 83);
            if !eligible {
                continue;
            }
            if matches!(zc, 6..=8) && hyb[j] != Hybridization::Sp2 {
                continue;
            }
            let nbrs = [adj[j][0], adj[j][1], adj[j][2]];
            let is_c_bound_to_sp2_o = zc == 6
                && nbrs
                    .iter()
                    .any(|&o| z[o] == 8 && hyb[o] == Hybridization::Sp2);
            let (k_inv, c0, c1, c2) = inversion_coeffs(zc, is_c_bound_to_sp2_o);
            // three permutations: centre j first, outer atoms in RDKit order
            let perms = [
                (nbrs[0], nbrs[1], nbrs[2]),
                (nbrs[0], nbrs[2], nbrs[1]),
                (nbrs[1], nbrs[2], nbrs[0]),
            ];
            for (a, b, c) in perms {
                let iid = graph
                    .add_improper(
                        atom_ids[j],
                        atom_ids[a],
                        atom_ids[b],
                        atom_ids[c],
                        Props::new(),
                    )
                    .map_err(TypingError::malformed)?;
                let ends = [&*labels[j], &*labels[a], &*labels[b], &*labels[c]];
                let name = TypeName::join(&ends).map_err(TypingError::malformed)?;
                inversions.insert(
                    iid,
                    typed(
                        "uff_inversion",
                        &name,
                        &ends,
                        Params::from_iter([("K", k_inv), ("c0", c0), ("c1", c1), ("c2", c2)]),
                    ),
                );
            }
        }
        // Positional against every improper of the graph; one the input
        // already carried gets nothing.
        *m.link_mut(IMPROPERS) = graph
            .impropers()
            .map(|(id, _)| inversions.remove(&id).unwrap_or_default())
            .collect();

        m.declare_styles_of(&self.ff);
        Ok(m)
    }

    /// The UFF style skeleton: five styles, no rows (every UFF parameter is
    /// resolved per instance), and UFF's special_bonds.
    fn source_forcefield(&self) -> &ForceField {
        &self.ff
    }
}

/// The element of atom `id`, read from its [`keys::ATOMIC_NUMBER`]. UFF types
/// real atoms only: a wildcard (`Z = 0`) has no UFF type and is refused.
fn element_of(mol: &Atomistic, id: NodeId) -> Result<Element, String> {
    let z = mol
        .get_atom(id)
        .map_err(|e| e.to_string())?
        .atomic_number()
        .ok_or_else(|| format!("UFF: atom {id:?} has no valid `{}`", keys::ATOMIC_NUMBER))?;
    Element::by_number(z).ok_or_else(|| {
        format!(
            "UFF: atom {id:?} is a wildcard (`{}` = 0) and has no UFF type",
            keys::ATOMIC_NUMBER
        )
    })
}

/// The UFF atom label of an atom: RDKit `Tools::getAtomLabel` +
/// `addAtomChargeFlags` (the full default UFF set).
///
/// `hyb` is RDKit's hybridization ([`perceive_hybridizations`]); `resonant` is whether
/// the atom is aromatic or carries a conjugated bond, which turns an sp²
/// C / N / O / S into its `_R` type. Shared with the ETKDG bounds builder,
/// whose 1-2 bounds are UFF rest lengths.
pub(crate) fn atom_label(
    element: Element,
    hyb: Hybridization,
    resonant: bool,
    valence: f64,
) -> String {
    let z = element.z();
    let mut key = element.symbol().to_string();
    if key.len() == 1 {
        key.push('_');
    }

    // No hybridization on alkali metals (group 1) or halogens (group 7).
    let n_outer = match z {
        1 | 3 | 11 | 19 | 37 | 55 | 87 => 1u8, // alkali
        9 | 17 | 35 | 53 | 85 => 7u8,          // halogen
        _ => 0u8,
    };
    let skip_hyb = n_outer == 1 || n_outer == 7;

    if !skip_hyb {
        // Main-group force SP3 suffix (RDKit cases 12–15, 50–52, 81–84).
        if matches!(z, 12 | 13 | 14 | 15 | 50 | 51 | 52 | 81 | 82 | 83 | 84) {
            key.push('3');
        } else if z == 80 {
            // Hg → Hg1
            key.push('1');
        } else {
            match hyb {
                Hybridization::Sp => key.push('1'),
                Hybridization::Sp2 => {
                    if resonant && matches!(z, 6 | 7 | 8 | 16) {
                        key.push('R');
                    } else {
                        key.push('2');
                    }
                }
                Hybridization::Sp3 => key.push('3'),
                Hybridization::Sp3d => key.push('5'),
                Hybridization::Sp3d2 => key.push('6'),
                Hybridization::S | Hybridization::Other => {}
            }
        }
    }

    add_charge_flags(&mut key, z, valence, hyb);
    key
}

/// RDKit `addAtomChargeFlags` (tolerateChargeMismatch = true for robustness).
fn add_charge_flags(key: &mut String, z: u8, valence: f64, hyb: Hybridization) {
    let v = valence.round() as i32;
    let push = |k: &mut String, s: &str| k.push_str(s);

    match z {
        // only +1
        29 | 47 => push(key, "+1"),
        // only +2
        4 | 20 | 25 | 26 | 28 | 46 | 78 => push(key, "+2"),
        // only +3
        21 | 24 | 27 | 79 | 89 | 96..=103 => push(key, "+3"),
        // only +4
        2 | 18 | 22 | 36 | 54 | 90..=95 => push(key, "+4"),
        // only +5
        23 | 41 | 43 | 73 => push(key, "+5"),
        // only +6
        42 => push(key, "+6"),
        12 => push(key, "+2"), // Mg
        15 => {
            // P
            if v <= 3 {
                push(key, "+3");
            } else {
                push(key, "+5");
            }
        }
        16 => {
            // S — skip charge flag for SP2 (S_2 / S_R)
            if hyb != Hybridization::Sp2 {
                match v {
                    2 => push(key, "+2"),
                    4 => push(key, "+4"),
                    _ => push(key, "+6"),
                }
            }
        }
        30 => push(key, "+2"), // Zn
        31 => push(key, "+3"), // Ga
        33 => push(key, "+3"), // As
        34 => push(key, "+2"), // Se
        48 => push(key, "+2"), // Cd
        49 => push(key, "+3"), // In
        51 => push(key, "+3"), // Sb
        52 => push(key, "+2"), // Te
        75 => {
            // Re special
            if key.starts_with("Re6") {
                *key = "Re6+5".into();
            } else if key.starts_with("Re3") {
                *key = "Re3+7".into();
            }
        }
        80 => push(key, "+2"),      // Hg
        81 => push(key, "+3"),      // Tl
        82 => push(key, "+3"),      // Pb — RDKit uses +3 default
        83 => push(key, "+3"),      // Bi
        84 => push(key, "+2"),      // Po
        57..=71 => push(key, "+3"), // lanthanides
        _ => {}
    }
}

fn inversion_coeffs(at2_z: u8, is_c_bound_to_o: bool) -> (f64, f64, f64, f64) {
    // RDKit Utils::calcInversionCoefficientsAndForceConstant
    if matches!(at2_z, 6..=8) {
        let res = if is_c_bound_to_o { 50.0 } else { 6.0 } / 3.0;
        return (res, 1.0, -1.0, 0.0);
    }
    let w0_deg: f64 = match at2_z {
        15 => 84.4339,
        33 => 86.9735,
        51 => 87.7047,
        83 => 90.0,
        _ => 90.0,
    };
    let w0 = w0_deg.to_radians();
    let c2: f64 = 1.0;
    let c1 = -4.0 * w0.cos();
    let c0 = -(c1 * w0.cos() + c2 * (2.0 * w0).cos());
    let res = 22.0 / (c0 + c1 + c2) / 3.0;
    (res, c0, c1, c2)
}

/// UFF's bond rest length between two labelled atoms (RDKit
/// `calcBondRestLength`), or `None` when a label has no UFF row.
///
/// The ETKDG bounds builder sets its 1-2 bounds from this, as RDKit's does.
#[cfg(any(feature = "conformer", test))]
pub(crate) fn bond_rest_length(label_i: &str, label_j: &str, bond_order: f64) -> Option<f64> {
    let (pi, pj) = (params_for_label(label_i)?, params_for_label(label_j)?);
    Some(bond_rest_and_k(pi, pj, bond_order).0)
}

fn bond_rest_and_k(p1: &AtomicParams, p2: &AtomicParams, bond_order: f64) -> (f64, f64) {
    let bo = bond_order.max(1e-6);
    let (ri, rj) = (p1.r1, p2.r1);
    let r_bo = -LAMBDA * (ri + rj) * bo.ln();
    let (xi, xj) = (p1.xi, p2.xi);
    let dx = xi.sqrt() - xj.sqrt();
    let r_en = ri * rj * dx * dx / (xi * ri + xj * rj);
    let r0 = ri + rj + r_bo - r_en;
    let kb = 2.0 * UFF_COULOMB * p1.z1 * p2.z1 / (r0 * r0 * r0);
    (r0, kb)
}

fn angle_force_constant(
    theta0: f64,
    bo12: f64,
    bo23: f64,
    p1: &AtomicParams,
    p2: &AtomicParams,
    p3: &AtomicParams,
) -> f64 {
    let cos0 = theta0.cos();
    let r12 = bond_rest_and_k(p1, p2, bo12).0;
    let r23 = bond_rest_and_k(p2, p3, bo23).0;
    let r13 = (r12 * r12 + r23 * r23 - 2.0 * r12 * r23 * cos0).sqrt();
    let beta = 2.0 * UFF_COULOMB / (r12 * r23);
    let pref = beta * p1.z1 * p3.z1 / r13.powi(5);
    let r_term = r12 * r23;
    let inner = 3.0 * r_term * (1.0 - cos0 * cos0) - r13 * r13 * cos0;
    pref * r_term * inner
}

fn fourier_coeffs(theta0: f64) -> (f64, f64, f64) {
    let sin0 = theta0.sin();
    let cos0 = theta0.cos();
    let c2 = 1.0 / (4.0 * (sin0 * sin0).max(1e-8));
    let c1 = -4.0 * c2 * cos0;
    let c0 = c2 * (2.0 * cos0 * cos0 + 1.0);
    (c0, c1, c2)
}

fn is_group6(z: u8) -> bool {
    matches!(z, 8 | 16 | 34 | 52 | 84)
}

#[allow(clippy::too_many_arguments)]
fn torsion_params(
    bo23: f64,
    z2: u8,
    z3: u8,
    h2: Hybridization,
    h3: Hybridization,
    p2: &AtomicParams,
    p3: &AtomicParams,
    end_sp2: bool,
) -> (f64, u8, f64) {
    if h2 == Hybridization::Sp3 && h3 == Hybridization::Sp3 {
        let mut v = (p2.v1 * p3.v1).sqrt();
        let mut order = 3u8;
        let mut cos_term = -1.0;
        if (bo23 - 1.0).abs() < 1e-6 && is_group6(z2) && is_group6(z3) {
            let v2: f64 = if z2 == 8 { 2.0 } else { 6.8 };
            let v3: f64 = if z3 == 8 { 2.0 } else { 6.8 };
            v = (v2 * v3).sqrt();
            order = 2;
            cos_term = -1.0;
        }
        return (v, order, cos_term);
    }
    if h2 == Hybridization::Sp2 && h3 == Hybridization::Sp2 {
        let v = 5.0 * (p2.u1 * p3.u1).sqrt() * (1.0 + 4.18 * bo23.ln());
        return (v, 2, 1.0);
    }
    // SP2-SP3
    let mut v = 1.0;
    let mut order = 6u8;
    let mut cos_term = 1.0;
    if (bo23 - 1.0).abs() < 1e-6 {
        if (h2 == Hybridization::Sp3 && is_group6(z2) && !is_group6(z3))
            || (h3 == Hybridization::Sp3 && is_group6(z3) && !is_group6(z2))
        {
            v = 5.0 * (p2.u1 * p3.u1).sqrt() * (1.0 + 4.18 * bo23.ln());
            order = 2;
            cos_term = -1.0;
        } else if end_sp2 {
            v = 2.0;
            order = 3;
            cos_term = -1.0;
        }
    }
    (v, order, cos_term)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::props;
    use molrs::core::Element;
    use molrs::core::TypeName;
    use molrs::core::keys;
    use molrs::core::{Atomistic, BondNumber, BondOrder};
    use std::collections::{BTreeMap, BTreeSet};

    fn ethanol() -> Atomistic {
        // Build C-C-O + hydrogens with rough coords
        let mut m = Atomistic::new();
        let c0 = m
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 0.9,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let c1 = m
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => -0.5,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let o = m
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::O,
                keys::X => -1.2,
                keys::Y => 0.9,
                keys::Z => 0.0,
            })
            .unwrap();
        let _ = m.add_bond(c0, c1, props! {});
        let _ = m.add_bond(c1, o, props! {});
        // add H
        let h1 = m
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => 1.3,
                keys::Y => 0.9,
                keys::Z => 0.0,
            })
            .unwrap();
        let h2 = m
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => 1.3,
                keys::Y => -0.5,
                keys::Z => 0.9,
            })
            .unwrap();
        let h3 = m
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => 1.3,
                keys::Y => -0.5,
                keys::Z => -0.9,
            })
            .unwrap();
        let h4 = m
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => -0.9,
                keys::Y => -0.9,
                keys::Z => 0.5,
            })
            .unwrap();
        let h5 = m
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => -0.9,
                keys::Y => -0.5,
                keys::Z => -0.9,
            })
            .unwrap();
        let h6 = m
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => -1.1,
                keys::Y => 1.5,
                keys::Z => -0.5,
            })
            .unwrap();
        for h in [h1, h2, h3] {
            let _ = m.add_bond(c0, h, props! {});
        }
        for h in [h4, h5] {
            let _ = m.add_bond(c1, h, props! {});
        }
        let _ = m.add_bond(o, h6, props! {});
        m
    }

    // -- Typing<UffTypifier> output (system-forcefield-07) --------------------

    /// N-methylacetamide `CH3-C(=O)-NH-CH3`, hand-built: methyl C is atom 0,
    /// carbonyl C atom 1, O atom 2 (C=O double), amide N atom 3, N-methyl C
    /// atom 4; hydrogens follow. Returns the graph and the carbonyl C and N.
    fn n_methylacetamide() -> (Atomistic, NodeId, NodeId) {
        let mut m = Atomistic::new();
        let c_me = m
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let c_co = m
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let o = m
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::O })
            .unwrap();
        let n = m
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::N })
            .unwrap();
        let c_nme = m
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        m.add_bond(c_me, c_co, props! {}).unwrap();
        let co = m.add_bond(c_co, o, props! {}).unwrap();
        m.set_bond_class(co, BondOrder::Double, BondNumber::Double)
            .unwrap();
        m.add_bond(c_co, n, props! {}).unwrap();
        m.add_bond(n, c_nme, props! {}).unwrap();
        for (heavy, n_h) in [(c_me, 3), (n, 1), (c_nme, 3)] {
            for _ in 0..n_h {
                let h = m
                    .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
                    .unwrap();
                m.add_bond(heavy, h, props! {}).unwrap();
            }
        }
        (m, c_co, n)
    }

    /// Triphenylphosphine `P(C6H5)3`, hand-built with aromatic ring bonds.
    /// The P is the **last** atom, so in an ipso carbon's sorted neighbour
    /// list it comes after both ortho carbons: the ipso-centred inversion
    /// `C_R-C_R-P_3+3-C_R` is then generated, which is exactly the reversal of
    /// the P-centred `C_R-P_3+3-C_R-C_R` — the two collide only if impropers
    /// are wrongly oriented.
    fn triphenylphosphine() -> Atomistic {
        let mut m = Atomistic::new();
        let mut ipso = Vec::new();
        for _ in 0..3 {
            let ring: Vec<NodeId> = (0..6)
                .map(|_| {
                    m.add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
                        .unwrap()
                })
                .collect();
            for k in 0..6 {
                let b = m.add_bond(ring[k], ring[(k + 1) % 6], props! {}).unwrap();
                m.set_bond_class(b, BondOrder::Aromatic, BondNumber::Unknown)
                    .unwrap();
            }
            for &c in &ring[1..] {
                let h = m
                    .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
                    .unwrap();
                m.add_bond(c, h, props! {}).unwrap();
            }
            ipso.push(ring[0]);
        }
        let p = m
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::P })
            .unwrap();
        for c in ipso {
            m.add_bond(p, c, props! {}).unwrap();
        }
        m
    }

    fn uff_typed(mol: &Atomistic) -> (Atomistic, crate::ff::typifier::Typing<UffTypifier>) {
        let mut typing = crate::ff::typifier::Typing::new(UffTypifier::new());
        let typed = typing.typify(mol).expect("UFF types the molecule");
        (typed, typing)
    }

    fn str_prop(props: &Props, key: &str) -> Option<String> {
        props.get_str(key).map(str::to_owned)
    }

    /// The UFF atom label (`type`) of every atom, by id.
    fn atom_labels(typed: &Atomistic) -> HashMap<NodeId, String> {
        typed
            .atoms()
            .map(|(id, a)| (id, a.get_str("type").expect("every atom typed").to_owned()))
            .collect()
    }

    /// `(nodes, type label)` of every link of `kind` (`bonds`, `angles`,
    /// `dihedrals`, `impropers`); a missing `type` is a failure.
    fn link_labels(typed: &Atomistic, kind: &str) -> Vec<(Vec<NodeId>, String)> {
        let rows: Vec<(Vec<NodeId>, Props)> = match kind {
            "bonds" => typed
                .bonds()
                .map(|(_, r)| (r.nodes.to_vec(), r.props))
                .collect(),
            "angles" => typed
                .angles()
                .map(|(_, r)| (r.nodes.to_vec(), r.props))
                .collect(),
            "dihedrals" => typed
                .dihedrals()
                .map(|(_, r)| (r.nodes.to_vec(), r.props))
                .collect(),
            "impropers" => typed
                .impropers()
                .map(|(_, r)| (r.nodes.to_vec(), r.props))
                .collect(),
            other => panic!("no link kind {other}"),
        };
        rows.into_iter()
            .map(|(nodes, props)| {
                let label = str_prop(&props, "type")
                    .unwrap_or_else(|| panic!("{kind} {nodes:?} has no type"));
                (nodes, label)
            })
            .collect()
    }

    /// Per `(category, style)` of `ff`, the set of type names; styles holding
    /// no type are left out.
    fn output_names(ff: &ForceField) -> BTreeMap<(String, String), BTreeSet<String>> {
        ff.styles()
            .iter()
            .filter_map(|s| {
                let names: BTreeSet<String> = s
                    .defs()
                    .collect_type_params()
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect();
                (!names.is_empty()).then(|| ((s.category().to_owned(), s.name().to_owned()), names))
            })
            .collect()
    }

    /// Output name sets per `(category, style)` equal the stamped label sets:
    /// bonded `type`s under `uff_bond` / `uff_angle` / `uff_torsion` /
    /// `uff_inversion`, atom `type`s as the `uff_lj` pair rows.
    fn assert_output_names_equal_stamped_labels(mol: &Atomistic) {
        let (typed, typing) = uff_typed(mol);
        let set = |kind: &str| -> BTreeSet<String> {
            link_labels(&typed, kind)
                .into_iter()
                .map(|(_, label)| label)
                .collect()
        };
        let key = |c: &str, s: &str| (c.to_owned(), s.to_owned());
        let mut expected = BTreeMap::from([
            (
                key("pair", "uff_lj"),
                atom_labels(&typed).into_values().collect(),
            ),
            (key("bond", "uff_bond"), set("bonds")),
            (key("angle", "uff_angle"), set("angles")),
            (key("dihedral", "uff_torsion"), set("dihedrals")),
            (key("improper", "uff_inversion"), set("impropers")),
        ]);
        expected.retain(|_, names| !names.is_empty());
        assert_eq!(output_names(typing.forcefield()), expected);
    }

    #[test]
    fn typing_ethanol_output_names_equal_the_stamped_labels() {
        assert_output_names_equal_stamped_labels(&ethanol());
    }

    #[test]
    fn typing_n_methylacetamide_output_names_equal_the_stamped_labels() {
        assert_output_names_equal_stamped_labels(&n_methylacetamide().0);
    }

    /// Every bond, angle and dihedral is typed on the smaller (slot by slot)
    /// of its atom labels read forward and reversed, and its label is the
    /// `TypeName` join of those endpoints plus a qualifier; no two output rows
    /// of those styles hold endpoint tuples that are reversals of each other.
    fn assert_proper_types_are_oriented(mol: &Atomistic) {
        let (typed, typing) = uff_typed(mol);
        let atoms = atom_labels(&typed);
        for (kind, category, style) in [
            ("bonds", "bond", "uff_bond"),
            ("angles", "angle", "uff_angle"),
            ("dihedrals", "dihedral", "uff_torsion"),
        ] {
            let defs = typing
                .forcefield()
                .get_style(category, style)
                .expect("the UFF style is declared");
            for (nodes, label) in link_labels(&typed, kind) {
                let forward: Vec<&str> = nodes.iter().map(|id| atoms[id].as_str()).collect();
                let reversed: Vec<&str> = forward.iter().rev().copied().collect();
                let ends = defs
                    .type_endpoints(&label)
                    .unwrap_or_else(|| panic!("{kind} {label} is defined"));
                let ends: Vec<&str> = ends.iter().map(String::as_str).collect();
                assert_eq!(ends, forward.clone().min(reversed), "{kind} {label}");
                let joined = TypeName::join(&ends).expect("UFF labels hold no '@'");
                assert!(
                    label.starts_with(&format!("{joined}@")),
                    "{kind} {label}: named from its endpoints {ends:?}"
                );
            }
            let rows: BTreeSet<Vec<String>> = defs
                .defs()
                .collect_type_params()
                .into_iter()
                .map(|(n, _)| defs.type_endpoints(&n).expect("a stored type"))
                .collect();
            for e in &rows {
                let rev: Vec<String> = e.iter().rev().cloned().collect();
                assert!(
                    rev == *e || !rows.contains(&rev),
                    "{style} holds both {e:?} and its reversal"
                );
            }
        }
    }

    #[test]
    fn typing_ethanol_proper_types_are_oriented_over_their_atom_labels() {
        assert_proper_types_are_oriented(&ethanol());
    }

    #[test]
    fn typing_n_methylacetamide_proper_types_are_oriented_over_their_atom_labels() {
        assert_proper_types_are_oriented(&n_methylacetamide().0);
    }

    /// Every improper label is the `TypeName` join of its atom labels in node
    /// order, unqualified and not oriented, and its first node is the
    /// centre (bonded to the other three).
    fn assert_improper_labels_keep_node_order(mol: &Atomistic) -> usize {
        let (typed, _) = uff_typed(mol);
        let atoms = atom_labels(&typed);
        let bonded = |a: NodeId, b: NodeId| {
            typed
                .bonds()
                .any(|(_, r)| r.nodes.contains(&a) && r.nodes.contains(&b))
        };
        let impropers = link_labels(&typed, "impropers");
        for (nodes, label) in &impropers {
            let parts: Vec<&str> = nodes.iter().map(|id| atoms[id].as_str()).collect();
            let expected = TypeName::join(&parts).expect("UFF labels hold no '@'");
            assert_eq!(label, expected.as_str(), "improper {nodes:?}");
            assert!(!label.contains('@'), "{label} is unqualified");
            for &other in [nodes[1], nodes[2], nodes[3]].iter() {
                assert!(bonded(nodes[0], other), "improper {label}: centre is first");
            }
        }
        impropers.len()
    }

    #[test]
    fn typing_n_methylacetamide_improper_labels_keep_node_order_centre_first() {
        let n = assert_improper_labels_keep_node_order(&n_methylacetamide().0);
        assert!(n > 0, "the carbonyl C is an inversion centre");
    }

    /// Triphenylphosphine types without a `TypeConflict`: the P-centred and
    /// ipso-C-centred inversions keep distinct names (group-15 vs sp2-C
    /// params), and their labels keep node order with the centre first.
    #[test]
    fn typing_triphenylphosphine_has_no_type_conflict() {
        let mut typing = crate::ff::typifier::Typing::new(UffTypifier::new());
        let result = typing.typify(&triphenylphosphine());
        assert!(result.is_ok(), "{:?}", result.err());
        let n = assert_improper_labels_keep_node_order(&triphenylphosphine());
        assert!(n > 0, "P and the ring carbons are inversion centres");
    }

    /// The amide C-N bond is priced at its graph order, 1, as RDKit prices it:
    /// RDKit 2026.03's `GetUFFBondStretchParams` gives N-methylacetamide's
    /// C(=O)-N the `C_R`-`N_R` rest length at order 1 (1.4222 Å), and its
    /// ETKDG 1-2 bound is that length too. UFF's `amideBondOrder` (1.41)
    /// enters neither.
    #[test]
    fn typing_n_methylacetamide_prices_the_amide_bond_at_order_one() {
        let (mol, c_co, n) = n_methylacetamide();
        let (typed, _) = uff_typed(&mol);
        let (_, label) = link_labels(&typed, "bonds")
            .into_iter()
            .find(|(nodes, _)| nodes.contains(&c_co) && nodes.contains(&n))
            .expect("the amide C-N bond");
        assert_eq!(label, "C_R-N_R@1", "{label}");
    }

    /// Every label has the documented qualifier: bonds `@{bo}`; angles
    /// `@{bo_ij}_{bo_jk}_{code}` with `code` an RDKit coordination code;
    /// torsions `@{V}_{n}_{0|180}`; inversions none.
    fn assert_labels_follow_the_qualifier_form(mol: &Atomistic) {
        let (typed, _) = uff_typed(mol);
        let fields = |label: &str| -> Vec<String> {
            label
                .split_once('@')
                .map(|(_, qualifier)| qualifier)
                .unwrap_or_else(|| panic!("{label} has a qualifier"))
                .split('_')
                .map(str::to_owned)
                .collect()
        };
        let is_f64 = |s: &str| s.parse::<f64>().is_ok();
        for (_, label) in link_labels(&typed, "bonds") {
            let f = fields(&label);
            assert!(f.len() == 1 && is_f64(&f[0]), "bond {label}");
        }
        for (_, label) in link_labels(&typed, "angles") {
            let f = fields(&label);
            assert!(
                f.len() == 3
                    && is_f64(&f[0])
                    && is_f64(&f[1])
                    && ["0", "1", "2", "3", "4", "30", "35", "40", "45"].contains(&f[2].as_str()),
                "angle {label}"
            );
        }
        for (_, label) in link_labels(&typed, "dihedrals") {
            let f = fields(&label);
            assert!(
                f.len() == 3
                    && is_f64(&f[0])
                    && f[1].parse::<u32>().is_ok()
                    && (f[2] == "0" || f[2] == "180"),
                "dihedral {label}"
            );
        }
        for (_, label) in link_labels(&typed, "impropers") {
            assert!(!label.contains('@'), "{label} is unqualified");
        }
    }

    #[test]
    fn typing_ethanol_labels_follow_the_qualifier_form() {
        assert_labels_follow_the_qualifier_form(&ethanol());
    }

    #[test]
    fn typing_n_methylacetamide_labels_follow_the_qualifier_form() {
        assert_labels_follow_the_qualifier_form(&n_methylacetamide().0);
    }

    /// The five UFF styles define without a conflict. `new` `expect`s this
    /// result and names this test.
    #[test]
    fn new_defines_without_conflict() {
        assert_eq!(UffTypifier::try_new().err(), None);
    }

    #[test]
    fn uff_types_ethanol() {
        let typed = crate::ff::typifier::Typing::new(UffTypifier::new())
            .typify(&ethanol())
            .unwrap();
        let mut c3 = 0;
        let mut o3 = 0;
        for (_, a) in typed.atoms() {
            match a.get_str("type") {
                Some("C_3") => c3 += 1,
                Some("O_3") => o3 += 1,
                _ => {}
            }
        }
        assert_eq!(c3, 2);
        assert_eq!(o3, 1);
    }

    /// UFF Eqs. 2–3, one correction term at a time.
    #[test]
    fn bond_rest_length_and_force_constant_follow_the_uff_formulas() {
        let c = params_for_label("C_3").unwrap();
        let o = params_for_label("O_3").unwrap();
        // Homonuclear single bond: the bond-order term is λ(rᵢ+rⱼ)·ln 1 = 0 and
        // the electronegativity term needs χᵢ ≠ χⱼ, so r₀ = 2·r₁ and
        // k = 2G·Z²/r₀³.
        let (r, k) = bond_rest_and_k(c, c, 1.0);
        let r_single = 2.0 * c.r1;
        assert!((r - r_single).abs() < 1e-12);
        assert!((k - 2.0 * UFF_COULOMB * c.z1 * c.z1 / r_single.powi(3)).abs() < 1e-9);
        // A double bond is shorter by λ(rᵢ+rⱼ)·ln 2 and nothing else moves.
        let (r_double, _) = bond_rest_and_k(c, c, 2.0);
        assert!((r_double - (r_single - LAMBDA * r_single * 2f64.ln())).abs() < 1e-12);
        // Heteronuclear: the electronegativity correction pulls r₀ below rᵢ + rⱼ.
        let (r_co, k_co) = bond_rest_and_k(c, o, 1.0);
        let d = c.xi.sqrt() - o.xi.sqrt();
        let r_en = c.r1 * o.r1 * d * d / (c.xi * c.r1 + o.xi * o.r1);
        assert!((r_co - (c.r1 + o.r1 - r_en)).abs() < 1e-12);
        assert!(r_co < c.r1 + o.r1);
        assert!((k_co - 2.0 * UFF_COULOMB * c.z1 * o.z1 / r_co.powi(3)).abs() < 1e-9);
    }

    /// `getAtomLabel`: the hybridization suffix, `R` for a resonant sp² C / N /
    /// O / S, none for hydrogen and the halogens.
    #[test]
    fn labels_follow_element_and_hybridization() {
        use Hybridization::{Sp, Sp2, Sp3};
        let label = |el, h, resonant| atom_label(el, h, resonant, 0.0);
        assert_eq!(label(Element::H, Sp3, false), "H_");
        assert_eq!(label(Element::Cl, Sp3, false), "Cl");
        assert_eq!(label(Element::C, Sp3, false), "C_3");
        assert_eq!(label(Element::C, Sp2, false), "C_2");
        assert_eq!(label(Element::C, Sp2, true), "C_R");
        assert_eq!(label(Element::N, Sp, false), "N_1");
        // Phosphorus is always `3`, with its valence as a charge flag.
        assert_eq!(atom_label(Element::P, Sp2, false, 3.0), "P_3+3");
    }

    /// Sulfur carries its valence as a charge flag, unless it is sp².
    #[test]
    fn sulfur_carries_its_valence_as_a_charge_flag() {
        assert_eq!(
            atom_label(Element::S, Hybridization::Sp3, false, 2.0),
            "S_3+2"
        );
        assert_eq!(
            atom_label(Element::S, Hybridization::Sp3, false, 6.0),
            "S_3+6"
        );
        assert_eq!(
            atom_label(Element::S, Hybridization::Sp2, false, 2.0),
            "S_2"
        );
    }

    /// The rest length is the labelled pair's; an unknown label has none.
    #[test]
    fn bond_rest_length_reads_the_labels_rows() {
        let c = params_for_label("C_3").unwrap();
        assert_eq!(bond_rest_length("C_3", "C_3", 1.0), Some(2.0 * c.r1));
        assert_eq!(bond_rest_length("C_3", "no_such_label", 1.0), None);
    }
}
