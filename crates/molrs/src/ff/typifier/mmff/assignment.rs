//! MMFF atom typing — the [`TypeAssignment`] of an [`Atomistic`]: MMFF type labels,
//! partial charges, and the per-instance force constants the kernels read.
//!
//! This is the **typifier** half of MMFF: it takes a molecular graph and returns
//! its annotations (atoms typed + charged; bonds/angles/dihedrals/impropers
//! labeled, each distinct parameter set one type named by its label). The
//! typing base stamps them and defines the types. Materializing the typed
//! graph into a [`Frame`](molrs::core::Frame) for the generic
//! `PotentialCompiler::compile` path is the caller's job (via
//! [`Atomistic::to_frame`]); building the neighbour list is the consumer's. Atom
//! types + partial charges are reused from the RDKit-validated MMFF front-end
//! ([`MmffMolProperties`]).
//!
//! # One resolver, for the numbers AND the labels
//!
//! Every number and every type code on this graph comes from
//! `resolve` — the RDKit-faithful resolver, with the ring rules,
//! the four-level equivalence degradation and the empirical fallbacks. There used
//! to be a second classifier (`typifier/mmff/classify.rs`) that produced the
//! *labels* while the resolver produced the *parameters*, so a single row could
//! carry a force constant from one rule set and a type code from another. It was
//! also wrong — it read raw bond orders, so an aromatic bond came out as bond
//! type 1 (RDKit: 0, because after aromaticity perception the bond is `AROMATIC`,
//! not `SINGLE`), and its `typify_angle(bt_ij, bt_jk)` could not see ring
//! membership at all, so a cyclopropane C-C-C angle could never reach its true
//! type 3. It is deleted.
//!
//! The labels are only provenance and conflict records — the per-instance
//! kernels ([`ParamSource::PerInstance`](crate::ff::ir::ParamSource)) read
//! Frame *columns*, not type rows — but "only provenance" is not a licence to be
//! wrong: one label must name one parameter set, or the output force field
//! refuses the second definition.
//!
//! # The variant is a parameter, never a constant
//!
//! Every parameter resolved here is resolved **for the caller's [`MmffVariant`]**.
//! This is the path that bakes `koop` (impropers) and `(v1, v2, v3)` (dihedrals)
//! into the Frame columns that `mmff_oop` / `mmff_torsion` consume, so a hardcoded
//! variant here silently produces MMFF94 numbers no matter which typifier the user
//! constructed — the exact bug `Mmff94sTypifier` exists to make impossible.

use std::collections::{HashMap, HashSet};

use molrs::core::PropValue;
use molrs::core::schema::block_names::{ANGLES, BONDS, DIHEDRALS, IMPROPERS};
use molrs::core::{Atomistic, NodeId, Props};

use super::properties::{MmffMolProperties, MmffVariant};
use super::resolve as eparams;
use crate::ff::forcefield::ForceField;
use crate::ff::ir::Params;
use crate::ff::typifier::{Annotation, TypeAssignment};
use crate::perceive::mmff_aromaticity::MmffTopology;

use super::atom_properties::MmffAtomProperties;

/// Positional annotations of one kind of graph element.
type Annotations = Vec<Vec<(String, Annotation)>>;

/// Everything the annotation steps below share: the molecule's atom order,
/// the MMFF topology (with aromaticity perceived) and numeric atom types the
/// resolver keys off, and the caller's variant.
///
/// Assembled once. `MmffTopology::build` + `set_mmff_aromaticity` is the expensive part
/// of MMFF typing and every step needs the result, so it is not re-derived.
struct MmffTypedMolecule<'a> {
    /// Molecule atom-iteration order — the index space `props` / `types` use.
    atom_ids: Vec<NodeId>,
    idx_of: HashMap<NodeId, usize>,
    props: MmffMolProperties,
    /// MMFF topology with perceived aromaticity — the resolver's ring / bond-order
    /// source, and the reason the type codes below can see what `classify.rs` could not.
    topo: MmffTopology,
    /// Numeric MMFF atom types, indexed as `atom_ids`.
    types: Vec<u8>,
    /// Typing metadata (atom-type properties), for the linear-centre flag.
    params: &'a MmffAtomProperties,
    variant: MmffVariant,
}

impl MmffTypedMolecule<'_> {
    /// MMFF numeric type of an atom, by id.
    fn type_of(&self, aid: NodeId) -> u32 {
        self.props.atom_type(self.idx_of[&aid]) as u32
    }

    /// Zero-based index of an atom, by id.
    fn idx(&self, aid: NodeId) -> usize {
        self.idx_of[&aid]
    }
}

/// `key` → a type named `name` under `style`, with the MMFF atom types
/// `endpoints` and `params`.
fn typed(
    key: &str,
    style: &str,
    name: String,
    endpoints: &[u32],
    params: Params,
) -> (String, Annotation) {
    (
        key.to_owned(),
        Annotation::Type {
            style: style.to_owned(),
            name,
            endpoints: endpoints.iter().map(u32::to_string).collect(),
            params,
        },
    )
}

/// The MMFF [`TypeAssignment`] of `graph` for `variant`:
/// - atoms: `type` (MMFF numeric type as string) + `charge` (MMFF partial
///   charge), both plain values — MMFF declares no atom style and its charges
///   are per-instance;
/// - bonds: `type` (e.g. `"0_1_5"`) → `mmff_bond {kb, r0}`
/// - angles: `type` (e.g. `"0_1_2_1"`) → `mmff_angle {ka, theta0}` (degrees);
///   `stbn_type` → `mmff_stbn {kba_ijk, kba_kji, r0_ij, r0_kj}`, named
///   `{sbt}_{i}_{j}_{k}` by the MMFF stretch-bend class of the angle read in its
///   own node order; `linear` (0/1: the central atom is a linear centre,
///   `linh != 0`) as an integer value
/// - dihedrals: `type` (e.g. `"0_5_1_1_5"`) → `mmff_torsion {v1, v2, v3}`
///   (kcal·mol⁻¹, **variant-dependent**); the type code carries an `@`
///   qualifier (`{tt}@{sec}` / `{tt}@{bond}`) when the principal lookup missed
///   (grammar in `annotate_dihedrals`)
/// - impropers: `type` = canonical MMFF out-of-plane key (e.g. `"0_37_37_37"`)
///   → `mmff_oop {koop}` (md·Å·rad⁻², **variant-dependent**); three Wilson rows
///   per trigonal centre, centre in the `atomi` position, sharing one `koop`
/// - styles: every style of `library`, in its order (compiled member order is
///   unchanged); pairs: the `mmff_vdw` rows of the atom types used.
///
/// Angles, dihedrals and impropers are enumerated onto `graph`. Every bonded
/// type is defined with explicit endpoints (the MMFF atom types its label
/// resolved through); the label grammar is `{type_code}_{atom_types...}`, and
/// both halves come from the same resolver call as the row's numbers.
///
/// `variant` is supplied by the typifier front door
/// ([`Mmff94Typifier`](super::Mmff94Typifier) /
/// [`Mmff94sTypifier`](super::Mmff94sTypifier)) and is threaded to **every**
/// parameter lookup below. Atom types and charges are variant-independent by
/// construction (MMFF94 and MMFF94s share all 95 types); `koop` and `(v1, v2, v3)`
/// are not.
pub(crate) fn annotate_mmff(
    graph: &mut Atomistic,
    params: &MmffAtomProperties,
    library: &ForceField,
    variant: MmffVariant,
) -> Result<TypeAssignment, String> {
    let typed_mol = type_molecule(graph, params, variant)?;
    let mut m = TypeAssignment {
        nodes: annotate_atoms(&typed_mol),
        ..TypeAssignment::default()
    };
    *m.link_mut(BONDS) = annotate_bonds(graph, &typed_mol);

    // Enumerate angles + dihedrals on the graph (impropers are MMFF-specific and
    // are enumerated by `annotate_impropers` below).
    crate::ff::typifier::topology::typify_bonded_topology(graph)?;

    *m.link_mut(ANGLES) = annotate_angles(graph, &typed_mol);
    *m.link_mut(DIHEDRALS) = annotate_dihedrals(graph, &typed_mol);
    *m.link_mut(IMPROPERS) = annotate_impropers(graph, &typed_mol)?;

    m.declare_styles_of(library);
    let used: Vec<String> = typed_mol.types.iter().map(u8::to_string).collect();
    let used: HashSet<&str> = used.iter().map(String::as_str).collect();
    m.add_pairs_among(library, &used);
    Ok(m)
}

/// The shared front-end: atom types, partial charges, MMFF topology.
fn type_molecule<'a>(
    mol: &Atomistic,
    params: &'a MmffAtomProperties,
    variant: MmffVariant,
) -> Result<MmffTypedMolecule<'a>, String> {
    // The RDKit-validated front-end for atom types + MMFF partial charges. Its
    // per-atom index is the molecule's atom iteration order — the same order as
    // `atom_ids`.
    let props = MmffMolProperties::compute(mol).map_err(|e| e.to_string())?;

    let atom_ids: Vec<NodeId> = mol.atoms().map(|(id, _)| id).collect();
    let idx_of: HashMap<NodeId, usize> = atom_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // The MMFF topology drives every per-instance parameter and type-code lookup
    // below. Aromaticity is *perceived* here — which is precisely the fact the
    // deleted classifier never saw, because it was handed raw bond orders instead.
    let base = MmffTopology::build(mol).map_err(|s| format!("MMFF MmffTopology: {s}"))?;
    let topo = crate::perceive::mmff_aromaticity::set_mmff_aromaticity(&base);
    let types: Vec<u8> = (0..atom_ids.len()).map(|i| props.atom_type(i)).collect();

    Ok(MmffTypedMolecule {
        atom_ids,
        idx_of,
        props,
        topo,
        types,
        params,
        variant,
    })
}

// --- 1. Atoms ------------------------------------------------------------

/// Validated MMFF numeric type + MMFF partial charge on every atom.
fn annotate_atoms(typed_mol: &MmffTypedMolecule) -> Annotations {
    (0..typed_mol.atom_ids.len())
        .map(|i| {
            vec![
                (
                    "type".to_owned(),
                    Annotation::Value(PropValue::Str(typed_mol.props.atom_type(i).to_string())),
                ),
                (
                    "charge".to_owned(),
                    Annotation::Value(PropValue::F64(typed_mol.props.partial_charge(i))),
                ),
            ]
        })
        .collect()
}

// --- 2. Bonds ------------------------------------------------------------

/// MMFF bond type + the per-bond `kb` / `r0` (table → equivalence → empirical).
fn annotate_bonds(graph: &Atomistic, typed_mol: &MmffTypedMolecule) -> Annotations {
    graph
        .bonds()
        .map(|(_, bond)| {
            let (a, b) = (bond.nodes[0], bond.nodes[1]);
            let (ia, ib) = (typed_mol.idx(a), typed_mol.idx(b));
            let (t1, t2) = (typed_mol.type_of(a), typed_mol.type_of(b));
            let (lo, hi) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
            let bt = eparams::bond_type(&typed_mol.topo, &typed_mol.types, ia, ib);

            let (kb, r0) = eparams::bond_params(&typed_mol.topo, &typed_mol.types, ia, ib)
                .map(|bp| (bp.kb, bp.r0))
                .unwrap_or((0.0, 0.0));
            vec![typed(
                "type",
                "mmff_bond",
                format!("{bt}_{lo}_{hi}"),
                &[lo, hi],
                Params::from_iter([("kb", kb), ("r0", r0)]),
            )]
        })
        .collect()
}

// --- 3. Angles (+ stretch-bend) ------------------------------------------

/// MMFF angle type + `ka` / `theta0` / the stretch-bend constants and their two
/// reference bond lengths, plus the linear-centre flag.
fn annotate_angles(graph: &Atomistic, typed_mol: &MmffTypedMolecule) -> Annotations {
    graph
        .angles()
        .map(|(_, angle)| {
            let (a, b, c) = (angle.nodes[0], angle.nodes[1], angle.nodes[2]);
            let (ia, ib, ic) = (typed_mol.idx(a), typed_mol.idx(b), typed_mol.idx(c));
            let ends = [
                typed_mol.type_of(a),
                typed_mol.type_of(b),
                typed_mol.type_of(c),
            ];
            let [ta, tb, tc] = ends;
            // The ring-aware angle type: an angle inside a 3-/4-membered ring is
            // promoted to 3..8, which is exactly what a `(bt_ij, bt_jk)` signature
            // could never express.
            let at = eparams::angle_type(&typed_mol.topo, &typed_mol.types, ia, ib, ic);
            // The stretch-bend class of this angle read in its OWN node order:
            // it says which of the two bonds is the type-1 bond, so the label
            // tells the two orientations of `(kba, r0)` apart. (The resolver's
            // own class swaps its bond types by atom-type order to find the
            // table row; for `ti == tk` both orientations get one class there.)
            let sbt = eparams::stretch_bend_type(
                at,
                eparams::bond_type(&typed_mol.topo, &typed_mol.types, ia, ib),
                eparams::bond_type(&typed_mol.topo, &typed_mol.types, ib, ic),
            );

            // Linear-centre flag, from the CENTRAL atom's `linh` property (nitrile,
            // alkyne, allene, isocyanate…). It selects a different functional form for
            // the bend — `E = 143.9325·ka·(1 + cos θ)` instead of the cubic expansion
            // about theta0 — and suppresses the stretch-bend term at that centre; both
            // kernels (`mmff_angle`, `mmff_stbn`) read this one column. Baked as 0/1
            // rather than a bool because `Atomistic::to_frame` carries only f64 / i32 /
            // string columns into the Frame; a bool would be silently dropped.
            let linear = typed_mol
                .params
                .get(tb as u8)
                .map(|p| p.linh != 0)
                .unwrap_or(false);

            // `theta0` in degrees, as MMFF's tables and every molrs angle parameter
            // are; the angle / stretch-bend kernels convert it once.
            let (ka, theta0) = eparams::angle_params(&typed_mol.topo, &typed_mol.types, ia, ib, ic)
                .map(|p| (p.ka, p.theta0))
                .unwrap_or((0.0, 0.0));

            // Stretch-bend force constants — `stretch_bend_params` carries the `dfsb`
            // period-row default fallback that a table-keyed path lacks (the benzene
            // `mmff_stbn: unknown` blocker). The two reference bond lengths are the
            // per-bond r0, taken straight from the bond resolver.
            let (kba_ijk, kba_kji) =
                eparams::stretch_bend_params(&typed_mol.topo, &typed_mol.types, ia, ib, ic)
                    .map(|(s, _, _, _)| (s.kba_ijk, s.kba_kji))
                    .unwrap_or((0.0, 0.0));
            let r0_ij = eparams::bond_params(&typed_mol.topo, &typed_mol.types, ia, ib)
                .map(|b| b.r0)
                .unwrap_or(0.0);
            let r0_kj = eparams::bond_params(&typed_mol.topo, &typed_mol.types, ic, ib)
                .map(|b| b.r0)
                .unwrap_or(0.0);

            vec![
                typed(
                    "type",
                    "mmff_angle",
                    format!("{at}_{ta}_{tb}_{tc}"),
                    &ends,
                    Params::from_iter([("ka", ka), ("theta0", theta0)]),
                ),
                typed(
                    "stbn_type",
                    "mmff_stbn",
                    format!("{sbt}_{ta}_{tb}_{tc}"),
                    &ends,
                    Params::from_iter([
                        ("kba_ijk", kba_ijk),
                        ("kba_kji", kba_kji),
                        ("r0_ij", r0_ij),
                        ("r0_kj", r0_kj),
                    ]),
                ),
                (
                    "linear".to_owned(),
                    Annotation::Value(PropValue::Int(i32::from(linear))),
                ),
            ]
        })
        .collect()
}

// --- 4. Dihedrals --------------------------------------------------------

/// MMFF torsion type + the variant's Fourier coefficients `(v1, v2, v3)`.
///
/// 42 of the torsion rows are re-parameterised by MMFF94s, all on delocalised
/// trivalent nitrogen — which is why the variant has to reach this lookup.
///
/// # The torsion label
///
/// The label names the rule the numbers came from
/// ([`TorSource`](eparams::TorSource)) together with every input that rule
/// reads, so one label names one parameter set:
///
/// - `{tt}_{i}_{j}_{k}_{l}` — a table row under the principal torsion type
///   `tt`. Whether this pass hits is fixed by `tt`, the four MMFF atom types
///   and the variant, so this form never collides with the two below.
/// - `{tt}@{sec}_{i}_{j}_{k}_{l}` — no row under `tt`; a row found on the
///   restart under the secondary torsion type `sec` (the one a 4-/5-ring
///   promotion displaced). Two ring torsions with equal atom types but
///   different `sec` resolve to different rows.
/// - `{tt}@{b}_{i}_{j}_{k}_{l}` — no row under either type: the MMFF.V
///   empirical rules, which read the central types (they fix the element and
///   every MMFF atom property the rules use) and the j–k bond class `b`,
///   written as a SMILES bond symbol: `-` single, `=` double, `:` aromatic
///   (both central atoms aromatic MMFF types on a perceived-aromatic bond —
///   rule (b)), `~` anything else (the rules treat triple and a non-rule-(b)
///   aromatic bond alike). `sec` is omitted — the rules do not read it.
///
/// The qualifier appears only when the principal lookup missed, so every
/// label a table hit on the principal type produced is unchanged.
fn annotate_dihedrals(graph: &Atomistic, typed_mol: &MmffTypedMolecule) -> Annotations {
    graph
        .dihedrals()
        .map(|(_, dihedral)| {
            let [a, b, c, d] = [
                dihedral.nodes[0],
                dihedral.nodes[1],
                dihedral.nodes[2],
                dihedral.nodes[3],
            ];
            let (ia, ib, ic, il) = (
                typed_mol.idx(a),
                typed_mol.idx(b),
                typed_mol.idx(c),
                typed_mol.idx(d),
            );
            let ends = [
                typed_mol.type_of(a),
                typed_mol.type_of(b),
                typed_mol.type_of(c),
                typed_mol.type_of(d),
            ];
            // `torsion_type` returns `(principal, secondary)`; the principal code
            // leads the label (the 4-/5-ring promotions live in it), and the
            // secondary one enters through the source when the lookup reads it.
            let (tt, _) = eparams::torsion_type(&typed_mol.topo, &typed_mol.types, ia, ib, ic, il);
            let (source, p) = eparams::torsion_params(
                typed_mol.variant,
                &typed_mol.topo,
                &typed_mol.types,
                ia,
                ib,
                ic,
                il,
            );
            let (v1, v2, v3) = p.map(|t| (t.v1, t.v2, t.v3)).unwrap_or((0.0, 0.0, 0.0));
            let code = match source {
                eparams::TorSource::Principal => tt.to_string(),
                eparams::TorSource::Secondary(sec) => format!("{tt}@{sec}"),
                eparams::TorSource::Empirical(bond) => {
                    let symbol = match bond {
                        eparams::JkBond::Single => '-',
                        eparams::JkBond::Double => '=',
                        eparams::JkBond::Aromatic => ':',
                        eparams::JkBond::Other => '~',
                    };
                    format!("{tt}@{symbol}")
                }
            };
            vec![typed(
                "type",
                "mmff_torsion",
                format!("{code}_{}_{}_{}_{}", ends[0], ends[1], ends[2], ends[3]),
                &ends,
                Params::from_iter([("v1", v1), ("v2", v2), ("v3", v3)]),
            )]
        })
        .collect()
}

// --- 5. Out-of-plane (Wilson) --------------------------------------------

/// MMFF-specific out-of-plane enumeration.
///
/// Only atoms with *exactly three* neighbours are trigonal centres; each
/// contributes three Wilson permutations that share one `koop`. The centre is
/// placed **first** (`atomi`), the improper order of every molrs out-of-plane
/// style (LAMMPS's `fourier` / `umbrella`), which the `mmff_oop` kernel reads.
/// The `type` label is the canonical OOP key that [`eparams::out_of_plane_params`]
/// matched on (peripherals equivalence-degraded and sorted, centre second, as
/// MMFF writes it), so the label names the row the `koop` came from; the type's
/// endpoints are its four fields with the centre moved first. Centres for which
/// MMFF defines no out-of-plane term are skipped.
///
/// The impropers are added to `graph`; the returned annotations are positional
/// against all of its impropers, a pre-existing one getting none.
fn annotate_impropers(
    graph: &mut Atomistic,
    typed_mol: &MmffTypedMolecule,
) -> Result<Annotations, String> {
    let n = typed_mol.atom_ids.len();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (_, bond) in graph.bonds() {
        let (a, b) = (typed_mol.idx(bond.nodes[0]), typed_mol.idx(bond.nodes[1]));
        adjacency[a].push(b);
        adjacency[b].push(a);
    }

    let mut added = HashMap::new();
    for (center, neighbours) in adjacency.iter().enumerate() {
        // Exactly three neighbours — the definition of a trigonal centre — said as
        // a pattern, so the arity check and the destructuring cannot disagree.
        let &[a, b, c] = &neighbours[..] else {
            continue;
        };

        // Per-centre out-of-plane force constant `koop` (md·Å·rad⁻²), shared by all
        // three Wilson permutations — the OOP lookup is symmetric in the peripheral
        // atoms — resolved for the caller's variant; the kernel reads the column and
        // evaluates `E_oop = 0.5 · 143.9325 · koop · χ²` with χ in radians. This is
        // the one number MMFF94s changes on a delocalised trivalent nitrogen.
        let Some((label, koop)) =
            eparams::out_of_plane_params(typed_mol.variant, &typed_mol.types, a, center, b, c)
        else {
            continue;
        };
        let mut ends: Vec<u32> = label
            .split('_')
            .map(|t| {
                t.parse::<u32>()
                    .map_err(|_| format!("MMFF out-of-plane key {label:?} is not four atom types"))
            })
            .collect::<Result<_, _>>()?;
        if ends.len() != 4 {
            return Err(format!(
                "MMFF out-of-plane key {label:?} is not four atom types"
            ));
        }
        // MMFF's key lists the centre second; the improper lists it first.
        ends.swap(0, 1);

        let center_id = typed_mol.atom_ids[center];
        for &(i, k, l) in &[(a, b, c), (a, c, b), (b, c, a)] {
            let id = graph
                .add_improper(
                    center_id,
                    typed_mol.atom_ids[i],
                    typed_mol.atom_ids[k],
                    typed_mol.atom_ids[l],
                    Props::new(),
                )
                .map_err(|e| e.to_string())?;
            added.insert(
                id,
                vec![typed(
                    "type",
                    "mmff_oop",
                    label.clone(),
                    &ends,
                    Params::from_iter([("koop", koop)]),
                )],
            );
        }
    }

    Ok(graph
        .impropers()
        .map(|(id, _)| added.remove(&id).unwrap_or_default())
        .collect())
}
