//! The core data model: one public namespace, `molrs::core`.
//!
//! Every public name is flat on `molrs::core` — `molrs::core::Frame`,
//! `molrs::core::SimBox`, `molrs::core::Atomistic`, `molrs::core::Quantity`.
//! The implementation files are private. Three public submodules remain, each
//! a vocabulary whose names belong together:
//!
//! - [`keys`] — canonical column, block-group and frame-meta keys;
//! - [`schema`] — the column and block specifications those keys come from,
//!   and the validator that judges a frame against them;
//! - [`constants`] — physical and engine constants (CODATA 2018 / SI 2019,
//!   and the Coulomb and scaling constants each engine uses).
//!
//! What the flat surface holds:
//!
//! - columnar data: [`Block`], [`Frame`], [`Trajectory`] and their metadata;
//! - the molecular graph: [`MolGraph`], [`Atomistic`], [`CoarseGrain`],
//!   [`Port`], [`Topology`], [`Element`], [`BondOrder`];
//! - space: [`SimBox`], the regions ([`Region`], [`Sphere`], …), neighbour
//!   search ([`NeighborList`], [`VerletSkin`], …), [`GhostSet`],
//!   [`TriMesh`], [`Trace`];
//! - numerics and units: [`Complex`], the spherical harmonics and Wigner
//!   symbols, [`Virial`], [`Unit`], [`Quantity`], [`UnitRegistry`],
//!   [`UnitFactor`], [`UnitPreset`];
//! - the crate error, [`MolRsError`].
//!
//! # Units: one definition each
//!
//! Every unit conversion in molrs goes through the unit registry: a
//! [`UnitFactor`] written as its two units, each defined once in
//! [`unit_factors`] (`unit_factors::KCAL_TO_KJ`, resolved once),
//! [`UnitRegistry::factor`] or [`Quantity::to`]. No module writes a factor by hand (`* 4.184`,
//! `/ 10.0` for nm), and [`constants`] holds physical constants and the
//! constants engines define as data, never a conversion factor; the units
//! are built from those constants (`bohr` from [`constants::BOHR_RADIUS`],
//! `eV` from [`constants::ELEMENTARY_CHARGE`]). `module_boundaries` fails on
//! a conversion-factor constant or literal outside the units module.
//! Degrees ↔ radians is `to_radians` / `to_degrees` (the registry's `deg`
//! is the same π/180).
//!
//! Whole-graph transforms are methods of every graph type
//! ([`Atomistic::translate`], [`Atomistic::rotate`], [`Atomistic::scale`],
//! [`Atomistic::center`], and the same on [`CoarseGrain`] and [`MolGraph`]);
//! an algorithm that serves every graph type takes the sealed [`Graph`]
//! bound; ring
//! perception is `crate::perceive::perceive_rings`; the `*.mrec` record and
//! its force-field section are `crate::io::mrec`.
//!
//! # Examples
//!
//! ```
//! use molrs::core::Element;
//!
//! let hydrogen = Element::by_number(1).unwrap();
//! assert_eq!(hydrogen.symbol(), "H");
//! assert_eq!(Element::by_symbol("h").unwrap().name(), "Hydrogen");
//! ```

#![allow(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

// Vocabularies: the only public submodules.
pub mod constants;
pub mod keys;
pub mod schema;
pub mod unit_factors;

// Columnar data.
mod block;
mod frame;
mod frame_access;
mod frame_view;
mod metadata;
pub(crate) mod precision;
mod trajectory;
pub(crate) mod type_labels;
pub(crate) mod typed_json;

// The molecular graph.
mod atomistic;
pub(crate) mod bond_order;
mod bond_weights;
mod coarsegrain;
mod element;
mod entity_table;
mod extract;
mod graph;
pub(crate) mod graph_hash;
mod link;
mod molgraph;
mod molgraph_geometry;
mod port;
mod topology;

// Space.
pub(crate) mod bvh;
mod mesh;
pub(crate) mod neighbors;
mod periodic;
mod region;
mod simbox;
mod trace;

// Numerics, units, errors.
mod error;
mod math;
mod units;

#[cfg(all(test, feature = "rayon"))]
pub(crate) mod test_thread_pool;

pub use block::{
    Block, BlockAccess, BlockDtype, BlockError, BlockView, Column, ColumnArray, ColumnView, DType,
};
pub use frame::Frame;
pub use frame_access::FrameAccess;
pub use frame_view::FrameView;
pub use metadata::{MetaIter, MetaMap, MetaValue};
pub use precision::{
    PRECISION_MAX, PRECISION_MIN, check_precision, quantize, quantize_in_place, quantum,
};
pub use trajectory::{ObservableKind, ObservableRecord, ObservableValues, Trajectory};
pub use type_labels::{BlockTypeLabels, TypeLabels, TypeName};

pub use atomistic::{Atomistic, ExtractedAtomistic};
pub use bond_order::{BondNumber, BondOrder};
pub use bond_weights::BondDistanceWeights;
pub use coarsegrain::{CoarseGrain, ExtractedCoarseGrain};
pub use element::Element;
pub use entity_table::{EntityCell, EntityColumn, EntityDtype, EntityTable, Validity};
pub use extract::{ExtractedBall, InducedSubgraph};
pub use graph::Graph;
pub(crate) use graph::GraphAccess;
pub use link::{LinkError, LinkManyError};
pub use molgraph::{KindId, MolGraph, NodeId, PropValue, Props, Relation, RelationId};
pub use molgraph_geometry::CenterError;
pub use port::{Port, PortKind};
pub use topology::{Topology, TopologyError};

pub use mesh::{DEGENERATE_AREA2, TriMesh};
pub use neighbors::{
    AabbQuery, BruteForce, CellGrid, LinkCell, NeighborColumns, NeighborList, NeighborPair,
    NeighborPolicy, NeighborQuery, Neighbors, PairSource, QueryMode, SkinError, SkinPair,
    VerletSkin, filter_rad, filter_sann,
};
pub use periodic::{GhostError, GhostHalo, GhostSet, ImageRange};
pub use region::{
    AndRegion, Cuboid, Cylinder, Ellipsoid, HalfSpace, NotRegion, OrRegion, Parallelepiped,
    Polyhedron, PolyhedronError, Region, Sphere, SphereUnion, SphereUnionError,
};
pub use simbox::{BoxError, BoxKind, Cell, Mic, Points, SimBox};
pub use trace::Trace;

pub use error::MolRsError;
pub use math::{
    Complex, Virial, legendre_plm, wigner_3j, wigner_d_element, wigner_d_matrix, wigner_small_d,
    ylm_all, ylm_complex, ylm_normalization, ylm_real,
};
#[cfg(feature = "compute")]
pub(crate) use math::{FOUR_PI, FOUR_THIRDS_PI};
pub use units::{
    Dimension, PresetDim, Quantity, Unit, UnitDef, UnitFactor, UnitPreset, UnitPresetRegistry,
    UnitRegistry, UnitsError, lookup_unit_preset, register_unit_preset, replace_unit_preset,
    unit_preset_names,
};

#[cfg(test)]
mod contract_regressions {
    use super::*;
    use crate::props;
    use ndarray::Array1;

    fn block_u64(key: &str, values: &[u64]) -> Block {
        let mut b = Block::new();
        b.insert(key, Array1::from_vec(values.to_vec()).into_dyn(), None)
            .unwrap();
        b
    }

    #[test]
    fn tilted_cell_uses_the_nearest_image() {
        let bx = SimBox::new(
            Cell::Matrix([[10., 9., 0.], [0., 2., 0.], [0., 0., 10.]]),
            [0.; 3],
            [true; 3],
        )
        .unwrap();
        let d = bx.mic().apply([9.31, 0.98, 0.]);
        assert!(
            (d[0] - 0.31).abs() < 1e-12 && (d[1] + 1.02).abs() < 1e-12,
            "{d:?}"
        );
        let partial = SimBox::new(
            Cell::Matrix([[10., 9., 0.], [0., 2., 0.], [0., 0., 10.]]),
            [0.; 3],
            [true, false, false],
        )
        .unwrap();
        let d = partial.mic().apply([9.31, 0.98, 0.]);
        assert!((d[0] + 0.69).abs() < 1e-12);
    }

    #[test]
    fn concatenation_refuses_conflicting_reference_targets() {
        let mut a = block_u64("ref", &[0]);
        a.set_target("ref", "atoms").unwrap();
        let mut b = block_u64("ref", &[0]);
        b.set_target("ref", "sites").unwrap();
        assert!(Block::stack([&a, &b]).is_err());
        assert!(a.merge(&b).is_err());
    }

    #[test]
    fn subset_keeps_indirect_references_consistent() {
        let mut f = Frame::new();
        f.insert("atoms", block_u64("id", &[0, 1, 2]));
        let mut bonds = block_u64("atomi", &[0, 1]);
        bonds
            .insert("atomj", Array1::from_vec(vec![1u64, 2]).into_dyn(), None)
            .unwrap();
        f.insert("bonds", bonds);
        let mut refs = block_u64("edge", &[1]);
        refs.set_target("edge", "bonds").unwrap();
        f.insert("refs", refs);
        let s = f.subset("atoms", &[1, 2]).unwrap();
        s.validate().unwrap();
        assert_eq!(s["refs"].get("edge").unwrap().get::<u64>().unwrap()[[0]], 0);
    }

    #[test]
    fn block_transforms_preserve_cardinality_and_grid_shape() {
        let mut b = Block::new();
        b.resize(3).unwrap();
        assert_eq!(b.select_rows(&[1]).unwrap().n_rows(), Some(1));
        assert_eq!(b.select_columns(&[]).unwrap().n_rows(), Some(3));
        assert_eq!(BlockView::from(&b).to_owned().unwrap().n_rows(), Some(3));
        let mut grid = block_u64("value", &[0, 1, 2, 3, 4, 5]);
        grid.set_shape(&[2, 3]).unwrap();
        assert_eq!(grid.select_columns(&["value"]).unwrap().shape(), vec![2, 3]);
        let mut empty = Block::new();
        empty.set_shape(&[2, 3]).unwrap();
        empty.resize(4).unwrap();
        assert_eq!(empty.shape(), vec![4]);
    }

    #[test]
    fn borrowed_conversion_reports_invalid_columns() {
        let data = Array1::from_vec(vec![1i32, 2]);
        let mut view = BlockView::new();
        view.insert("x", ColumnView::Int(data.view().into_dyn()));
        assert!(view.to_owned().is_err());
    }

    #[test]
    fn graph_import_refuses_unrepresentable_columns() {
        let mut atoms = block_u64("id", &[0]);
        atoms
            .insert(
                "atomic_number",
                Array1::from_vec(vec![6u64]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert("tiny", Array1::from_vec(vec![7u8]).into_dyn(), None)
            .unwrap();
        let mut f = Frame::new();
        f.insert("atoms", atoms);
        assert!(Atomistic::from_frame(&f).is_err());
    }

    #[test]
    fn topology_refuses_null_endpoints() {
        let mut bonds = block_u64("atomi", &[0]);
        bonds.set_validity("atomi", vec![false]).unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1u64]).into_dyn(), None)
            .unwrap();
        let mut f = Frame::new();
        f.insert("atoms", block_u64("id", &[0, 1]));
        f.insert("bonds", bonds);
        assert!(Topology::from_frame(&f).is_err());
    }

    #[test]
    fn membership_rejects_duplicates_and_nulls() {
        let mut cg = CoarseGrain::new();
        let bead = cg.add_bead(props! { "bead_type" => "A" }).unwrap();
        assert!(cg.set_bead_members(bead, vec![1, 1]).is_err());
        cg.set_bead_members(bead, vec![1]).unwrap();
        let mut frame = cg.to_frame().unwrap();
        frame
            .get_mut("members")
            .unwrap()
            .set_validity("atom", vec![false])
            .unwrap();
        assert!(CoarseGrain::from_frame(&frame).is_err());
    }

    #[test]
    fn incomplete_coordinates_are_not_transformed() {
        let mut mol = Atomistic::new();
        let atom = mol
            .add_atom(props! { "atomic_number" => 6, "x" => 2.0 })
            .unwrap();
        mol.translate([3., 4., 5.]);
        mol.scale([2.; 3], None);
        assert_eq!(mol.get_atom(atom).unwrap().get_f64("x"), Some(2.));
    }

    #[test]
    fn identity_uses_formal_charge_not_partial_charge() {
        let mut a = Atomistic::new();
        let atom = a.add_atom(props! { "atomic_number" => 6 }).unwrap();
        let mut b = a.clone();
        b.set_atom(atom, "formal_charge", 1).unwrap();
        assert!(!a.is_isomorphic(&b));
        let mut c = a.clone();
        c.set_atom(atom, "charge", 0.5).unwrap();
        assert!(a.is_isomorphic(&c));
    }

    #[test]
    fn canonical_mapping_preserves_edges_in_symmetric_graphs() {
        fn ring(edges: &[(usize, usize)]) -> Atomistic {
            let mut m = Atomistic::new();
            let a: Vec<_> = (0..4)
                .map(|_| m.add_atom(props! { "atomic_number" => 6 }).unwrap())
                .collect();
            for &(i, j) in edges {
                m.add_bond(a[i], a[j], props! {}).unwrap();
            }
            m
        }
        let a = ring(&[(0, 1), (1, 2), (2, 3), (3, 0)]);
        let b = ring(&[(0, 2), (2, 1), (1, 3), (3, 0)]);
        let oa = a.canonical_order();
        let ob = b.canonical_order();
        for i in 0..4 {
            for j in 0..4 {
                let has = |m: &Atomistic, x, y| m.neighbor_bonds(x).any(|(n, _)| n == y);
                assert_eq!(has(&a, oa[i], oa[j]), has(&b, ob[i], ob[j]));
            }
        }
    }
    #[test]
    fn large_tilt_nearest_image_does_not_enumerate_a_cartesian_box() {
        let bx = SimBox::new(
            Cell::Matrix([[1., 1000., 0.], [0., 1., 0.], [0., 0., 1.]]),
            [0.; 3],
            [true; 3],
        )
        .unwrap();
        let d = bx.mic().apply([490.49, 0.49, 0.49]);
        for value in d {
            assert!((value - 0.49).abs() < 1e-10, "{d:?}");
        }
    }

    #[test]
    fn nearest_images_match_an_independent_image_enumeration() {
        for pbc in [[true; 3], [true, false, true], [false, true, false]] {
            let h = [[3., 2.7, 1.4], [0., 2., 1.2], [0., 0., 2.5]];
            let bx = SimBox::new(Cell::Matrix(h), [0.; 3], pbc).unwrap();
            for f in [[0.49, 0.49, 0.49], [-0.7, 0.8, -0.9], [0.2, -0.8, 0.6]] {
                let d: [f64; 3] = std::array::from_fn(|i| (0..3).map(|j| h[i][j] * f[j]).sum());
                let got = bx.mic().apply(d);
                let got2 = got.iter().map(|v| v * v).sum::<f64>();
                let mut want = f64::INFINITY;
                for i in -4..=4 {
                    for j in -4..=4 {
                        for k in -4..=4 {
                            let n = [i, j, k];
                            if (0..3).any(|a| !pbc[a] && n[a] != 0) {
                                continue;
                            }
                            let image: [f64; 3] = std::array::from_fn(|a| {
                                d[a] - (0..3).map(|b| h[a][b] * n[b] as f64).sum::<f64>()
                            });
                            want = want.min(image.iter().map(|v| v * v).sum());
                        }
                    }
                }
                assert!(
                    (got2 - want).abs() < 1e-11,
                    "{pbc:?} {f:?}: {got2} != {want}"
                );
            }
        }
    }

    #[test]
    fn canonical_labeling_prunes_symmetric_parallel_relations() {
        let mut cg = CoarseGrain::new();
        let center = cg.add_bead(props! {"bead_type" => "A"}).unwrap();
        for _ in 0..20 {
            let leaf = cg.add_bead(props! {"bead_type" => "A"}).unwrap();
            cg.add_bond(center, leaf, props! {}).unwrap();
            cg.add_port(center, leaf, PortKind::Symmetric, "", BondNumber::Single)
                .unwrap();
        }
        assert_eq!(cg.canonical_order().len(), 21);
        let restored = CoarseGrain::from_frame(&cg.to_frame().unwrap()).unwrap();
        assert_eq!(restored.n_ports(), 20);
    }

    #[test]
    fn empty_relations_need_no_target_rows_for_subset() {
        let mut f = Frame::new();
        let mut bonds = block_u64("atomi", &[]);
        bonds
            .insert("atomj", Array1::<u64>::from_vec(vec![]).into_dyn(), None)
            .unwrap();
        f.insert("bonds", bonds);
        let out = f.subset("bonds", &[]).unwrap();
        assert_eq!(out["bonds"].n_rows(), Some(0));
    }

    #[test]
    fn replacing_a_borrowed_column_cannot_install_stale_row_counts() {
        let first = Array1::from_vec(vec![1., 2.]);
        let second = Array1::from_vec(vec![1., 2., 3.]);
        let mut view = BlockView::new();
        view.insert("custom", ColumnView::Float(first.view().into_dyn()));
        view.insert("custom", ColumnView::Float(second.view().into_dyn()));
        let owned = view.to_owned().unwrap();
        assert_eq!(owned.n_rows(), Some(3));
    }

    #[test]
    fn membership_refuses_multidimensional_atom_handles() {
        let mut cg = CoarseGrain::new();
        let bead = cg.add_bead(props! {"bead_type" => "A"}).unwrap();
        cg.set_bead_members(bead, vec![10]).unwrap();
        let mut frame = cg.to_frame().unwrap();
        frame
            .get_mut("members")
            .unwrap()
            .insert("atom", ndarray::arr2(&[[10u64, 11]]).into_dyn(), None)
            .unwrap();
        assert!(CoarseGrain::from_frame(&frame).is_err());
    }
    #[test]
    fn canonical_order_handles_many_interchangeable_components() {
        let mut graph = MolGraph::new();
        for _ in 0..10_000 {
            graph.add_node(props! {}).unwrap();
        }
        assert_eq!(graph.canonical_order().len(), 10_000);
        let mut dimers = MolGraph::new();
        let kind = dimers.register_kind("links", 2);
        for _ in 0..500 {
            let a = dimers.add_node(props! {}).unwrap();
            let b = dimers.add_node(props! {}).unwrap();
            dimers.add_relation(kind, &[a, b], props! {}).unwrap();
        }
        let order = dimers.canonical_order();
        for pair in order.as_chunks::<2>().0 {
            assert!(dimers.neighbors(pair[0]).any(|n| n == pair[1]));
        }
    }

    #[test]
    fn borrowed_conversion_validates_stale_masks() {
        let mut block = Block::new();
        block
            .insert(
                "custom",
                Array1::from_vec(vec![1., 2.]).into_dyn(),
                Some(vec![false, true]),
            )
            .unwrap();
        let longer = Array1::from_vec(vec![1., 2., 3.]);
        let mut view = BlockView::from(&block);
        view.insert("custom", ColumnView::Float(longer.view().into_dyn()));
        assert!(view.to_owned().is_err());
    }

    #[test]
    fn null_relation_rows_do_not_require_a_missing_default_target() {
        let mut bonds = block_u64("atomi", &[0]);
        bonds.set_validity("atomi", vec![false]).unwrap();
        bonds
            .insert(
                "atomj",
                Array1::from_vec(vec![0u64]).into_dyn(),
                Some(vec![false]),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("bonds", bonds);
        frame.validate().unwrap();
        frame.subset("bonds", &[0]).unwrap().validate().unwrap();
    }
}
