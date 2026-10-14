//! Radical (Laguerre) Voronoi tessellation + its first two consumers.
//!
//! Gated behind the `voronoi` feature (which implies `compute`). The default
//! backend is a **native pure-Rust** cell-by-cell radical tessellation
//! ([`RadicalVoronoi`]) — no C/C++ FFI, WASM-clean — ported from voro++
//! (`src/v_cell.cpp`, `src/v_rad_option.h`, `src/v_container_prd.cpp`) as used
//! by the reference implementation (`vorowrapper.cpp`). Its consumers:
//! [`VoronoiDomainAnalysis`] (microheterogeneity / ionic-liquid domains, `domain.cpp`),
//! [`VoronoiVoidAnalysis`] (cavity / free-volume, `void.cpp`),
//! [`VoronoiIntegration`] (per-molecule charges and dipoles from an electron
//! density) and [`FiniteFieldPolarizability`] (α from ±E dipoles).
//!
//! Every type here is a per-snapshot [`Compute`](crate::compute::Compute);
//! many snapshots go through [`Batch`](crate::compute::Batch). Candidate
//! search runs on the core cell list
//! ([`LinkCell`](molrs::core::LinkCell)), and the per-molecule reference of
//! [`VoronoiIntegration`] on the one group reduction.
//!
//! Layer: `compute` → `core` (`SimBox`, `LinkCell`, `GhostSet`); no new dependency.

mod cell;
mod domain;
mod integrate;
mod polarizability;
mod radical;
mod void;

pub use cell::{VORONOI_BOUNDARY, VoronoiCells, VoronoiFace};
pub use domain::{VoronoiDomainAnalysis, VoronoiDomainInput, VoronoiDomainResult};
pub use integrate::{LengthUnit, MolecularMoments, VoronoiIntegration, VoronoiIntegrationInput};
pub use polarizability::{FiniteFieldPolarizability, FiniteFieldPolarizabilityInput};
pub use radical::{RadicalVoronoi, RadicalVoronoiInput};
pub use void::{VoronoiVoidAnalysis, VoronoiVoidInput, VoronoiVoidResult};
