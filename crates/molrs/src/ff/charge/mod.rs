//! Charge models: molecule in, charges out.
//!
//! One trait, [`ChargeModel`], and the models behind it:
//!
//! | model | needs QM input? | topology correction? | [`needs_equivalencing`] |
//! |---|---|---|---|
//! | [`MullikenModel`] | yes | no (pass-through) | `false` |
//! | [`BccModel`] (AM1-BCC) | yes | yes (bond increments) | `true` |
//! | [`BccModel`] (ABCG2) | yes | yes (bond increments) | `true` |
//! | [`GasteigerModel`] | **no** | yes (iterative PEOE) | `false` |
//!
//! The last row is the one that keeps the trait honest. Gasteiger is handed a
//! molecule and nothing else — `assign(&mol, None)` — and still lands on an
//! antechamber column, with no branch for it anywhere in this module. A trait that
//! needed a special case to host a model with no QM input would be a trait that had
//! quietly assumed "QM base charges plus a correction".
//!
//! # The seam is a push, not a pull
//!
//! QM charges are an **argument** — `assign(&mol, Some(&am1))`,
//! `BccModel::correct(&mol, &am1)` — because molrs does not compute them. A
//! backend trait that asks for them would have implementors that ignore the
//! molecule they are handed and return a vector computed elsewhere, a
//! `Vec<f64>` dressed up as a solver. The charges go in the direction they
//! actually travel.
//!
//! # Nothing is written into the molecule
//!
//! Every model takes `&Atomistic` and returns `Vec<f64>`. The charges are the return
//! value, and the models' internal atom types (BCC codes like `11` / `91`) stay
//! internal — a caller keeps their GAFF or OPLS types in [`keys::TYPE`](molrs::core::keys::TYPE)
//! and gets BCC charges back, which is what the standard AM1-BCC workflow needs.
//!
//! # Equivalencing is the model's declaration, and the model honours it
//!
//! [`ChargeModel::needs_equivalencing`] is antechamber's `-eq`, whose default is per
//! charge *method*: `1` for `bcc` / `abcg2` / `resp`, `0` for everything else. So
//! [`assign`](ChargeModel::assign) applies the class-mean itself when the model
//! declares it needs it. Handed the raw `sqm` Mulliken charges an AM1 solver really
//! produces, [`BccModel`] averages them over the topological-equivalence classes and
//! then corrects; [`MullikenModel`] hands the same bits back.
//!
//! ```
//! use molrs::core::{Atomistic, Element, keys};
//! use molrs::props;
//! use molrs::ff::charge::{BccModel, BccParameterSet, ChargeModel, MullikenModel};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut mol = Atomistic::new();
//! let o = mol.add_atom(props! {
//!     keys::ATOMIC_NUMBER => Element::O,
//!     keys::X => 0.0,
//!     keys::Y => 0.0,
//!     keys::Z => 0.0,
//! })?;
//! let h1 = mol.add_atom(props! {
//!     keys::ATOMIC_NUMBER => Element::H,
//!     keys::X => 0.96,
//!     keys::Y => 0.0,
//!     keys::Z => 0.0,
//! })?;
//! let h2 = mol.add_atom(props! {
//!     keys::ATOMIC_NUMBER => Element::H,
//!     keys::X => -0.24,
//!     keys::Y => 0.93,
//!     keys::Z => 0.0,
//! })?;
//! mol.add_bond(o, h1, props! {})?;
//! mol.add_bond(o, h2, props! {})?;
//!
//! // The model is chosen at runtime — the trait is object-safe, and no caller has
//! // to know which one it is holding.
//! let models: Vec<Box<dyn ChargeModel>> = vec![
//!     Box::new(BccModel::new(BccParameterSet::Bcc)),
//!     Box::new(MullikenModel),
//! ];
//! let am1 = [-0.826, 0.417, 0.409];
//! for model in &models {
//!     let q: Vec<f64> = model.assign(&mol, Some(&am1))?;
//!     assert_eq!(q.len(), mol.n_atoms());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! [`needs_equivalencing`]: ChargeModel::needs_equivalencing

mod bcc;
mod error;
mod gasteiger;
mod model;
mod mulliken;

pub use bcc::{BccModel, BccParameterSet};
pub use error::ChargeError;
pub use gasteiger::GasteigerModel;
pub use model::ChargeModel;
pub use mulliken::MullikenModel;
