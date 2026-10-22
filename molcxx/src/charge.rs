//! Partial charges for the C++ engine — the CXX face of `molrs::ff::charge`
//! (AM1-BCC corrections over engine-supplied AM1 base charges).

use molrs::core::keys;
use molrs::ff::charge::{BccParameterSet, ChargeModel};
use ndarray::Array1;

use crate::frame::{FrameRef, with_block_inserted_res};

/// `molrs::ff::charge::BccModel` across the bridge: AM1-BCC / ABCG2 bond-charge
/// corrections over AM1 base charges the engine supplies.
///
/// All BCC atom typing and correction lookup stays in molrs. The frame is the
/// engine's: `atoms.{atomic_number, x, y, z}` and `bonds.{atomi, atomj}` — the
/// model reads species from `atomic_number` only and needs no `element`
/// column (a symbol frame is converted first with `frame_fill_atomic_numbers`).
/// Both methods write the result into the frame's `atoms.charge` column and
/// return it;
/// other frame blocks and non-charge atom columns are left untouched — in
/// particular the caller's `atoms.type` column survives, because the model
/// keeps its BCC codes to itself.
///
/// # Errors
///
/// Every error here is the *caller's chemistry*, not a programmer bug: a molecule
/// with no BCC correction row (BF4⁻ — boron has none), a missing atom type, a
/// missing bond order, or a parameter-set name molrs does not know. They are
/// returned rather than panicked precisely so that cxx renders them as a
/// `rust::Error` C++ can catch. A panic here would cross an `extern "C"` shim and
/// abort the engine that asked.
///
/// There is no total-charge argument. AM1-BCC does not renormalize to a target:
/// the reference AM1-BCC algorithm ends at the increment loop; spreading the AM1
/// rounding residual over the atoms would make molrs *diverge* from the reference
/// while hiding a non-converged AM1 behind a plausible-looking answer.
pub struct BccModel(molrs::ff::charge::BccModel);

/// `BccModel::new` for the correction family named `parameter_set`, `"bcc"`
/// or `"abcg2"` (see [`BccParameterSet::from_name`]). A name molrs does not
/// know is an error, never a default.
pub(crate) fn bcc_model_new(parameter_set: &str) -> Result<Box<BccModel>, String> {
    let set = BccParameterSet::from_name(parameter_set)?;
    Ok(Box::new(BccModel(molrs::ff::charge::BccModel::new(set))))
}

impl BccModel {
    /// `BccModel::assign`: the whole model — the AM1 charges are averaged over
    /// the topological-equivalence classes, then corrected.
    pub(crate) fn assign(
        &self,
        fref: &mut FrameRef,
        am1_charges: &[f64],
    ) -> Result<Vec<f64>, String> {
        self.write_charges(fref, |mol| self.0.assign(mol, Some(am1_charges)))
    }

    /// `BccModel::correct`: the correction stage alone, for base charges that
    /// are already equivalenced.
    pub(crate) fn correct(
        &self,
        fref: &mut FrameRef,
        am1_charges: &[f64],
    ) -> Result<Vec<f64>, String> {
        self.write_charges(fref, |mol| self.0.correct(mol, am1_charges))
    }

    fn write_charges(
        &self,
        fref: &mut FrameRef,
        charges_of: impl FnOnce(
            &molrs::core::Atomistic,
        ) -> Result<Vec<f64>, molrs::ff::charge::ChargeError>,
    ) -> Result<Vec<f64>, String> {
        fref.0
            .with_mut(|frame| -> Result<Vec<f64>, String> {
                let mol = molrs::core::Atomistic::from_frame(frame).map_err(|e| e.to_string())?;
                let charges = charges_of(&mol).map_err(|e| e.to_string())?;
                with_block_inserted_res(frame, "atoms", |blk| {
                    blk.insert(
                        keys::CHARGE,
                        Array1::from_vec(charges.clone()).into_dyn(),
                        None,
                    )
                    .map_err(|e| format!("BccModel: insert charge: {e}"))
                })?;
                Ok(charges)
            })
            .map_err(|e| e.to_string())?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{frame_column_f64, frame_new, frame_set_column_f64, frame_set_column_u64};
    use molrs::core::Element;
    use molrs::props;

    /// Tetrahedral C–H offsets of the methane fixtures (Å).
    const H_OFFSETS: [[f64; 3]; 4] = [
        [0.63, 0.63, 0.63],
        [-0.63, -0.63, 0.63],
        [-0.63, 0.63, -0.63],
        [0.63, -0.63, -0.63],
    ];

    /// The BCC methane result for AM1 base charges `[-0.266, 0.066 × 4]`.
    const METHANE_BCC: [f64; 5] = [-0.1088, 0.0267, 0.0267, 0.0267, 0.0267];

    fn assert_charges(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-12,
                "{actual} != {expected}"
            );
        }
    }

    /// Methane built through molrs' graph API and handed over as its frame.
    fn methane_frame_ref() -> Box<FrameRef> {
        let mut mol = molrs::core::Atomistic::new();
        let c = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 0.0, keys::Y => 0.0, keys::Z => 0.0,
            })
            .unwrap();
        for [x, y, z] in H_OFFSETS {
            let h = mol
                .add_atom(props! {
                    keys::ATOMIC_NUMBER => Element::H,
                    keys::X => x, keys::Y => y, keys::Z => z,
                })
                .unwrap();
            mol.add_bond(c, h, props! {}).expect("add C-H");
        }
        let fref = frame_new();
        fref.0
            .with_mut(|frame| *frame = mol.to_frame().expect("methane converts"))
            .unwrap();
        fref
    }

    /// Methane exactly as Atomiverse sends it across the bridge:
    /// `atoms.{atomic_number (u64), x, y, z}` + `bonds.{atomi, atomj}` and
    /// nothing else — no `element`, no bond orders, no atom types.
    fn engine_methane_frame_ref() -> Box<FrameRef> {
        let mut fref = frame_new();
        let mut xs = vec![0.0];
        let mut ys = vec![0.0];
        let mut zs = vec![0.0];
        for [x, y, z] in H_OFFSETS {
            xs.push(x);
            ys.push(y);
            zs.push(z);
        }
        frame_set_column_u64(&mut fref, "atoms", keys::ATOMIC_NUMBER, &[6, 1, 1, 1, 1]).unwrap();
        frame_set_column_f64(&mut fref, "atoms", keys::X, &xs).unwrap();
        frame_set_column_f64(&mut fref, "atoms", keys::Y, &ys).unwrap();
        frame_set_column_f64(&mut fref, "atoms", keys::Z, &zs).unwrap();
        frame_set_column_u64(&mut fref, "bonds", keys::ATOMI, &[0, 0, 0, 0]).unwrap();
        frame_set_column_u64(&mut fref, "bonds", keys::ATOMJ, &[1, 2, 3, 4]).unwrap();
        fref
    }

    #[test]
    fn bcc_model_correct_writes_the_corrected_charges_into_the_frame() {
        let mut fref = methane_frame_ref();
        let charges = bcc_model_new("bcc")
            .expect("bcc is a parameter set")
            .correct(
                &mut fref,
                &[-0.266000, 0.066000, 0.066000, 0.066000, 0.066000],
            )
            .expect("methane is covered by the BCC table");
        assert_charges(&charges, &METHANE_BCC);
        assert_charges(
            &frame_column_f64(&fref, "atoms", keys::CHARGE).unwrap(),
            &charges,
        );
    }

    #[test]
    fn bcc_model_assign_equivalences_then_corrects() {
        let mut fref = methane_frame_ref();
        // Unequal hydrogens average to 0.066 before the correction, so the
        // result is the one `correct` gives on equivalenced input.
        let charges = bcc_model_new("bcc")
            .unwrap()
            .assign(&mut fref, &[-0.266, 0.056, 0.076, 0.066, 0.066])
            .expect("methane is covered by the BCC table");
        assert_charges(&charges, &METHANE_BCC);
    }

    /// The Atomiverse AM1-BCC path: a Z-only engine frame with bare bonds.
    #[test]
    fn bcc_model_corrects_an_engine_frame_without_element_symbols() {
        let mut fref = engine_methane_frame_ref();
        let charges = bcc_model_new("bcc")
            .unwrap()
            .correct(&mut fref, &[-0.266, 0.066, 0.066, 0.066, 0.066])
            .expect("a Z-only methane frame is enough for BCC typing");
        assert!(charges.iter().all(|q| q.is_finite()), "{charges:?}");
        assert_charges(&charges, &METHANE_BCC);
        assert_charges(
            &frame_column_f64(&fref, "atoms", keys::CHARGE).unwrap(),
            &charges,
        );
    }

    #[test]
    fn bcc_model_assigns_an_engine_frame_without_element_symbols() {
        let mut fref = engine_methane_frame_ref();
        let charges = bcc_model_new("bcc")
            .unwrap()
            .assign(&mut fref, &[-0.266, 0.056, 0.076, 0.066, 0.066])
            .expect("a Z-only methane frame is enough for BCC typing");
        assert_charges(&charges, &METHANE_BCC);
    }

    #[test]
    fn bcc_model_new_refuses_an_unknown_parameter_set() {
        assert!(bcc_model_new("am1").is_err());
    }
}
