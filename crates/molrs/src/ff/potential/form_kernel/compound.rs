//! [`CompoundTerms`]: N-body terms of any block priced by a [`CompoundForm`].

use std::sync::Arc;

use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::ir::{CategorySpec, StyleSpec};
use crate::ff::potential::flat_coords::{term_table, validate_coords};
use crate::ff::potential::form_kernel::Probe;
use crate::ff::potential::form_kernel::{CompoundForm, TermParams, resolve_terms};
use crate::ff::potential::{Accumulator, IndexedTerms, Potential};
use molrs::core::Frame;
use molrs::op::F;

/// The rows of a block, `arity` atoms each, priced by one N-body form.
///
/// Per evaluation it gathers every term's positions, makes **one** call to
/// the form, and adds `−∂E/∂x` onto the atoms.
pub struct CompoundTerms<Fm: ?Sized = dyn CompoundForm> {
    form: Arc<Fm>,
    /// One column per position of a term: `atoms[a][t]`.
    atoms: Vec<Vec<usize>>,
    params: TermParams,
}

impl<Fm: CompoundForm + ?Sized> CompoundTerms<Fm> {
    /// The kernel of `spec` over `category`'s block in `frame`, `arity`
    /// atoms per term.
    pub fn build(
        form: Arc<Fm>,
        category: &CategorySpec,
        spec: &StyleSpec,
        style: &Params,
        tp: &[(&str, &Params)],
        frame: &Frame,
    ) -> Result<Self, crate::ff::potential::CompileError> {
        let arity = category.arity.endpoints();
        if arity == 0 {
            return Err(format!(
                "{} `{}`: a compound term names at least one atom",
                spec.category, spec.name
            )
            .into());
        }
        let (atoms, params) = resolve_terms(
            spec,
            &form.inputs(),
            &category.block,
            arity,
            style,
            tp,
            frame,
        )?;
        params.require(spec, &form.inputs())?;
        params
            .style_cols(|p| form.check_style(p))
            .map_err(|r| TermParams::refused(spec, r))?;
        Ok(Self {
            form,
            atoms,
            params,
        })
    }

    /// Up to `limit` of its terms at `coords`, for the conformance check a
    /// style without registration samples gets at its first compile.
    pub(crate) fn probe(&self, coords: &[F], limit: usize) -> Probe {
        let picked: Vec<usize> = (0..self.n_terms().min(limit)).collect();
        let x = picked
            .iter()
            .flat_map(|&t| {
                self.atoms.iter().map(move |col| {
                    let a = col[t];
                    [coords[a * 3], coords[a * 3 + 1], coords[a * 3 + 2]]
                })
            })
            .collect();
        Probe {
            params: self.params.select(&picked),
            q: Vec::new(),
            x,
            arity: self.atoms.len(),
        }
    }

    fn n_terms(&self) -> usize {
        self.atoms[0].len()
    }

    fn fold(&self, coords: &[F], out: &mut [F], atoms: impl Fn(usize, usize) -> usize) -> F {
        let _ = validate_coords(coords);
        let (n, arity) = (self.n_terms(), self.atoms.len());
        if n == 0 {
            return 0.0;
        }
        let mut x = Vec::with_capacity(n * arity);
        for t in 0..n {
            for p in 0..arity {
                let a = atoms(t, p);
                x.push([coords[a * 3], coords[a * 3 + 1], coords[a * 3 + 2]]);
            }
        }
        let mut e = vec![0.0; n];
        let mut grad = vec![[0.0; 3]; n * arity];
        self.params
            .with_cols(0..n, |p| self.form.eval(&x, arity, p, &mut e, &mut grad));
        for t in 0..n {
            for p in 0..arity {
                let a = atoms(t, p);
                let g = grad[t * arity + p];
                for d in 0..3 {
                    out[a * 3 + d] -= g[d];
                }
            }
        }
        e.iter().sum()
    }
}

impl<Fm: CompoundForm + ?Sized> Potential for CompoundTerms<Fm> {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let energy = self.fold(coords, acc.forces_without_virial(), |t, p| self.atoms[p][t]);
        acc.add_energy(energy);
    }
}

impl<Fm: CompoundForm + ?Sized> IndexedTerms for CompoundTerms<Fm> {
    fn terms(&self) -> Array2<u32> {
        let cols: Vec<&[usize]> = self.atoms.iter().map(Vec::as_slice).collect();
        term_table(&cols)
    }

    fn evaluate_terms(&self, coords: &[F], terms: ArrayView2<'_, u32>, acc: &mut Accumulator<'_>) {
        debug_assert_eq!(
            terms.nrows(),
            self.n_terms(),
            "the row set is the force field's; only the atoms a row names may be rebound"
        );
        let energy = self.fold(coords, acc.forces_without_virial(), |t, p| {
            terms[[t, p]] as usize
        });
        acc.add_energy(energy);
    }
}
