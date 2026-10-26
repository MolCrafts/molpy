//! [`ScalarBonded`]: a bonded category's terms priced by a [`ScalarForm`].

use std::sync::Arc;

use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::ir::{CategorySpec, Coordinate, StyleSpec};
use crate::ff::potential::flat_coords::{
    accumulate_angle_forces, accumulate_dihedral_forces, compute_angle, compute_dihedral,
    term_table, validate_coords,
};
use crate::ff::potential::form_kernel::Probe;
use crate::ff::potential::form_kernel::{ScalarForm, TermParams, resolve_terms};
use crate::ff::potential::{Accumulator, IndexedTerms, Potential};
use molrs::core::Frame;
use molrs::op::F;

/// The rows of a bonded category's block, each priced by one form of one
/// coordinate.
///
/// Per evaluation it computes every term's coordinate, makes **one** call to
/// the form, and projects each `dE/dq` onto the term's atoms with the chain
/// rule over the flat coordinate array: along the bond for
/// `r`, `accumulate_angle_forces` for `theta`, `accumulate_dihedral_forces`
/// for `phi`.
pub struct ScalarBonded<Fm: ?Sized = dyn ScalarForm> {
    form: Arc<Fm>,
    coordinate: Coordinate,
    /// One column per position of a term: `atoms[a][t]`.
    atoms: Vec<Vec<usize>>,
    params: TermParams,
}

impl<Fm: ScalarForm + ?Sized> ScalarBonded<Fm> {
    /// The kernel of `spec` over `category`'s block in `frame`, its terms'
    /// parameters resolved from `style` and `tp` (see
    /// [`generic`](crate::ff::potential::form_kernel)'s resolution order).
    pub fn build(
        form: Arc<Fm>,
        category: &CategorySpec,
        spec: &StyleSpec,
        style: &Params,
        tp: &[(&str, &Params)],
        frame: &Frame,
    ) -> Result<Self, crate::ff::potential::CompileError> {
        let coordinate = category.coordinate;
        let arity = coordinate.atoms().ok_or_else(|| {
            format!(
                "{} `{}`: a scalar form needs a scalar coordinate, the category's is {coordinate:?}",
                spec.category, spec.name
            )
        })?;
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
            coordinate,
            atoms,
            params,
        })
    }

    /// The coordinate of the term whose atoms are `at`.
    fn coordinate_of(&self, coords: &[F], at: [usize; 4]) -> F {
        let [i, j, k, l] = at;
        match self.coordinate {
            Coordinate::Distance => {
                let d = [
                    coords[j * 3] - coords[i * 3],
                    coords[j * 3 + 1] - coords[i * 3 + 1],
                    coords[j * 3 + 2] - coords[i * 3 + 2],
                ];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            }
            Coordinate::Angle => compute_angle(coords, i, j, k),
            Coordinate::Dihedral | Coordinate::Improper => compute_dihedral(coords, i, j, k, l),
            Coordinate::None | Coordinate::Compound => unreachable!("checked in build"),
        }
    }

    fn atoms_of(&self, t: usize) -> [usize; 4] {
        let mut a = [0; 4];
        for (p, slot) in a.iter_mut().enumerate().take(self.atoms.len()) {
            *slot = self.atoms[p][t];
        }
        a
    }

    /// Up to `limit` of its terms at `coords`, for the conformance check a
    /// style without registration samples gets at its first compile.
    pub(crate) fn probe(&self, coords: &[F], limit: usize) -> Probe {
        let picked: Vec<usize> = (0..self.n_terms().min(limit)).collect();
        Probe {
            params: self.params.select(&picked),
            q: picked
                .iter()
                .map(|&t| self.coordinate_of(coords, self.atoms_of(t)))
                .collect(),
            x: Vec::new(),
            arity: self.atoms.len(),
        }
    }

    fn n_terms(&self) -> usize {
        self.atoms[0].len()
    }

    /// The physics, once: which atoms term `t` names is the only thing that
    /// differs between the two entry points.
    fn fold(&self, coords: &[F], out: &mut [F], atoms: impl Fn(usize, usize) -> usize) -> F {
        let _ = validate_coords(coords);
        let n = self.n_terms();
        if n == 0 {
            return 0.0;
        }
        let at = |t: usize| -> [usize; 4] {
            let mut a = [0; 4];
            for (p, slot) in a.iter_mut().enumerate().take(self.atoms.len()) {
                *slot = atoms(t, p);
            }
            a
        };
        let q: Vec<F> = (0..n).map(|t| self.coordinate_of(coords, at(t))).collect();
        let mut e = vec![0.0; n];
        let mut de = vec![0.0; n];
        self.params
            .with_cols(0..n, |p| self.form.eval(&q, p, &mut e, &mut de));
        for t in 0..n {
            let [i, j, k, l] = at(t);
            match self.coordinate {
                Coordinate::Distance => {
                    let r = q[t];
                    if r < 1e-12 {
                        continue;
                    }
                    let factor = -de[t] / r;
                    for d in 0..3 {
                        let f = factor * (coords[j * 3 + d] - coords[i * 3 + d]);
                        out[j * 3 + d] += f;
                        out[i * 3 + d] -= f;
                    }
                }
                Coordinate::Angle => accumulate_angle_forces(coords, i, j, k, de[t], out),
                Coordinate::Dihedral | Coordinate::Improper => {
                    accumulate_dihedral_forces(coords, i, j, k, l, de[t], out)
                }
                Coordinate::None | Coordinate::Compound => unreachable!("checked in build"),
            }
        }
        e.iter().sum()
    }
}

impl<Fm: ScalarForm + ?Sized> Potential for ScalarBonded<Fm> {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let energy = self.fold(coords, acc.forces_without_virial(), |t, p| self.atoms[p][t]);
        acc.add_energy(energy);
    }
}

impl<Fm: ScalarForm + ?Sized> IndexedTerms for ScalarBonded<Fm> {
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
