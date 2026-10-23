//! One style's kernel over **explicit instances**: the terms and their
//! parameters given by hand, no typifier and no type table.

use crate::ff::style_registry;
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::keys::ENDPOINTS;
use molrs::core::schema::block_names::{ATOMS, PAIRS};
use molrs::op::{F, Idx};
use ndarray::Array1;

use crate::ff::compile::PotentialCompiler;
use crate::ff::forcefield::{DefError, ForceField};
use crate::ff::ir::Params;
use crate::ff::ir::{CategorySpec, IrError};
use crate::ff::potential::{CompileError, Potentials};
use crate::ff::style_registry::{Registry, RegistryRef};
use std::sync::Arc;

/// One style's terms, each its atoms and its own parameter row (as stored:
/// the force-field IR's units, angle values in degrees).
///
/// A pair style's terms are atom pairs, each priced with its row as the
/// pair's cross row; its per-atom data (`charge`, read by `coul/cut` and by
/// expressions through `q1`, `q2`) is [`charges`](Self::charges). A style
/// that reads its numbers per instance (`coul/cut`, MMFF) may have terms
/// without rows ([`atoms`](Self::atoms)).
///
/// [`ExplicitTerms::compile`] is the generic way to build the kernel of *any*
/// registered `(category, style)` — a built-in, a custom style priced by its
/// expression or a Python callable, a style of a custom category — from
/// atom indices and one parameter row per term. It goes through
/// [`PotentialCompiler`] (one type per term, labelled by its row), so a hand
/// assembled kernel is priced by exactly the code a compiled force field is,
/// with the same registry, fallback and refusals.
///
/// ```
/// use molrs::ff::ir::Params;
/// use molrs::ff::compile::ExplicitTerms;
/// use molrs::ff::potential::{Accumulator, Potential};
///
/// // LAMMPS `bond_style harmonic`, one bond: k (r − r0)².
/// let pots = ExplicitTerms::new("bond", "harmonic")
///     .term(&[0, 1], Params::from_iter([("k", 300.0), ("r0", 1.5)]))
///     .compile()
///     .unwrap();
/// let x = [0.0, 0.0, 0.0, 1.6, 0.0, 0.0];
/// let mut acc = Accumulator::new(x.len());
/// pots.evaluate(&x, &mut acc);
/// assert!((acc.energy() - 3.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct ExplicitTerms {
    category: String,
    style: String,
    style_params: Params,
    atoms: Vec<Vec<usize>>,
    rows: Vec<Params>,
    charges: Option<Vec<F>>,
    registry: RegistryRef,
}

impl ExplicitTerms {
    /// No terms yet, of `style` in `category`.
    pub fn new(category: &str, style: &str) -> Self {
        Self {
            category: category.to_owned(),
            style: style.to_owned(),
            style_params: Params::new(),
            atoms: Vec::new(),
            rows: Vec::new(),
            charges: None,
            registry: RegistryRef::Global,
        }
    }

    /// Build against `registry` instead of the process-wide one — a test's
    /// own styles, seen by nothing else.
    pub fn with_registry(mut self, registry: Arc<Registry>) -> Self {
        self.registry = RegistryRef::Own(registry);
        self
    }

    /// The style-level parameters (`cutoff`, `mixing`, `coulomb`, an
    /// unregistered style's `expression`, …).
    pub fn style_params(mut self, params: Params) -> Self {
        self.style_params = params;
        self
    }

    /// One term: its atoms and its parameter row.
    pub fn term(mut self, atoms: &[usize], row: Params) -> Self {
        self.atoms.push(atoms.to_vec());
        self.rows.push(row);
        self
    }

    /// Terms without parameter rows, for a style whose numbers are per
    /// instance (`coul/cut`, MMFF). Not mixed with [`term`](Self::term).
    pub fn atoms(mut self, atoms: impl IntoIterator<Item = Vec<usize>>) -> Self {
        self.atoms.extend(atoms);
        self
    }

    /// The per-atom charges (`atoms.charge`).
    pub fn charges(mut self, charges: Vec<F>) -> Self {
        self.charges = Some(charges);
        self
    }

    /// The kernel, against the registry of [`with_registry`](Self::with_registry)
    /// (the process-wide one by default).
    pub fn compile(&self) -> Result<Potentials, CompileError> {
        let lookup = |r: &Registry| r.category(&self.category).cloned();
        let category = match &self.registry {
            RegistryRef::Global => style_registry::with_global_registry(lookup),
            RegistryRef::Own(r) => lookup(r),
        };
        self.build(category)
    }

    fn build(&self, category: Option<CategorySpec>) -> Result<Potentials, CompileError> {
        let spec = category.ok_or_else(|| IrError::UnknownCategory {
            category: self.category.clone(),
        })?;
        let arity = spec.arity.endpoints();
        if let Some(bad) = self.atoms.iter().find(|a| a.len() != arity) {
            return Err(IrError::Arity {
                category: self.category.clone(),
                arity: bad.len(),
            }
            .into());
        }
        if !self.rows.is_empty() && self.rows.len() != self.atoms.len() {
            return Err(format!(
                "{} `{}`: {} parameter rows for {} terms (one per term, or none)",
                self.category,
                self.style,
                self.rows.len(),
                self.atoms.len()
            )
            .into());
        }
        let pair = spec.is_pair_driven();
        let n_atoms = self
            .atoms
            .iter()
            .flatten()
            .map(|&a| a + 1)
            .max()
            .unwrap_or(0)
            .max(self.charges.as_ref().map_or(0, Vec::len));
        let atom_type = |a: usize| format!("a{a}");

        let mut ff = ForceField::new("instances");
        ff.registry = self.registry.clone();
        // Each explicit term is priced at full strength: the `pairs` rows
        // below carry no `is_14`, and no 1-2 / 1-3 class is excluded from
        // terms the caller listed by hand.
        ff.set_special_bonds(crate::ff::ir::SpecialBonds {
            lj: [1.0; 3],
            coul: [1.0; 3],
        });
        let style = ff
            .def_style(&self.category, &self.style, self.style_params.clone())
            .map_err(def_err)?;
        for (t, row) in self.rows.iter().enumerate() {
            // A pair term is the cross row of its two atoms' own types; any
            // other term is its own type (the endpoints only name it).
            let endpoints: Vec<String> = if pair {
                self.atoms[t].iter().map(|&a| atom_type(a)).collect()
            } else {
                vec!["X".to_owned(); arity]
            };
            let endpoints: Vec<&str> = endpoints.iter().map(String::as_str).collect();
            style
                .def_type(&t.to_string(), &endpoints, row.clone())
                .map_err(def_err)?;
        }

        let mut frame = Frame::new();
        let mut terms = Block::new();
        for (col, name) in ENDPOINTS.iter().take(arity).enumerate() {
            let column: Vec<Idx> = self.atoms.iter().map(|a| a[col] as Idx).collect();
            terms
                .insert(*name, Array1::from_vec(column).into_dyn(), None)
                .map_err(|e| e.to_string())?;
        }
        if !pair && !self.rows.is_empty() {
            let labels: Vec<String> = (0..self.atoms.len()).map(|t| t.to_string()).collect();
            terms
                .insert("type", Array1::from_vec(labels).into_dyn(), None)
                .map_err(|e| e.to_string())?;
        }
        frame.insert(if pair { PAIRS } else { spec.block.as_ref() }, terms);
        if pair || self.charges.is_some() {
            let mut atoms = Block::new();
            let types: Vec<String> = (0..n_atoms).map(atom_type).collect();
            atoms
                .insert("type", Array1::from_vec(types).into_dyn(), None)
                .map_err(|e| e.to_string())?;
            if let Some(q) = &self.charges {
                let mut q = q.clone();
                q.resize(n_atoms, 0.0);
                atoms
                    .insert("charge", Array1::from_vec(q).into_dyn(), None)
                    .map_err(|e| e.to_string())?;
            }
            frame.insert(ATOMS, atoms);
        }

        PotentialCompiler::new(&ff).compile(&frame)
    }
}

fn def_err(e: DefError) -> CompileError {
    match e.ir() {
        Some(refusal) => refusal.into(),
        None => e.to_string().into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::ir::{ParamSpec, StyleSpec};
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::style_registry::Kernel;

    const XYZ: [F; 12] = [
        1.2, -0.4, 0.3, 0.0, 0.0, 0.0, -0.2, 1.5, 0.1, 0.9, 2.1, -0.8,
    ];

    fn p(i: usize) -> [F; 3] {
        [XYZ[3 * i], XYZ[3 * i + 1], XYZ[3 * i + 2]]
    }

    fn dist(a: usize, b: usize) -> F {
        let (a, b) = (p(a), p(b));
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    fn angle(a: usize, b: usize, c: usize) -> F {
        let (a, b, c) = (p(a), p(b), p(c));
        let u = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let v = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
        let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        (dot / (nu * nv)).acos()
    }

    fn close(a: F, b: F) {
        assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn a_bonded_style_prices_one_row_per_term() {
        let pots = ExplicitTerms::new("bond", "harmonic")
            .term(&[0, 1], Params::from_iter([("k", 300.0), ("r0", 1.4)]))
            .term(&[1, 2], Params::from_iter([("k", 200.0), ("r0", 1.5)]))
            .compile()
            .unwrap();
        let want = 300.0 * (dist(0, 1) - 1.4).powi(2) + 200.0 * (dist(1, 2) - 1.5).powi(2);
        close(pots.energy_at(&XYZ), want);
    }

    #[test]
    fn angle_values_are_degrees_as_stored() {
        let pots = ExplicitTerms::new("angle", "harmonic")
            .term(
                &[0, 1, 2],
                Params::from_iter([("k", 50.0), ("theta0", 109.5)]),
            )
            .compile()
            .unwrap();
        let want = 50.0 * (angle(0, 1, 2) - 109.5_f64.to_radians()).powi(2);
        close(pots.energy_at(&XYZ), want);
    }

    #[test]
    fn a_pair_term_is_priced_with_its_own_row_and_charges_per_atom() {
        let lj = ExplicitTerms::new("pair", "lj/cut")
            .term(
                &[0, 3],
                Params::from_iter([("epsilon", 0.2), ("sigma", 3.1)]),
            )
            .compile()
            .unwrap();
        let s = 3.1 / dist(0, 3);
        close(lj.energy_at(&XYZ), 4.0 * 0.2 * (s.powi(12) - s.powi(6)));

        let coul = ExplicitTerms::new("pair", "coul/cut")
            .style_params(Params::from_iter([
                ("coulomb", 332.06371),
                ("dielectric", 1.0),
            ]))
            .atoms([vec![0, 3]])
            .charges(vec![0.3, 0.0, 0.0, -0.4])
            .compile()
            .unwrap();
        close(coul.energy_at(&XYZ), 332.06371 * 0.3 * -0.4 / dist(0, 3));
    }

    #[test]
    fn a_custom_style_in_a_private_registry_is_built_the_same_way() {
        let mut reg = Registry::builtin();
        reg.register_style(
            StyleSpec::new("bond", "quartic")
                .params(vec![
                    ParamSpec::new("k", "E/L^4".parse().unwrap()),
                    ParamSpec::new("r0", "L".parse().unwrap()),
                ])
                .expression("k*(r-r0)^4"),
            None::<Kernel>,
        )
        .unwrap();
        let terms = ExplicitTerms::new("bond", "quartic")
            .term(&[0, 1], Params::from_iter([("k", 2.0), ("r0", 1.0)]));
        close(
            terms
                .clone()
                .with_registry(Arc::new(reg))
                .compile()
                .unwrap()
                .energy_at(&XYZ),
            2.0 * (dist(0, 1) - 1.0).powi(4),
        );
        // The process-wide registry has no such style.
        let err = terms.compile().unwrap_err();
        assert!(matches!(err.ir(), Some(IrError::NoKernel { .. })), "{err}");
    }

    #[test]
    fn refusals_are_typed() {
        let err = ExplicitTerms::new("nope", "x").compile().unwrap_err();
        assert!(matches!(err.ir(), Some(IrError::UnknownCategory { .. })));
        let err = ExplicitTerms::new("bond", "harmonic")
            .term(&[0, 1, 2], Params::from_iter([("k", 1.0), ("r0", 1.0)]))
            .compile()
            .unwrap_err();
        assert!(matches!(err.ir(), Some(IrError::Arity { arity: 3, .. })));
    }
}
