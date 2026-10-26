//! Reads of a style's parameters outside a form, refused typed: the form
//! kernels' `cutoff`s and 1-4 weight, PME's reciprocal-space constructor,
//! and the Lennard-Jones rows of the 1-4 exceptions.
//!
//! A parameter one needs and does not find is [`IrError::MissingParam`]
//! naming the style, the type and the parameter — the refusal a form kernel
//! or an expression makes of the same row — and never a message. The values
//! arrive gathered ([`StyleSpec::gather`](crate::ff::ir::StyleSpec::gather)):
//! a declared default is already in place, so none is stated here.

use crate::ff::ir::IrError;
use crate::ff::ir::Params;
use crate::ff::ir::StyleSpec;
use molrs::op::F;

/// `style` lacks `param`: in the type row `type_`, or (`type_` empty) among
/// its style params or its per-instance columns.
pub(crate) fn missing(style: &str, type_: &str, param: &str) -> IrError {
    IrError::MissingParam {
        style: style.to_owned(),
        type_: type_.to_owned(),
        param: param.to_owned(),
    }
}

/// `param` of `style` (in the type row `type_`, or the style when empty)
/// is out of its domain, for `reason`.
pub(crate) fn bad(style: &str, type_: &str, param: &str, reason: impl Into<String>) -> IrError {
    IrError::BadValue {
        style: style.to_owned(),
        type_: type_.to_owned(),
        param: param.to_owned(),
        reason: reason.into(),
    }
}

/// The number `key` of the type row `label` of `style`.
pub(crate) fn type_num(style: &str, label: &str, row: &Params, key: &str) -> Result<F, IrError> {
    row.get(key)
        .and_then(|v| v.as_num())
        .ok_or_else(|| missing(style, label, key))
}

/// The style param `key` of `style`.
pub(crate) fn style_num(style: &str, params: &Params, key: &str) -> Result<F, IrError> {
    params
        .get(key)
        .and_then(|v| v.as_num())
        .ok_or_else(|| missing(style, "", key))
}

/// The text style param `key` of `style`.
pub(crate) fn style_text<'p>(
    style: &str,
    params: &'p Params,
    key: &str,
) -> Result<&'p str, IrError> {
    params
        .get(key)
        .and_then(|v| v.as_text())
        .ok_or_else(|| missing(style, "", key))
}

/// The `cutoff` of a pair style, as LAMMPS truncates every pair style: a
/// pair prices only at `r < cutoff`, at both compile doors.
///
/// A style whose spec declares a `cutoff` states it: positive, and `∞` only
/// as stated (a field priced with no cutoff, OpenMM's `NoCutoff`). Where a
/// pair style stops is the force field's decision — a typifier declares its
/// convention, a reader the file's — never a default. A style whose spec
/// declares none bounds itself (`thole`, `mmff_vdw`): ∞.
pub(crate) fn pair_cutoff(spec: &StyleSpec, params: &Params) -> Result<F, IrError> {
    if spec.style_param("cutoff").is_none() {
        return Ok(F::INFINITY);
    }
    let style = spec.name.as_ref();
    let Some(cutoff) = params.get("cutoff").and_then(|v| v.as_num()) else {
        return Err(bad(
            style,
            "",
            "cutoff",
            "is not stated: where a pair style stops is the force field's to declare \
             (`inf` prices every pair)",
        ));
    };
    if cutoff.is_nan() || cutoff <= 0.0 {
        return Err(bad(
            style,
            "",
            "cutoff",
            format!("= {cutoff}: a pair cutoff is positive"),
        ));
    }
    Ok(cutoff)
}

/// The `cutoff` of a pair style evaluated over a neighbour search: stated,
/// finite and positive — a periodic neighbour sum is not finite without one,
/// and the ∞ an untruncated style's spec defaults to is no neighbour cutoff.
pub(crate) fn neighbour_cutoff(style: &str, params: &Params) -> Result<F, IrError> {
    let cutoff = style_num(style, params, "cutoff")?;
    if !(cutoff.is_finite() && cutoff > 0.0) {
        return Err(bad(
            style,
            "",
            "cutoff",
            format!("= {cutoff}: a neighbour-driven pair style needs a finite, positive cutoff"),
        ));
    }
    Ok(cutoff)
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::IrError;
    use crate::ff::ir::Params;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::{CompileError, Potentials};
    use molrs::op::F;

    const XYZ: [F; 12] = [
        1.2, -0.4, 0.3, 0.0, 0.0, 0.0, -0.2, 1.5, 0.1, 0.9, 2.1, -0.8,
    ];

    fn one(
        category: &str,
        style: &str,
        atoms: &[usize],
        row: Params,
    ) -> Result<Potentials, CompileError> {
        ExplicitTerms::new(category, style)
            .term(atoms, row)
            .compile()
    }

    fn missing(err: CompileError) -> (String, String) {
        match err.ir() {
            Some(IrError::MissingParam { style, param, .. }) => (style.clone(), param.clone()),
            _ => panic!("not MissingParam: {err:?}"),
        }
    }

    fn bad(err: CompileError) -> (String, String) {
        match err.ir() {
            Some(IrError::BadValue { style, param, .. }) => (style.clone(), param.clone()),
            _ => panic!("not BadValue: {err:?}"),
        }
    }

    /// Every built-in kernel refuses a parameter it needs as `MissingParam`
    /// naming the style and the parameter, not a message.
    #[test]
    fn a_built_in_kernel_refuses_a_missing_parameter_typed() {
        let p = |pairs: &[(&str, f64)]| pairs.iter().copied().collect::<Params>();
        for (category, style, atoms, row, param) in [
            ("bond", "harmonic", &[0, 1][..], p(&[("k", 1.0)]), "r0"),
            (
                "bond",
                "morse",
                &[0, 1],
                p(&[("d0", 1.0), ("r0", 1.0)]),
                "alpha",
            ),
            (
                "bond",
                "class2",
                &[0, 1],
                p(&[("r0", 1.0), ("k2", 1.0)]),
                "k3",
            ),
            (
                "angle",
                "harmonic",
                &[0, 1, 2],
                p(&[("theta0", 100.0)]),
                "k",
            ),
            (
                "angle",
                "charmm",
                &[0, 1, 2],
                p(&[("k", 1.0), ("theta0", 100.0)]),
                "k_ub",
            ),
            (
                "dihedral",
                "charmm",
                &[0, 1, 2, 3],
                p(&[("k", 1.0)]),
                "periodicity",
            ),
            (
                "dihedral",
                "periodic",
                &[0, 1, 2, 3],
                p(&[("k1", 1.0)]),
                "periodicity1",
            ),
            (
                "dihedral",
                "harmonic",
                &[0, 1, 2, 3],
                p(&[("k", 1.0), ("sign", 1.0)]),
                "periodicity",
            ),
            (
                "dihedral",
                "nharmonic",
                &[0, 1, 2, 3],
                p(&[("a2", 1.0)]),
                "a1",
            ),
            ("improper", "cvff", &[0, 1, 2, 3], p(&[("k", 1.0)]), "sign"),
            (
                "improper",
                "harmonic",
                &[0, 1, 2, 3],
                p(&[("chi0", 1.0)]),
                "k",
            ),
            (
                "improper",
                "periodic",
                &[0, 1, 2, 3],
                p(&[("k", 1.0)]),
                "periodicity",
            ),
        ] {
            let err = one(category, style, atoms, row).map(|_| ()).unwrap_err();
            assert_eq!(
                missing(err),
                (style.to_owned(), param.to_owned()),
                "{category} {style}"
            );
        }
        // A per-instance style: the column its typifier bakes.
        let err = ExplicitTerms::new("bond", "mmff_bond")
            .atoms([vec![0, 1]])
            .compile()
            .map(|_| ())
            .unwrap_err();
        assert_eq!(missing(err).1, "kb");
        // A style parameter.
        let err = ExplicitTerms::new("pair", "lj/charmm")
            .style_params(Params::from_iter([("cutoff", 10.0)]))
            .term(&[0, 1], p(&[("epsilon", 0.1), ("sigma", 3.0)]))
            .compile()
            .map(|_| ())
            .unwrap_err();
        assert_eq!(missing(err), ("lj/charmm".to_owned(), "inner".to_owned()));
    }

    /// A value of the wrong kind or outside its choices is `BadValue`.
    #[test]
    fn a_value_of_the_wrong_kind_is_bad_value() {
        let mut row = Params::from_iter([("k", 1.0)]);
        row.set("r0", "1.5");
        let err = one("bond", "harmonic", &[0, 1], row)
            .map(|_| ())
            .unwrap_err();
        assert_eq!(bad(err), ("harmonic".to_owned(), "r0".to_owned()));

        let mut style = Params::from_iter([("cutoff", 10.0)]);
        style.set("mixing", "lorentz");
        let err = ExplicitTerms::new("pair", "lj/cut")
            .style_params(style)
            .term(
                &[0, 1],
                Params::from_iter([("epsilon", 0.1), ("sigma", 3.0)]),
            )
            .compile()
            .map(|_| ())
            .unwrap_err();
        assert_eq!(bad(err), ("lj/cut".to_owned(), "mixing".to_owned()));

        // ∞, stated to price every pair, is no neighbour cutoff.
        let err = neighbour_cutoff_of(Params::from_iter([("cutoff", F::INFINITY)])).unwrap_err();
        assert!(err.to_string().contains("finite"), "{err}");
    }

    /// A pair style that declares a cutoff states it: unstated is refused by
    /// name, with what to write, and `inf` prices every pair. A style that
    /// declares none (`thole`) bounds itself.
    #[test]
    fn a_declared_cutoff_is_stated_and_inf_prices_every_pair() {
        let lj = |style: Params| {
            ExplicitTerms::new("pair", "lj/cut")
                .style_params(style)
                .term(
                    &[0, 1],
                    Params::from_iter([("epsilon", 0.2), ("sigma", 1.1)]),
                )
                .compile()
        };
        let err = lj(Params::new()).map(|_| ()).unwrap_err();
        assert_eq!(bad(err), ("lj/cut".to_owned(), "cutoff".to_owned()));
        let err = lj(Params::new()).map(|_| ()).unwrap_err().to_string();
        assert!(err.contains("`inf`"), "{err}");
        let r = XYZ[0].hypot(XYZ[1]).hypot(XYZ[2]); // atom 1 sits at the origin
        let far = lj(Params::from_iter([("cutoff", F::INFINITY)])).unwrap();
        assert!(far.energy_at(&XYZ) != 0.0, "every pair is priced under inf");
        let near = lj(Params::from_iter([("cutoff", r * 0.5)])).unwrap();
        assert_eq!(near.energy_at(&XYZ), 0.0, "a pair past the cutoff is not");
        let spec = crate::ff::ir::builtin_style("pair", "thole").unwrap();
        assert_eq!(
            super::pair_cutoff(spec, &Params::new()).unwrap(),
            F::INFINITY
        );
    }

    fn neighbour_cutoff_of(style: Params) -> Result<F, IrError> {
        let gathered = crate::ff::style_registry::GlobalRegistry
            .with(|r| r.style("pair", "lj/cut").unwrap().0.gather(&style, &[]))?;
        super::neighbour_cutoff("lj/cut", &gathered.0)
    }

    /// A declared default prices an absent parameter exactly as stating it
    /// does: the built-in kernels state no default of their own.
    #[test]
    fn a_declared_default_prices_as_the_stated_value() {
        let energy = |category: &str, style: &str, atoms: &[usize], row: Params| {
            one(category, style, atoms, row).unwrap().energy_at(&XYZ)
        };
        let p = |pairs: &[(&str, f64)]| pairs.iter().copied().collect::<Params>();
        for (category, style, atoms, bare, stated) in [
            (
                "dihedral",
                "charmm",
                &[0, 1, 2, 3][..],
                p(&[("k", 1.3), ("periodicity", 3.0)]),
                p(&[("k", 1.3), ("periodicity", 3.0), ("phase", 0.0), ("w", 0.0)]),
            ),
            (
                "dihedral",
                "periodic",
                &[0, 1, 2, 3],
                p(&[
                    ("k1", 1.3),
                    ("periodicity1", 1.0),
                    ("k2", 0.4),
                    ("periodicity2", 2.0),
                ]),
                p(&[
                    ("k1", 1.3),
                    ("periodicity1", 1.0),
                    ("phase1", 0.0),
                    ("k2", 0.4),
                    ("periodicity2", 2.0),
                    ("phase2", 0.0),
                ]),
            ),
            (
                "dihedral",
                "opls",
                &[0, 1, 2, 3],
                p(&[("k1", 1.3), ("k3", 0.2)]),
                p(&[("k1", 1.3), ("k2", 0.0), ("k3", 0.2), ("k4", 0.0)]),
            ),
            (
                "improper",
                "harmonic",
                &[0, 1, 2, 3],
                p(&[("k", 2.0)]),
                p(&[("k", 2.0), ("chi0", 0.0)]),
            ),
        ] {
            let e = energy(category, style, atoms, bare);
            assert!(e != 0.0, "{category} {style}");
            assert_eq!(
                e,
                energy(category, style, atoms, stated),
                "{category} {style}"
            );
        }
        // A style default on the pair kernels: lj/cut's n, m, shift.
        let lj = |mut style: Params| {
            style.set("cutoff", F::INFINITY);
            ExplicitTerms::new("pair", "lj/cut")
                .style_params(style)
                .term(
                    &[0, 1],
                    Params::from_iter([("epsilon", 0.2), ("sigma", 1.1)]),
                )
                .compile()
                .unwrap()
                .energy_at(&XYZ)
        };
        assert_eq!(
            lj(Params::new()),
            lj(Params::from_iter([("n", 12.0), ("m", 6.0), ("shift", 0.0)]))
        );
    }
}
