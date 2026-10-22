//! Form conversion: a force field rewritten between the styles of one form
//! family (`ff-ir-01` P3, on the protocol of `ff-ir-02` §9), through the
//! [`FormCodec`](crate::ff::ir::FormCodec)s the style registry holds.
//!
//! A **form family** is a set of styles whose energies are one function
//! space parametrised differently — the torsions (Fourier series in φ), the
//! harmonic bonds and angles, the Lennard-Jones pairs. Each member registers a
//! [`FormCodec`](crate::ff::ir::FormCodec) beside its kernel ([`Registry::register_form`]): its family,
//! and two exact maps between its own parameters and the family's
//! **canonical** style's:
//!
//! * `embed` — this style → canonical (exact: the same energy, constant
//!   included), refusing a row the canonical style cannot hold;
//! * `project` — canonical → this style, exact on its image, refusing with a
//!   [`FormRefusal`] that names the condition otherwise (a Fourier series with
//!   `bₙ ≠ 0` has no RB form);
//! * optionally `seed` — canonical → the nearest member of this style's
//!   image by a rule of thumb, where [`fit`](ForceField::fit_form) starts.
//!
//! Three operations on a [`ForceField`] use them, one job each:
//!
//! * [`ForceField::canonical`] maps every style of a family **in the
//!   canonical style's category** onto the canonical style (styles of the
//!   family in another category — impropers in the torsion family — stay);
//! * [`ForceField::to_form`] converts one category's styles of a family to
//!   one style of it, through the canonical parameters, exactly or not at all
//!   ([`IrError::OutOfImage`] names the type and the condition);
//! * [`ForceField::fit_form`] is the projection that always answers: least
//!   squares under a declared [`FitMetric`], using nothing but each style's
//!   energy `E(q)` from the registry's kernels — so it works for an
//!   expression or a Python style as for a native one — and returns the
//!   [`FitResidual`] beside the parameters.
//!

mod fit;
#[cfg(test)]
mod tests;

pub use fit::{FitMetric, FitResidual, TypeResidual};

use crate::ff::forcefield::{DefError, ForceField, Style};
use crate::ff::ir::form::declared;
use crate::ff::ir::{FormRefusal, IrError, Params, StyleSpec, TypeParams};
use crate::ff::style_registry::Registry;

// ── the operations ──────────────────────────────────────────────────────────

/// `"<category> <style>"`, as messages name a style.
fn named(category: &str, style: &str) -> String {
    format!("{category} {style}")
}

/// The output row of a conversion: `converted`, plus every key of the source
/// row that is no parameter of the source style (`id`, `desc`, `smarts`, …),
/// carried as it is.
fn carry_annotations(source: &Params, spec: Option<&StyleSpec>, mut converted: Params) -> Params {
    let keep = |key: &str| spec.is_some_and(|s| !declared(s, key));
    for (key, value) in source
        .iter()
        .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
    {
        if keep(key) && converted.get(key).and_then(|v| v.as_num()).is_none() {
            converted.set(key, value);
        }
    }
    for (key, value) in source
        .iter()
        .filter_map(|(k, v)| v.as_text().map(|s| (k, s)))
    {
        if keep(key) && converted.get(key).and_then(|v| v.as_text()).is_none() {
            converted.set(key, value);
        }
    }
    for (key, value) in source
        .iter()
        .filter_map(|(k, v)| v.as_array().map(|a| (k, a)))
    {
        if keep(key) && converted.get(key).and_then(|v| v.as_array()).is_none() {
            converted.set(key, value.clone());
        }
    }
    converted
}

/// One converted row: its name, endpoints and parameters in the target style.
pub(crate) struct Converted {
    pub name: String,
    pub endpoints: Vec<String>,
    pub params: TypeParams,
}

/// Every row of `style` converted by `convert`, annotations carried.
pub(crate) fn convert_rows(
    registry: &Registry,
    style: &Style,
    mut convert: impl FnMut(&str, TypeParams) -> Result<TypeParams, IrError>,
) -> Result<Vec<Converted>, IrError> {
    let spec = registry
        .style(style.category(), style.name())
        .map(|(spec, _)| spec);
    style
        .type_rows()
        .into_iter()
        .map(|(name, endpoints, row)| {
            let mut out = convert(name, TypeParams::new(style.params().clone(), row.clone()))?;
            out.row = carry_annotations(row, spec, out.row);
            Ok(Converted {
                name: name.to_owned(),
                endpoints: endpoints.into_iter().map(str::to_owned).collect(),
                params: out,
            })
        })
        .collect()
}

/// `ff` with the styles `sources` of `category` replaced by `target`, holding
/// `rows` (the sources' rows converted) after the target's own rows when it
/// is not among the sources. The target takes the place of the first of
/// them; every other style keeps its place and its rows.
pub(crate) fn rewrite(
    ff: &ForceField,
    family: &str,
    category: &str,
    target: &str,
    sources: &[String],
    rows: Vec<Converted>,
) -> Result<ForceField, IrError> {
    let conflict = |reason: String| IrError::FormConflict {
        family: family.to_owned(),
        reason,
    };
    // One set of style parameters for the target.
    let existing = ff
        .get_style(category, target)
        .filter(|_| !sources.iter().any(|s| s == target));
    let mut style_params: Option<(String, Params)> =
        existing.map(|s| (named(category, target), s.params().clone()));
    for row in &rows {
        match &style_params {
            Some((from, p)) if *p != row.params.style => {
                return Err(conflict(format!(
                    "`{}` needs style parameters {:?} for the row '{}', and {from} {:?}",
                    named(category, target),
                    row.params.style,
                    row.name,
                    p
                )));
            }
            Some(_) => {}
            None => style_params = Some((format!("row '{}'", row.name), row.params.style.clone())),
        }
    }
    let style_params = style_params.map(|(_, p)| p).unwrap_or_default();

    let def = |e: DefError| match e {
        DefError::TypeConflict { name, .. } => conflict(format!(
            "two rows named '{name}' convert to different `{}` rows",
            named(category, target)
        )),
        DefError::PairConflict {
            itom, jtom, name, ..
        } => conflict(format!(
            "the pair {itom}-{jtom} of '{name}' converts to a `{}` row another type states \
             differently",
            named(category, target)
        )),
        other => conflict(other.to_string()),
    };
    let is_replaced = |s: &Style| {
        s.category() == category && (s.name() == target || sources.iter().any(|n| n == s.name()))
    };
    let mut out = ff.empty_like();
    let mut rows = Some(rows);
    for style in ff.styles() {
        if !is_replaced(style) {
            let copy = out
                .def_style(style.category(), style.name(), style.params().clone())
                .map_err(def)?;
            for (name, endpoints, params) in style.type_rows() {
                copy.def_type(name, &endpoints, params.clone())
                    .map_err(def)?;
            }
            continue;
        }
        let Some(rows) = rows.take() else {
            continue; // the target is already in place
        };
        let t = out
            .def_style(category, target, style_params.clone())
            .map_err(def)?;
        if let Some(own) = existing {
            for (name, endpoints, params) in own.type_rows() {
                t.def_type(name, &endpoints, params.clone()).map_err(def)?;
            }
        }
        for row in rows {
            let endpoints: Vec<&str> = row.endpoints.iter().map(String::as_str).collect();
            t.def_type(&row.name, &endpoints, row.params.row)
                .map_err(def)?;
        }
    }
    Ok(out)
}

/// The styles of `category` in `ff` that hold rows and register a codec of
/// `family`, by name, in their order.
fn members(registry: &Registry, ff: &ForceField, category: &str, family: &str) -> Vec<String> {
    ff.get_styles(category)
        .into_iter()
        .filter(|s| !s.type_rows().is_empty())
        .filter(|s| {
            registry
                .form(category, s.name())
                .is_some_and(|c| c.family == family)
        })
        .map(|s| s.name().to_owned())
        .collect()
}

impl Registry {
    /// [`ForceField::canonical`] against this registry.
    pub(crate) fn canonical(&self, ff: &ForceField) -> Result<ForceField, IrError> {
        // A family a style of `ff` belongs to must have its canonical style.
        for style in ff.styles() {
            if let Some(codec) = self.form(style.category(), style.name()) {
                self.canonical_form(&codec.family)?;
            }
        }
        let families: Vec<(String, String, String)> = self
            .forms()
            .filter(|(_, _, f)| f.canonical)
            .map(|(c, s, f)| (f.family.to_string(), c.to_owned(), s.to_owned()))
            .collect();
        let mut out = ff.clone();
        for (family, category, target) in families {
            let sources = members(self, &out, &category, &family);
            if sources.is_empty() {
                continue;
            }
            let mut rows = Vec::new();
            for source in &sources {
                let style = out.get_style(&category, source).expect("a member");
                let codec = self.form(&category, source).expect("a member");
                rows.extend(convert_rows(self, style, |name, tp| {
                    (codec.embed)(&tp).map_err(|e| IrError::OutOfImage {
                        from: named(&category, source),
                        to: named(&category, &target),
                        type_: name.to_owned(),
                        reason: e.reason,
                    })
                })?);
            }
            out = rewrite(&out, &family, &category, &target, &sources, rows)?;
        }
        Ok(out)
    }

    /// [`ForceField::to_form`] against this registry.
    pub(crate) fn to_form(
        &self,
        ff: &ForceField,
        category: &str,
        style: &str,
    ) -> Result<ForceField, IrError> {
        let codec = self.form(category, style).ok_or_else(|| IrError::NoForm {
            category: category.to_owned(),
            style: style.to_owned(),
        })?;
        let family = codec.family.to_string();
        let (cc, cs) = self.canonical_form(&family)?;
        let canonical = named(cc, cs);
        let sources: Vec<String> = members(self, ff, category, &family)
            .into_iter()
            .filter(|s| s != style)
            .collect();
        let mut rows = Vec::new();
        for source in &sources {
            let from = self.form(category, source).expect("a member");
            let style_of = ff.get_style(category, source).expect("a member");
            rows.extend(convert_rows(self, style_of, |name, tp| {
                let refused = |to: &str, e: FormRefusal| IrError::OutOfImage {
                    from: named(category, source),
                    to: to.to_owned(),
                    type_: name.to_owned(),
                    reason: e.reason,
                };
                let c = (from.embed)(&tp).map_err(|e| refused(&canonical, e))?;
                (codec.project)(&c).map_err(|e| refused(&named(category, style), e))
            })?);
        }
        if rows.is_empty() {
            return Ok(ff.clone());
        }
        rewrite(ff, &family, category, style, &sources, rows)
    }
}

impl ForceField {
    /// The force field in canonical form: every style of a form family in the
    /// family's canonical style's category mapped onto that style, exactly
    /// (the same energy, constant included), its rows in their canonical
    /// spelling — `dihedral opls`, `charmm`, `rb`, … → `dihedral periodic`,
    /// `bond class2` → `bond harmonic`, `pair lj/class2` → `pair lj/cut`.
    /// Styles of no family, and of a family whose canonical style is in
    /// another category (impropers), stay as they are. Idempotent.
    ///
    /// # Errors
    ///
    /// [`IrError::OutOfImage`] for a row the canonical style cannot hold
    /// (a charmm `w ≠ 0`, a class2 bond with `k3 ≠ 0`), naming the type and
    /// the condition; [`IrError::FormConflict`] for two rows of one name
    /// that become different rows, two styles that need different style
    /// parameters, or a family without a canonical style.
    pub fn canonical(&self) -> Result<ForceField, IrError> {
        self.registry().with(|r| r.canonical(self))
    }

    /// The force field with every style of `category` in the form family of
    /// `style` converted to `style`, exactly, through the family's canonical
    /// parameters; the other styles stay.
    ///
    /// # Errors
    ///
    /// [`IrError::NoForm`] when `style` registers no form codec;
    /// [`IrError::OutOfImage`] naming the first row outside `style`'s image
    /// and the condition (`sin(2φ) coefficient … ≠ 0`, `the constant term …`);
    /// [`IrError::FormConflict`] as for [`canonical`](Self::canonical).
    pub fn to_form(&self, category: &str, style: &str) -> Result<ForceField, IrError> {
        self.registry().with(|r| r.to_form(self, category, style))
    }

    /// Fit every other style of `category` that holds rows to `style`, row
    /// by row, by least squares under `metric`; return the force field with
    /// those styles replaced by `style` and the [`FitResidual`] of each row.
    ///
    /// Per row: the source's energies at the metric's points come from its
    /// kernel. When source and target are of one form family and the row is
    /// in the target's image, the exact conversion is taken
    /// ([`TypeResidual::exact`]). Otherwise the fit starts from the target
    /// codec's seed (else the source row's same-named values, else the
    /// spec's default, else 1) and minimises the weighted squared residual by
    /// Levenberg–Marquardt over the target's **dimensioned** scalar
    /// parameters; a dimensionless parameter (a periodicity, a sign, a 1-4
    /// weight) is a choice of form, held at its seed.
    ///
    /// # Errors
    ///
    /// [`IrError::NoKernel`] for a `style` not registered;
    /// [`IrError::OutOfImage`] (no type) for a category with no scalar
    /// coordinate or a pair category (whose unlike pairs follow a mixing
    /// rule a per-row fit cannot hold), and for a row whose source cannot
    /// be priced; [`IrError::Malformed`] for a malformed metric;
    /// [`IrError::FormConflict`] as for
    /// [`canonical`](crate::ff::forcefield::ForceField::canonical).
    pub fn fit_form(
        &self,
        category: &str,
        style: &str,
        metric: &FitMetric,
    ) -> Result<(ForceField, FitResidual), IrError> {
        self.registry()
            .with(|r| r.fit_form(self, category, style, metric))
    }
}
