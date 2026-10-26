//! The built-in form families: `torsion` (in
//! [`crate::ff::ir::torsion`], beside its algebra), `bond`, `angle`
//! and `lj`.

use crate::ff::ir::Params;
use crate::ff::ir::torsion;
use crate::ff::ir::{FormCodec, FormRefusal, TypeParams};

/// Every built-in `(category, style, codec)`, registered and sealed by
/// [`Registry::builtin`](crate::ff::style_registry::Registry::builtin).
pub fn builtin_forms() -> Vec<(&'static str, &'static str, FormCodec)> {
    let mut out = torsion::codecs();
    out.extend(polynomial("bond", "r0"));
    out.extend(polynomial("angle", "theta0"));
    out.push(("angle", "charmm", angle_charmm()));
    out.extend(lj());
    out
}

/// A required numeric param.
fn need(p: &Params, what: &str, key: &str) -> Result<f64, FormRefusal> {
    p.get(key)
        .and_then(|v| v.as_num())
        .ok_or_else(|| FormRefusal::new(format!("{what}: missing param `{key}`")))
}

/// `harmonic` (canonical: `k(q − q0)²`, the identity on its two
/// parameters) and `class2` (`k2 d² + k3 d³ + k4 d⁴`, `d = q − q0`), whose
/// image in the harmonic is `k3 = k4 = 0`, of `category` with equilibrium
/// parameter `q0`.
fn polynomial(
    category: &'static str,
    q0: &'static str,
) -> Vec<(&'static str, &'static str, FormCodec)> {
    let harmonic = move |tp: &TypeParams| -> Result<TypeParams, FormRefusal> {
        let what = format!("{category} harmonic");
        Ok(TypeParams::row(Params::from_iter([
            ("k", need(&tp.row, &what, "k")?),
            (q0, need(&tp.row, &what, q0)?),
        ])))
    };
    let embed = move |tp: &TypeParams| -> Result<TypeParams, FormRefusal> {
        let what = format!("{category} class2");
        for key in ["k3", "k4"] {
            let v = need(&tp.row, &what, key)?;
            if v != 0.0 {
                let power = if key == "k3" { "cubic" } else { "quartic" };
                return Err(FormRefusal::new(format!(
                    "{key} = {v} ≠ 0: a {power} term has no harmonic form"
                )));
            }
        }
        Ok(TypeParams::row(Params::from_iter([
            ("k", need(&tp.row, &what, "k2")?),
            (q0, need(&tp.row, &what, q0)?),
        ])))
    };
    let project = move |tp: &TypeParams| -> Result<TypeParams, FormRefusal> {
        let h = harmonic(tp)?.row;
        Ok(TypeParams::row(Params::from_iter([
            (q0, h.get(q0).and_then(|v| v.as_num()).expect("set")),
            ("k2", h.get("k").and_then(|v| v.as_num()).expect("set")),
            ("k3", 0.0),
            ("k4", 0.0),
        ])))
    };
    vec![
        (
            category,
            "harmonic",
            FormCodec::new(category, harmonic, harmonic).as_canonical(),
        ),
        (category, "class2", FormCodec::new(category, embed, project)),
    ]
}

/// `angle charmm`, `k(θ − θ0)² + k_ub(r13 − r_ub)²`: a harmonic angle when
/// the Urey–Bradley term is off (`k_ub = 0`).
fn angle_charmm() -> FormCodec {
    const WHAT: &str = "angle charmm";
    FormCodec::new(
        "angle",
        |tp: &TypeParams| {
            let k_ub = need(&tp.row, WHAT, "k_ub")?;
            if k_ub != 0.0 {
                return Err(FormRefusal::new(format!(
                    "k_ub = {k_ub} ≠ 0: the Urey–Bradley 1-3 term is no function of θ"
                )));
            }
            Ok(TypeParams::row(Params::from_iter([
                ("k", need(&tp.row, WHAT, "k")?),
                ("theta0", need(&tp.row, WHAT, "theta0")?),
            ])))
        },
        |tp: &TypeParams| {
            Ok(TypeParams::row(Params::from_iter([
                ("k", need(&tp.row, "angle harmonic", "k")?),
                ("theta0", need(&tp.row, "angle harmonic", "theta0")?),
                ("k_ub", 0.0),
                ("r_ub", 0.0),
            ])))
        },
    )
}

/// `σ' = (2/3)^⅓ σ`: the zero of the Mie 9-6 whose minimum is at `σ` (the
/// `lj/class2` σ is the minimum, `2ε(σ/r)⁹ − 3ε(σ/r)⁶`).
fn class2_sigma_scale() -> f64 {
    (2.0_f64 / 3.0).cbrt()
}

/// The `lj` family: `pair lj/cut` (canonical, the Mie `n-m` form with the
/// zero at σ) and `pair lj/class2`, which is `lj/cut` at `n = 9, m = 6` with
/// `σ' = (2/3)^⅓ σ`. Every mixing rule is homogeneous in σ (degree 1 for σ,
/// 0 for ε), so the unlike pairs map as the self rows do.
fn lj() -> Vec<(&'static str, &'static str, FormCodec)> {
    // `lj/cut`'s own style params, its exponents and shift stated: the
    // spec's declared defaults where the style states none.
    let spec =
        crate::ff::ir::builtin_style("pair", "lj/cut").expect("pair lj/cut is a built-in style");
    let default = |k: &str| {
        spec.style_param(k)
            .and_then(|p| p.default.as_ref())
            .and_then(crate::ff::ir::ParamValue::as_num)
            .expect("pair lj/cut declares n, m and shift defaults")
    };
    let (n0, m0, shift0) = (default("n"), default("m"), default("shift"));
    let lj_cut_style = move |style: &Params| -> Params {
        let mut out = Params::new();
        if let Some(c) = style.get("cutoff").and_then(|v| v.as_num()) {
            out.set("cutoff", c);
        }
        if let Some(m) = style.get("mixing").and_then(|v| v.as_text()) {
            out.set("mixing", m);
        }
        let num = |k: &str, d: f64| style.get(k).and_then(|v| v.as_num()).unwrap_or(d);
        out.set("n", num("n", n0));
        out.set("m", num("m", m0));
        out.set("shift", num("shift", shift0));
        out
    };
    let scale_sigma = |row: &Params, by: f64| -> Params {
        let mut out = Params::new();
        if let Some(e) = row.get("epsilon").and_then(|v| v.as_num()) {
            out.set("epsilon", e);
        }
        if let Some(s) = row.get("sigma").and_then(|v| v.as_num()) {
            out.set("sigma", s * by);
        }
        out
    };
    let lj_cut = FormCodec::new(
        "lj",
        move |tp: &TypeParams| {
            Ok(TypeParams::new(
                lj_cut_style(&tp.style),
                scale_sigma(&tp.row, 1.0),
            ))
        },
        move |tp: &TypeParams| {
            Ok(TypeParams::new(
                lj_cut_style(&tp.style),
                scale_sigma(&tp.row, 1.0),
            ))
        },
    )
    .as_canonical();
    let class2 = FormCodec::new(
        "lj",
        move |tp: &TypeParams| {
            let mut style = Params::new();
            if let Some(c) = tp.style.get("cutoff").and_then(|v| v.as_num()) {
                style.set("cutoff", c);
            }
            if let Some(m) = tp.style.get("mixing").and_then(|v| v.as_text()) {
                style.set("mixing", m);
            }
            style.set("n", 9.0);
            style.set("m", 6.0);
            style.set("shift", 0.0);
            Ok(TypeParams::new(
                style,
                scale_sigma(&tp.row, class2_sigma_scale()),
            ))
        },
        move |tp: &TypeParams| {
            let style = lj_cut_style(&tp.style);
            let (n, m) = (
                style.get("n").and_then(|v| v.as_num()).expect("set"),
                style.get("m").and_then(|v| v.as_num()).expect("set"),
            );
            if (n, m) != (9.0, 6.0) {
                return Err(FormRefusal::new(format!(
                    "exponents n = {n}, m = {m}: lj/class2 is the 9-6 form"
                )));
            }
            let shift = style.get("shift").and_then(|v| v.as_num()).expect("set");
            if shift != 0.0 {
                return Err(FormRefusal::new(
                    "shift = 1: lj/class2 holds no energy shift at the cutoff",
                ));
            }
            let mut out = Params::new();
            if let Some(c) = style.get("cutoff").and_then(|v| v.as_num()) {
                out.set("cutoff", c);
            }
            if let Some(m) = style.get("mixing").and_then(|v| v.as_text()) {
                out.set("mixing", m);
            }
            Ok(TypeParams::new(
                out,
                scale_sigma(&tp.row, 1.0 / class2_sigma_scale()),
            ))
        },
    );
    vec![("pair", "lj/cut", lj_cut), ("pair", "lj/class2", class2)]
}
