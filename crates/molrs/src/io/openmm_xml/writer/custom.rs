//! Expression styles as OpenMM `Custom*Force`s (`ff-ir-02-protocol` §8).
//!
//! A style with no native OpenMM tag is written from its expression — the
//! registered one, or the instance's, or for an unregistered style the
//! instance's alone (the compile fallback's spec) — as the custom force of
//! its category:
//!
//! | category | force | rows | geometric variable |
//! |---|---|---|---|
//! | `r` (bond, …) | `<CustomBondForce>` | `<Bond>` | `r` → `10*r` (nm → Å) |
//! | `theta` (angle) | `<CustomAngleForce>` | `<Angle>` | `theta` (radians both) |
//! | `phi` (dihedral) | `<CustomTorsionForce>` | `<Proper>` | `phi` → `theta` |
//! | `phi`, `chi` (improper) | `<CustomTorsionForce ordering="charmm">` | `<Improper>`, the centre first | `phi` → `theta`, `chi` → `abs(theta)` |
//! | points (a compound category, or an expression calling `distance`, `angle`, `dihedral`) | `CustomCompoundBondForce`, built by a `<Script>` over OpenMM's bonds, angles or propers | the rows' labels | `distance(…)` → `10*distance(…)` |
//! | pair | `<CustomNonbondedForce>` | `<Atom>` per type | `r` → `10*r`; bare `x` → its mixing rule of `x1`, `x2`; `q1`, `q2` → `charge1`, `charge2` |
//!
//! The parameters stay in IR units (`PerBondParameter`, …,
//! `GlobalParameter` for the style-level ones): the expression is rewritten
//! instead, exactly — `4.184*(E[r → 10*r])`, the energy from kcal/mol to
//! kJ/mol and the length from nm to Å — through the expression printer
//! ([`Parsed`]), which prints numbers that read back to the same `f64`.
//!
//! OpenMM's ForceField XML has no `<CustomCompoundBondForce>` tag (an unknown
//! tag would be ignored, its energy lost), so a compound term is a
//! `<Script>`: Python `createSystem` runs, building the force over the
//! bonds, angles or propers of the topology whose atom types match a row,
//! either way round — a compound category of 2, 3 or 4 atoms in a chain.
//!
//! Refused ([`IrError::NoEngineForm`](crate::ff::ir::IrError::NoEngineForm)):
//! a style without an expression, an indexed parameter, a row without a
//! value its expression reads, a wildcard endpoint where OpenMM would
//! re-order the atoms (impropers, compound rows), a compound category that
//! is not a chain of 2 to 4 atoms, two custom forces stating one global
//! parameter at different values; for a pair, a cross row, a bare parameter
//! that does not mix, an atom type without a row, and special-bonds weights
//! other than whole neighbours excluded (`bondCutoff`).

use std::collections::BTreeMap;

use super::{
    Charges, Endpoints, KCAL_TO_KJ, NM_TO_ANGSTROM, NO_CHARGES, OpenmmXmlWriter, Out, centre_first,
    either_way, esc,
};
use crate::ff::forcefield::{ForceField, Style};
use crate::ff::ir::CombiningRule;
use crate::ff::ir::Params;
use crate::ff::ir::expression::{self, BinOp, Definition, Expr, Func, Parsed};
use crate::ff::ir::{
    CategorySpec, Coordinate, EndpointOrder, Engine, ParamCombination, ParamKind, ParamValue,
    SpecialClass, StyleSpec,
};
use crate::ff::style_registry::Registry;
use crate::ff::style_registry::fallback_spec;
use crate::io::writer::ForceFieldWriteError;
use molrs::op::F;

fn refuse(style: &Style, why: impl Into<String>) -> ForceFieldWriteError {
    Engine::OpenmmXml
        .refuse(style.category(), style.name(), why)
        .into()
}

/// The category, spec and parsed expression a style is written under.
struct Custom {
    cat: CategorySpec,
    spec: StyleSpec,
    parsed: Parsed,
}

/// `style`'s expression and the spec stating its parameters: the registered
/// spec (with the instance's own expression when it carries one), else the
/// compile fallback's spec of an unregistered style's instance expression.
fn custom_of(reg: &Registry, style: &Style) -> Result<Custom, ForceFieldWriteError> {
    let cat = crate::ff::compile::category_of(reg, style)
        .map_err(|_| refuse(style, "its category is not registered"))?
        .into_owned();
    let instance = style.params().get("expression").and_then(|v| v.as_text());
    let no_expression = || {
        refuse(
            style,
            "OpenMM has no native tag of its form, and it has no expression to write a \
             Custom*Force from",
        )
    };
    let spec = match reg.style(style.category(), style.name()) {
        Some((spec, _)) => {
            let mut spec = spec.clone();
            if let Some(e) = instance {
                spec.expression = Some(e.to_owned());
            }
            spec
        }
        None => {
            let expression = instance.ok_or_else(no_expression)?;
            let rows = style.type_rows();
            let tp: Vec<(&str, &Params)> = rows.iter().map(|(n, _, p)| (*n, *p)).collect();
            fallback_spec(&cat, style.name(), style.params(), &tp, expression)
        }
    };
    let source = spec.expression.as_deref().ok_or_else(no_expression)?;
    let parsed =
        expression::parse(source).map_err(|e| refuse(style, format!("its expression: {e}")))?;
    Ok(Custom { cat, spec, parsed })
}

// ── tree rewriting ──────────────────────────────────────────────────────────

/// `e` with every node `f` maps replaced (top-down; a replacement is not
/// visited again).
fn rewrite(e: &Expr, f: &dyn Fn(&Expr) -> Option<Expr>) -> Expr {
    if let Some(new) = f(e) {
        return new;
    }
    match e {
        Expr::Num(_) | Expr::Var(_) => e.clone(),
        Expr::Neg(a) => Expr::neg(rewrite(a, f)),
        Expr::Bin(op, l, r) => Expr::bin(*op, rewrite(l, f), rewrite(r, f)),
        Expr::Call(name, args) => {
            Expr::Call(name.clone(), args.iter().map(|a| rewrite(a, f)).collect())
        }
    }
}

/// The energy `4.184*(main')` with every node rewritten by `f`: the
/// expression in kJ/mol of OpenMM's coordinates, its parameters in IR units.
fn in_openmm_units(p: &Parsed, f: &dyn Fn(&Expr) -> Option<Expr>) -> Parsed {
    let main = Expr::bin(
        BinOp::Mul,
        Expr::num(KCAL_TO_KJ.get()),
        rewrite(p.main(), f),
    );
    let defs = p
        .defs()
        .iter()
        .map(|d| Definition {
            name: d.name.clone(),
            expr: rewrite(&d.expr, f),
        })
        .collect();
    Parsed::from_tree(main, defs)
}

/// `10*e`: an OpenMM length (nm) in Å.
fn angstrom(e: Expr) -> Expr {
    Expr::bin(BinOp::Mul, Expr::num(NM_TO_ANGSTROM.get()), e)
}

/// Every identifier the expression names (definitions included).
fn identifiers(p: &Parsed) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut add = |e: &Expr| {
        for id in e.identifiers() {
            if !out.iter().any(|o| o == id) {
                out.push(id.to_owned());
            }
        }
    };
    add(p.main());
    for d in p.defs() {
        add(&d.expr);
    }
    out
}

/// Whether the expression reads the atoms' positions: a geometric function
/// or a point.
fn uses_points(p: &Parsed) -> bool {
    fn walk(e: &Expr) -> bool {
        match e {
            Expr::Num(_) => false,
            Expr::Var(v) => is_point(v),
            Expr::Neg(a) => walk(a),
            Expr::Bin(_, l, r) => walk(l) || walk(r),
            Expr::Call(name, args) => {
                Func::from_name(name).is_some_and(Func::is_geometric) || args.iter().any(walk)
            }
        }
    }
    walk(p.main()) || p.defs().iter().any(|d| walk(&d.expr))
}

fn is_point(name: &str) -> bool {
    name.strip_prefix('p')
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

fn points(n: usize) -> Vec<Expr> {
    (1..=n).map(|i| Expr::var(format!("p{i}"))).collect()
}

/// The rewrite of a scalar category's coordinate into OpenMM's.
fn scalar_rewrite(coordinate: Coordinate) -> impl Fn(&Expr) -> Option<Expr> {
    move |e| match (e, coordinate) {
        (Expr::Var(v), Coordinate::Distance) if v == "r" => Some(angstrom(Expr::var("r"))),
        (Expr::Var(v), Coordinate::Dihedral | Coordinate::Improper) if v == "phi" => {
            Some(Expr::var("theta"))
        }
        (Expr::Var(v), Coordinate::Improper) if v == "chi" => {
            Some(Expr::call("abs", vec![Expr::var("theta")]))
        }
        _ => None,
    }
}

/// The rewrite of a compound expression: a geometric variable as its
/// function of the points, every `distance` in Å.
fn compound_rewrite(e: &Expr) -> Option<Expr> {
    match e {
        Expr::Var(v) if v == "r" => Some(angstrom(Expr::call("distance", points(2)))),
        Expr::Var(v) if v == "theta" => Some(Expr::call("angle", points(3))),
        Expr::Var(v) if v == "phi" => Some(Expr::call("dihedral", points(4))),
        Expr::Var(v) if v == "chi" => {
            Some(Expr::call("abs", vec![Expr::call("dihedral", points(4))]))
        }
        Expr::Call(name, args) if name == "distance" => Some(angstrom(Expr::call(
            "distance",
            args.iter().map(|a| rewrite(a, &compound_rewrite)).collect(),
        ))),
        _ => None,
    }
}

// ── parameters ──────────────────────────────────────────────────────────────

/// The per-type parameters the expression reads, in declared order.
fn per_type<'s>(
    style: &Style,
    spec: &'s StyleSpec,
    used: &[String],
) -> Result<Vec<&'s str>, ForceFieldWriteError> {
    let mut out = Vec::new();
    for p in &spec.params {
        if p.kind != ParamKind::Scalar {
            continue;
        }
        if p.indexed {
            return Err(refuse(
                style,
                format!(
                    "parameter `{}` is indexed, which a Custom*Force cannot count",
                    p.name
                ),
            ));
        }
        if used.iter().any(|u| u == p.name.as_ref()) {
            out.push(p.name.as_ref());
        }
    }
    Ok(out)
}

/// A row's value of `name`, else the spec's default, else a refusal naming
/// the type.
fn row_value(
    style: &Style,
    spec: &StyleSpec,
    ty: &str,
    row: &Params,
    name: &str,
) -> Result<F, ForceFieldWriteError> {
    row.get(name)
        .and_then(|v| v.as_num())
        .or_else(|| {
            spec.param(name)
                .and_then(|p| p.default.as_ref())
                .and_then(ParamValue::as_num)
        })
        .ok_or_else(|| refuse(style, format!("type '{ty}' has no `{name}`")))
}

/// The style-level parameters the expression reads, with their values, as
/// `<GlobalParameter>`s: one name, one value across the file (OpenMM's
/// global parameters are the context's).
fn globals(
    w: &OpenmmXmlWriter,
    style: &Style,
    spec: &StyleSpec,
    used: &[String],
    out: &mut Out,
) -> Result<String, ForceFieldWriteError> {
    let mut xml = String::new();
    for p in spec
        .style_params
        .iter()
        .filter(|p| p.kind == ParamKind::Scalar)
    {
        if !used.iter().any(|u| u == p.name.as_ref()) {
            continue;
        }
        let v = style
            .params()
            .get(&p.name)
            .and_then(|v| v.as_num())
            .or_else(|| p.default.as_ref().and_then(ParamValue::as_num))
            .ok_or_else(|| refuse(style, format!("style parameter `{}` has no value", p.name)))?;
        match out.globals.get(p.name.as_ref()) {
            Some(&other) if other != v => {
                return Err(refuse(
                    style,
                    format!(
                        "its `{}` = {v} and another Custom force's = {other}: OpenMM's global \
                         parameters are one per context",
                        p.name
                    ),
                ));
            }
            _ => {
                out.globals.insert(p.name.to_string(), v);
            }
        }
        xml.push_str(&format!(
            "    <GlobalParameter name=\"{}\" defaultValue=\"{}\"/>\n",
            esc(&p.name),
            w.fmt_f(v)
        ));
    }
    Ok(xml)
}

// ── bonded ──────────────────────────────────────────────────────────────────

/// A bonded expression style (any category but pair) as its custom force.
pub(super) fn bonded(
    w: &OpenmmXmlWriter,
    reg: &Registry,
    style: &Style,
    ends: &Endpoints,
    out: &mut Out,
) -> Result<(), ForceFieldWriteError> {
    let rows = style.type_rows();
    let c = match custom_of(reg, style) {
        Ok(c) => c,
        Err(_) if rows.is_empty() => return Ok(()),
        Err(e) => return Err(e),
    };
    if !c.cat.prices_energy() {
        return if rows.is_empty() {
            Ok(())
        } else {
            Err(refuse(
                style,
                "OpenMM's ForceField XML has no form for a category that prices no energy",
            ))
        };
    }
    if c.cat.coordinate == Coordinate::Compound || uses_points(&c.parsed) {
        return compound(w, &c, style, ends, out);
    }
    let (force, param_tag, row_tag, ordering) = match c.cat.coordinate {
        Coordinate::Distance => ("CustomBondForce", "PerBondParameter", "Bond", ""),
        Coordinate::Angle => ("CustomAngleForce", "PerAngleParameter", "Angle", ""),
        Coordinate::Dihedral => ("CustomTorsionForce", "PerTorsionParameter", "Proper", ""),
        Coordinate::Improper => (
            "CustomTorsionForce",
            "PerTorsionParameter",
            "Improper",
            " ordering=\"charmm\"",
        ),
        Coordinate::None | Coordinate::Compound => unreachable!("handled above"),
    };
    let energy = in_openmm_units(&c.parsed, &scalar_rewrite(c.cat.coordinate));
    let used = identifiers(&c.parsed);
    let names = per_type(style, &c.spec, &used)?;
    let mut xml = format!(
        "  <{force} energy=\"{}\"{ordering}>\n",
        esc(&energy.to_string())
    );
    xml.push_str(&globals(w, style, &c.spec, &used, out)?);
    for n in &names {
        xml.push_str(&format!("    <{param_tag} name=\"{}\"/>\n", esc(n)));
    }
    let mut written = 0;
    for (ty, labels, params) in &rows {
        let mut body = String::new();
        for n in &names {
            let v = row_value(style, &c.spec, ty, params, n)?;
            body.push_str(&format!(" {}=\"{}\"", esc(n), w.fmt_f(v)));
        }
        let key = if row_tag == "Improper" {
            if labels.iter().any(|l| l.is_empty()) {
                return Err(refuse(
                    style,
                    format!(
                        "type '{ty}': a wildcard endpoint makes OpenMM re-order the improper, so \
                         it would price another dihedral"
                    ),
                ));
            }
            centre_first(labels[0], [labels[1], labels[2], labels[3]])
        } else {
            either_way(labels)
        };
        if out.admit(row_tag, key, ty, &format!("{} {body}", style.name()))? {
            xml.push_str(&format!("    <{row_tag}{}{body}/>\n", ends.attrs(labels)));
            written += 1;
        }
    }
    xml.push_str(&format!("  </{force}>\n"));
    if written > 0 {
        out.custom_forces.push(xml);
    }
    Ok(())
}

/// A compound term as a `<Script>` building a `CustomCompoundBondForce`
/// over OpenMM's bonds, angles or propers.
fn compound(
    w: &OpenmmXmlWriter,
    c: &Custom,
    style: &Style,
    ends: &Endpoints,
    out: &mut Out,
) -> Result<(), ForceFieldWriteError> {
    let arity = c.cat.arity.endpoints();
    let tuples = match (arity, c.cat.order) {
        (2, EndpointOrder::Reversible) => "[(_b.atom1, _b.atom2) for _b in data.bonds]",
        (3, EndpointOrder::Reversible) => "data.angles",
        (4, EndpointOrder::Reversible) => "data.propers",
        _ => {
            return Err(refuse(
                style,
                format!(
                    "a {arity}-atom {:?} compound term: OpenMM's templates generate bonds, \
                     angles and propers, chains of 2 to 4 atoms read either way",
                    c.cat.order
                ),
            ));
        }
    };
    let energy = in_openmm_units(&c.parsed, &compound_rewrite);
    let used = identifiers(&c.parsed);
    let names = per_type(style, &c.spec, &used)?;
    // The globals' XML is not written (the script adds them), but they are
    // held to one value across the file all the same.
    globals(w, style, &c.spec, &used, out)?;
    let global_values: Vec<String> = c
        .spec
        .style_params
        .iter()
        .filter(|p| p.kind == ParamKind::Scalar && used.iter().any(|u| u == p.name.as_ref()))
        .map(|p| format!("({}, {:?})", py_str(&p.name), out.globals[p.name.as_ref()]))
        .collect();
    let mut rows = Vec::new();
    for (ty, labels, params) in style.type_rows() {
        if labels.iter().any(|l| l.is_empty()) {
            return Err(refuse(
                style,
                format!("type '{ty}': a wildcard endpoint, which the script does not match"),
            ));
        }
        let values = names
            .iter()
            .map(|n| row_value(style, &c.spec, ty, params, n).map(|v| format!("{v:?}")))
            .collect::<Result<Vec<_>, _>>()?;
        if !out.admit(
            c.cat.name.as_ref(),
            either_way(&labels),
            ty,
            &format!("{} {values:?}", style.name()),
        )? {
            continue;
        }
        let attrs: Vec<String> = ends
            .keys(&labels)
            .into_iter()
            .map(|(k, l)| format!("{}: {}", py_str(&k), py_str(&l)))
            .collect();
        rows.push(format!(
            "({{{}}}, [{}])",
            attrs.join(", "),
            values.join(", ")
        ));
    }
    if rows.is_empty() {
        return Ok(());
    }
    let names: Vec<String> = names.iter().map(|n| py_str(n)).collect();
    let script = format!(
        "# {category} {name}: CustomCompoundBondForce (written by molrs)\n\
         import openmm as _mm\n\
         _f = _mm.CustomCompoundBondForce({arity}, {energy})\n\
         for _n in [{names}]:\n    _f.addPerBondParameter(_n)\n\
         for _n, _v in [{globals}]:\n    _f.addGlobalParameter(_n, _v)\n\
         _rows = []\n\
         for _a, _v in [{rows}]:\n    _t = self._findAtomTypes(_a, {arity})\n    if None not in _t:\n        _rows.append((_t, _v))\n\
         for _p in {tuples}:\n    _ty = [data.atomType[data.atoms[_i]] for _i in _p]\n    for _t, _v in _rows:\n        if all(_x in _s for _x, _s in zip(_ty, _t)):\n            _f.addBond(list(_p), _v)\n            break\n        if all(_x in _s for _x, _s in zip(_ty[::-1], _t)):\n            _f.addBond(list(_p)[::-1], _v)\n            break\n\
         sys.addForce(_f)\n",
        category = c.cat.name,
        name = style.name(),
        energy = py_str(&energy.to_string()),
        names = names.join(", "),
        globals = global_values.join(", "),
        rows = rows.join(", "),
    );
    out.scripts.push(script);
    Ok(())
}

/// A Python string literal (JSON's escaping is Python's for these).
fn py_str(s: &str) -> String {
    serde_json::to_string(s).expect("a string serializes")
}

// ── pair ────────────────────────────────────────────────────────────────────

/// A pair expression style as a `<CustomNonbondedForce>`.
pub(super) fn pair(
    w: &OpenmmXmlWriter,
    reg: &Registry,
    ff: &ForceField,
    style: &Style,
    out: &mut Out,
) -> Result<String, ForceFieldWriteError> {
    let c = custom_of(reg, style)?;
    let used = identifiers(&c.parsed);
    let types = style.defs().types();
    if let Some(t) = types.iter().find(|t| t.endpoints[0] != t.endpoints[1]) {
        return Err(refuse(
            style,
            format!(
                "cross row '{}': a CustomNonbondedForce prices every pair from its two atoms' \
                 parameters",
                t.name
            ),
        ));
    }
    // Bare names: the pair value, by its mixing rule of the two self rows.
    let rule = match style.params().get("mixing").and_then(|v| v.as_text()) {
        Some(m) => CombiningRule::parse(m)?,
        None => CombiningRule::UNDECLARED,
    };
    let mut bare: BTreeMap<String, Expr> = BTreeMap::new();
    let mut per_particle: Vec<&str> = Vec::new();
    let one = |n: &str, i: u8| Expr::var(format!("{n}{i}"));
    let call = |f: &str, a: Expr| Expr::call(f, vec![a]);
    let mul = |a: Expr, b: Expr| Expr::bin(BinOp::Mul, a, b);
    let pow = |a: Expr, n: F| Expr::bin(BinOp::Pow, a, Expr::num(n));
    for p in c.spec.params.iter().filter(|p| p.kind == ParamKind::Scalar) {
        let name = p.name.as_ref();
        let pair_used = used.iter().any(|u| u == name);
        let self_used = used
            .iter()
            .any(|u| *u == format!("{name}1") || *u == format!("{name}2"));
        if !pair_used && !self_used {
            continue;
        }
        if p.indexed {
            return Err(refuse(style, format!("parameter `{name}` is indexed")));
        }
        if !per_particle.contains(&name) {
            per_particle.push(name);
        }
        if !pair_used {
            continue;
        }
        let geometric = || call("sqrt", mul(one(name, 1), one(name, 2)));
        let formula = match (&p.mix, rule) {
            (ParamCombination::None, _) => {
                return Err(refuse(
                    style,
                    format!(
                        "`{name}` does not mix, so its pair value is a cross row's, which a \
                         CustomNonbondedForce has not"
                    ),
                ));
            }
            (ParamCombination::Arithmetic, _) => mul(
                Expr::num(0.5),
                Expr::bin(BinOp::Add, one(name, 1), one(name, 2)),
            ),
            (ParamCombination::Geometric, _)
            | (
                ParamCombination::LjEpsilon { .. },
                CombiningRule::Arithmetic | CombiningRule::Geometric,
            ) => geometric(),
            (ParamCombination::LjSigma { .. }, CombiningRule::Arithmetic) => mul(
                Expr::num(0.5),
                Expr::bin(BinOp::Add, one(name, 1), one(name, 2)),
            ),
            (ParamCombination::LjSigma { .. }, CombiningRule::Geometric) => geometric(),
            (ParamCombination::LjSigma { .. }, CombiningRule::SixthPower) => pow(
                mul(
                    Expr::num(0.5),
                    Expr::bin(BinOp::Add, pow(one(name, 1), 6.0), pow(one(name, 2), 6.0)),
                ),
                1.0 / 6.0,
            ),
            (ParamCombination::LjEpsilon { sigma }, CombiningRule::SixthPower) => {
                let s = sigma.as_ref();
                if !per_particle.contains(&s) && c.spec.param(s).is_some() {
                    per_particle.push(c.spec.param(s).unwrap().name.as_ref());
                }
                Expr::bin(
                    BinOp::Div,
                    mul(
                        mul(mul(Expr::num(2.0), geometric()), pow(one(s, 1), 3.0)),
                        pow(one(s, 2), 3.0),
                    ),
                    Expr::bin(BinOp::Add, pow(one(s, 1), 6.0), pow(one(s, 2), 6.0)),
                )
            }
        };
        bare.insert(name.to_owned(), formula);
    }
    let charged = used.iter().any(|u| u == "q1" || u == "q2");
    if charged && c.spec.param("charge").is_some() {
        return Err(refuse(
            style,
            "its parameter `charge` is the name OpenMM's per-particle charge takes here",
        ));
    }
    let f = |e: &Expr| match e {
        Expr::Var(v) if v == "r" => Some(angstrom(Expr::var("r"))),
        Expr::Var(v) if v == "q1" => Some(Expr::var("charge1")),
        Expr::Var(v) if v == "q2" => Some(Expr::var("charge2")),
        Expr::Var(v) => bare.get(v).cloned(),
        _ => None,
    };
    let energy = in_openmm_units(&c.parsed, &f);

    // Excluded whole neighbours: weights [0, …, 0, 1, …, 1].
    let sb = ff.required_special_bonds()?;
    let weights = match c.spec.special_class() {
        SpecialClass::Vdw => sb.lj,
        SpecialClass::Coulomb => sb.coul,
    };
    let bond_cutoff = match weights {
        [1.0, 1.0, 1.0] => 0,
        [0.0, 1.0, 1.0] => 1,
        [0.0, 0.0, 1.0] => 2,
        [0.0, 0.0, 0.0] => 3,
        other => {
            return Err(refuse(
                style,
                format!(
                    "special_bonds weights {other:?}: a CustomNonbondedForce excludes whole \
                     bonded neighbours (bondCutoff) and scales none"
                ),
            ));
        }
    };

    let mut xml = format!(
        "  <CustomNonbondedForce energy=\"{}\" bondCutoff=\"{bond_cutoff}\">\n",
        esc(&energy.to_string())
    );
    xml.push_str(&globals(w, style, &c.spec, &used, out)?);
    for n in &per_particle {
        xml.push_str(&format!(
            "    <PerParticleParameter name=\"{}\"/>\n",
            esc(n)
        ));
    }
    let atoms: Vec<_> = ff
        .get_types("atom")
        .into_iter()
        .filter(|t| t.params.get("type_").and_then(|v| v.as_text()) != Some("*"))
        .collect();
    // Charges from the residue templates (a frame's) are not the types'.
    let all_charged = !matches!(out.charges, Charges::Residues { .. })
        && atoms.iter().all(|t| t.params.get("charge").is_some());
    if charged {
        xml.push_str("    <PerParticleParameter name=\"charge\"/>\n");
        if !all_charged {
            if out.charges == Charges::TypesOnly {
                return Err(NO_CHARGES.into());
            }
            xml.push_str("    <UseAttributeFromResidue name=\"charge\"/>\n");
        }
    }
    let mut sorted = atoms;
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    for t in sorted {
        let row = types
            .iter()
            .find(|p| p.endpoints[0] == t.name)
            .ok_or_else(|| {
                refuse(
                    style,
                    format!(
                        "atom type '{}' has no row, and a CustomNonbondedForce needs one for \
                         every type",
                        t.name
                    ),
                )
            })?;
        let mut attrs = format!("    <Atom type=\"{}\"", esc(&t.name));
        for n in &per_particle {
            let v = row_value(style, &c.spec, &row.name, &row.params, n)?;
            attrs.push_str(&format!(" {}=\"{}\"", esc(n), w.fmt_f(v)));
        }
        if charged && all_charged {
            attrs.push_str(&format!(
                " charge=\"{}\"",
                w.fmt_f(t.params.get("charge").and_then(|v| v.as_num()).unwrap())
            ));
        }
        attrs.push_str("/>\n");
        xml.push_str(&attrs);
    }
    xml.push_str("  </CustomNonbondedForce>\n");
    Ok(xml)
}
