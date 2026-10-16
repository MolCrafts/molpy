//! The LAMMPS codecs of the built-in styles whose lines are not positional
//! ([`LammpsForm::Custom`]); every other built-in LAMMPS has is
//! [`LammpsForm::Positional`] (`ff-ir-02-protocol` §8).
//!
//! | style | LAMMPS | why a codec of its own |
//! |---|---|---|
//! | `dihedral periodic` | `fourier`: `m K1 n1 d1 …` | the term count `m`; indexed parameters |
//! | `dihedral nharmonic` | `N A1 … AN` | the count `N` |
//! | `dihedral charmm` | `K n d w` | `n` and `d` are integers to LAMMPS |
//! | `dihedral harmonic`, `improper cvff` | `K d n` | `d` = ±1 and `n` integers |
//! | `improper periodic` | `cvff` `K d n` | AMBER's phase 0° / 180° as LAMMPS's sign |
//! | `angle class2`, `dihedral class2` | `class2` and its cross-term lines | LAMMPS needs the `bb`/`ba` (`mbt`/`ebt`/`at`/`aat`/`bb13`) lines, written at zero: the IR has no cross terms |
//! | `pair lj/cut` | `lj/cut` | `shift` is `pair_modify shift`; `n`, `m` only 12-6 |
//! | `pair lj/charmm` | `lj/charmm/coul/charmm` | the 1-4 pair, the two switching cutoffs |
//! | `pair lj/class2` | `lj/class2` | LAMMPS always mixes it `sixthpower` |
//! | `pair coul/cut`, `coul/long/pme`, `coul/charmm` | the Coulomb half of a pair style | no per-type line; the Coulomb constant is LAMMPS's own |
//! | `cmap charmm` | `fix cmap` | a grid file, not a coefficient line |

use std::sync::{Arc, LazyLock};

use crate::ff::ir::OneFour;
use crate::ff::ir::Params;
use crate::ff::ir::positional::{self, number, parse_named, value};
use crate::ff::ir::torsion::nharmonic_coefficients;
use crate::ff::ir::{Engine, EngineCodec, LammpsCodec, LammpsCoeffs, LammpsForm, Token, UnitScale};
use crate::ff::ir::{ParamDimension, StyleSpec};
use molrs::op::F;

type Codec = LazyLock<Arc<dyn LammpsCodec>>;

/// `codec` as a style's [`LammpsForm`].
pub(crate) fn custom(codec: &Codec) -> LammpsForm {
    LammpsForm::Custom(Arc::clone(codec))
}

macro_rules! codec {
    ($(#[$doc:meta])* $static:ident = $ty:ident, $name:expr) => {
        $(#[$doc])*
        pub(crate) static $static: Codec = LazyLock::new(|| Arc::new($ty));

        impl EngineCodec for $ty {
            fn engine(&self) -> Engine {
                Engine::Lammps
            }

            fn name(&self, spec: &StyleSpec) -> String {
                let name: Option<&str> = $name;
                name.unwrap_or(&spec.name).to_owned()
            }
        }
    };
}

fn what(spec: &StyleSpec) -> String {
    format!("{} {}", spec.category, spec.name)
}

/// `v` as a LAMMPS integer, refused with `key` named otherwise.
fn whole(spec: &StyleSpec, key: &str, v: F) -> Result<Token, String> {
    Token::integral(v).ok_or_else(|| {
        format!(
            "{}: `{key}` = {v} is not an integer, which LAMMPS reads it as",
            what(spec)
        )
    })
}

fn sign(spec: &StyleSpec, v: F) -> Result<Token, String> {
    if v == 1.0 || v == -1.0 {
        Ok(Token::Int(v as i64))
    } else {
        Err(format!("{}: `sign` = {v} is not ±1", what(spec)))
    }
}

fn energy(units: &UnitScale, v: F) -> Token {
    Token::Real(units.apply(v, ParamDimension::ENERGY))
}

fn no_rows(spec: &StyleSpec) -> String {
    format!(
        "{}: a Coulomb style has no per-type coefficient line (its charges are the atoms')",
        what(spec)
    )
}

// ── dihedral periodic: fourier ──────────────────────────────────────────────

pub(crate) struct FourierCodec;
codec!(
    /// `dihedral periodic` is LAMMPS's `fourier`, term for term.
    FOURIER = FourierCodec,
    Some("fourier")
);

impl LammpsCodec for FourierCodec {
    /// `m K1 n1 d1 [K2 n2 d2 …]` from `k<i>` / `periodicity<i>` / `phase<i>`
    /// (`phase` absent: 0°); the unindexed `k` / `periodicity` / `phase` is
    /// the one-term case.
    fn write(
        &self,
        spec: &StyleSpec,
        p: &Params,
        units: &UnitScale,
    ) -> Result<LammpsCoeffs, String> {
        let need = |key: &str| {
            p.get(key)
                .ok_or_else(|| format!("{}: missing parameter `{key}`", what(spec)))
        };
        let mut terms = Vec::new();
        let mut known = vec!["k".to_owned(), "periodicity".to_owned(), "phase".to_owned()];
        if p.get("k1").is_none()
            && let Some(k) = p.get("k")
        {
            terms.push(energy(units, k));
            terms.push(whole(spec, "periodicity", need("periodicity")?)?);
            terms.push(Token::Real(p.get("phase").unwrap_or(0.0)));
        }
        let mut i = 1usize;
        while let Some(k) = p.get(&format!("k{i}")) {
            let n = format!("periodicity{i}");
            terms.push(energy(units, k));
            terms.push(whole(spec, &n, need(&n)?)?);
            terms.push(Token::Real(p.get(&format!("phase{i}")).unwrap_or(0.0)));
            known.extend([format!("k{i}"), n, format!("phase{i}")]);
            i += 1;
        }
        if terms.is_empty() {
            return Err(format!("{}: missing parameter `k1`", what(spec)));
        }
        if let Some((key, _)) = p
            .iter_parameters()
            .find(|(k, _)| !known.iter().any(|n| n == k))
        {
            return Err(format!(
                "{}: parameter `{key}` has no place on the fourier line",
                what(spec)
            ));
        }
        let mut values = vec![Token::Int(terms.len() as i64 / 3)];
        values.extend(terms);
        Ok(LammpsCoeffs {
            values,
            extra: Vec::new(),
        })
    }

    fn read(&self, spec: &StyleSpec, values: &[&str]) -> Result<Params, String> {
        let m: usize = values
            .first()
            .ok_or_else(|| format!("{}: missing the term count m", what(spec)))?
            .parse()
            .map_err(|_| format!("{}: the term count m is not an integer", what(spec)))?;
        if values.len() != 1 + 3 * m {
            return Err(format!(
                "{} (fourier) takes m = {m} and {} more coefficients (K n d per term), got {}",
                what(spec),
                3 * m,
                values.len() - 1
            ));
        }
        let mut params = Params::new();
        for t in 0..m {
            let i = t + 1;
            let base = 1 + 3 * t;
            params.set(&format!("k{i}"), number("dihedral K", values[base])?);
            params.set(
                &format!("periodicity{i}"),
                number("dihedral n", values[base + 1])?,
            );
            params.set(
                &format!("phase{i}"),
                number("dihedral d", values[base + 2])?,
            );
        }
        Ok(params)
    }
}

// ── dihedral nharmonic ──────────────────────────────────────────────────────

pub(crate) struct NharmonicCodec;
codec!(
    /// `N A1 … AN`.
    NHARMONIC = NharmonicCodec,
    None
);

impl LammpsCodec for NharmonicCodec {
    fn write(
        &self,
        spec: &StyleSpec,
        p: &Params,
        units: &UnitScale,
    ) -> Result<LammpsCoeffs, String> {
        let a = nharmonic_coefficients(p)?;
        if let Some((key, _)) = p.iter_parameters().find(|(k, _)| {
            k.strip_prefix('a')
                .and_then(|i| i.parse::<usize>().ok())
                .is_none()
        }) {
            return Err(format!(
                "{}: parameter `{key}` has no place on the nharmonic line",
                what(spec)
            ));
        }
        let mut values = vec![Token::Int(a.len() as i64)];
        values.extend(a.into_iter().map(|v| energy(units, v)));
        Ok(LammpsCoeffs {
            values,
            extra: Vec::new(),
        })
    }

    fn read(&self, spec: &StyleSpec, values: &[&str]) -> Result<Params, String> {
        let n: usize = values
            .first()
            .and_then(|t| t.parse().ok())
            .filter(|&n| n >= 1)
            .ok_or_else(|| format!("{}: the count N is not an integer ≥ 1", what(spec)))?;
        if values.len() != 1 + n {
            return Err(format!(
                "{} takes N = {n} and {n} coefficients, got {}",
                what(spec),
                values.len() - 1
            ));
        }
        let mut params = Params::new();
        for (i, raw) in values.iter().enumerate().skip(1) {
            params.set(&format!("a{i}"), number("dihedral A", raw)?);
        }
        Ok(params)
    }
}

// ── dihedral charmm ─────────────────────────────────────────────────────────

pub(crate) struct DihedralCharmmCodec;
codec!(
    /// `K n d w`, `n` and `d` integers (`dihedral_charmm.cpp` reads both
    /// with `inumeric`).
    DIHEDRAL_CHARMM = DihedralCharmmCodec,
    None
);

impl LammpsCodec for DihedralCharmmCodec {
    fn write(
        &self,
        spec: &StyleSpec,
        p: &Params,
        units: &UnitScale,
    ) -> Result<LammpsCoeffs, String> {
        positional::refuse_undeclared_params(spec, p)?;
        let get = |name: &str| value(spec, spec.param(name).expect("a charmm parameter"), p);
        let phase = get("phase")?;
        if phase.fract() != 0.0 {
            return Err(format!(
                "dihedral charmm: phase = {phase}° is not an integer number of degrees, which \
                 LAMMPS's dihedral_style charmm requires"
            ));
        }
        Ok(LammpsCoeffs {
            values: vec![
                energy(units, get("k")?),
                whole(spec, "periodicity", get("periodicity")?)?,
                Token::Int(phase as i64),
                Token::Real(get("w")?),
            ],
            extra: Vec::new(),
        })
    }

    fn read(&self, spec: &StyleSpec, values: &[&str]) -> Result<Params, String> {
        positional::read(spec, values)
    }
}

// ── dihedral harmonic, improper cvff: K d n ─────────────────────────────────

pub(crate) struct SignedCosineCodec;
codec!(
    /// `K d n`: `E = K[1 + d cos(nφ)]`, `d` a sign (±1) and `n` an integer.
    SIGNED_COSINE = SignedCosineCodec,
    None
);

impl LammpsCodec for SignedCosineCodec {
    fn write(
        &self,
        spec: &StyleSpec,
        p: &Params,
        units: &UnitScale,
    ) -> Result<LammpsCoeffs, String> {
        positional::refuse_undeclared_params(spec, p)?;
        let get = |name: &str| value(spec, spec.param(name).expect("a K d n parameter"), p);
        Ok(LammpsCoeffs {
            values: vec![
                energy(units, get("k")?),
                sign(spec, get("sign")?)?,
                whole(spec, "periodicity", get("periodicity")?)?,
            ],
            extra: Vec::new(),
        })
    }

    fn read(&self, spec: &StyleSpec, values: &[&str]) -> Result<Params, String> {
        positional::read(spec, values)
    }
}

// ── improper periodic: LAMMPS cvff ──────────────────────────────────────────

pub(crate) struct PeriodicAsCvffCodec;
codec!(
    /// AMBER's `improper periodic`, `E = K[1 + cos(nφ − φ0)]`, is LAMMPS's
    /// `improper_style cvff`, `E = K[1 + d cos(nφ)]`, when φ0 is 0° (`d` =
    /// +1) or 180° (`d` = −1) — every GAFF improper; both price the dihedral
    /// I-J-K-L of the stored order. AMBER writes π as 3.1416, which a reader
    /// in degrees stores as 180.0004, hence a tolerance of 1e-3 rad; any
    /// other phase is refused, not rounded.
    PERIODIC_AS_CVFF = PeriodicAsCvffCodec,
    Some("cvff")
);

impl LammpsCodec for PeriodicAsCvffCodec {
    fn write(
        &self,
        spec: &StyleSpec,
        p: &Params,
        units: &UnitScale,
    ) -> Result<LammpsCoeffs, String> {
        positional::refuse_undeclared_params(spec, p)?;
        let get = |name: &str| value(spec, spec.param(name).expect("a periodic parameter"), p);
        let phase = get("phase")?.rem_euclid(360.0);
        let near = |x: F| (phase - x).abs() < 1e-3_f64.to_degrees();
        let d = if near(0.0) || near(360.0) {
            1
        } else if near(180.0) {
            -1
        } else {
            return Err(format!(
                "{}: phase {phase}° has no LAMMPS `cvff` form (needs 0° or 180°)",
                what(spec)
            ));
        };
        Ok(LammpsCoeffs {
            values: vec![
                energy(units, get("k")?),
                Token::Int(d),
                whole(spec, "periodicity", get("periodicity")?)?,
            ],
            extra: Vec::new(),
        })
    }

    fn read(&self, spec: &StyleSpec, values: &[&str]) -> Result<Params, String> {
        let p = parse_named(spec, &["k", "d", "periodicity"], values)?;
        let phase = match p.get("d") {
            Some(1.0) => 0.0,
            Some(-1.0) => 180.0,
            d => return Err(format!("{}: d = {d:?} is not ±1", what(spec))),
        };
        let mut out = Params::new();
        out.set("k", p.get("k").unwrap());
        out.set("periodicity", p.get("periodicity").unwrap());
        out.set("phase", phase);
        Ok(out)
    }
}

// ── class2: the cross-term lines at zero ────────────────────────────────────

/// A class2 cross-term line: its keyword, its number of values, and which
/// of them are force constants (the rest are reference geometries).
struct CrossTerm {
    keyword: &'static str,
    values: usize,
    constants: &'static [usize],
}

fn zero_cross_lines(terms: &[CrossTerm]) -> Vec<(&'static str, Vec<Token>)> {
    terms
        .iter()
        .map(|t| (t.keyword, vec![Token::Int(0); t.values]))
        .collect()
}

fn check_zero_cross_line(
    spec: &StyleSpec,
    terms: &[CrossTerm],
    keyword: &str,
    values: &[&str],
) -> Result<(), String> {
    let term = terms
        .iter()
        .find(|t| t.keyword == keyword)
        .ok_or_else(|| format!("{}: no `{keyword}` coefficient line", what(spec)))?;
    if values.len() != term.values {
        return Err(format!(
            "{} `{keyword}` takes {} coefficients, got {}",
            what(spec),
            term.values,
            values.len()
        ));
    }
    for &i in term.constants {
        let v = number(keyword, values[i])?;
        if v != 0.0 {
            return Err(format!(
                "{} `{keyword}` coefficient {} = {v}: a class2 cross term, which the \
                 force-field IR has no form for (only a zero one reads)",
                what(spec),
                i + 1
            ));
        }
    }
    for raw in values {
        number(keyword, raw)?;
    }
    Ok(())
}

const ANGLE_CROSS: [CrossTerm; 2] = [
    // bb M r1 r2
    CrossTerm {
        keyword: "bb",
        values: 3,
        constants: &[0],
    },
    // ba N1 N2 r1 r2
    CrossTerm {
        keyword: "ba",
        values: 4,
        constants: &[0, 1],
    },
];

const DIHEDRAL_CROSS: [CrossTerm; 5] = [
    // mbt A1 A2 A3 r2
    CrossTerm {
        keyword: "mbt",
        values: 4,
        constants: &[0, 1, 2],
    },
    // ebt B1 B2 B3 C1 C2 C3 r1 r3
    CrossTerm {
        keyword: "ebt",
        values: 8,
        constants: &[0, 1, 2, 3, 4, 5],
    },
    // at D1 D2 D3 E1 E2 E3 theta1 theta2
    CrossTerm {
        keyword: "at",
        values: 8,
        constants: &[0, 1, 2, 3, 4, 5],
    },
    // aat M theta1 theta2
    CrossTerm {
        keyword: "aat",
        values: 3,
        constants: &[0],
    },
    // bb13 N r1 r3
    CrossTerm {
        keyword: "bb13",
        values: 3,
        constants: &[0],
    },
];

pub(crate) struct AngleClass2Codec;
codec!(
    /// `theta0 K2 K3 K4`, and `bb`/`ba` lines at zero.
    ANGLE_CLASS2 = AngleClass2Codec,
    None
);

impl LammpsCodec for AngleClass2Codec {
    fn write(
        &self,
        spec: &StyleSpec,
        p: &Params,
        units: &UnitScale,
    ) -> Result<LammpsCoeffs, String> {
        Ok(LammpsCoeffs {
            values: positional::write(spec, p, units)?,
            extra: zero_cross_lines(&ANGLE_CROSS),
        })
    }

    fn read(&self, spec: &StyleSpec, values: &[&str]) -> Result<Params, String> {
        positional::read(spec, values)
    }

    fn extra_keywords(&self) -> &'static [&'static str] {
        &["bb", "ba"]
    }

    fn parse_extra(
        &self,
        spec: &StyleSpec,
        keyword: &str,
        values: &[&str],
        _: &mut Params,
    ) -> Result<(), String> {
        check_zero_cross_line(spec, &ANGLE_CROSS, keyword, values)
    }
}

pub(crate) struct DihedralClass2Codec;
codec!(
    /// `K1 phi1 K2 phi2 K3 phi3`, and `mbt`/`ebt`/`at`/`aat`/`bb13` lines at
    /// zero.
    DIHEDRAL_CLASS2 = DihedralClass2Codec,
    None
);

impl LammpsCodec for DihedralClass2Codec {
    fn write(
        &self,
        spec: &StyleSpec,
        p: &Params,
        units: &UnitScale,
    ) -> Result<LammpsCoeffs, String> {
        Ok(LammpsCoeffs {
            values: positional::write(spec, p, units)?,
            extra: zero_cross_lines(&DIHEDRAL_CROSS),
        })
    }

    fn read(&self, spec: &StyleSpec, values: &[&str]) -> Result<Params, String> {
        positional::read(spec, values)
    }

    fn extra_keywords(&self) -> &'static [&'static str] {
        &["mbt", "ebt", "at", "aat", "bb13"]
    }

    fn parse_extra(
        &self,
        spec: &StyleSpec,
        keyword: &str,
        values: &[&str],
        _: &mut Params,
    ) -> Result<(), String> {
        check_zero_cross_line(spec, &DIHEDRAL_CROSS, keyword, values)
    }
}

// ── pair styles ─────────────────────────────────────────────────────────────

pub(crate) struct LjCutCodec;
codec!(
    /// `epsilon sigma`; `shift` is `pair_modify shift yes`; only the 12-6
    /// (`n`, `m` at their defaults: LAMMPS's Mie form is `mie/cut`).
    LJ_CUT = LjCutCodec,
    None
);

impl LammpsCodec for LjCutCodec {
    fn write(
        &self,
        spec: &StyleSpec,
        p: &Params,
        units: &UnitScale,
    ) -> Result<LammpsCoeffs, String> {
        Ok(LammpsCoeffs {
            values: positional::write(spec, p, units)?,
            extra: Vec::new(),
        })
    }

    fn read(&self, spec: &StyleSpec, values: &[&str]) -> Result<Params, String> {
        positional::read(spec, values)
    }

    fn check_style(&self, spec: &StyleSpec, style: &Params) -> Result<(), String> {
        let (n, m) = (
            style.get("n").unwrap_or(12.0),
            style.get("m").unwrap_or(6.0),
        );
        if (n, m) != (12.0, 6.0) {
            return Err(format!(
                "pair lj/cut with n = {n}, m = {m}: LAMMPS's lj/cut is 12-6 (its Mie form, \
                 mie/cut, is not written)"
            ));
        }
        positional::check_style_except(spec, style, &["shift", "n", "m"])
    }

    fn pair_modify(&self, spec: &StyleSpec, style: &Params) -> Result<Vec<String>, String> {
        let mut out = positional::pair_modify(None, spec, style)?;
        if style.get("shift").is_some_and(|s| s != 0.0) {
            out.push("shift yes".into());
        }
        Ok(out)
    }
}

pub(crate) struct LjCharmmCodec;
codec!(
    /// `epsilon sigma epsilon14 sigma14` (the 1-4 pair always written, the
    /// regular one when the row has none); `inner cutoff` on the style line.
    LJ_CHARMM = LjCharmmCodec,
    Some("lj/charmm/coul/charmm")
);

impl LammpsCodec for LjCharmmCodec {
    fn write(
        &self,
        spec: &StyleSpec,
        p: &Params,
        units: &UnitScale,
    ) -> Result<LammpsCoeffs, String> {
        positional::refuse_undeclared_params(spec, p)?;
        let need = |key: &str| {
            p.get(key)
                .ok_or_else(|| format!("{}: missing parameter `{key}`", what(spec)))
        };
        let (eps, sigma) = (need("epsilon")?, need("sigma")?);
        Ok(LammpsCoeffs {
            values: vec![
                energy(units, eps),
                Token::Real(units.apply(sigma, ParamDimension::LENGTH)),
                energy(units, p.get("epsilon14").unwrap_or(eps)),
                Token::Real(units.apply(p.get("sigma14").unwrap_or(sigma), ParamDimension::LENGTH)),
            ],
            extra: Vec::new(),
        })
    }

    /// `epsilon sigma [epsilon14 sigma14]`: LAMMPS's two-number form means
    /// the 1-4 pair is the regular one, stored as such.
    fn read(&self, spec: &StyleSpec, values: &[&str]) -> Result<Params, String> {
        match values.len() {
            2 => {
                let mut p = parse_named(spec, &["epsilon", "sigma"], values)?;
                p.set("epsilon14", p.get("epsilon").unwrap());
                p.set("sigma14", p.get("sigma").unwrap());
                Ok(p)
            }
            4 => positional::read(spec, values),
            n => Err(format!(
                "pair_coeff for lj/charmm/coul/charmm takes `epsilon sigma [epsilon14 sigma14]`, \
                 got {n} numbers"
            )),
        }
    }

    fn check_style(&self, spec: &StyleSpec, style: &Params) -> Result<(), String> {
        if OneFour::of(style)? == OneFour::Epsilon14 {
            return Err(
                "pair lj/charmm has one_four = \"epsilon14\" (its special_bonds 1-4 pairs at \
                 epsilon14/sigma14), which LAMMPS's lj/charmm/coul/charmm prices at the regular \
                 epsilon/sigma; LAMMPS reaches epsilon14/sigma14 only through dihedral charmm w"
                    .into(),
            );
        }
        positional::check_style_except(spec, style, &["inner", "one_four"])
    }

    fn style_args(
        &self,
        spec: &StyleSpec,
        style: &Params,
        units: &UnitScale,
    ) -> Result<Vec<Token>, String> {
        switch_args(spec, style, units)
    }

    fn parse_style_args(&self, spec: &StyleSpec, args: &[&str]) -> Result<Params, String> {
        parse_named(spec, &["inner", "cutoff"], args)
    }
}

/// `inner cutoff` of a CHARMM style, both required.
fn switch_args(spec: &StyleSpec, style: &Params, units: &UnitScale) -> Result<Vec<Token>, String> {
    ["inner", "cutoff"]
        .iter()
        .map(|key| {
            style
                .get(key)
                .map(|v| Token::Real(units.apply(v, ParamDimension::LENGTH)))
                .ok_or_else(|| {
                    format!(
                        "pair style '{}' has no '{key}': lj/charmm/coul/charmm needs its inner \
                         and outer switching cutoffs",
                        spec.name
                    )
                })
        })
        .collect()
}

pub(crate) struct LjClass2Codec;
codec!(
    /// `epsilon sigma`; LAMMPS's `lj/class2` mixes `sixthpower` whatever
    /// `pair_modify` says (`pair_lj_class2.cpp`, `init_one`).
    LJ_CLASS2 = LjClass2Codec,
    None
);

impl LammpsCodec for LjClass2Codec {
    fn write(
        &self,
        spec: &StyleSpec,
        p: &Params,
        units: &UnitScale,
    ) -> Result<LammpsCoeffs, String> {
        Ok(LammpsCoeffs {
            values: positional::write(spec, p, units)?,
            extra: Vec::new(),
        })
    }

    fn read(&self, spec: &StyleSpec, values: &[&str]) -> Result<Params, String> {
        positional::read(spec, values)
    }

    fn fixed_mixing(&self) -> Option<&'static str> {
        Some("sixthpower")
    }
}

/// The checks of a Coulomb style LAMMPS prices with its own constant.
fn check_coulomb(spec: &StyleSpec, style: &Params, carried: &[&str]) -> Result<(), String> {
    if let Some(delta) = style.get("delta").filter(|d| *d != 0.0) {
        return Err(format!(
            "pair {}: the buffer delta = {delta} (E = k·qq/(D·(r + delta))) has no LAMMPS \
             Coulomb style",
            spec.name
        ));
    }
    if let Some(d) = style.get("dielectric").filter(|d| *d != 1.0) {
        return Err(format!(
            "pair {}: dielectric = {d} is a LAMMPS input-script `dielectric` command, not a \
             coefficient this include can carry",
            spec.name
        ));
    }
    // The Coulomb constant is LAMMPS's own `qqr2e` for the `units` and is not
    // written; a field stating another (AMBER's 332.0522173) is priced by
    // LAMMPS at LAMMPS's, a documented difference of ~3e-5 relative.
    let mut all = vec!["coulomb", "dielectric", "delta"];
    all.extend_from_slice(carried);
    positional::check_style_except(spec, style, &all)
}

pub(crate) struct CoulCutCodec;
codec!(
    /// The `coul/cut` half of `lj/cut/coul/cut` (or a `hybrid/overlay`).
    COUL_CUT = CoulCutCodec,
    Some("coul/cut")
);

impl LammpsCodec for CoulCutCodec {
    fn write(&self, spec: &StyleSpec, _: &Params, _: &UnitScale) -> Result<LammpsCoeffs, String> {
        Err(no_rows(spec))
    }

    fn read(&self, spec: &StyleSpec, _: &[&str]) -> Result<Params, String> {
        Err(no_rows(spec))
    }

    fn check_style(&self, spec: &StyleSpec, style: &Params) -> Result<(), String> {
        check_coulomb(spec, style, &[])
    }
}

pub(crate) struct CoulLongCodec;
codec!(
    /// The real-space `coul/long` of `lj/cut/coul/long`: its Ewald
    /// parameters are the input script's `kspace_style`, an accuracy, so a
    /// field stating its own is refused.
    COUL_LONG = CoulLongCodec,
    Some("coul/long")
);

impl LammpsCodec for CoulLongCodec {
    fn write(&self, spec: &StyleSpec, _: &Params, _: &UnitScale) -> Result<LammpsCoeffs, String> {
        Err(no_rows(spec))
    }

    fn read(&self, spec: &StyleSpec, _: &[&str]) -> Result<Params, String> {
        Err(no_rows(spec))
    }

    fn check_style(&self, spec: &StyleSpec, style: &Params) -> Result<(), String> {
        if let Some(key) = ["alpha", "order", "grid_x", "grid_y", "grid_z"]
            .into_iter()
            .find(|k| style.get(k).is_some())
        {
            return Err(format!(
                "pair coul/long/pme states its Ewald '{key}': molrs's smooth PME is not LAMMPS's \
                 PPPM, whose kspace_style sets the mesh by an accuracy; only the real-space \
                 lj/cut/coul/long is written"
            ));
        }
        check_coulomb(spec, style, &[])
    }
}

pub(crate) struct CoulCharmmCodec;
codec!(
    /// The `coul/charmm` half of `lj/charmm/coul/charmm`.
    COUL_CHARMM = CoulCharmmCodec,
    Some("coul/charmm")
);

impl LammpsCodec for CoulCharmmCodec {
    fn write(&self, spec: &StyleSpec, _: &Params, _: &UnitScale) -> Result<LammpsCoeffs, String> {
        Err(no_rows(spec))
    }

    fn read(&self, spec: &StyleSpec, _: &[&str]) -> Result<Params, String> {
        Err(no_rows(spec))
    }

    fn check_style(&self, spec: &StyleSpec, style: &Params) -> Result<(), String> {
        check_coulomb(spec, style, &["inner"])
    }

    fn style_args(
        &self,
        spec: &StyleSpec,
        style: &Params,
        units: &UnitScale,
    ) -> Result<Vec<Token>, String> {
        switch_args(spec, style, units)
    }

    fn parse_style_args(&self, spec: &StyleSpec, args: &[&str]) -> Result<Params, String> {
        parse_named(spec, &["inner", "cutoff"], args)
    }
}

// ── cmap: fix cmap ──────────────────────────────────────────────────────────

pub(crate) struct FixCmapCodec;
codec!(
    /// `cmap charmm` is LAMMPS's `fix cmap`: a grid file the include names,
    /// written by `LammpsForcefieldWriter::write_cmap_str`, not a coefficient line.
    FIX_CMAP = FixCmapCodec,
    Some("cmap")
);

impl LammpsCodec for FixCmapCodec {
    fn write(&self, spec: &StyleSpec, _: &Params, _: &UnitScale) -> Result<LammpsCoeffs, String> {
        Err(format!(
            "{}: LAMMPS's fix cmap reads a grid file (LammpsForcefieldWriter::write_cmap_str), not a \
             coefficient line",
            what(spec)
        ))
    }

    fn read(&self, spec: &StyleSpec, _: &[&str]) -> Result<Params, String> {
        Err(format!(
            "{}: LAMMPS's fix cmap reads a grid file (read_lammps_cmap_str)",
            what(spec)
        ))
    }
}
