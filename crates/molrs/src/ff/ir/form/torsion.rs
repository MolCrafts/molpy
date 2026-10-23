//! Torsion algebra: the exact linear maps between every proper-torsion and
//! improper form molrs has, and their registration as the force-field IR's
//! `torsion` form family (`builtin_forms` in `ff::ir`).
//!
//! Every Class-I torsion form is a finite Fourier series in the dihedral angle
//! φ (LAMMPS's signed φ of the atoms I-J-K-L as stored, the angle every molrs
//! dihedral kernel computes):
//!
//! ```text
//! E(φ) = Σₙ₌₀ aₙ cos nφ + bₙ sin nφ           (FourierSeries)
//! ```
//!
//! Each form has an exact **embedding** into that series (`to_series`: the
//! energy is the same function of φ, constant included) and an exact
//! **projection** back (`from_series`), which reproduces the series — every
//! coefficient, the constant `a₀` included — or refuses with a
//! [`TorsionRefusal`] naming the condition that prevents it. The forms, in
//! the force-field IR (adopts the LAMMPS standard; parameters as stored, phases in
//! degrees):
//!
//! | form | style | energy | image condition |
//! |---|---|---|---|
//! | [`PeriodicForm`] | `dihedral periodic` (LAMMPS `fourier`) | Σₘ kₘ[1 + cos(nₘφ − γₘ)] | none: every series (a periodicity-0 term holds the constant) |
//! | [`CharmmForm`] | `dihedral charmm` | k[1 + cos(nφ − d)] | one order, `√(aₙ² + bₙ²) = |a₀|` |
//! | [`CosineTermForm`] | `improper periodic` | k[1 + cos(nφ − γ)] | as charmm |
//! | [`SignedCosineForm`] | `dihedral harmonic`, `improper cvff` | k[1 + d cos nφ] | one order, `bₙ = 0`, `|aₙ| = |a₀|` |
//! | [`OplsForm`] | `dihedral opls` | ½Σ kₙ[1 ∓ cos nφ] | `bₙ = 0`, n ≤ 4, `a₀ = a₁ − a₂ + a₃ − a₄` |
//! | [`Class2Form`] | `dihedral class2` (torsion part) | Σₙ₌₁³ kₙ[1 − cos(nφ − φₙ)] | n ≤ 3, `a₀ = Σ ±√(aₙ² + bₙ²)` |
//! | [`MultiHarmonicForm`] | `dihedral multi/harmonic` | Σₙ₌₁⁵ Aₙ cosⁿ⁻¹φ | `bₙ = 0`, n ≤ 4 |
//! | [`NHarmonicForm`] | `dihedral nharmonic` | Σᵢ₌₁ᴺ Aᵢ cosⁱ⁻¹φ | `bₙ = 0` |
//! | [`RyckaertBellemansForm`] | `dihedral rb` (GROMACS funct 3, OpenMM `RBTorsionForce`) | Σₙ₌₀⁵ Cₙ cosⁿ(φ − 180°) | `bₙ = 0`, n ≤ 5 |
//!
//! Every periodicity must be an integer ([`TorsionRefusal::NonIntegerPeriodicity`]);
//! LAMMPS requires that of every style above.
//!
//! # The constant term
//!
//! `a₀` is part of the energy, so it is part of every map: `to_series` is the
//! exact energy, constant included, and `from_series` reproduces it or
//! refuses ([`TorsionRefusal::ConstantOffset`]). The polynomial forms
//! (multi/harmonic, nharmonic, RB) carry any constant; `dihedral periodic`
//! carries any constant through a periodicity-0 term (`k₀[1 + cos 0] = 2k₀`);
//! the other forms fix their constant by their other parameters, which is
//! their image condition above. A single-term form takes `k = a₀`, so a
//! negative `k` survives the round trip.
//!
//! A constant shifts no force: [`FourierSeries::canonical`] drops it, and
//! two torsions are the same physics iff their canonical series are equal.
//! Converting *up to* a constant is a fit, not a projection:
//! [`ForceField::fit_form`](crate::ff::forcefield::ForceField::fit_form)
//! with a free offset reports it as the fit's offset.
//!
//! # Exactness
//!
//! Phases are degrees, and multiples of 90° are evaluated exactly (`cos 180°`
//! is `−1`, `sin 180°` is `0`, not `1.2e−16`), so a 0°/180° phase lands as
//! `bₙ = 0` exactly and passes a sine-free projection. The `n ≥ 1`
//! coefficient tests are exact (`!= 0.0`); a caller holding rounded input
//! (GROMACS prints five decimals) chops it first with
//! [`FourierSeries::chopped`]. The constant is a sum of products, so it is
//! compared to [`CONSTANT_RTOL`] of the series' scale.
//!
//! # Outside the image
//!
//! - `improper harmonic`, `K(|φ| − χ0)²`, and LAMMPS `dihedral quadratic`,
//!   `K(φ − φ0)²`, are not finite Fourier series
//!   ([`TorsionRefusal::NotAFourierForm`]); `improper harmonic` registers no
//!   form codec. A periodic improper and a harmonic one agree to second order
//!   about the minimum only: `K = n²·k/2` in LAMMPS's un-halved `K(χ − χ0)²`
//!   (`2k` at `n = 2`; the `k_h = n²k` of a ½-form harmonic) — see
//!   [`CosineTermForm::second_order_harmonic`] and
//!   [`ImproperHarmonicForm::second_order_periodic`]. Their quartic terms differ
//!   by `−k n⁴ δ⁴/24`; `fit_form` gives the projection under a declared
//!   metric with its residual.
//! - `dihedral charmm`'s `w` weights the 1-4 pair of the dihedral; it is not a
//!   torsion parameter, so a row with `w ≠ 0` does not embed
//!   ([`TorsionRefusal::OneFourWeight`]) and a projection onto charmm gives
//!   `w = 0`.
//! - `dihedral class2`'s cross terms (`mbt`, `ebt`, `at`, `aat`, `bb13`) couple
//!   the torsion to bonds and angles; they are outside the Class-I IR (and
//!   molrs stores none of them). [`Class2Form`] is the torsion part only.
//! - The out-of-plane impropers (`improper fourier` / molrs `uff_inversion`,
//!   `mmff_oop`) are functions of a Wilson angle, not of a dihedral.
//!
//! Units: every map is linear and unit-agnostic (energy in = energy out).
//! These live in `ff::forcefield`, not `ff::potential`, so readers and writers
//! never import from a kernel module; the kernels' agreement with the algebra
//! is tested here, on the registered kernels.

use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign};

use super::Params;
use crate::ff::ir::{FormCodec, FormRefusal, TypeParams};

/// The constant term's rounding allowance, relative to the series' scale
/// ([`FourierSeries::scale`]): `a₀` is a sum of products, so a projection
/// that fixes it compares it to this, never exactly. The `n ≥ 1`
/// coefficients are compared exactly.
pub const CONSTANT_RTOL: f64 = 1e-12;

// ── the series ──────────────────────────────────────────────────────────────

/// `E(φ) = Σₙ aₙ cos nφ + bₙ sin nφ`, `n = 0, 1, …`, φ in radians when
/// evaluated. `a[0]` is the constant term; `b[0]` multiplies `sin 0 = 0` and
/// is held at zero. Equality is exact on every coefficient, trailing zeros
/// aside.
#[derive(Debug, Clone, Default)]
pub struct FourierSeries {
    a: Vec<f64>,
    b: Vec<f64>,
}

impl FourierSeries {
    /// The zero series.
    pub fn zero() -> Self {
        Self::default()
    }

    /// The series with cosine coefficients `a` and sine coefficients `b`,
    /// both indexed from `n = 0` (the shorter is padded with zeros; `b[0]`,
    /// which multiplies `sin 0`, is dropped).
    pub fn new(a: Vec<f64>, b: Vec<f64>) -> Self {
        let mut s = Self::zero();
        for (n, &an) in a.iter().enumerate() {
            s.add_cos(n, an);
        }
        for (n, &bn) in b.iter().enumerate().skip(1) {
            s.add_sin(n, bn);
        }
        s
    }

    /// `aₙ` (zero beyond the stored length).
    pub fn a(&self, n: usize) -> f64 {
        self.a.get(n).copied().unwrap_or(0.0)
    }

    /// `bₙ` (zero beyond the stored length, and at `n = 0`).
    pub fn b(&self, n: usize) -> f64 {
        self.b.get(n).copied().unwrap_or(0.0)
    }

    /// The constant term `a₀`.
    pub fn constant(&self) -> f64 {
        self.a(0)
    }

    /// The highest `n ≥ 1` with `aₙ ≠ 0` or `bₙ ≠ 0`; 0 for a constant.
    pub fn order(&self) -> usize {
        (1..self.a.len())
            .rev()
            .find(|&n| self.a[n] != 0.0 || self.b[n] != 0.0)
            .unwrap_or(0)
    }

    /// The orders `n ≥ 1` with a non-zero coefficient, ascending.
    pub fn orders(&self) -> Vec<usize> {
        (1..=self.order())
            .filter(|&n| self.a(n) != 0.0 || self.b(n) != 0.0)
            .collect()
    }

    /// The energy at `phi` (radians).
    pub fn energy(&self, phi: f64) -> f64 {
        self.a
            .iter()
            .zip(&self.b)
            .enumerate()
            .map(|(n, (&an, &bn))| {
                let (s, c) = (n as f64 * phi).sin_cos();
                an * c + bn * s
            })
            .sum()
    }

    /// The canonical form: constant dropped, trailing zeros trimmed. Two
    /// torsions are the same physics iff their canonical series are equal.
    pub fn canonical(&self) -> Self {
        let len = self.order() + 1;
        let mut a = self.a.clone();
        let mut b = self.b.clone();
        a.resize(len, 0.0);
        b.resize(len, 0.0);
        a[0] = 0.0;
        Self { a, b }
    }

    /// Every coefficient with `|c| ≤ tol` set to exactly zero — for input
    /// rounded on print (the projections test coefficients exactly).
    pub fn chopped(&self, tol: f64) -> Self {
        let chop = |c: &f64| if c.abs() <= tol { 0.0 } else { *c };
        Self {
            a: self.a.iter().map(chop).collect(),
            b: self.b.iter().map(chop).collect(),
        }
    }

    /// `max |Δaₙ|, |Δbₙ|` over every order, the constant included.
    pub fn max_abs_diff(&self, other: &Self) -> f64 {
        let len = self.a.len().max(other.a.len());
        (0..len)
            .map(|n| {
                (self.a(n) - other.a(n))
                    .abs()
                    .max((self.b(n) - other.b(n)).abs())
            })
            .fold(0.0, f64::max)
    }

    /// `Σₙ |aₙ| + |bₙ|`, the constant included: the scale a rounding
    /// allowance is relative to.
    pub fn scale(&self) -> f64 {
        self.a.iter().chain(&self.b).map(|c| c.abs()).sum()
    }

    fn grow(&mut self, n: usize) {
        if self.a.len() <= n {
            self.a.resize(n + 1, 0.0);
            self.b.resize(n + 1, 0.0);
        }
    }

    fn add_cos(&mut self, n: usize, c: f64) {
        self.grow(n);
        self.a[n] += c;
    }

    fn add_sin(&mut self, n: usize, s: f64) {
        if n > 0 {
            self.grow(n);
            self.b[n] += s;
        }
    }
}

impl PartialEq for FourierSeries {
    fn eq(&self, other: &Self) -> bool {
        let len = self.a.len().max(other.a.len());
        (0..len).all(|n| self.a(n) == other.a(n) && self.b(n) == other.b(n))
    }
}

impl AddAssign<&FourierSeries> for FourierSeries {
    fn add_assign(&mut self, rhs: &FourierSeries) {
        for n in 0..rhs.a.len() {
            self.add_cos(n, rhs.a[n]);
            self.add_sin(n, rhs.b[n]);
        }
    }
}

impl Add for FourierSeries {
    type Output = FourierSeries;
    fn add(mut self, rhs: FourierSeries) -> FourierSeries {
        self += &rhs;
        self
    }
}

/// Several rows on one quadruple are one torsion: their series summed.
impl Sum for FourierSeries {
    fn sum<I: Iterator<Item = FourierSeries>>(iter: I) -> Self {
        iter.fold(Self::zero(), Add::add)
    }
}

// ── refusals ────────────────────────────────────────────────────────────────

/// Why a conversion is not exact, naming the term that prevents it.
#[derive(Debug, Clone, PartialEq)]
pub enum TorsionRefusal {
    /// `bₙ ≠ 0` — a phase off {0°, 180°} — where the target is cosines only.
    SineTerm { n: usize, b: f64 },
    /// An order above the target's highest.
    OrderTooHigh { n: usize, max: usize },
    /// More than one order for a single-term target.
    MultiTerm { orders: Vec<usize> },
    /// The constant term `a₀` is not the one the target's other parameters
    /// fix (`implied`).
    ConstantOffset { constant: f64, implied: f64 },
    /// A periodicity that is not an integer (or not finite).
    NonIntegerPeriodicity { periodicity: f64 },
    /// A `dihedral harmonic` / `improper cvff` sign that is not ±1.
    InvalidSign { sign: f64 },
    /// A `dihedral charmm` 1-4 weight `w ≠ 0`, which prices a pair, not φ.
    OneFourWeight { w: f64 },
    /// A form that is not a finite Fourier series in φ.
    NotAFourierForm {
        style: &'static str,
        reason: &'static str,
    },
    /// A second-order relation asked of a term with no curvature (k = 0 or n = 0).
    NoCurvature,
    /// A parameter the style requires is absent.
    MissingParam { style: &'static str, key: String },
}

impl fmt::Display for TorsionRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SineTerm { n, b } => write!(
                f,
                "sin({n}φ) coefficient {b} ≠ 0 (a phase other than 0° or 180°): \
                 the target form has cosines only"
            ),
            Self::OrderTooHigh { n, max } => write!(
                f,
                "order n = {n} is above the target form's highest order {max}"
            ),
            Self::MultiTerm { orders } => write!(
                f,
                "orders {orders:?} are present: the target form holds one term"
            ),
            Self::ConstantOffset { constant, implied } => write!(
                f,
                "the constant term a₀ = {constant} is not the {implied} the target form's \
                 terms fix (a constant shifts no force: fit_form with a free offset converts it)"
            ),
            Self::NonIntegerPeriodicity { periodicity } => {
                write!(f, "periodicity {periodicity} is not an integer")
            }
            Self::InvalidSign { sign } => write!(f, "sign {sign} is not ±1"),
            Self::OneFourWeight { w } => write!(
                f,
                "the 1-4 weight w = {w} prices the dihedral's 1-4 pair, which no other \
                 torsion form holds"
            ),
            Self::NotAFourierForm { style, reason } => {
                write!(f, "{style} is not a Fourier form: {reason}")
            }
            Self::NoCurvature => write!(f, "the term has no curvature (k = 0 or n = 0)"),
            Self::MissingParam { style, key } => write!(f, "{style}: missing param `{key}`"),
        }
    }
}

impl std::error::Error for TorsionRefusal {}

impl From<TorsionRefusal> for FormRefusal {
    fn from(e: TorsionRefusal) -> Self {
        FormRefusal::new(e.to_string())
    }
}

type Result<T> = std::result::Result<T, TorsionRefusal>;

// ── helpers ─────────────────────────────────────────────────────────────────

/// `(cos d, sin d)` of `d` degrees, exact at multiples of 90°.
fn cos_sin_deg(deg: f64) -> (f64, f64) {
    let r = deg.rem_euclid(360.0);
    if r == 0.0 {
        (1.0, 0.0)
    } else if r == 90.0 {
        (0.0, 1.0)
    } else if r == 180.0 {
        (-1.0, 0.0)
    } else if r == 270.0 {
        (0.0, -1.0)
    } else {
        let (s, c) = r.to_radians().sin_cos();
        (c, s)
    }
}

/// The phase, in degrees on (−180°, 180°], of `c·cos x + s·sin x = A cos(x − phase)`;
/// exact on the axes, 0° for a zero amplitude.
fn phase_deg(s: f64, c: f64) -> f64 {
    if s == 0.0 {
        if c >= 0.0 { 0.0 } else { 180.0 }
    } else if c == 0.0 {
        if s > 0.0 { 90.0 } else { -90.0 }
    } else {
        s.atan2(c).to_degrees()
    }
}

/// `|n|` and the sign of `n` for an integral periodicity.
fn order_of(periodicity: f64) -> Result<(usize, f64)> {
    if !periodicity.is_finite() || periodicity.fract() != 0.0 {
        return Err(TorsionRefusal::NonIntegerPeriodicity { periodicity });
    }
    let sign = if periodicity < 0.0 { -1.0 } else { 1.0 };
    Ok((periodicity.abs() as usize, sign))
}

/// The series has only cosines, of order ≤ `max`.
fn require_cosines(s: &FourierSeries, max: Option<usize>) -> Result<()> {
    for n in 1..=s.order() {
        if let Some(max) = max
            && n > max
            && (s.a(n) != 0.0 || s.b(n) != 0.0)
        {
            return Err(TorsionRefusal::OrderTooHigh { n, max });
        }
        if s.b(n) != 0.0 {
            return Err(TorsionRefusal::SineTerm { n, b: s.b(n) });
        }
    }
    Ok(())
}

/// The one order `n ≥ 1` of a single-term series (`None` for a constant).
fn single_order(s: &FourierSeries) -> Result<Option<usize>> {
    let orders = s.orders();
    match orders.as_slice() {
        [] => Ok(None),
        [n] => Ok(Some(*n)),
        _ => Err(TorsionRefusal::MultiTerm { orders }),
    }
}

/// `Σₖ pₖ cosᵏφ` as a Fourier series: `cosᵏφ = 2⁻ᵏ Σⱼ C(k,j) cos((k − 2j)φ)`.
fn cos_poly_to_series(p: &[f64]) -> FourierSeries {
    let mut s = FourierSeries::zero();
    for (k, &pk) in p.iter().enumerate() {
        if pk == 0.0 {
            continue;
        }
        let scale = pk * 0.5_f64.powi(k as i32);
        let mut binom = 1.0; // C(k, j)
        for j in 0..=k {
            s.add_cos(k.abs_diff(2 * j), scale * binom);
            binom = binom * (k - j) as f64 / (j + 1) as f64;
        }
    }
    s
}

/// A cosine-only series as `Σₖ pₖ cosᵏφ`, `cos nφ = Tₙ(cos φ)` (Chebyshev),
/// with at least `min_len` coefficients.
fn series_to_cos_poly(s: &FourierSeries, min_len: usize) -> Vec<f64> {
    let order = s.order();
    let mut p = vec![0.0; (order + 1).max(min_len)];
    // T₀ = 1, T₁ = x, Tₙ₊₁ = 2x Tₙ − Tₙ₋₁ (integer coefficients, exact).
    let (mut prev, mut cur) = (vec![1.0], vec![0.0, 1.0]);
    p[0] += s.a(0);
    for n in 1..=order {
        let an = s.a(n);
        if an != 0.0 {
            for (i, &t) in cur.iter().enumerate() {
                p[i] += an * t;
            }
        }
        let mut next = vec![0.0; cur.len() + 1];
        for (i, &t) in cur.iter().enumerate() {
            next[i + 1] += 2.0 * t;
        }
        for (i, &t) in prev.iter().enumerate() {
            next[i] -= t;
        }
        prev = std::mem::replace(&mut cur, next);
    }
    p
}

/// Whether the constant `a₀` of `s` is `implied`, to [`CONSTANT_RTOL`] of
/// the series' scale.
fn constant_is(s: &FourierSeries, implied: f64) -> Result<()> {
    let constant = s.constant();
    if (constant - implied).abs() <= CONSTANT_RTOL * s.scale() {
        Ok(())
    } else {
        Err(TorsionRefusal::ConstantOffset { constant, implied })
    }
}

/// The order `n ≥ 1` of largest amplitude `√(aₙ² + bₙ²)` (the lowest on a
/// tie), or `None` for a constant.
fn dominant_order(s: &FourierSeries) -> Option<usize> {
    s.orders().into_iter().fold(None, |best, n| match best {
        Some(m) if s.a(m).hypot(s.b(m)) >= s.a(n).hypot(s.b(n)) => Some(m),
        _ => Some(n),
    })
}

/// `s` with its sine terms and every order above `max` dropped, the constant
/// kept: the nearest cosine-only series of order ≤ `max` on a full period.
fn cosines_up_to(s: &FourierSeries, max: usize) -> FourierSeries {
    FourierSeries::new((0..=max.min(s.order())).map(|n| s.a(n)).collect(), vec![])
}

/// A required numeric param of a row.
fn need(p: &Params, style: &'static str, key: &str) -> Result<f64> {
    p.get(key)
        .and_then(|v| v.as_num())
        .ok_or_else(|| TorsionRefusal::MissingParam {
            style,
            key: key.to_owned(),
        })
}

/// An optional numeric param of a row of the built-in `style` (`"category
/// name"`): stated, else its spec's declared default — of the indexed
/// family a member `<base><m>` belongs to.
///
/// # Panics
///
/// When the spec declares no default for `key`: a built-in's codec reads an
/// optional parameter only where its spec makes it one.
fn or_default(p: &Params, style: &'static str, key: &str) -> f64 {
    p.get(key).and_then(|v| v.as_num()).unwrap_or_else(|| {
        let (category, name) = style.split_once(' ').expect("`category name`");
        let spec = crate::ff::ir::builtin_style(category, name)
            .unwrap_or_else(|| panic!("{style} is a built-in style"));
        spec.param(key)
            .or_else(|| {
                spec.params.iter().find(|d| {
                    d.indexed
                        && key
                            .strip_prefix(d.name.as_ref())
                            .is_some_and(|m| m.parse::<usize>().is_ok())
                })
            })
            .and_then(|d| d.default.as_ref())
            .and_then(|v| v.as_num())
            .unwrap_or_else(|| panic!("{style} declares a default for `{key}`"))
    })
}

// ── single cosine terms ─────────────────────────────────────────────────────

/// `k[1 + cos(nφ − γ)]`, `γ` = `phase` in degrees: one term of `dihedral
/// periodic` (LAMMPS `fourier`), and the whole of molrs `improper periodic`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CosineTermForm {
    pub k: f64,
    pub periodicity: f64,
    /// degrees
    pub phase: f64,
}

impl CosineTermForm {
    /// The energy at `phi` (radians), straight from the formula.
    pub fn energy(&self, phi: f64) -> f64 {
        self.k * (1.0 + (self.periodicity * phi - self.phase.to_radians()).cos())
    }

    /// `k + k cos γ cos nφ + k sin γ sin nφ` (`n < 0`: `cos` is even, `sin` odd).
    pub fn to_series(&self) -> Result<FourierSeries> {
        let mut s = FourierSeries::zero();
        self.add_to(&mut s)?;
        Ok(s)
    }

    fn add_to(&self, s: &mut FourierSeries) -> Result<()> {
        let (n, sign) = order_of(self.periodicity)?;
        let (c, sn) = cos_sin_deg(self.phase);
        s.add_cos(0, self.k);
        s.add_cos(n, self.k * c);
        s.add_sin(n, sign * self.k * sn);
        Ok(())
    }

    /// `aₙ cos nφ + bₙ sin nφ` as `k cos(nφ − γ)`: `k = √(a² + b²) ≥ 0`,
    /// `γ ∈ (−180°, 180°]`.
    fn from_coefficients(n: usize, a: f64, b: f64) -> Self {
        Self {
            k: a.hypot(b),
            periodicity: n as f64,
            phase: phase_deg(b, a),
        }
    }

    /// The one term of a single-order series, constant included: `k = a₀`
    /// (so its sign is the series'), `γ` from `(aₙ, bₙ) = k(cos γ, sin γ)`. A
    /// constant series is a periodicity-0 term `k = a₀/2` (`k = 0, n = 1`
    /// for the zero series).
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::MultiTerm`] for more than one order;
    /// [`TorsionRefusal::ConstantOffset`] when `√(aₙ² + bₙ²) ≠ |a₀|`.
    pub fn from_series(s: &FourierSeries) -> Result<Self> {
        let k = s.constant();
        let Some(n) = single_order(s)? else {
            return Ok(Self::constant(k));
        };
        let (a, b) = (s.a(n), s.b(n));
        let amplitude = a.hypot(b);
        constant_is(s, if k < 0.0 { -amplitude } else { amplitude })?;
        let phase = if k < 0.0 {
            phase_deg(-b, -a)
        } else {
            phase_deg(b, a)
        };
        Ok(Self {
            k,
            periodicity: n as f64,
            phase,
        })
    }

    /// The constant `c` as a term: `n = 0`, `k = c/2`; the zero term
    /// `k = 0, n = 1` for `c = 0`.
    fn constant(c: f64) -> Self {
        Self {
            k: c / 2.0,
            periodicity: if c == 0.0 { 1.0 } else { 0.0 },
            phase: 0.0,
        }
    }

    /// The term of the dominant order of `s` (`k ≥ 0`), its constant and
    /// every other order dropped: the starting point of a fit.
    pub fn nearest(s: &FourierSeries) -> Self {
        match dominant_order(s) {
            Some(n) => Self::from_coefficients(n, s.a(n), s.b(n)),
            None => Self::constant(s.constant()),
        }
    }

    /// The `improper periodic` row `k`, `periodicity`, `phase` (absent: the
    /// spec's default, 0).
    pub fn from_params(p: &Params) -> Result<Self> {
        Ok(Self {
            k: need(p, "improper periodic", "k")?,
            periodicity: need(p, "improper periodic", "periodicity")?,
            phase: or_default(p, "improper periodic", "phase"),
        })
    }

    pub fn to_params(&self) -> Params {
        Params::from_iter([
            ("k", self.k),
            ("periodicity", self.periodicity),
            ("phase", self.phase),
        ])
    }

    /// The harmonic improper `K(χ − χ0)²` (LAMMPS's un-halved `K`) that
    /// agrees with this term to second order about a minimum:
    /// `K = |k| n²/2`, `χ0` the minimum `φ*` (`nφ* − γ = 180°` for `k > 0`,
    /// `0°` for `k < 0`) nearest 0, as `|φ*|`. The quartic terms differ by
    /// `−|k| n⁴ δ⁴/24`.
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::NoCurvature`] for `k = 0` or `n = 0`;
    /// [`TorsionRefusal::NonIntegerPeriodicity`].
    pub fn second_order_harmonic(&self) -> Result<ImproperHarmonicForm> {
        let (n, _) = order_of(self.periodicity)?;
        if n == 0 || self.k == 0.0 {
            return Err(TorsionRefusal::NoCurvature);
        }
        let shift = if self.k > 0.0 { 180.0 } else { 0.0 };
        let period = 360.0 / n as f64;
        let base = ((self.phase + shift) / self.periodicity).rem_euclid(period);
        Ok(ImproperHarmonicForm {
            k: self.k.abs() * (n * n) as f64 / 2.0,
            chi0: base.min(period - base),
        })
    }
}

/// `k[1 + d cos nφ]`, `d` = `sign` ∈ {+1, −1}: LAMMPS `dihedral harmonic` and
/// `improper cvff` (`K d n`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignedCosineForm {
    pub k: f64,
    pub sign: f64,
    pub periodicity: f64,
}

impl SignedCosineForm {
    /// The energy at `phi` (radians), straight from the formula.
    pub fn energy(&self, phi: f64) -> f64 {
        self.k * (1.0 + self.sign * (self.periodicity * phi).cos())
    }

    /// `k + k d cos nφ`.
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::InvalidSign`], [`TorsionRefusal::NonIntegerPeriodicity`].
    pub fn to_series(&self) -> Result<FourierSeries> {
        if self.sign != 1.0 && self.sign != -1.0 {
            return Err(TorsionRefusal::InvalidSign { sign: self.sign });
        }
        let (n, _) = order_of(self.periodicity)?;
        let mut s = FourierSeries::zero();
        s.add_cos(0, self.k);
        s.add_cos(n, self.k * self.sign);
        Ok(s)
    }

    /// `a₀ + aₙ cos nφ` as `k = a₀`, `d = aₙ/a₀`, constant included. A
    /// constant series is `n = 0, d = 1, k = a₀/2` (`k = 0, n = 1` for zero).
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::MultiTerm`], [`TorsionRefusal::SineTerm`],
    /// [`TorsionRefusal::ConstantOffset`] when `|aₙ| ≠ |a₀|`.
    pub fn from_series(s: &FourierSeries) -> Result<Self> {
        let k = s.constant();
        let Some(n) = single_order(s)? else {
            let c = CosineTermForm::constant(k);
            return Ok(Self {
                k: c.k,
                sign: 1.0,
                periodicity: c.periodicity,
            });
        };
        require_cosines(s, None)?;
        let a = s.a(n);
        constant_is(s, if k < 0.0 { -a.abs() } else { a.abs() })?;
        Ok(Self {
            k,
            sign: if (a < 0.0) == (k < 0.0) { 1.0 } else { -1.0 },
            periodicity: n as f64,
        })
    }

    /// `k = |aₙ|`, `d = sign(aₙ)` of the dominant order of `s`, everything
    /// else dropped: the starting point of a fit.
    pub fn nearest(s: &FourierSeries) -> Self {
        match dominant_order(s) {
            Some(n) => Self {
                k: s.a(n).abs(),
                sign: if s.a(n) < 0.0 { -1.0 } else { 1.0 },
                periodicity: n as f64,
            },
            None => Self {
                k: 0.0,
                sign: 1.0,
                periodicity: 1.0,
            },
        }
    }

    /// The `dihedral harmonic` / `improper cvff` row `k`, `sign`,
    /// `periodicity` (all required); `style` names it in a refusal.
    pub fn from_params(style: &'static str, p: &Params) -> Result<Self> {
        Ok(Self {
            k: need(p, style, "k")?,
            sign: need(p, style, "sign")?,
            periodicity: need(p, style, "periodicity")?,
        })
    }

    pub fn to_params(&self) -> Params {
        Params::from_iter([
            ("k", self.k),
            ("sign", self.sign),
            ("periodicity", self.periodicity),
        ])
    }
}

// ── proper forms ────────────────────────────────────────────────────────────

/// molrs `dihedral periodic` (LAMMPS `dihedral fourier`):
/// `Σₘ kₘ[1 + cos(nₘφ − γₘ)]`. Every series has this form, and its
/// canonical one is the force-field IR's canonical torsion
/// ([`PeriodicForm::from_series`]).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PeriodicForm {
    pub terms: Vec<CosineTermForm>,
}

impl PeriodicForm {
    /// The sum of the terms' series.
    pub fn to_series(&self) -> Result<FourierSeries> {
        let mut s = FourierSeries::zero();
        for t in &self.terms {
            t.add_to(&mut s)?;
        }
        Ok(s)
    }

    /// The canonical `dihedral periodic` of `s`, constant included: one term
    /// per order `n ≥ 1` present, ascending, `k > 0`, `γ ∈ (−180°, 180°]`,
    /// preceded by a periodicity-0 term `k₀ = (a₀ − Σkₘ)/2`, `γ₀ = 0` when
    /// the constant is not the terms' own (beyond [`CONSTANT_RTOL`]). Never
    /// refuses.
    pub fn from_series(s: &FourierSeries) -> Self {
        let mut terms: Vec<CosineTermForm> = s
            .orders()
            .into_iter()
            .map(|n| CosineTermForm::from_coefficients(n, s.a(n), s.b(n)))
            .collect();
        let rest = s.constant() - terms.iter().map(|t| t.k).sum::<f64>();
        if rest.abs() > CONSTANT_RTOL * s.scale() {
            terms.insert(0, CosineTermForm::constant(rest));
        }
        Self { terms }
    }

    /// Whether the terms are already [`from_series`](Self::from_series)'s
    /// spelling: ascending integral periodicities, at most one periodicity-0
    /// term (first, `γ = 0`, `k ≠ 0`), every other `k > 0` and
    /// `γ ∈ (−180°, 180°]`; or the single zero term.
    pub fn is_canonical(&self) -> bool {
        let zero = CosineTermForm::constant(0.0);
        if self.terms.is_empty() || self.terms == [zero] {
            return true;
        }
        let ascending = self
            .terms
            .windows(2)
            .all(|w| w[0].periodicity < w[1].periodicity);
        ascending
            && self.terms.iter().all(|t| {
                t.periodicity.fract() == 0.0
                    && if t.periodicity == 0.0 {
                        t.phase == 0.0 && t.k != 0.0
                    } else {
                        t.periodicity > 0.0 && t.k > 0.0 && t.phase > -180.0 && t.phase <= 180.0
                    }
            })
    }

    /// The `dihedral periodic` row: the indexed terms `k<m>`,
    /// `periodicity<m>`, `phase<m>` from `m = 1`, else the one-term spelling
    /// `k`, `periodicity`, `phase` (an absent phase is 0).
    pub fn from_params(p: &Params) -> Result<Self> {
        const WHAT: &str = "dihedral periodic";
        let mut terms = Vec::new();
        let mut m = 1;
        while let Some(k) = p.get(&format!("k{m}")).and_then(|v| v.as_num()) {
            terms.push(CosineTermForm {
                k,
                periodicity: need(p, WHAT, &format!("periodicity{m}"))?,
                phase: or_default(p, WHAT, &format!("phase{m}")),
            });
            m += 1;
        }
        if terms.is_empty() {
            terms.push(CosineTermForm {
                k: need(p, WHAT, "k")?,
                periodicity: need(p, WHAT, "periodicity")?,
                phase: or_default(p, WHAT, "phase"),
            });
        }
        Ok(Self { terms })
    }

    /// The indexed spelling `k<m>`, `periodicity<m>`, `phase<m>`; no terms is
    /// the one zero term.
    pub fn to_params(&self) -> Params {
        let zero = [CosineTermForm::constant(0.0)];
        let terms = if self.terms.is_empty() {
            &zero[..]
        } else {
            &self.terms
        };
        let mut p = Params::new();
        for (i, t) in terms.iter().enumerate() {
            p.set(&format!("k{}", i + 1), t.k);
            p.set(&format!("periodicity{}", i + 1), t.periodicity);
            p.set(&format!("phase{}", i + 1), t.phase);
        }
        p
    }
}

/// LAMMPS `dihedral charmm`: `k[1 + cos(nφ − d)]`. `w` weights the 1-4 pair
/// the dihedral prices; it is not a torsion parameter, so the series does not
/// carry it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharmmForm {
    pub term: CosineTermForm,
    pub w: f64,
}

impl CharmmForm {
    /// The torsion's series (the 1-4 pair `w` prices is no part of it).
    pub fn to_series(&self) -> Result<FourierSeries> {
        self.term.to_series()
    }

    /// The single term of `s` (see [`CosineTermForm::from_series`]), `w = 0`.
    pub fn from_series(s: &FourierSeries) -> Result<Self> {
        Ok(Self {
            term: CosineTermForm::from_series(s)?,
            w: 0.0,
        })
    }

    /// `k`, `periodicity` required; `phase`, `w` absent: 0.
    pub fn from_params(p: &Params) -> Result<Self> {
        const WHAT: &str = "dihedral charmm";
        Ok(Self {
            term: CosineTermForm {
                k: need(p, WHAT, "k")?,
                periodicity: need(p, WHAT, "periodicity")?,
                phase: or_default(p, WHAT, "phase"),
            },
            w: or_default(p, WHAT, "w"),
        })
    }

    pub fn to_params(&self) -> Params {
        let mut p = self.term.to_params();
        p.set("w", self.w);
        p
    }
}

/// LAMMPS `dihedral opls`:
/// `½[k₁(1 + cos φ) + k₂(1 − cos 2φ) + k₃(1 + cos 3φ) + k₄(1 − cos 4φ)]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OplsForm {
    pub k: [f64; 4],
}

impl OplsForm {
    /// `a₀ = ½Σkₙ`, `aₙ = ±½kₙ` (`−` at even `n`).
    pub fn to_series(&self) -> FourierSeries {
        let [k1, k2, k3, k4] = self.k;
        FourierSeries::new(
            vec![
                0.5 * (k1 + k2 + k3 + k4),
                0.5 * k1,
                -0.5 * k2,
                0.5 * k3,
                -0.5 * k4,
            ],
            vec![],
        )
    }

    /// `kₙ = ±2aₙ`, constant included (`a₀ = a₁ − a₂ + a₃ − a₄`: OPLS is
    /// zero at φ = 180°).
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::OrderTooHigh`] above `n = 4`, [`TorsionRefusal::SineTerm`],
    /// [`TorsionRefusal::ConstantOffset`].
    pub fn from_series(s: &FourierSeries) -> Result<Self> {
        require_cosines(s, Some(4))?;
        let opls = Self::nearest(s);
        constant_is(s, opls.to_series().constant())?;
        Ok(opls)
    }

    /// `kₙ = ±2aₙ` of the cosines up to `n = 4`, the constant and sines
    /// dropped: the starting point of a fit (on a full period of φ, the
    /// least-squares one up to the constant).
    pub fn nearest(s: &FourierSeries) -> Self {
        Self {
            k: [2.0 * s.a(1), -2.0 * s.a(2), 2.0 * s.a(3), -2.0 * s.a(4)],
        }
    }

    /// `k1..k4`, absent: the spec's default (0).
    pub fn from_params(p: &Params) -> Self {
        Self {
            k: ["k1", "k2", "k3", "k4"].map(|k| or_default(p, "dihedral opls", k)),
        }
    }

    pub fn to_params(&self) -> Params {
        let mut p = Params::new();
        for (i, &k) in self.k.iter().enumerate() {
            p.set(&format!("k{}", i + 1), k);
        }
        p
    }
}

/// LAMMPS `dihedral class2`, torsion part: `Σₙ₌₁³ kₙ[1 − cos(nφ − φₙ)]`,
/// `φₙ` in degrees. The cross terms are outside the Class-I IR.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Class2Form {
    pub k: [f64; 3],
    /// degrees
    pub phi: [f64; 3],
}

impl Class2Form {
    /// `a₀ = Σkₙ`, `aₙ = −kₙ cos φₙ`, `bₙ = −kₙ sin φₙ`.
    pub fn to_series(&self) -> FourierSeries {
        let mut s = FourierSeries::zero();
        for (i, (&k, &phi)) in self.k.iter().zip(&self.phi).enumerate() {
            let (c, sn) = cos_sin_deg(phi);
            s.add_cos(0, k);
            s.add_cos(i + 1, -k * c);
            s.add_sin(i + 1, -k * sn);
        }
        s
    }

    /// `kₙ = ±√(aₙ² + bₙ²)`, the signs those (all `+` first) whose sum is
    /// `a₀`; `φₙ` from `(aₙ, bₙ) = −kₙ(cos φₙ, sin φₙ)`.
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::OrderTooHigh`] above `n = 3`;
    /// [`TorsionRefusal::ConstantOffset`] when no choice of signs sums to
    /// `a₀`.
    pub fn from_series(s: &FourierSeries) -> Result<Self> {
        if s.order() > 3 {
            return Err(TorsionRefusal::OrderTooHigh {
                n: s.order(),
                max: 3,
            });
        }
        let amplitude = [1, 2, 3].map(|n| s.a(n).hypot(s.b(n)));
        let mut first = None;
        for mask in 0..8u32 {
            let sign = [0, 1, 2].map(|i| if (mask >> i) & 1 == 1 { -1.0 } else { 1.0 });
            if (0..3).any(|i| sign[i] < 0.0 && amplitude[i] == 0.0) {
                continue;
            }
            let mut out = Self {
                k: [0.0; 3],
                phi: [0.0; 3],
            };
            for i in 0..3 {
                let (a, b) = (s.a(i + 1), s.b(i + 1));
                out.k[i] = sign[i] * amplitude[i];
                out.phi[i] = phase_deg(-sign[i] * b, -sign[i] * a);
            }
            let implied = out.k.iter().sum();
            match constant_is(s, implied) {
                Ok(()) => return Ok(out),
                Err(e) => {
                    first.get_or_insert(e);
                }
            }
        }
        Err(first.expect("the all-positive choice is always tried"))
    }

    /// `kₙ = √(aₙ² + bₙ²) ≥ 0` of the orders up to 3, the constant and
    /// higher orders dropped: the starting point of a fit.
    pub fn nearest(s: &FourierSeries) -> Self {
        let mut out = Self {
            k: [0.0; 3],
            phi: [0.0; 3],
        };
        for i in 0..3 {
            let (a, b) = (s.a(i + 1), s.b(i + 1));
            out.k[i] = a.hypot(b);
            out.phi[i] = phase_deg(-b, -a);
        }
        out
    }

    /// `k1..k3`, `phi1..phi3`, absent: the spec's default (0).
    pub fn from_params(p: &Params) -> Self {
        Self {
            k: ["k1", "k2", "k3"].map(|k| or_default(p, "dihedral class2", k)),
            phi: ["phi1", "phi2", "phi3"].map(|k| or_default(p, "dihedral class2", k)),
        }
    }

    pub fn to_params(&self) -> Params {
        let mut p = Params::new();
        for (i, (&k, &phi)) in self.k.iter().zip(&self.phi).enumerate() {
            p.set(&format!("k{}", i + 1), k);
            p.set(&format!("phi{}", i + 1), phi);
        }
        p
    }
}

/// LAMMPS `dihedral nharmonic`: `Σᵢ₌₁ᴺ Aᵢ cosⁱ⁻¹φ` (`a[i−1]` = `Aᵢ`).
#[derive(Debug, Clone, PartialEq)]
pub struct NHarmonicForm {
    pub a: Vec<f64>,
}

impl NHarmonicForm {
    pub fn to_series(&self) -> FourierSeries {
        cos_poly_to_series(&self.a)
    }

    /// `cos nφ = Tₙ(cos φ)`, constant included; `N = order + 1` (at least 1).
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::SineTerm`].
    pub fn from_series(s: &FourierSeries) -> Result<Self> {
        require_cosines(s, None)?;
        Ok(Self {
            a: series_to_cos_poly(s, 1),
        })
    }

    /// The cosines of `s`, sines dropped: the starting point of a fit.
    pub fn nearest(s: &FourierSeries) -> Self {
        Self {
            a: series_to_cos_poly(&cosines_up_to(s, s.order()), 1),
        }
    }

    /// `a1..aN`, contiguous from `a1`, at least one.
    pub fn from_params(p: &Params) -> Result<Self> {
        Ok(Self {
            a: nharmonic_coefficients(p).map_err(|_| TorsionRefusal::MissingParam {
                style: "dihedral nharmonic",
                key: "a1..aN".into(),
            })?,
        })
    }

    pub fn to_params(&self) -> Params {
        let mut p = Params::new();
        for (i, &a) in self.a.iter().enumerate() {
            p.set(&format!("a{}", i + 1), a);
        }
        p
    }
}

/// LAMMPS `dihedral multi/harmonic`: `Σₙ₌₁⁵ Aₙ cosⁿ⁻¹φ` — `nharmonic` at `N = 5`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiHarmonicForm {
    pub a: [f64; 5],
}

impl MultiHarmonicForm {
    pub fn to_series(&self) -> FourierSeries {
        cos_poly_to_series(&self.a)
    }

    /// As [`NHarmonicForm::from_series`], constant included.
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::OrderTooHigh`] above `n = 4`, [`TorsionRefusal::SineTerm`].
    pub fn from_series(s: &FourierSeries) -> Result<Self> {
        require_cosines(s, Some(4))?;
        let p = series_to_cos_poly(s, 5);
        Ok(Self {
            a: [p[0], p[1], p[2], p[3], p[4]],
        })
    }

    /// The cosines of `s` up to `n = 4`, sines and higher orders dropped: the
    /// starting point of a fit.
    pub fn nearest(s: &FourierSeries) -> Self {
        Self::from_series(&cosines_up_to(s, 4)).expect("cosines of order ≤ 4")
    }

    /// `a1..a5`, absent: the spec's default (0).
    pub fn from_params(p: &Params) -> Self {
        Self {
            a: ["a1", "a2", "a3", "a4", "a5"].map(|a| or_default(p, "dihedral multi/harmonic", a)),
        }
    }

    pub fn to_params(&self) -> Params {
        let mut p = Params::new();
        for (i, &a) in self.a.iter().enumerate() {
            p.set(&format!("a{}", i + 1), a);
        }
        p
    }
}

/// Ryckaert–Bellemans, `Σₙ₌₀⁵ Cₙ cosⁿψ` with `ψ = φ − 180°`: molrs `dihedral
/// rb` (molrec's registry style, priced by its expression), GROMACS funct 3,
/// OpenMM `RBTorsionForce`. Since `cos ψ = −cos φ` it is
/// `multi/harmonic` / `nharmonic` with `Aₙ₊₁ = (−1)ⁿ Cₙ`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RyckaertBellemansForm {
    pub c: [f64; 6],
}

impl RyckaertBellemansForm {
    /// The `nharmonic` coefficients `Aₙ₊₁ = (−1)ⁿ Cₙ` (N = 6).
    pub fn to_nharmonic(&self) -> NHarmonicForm {
        NHarmonicForm {
            a: self
                .c
                .iter()
                .enumerate()
                .map(|(n, &c)| if n % 2 == 0 { c } else { -c })
                .collect(),
        }
    }

    pub fn to_series(&self) -> FourierSeries {
        self.to_nharmonic().to_series()
    }

    /// `Cₙ = (−1)ⁿ Aₙ₊₁`, constant included.
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::OrderTooHigh`] above `n = 5`, [`TorsionRefusal::SineTerm`].
    pub fn from_series(s: &FourierSeries) -> Result<Self> {
        require_cosines(s, Some(5))?;
        let p = series_to_cos_poly(s, 6);
        let mut c = [0.0; 6];
        for (n, slot) in c.iter_mut().enumerate() {
            *slot = if n % 2 == 0 { p[n] } else { -p[n] };
        }
        Ok(Self { c })
    }

    /// The cosines of `s` up to `n = 5`, sines and higher orders dropped: the
    /// starting point of a fit.
    pub fn nearest(s: &FourierSeries) -> Self {
        Self::from_series(&cosines_up_to(s, 5)).expect("cosines of order ≤ 5")
    }

    /// The `dihedral rb` row `c0..c5`, all required (its expression reads
    /// every one).
    pub fn from_params(p: &Params) -> Result<Self> {
        let mut c = [0.0; 6];
        for (n, slot) in c.iter_mut().enumerate() {
            *slot = need(p, "dihedral rb", &format!("c{n}"))?;
        }
        Ok(Self { c })
    }

    pub fn to_params(&self) -> Params {
        let mut p = Params::new();
        for (n, &c) in self.c.iter().enumerate() {
            p.set(&format!("c{n}"), c);
        }
        p
    }
}

// ── the harmonic improper (not a Fourier form) ──────────────────────────────

/// LAMMPS `improper harmonic`: `K(χ − χ0)²`, `χ = |φ|`, `χ0` in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImproperHarmonicForm {
    pub k: f64,
    /// degrees
    pub chi0: f64,
}

impl ImproperHarmonicForm {
    /// The energy at `phi` (radians), straight from the formula.
    pub fn energy(&self, phi: f64) -> f64 {
        let d = phi.abs() - self.chi0.to_radians();
        self.k * d * d
    }

    /// Always refuses: `K(|φ| − χ0)²` is not a finite Fourier series.
    pub fn to_series(&self) -> Result<FourierSeries> {
        Err(TorsionRefusal::NotAFourierForm {
            style: "improper harmonic",
            reason: "K(|φ| − χ0)² is not a finite Fourier series; a periodic improper \
                     matches it to second order only (K = n²k/2)",
        })
    }

    /// The periodic improper of periodicity `n` agreeing with this one to
    /// second order about `φ = χ0`: `k = 2K/n²`, `γ = nχ0 + 180°` (on
    /// (−180°, 180°]), the inverse of [`CosineTermForm::second_order_harmonic`].
    ///
    /// # Errors
    ///
    /// [`TorsionRefusal::NonIntegerPeriodicity`]; [`TorsionRefusal::NoCurvature`]
    /// for `n = 0`.
    pub fn second_order_periodic(&self, periodicity: f64) -> Result<CosineTermForm> {
        let (n, _) = order_of(periodicity)?;
        if n == 0 {
            return Err(TorsionRefusal::NoCurvature);
        }
        let n = n as f64;
        let mut phase = (n * self.chi0 + 180.0).rem_euclid(360.0);
        if phase > 180.0 {
            phase -= 360.0;
        }
        Ok(CosineTermForm {
            k: 2.0 * self.k / (n * n),
            periodicity: n,
            phase,
        })
    }
}

// ── rows of the readers ─────────────────────────────────────────────────────

/// A Ryckaert–Bellemans row `Σₙ₌₀⁵ Cₙ cosⁿ(φ − 180°)` as the IR's polynomial
/// `Σ Aₙ₊₁ cosⁿφ`, `Aₙ₊₁ = (−1)ⁿ Cₙ` (`cos(φ − 180°) = −cos φ`), same energy
/// unit in and out: `dihedral multi/harmonic` (`a1..a5`) when `C₅ = 0`,
/// `dihedral nharmonic` (N = 6) otherwise — exact, constant included. The
/// GROMACS (funct 3) and OpenMM (`<RBTorsionForce>`) readers share it.
pub(crate) fn rb_polynomial(c: [f64; 6]) -> (&'static str, Params) {
    let (style, keep) = if c[5] == 0.0 {
        ("multi/harmonic", 5)
    } else {
        ("nharmonic", 6)
    };
    let mut params = Params::new();
    for (n, &cn) in c[..keep].iter().enumerate() {
        params.set(&format!("a{}", n + 1), if n % 2 == 0 { cn } else { -cn });
    }
    (style, params)
}

/// The `nharmonic` coefficients `a1..aN` of a params bag: contiguous from
/// `a1`, at least one, none past the first gap. Shared by the kernel, the
/// algebra and the LAMMPS writer, so the three read one `N`.
pub(crate) fn nharmonic_coefficients(p: &Params) -> std::result::Result<Vec<f64>, String> {
    let mut a = Vec::new();
    while let Some(v) = p.get(&format!("a{}", a.len() + 1)).and_then(|v| v.as_num()) {
        a.push(v);
    }
    if a.is_empty() {
        return Err("dihedral nharmonic: missing param `a1` (needs a1..aN, N ≥ 1)".into());
    }
    if let Some((key, _)) = p
        .iter()
        .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
        .find(|(key, _)| {
            key.strip_prefix('a')
                .and_then(|i| i.parse::<usize>().ok())
                .is_some_and(|i| i > a.len())
        })
    {
        return Err(format!(
            "dihedral nharmonic: `{key}` follows a gap after `a{}`",
            a.len()
        ));
    }
    Ok(a)
}

// ── the `torsion` form family ───────────────────────────────────────────────

/// The canonical torsion parameters of `s`: the `dihedral periodic` row of
/// [`PeriodicForm::from_series`].
fn canonical_row(s: &FourierSeries) -> TypeParams {
    TypeParams::row(PeriodicForm::from_series(s).to_params())
}

/// The series of canonical (`dihedral periodic`) parameters.
pub(crate) fn canonical_series(tp: &TypeParams) -> Result<FourierSeries> {
    PeriodicForm::from_params(&tp.row)?.to_series()
}

/// The family name of every torsion form.
pub const FAMILY: &str = "torsion";

/// A codec of the `torsion` family from a form's three maps: its row's
/// series (`embed`), the form of a series (`project`, exact) and the nearest
/// form (`seed`, a fit's start).
fn codec(
    canonical: bool,
    series: fn(&Params) -> Result<FourierSeries>,
    project: fn(&FourierSeries) -> Result<Params>,
    seed: fn(&FourierSeries) -> Params,
) -> FormCodec {
    let mut codec = FormCodec::new(
        FAMILY,
        move |tp: &TypeParams| Ok(canonical_row(&series(&tp.row)?)),
        move |tp: &TypeParams| Ok(TypeParams::row(project(&canonical_series(tp)?)?)),
    )
    .seed(move |tp: &TypeParams| Ok(TypeParams::row(seed(&canonical_series(tp)?))));
    codec.canonical = canonical;
    codec
}

/// The `torsion` family: every Fourier-series torsion style molrs registers,
/// as `(category, style, codec)`. The canonical style is `dihedral
/// periodic`; its embedding is the identity on a row already canonical
/// ([`PeriodicForm::is_canonical`]) so that `canonical()` is idempotent.
/// `improper harmonic` and the per-instance styles (MMFF, UFF) register
/// none.
pub(crate) fn codecs() -> Vec<(&'static str, &'static str, FormCodec)> {
    let mut periodic = codec(
        true,
        |p| PeriodicForm::from_params(p)?.to_series(),
        |s| Ok(PeriodicForm::from_series(s).to_params()),
        |s| PeriodicForm::from_series(s).to_params(),
    );
    periodic.embed = std::sync::Arc::new(|tp: &TypeParams| {
        let row = PeriodicForm::from_params(&tp.row)?;
        Ok(if row.is_canonical() {
            TypeParams::row(row.to_params())
        } else {
            canonical_row(&row.to_series()?)
        })
    });
    vec![
        ("dihedral", "periodic", periodic),
        (
            "dihedral",
            "charmm",
            codec(
                false,
                |p| {
                    let f = CharmmForm::from_params(p)?;
                    if f.w != 0.0 {
                        return Err(TorsionRefusal::OneFourWeight { w: f.w });
                    }
                    f.to_series()
                },
                |s| Ok(CharmmForm::from_series(s)?.to_params()),
                |s| {
                    CharmmForm {
                        term: CosineTermForm::nearest(s),
                        w: 0.0,
                    }
                    .to_params()
                },
            ),
        ),
        (
            "dihedral",
            "opls",
            codec(
                false,
                |p| Ok(OplsForm::from_params(p).to_series()),
                |s| Ok(OplsForm::from_series(s)?.to_params()),
                |s| OplsForm::nearest(s).to_params(),
            ),
        ),
        (
            "dihedral",
            "multi/harmonic",
            codec(
                false,
                |p| Ok(MultiHarmonicForm::from_params(p).to_series()),
                |s| Ok(MultiHarmonicForm::from_series(s)?.to_params()),
                |s| MultiHarmonicForm::nearest(s).to_params(),
            ),
        ),
        (
            "dihedral",
            "nharmonic",
            codec(
                false,
                |p| Ok(NHarmonicForm::from_params(p)?.to_series()),
                |s| Ok(NHarmonicForm::from_series(s)?.to_params()),
                |s| NHarmonicForm::nearest(s).to_params(),
            ),
        ),
        (
            "dihedral",
            "harmonic",
            codec(
                false,
                |p| SignedCosineForm::from_params("dihedral harmonic", p)?.to_series(),
                |s| Ok(SignedCosineForm::from_series(s)?.to_params()),
                |s| SignedCosineForm::nearest(s).to_params(),
            ),
        ),
        (
            "dihedral",
            "class2",
            codec(
                false,
                |p| Ok(Class2Form::from_params(p).to_series()),
                |s| Ok(Class2Form::from_series(s)?.to_params()),
                |s| Class2Form::nearest(s).to_params(),
            ),
        ),
        (
            "dihedral",
            "rb",
            codec(
                false,
                |p| Ok(RyckaertBellemansForm::from_params(p)?.to_series()),
                |s| Ok(RyckaertBellemansForm::from_series(s)?.to_params()),
                |s| RyckaertBellemansForm::nearest(s).to_params(),
            ),
        ),
        (
            "improper",
            "cvff",
            codec(
                false,
                |p| SignedCosineForm::from_params("improper cvff", p)?.to_series(),
                |s| Ok(SignedCosineForm::from_series(s)?.to_params()),
                |s| SignedCosineForm::nearest(s).to_params(),
            ),
        ),
        (
            "improper",
            "periodic",
            codec(
                false,
                |p| CosineTermForm::from_params(p)?.to_series(),
                |s| Ok(CosineTermForm::from_series(s)?.to_params()),
                |s| CosineTermForm::nearest(s).to_params(),
            ),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::f64::consts::PI;

    /// An absent optional parameter reads its spec's declared default — a
    /// member of an indexed family its family's — and a stated one reads as
    /// stated.
    #[test]
    fn an_absent_optional_parameter_is_its_spec_default() {
        for (style, key) in [
            ("improper periodic", "phase"),
            ("dihedral periodic", "phase"),
            ("dihedral periodic", "phase3"),
            ("dihedral charmm", "phase"),
            ("dihedral charmm", "w"),
            ("dihedral opls", "k4"),
            ("dihedral class2", "phi2"),
            ("dihedral multi/harmonic", "a5"),
        ] {
            let (category, name) = style.split_once(' ').unwrap();
            let want = crate::ff::ir::builtin_style(category, name)
                .and_then(|spec| {
                    spec.param(key.trim_end_matches(|c: char| c.is_ascii_digit()))
                        .or_else(|| spec.param(key))
                })
                .and_then(|d| d.default.as_ref())
                .and_then(|v| v.as_num())
                .unwrap_or_else(|| panic!("{style} `{key}` has a default"));
            assert_eq!(
                or_default(&Params::new(), style, key),
                want,
                "{style} `{key}`"
            );
            let stated = Params::from_iter([(key, 7.5)]);
            assert_eq!(or_default(&stated, style, key), 7.5, "{style} `{key}`");
        }
    }

    // ── random rows ──

    const SEED: u64 = 0x0070_1510_2026;
    const CASES: usize = 200;

    fn coeff(rng: &mut StdRng) -> f64 {
        rng.random_range(-5.0..5.0)
    }

    fn positive(rng: &mut StdRng) -> f64 {
        rng.random_range(0.05..5.0)
    }

    /// A phase on (−180°, 180°], a quarter of the time exactly 0° or 180°.
    fn phase(rng: &mut StdRng) -> f64 {
        match rng.random_range(0..8) {
            0 => 0.0,
            1 => 180.0,
            _ => {
                let p: f64 = rng.random_range(-180.0..180.0);
                if p == -180.0 { 180.0 } else { p }
            }
        }
    }

    fn sign(rng: &mut StdRng) -> f64 {
        if rng.random_range(0..2) == 0 {
            1.0
        } else {
            -1.0
        }
    }

    /// Any term: negative k, n ∈ 0..=6, any phase (also beyond ±180°).
    fn any_term(rng: &mut StdRng) -> CosineTermForm {
        CosineTermForm {
            k: coeff(rng),
            periodicity: rng.random_range(0..=6) as f64,
            phase: rng.random_range(-540.0..540.0),
        }
    }

    /// A term `from_series` reproduces: k > 0, n ∈ 1..=6, γ ∈ (−180°, 180°].
    fn canonical_term(rng: &mut StdRng, n: usize) -> CosineTermForm {
        CosineTermForm {
            k: positive(rng),
            periodicity: n as f64,
            phase: phase(rng),
        }
    }

    fn random_series(rng: &mut StdRng, order: usize, sines: bool) -> FourierSeries {
        let a = (0..=order).map(|_| coeff(rng)).collect();
        let b = if sines {
            (0..=order).map(|_| coeff(rng)).collect()
        } else {
            vec![]
        };
        FourierSeries::new(a, b)
    }

    type Energy = Box<dyn Fn(f64) -> f64>;

    /// A stored row of a registered torsion style, with its energy straight
    /// from the style's formula.
    struct Row {
        category: &'static str,
        style: &'static str,
        params: Params,
        energy: Energy,
    }

    impl std::fmt::Debug for Row {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{} {} {:?}", self.category, self.style, self.params)
        }
    }

    fn row(
        category: &'static str,
        style: &'static str,
        params: Params,
        energy: impl Fn(f64) -> f64 + 'static,
    ) -> Row {
        Row {
            category,
            style,
            params,
            energy: Box::new(energy),
        }
    }

    /// A row of every style of the `torsion` family, from random coefficients
    /// (negative k, arbitrary phases, n ≤ 6). `charmm`'s `w` is 0: a
    /// non-zero one does not embed.
    fn random_rows(rng: &mut StdRng) -> Vec<Row> {
        let periodic = PeriodicForm {
            terms: (0..rng.random_range(1..=4))
                .map(|_| any_term(rng))
                .collect(),
        };
        let charmm = CharmmForm {
            term: any_term(rng),
            w: 0.0,
        };
        let improper = any_term(rng);
        let opls = OplsForm {
            k: [0; 4].map(|_| coeff(rng)),
        };
        let multi = MultiHarmonicForm {
            a: [0; 5].map(|_| coeff(rng)),
        };
        let nharm = NHarmonicForm {
            a: (0..rng.random_range(1..=7)).map(|_| coeff(rng)).collect(),
        };
        let harmonic = SignedCosineForm {
            k: coeff(rng),
            sign: sign(rng),
            periodicity: rng.random_range(0..=6) as f64,
        };
        let cvff = SignedCosineForm {
            k: coeff(rng),
            sign: sign(rng),
            periodicity: rng.random_range(0..=6) as f64,
        };
        let class2 = Class2Form {
            k: [0; 3].map(|_| coeff(rng)),
            phi: [phase(rng), phase(rng), rng.random_range(-540.0..540.0)],
        };
        let rb = RyckaertBellemansForm {
            c: [0; 6].map(|_| coeff(rng)),
        };
        let poly = |a: Vec<f64>| {
            move |phi: f64| -> f64 {
                a.iter()
                    .enumerate()
                    .map(|(n, a)| a * phi.cos().powi(n as i32))
                    .sum()
            }
        };
        let p2 = periodic.clone();
        vec![
            row("dihedral", "periodic", periodic.to_params(), move |phi| {
                p2.terms.iter().map(|t| t.energy(phi)).sum()
            }),
            row("dihedral", "charmm", charmm.to_params(), move |phi| {
                charmm.term.energy(phi)
            }),
            row("improper", "periodic", improper.to_params(), move |phi| {
                improper.energy(phi)
            }),
            row("dihedral", "opls", opls.to_params(), move |phi| {
                let [k1, k2, k3, k4] = opls.k;
                0.5 * (k1 * (1.0 + phi.cos())
                    + k2 * (1.0 - (2.0 * phi).cos())
                    + k3 * (1.0 + (3.0 * phi).cos())
                    + k4 * (1.0 - (4.0 * phi).cos()))
            }),
            row(
                "dihedral",
                "multi/harmonic",
                multi.to_params(),
                poly(multi.a.to_vec()),
            ),
            row(
                "dihedral",
                "nharmonic",
                nharm.to_params(),
                poly(nharm.a.clone()),
            ),
            row("dihedral", "harmonic", harmonic.to_params(), move |phi| {
                harmonic.energy(phi)
            }),
            row("improper", "cvff", cvff.to_params(), move |phi| {
                cvff.energy(phi)
            }),
            row("dihedral", "class2", class2.to_params(), move |phi| {
                class2
                    .k
                    .iter()
                    .zip(&class2.phi)
                    .enumerate()
                    .map(|(i, (k, p))| k * (1.0 - ((i + 1) as f64 * phi - p.to_radians()).cos()))
                    .sum()
            }),
            row("dihedral", "rb", rb.to_params(), move |phi| {
                (0..6)
                    .map(|n| rb.c[n] * (phi - PI).cos().powi(n as i32))
                    .sum()
            }),
        ]
    }

    /// The registered codec of `(category, style)`.
    fn codec_of(category: &str, style: &str) -> FormCodec {
        codecs()
            .into_iter()
            .find(|(c, s, _)| *c == category && *s == style)
            .map(|(_, _, codec)| codec)
            .unwrap_or_else(|| panic!("no codec for {category} {style}"))
    }

    fn series_of(r: &Row) -> FourierSeries {
        let canonical = (codec_of(r.category, r.style).embed)(&TypeParams::row(r.params.clone()))
            .unwrap_or_else(|e| panic!("{r:?}: {e}"));
        canonical_series(&canonical).unwrap()
    }

    fn assert_energy(got: f64, want: f64, scale: f64, what: &str) {
        assert!(
            (got - want).abs() <= 1e-12 * scale.max(want.abs()),
            "{what}: {got} vs {want} (scale {scale})"
        );
    }

    /// Every style's embedding is its energy, constant included, to 1e-12.
    #[test]
    fn every_embedding_is_the_energy() {
        let mut rng = StdRng::seed_from_u64(SEED);
        for _ in 0..CASES {
            for r in random_rows(&mut rng) {
                let s = series_of(&r);
                let scale = 64.0 * s.scale();
                for _ in 0..8 {
                    let phi = rng.random_range(-PI..PI);
                    assert_energy(s.energy(phi), (r.energy)(phi), scale, &format!("{r:?}"));
                }
            }
        }
    }

    /// Embed then project back onto the same style is the same energy,
    /// constant included, for every row; on a form's canonical inputs (k ≥ 0,
    /// distinct ascending orders, phases on (−180°, 180°]) it is the same
    /// parameters.
    #[test]
    fn round_trips_are_the_identity() {
        let mut rng = StdRng::seed_from_u64(SEED + 1);
        let close = |x: f64, y: f64| (x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0);
        for _ in 0..CASES {
            for r in random_rows(&mut rng) {
                let codec = codec_of(r.category, r.style);
                let canonical = (codec.embed)(&TypeParams::row(r.params.clone())).unwrap();
                let back = (codec.project)(&canonical).unwrap_or_else(|e| panic!("{r:?}: {e}"));
                let s = canonical_series(&canonical).unwrap();
                let again = canonical_series(&(codec.embed)(&back).unwrap()).unwrap();
                let d = s.max_abs_diff(&again);
                assert!(
                    d < 1e-12 * 64.0 * s.scale().max(1.0),
                    "{r:?} → {back:?}: Δ = {d}"
                );
            }

            // Canonical inputs come back identical.
            let mut orders: Vec<usize> = (1..=6).filter(|_| rng.random_range(0..2) == 0).collect();
            if orders.is_empty() {
                orders.push(1);
            }
            let periodic = PeriodicForm {
                terms: orders
                    .iter()
                    .map(|&n| canonical_term(&mut rng, n))
                    .collect(),
            };
            assert!(periodic.is_canonical(), "{periodic:?}");
            let back = PeriodicForm::from_series(&periodic.to_series().unwrap());
            assert_eq!(back.terms.len(), periodic.terms.len(), "{back:?}");
            for (t, u) in periodic.terms.iter().zip(&back.terms) {
                assert!(
                    close(t.k, u.k) && t.periodicity == u.periodicity,
                    "{t:?} {u:?}"
                );
                assert!((t.phase - u.phase).abs() < 1e-10, "{t:?} {u:?}");
            }

            // A single term keeps its sign: k = a₀.
            let n = rng.random_range(1..=6);
            let term = CosineTermForm {
                k: coeff(&mut rng),
                ..canonical_term(&mut rng, n)
            };
            let back = CharmmForm::from_series(&term.to_series().unwrap()).unwrap();
            assert!(close(back.term.k, term.k), "{term:?} {back:?}");
            assert_eq!((back.term.periodicity, back.w), (term.periodicity, 0.0));
            let s = back.to_series().unwrap();
            assert!(s.max_abs_diff(&term.to_series().unwrap()) < 1e-12 * 16.0);

            let sc = SignedCosineForm {
                k: coeff(&mut rng),
                sign: sign(&mut rng),
                periodicity: rng.random_range(1..=6) as f64,
            };
            assert_eq!(
                SignedCosineForm::from_series(&sc.to_series().unwrap()),
                Ok(sc)
            );

            let opls = OplsForm {
                k: [0; 4].map(|_| coeff(&mut rng)),
            };
            assert_eq!(OplsForm::from_series(&opls.to_series()), Ok(opls));

            let multi = MultiHarmonicForm {
                a: [0; 5].map(|_| coeff(&mut rng)),
            };
            let back = MultiHarmonicForm::from_series(&multi.to_series()).unwrap();
            for (x, y) in multi.a.iter().zip(&back.a) {
                assert!((x - y).abs() < 1e-12 * 64.0, "{multi:?} → {back:?}");
            }

            // The top coefficient non-zero, so N survives the trip.
            let mut a: Vec<f64> = (0..rng.random_range(1..=7))
                .map(|_| coeff(&mut rng))
                .collect();
            let top = a.len() - 1;
            if top > 0 {
                a[top] = positive(&mut rng);
            }
            let nh = NHarmonicForm { a };
            let back = NHarmonicForm::from_series(&nh.to_series()).unwrap();
            assert_eq!(back.a.len(), nh.a.len(), "{nh:?} → {back:?}");
            for (x, y) in nh.a.iter().zip(&back.a) {
                assert!((x - y).abs() < 1e-12 * 64.0, "{nh:?} → {back:?}");
            }

            let rb = RyckaertBellemansForm {
                c: [0; 6].map(|_| coeff(&mut rng)),
            };
            let back = RyckaertBellemansForm::from_series(&rb.to_series()).unwrap();
            for (x, y) in rb.c.iter().zip(&back.c) {
                assert!((x - y).abs() < 1e-12 * 64.0, "{rb:?} → {back:?}");
            }

            let class2 = Class2Form {
                k: [0; 3].map(|_| positive(&mut rng)),
                phi: [0; 3].map(|_| phase(&mut rng)),
            };
            let back = Class2Form::from_series(&class2.to_series()).unwrap();
            for i in 0..3 {
                assert!(close(class2.k[i], back.k[i]), "{class2:?} → {back:?}");
                assert!(
                    (class2.phi[i] - back.phi[i]).abs() < 1e-10,
                    "{class2:?} → {back:?}"
                );
            }
            // A negative class2 k is found among the sign choices.
            let mixed = Class2Form {
                k: [positive(&mut rng), -positive(&mut rng), positive(&mut rng)],
                ..class2
            };
            let back = Class2Form::from_series(&mixed.to_series()).unwrap();
            assert!(
                back.to_series().max_abs_diff(&mixed.to_series()) < 1e-12 * 64.0,
                "{mixed:?} → {back:?}"
            );
        }
    }

    /// The canonical periodic row is a fixed point, and carries the constant
    /// in one periodicity-0 term.
    #[test]
    fn the_canonical_periodic_row_is_a_fixed_point() {
        let mut rng = StdRng::seed_from_u64(SEED + 6);
        let codec = codec_of("dihedral", "periodic");
        for _ in 0..CASES {
            let s = random_series(&mut rng, 5, true);
            let p = PeriodicForm::from_series(&s);
            assert!(p.is_canonical(), "{p:?}");
            assert!(p.to_series().unwrap().max_abs_diff(&s) < 1e-12 * 64.0 * s.scale());
            let once = (codec.embed)(&TypeParams::row(p.to_params())).unwrap();
            assert_eq!(
                once.row,
                p.to_params(),
                "canonical rows are kept as they are"
            );
            let zeroth: Vec<_> = p.terms.iter().filter(|t| t.periodicity == 0.0).collect();
            assert!(zeroth.len() <= 1 && p.terms[0].periodicity <= 1.0, "{p:?}");
        }
        // A non-canonical row (negative k, two terms of one order) is not.
        let messy = PeriodicForm {
            terms: vec![
                CosineTermForm {
                    k: -1.0,
                    periodicity: 3.0,
                    phase: 0.0,
                },
                CosineTermForm {
                    k: 2.0,
                    periodicity: 3.0,
                    phase: 0.0,
                },
            ],
        };
        assert!(!messy.is_canonical());
        let canonical = PeriodicForm::from_params(
            &(codec.embed)(&TypeParams::row(messy.to_params()))
                .unwrap()
                .row,
        )
        .unwrap();
        assert_eq!(
            canonical.terms,
            vec![CosineTermForm {
                k: 1.0,
                periodicity: 3.0,
                phase: 0.0
            }],
            "−1[1 + cos 3φ] + 2[1 + cos 3φ] = 1 + cos 3φ: no constant left over"
        );
    }

    /// RB ↔ multi/harmonic ↔ OPLS ↔ periodic: every link is the same series,
    /// constant included, and the RB ↔ multi/harmonic link is the sign map
    /// `Aₙ₊₁ = (−1)ⁿCₙ`.
    #[test]
    fn the_rb_multi_harmonic_opls_periodic_chain() {
        let mut rng = StdRng::seed_from_u64(SEED + 2);
        for _ in 0..CASES {
            let mut c = [0; 6].map(|_| coeff(&mut rng));
            // In OPLS's image: C₅ = 0, and ΣCₙ = E(180°) = 0.
            c[5] = 0.0;
            c[0] = -(c[1] + c[2] + c[3] + c[4]);
            let rb = RyckaertBellemansForm { c };
            let s = rb.to_series();

            assert_eq!(
                rb.to_nharmonic().a,
                [c[0], -c[1], c[2], -c[3], c[4], -c[5]],
                "A(n+1) = (−1)ⁿ C(n), exactly"
            );
            let multi = MultiHarmonicForm::from_series(&s).unwrap();
            for (n, (&a, &cn)) in multi.a.iter().zip(&c).enumerate() {
                let want = if n % 2 == 0 { cn } else { -cn };
                assert!((a - want).abs() < 1e-12 * 64.0, "A{}: {a} vs {want}", n + 1);
            }
            let opls = OplsForm::from_series(&multi.to_series()).unwrap();
            let periodic = PeriodicForm::from_series(&opls.to_series());
            let rb2 = RyckaertBellemansForm::from_series(&periodic.to_series().unwrap()).unwrap();

            for (name, other) in [
                ("multi/harmonic", multi.to_series()),
                ("opls", opls.to_series()),
                ("periodic", periodic.to_series().unwrap()),
                ("rb", rb2.to_series()),
            ] {
                let d = s.max_abs_diff(&other);
                assert!(d < 1e-12 * 64.0, "{name}: Δ = {d}");
            }
            for phi in [0.0, 0.7, PI / 2.0, 2.0, PI] {
                let e_rb: f64 = (0..6).map(|n| c[n] * (phi - PI).cos().powi(n as i32)).sum();
                for (name, e) in [
                    ("multi", multi.to_series().energy(phi)),
                    ("rb via periodic", rb2.to_series().energy(phi)),
                ] {
                    assert_energy(e, e_rb, 64.0 * 5.0, name);
                }
            }
            // Every OPLS phase is 0° or 180°.
            for t in &periodic.terms {
                assert!(t.phase == 0.0 || t.phase == 180.0, "{t:?}");
            }
        }
    }

    /// Out-of-image inputs refuse, naming the term.
    #[test]
    fn out_of_image_inputs_refuse_with_the_reason() {
        let mut rng = StdRng::seed_from_u64(SEED + 3);
        for _ in 0..CASES {
            // A phase off {0°, 180°}: a sine term at its order.
            let n = rng.random_range(1..=4);
            let p = loop {
                let p: f64 = rng.random_range(-179.0..179.0);
                if p.abs() > 1e-3 {
                    break p;
                }
            };
            let s = CosineTermForm {
                k: positive(&mut rng),
                periodicity: n as f64,
                phase: p,
            }
            .to_series()
            .unwrap();
            for (target, got) in [
                ("opls", OplsForm::from_series(&s).map(|_| ())),
                (
                    "multi/harmonic",
                    MultiHarmonicForm::from_series(&s).map(|_| ()),
                ),
                ("nharmonic", NHarmonicForm::from_series(&s).map(|_| ())),
                ("harmonic", SignedCosineForm::from_series(&s).map(|_| ())),
                ("rb", RyckaertBellemansForm::from_series(&s).map(|_| ())),
            ] {
                match got {
                    Err(TorsionRefusal::SineTerm { n: m, .. }) => assert_eq!(m, n),
                    other => panic!("{target}: {other:?}"),
                }
            }

            // An order above the form's highest.
            let mut a6 = vec![0.0; 7];
            a6[6] = 1.0;
            let top = random_series(&mut rng, 5, false) + FourierSeries::new(a6, vec![]);
            assert_eq!(top.order(), 6);
            for (target, max, got) in [
                ("opls", 4, OplsForm::from_series(&top).map(|_| ())),
                (
                    "multi/harmonic",
                    4,
                    MultiHarmonicForm::from_series(&top).map(|_| ()),
                ),
                ("class2", 3, Class2Form::from_series(&top).map(|_| ())),
                (
                    "rb",
                    5,
                    RyckaertBellemansForm::from_series(&top).map(|_| ()),
                ),
            ] {
                match got {
                    Err(TorsionRefusal::OrderTooHigh { n, max: m }) => {
                        assert_eq!(m, max, "{target}");
                        assert!(n > max);
                    }
                    other => panic!("{target}: {other:?}"),
                }
            }

            // Two orders into a single-term form.
            let two = canonical_term(&mut rng, 1).to_series().unwrap()
                + canonical_term(&mut rng, 3).to_series().unwrap();
            let multi = Err(TorsionRefusal::MultiTerm { orders: vec![1, 3] });
            assert_eq!(CosineTermForm::from_series(&two).map(|_| ()), multi);
            assert_eq!(SignedCosineForm::from_series(&two).map(|_| ()), multi);

            // A constant the form's terms do not fix.
            let k = positive(&mut rng);
            let shifted = FourierSeries::new(vec![k + 1.0, 0.0, -k], vec![]);
            for (target, got) in [
                ("charmm", CharmmForm::from_series(&shifted).map(|_| ())),
                (
                    "harmonic",
                    SignedCosineForm::from_series(&shifted).map(|_| ()),
                ),
                ("opls", OplsForm::from_series(&shifted).map(|_| ())),
                ("class2", Class2Form::from_series(&shifted).map(|_| ())),
            ] {
                assert!(
                    matches!(got, Err(TorsionRefusal::ConstantOffset { constant, .. }) if constant == k + 1.0),
                    "{target}: {got:?}"
                );
            }
            // … which the polynomial forms carry.
            assert!(MultiHarmonicForm::from_series(&shifted).is_ok());
            assert!(RyckaertBellemansForm::from_series(&shifted).is_ok());
        }

        // Non-integer n, a sign that is not ±1, a 1-4 weight, a non-Fourier form.
        let half = CosineTermForm {
            k: 1.0,
            periodicity: 2.5,
            phase: 0.0,
        };
        assert_eq!(
            half.to_series(),
            Err(TorsionRefusal::NonIntegerPeriodicity { periodicity: 2.5 })
        );
        let bad = SignedCosineForm {
            k: 1.0,
            sign: 0.5,
            periodicity: 2.0,
        };
        assert_eq!(
            bad.to_series(),
            Err(TorsionRefusal::InvalidSign { sign: 0.5 })
        );
        let charmm = CharmmForm { term: half, w: 0.5 };
        let refused = (codec_of("dihedral", "charmm").embed)(&TypeParams::row(charmm.to_params()))
            .unwrap_err();
        assert!(refused.reason.contains("w = 0.5"), "{refused}");
        assert!(matches!(
            ImproperHarmonicForm { k: 1.0, chi0: 0.0 }.to_series(),
            Err(TorsionRefusal::NotAFourierForm { .. })
        ));
        assert!(
            !codecs()
                .iter()
                .any(|(c, s, _)| (*c, *s) == ("improper", "harmonic")),
            "improper harmonic is no Fourier series and registers no codec"
        );
    }

    /// A 0°/180° phase is a cosine, exactly: no rounding residue in `bₙ`.
    #[test]
    fn phases_on_the_axes_are_exact() {
        for (phase, a, b) in [
            (0.0, 1.0, 0.0),
            (180.0, -1.0, 0.0),
            (-180.0, -1.0, 0.0),
            (540.0, -1.0, 0.0),
            (90.0, 0.0, 1.0),
            (-90.0, 0.0, -1.0),
        ] {
            let s = CosineTermForm {
                k: 1.0,
                periodicity: 2.0,
                phase,
            }
            .to_series()
            .unwrap();
            assert_eq!((s.a(2), s.b(2)), (a, b), "phase {phase}");
        }
    }

    /// Rows on one quadruple sum into one series; the canonical series drops
    /// the constant, the canonical periodic row keeps it.
    #[test]
    fn rows_on_one_quadruple_sum() {
        let rows = [
            PeriodicForm {
                terms: vec![CosineTermForm {
                    k: 1.0,
                    periodicity: 3.0,
                    phase: 0.0,
                }],
            }
            .to_series()
            .unwrap(),
            CharmmForm {
                term: CosineTermForm {
                    k: 0.5,
                    periodicity: 3.0,
                    phase: 180.0,
                },
                w: 0.0,
            }
            .to_series()
            .unwrap(),
            OplsForm {
                k: [2.0, 0.0, 0.0, 0.0],
            }
            .to_series(),
        ];
        let s: FourierSeries = rows.into_iter().sum();
        // a0 = 1 + 0.5 + 1; a1 = 1 (opls ½k1); a3 = 1 − 0.5.
        assert_eq!(s, FourierSeries::new(vec![2.5, 1.0, 0.0, 0.5], vec![]));
        assert_eq!(
            s.canonical(),
            FourierSeries::new(vec![0.0, 1.0, 0.0, 0.5], vec![])
        );
        // 2.5 − (1 + 0.5) = 1 left over: k₀ = ½ at periodicity 0.
        assert_eq!(
            PeriodicForm::from_series(&s).terms,
            vec![
                CosineTermForm {
                    k: 0.5,
                    periodicity: 0.0,
                    phase: 0.0
                },
                CosineTermForm {
                    k: 1.0,
                    periodicity: 1.0,
                    phase: 0.0
                },
                CosineTermForm {
                    k: 0.5,
                    periodicity: 3.0,
                    phase: 0.0
                },
            ]
        );
        assert_eq!(
            CharmmForm::from_series(&FourierSeries::new(vec![0.5, 0.0, 0.0, 0.5], vec![])),
            Ok(CharmmForm {
                term: CosineTermForm {
                    k: 0.5,
                    periodicity: 3.0,
                    phase: 0.0
                },
                w: 0.0
            })
        );
        assert!(matches!(
            CharmmForm::from_series(&FourierSeries::new(vec![9.0, 0.0, 0.0, 0.5], vec![])),
            Err(TorsionRefusal::ConstantOffset { .. })
        ));
    }

    /// The seeds of a fit lie in their form's image, and on a series already
    /// in it they are the projection.
    #[test]
    fn nearest_forms_are_in_the_image() {
        let mut rng = StdRng::seed_from_u64(SEED + 7);
        for _ in 0..CASES {
            let s = random_series(&mut rng, 6, true);
            assert!(OplsForm::from_series(&OplsForm::nearest(&s).to_series()).is_ok());
            assert!(
                MultiHarmonicForm::from_series(&MultiHarmonicForm::nearest(&s).to_series()).is_ok()
            );
            assert!(NHarmonicForm::from_series(&NHarmonicForm::nearest(&s).to_series()).is_ok());
            assert!(
                RyckaertBellemansForm::from_series(&RyckaertBellemansForm::nearest(&s).to_series())
                    .is_ok()
            );
            assert!(Class2Form::from_series(&Class2Form::nearest(&s).to_series()).is_ok());
            assert!(
                CosineTermForm::from_series(&CosineTermForm::nearest(&s).to_series().unwrap())
                    .is_ok()
            );
            let sc = SignedCosineForm::nearest(&s);
            assert!(SignedCosineForm::from_series(&sc.to_series().unwrap()).is_ok());

            let opls = OplsForm {
                k: [0; 4].map(|_| coeff(&mut rng)),
            };
            assert_eq!(OplsForm::nearest(&opls.to_series()), opls);
        }
    }

    /// Periodic ↔ harmonic improper agree to second order about the minimum,
    /// with LAMMPS's un-halved K = n²k/2, and differ at fourth order by
    /// −k n⁴ δ⁴/24.
    #[test]
    fn periodic_and_harmonic_impropers_agree_to_second_order() {
        // AMBER's improper: k[1 + cos(2φ − 180°)] = k[1 − cos 2φ], minimum at 0.
        let amber = CosineTermForm {
            k: 1.1,
            periodicity: 2.0,
            phase: 180.0,
        };
        let h = amber.second_order_harmonic().unwrap();
        assert_eq!(
            h,
            ImproperHarmonicForm { k: 2.2, chi0: 0.0 },
            "K = 2k at n = 2"
        );
        assert_eq!(h.second_order_periodic(2.0), Ok(amber));

        let mut rng = StdRng::seed_from_u64(SEED + 4);
        for _ in 0..CASES {
            let n = rng.random_range(1..=6);
            let t = CosineTermForm {
                k: coeff(&mut rng),
                periodicity: n as f64,
                phase: phase(&mut rng),
            };
            let h = t.second_order_harmonic().unwrap();
            assert!((0.0..=180.0).contains(&h.chi0), "{h:?}");
            assert!((h.k - t.k.abs() * (n * n) as f64 / 2.0).abs() < 1e-15 * h.k);
            // The term's minimum is at +χ0 or −χ0; the harmonic's at both.
            let chi0 = h.chi0.to_radians();
            let side = if t.energy(chi0) <= t.energy(-chi0) {
                1.0
            } else {
                -1.0
            };
            let e_min = t.energy(side * chi0);
            // About it: E_p − E_p(min) = Kδ² − |k| n⁴ δ⁴/24 + O(δ⁶).
            let delta = 1e-2;
            for d in [delta, -delta] {
                if chi0 + d <= 0.0 {
                    continue;
                }
                let ep = t.energy(side * (chi0 + d)) - e_min;
                let eh = h.energy(side * (chi0 + d));
                let quartic = -t.k.abs() * (n as f64).powi(4) * d.powi(4) / 24.0;
                assert!(
                    (ep - eh - quartic).abs() < 1e-3 * quartic.abs() + 1e-15,
                    "{t:?} {h:?}: {ep} − {eh} vs {quartic}"
                );
            }
            // Back: a periodic term with the same minimum and curvature.
            let back = h.second_order_periodic(n as f64).unwrap();
            assert!((back.k - t.k.abs()).abs() < 1e-12 * t.k.abs());
            let back_h = back.second_order_harmonic().unwrap();
            assert!((back_h.k - h.k).abs() < 1e-12 * h.k);
            assert!((back_h.chi0 - h.chi0).abs() < 1e-9, "{h:?} {back_h:?}");
        }
        assert_eq!(
            CosineTermForm {
                k: 1.0,
                periodicity: 0.0,
                phase: 0.0
            }
            .second_order_harmonic(),
            Err(TorsionRefusal::NoCurvature)
        );
    }

    // ── the registered kernels price the series ──

    /// Every registered kernel of a torsion style, evaluated through the
    /// compiler on a random 4-atom geometry, equals the series its codec
    /// embeds the row as — so the algebra, the codecs and the kernels cannot
    /// drift.
    #[test]
    fn registered_kernels_price_the_series() {
        use crate::ff::compile::PotentialCompiler;
        use crate::ff::forcefield::ForceField;
        use crate::ff::potential::flat_coords::compute_dihedral;
        use molrs::core::Block;
        use molrs::core::Frame;
        use molrs::op::Idx;
        use ndarray::Array1;

        let mut rng = StdRng::seed_from_u64(SEED + 5);
        for _ in 0..40 {
            for r in random_rows(&mut rng) {
                let mut ff = ForceField::new("t");
                ff.def_style(r.category, r.style, Params::new())
                    .unwrap()
                    .def_type("t", &["a", "b", "c", "d"], r.params.clone())
                    .unwrap();
                let mut block = Block::new();
                for (key, atom) in [("atomi", 0), ("atomj", 1), ("atomk", 2), ("atoml", 3)] {
                    block
                        .insert(key, Array1::from_vec(vec![atom as Idx]).into_dyn(), None)
                        .unwrap();
                }
                block
                    .insert(
                        "type",
                        Array1::from_vec(vec!["t".to_owned()]).into_dyn(),
                        None,
                    )
                    .unwrap();
                let mut frame = Frame::new();
                frame.insert(
                    if r.category == "dihedral" {
                        "dihedrals"
                    } else {
                        "impropers"
                    },
                    block,
                );
                let pots = PotentialCompiler::new(&ff)
                    .compile(&frame)
                    .unwrap_or_else(|e| panic!("{r:?}: {e}"));
                let s = series_of(&r);
                let scale = 64.0 * s.scale();
                for _ in 0..4 {
                    let coords: Vec<f64> = (0..12).map(|_| rng.random_range(-1.5..1.5)).collect();
                    let phi = compute_dihedral(&coords, 0, 1, 2, 3);
                    assert_energy(
                        pots.energy_at(&coords),
                        s.energy(phi),
                        scale,
                        &format!("{r:?}"),
                    );
                }
            }
        }
    }
}
