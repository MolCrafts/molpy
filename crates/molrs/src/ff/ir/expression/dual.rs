//! Forward-mode dual numbers with `N` derivative directions.
//!
//! `N = 1` carries dE/dq for a scalar coordinate; `N = 3·arity` carries the
//! gradient over a compound term's point coordinates. Every operation applies
//! the chain rule exactly, so a derivative is as exact as the value.

use molrs::op::F;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Dual<const N: usize> {
    pub v: F,
    pub d: [F; N],
}

impl<const N: usize> Dual<N> {
    pub fn constant(v: F) -> Self {
        Dual { v, d: [0.0; N] }
    }

    /// The independent variable `i` at value `v`.
    pub fn var(v: F, i: usize) -> Self {
        let mut d = [0.0; N];
        d[i] = 1.0;
        Dual { v, d }
    }

    #[inline]
    fn chain(self, v: F, dv: F) -> Self {
        let mut d = self.d;
        for x in &mut d {
            *x *= dv;
        }
        Dual { v, d }
    }

    fn is_constant(&self) -> bool {
        self.d.iter().all(|&x| x == 0.0)
    }

    pub fn exp(self) -> Self {
        let e = self.v.exp();
        self.chain(e, e)
    }

    pub fn ln(self) -> Self {
        self.chain(self.v.ln(), 1.0 / self.v)
    }

    pub fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.chain(s, 0.5 / s)
    }

    pub fn sin(self) -> Self {
        let (s, c) = self.v.sin_cos();
        self.chain(s, c)
    }

    pub fn cos(self) -> Self {
        let (s, c) = self.v.sin_cos();
        self.chain(c, -s)
    }

    pub fn tan(self) -> Self {
        let t = self.v.tan();
        self.chain(t, 1.0 + t * t)
    }

    pub fn asin(self) -> Self {
        self.chain(self.v.asin(), 1.0 / (1.0 - self.v * self.v).sqrt())
    }

    pub fn acos(self) -> Self {
        self.chain(self.v.acos(), -1.0 / (1.0 - self.v * self.v).sqrt())
    }

    pub fn atan(self) -> Self {
        self.chain(self.v.atan(), 1.0 / (1.0 + self.v * self.v))
    }

    /// The derivative is sign(x), 0 at x = 0 (the protocol's rule).
    pub fn abs(self) -> Self {
        if self.v > 0.0 {
            self
        } else if self.v < 0.0 {
            -self
        } else {
            Dual::constant(self.v.abs())
        }
    }

    /// `x^n`, n a constant integer.
    pub fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Dual::constant(1.0);
        }
        let v = self.v.powi(n);
        self.chain(v, n as F * self.v.powi(n - 1))
    }

    /// `x^c`, c a constant.
    pub fn powf(self, c: F) -> Self {
        let v = self.v.powf(c);
        self.chain(v, c * self.v.powf(c - 1.0))
    }

    /// `x^y`, both varying: d = y·x^(y−1)·dx + x^y·ln(x)·dy. The second term
    /// is skipped where y does not vary, so a negative base with a constant
    /// exponent stays finite.
    pub fn pow(self, y: Self) -> Self {
        if y.is_constant() {
            return match super::compile::as_small_int(y.v) {
                Some(n) => self.powi(n),
                None => self.powf(y.v),
            };
        }
        let v = self.v.powf(y.v);
        let a = y.v * self.v.powf(y.v - 1.0);
        let b = v * self.v.ln();
        let mut d = [0.0; N];
        for (i, out) in d.iter_mut().enumerate() {
            let dx = if self.d[i] == 0.0 { 0.0 } else { a * self.d[i] };
            *out = dx + b * y.d[i];
        }
        Dual { v, d }
    }

    pub fn min(self, o: Self) -> Self {
        if self.v <= o.v { self } else { o }
    }

    pub fn max(self, o: Self) -> Self {
        if self.v >= o.v { self } else { o }
    }

    /// `atan2(self, x)`: d = (x·dy − y·dx) / (x² + y²).
    pub fn atan2(self, x: Self) -> Self {
        let r2 = x.v * x.v + self.v * self.v;
        let mut d = [0.0; N];
        for (i, out) in d.iter_mut().enumerate() {
            *out = (x.v * self.d[i] - self.v * x.d[i]) / r2;
        }
        Dual {
            v: self.v.atan2(x.v),
            d,
        }
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    #[inline]
    fn add(mut self, o: Self) -> Self {
        self.v += o.v;
        for i in 0..N {
            self.d[i] += o.d[i];
        }
        self
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    #[inline]
    fn sub(mut self, o: Self) -> Self {
        self.v -= o.v;
        for i in 0..N {
            self.d[i] -= o.d[i];
        }
        self
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        let mut d = [0.0; N];
        for (i, out) in d.iter_mut().enumerate() {
            *out = self.d[i] * o.v + o.d[i] * self.v;
        }
        Dual { v: self.v * o.v, d }
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        let v = self.v / o.v;
        let mut d = [0.0; N];
        for (i, out) in d.iter_mut().enumerate() {
            *out = (self.d[i] - v * o.d[i]) / o.v;
        }
        Dual { v, d }
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        self.v = -self.v;
        for x in &mut self.d {
            *x = -*x;
        }
        self
    }
}

/// A point of three dual coordinates.
pub(crate) type P3<const N: usize> = [Dual<N>; 3];

fn sub3<const N: usize>(a: P3<N>, b: P3<N>) -> P3<N> {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3<const N: usize>(a: P3<N>, b: P3<N>) -> Dual<N> {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3<const N: usize>(a: P3<N>, b: P3<N>) -> P3<N> {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `distance(p1, p2)` = |p2 − p1|; gradient 0 where the points coincide.
pub(crate) fn distance<const N: usize>(a: P3<N>, b: P3<N>) -> Dual<N> {
    let d = sub3(b, a);
    let d2 = dot3(d, d);
    if d2.v == 0.0 {
        return Dual::constant(0.0);
    }
    d2.sqrt()
}

/// `angle(p1, p2, p3)`: the angle at p2, atan2(|u×v|, u·v) — stable at
/// every angle, unlike acos near 0 and π. Collinear points (|u×v| = 0):
/// the value as computed, gradient 0.
pub(crate) fn angle<const N: usize>(a: P3<N>, b: P3<N>, c: P3<N>) -> Dual<N> {
    let u = sub3(a, b);
    let v = sub3(c, b);
    let w = cross3(u, v);
    let w2 = dot3(w, w);
    let x = dot3(u, v);
    if w2.v == 0.0 {
        return Dual::constant((0.0 as F).atan2(x.v));
    }
    w2.sqrt().atan2(x)
}

/// `dihedral(p1, p2, p3, p4)`: the signed dihedral in molrs's convention
/// ([`compute_dihedral`](crate::ff::potential::flat_coords::compute_dihedral)):
/// φ = atan2(|b2|·(b1·n2), n1·n2), b1 = p2 − p1, b2 = p3 − p2, b3 = p4 − p3,
/// n1 = b1×b2, n2 = b2×b3 (IUPAC: trans = ±π).
///
/// A degenerate geometry (three collinear points: n1 or n2 = 0): the value
/// as computed, gradient 0.
pub(crate) fn dihedral<const N: usize>(a: P3<N>, b: P3<N>, c: P3<N>, d: P3<N>) -> Dual<N> {
    let b1 = sub3(b, a);
    let b2 = sub3(c, b);
    let b3 = sub3(d, c);
    let n1 = cross3(b1, b2);
    let n2 = cross3(b2, b3);
    let x = dot3(n1, n2);
    let y = dot3(b2, b2).sqrt() * dot3(b1, n2);
    if dot3(n1, n1).v == 0.0 || dot3(n2, n2).v == 0.0 {
        return Dual::constant(y.v.atan2(x.v));
    }
    y.atan2(x)
}
