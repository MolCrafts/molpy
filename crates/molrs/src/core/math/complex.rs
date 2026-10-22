//! Minimal complex-number type: [`Complex`].

use crate::op::F;

/// Complex number with `f64` real and imaginary parts, used by the spherical
/// harmonics and Wigner-D kernels.
///
/// molrs does not pull in `num-complex` — only a handful of operations are
/// needed (add, sub, mul, conjugate, modulus, polar construction), so a
/// ~100-line newtype avoids the extra crate dependency. The layout matches
/// `num_complex::Complex<F>` (two contiguous `F`s) so a downstream crate that
/// does depend on `num-complex` can transmute slices if profiling warrants.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[repr(C)]
pub struct Complex {
    pub re: F,
    pub im: F,
}

impl Complex {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    #[inline]
    pub const fn new(re: F, im: F) -> Self {
        Self { re, im }
    }

    /// `e^{i·θ}` via Euler's formula.
    #[inline]
    pub fn from_polar(r: F, theta: F) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }

    /// Complex conjugate.
    #[inline]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Modulus squared `|z|²`.
    #[inline]
    pub fn norm_sqr(self) -> F {
        self.re * self.re + self.im * self.im
    }

    /// Modulus `|z|`.
    #[inline]
    pub fn norm(self) -> F {
        self.norm_sqr().sqrt()
    }

    /// Multiplication by a real scalar.
    #[inline]
    pub fn scale(self, k: F) -> Self {
        Self::new(self.re * k, self.im * k)
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl std::ops::AddAssign for Complex {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl std::ops::Mul<F> for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: F) -> Self {
        self.scale(rhs)
    }
}

impl std::ops::Neg for Complex {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const TOL: F = 1e-12;

    #[test]
    fn polar_and_norm() {
        let z = Complex::from_polar(2.0, std::f64::consts::FRAC_PI_4);
        assert!((z.re - 2.0_f64.sqrt()).abs() < TOL);
        assert!((z.im - 2.0_f64.sqrt()).abs() < TOL);
        assert!((z.norm() - 2.0).abs() < TOL);
    }

    #[test]
    fn conj_mul_gives_norm_sqr() {
        let z = Complex::new(3.0, -4.0);
        let n2 = (z * z.conj()).re;
        assert!((n2 - 25.0).abs() < TOL);
        assert!((z * z.conj()).im.abs() < TOL);
        assert!((z.norm_sqr() - 25.0).abs() < TOL);
    }

    #[test]
    fn euler_identity() {
        // e^{iπ} = -1
        let z = Complex::from_polar(1.0, std::f64::consts::PI);
        assert!((z.re + 1.0).abs() < 1e-15);
        assert!(z.im.abs() < 1e-15);
    }

    #[test]
    fn arithmetic_ops() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, -1.0);
        assert_eq!(a + b, Complex::new(4.0, 1.0));
        assert_eq!(a - b, Complex::new(-2.0, 3.0));
        assert_eq!(a * b, Complex::new(1.0 * 3.0 + 2.0, -1.0 + 2.0 * 3.0));
        assert_eq!(-a, Complex::new(-1.0, -2.0));
    }
}
