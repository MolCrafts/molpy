//! Declared precision of an `f64` column: the binary-grid rounding a writer applies before a value reaches the codec pipeline.

use crate::core::MolRsError;

/// Smallest admissible precision, `2⁻¹⁰⁰⁰`: keeps the quantum a normal number.
pub const PRECISION_MIN: f64 = f64::from_bits(((1023 - 1000) as u64) << 52);

/// Largest admissible precision, `2¹⁰⁰⁰`: keeps `x / q` exact.
pub const PRECISION_MAX: f64 = f64::from_bits(((1023 + 1000) as u64) << 52);

/// The sign and exponent bits of a binary64.
const EXPONENT_MASK: u64 = 0xFFF0_0000_0000_0000;

/// `2⁵²`: at or above `2⁵² · q` a value already is a multiple of `q`, and
/// `x / q` would no longer be an exactly representable integer step.
const TWO_POW_52: f64 = 4_503_599_627_370_496.0;

/// Check that `precision` is a declarable precision.
///
/// # Errors
///
/// A [`MolRsError::Validation`] naming the value when it is NaN, infinite, or
/// outside `[2⁻¹⁰⁰⁰, 2¹⁰⁰⁰]` (zero and negatives included).
pub fn check_precision(precision: f64) -> Result<(), MolRsError> {
    if is_admissible(precision) {
        Ok(())
    } else {
        Err(MolRsError::validation(inadmissible(precision)))
    }
}

/// Whether `precision` is finite and within `[2⁻¹⁰⁰⁰, 2¹⁰⁰⁰]`.
pub(crate) fn is_admissible(precision: f64) -> bool {
    precision.is_finite() && (PRECISION_MIN..=PRECISION_MAX).contains(&precision)
}

/// The message every refusal of an inadmissible precision carries.
pub(crate) fn inadmissible(precision: f64) -> String {
    format!(
        "precision {precision} is not admissible: it must be finite and within [2^-1000, 2^1000]"
    )
}

/// The quantum of `precision`: `2^(e−1)` for `precision = m · 2^e`,
/// `0.5 ≤ m < 1` — the largest power of two not above it.
///
/// Computed on the bits: an admissible precision is a positive normal number,
/// so clearing its mantissa leaves exactly that power of two.
///
/// # Declared precision
///
/// A column may declare a precision `p`, an absolute tolerance in the column's
/// own units. From it a writer derives the **quantum** `q`, the largest power
/// of two not above `p`, and stores every value rounded to a multiple of `q`
/// (ties to even, [`quantize`]). Both operations are exact in binary64, so two
/// writers store the same bits, and `|x − stored(x)| ≤ q/2 ≤ p/2`.
///
/// The grid is binary on purpose: a multiple of `2⁻¹⁰` has zero low-order
/// mantissa bits, which a byte shuffle gathers into runs a lossless compressor
/// removes; a multiple of `10⁻³` has a full mantissa and compresses no better
/// than raw data. See molrec `frame.md` § "Declared precision".
///
/// A reader does nothing with the declaration but carry it: it does not
/// re-round, re-check or refuse.
///
/// # Errors
///
/// [`check_precision`]'s.
///
/// # Examples
///
/// ```
/// use molrs::core::quantum;
///
/// assert_eq!(quantum(1e-3).unwrap(), 2f64.powi(-10));
/// assert_eq!(quantum(0.5).unwrap(), 0.5);
/// assert!(quantum(0.0).is_err());
/// ```
pub fn quantum(precision: f64) -> Result<f64, MolRsError> {
    check_precision(precision)?;
    Ok(f64::from_bits(precision.to_bits() & EXPONENT_MASK))
}

/// `stored(x)` for quantum `q`: `x` itself when it is NaN, ±∞ or
/// `|x| ≥ 2⁵² · q`, else `roundTiesToEven(x / q) · q`.
///
/// `q` must be a quantum ([`quantum`]); both the division and the product are
/// then exact, so the result is deterministic across implementations. A small
/// negative value rounds to `-0.0`, keeping its sign.
///
/// # Examples
///
/// ```
/// use molrs::core::quantize;
///
/// let q = 0.25;
/// assert_eq!(quantize(0.3, q), 0.25);
/// assert_eq!(quantize(2.5 * q, q), 2.0 * q); // a tie goes to even
/// assert_eq!(quantize(3.5 * q, q), 4.0 * q);
/// assert!(quantize(f64::NAN, q).is_nan());
/// ```
#[inline]
pub fn quantize(x: f64, q: f64) -> f64 {
    if !x.is_finite() || x.abs() >= TWO_POW_52 * q {
        x
    } else {
        (x / q).round_ties_even() * q
    }
}

/// [`quantize`] every value of `values` in place.
pub fn quantize_in_place(values: &mut [f64], q: f64) {
    for value in values {
        *value = quantize(*value, q);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_are_the_powers_of_two_the_spec_names() {
        assert_eq!(PRECISION_MIN, 2f64.powi(-1000));
        assert_eq!(PRECISION_MAX, 2f64.powi(1000));
    }

    #[test]
    fn the_quantum_is_the_largest_power_of_two_not_above_p() {
        assert_eq!(quantum(1e-3).unwrap(), 2f64.powi(-10));
        assert_eq!(quantum(1e-2).unwrap(), 2f64.powi(-7));
        assert_eq!(quantum(0.5).unwrap(), 0.5);
        assert_eq!(quantum(0.75).unwrap(), 0.5);
        assert_eq!(quantum(1.0).unwrap(), 1.0);
        assert_eq!(quantum(PRECISION_MIN).unwrap(), PRECISION_MIN);
        assert_eq!(quantum(PRECISION_MAX).unwrap(), PRECISION_MAX);
        // Agrees with frexp: p = m·2^e, 0.5 <= m < 1, q = 2^(e-1).
        for p in [3.7e-5_f64, 0.1, 0.3, 7.0, 1234.5] {
            let e = p.log2().floor() as i32 + 1;
            assert_eq!(quantum(p).unwrap(), 2f64.powi(e - 1), "p = {p}");
        }
    }

    #[test]
    fn inadmissible_precisions_are_refused() {
        for p in [
            0.0,
            -1.0,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            PRECISION_MIN / 2.0,
            PRECISION_MAX * 2.0,
        ] {
            assert!(quantum(p).is_err(), "accepted {p}");
        }
    }

    #[test]
    fn ties_go_to_even() {
        let q = 2f64.powi(-10);
        assert_eq!(quantize(2.5 * q, q), 2.0 * q);
        assert_eq!(quantize(3.5 * q, q), 4.0 * q);
        assert_eq!(quantize(-2.5 * q, q), -2.0 * q);
        assert_eq!(quantize(0.5 * q, q), 0.0);
    }

    #[test]
    fn a_small_negative_keeps_its_sign_as_negative_zero() {
        let q = 2f64.powi(-10);
        let stored = quantize(-0.1 * q, q);
        assert_eq!(stored, 0.0);
        assert!(stored.is_sign_negative());
        assert!(quantize(-0.0, q).is_sign_negative());
    }

    #[test]
    fn non_finite_and_huge_values_are_untouched() {
        let q = 2f64.powi(-10);
        assert!(quantize(f64::NAN, q).is_nan());
        assert_eq!(quantize(f64::INFINITY, q), f64::INFINITY);
        assert_eq!(quantize(f64::NEG_INFINITY, q), f64::NEG_INFINITY);
        let huge = 1.5 * TWO_POW_52 * q;
        assert_eq!(quantize(huge, q).to_bits(), huge.to_bits());
        assert_eq!(quantize(-huge, q).to_bits(), (-huge).to_bits());
        assert_eq!(quantize(f64::MAX, q), f64::MAX);
    }

    #[test]
    fn the_error_is_at_most_half_a_quantum_and_every_value_is_on_the_grid() {
        // A fixed-seed xorshift sweep over several magnitudes.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for p in [1e-3, 1e-2, 0.37, 1e-9] {
            let q = quantum(p).unwrap();
            for _ in 0..20_000 {
                let unit = (next() >> 11) as f64 / (1u64 << 53) as f64;
                let x = (unit - 0.5) * 2000.0;
                let stored = quantize(x, q);
                assert!((x - stored).abs() <= q / 2.0, "x = {x}, p = {p}");
                assert!((x - stored).abs() <= p / 2.0);
                let steps = stored / q;
                assert_eq!(steps, steps.trunc(), "{stored} is off the grid of {q}");
            }
        }
    }

    #[test]
    fn quantize_in_place_rounds_every_value() {
        let q = 0.25;
        let mut values = [0.1, 0.13, 0.374, 0.375, f64::NAN];
        quantize_in_place(&mut values, q);
        assert_eq!(values[..4], [0.0, 0.25, 0.25, 0.5]);
        assert!(values[4].is_nan());
    }

    #[test]
    fn rounding_twice_is_rounding_once() {
        let q = quantum(1e-3).unwrap();
        for x in [1.23456, -7.000_49, 1e6 + 0.1234] {
            let once = quantize(x, q);
            assert_eq!(quantize(once, q).to_bits(), once.to_bits());
        }
    }
}
