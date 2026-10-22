//! The log-factorial the Wigner symbols are summed in.

use libm::lgamma;

use crate::op::F;

/// `ln(n!)`, evaluated as `lgamma(n + 1)` so that the factorial ratios of the
/// Racah and Wigner sums stay finite for large angular momenta.
#[inline]
pub(super) fn ln_factorial(n: i64) -> F {
    debug_assert!(n >= 0, "ln_factorial: argument must be ≥ 0, got {n}");
    lgamma(n as F + 1.0)
}
