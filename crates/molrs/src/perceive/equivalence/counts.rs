//! Path counts and the accumulators that sum them, for the histogram engine
//! in [`super::paths`].

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// The multiply-rotate hash of rustc's `FxHasher`, for the accumulators.
///
/// Their keys are `f64` bit patterns and small integers the walk computes
/// itself — nothing adversarial — and SipHash, the std default, would cost more
/// than the accumulation it serves.
#[derive(Default, Clone, Copy)]
pub(super) struct FxHasher(u64);

impl FxHasher {
    const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
    fn mix(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(Self::SEED);
    }
}

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.mix(u64::from_le_bytes(word));
        }
    }
    fn write_u32(&mut self, word: u32) {
        self.mix(u64::from(word));
    }
    fn write_u64(&mut self, word: u64) {
        self.mix(word);
    }
    fn write_usize(&mut self, word: usize) {
        self.mix(word as u64);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

/// A count accumulator: key → summed count.
pub(super) type Acc<K, C> = HashMap<K, C, BuildHasherDefault<FxHasher>>;

/// Add `count` under `key`; `None` when the sum overflows.
pub(super) fn accumulate<K: std::hash::Hash + Eq, C: Count>(
    acc: &mut Acc<K, C>,
    key: K,
    count: C,
) -> Option<()> {
    match acc.entry(key) {
        std::collections::hash_map::Entry::Occupied(mut e) => e.get_mut().add_checked(&count),
        std::collections::hash_map::Entry::Vacant(e) => {
            e.insert(count);
            Some(())
        }
    }
}

/// An exact, unbounded count of paths.
///
/// A count only ever grows by addition, so it is a `u128` with overflow limbs
/// above it: the limbs stay empty (and allocation-free) until a single score is
/// reached by more than `2^128` paths — about 128 ring systems in series — and
/// the representation is canonical (no trailing zero limbs), so the derived
/// equality is numeric equality.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct PathCount {
    low: u128,
    /// Little-endian 64-bit limbs above `low`; never ends in a zero limb.
    high: Vec<u64>,
}

impl PathCount {
    /// A single path.
    pub(super) fn one() -> Self {
        Self {
            low: 1,
            high: Vec::new(),
        }
    }

    /// A count of `n` paths (tests build large ones directly).
    #[cfg(test)]
    pub(super) fn from_u128(n: u128) -> Self {
        Self {
            low: n,
            high: Vec::new(),
        }
    }

    /// The count, if it fits a `u128`.
    #[cfg(test)]
    pub(super) fn as_u128(&self) -> Option<u128> {
        self.high.is_empty().then_some(self.low)
    }

    /// `self × factor`, exactly.
    pub(super) fn scaled(&self, factor: u64) -> Self {
        if factor == 1 {
            return self.clone();
        }
        // Little-endian 64-bit limbs of the whole count, times `factor`.
        let limbs = [self.low as u64, (self.low >> 64) as u64]
            .into_iter()
            .chain(self.high.iter().copied());
        let mut product: Vec<u64> = Vec::with_capacity(self.high.len() + 3);
        let mut carry = 0u128;
        for limb in limbs {
            let wide = u128::from(limb) * u128::from(factor) + carry;
            product.push(wide as u64);
            carry = wide >> 64;
        }
        product.push(carry as u64);
        while product.len() > 2 && product.last() == Some(&0) {
            product.pop();
        }
        Self {
            low: u128::from(product[0]) | (u128::from(product[1]) << 64),
            high: product.split_off(2),
        }
    }

    /// `self += other`, exactly.
    pub(super) fn add(&mut self, other: &Self) {
        let (low, carry) = self.low.overflowing_add(other.low);
        self.low = low;
        if !carry && other.high.is_empty() {
            return;
        }
        let width = self.high.len().max(other.high.len());
        self.high.resize(width, 0);
        let mut carry = u64::from(carry);
        for (i, limb) in self.high.iter_mut().enumerate() {
            let (sum, c1) = limb.overflowing_add(other.high.get(i).copied().unwrap_or(0));
            let (sum, c2) = sum.overflowing_add(carry);
            *limb = sum;
            carry = u64::from(c1) + u64::from(c2);
        }
        if carry > 0 {
            self.high.push(carry);
        }
    }
}

/// A path count: a plain `u128` on the fast path (checked — `None` reports
/// an overflow), or the unbounded [`PathCount`] it falls back to.
pub(super) trait Count: Clone {
    /// A single path.
    fn one() -> Self;
    /// `self += other`; `None` on overflow.
    fn add_checked(&mut self, other: &Self) -> Option<()>;
    /// `self × factor`; `None` on overflow.
    fn scaled_checked(&self, factor: u64) -> Option<Self>;
    /// The count as the canonical unbounded type.
    fn into_path_count(self) -> PathCount;
}

impl Count for u128 {
    fn one() -> Self {
        1
    }
    fn add_checked(&mut self, other: &Self) -> Option<()> {
        *self = self.checked_add(*other)?;
        Some(())
    }
    fn scaled_checked(&self, factor: u64) -> Option<Self> {
        self.checked_mul(u128::from(factor))
    }
    fn into_path_count(self) -> PathCount {
        PathCount {
            low: self,
            high: Vec::new(),
        }
    }
}

impl Count for PathCount {
    fn one() -> Self {
        PathCount::one()
    }
    fn add_checked(&mut self, other: &Self) -> Option<()> {
        self.add(other);
        Some(())
    }
    fn scaled_checked(&self, factor: u64) -> Option<Self> {
        Some(self.scaled(factor))
    }
    fn into_path_count(self) -> PathCount {
        self
    }
}
