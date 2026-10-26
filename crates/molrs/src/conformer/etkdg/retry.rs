//! ETKDG retry loop: maxIterations attempts with a fresh sample each time,
//! plus a `useRandomCoords` fallback.
//!
//! Port of the retry structure in RDKit `EmbeddingOps::embedPoints`
//! (`$RDBASE/Code/GraphMol/DistGeomHelpers/Embedder.cpp`, BSD-3): each
//! iteration draws a new random distance matrix (here: a fresh RNG stream
//! seeded deterministically from the base seed + attempt index), embeds in 4D,
//! runs the first minimization and the chiral/tetrahedral checks, and accepts
//! the first attempt that passes. The default `maxIterations` heuristic is
//! `10 × n_atoms` (RDKit `embedParams.maxIterations == 0` branch). When all
//! plain attempts fail and the fallback is enabled, a final pass uses random
//! box coordinates (`useRandomCoords`).

/// Resolve the effective maximum number of embedding attempts.
///
/// `requested == 0` reproduces RDKit's heuristic `10 × n_atoms`
/// (`embedParams.maxIterations == 0`); a non-zero value is used verbatim.
pub fn effective_max_iterations(requested: usize, n_atoms: usize) -> usize {
    if requested == 0 {
        (10 * n_atoms).max(1)
    } else {
        requested
    }
}

/// Per-attempt seed derived from a base seed and the attempt index, so each
/// retry draws an independent — but fully reproducible — random stream.
pub fn attempt_seed(base_seed: u64, attempt: usize) -> u64 {
    // splitmix-style mixing so consecutive attempts decorrelate.
    let mut z = base_seed
        .wrapping_add(0x9E37_79B9_7F4A_7C15)
        .wrapping_mul((attempt as u64).wrapping_add(1));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_means_ten_per_atom_and_never_zero() {
        assert_eq!(effective_max_iterations(0, 7), 70);
        assert_eq!(effective_max_iterations(0, 0), 1);
        assert_eq!(effective_max_iterations(3, 7), 3);
    }

    #[test]
    fn attempt_seeds_are_reproducible_and_distinct() {
        assert_eq!(attempt_seed(42, 0), attempt_seed(42, 0));
        assert_ne!(attempt_seed(42, 0), attempt_seed(42, 1));
        assert_ne!(attempt_seed(42, 0), attempt_seed(43, 0));
    }
}
