//! Test-only rayon global pool: bounded workers, env override, installed once.
//!
//! `cargo test` runs many tests at once. The first `par_iter` in each of them
//! would otherwise try to spawn `available_parallelism()` workers into rayon's
//! **global** pool. Concurrent `build_global` races (`GlobalPoolAlreadyInitialized`)
//! and `pthread_create` returning `EAGAIN` (`WouldBlock`) leave the pool
//! uninitialised — every later `par_iter` then panics.
//!
//! Tests install a small pool **before** any parallel iterator. Default is 2
//! workers so the parallel path still runs. Override with `MOLRS_TEST_THREADS`
//! (clamped to 2..=8).

use rayon::prelude::*;
use std::sync::Once;

/// Environment variable that sets the test rayon worker count.
pub(crate) const THREADS_ENV: &str = "MOLRS_TEST_THREADS";

const DEFAULT_THREADS: usize = 2;
const MIN_THREADS: usize = 2;
const MAX_THREADS: usize = 8;

/// Worker count for the test pool: `MOLRS_TEST_THREADS`, else 2, clamped to 2..=8.
pub(crate) fn worker_count() -> usize {
    std::env::var(THREADS_ENV)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_THREADS)
        .clamp(MIN_THREADS, MAX_THREADS)
}

/// Install the test pool once. Safe to call from every rayon-using test.
pub(crate) fn ensure() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let n = worker_count();
        // AlreadyInitialized: another test or rust-analyzer won the race after
        // we picked `n`. The pool exists; thread count may not match `n`.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    });
}

#[test]
fn test_pool_is_multithreaded() {
    ensure();
    let n = rayon::current_num_threads();
    assert!(
        n >= MIN_THREADS,
        "rayon test pool must be multithreaded (got {n} workers; \
         set {THREADS_ENV}>={MIN_THREADS})"
    );

    // What the pool guarantees is its width, asserted above. *Which* workers a
    // given `par_iter` happens to use is not a property this code controls:
    // rayon steals work, and with a trivial body one thread can drain the
    // queue before any other looks. Counting distinct workers therefore
    // measures machine timing, not correctness -- it passed until enabling the
    // Zarr codecs changed how much work ran alongside it.
    //
    // So assert what is actually true and actually ours: every item runs, on
    // the pool, exactly once.
    use std::sync::atomic::{AtomicUsize, Ordering};
    let ran = AtomicUsize::new(0);
    let off_pool = AtomicUsize::new(0);
    (0..1024).into_par_iter().for_each(|_| {
        ran.fetch_add(1, Ordering::Relaxed);
        if rayon::current_thread_index().is_none() {
            off_pool.fetch_add(1, Ordering::Relaxed);
        }
    });
    assert_eq!(ran.load(Ordering::Relaxed), 1024, "par_iter dropped items");
    assert_eq!(
        off_pool.load(Ordering::Relaxed),
        0,
        "par_iter ran work outside the pool"
    );
}

#[test]
fn test_thread_count_follows_env_or_default() {
    ensure();
    let configured = worker_count();
    assert!((MIN_THREADS..=MAX_THREADS).contains(&configured));
    // If we won the install, current_num_threads matches. If rust-analyzer
    // already built a larger pool, we still have at least MIN_THREADS.
    assert!(rayon::current_num_threads() >= MIN_THREADS);
}
