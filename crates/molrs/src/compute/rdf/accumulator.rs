//! Streaming (frame-by-frame) RDF accumulation — the one g(r) algorithm.

use molrs::core::{PairSource, QueryMode};
use molrs::op::F;
use ndarray::Array1;

use super::{Rdf, RdfResult};
use crate::compute::{Accumulate, Axis, ComputeError, Histogram, require_dist_sq};

/// Streaming g(r) accumulator (bounded memory), obtained from a configured
/// [`Rdf`] through [`Stream::accumulator`](crate::compute::Stream::accumulator).
///
/// Push one pair source per frame; [`finish`](Accumulate::finish) normalizes
/// the running histogram into an [`RdfResult`] without consuming it, so a long
/// run can be inspected and continued. State is O(`n_bins`) — never
/// O(trajectory). It is the one form of the g(r): [`Rdf`] has no batch
/// `compute`.
#[derive(Debug, Clone)]
pub struct RdfAccumulator {
    rdf: Rdf,
    histogram: Histogram<1>,
    n_points: usize,
    n_query_points: usize,
    volume: F,
    n_frames: usize,
    /// Pairing latched from frame 0: `true` for a half-shell self-query. Only
    /// the pairing must stay constant; the point counts may vary per frame.
    self_query: Option<bool>,
}

impl RdfAccumulator {
    pub(super) fn new(rdf: Rdf) -> Self {
        Self {
            histogram: Histogram::new([rdf.axis()]),
            rdf,
            n_points: 0,
            n_query_points: 0,
            volume: 0.0,
            n_frames: 0,
            self_query: None,
        }
    }

    /// Number of frames accumulated so far.
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Bin every pair distance of one source into a fresh histogram, so a
    /// rejected frame leaves the running state untouched.
    ///
    /// Reads the distances through [`PairSource::visit_dist_sq`], the bulk
    /// column path: a table's column is binned in one tight loop, and a
    /// streaming list batches its distances, so no per-pair dynamic call or
    /// `Option` reaches this loop. The bin is [`Axis::index`](crate::compute::Axis::index)'s.
    fn bin(&self, source: &dyn PairSource) -> Result<Histogram<1>, ComputeError> {
        let axis = self.rdf.axis();
        let mut counts = vec![0u64; axis.n_bins()];
        let mut failure: Option<ComputeError> = None;
        let mut offset = 0usize;
        let visited = source.visit_dist_sq(&mut |chunk| {
            if failure.is_some() {
                return;
            }
            if let Err(k) = Self::bin_chunk(axis, chunk, &mut counts) {
                failure = Some(ComputeError::NonFinite {
                    where_: "pair dist_sq",
                    index: offset + k,
                });
            }
            offset += chunk.len();
        });
        // `push` ran `require_dist_sq` first, which vouched for the column.
        debug_assert!(visited, "require_dist_sq vouched for the dist_sq column");
        match failure {
            Some(err) => Err(err),
            // Integer counts convert exactly (a frame holds far fewer than
            // 2⁵³ pairs), so this equals adding 1.0 per pair.
            None => {
                let counts = counts.iter().map(|&c| c as F).collect::<Array1<F>>();
                Ok(Histogram::from_counts([axis], counts.into_dyn()))
            }
        }
    }
}

impl RdfAccumulator {
    /// Count one slice of squared distances into `counts`.
    ///
    /// The axis arrives by value and the counts as a plain slice, so the
    /// optimizer keeps the axis in registers across the loop and hoists the
    /// sample-independent part of [`Axis::index`](crate::compute::Axis::index).
    ///
    /// # Errors
    /// The slice position of the first non-finite distance; the counts up to
    /// it are already added, which is harmless because the caller discards
    /// the whole frame on error.
    #[inline]
    fn bin_chunk(axis: Axis, chunk: &[F], counts: &mut [u64]) -> Result<(), usize> {
        for (k, &d2) in chunk.iter().enumerate() {
            // A coincident pair (d² = 0) has no shell and is skipped; a NaN
            // goes on to `Axis::index`, which rejects it.
            if !(d2 > 0.0 || d2.is_nan()) {
                continue;
            }
            match axis.index(d2.sqrt()) {
                Ok(Some(i)) => counts[i] += 1,
                Ok(None) => {}
                Err(_) => return Err(k),
            }
        }
        Ok(())
    }
}

impl Accumulate for RdfAccumulator {
    type Item<'a> = &'a (dyn PairSource + Sync);
    type Output = RdfResult;

    /// Fold one frame's pairs into the running histogram.
    ///
    /// # Errors
    /// The frame is rejected — and the state left unchanged — when:
    /// - its pairing (self vs cross) differs from frame 0's
    ///   ([`ComputeError::BadShape`]);
    /// - the source carries no box ([`ComputeError::BadShape`]: g(r) needs
    ///   the box volume, and none is invented);
    /// - the box volume is not finite and positive
    ///   ([`ComputeError::OutOfRange`]);
    /// - the source has no `dist_sq` column ([`ComputeError::BadShape`]);
    /// - a distance is not finite ([`ComputeError::NonFinite`]).
    fn push(&mut self, source: &(dyn PairSource + Sync)) -> Result<(), ComputeError> {
        let mode = source.mode();
        let self_query = matches!(mode, QueryMode::SelfQuery { .. });
        if let Some(latched) = self.self_query
            && latched != self_query
        {
            return Err(ComputeError::BadShape {
                expected: format!("{} (frame 0)", pairing(latched)),
                got: format!("{} (frame {})", pairing(self_query), self.n_frames),
            });
        }
        let simbox = source.simbox().ok_or_else(|| ComputeError::BadShape {
            expected: "a pair source searched in a SimBox (its volume normalizes g(r))".into(),
            got: "a source with no box — attach one with Neighbors::with_simbox".into(),
        })?;
        let volume = simbox.volume();
        if !(volume.is_finite() && volume > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "Rdf::volume",
                value: volume.to_string(),
            });
        }
        require_dist_sq(source)?;
        let histogram = self.bin(source)?;

        self.histogram.merge(&histogram);
        self.self_query = Some(self_query);
        let (n_query_points, n_points) = match mode {
            QueryMode::SelfQuery { n_points } => (n_points, n_points),
            QueryMode::CrossQuery {
                n_query_points,
                n_points,
            } => (n_query_points, n_points),
        };
        self.n_points += n_points;
        self.n_query_points += n_query_points;
        self.volume += volume;
        self.n_frames += 1;
        Ok(())
    }

    /// Normalize the accumulated histogram into an [`RdfResult`].
    ///
    /// # Errors
    /// [`ComputeError::EmptyInput`] before any frame, or when the frames
    /// held no points (no density to normalize by).
    fn finish(&self) -> Result<RdfResult, ComputeError> {
        let Some(self_query) = self.self_query else {
            return Err(ComputeError::EmptyInput);
        };
        let nf = self.n_frames as F;
        let vol = self.volume / nf;
        let axis = self.rdf.axis();
        let shells = axis.shell_volumes(self.rdf.dimensionality())?;
        let n_r = self
            .histogram
            .counts()
            .iter()
            .copied()
            .collect::<Array1<F>>();

        let rdf = if self_query {
            let n = self.n_points as F / nf;
            if n <= 0.0 {
                return Err(ComputeError::EmptyInput);
            }
            let rho = n / vol;
            ndarray::Zip::from(&n_r)
                .and(&shells)
                .map_collect(|&count, &v_shell| 2.0 * count / (rho * v_shell * n * nf))
        } else {
            let n_a = self.n_query_points as F / nf;
            let n_b = self.n_points as F / nf;
            if n_a <= 0.0 || n_b <= 0.0 {
                return Err(ComputeError::EmptyInput);
            }
            ndarray::Zip::from(&n_r)
                .and(&shells)
                .map_collect(|&count, &v_shell| count / (n_a * n_b * v_shell / vol * nf))
        };

        Ok(RdfResult {
            bin_edges: axis.edges(),
            bin_centers: axis.centers(),
            rdf,
            n_r,
            n_points: self.n_points,
            n_query_points: self.n_query_points,
            self_query,
            volume: self.volume,
            r_min: axis.min(),
            n_frames: self.n_frames,
            dimensionality: self.rdf.dimensionality(),
        })
    }
}

fn pairing(self_query: bool) -> &'static str {
    if self_query {
        "half-shell self-query"
    } else {
        "directed cross-query"
    }
}
