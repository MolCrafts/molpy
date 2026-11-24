//! Fixed-origin transient correlations over an ensemble of driven trajectories.

use molrs::op::F;
use ndarray::{ArrayView2, Axis};

use super::{Correlator, CorrelatorInput, LagKernel, Origins, TimeCorrelation};
use crate::compute::{Compute, ComputeError};

/// Transient time-correlation function of a scalar flux and observable.
/// Each column is one equally weighted trajectory started from equilibrium
/// and driven from row zero. The single output channel is
/// `C(t) = mean_r[flux(0,r) observable(t,r)]`.
/// For a current response, supply current densities projected onto the field
/// direction. The raw correlation carries the product of the input units.
///
/// ```
/// use molrs::compute::{Compute, Ttcf, TtcfInput};
/// use ndarray::array;
///
/// let j = array![[2., -2.], [3., 1.], [1., 3.]];
/// let curve = Ttcf::new(2).compute(TtcfInput {
///     flux: j.view(), observable: None, dt: 0.5,
/// })?;
/// assert_eq!(curve.channel(0)?.to_vec(), vec![4., 2., -2.]);
/// # Ok::<(), molrs::compute::ComputeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ttcf {
    correlator: Correlator,
}

/// Equally spaced samples on aligned trajectories, including the switch-on frame.
#[derive(Debug, Clone, Copy)]
pub struct TtcfInput<'a> {
    /// Scalar flux, (frames, trajectories); each column supplies its own origin.
    pub flux: ArrayView2<'a, F>,
    /// Driven observable on the same grid, or the flux itself.
    pub observable: Option<ArrayView2<'a, F>>,
    /// Interval between samples, in the desired unit of lag time.
    pub dt: F,
}

impl Ttcf {
    /// Include lags from zero through `max_lag` frames.
    pub fn new(max_lag: usize) -> Self {
        Self {
            correlator: Correlator::new(LagKernel::Product, max_lag).with_origins(Origins::First),
        }
    }
}

impl Compute for Ttcf {
    type Input<'a> = TtcfInput<'a>;
    type Output = TimeCorrelation;

    fn compute(&self, input: TtcfInput<'_>) -> Result<TimeCorrelation, ComputeError> {
        self.correlator.compute(CorrelatorInput {
            a: input.flux.insert_axis(Axis(2)),
            b: input.observable.map(|obs| obs.insert_axis(Axis(2))),
            dt: input.dt,
            boxes: None,
        })
    }
}

crate::compute::spec::compute_spec! {
    Ttcf as "Ttcf";
    params |_s| {
        max_lag: Int;
    }
    inputs {
        flux: Series;
        observable: Series, optional;
        dt: Scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn fixed_origin_products_pair_the_same_trajectory() {
        let flux = array![[2., -2.], [3., 1.], [1., 3.]];
        let observable = array![[1., -1.], [0., 4.], [5., 2.]];
        let run = |b| {
            Ttcf::new(2)
                .compute(TtcfInput {
                    flux: flux.view(),
                    observable: b,
                    dt: 0.5,
                })
                .unwrap()
        };
        let auto = run(None);
        assert_eq!(auto.lag_times, array![0., 0.5, 1.]);
        assert_eq!(auto.values, array![[4.], [2.], [-2.]]);
        assert_eq!(
            run(Some(observable.view())).values,
            array![[2.], [-4.], [3.]]
        );
    }
}
