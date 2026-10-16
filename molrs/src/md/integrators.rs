//! The integrators, [`VelocityVerlet`] and [`Langevin`], that advance an [`MdState`].

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

use molrs::core::SimBox;

use super::forces::ForceProvider;
use crate::op::standard_normal;
use molrs::op::{F, Fnx3, I};

use super::error::MdError;
use super::state::{ForceOutput, MdState};

fn as_mass_col(mass: ArrayView1<'_, F>) -> Result<Array2<F>, MdError> {
    if mass.iter().any(|&m| !m.is_finite() || m <= 0.0) {
        return Err(MdError::Invalid("mass must be strictly positive".into()));
    }
    Ok(mass.to_owned().insert_axis(ndarray::Axis(1)))
}

/// Fold the drifted positions back into the cell and bank the crossings.
///
/// This runs after **every** position update, not only at a neighbour rebuild.
/// LAMMPS remaps at reneighbouring because its `exchange`/`borders` transaction
/// is bound to that point; here the cost is one pass that touches only the
/// atoms that actually crossed — `SimBox::wrap` returns an in-cell point
/// untouched to the bit — and in exchange the coordinates never grow past the
/// cell, so nothing downstream has to reason about how far they might have
/// drifted.
///
/// `m` is the shift the wrap actually applied, so the flags cannot disagree
/// with the positions they belong to.
fn wrap_and_bank(simbox: Option<&SimBox>, state: &mut MdState) -> Array2<I> {
    let Some(bx) = simbox else {
        return Array2::zeros((state.pos.nrows(), 3));
    };
    let (wrapped, m) = bx.wrap(state.pos.view());
    state.pos = wrapped;
    state.images += &m;
    m
}

fn check_state_shape(
    pos: ArrayView2<'_, F>,
    vel: ArrayView2<'_, F>,
    n_mass: usize,
) -> Result<(), MdError> {
    if pos.shape() != vel.shape() {
        return Err(MdError::Invalid(format!(
            "pos shape {:?} disagrees with vel shape {:?}",
            pos.shape(),
            vel.shape()
        )));
    }
    if pos.ncols() != 3 {
        return Err(MdError::Invalid(format!(
            "pos must have shape (N, 3), got {:?}",
            pos.shape()
        )));
    }
    if pos.nrows() != n_mass {
        return Err(MdError::Invalid(format!(
            "n_atoms {} disagrees with mass length {n_mass}",
            pos.nrows()
        )));
    }
    Ok(())
}

/// NVE velocity-Verlet (B-A-A-B).
///
/// Construct with timestep, the [`ForceProvider`] the loop evaluates, and
/// mass.
/// The mechanics both schemes share.
///
/// [`VelocityVerlet`] and [`Langevin`] differ in which moves they make and in
/// what order — B-A-A-B against B-A-O-A-B — and in nothing else. Everything
/// else was two copies: the force seam, the cell, the masses, the entry
/// evaluation, the half kick, the half drift, and the fold that has to sit
/// between the last drift and the force. Two copies of a thing that must agree
/// is two places to change it and one place to forget.
struct Stepper {
    dt: F,
    /// Everything that makes a force, behind one seam.
    forces: Box<dyn ForceProvider>,
    /// Cell the positions are folded into each step; `None` is free boundary.
    simbox: Option<SimBox>,
    mass_col: Array2<F>,
    /// Per-atom 1/m (`(N,)`).
    inv_mass: Array1<F>,
}

impl Stepper {
    fn new(
        dt: F,
        forces: impl ForceProvider + 'static,
        mass: ArrayView1<'_, F>,
        simbox: Option<SimBox>,
    ) -> Result<Self, MdError> {
        let mass_col = as_mass_col(mass)?;
        let inv_mass = mass_col.column(0).mapv(|m| 1.0 / m);
        Ok(Self {
            dt,
            forces: Box::new(forces),
            simbox,
            mass_col,
            inv_mass,
        })
    }

    /// Energy and forces at `pos`, for a caller that has not folded anything.
    fn eval_force(&mut self, pos: ArrayView2<'_, F>) -> Result<ForceOutput, MdError> {
        let no_fold = Array2::zeros((pos.nrows(), 3));
        self.forces.compute(pos, no_fold.view())
    }

    /// Seed an [`MdState`], evaluating the entry force.
    fn initial(&mut self, pos: Fnx3, vel: Fnx3) -> Result<MdState, MdError> {
        check_state_shape(pos.view(), vel.view(), self.mass_col.nrows())?;
        // Fold the entry configuration too, so step 0 already satisfies the
        // invariant every later step maintains. Flags start at zero: they count
        // crossings *during this run*, and an atom's history before it is not
        // this integrator's to claim.
        let n_atoms = pos.nrows();
        let mut state = MdState {
            pos,
            images: Array2::zeros((n_atoms, 3)),
            vel,
            forces: Fnx3::zeros((n_atoms, 3)),
            energy: 0.0,
            virial: None,
        };
        let folded = wrap_and_bank(self.simbox.as_ref(), &mut state);
        state.images.fill(0);
        let seeded = self.forces.compute(state.pos.view(), folded.view())?;
        state.forces = seeded.forces;
        state.energy = seeded.energy;
        state.virial = seeded.virial;
        Ok(state)
    }

    /// **B** — half kick, `v += (Δt/2)·f/m`.
    fn kick(&self, state: &mut MdState, half_dt: F) {
        Zip::from(state.vel.rows_mut())
            .and(state.forces.rows())
            .and(&self.inv_mass)
            .for_each(|mut v, f, &im| {
                v[0] += half_dt * f[0] * im;
                v[1] += half_dt * f[1] * im;
                v[2] += half_dt * f[2] * im;
            });
    }

    /// **A** — half drift, `x += (Δt/2)·v`. The two halves of a full drift stay
    /// separate adds rather than one `dt * v`.
    fn drift(&self, state: &mut MdState, half_dt: F) {
        Zip::from(state.pos.rows_mut())
            .and(state.vel.rows())
            .for_each(|mut p, v| {
                p[0] += half_dt * v[0];
                p[1] += half_dt * v[1];
                p[2] += half_dt * v[2];
            });
    }

    /// Fold the drifted positions back into the cell and evaluate the force
    /// there, caching both on `state`.
    ///
    /// The fold has to happen after the last drift and before the force, and
    /// this is the only place either scheme does it. A halo has to reconcile
    /// that fold in the same breath — which is why the shift is handed to
    /// [`ForceProvider::compute`] rather than re-derived from the positions,
    /// where it cannot be seen: a fold relabels an atom without moving it.
    fn refold_and_eval(&mut self, state: &mut MdState) -> Result<(), MdError> {
        let folded = wrap_and_bank(self.simbox.as_ref(), state);
        // Lend the state's force array to the provider and take it back: a
        // provider that accumulates in place swaps rather than clones, so the
        // step allocates nothing. On `Err` the state is left with an empty
        // array, which is fine — a failed force evaluation ends the run.
        let mut out = ForceOutput {
            energy: 0.0,
            forces: std::mem::replace(&mut state.forces, Fnx3::zeros((0, 3))),
            virial: None,
        };
        self.forces
            .compute_into(state.pos.view(), folded.view(), &mut out)?;
        state.forces = out.forces;
        state.energy = out.energy;
        state.virial = out.virial;
        Ok(())
    }
}

/// Velocity-Verlet (NVE): B-A-A-B, the two half-drifts kept as separate adds.
///
/// Required pieces go in the constructor — no `bind_*` afterthoughts:
///
/// ```ignore
/// VelocityVerlet::new(dt, MicPairs::new(ForceTerm::pair(lj), skin).unwrap(), mass, Some(bx))?;
/// ```
///
/// The second argument is a [`ForceProvider`], and it is the only thing an
/// integrator knows about force fields. The potential, the neighbour
/// bookkeeping and the periodic régime all live behind it:
/// [`SelfPairedForces`](crate::md::SelfPairedForces) hands the potential raw
/// coordinates, [`MicPairs`](crate::md::MicPairs) gives it minimum-image
/// pairs, and [`GhostPairs`](crate::md::GhostPairs) gives it periodic copies
/// and folds the forces back. An integrator holds no skin, no halo and no
/// potential, so adding a fourth way to make a force changes nothing here.
///
/// Two schemes, two types — no `gamma=0` switch: the thermostatted one is
/// [`Langevin`]. Units are the caller's; MD has no unit knowledge.
pub struct VelocityVerlet {
    inner: Stepper,
}

impl VelocityVerlet {
    /// `dt` (fs), the force provider, per-atom mass `(N,)`, and the cell
    /// positions are folded into each step (`None` = free boundary, no
    /// wrapping and image flags stay zero).
    pub fn new(
        dt: F,
        forces: impl ForceProvider + 'static,
        mass: ArrayView1<'_, F>,
        simbox: Option<SimBox>,
    ) -> Result<Self, MdError> {
        Ok(Self {
            inner: Stepper::new(dt, forces, mass, simbox)?,
        })
    }

    /// The force provider, for the counters it chooses to expose.
    pub fn forces(&self) -> &dyn ForceProvider {
        &*self.inner.forces
    }

    /// Timestep Δt in fs.
    pub fn dt(&self) -> F {
        self.inner.dt
    }

    /// Per-atom mass column `(N, 1)`.
    pub fn mass(&self) -> &Array2<F> {
        &self.inner.mass_col
    }

    /// Inverse mass `(N,)`.
    pub fn inv_mass(&self) -> &Array1<F> {
        &self.inner.inv_mass
    }

    /// Degrees of freedom the temperature estimator must not count (`3N − 3`).
    pub fn removed_dof(&self) -> usize {
        3
    }

    /// Energy and forces at `pos` (runs the neighbour update policy first).
    pub fn eval_force(&mut self, pos: ArrayView2<'_, F>) -> Result<ForceOutput, MdError> {
        self.inner.eval_force(pos)
    }

    /// Seed an [`MdState`], evaluating the entry force.
    pub fn initial(&mut self, pos: Fnx3, vel: Fnx3) -> Result<MdState, MdError> {
        self.inner.initial(pos, vel)
    }

    /// One NVE step from the cached entry force: **B-A-A-B**.
    pub fn step(&mut self, mut state: MdState) -> Result<MdState, MdError> {
        let half_dt = 0.5 * self.inner.dt;
        self.inner.kick(&mut state, half_dt);
        self.inner.drift(&mut state, half_dt);
        self.inner.drift(&mut state, half_dt);
        self.inner.refold_and_eval(&mut state)?;
        self.inner.kick(&mut state, half_dt);
        Ok(state)
    }

    /// One eager step.
    pub fn advance(&mut self, state: MdState) -> Result<MdState, MdError> {
        self.step(state)
    }

    /// Advance `n_steps` eagerly.
    pub fn advance_n(&mut self, mut state: MdState, n_steps: usize) -> Result<MdState, MdError> {
        for _ in 0..n_steps {
            state = self.advance(state)?;
        }
        Ok(state)
    }
}

/// Langevin velocity-Verlet (BAOAB). γ must be strictly positive.
///
/// Ordering (Leimkuhler & Matthews): B (half kick) → A (half drift) → O
/// (Ornstein-Uhlenbeck) → A → B. The O step is `v ← c1·v + c2·σ·ξ` with
/// `c1 = e^{-γΔt}`, `c2 = √(1-c1²)`, `σ = √(k_BT/m)`.
///
/// ```ignore
/// Langevin::new(dt, gamma, kbt, SelfPairedForces::new(potentials), mass, seed, None)?;
/// ```
///
/// NVE is [`VelocityVerlet`] — not this type with `gamma=0`. Units are the
/// caller's; MD has no unit knowledge.
///
/// Reference: Leimkuhler & Matthews, "Rational Construction of Stochastic
/// Numerical Methods for Molecular Sampling", Appl. Math. Res. Express 2013.
/// <https://doi.org/10.1093/amrx/abs010>
pub struct Langevin {
    inner: Stepper,
    gamma: F,
    c1: F,
    c2: F,
    kbt: F,
    sigma: Array1<F>,
    rng: rand::rngs::StdRng,
}

impl Langevin {
    /// BAOAB integrator with all required pieces at construction.
    ///
    /// Seven arguments is the cost of this module's rule that required pieces
    /// go in the constructor rather than into `bind_*` setters afterwards: a
    /// half-built integrator is a state nobody should be able to hold.
    ///
    /// `seed` fixes the internal noise stream, so [`advance`](Self::advance)
    /// is deterministic given the seed.
    pub fn new(
        dt: F,
        gamma: F,
        kbt: F,
        forces: impl ForceProvider + 'static,
        mass: ArrayView1<'_, F>,
        seed: u64,
        simbox: Option<SimBox>,
    ) -> Result<Self, MdError> {
        if gamma <= 0.0 {
            return Err(MdError::Invalid(
                "Langevin requires gamma > 0; use VelocityVerlet for NVE".into(),
            ));
        }
        if kbt <= 0.0 {
            return Err(MdError::Invalid("Langevin requires kbt > 0".into()));
        }
        let inner = Stepper::new(dt, forces, mass, simbox)?;
        let sigma = inner.mass_col.column(0).mapv(|m| (kbt / m).sqrt());
        let c1 = (-gamma * dt).exp();
        // `1 − e^{−2γΔt}` written directly loses a digit for every decade that
        // `γΔt` is below one — at `γΔt = 1e-8` the noise amplitude keeps barely
        // half its bits, and the sampled temperature carries the error.
        // `exp_m1` computes it to the last bit at any `γΔt`.
        let c2 = (-(-2.0 * gamma * dt).exp_m1()).max(0.0).sqrt();
        Ok(Self {
            inner,
            gamma,
            c1,
            c2,
            kbt,
            sigma,
            rng: rand::SeedableRng::seed_from_u64(seed),
        })
    }

    /// The force provider, for the counters it chooses to expose.
    pub fn forces(&self) -> &dyn ForceProvider {
        &*self.inner.forces
    }

    /// Timestep Δt in fs.
    pub fn dt(&self) -> F {
        self.inner.dt
    }

    /// Friction γ (1/fs).
    pub fn gamma(&self) -> F {
        self.gamma
    }

    /// `c1 = e^{−γΔt}` — the velocity damping of the O step.
    pub fn c1(&self) -> F {
        self.c1
    }

    /// `c2 = √(1 − c1²)` — the noise amplitude of the O step.
    pub fn c2(&self) -> F {
        self.c2
    }

    /// Target `k_B T` in the caller's energy units.
    pub fn kbt(&self) -> F {
        self.kbt
    }

    /// Per-atom mass column `(N, 1)`.
    pub fn mass(&self) -> &Array2<F> {
        &self.inner.mass_col
    }

    /// Per-atom `σ = √(k_BT/m)` `(N,)`.
    pub fn sigma(&self) -> &Array1<F> {
        &self.sigma
    }

    /// Inverse mass `(N,)`.
    pub fn inv_mass(&self) -> &Array1<F> {
        &self.inner.inv_mass
    }

    /// Degrees of freedom the temperature estimator must not count.
    ///
    /// Zero: the thermostat does not conserve momentum, so nothing is removed.
    pub fn removed_dof(&self) -> usize {
        0
    }

    /// Energy and forces at `pos` (runs the neighbour update policy first).
    pub fn eval_force(&mut self, pos: ArrayView2<'_, F>) -> Result<ForceOutput, MdError> {
        self.inner.eval_force(pos)
    }

    /// Seed an [`MdState`], evaluating the entry force.
    pub fn initial(&mut self, pos: Fnx3, vel: Fnx3) -> Result<MdState, MdError> {
        self.inner.initial(pos, vel)
    }

    /// One BAOAB step with caller-supplied standard-normal `noise` `(N, 3)`.
    ///
    /// [`advance`](Self::advance) draws from the seeded internal RNG instead.
    pub fn step(
        &mut self,
        mut state: MdState,
        noise: ArrayView2<'_, F>,
    ) -> Result<MdState, MdError> {
        if noise.shape() != state.vel.shape() {
            return Err(MdError::Invalid(format!(
                "noise shape {:?} disagrees with vel shape {:?}",
                noise.shape(),
                state.vel.shape()
            )));
        }
        let half_dt = 0.5 * self.inner.dt;
        self.inner.kick(&mut state, half_dt);
        self.inner.drift(&mut state, half_dt);
        self.ornstein_uhlenbeck(&mut state, noise);
        self.inner.drift(&mut state, half_dt);
        self.inner.refold_and_eval(&mut state)?;
        self.inner.kick(&mut state, half_dt);
        Ok(state)
    }

    /// **O** — `v ← c1·v + c2·σ·ξ`. The only move `VelocityVerlet` does not
    /// make, and the only reason these are two types.
    fn ornstein_uhlenbeck(&self, state: &mut MdState, noise: ArrayView2<'_, F>) {
        let (c1, c2) = (self.c1, self.c2);
        Zip::from(state.vel.rows_mut())
            .and(&self.sigma)
            .and(noise.rows())
            .for_each(|mut v, &sig, xi| {
                v[0] = c1 * v[0] + c2 * sig * xi[0];
                v[1] = c1 * v[1] + c2 * sig * xi[1];
                v[2] = c1 * v[2] + c2 * sig * xi[2];
            });
    }

    /// Draw `(n_atoms, 3)` standard normals from the seeded internal RNG.
    pub fn draw_noise(&mut self, n_atoms: usize) -> Array2<F> {
        let mut noise = Array2::<F>::zeros((n_atoms, 3));
        for x in noise.iter_mut() {
            *x = standard_normal(&mut self.rng);
        }
        noise
    }

    /// One step with noise drawn from the seeded internal RNG.
    pub fn advance(&mut self, state: MdState) -> Result<MdState, MdError> {
        let noise = self.draw_noise(state.vel.nrows());
        self.step(state, noise.view())
    }

    /// Advance `n_steps` eagerly.
    pub fn advance_n(&mut self, mut state: MdState, n_steps: usize) -> Result<MdState, MdError> {
        for _ in 0..n_steps {
            state = self.advance(state)?;
        }
        Ok(state)
    }
}

/// The `(n,)` per-atom masses of a homogeneous system: `mass` on every atom.
pub fn uniform_masses(mass: F, n: usize) -> Result<Array1<F>, MdError> {
    if !mass.is_finite() || mass <= 0.0 {
        return Err(MdError::Invalid("mass must be strictly positive".into()));
    }
    Ok(Array1::from_elem(n, mass))
}

#[cfg(test)]
mod tests {
    use crate::core::Cell;
    use ndarray::{Array2, ArrayView2, array};

    use molrs::core::{NeighborList, NeighborPolicy, VerletSkin};
    use molrs::ff::potential::{Accumulator, ForceTerm, Potential, Potentials};

    use super::super::forces::{MicPairs, SelfPairedForces};
    use super::*;
    use molrs::ff::potential::pair::PairLjCut;

    fn cube(a: F) -> SimBox {
        SimBox::new(Cell::Lengths([a; 3]), [0.0, 0.0, 0.0], [true, true, true]).unwrap()
    }

    fn soft_lj(n: usize, box_a: F) -> (PairLjCut, VerletSkin, Array2<F>) {
        let cutoff = 2.5;
        let skin = 0.5;
        let mut pos = Array2::<F>::zeros((n, 3));
        for i in 0..n {
            pos[[i, 0]] = (i as F) * 1.1;
        }
        let nl = VerletSkin::new(
            NeighborList::new(cutoff + skin),
            cutoff,
            NeighborPolicy {
                skin,
                ..NeighborPolicy::default()
            },
            pos.view(),
            cube(box_a),
        )
        .unwrap();
        let lj = PairLjCut::lj126(1.0, 1.0, cutoff).unwrap();
        (lj, nl, pos)
    }

    /// Fixed energy + uniform x-force — a bonded-category stand-in for
    /// merge tests (default no-op `set_pairs`).
    struct Uniform {
        energy: F,
        fx: F,
    }

    impl Potential for Uniform {
        fn evaluate(&self, _coords: &[F], acc: &mut Accumulator<'_>) {
            acc.add_energy(self.energy);
            for row in acc.forces_without_virial().chunks_mut(3) {
                row[0] += self.fx;
            }
        }
    }

    fn arrays_close(a: ArrayView2<'_, F>, b: ArrayView2<'_, F>, tol: F) -> bool {
        a.shape() == b.shape() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn step_constants_match_the_closed_form() {
        let dt = 0.05;
        let gamma = 2.0;
        let kbt = 1.5;
        let mass = 2.0;
        let (lj, nl, pos) = soft_lj(1, 20.0);
        let ig = Langevin::new(
            dt,
            gamma,
            kbt,
            MicPairs::new(ForceTerm::pair(lj), nl).unwrap(),
            uniform_masses(mass, 1).unwrap().view(),
            0,
            None,
        )
        .unwrap();
        assert!((ig.c1() - (-gamma * dt).exp()).abs() < 1e-15);
        assert!((ig.c2() - (1.0 - (-2.0 * gamma * dt).exp()).sqrt()).abs() < 1e-15);
        assert!((ig.sigma()[0] - (kbt / mass).sqrt()).abs() < 1e-15);
        assert!((ig.inv_mass()[0] - 1.0 / mass).abs() < 1e-15);
        assert!((ig.dt() - dt).abs() < 1e-15);
        assert!((ig.gamma() - gamma).abs() < 1e-15);
        let _ = pos;
    }

    #[test]
    fn removed_dof_follows_the_scheme() {
        let (lj, nl, _) = soft_lj(2, 40.0);
        let nve = VelocityVerlet::new(
            0.01,
            MicPairs::new(ForceTerm::pair(lj), nl).unwrap(),
            uniform_masses(1.0, 2).unwrap().view(),
            None,
        )
        .unwrap();
        assert_eq!(nve.removed_dof(), 3);
        let (lj, nl, _) = soft_lj(2, 40.0);
        let lgv = Langevin::new(
            0.01,
            1.0,
            1.0,
            MicPairs::new(ForceTerm::pair(lj), nl).unwrap(),
            uniform_masses(1.0, 2).unwrap().view(),
            0,
            None,
        )
        .unwrap();
        assert_eq!(lgv.removed_dof(), 0);
    }

    #[test]
    fn mass_must_be_positive() {
        let (lj, nl, _) = soft_lj(2, 40.0);
        assert!(
            VelocityVerlet::new(
                0.01,
                MicPairs::new(ForceTerm::pair(lj), nl).unwrap(),
                array![-1.0, 1.0].view(),
                None
            )
            .is_err()
        );
    }

    #[test]
    fn langevin_rejects_gamma_zero() {
        let (lj, nl, _) = soft_lj(1, 20.0);
        let err = Langevin::new(
            0.01,
            0.0,
            1.0,
            MicPairs::new(ForceTerm::pair(lj), nl).unwrap(),
            array![1.0].view(),
            0,
            None,
        );
        match err {
            Err(e) => assert!(e.to_string().contains("VelocityVerlet")),
            Ok(_) => panic!("expected Langevin gamma=0 to fail"),
        }
    }

    #[test]
    fn langevin_rejects_nonpositive_kbt() {
        let (lj, nl, _) = soft_lj(1, 20.0);
        assert!(
            Langevin::new(
                0.01,
                1.0,
                0.0,
                MicPairs::new(ForceTerm::pair(lj), nl).unwrap(),
                array![1.0].view(),
                0,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn langevin_advance_is_deterministic_given_the_seed() {
        let (lj, nl, pos) = soft_lj(4, 40.0);
        let vel = Array2::from_elem(pos.raw_dim(), 0.01);
        let mut a = Langevin::new(
            0.01,
            1.0,
            1.0,
            MicPairs::new(ForceTerm::pair(lj), nl).unwrap(),
            uniform_masses(1.0, 4).unwrap().view(),
            9,
            None,
        )
        .unwrap();
        let (lj, nl, _) = soft_lj(4, 40.0);
        let mut b = Langevin::new(
            0.01,
            1.0,
            1.0,
            MicPairs::new(ForceTerm::pair(lj), nl).unwrap(),
            uniform_masses(1.0, 4).unwrap().view(),
            9,
            None,
        )
        .unwrap();
        let sa = a.initial(pos.clone(), vel.clone()).unwrap();
        let sb = b.initial(pos, vel).unwrap();
        let ea = a.advance_n(sa, 3).unwrap();
        let eb = b.advance_n(sb, 3).unwrap();
        assert!(arrays_close(ea.pos.view(), eb.pos.view(), 0.0));
        assert!(arrays_close(ea.vel.view(), eb.vel.view(), 0.0));
    }

    #[test]
    fn force_caching_one_eval_per_step() {
        // Skin with check: force changes when atoms move inside half-skin.
        let (lj, nl, mut pos) = soft_lj(4, 40.0);
        let mut ig = VelocityVerlet::new(
            0.01,
            MicPairs::new(ForceTerm::pair(lj), nl).unwrap(),
            uniform_masses(1.0, 4).unwrap().view(),
            None,
        )
        .unwrap();
        let f0 = ig.eval_force(pos.view()).unwrap().forces;
        pos[[1, 0]] += 0.05;
        let f1 = ig.eval_force(pos.view()).unwrap().forces;
        assert!(!arrays_close(f0.view(), f1.view(), 1e-15));
    }

    #[test]
    fn advance_n_matches_manual_advance_loop() {
        let (lj, nl, pos) = soft_lj(4, 40.0);
        let vel = Array2::from_elem(pos.raw_dim(), 0.01);
        let mut a = VelocityVerlet::new(
            0.01,
            MicPairs::new(ForceTerm::pair(lj), nl).unwrap(),
            uniform_masses(1.0, 4).unwrap().view(),
            None,
        )
        .unwrap();
        let (lj, nl, _) = soft_lj(4, 40.0);
        let mut b = VelocityVerlet::new(
            0.01,
            MicPairs::new(ForceTerm::pair(lj), nl).unwrap(),
            uniform_masses(1.0, 4).unwrap().view(),
            None,
        )
        .unwrap();
        let s0 = a.initial(pos.clone(), vel.clone()).unwrap();
        let end_a = a.advance_n(s0, 5).unwrap();
        let mut state = b.initial(pos, vel).unwrap();
        for _ in 0..5 {
            state = b.advance(state).unwrap();
        }
        assert!(arrays_close(end_a.pos.view(), state.pos.view(), 0.0));
        assert!(arrays_close(end_a.vel.view(), state.vel.view(), 0.0));
    }

    #[test]
    fn potentials_merge_nonbond_and_bonded_terms() {
        // A Potentials collection [PairLjCut, Uniform] through the integrator
        // must equal the lone LJ evaluation plus the uniform offsets.
        let (lj, nl, pos) = soft_lj(2, 40.0);
        let mut lone = VelocityVerlet::new(
            0.01,
            MicPairs::new(ForceTerm::pair(lj), nl).unwrap(),
            uniform_masses(1.0, 2).unwrap().view(),
            None,
        )
        .unwrap();
        let base = lone.eval_force(pos.view()).unwrap();

        let (lj, nl, _) = soft_lj(2, 40.0);
        let mut pots = Potentials::new();
        pots.push(ForceTerm::pair(lj));
        pots.push(ForceTerm::plain(Uniform {
            energy: 0.25,
            fx: -1.5,
        }));
        let mut ig = VelocityVerlet::new(
            0.01,
            MicPairs::new(ForceTerm::pair(pots), nl).unwrap(),
            uniform_masses(1.0, 2).unwrap().view(),
            None,
        )
        .unwrap();
        let out = ig.eval_force(pos.view()).unwrap();
        assert!((out.energy - (base.energy + 0.25)).abs() < 1e-12);
        for i in 0..2 {
            assert!((out.forces[[i, 0]] - (base.forces[[i, 0]] - 1.5)).abs() < 1e-12);
            assert!((out.forces[[i, 1]] - base.forces[[i, 1]]).abs() < 1e-12);
        }
    }

    #[test]
    fn empty_potentials_is_the_zero_potential() {
        let pos = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut ig = VelocityVerlet::new(
            0.01,
            SelfPairedForces::new(Potentials::new()),
            uniform_masses(1.0, 2).unwrap().view(),
            None,
        )
        .unwrap();
        let out = ig.eval_force(pos.view()).unwrap();
        assert_eq!(out.energy, 0.0);
        assert!(out.forces.iter().all(|&f| f == 0.0));
    }

    #[test]
    fn neighbor_rebuilds_flow_from_the_loop_to_the_nonbond_potential() {
        // March one atom far enough to force repeated skin rebuilds; the
        // integrator must feed each fresh pair list to the potential, so the
        // final force must match a freshly wired integrator at the final
        // geometry.
        let cutoff = 2.0;
        let skin = 1.0;
        let pos0 = array![[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]];
        let make = |pos: ArrayView2<'_, F>| {
            let nl = VerletSkin::new(
                NeighborList::new(cutoff + skin),
                cutoff,
                NeighborPolicy {
                    skin,
                    ..NeighborPolicy::default()
                },
                pos,
                cube(20.0),
            )
            .unwrap();
            let lj = PairLjCut::lj126(1.0, 1.0, cutoff).unwrap();
            VelocityVerlet::new(
                0.01,
                MicPairs::new(ForceTerm::pair(lj), nl).unwrap(),
                uniform_masses(1.0, 2).unwrap().view(),
                None,
            )
            .unwrap()
        };
        let mut ig = make(pos0.view());
        let mut x = 1.1;
        let mut last = None;
        for _ in 0..8 {
            x += 0.4;
            let pos = array![[0.0, 0.0, 0.0], [x, 0.0, 0.0]];
            last = Some(ig.eval_force(pos.view()).unwrap());
        }
        let rebuilds = ig
            .forces()
            .neighbor_stats()
            .rebuilds
            .expect("this integrator was built with a minimum-image skin");
        assert!(
            rebuilds >= 2,
            "expected repeated rebuilds on a moving system, got {rebuilds}"
        );
        let pos_end = array![[0.0, 0.0, 0.0], [x, 0.0, 0.0]];
        let mut fresh = make(pos_end.view());
        let reference = fresh.eval_force(pos_end.view()).unwrap();
        let marched = last.unwrap();
        assert!((marched.energy - reference.energy).abs() < 1e-12);
        assert!(arrays_close(
            marched.forces.view(),
            reference.forces.view(),
            1e-12
        ));
    }
}

#[cfg(test)]
mod ghost_path_tests {
    use super::super::forces::{GhostPairs, MicPairs};
    use super::*;
    use crate::core::Cell;
    use molrs::core::SimBox;
    use molrs::core::{NeighborList, NeighborPolicy, VerletSkin};
    use molrs::ff::potential::ForceTerm;
    use molrs::ff::potential::pair::PairLjCut;
    use ndarray::array;

    use molrs::core::GhostHalo;

    /// A halo that outlives a fold has to be reconciled with it, and the
    /// observable consequence is that nothing happens: the energy does not jump
    /// when an atom crosses a face.
    ///
    /// This is deliberately *not* a comparison against the minimum-image route.
    /// A skin makes the two differ by list staleness, and with the skin set to
    /// zero — as the comparison test must — the halo is rebuilt every step and
    /// the reconciliation is never reached. So the property is asserted on its
    /// own terms: a crossing is not a physical event, so no physical quantity
    /// may notice one.
    ///
    /// Without `s_g += m` a copy jumps a whole cell when its owner folds, pairs
    /// appear and vanish, and the energy steps by kcal/mol between one
    /// femtosecond and the next.
    #[test]
    fn crossing_a_face_does_not_disturb_the_energy() {
        let l = 12.0_f64;
        let cutoff = 5.0;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();

        // Sitting a tenth of an Ångström inside the +x face, so the crossing
        // happens within the first few steps and the configuration has no time
        // to evolve far from where it started. Neighbours are still 4 Å apart
        // the short way round (12 − 11.9 + 3.9).
        let pos0 = array![
            [11.9_f64, 11.9, 11.9],
            [3.9, 11.9, 11.9],
            [11.9, 3.9, 11.9],
            [11.9, 11.9, 3.9],
            [3.9, 3.9, 11.9],
            [3.9, 11.9, 3.9],
            [11.9, 3.9, 3.9],
            [3.9, 3.9, 3.9],
        ];
        // A uniform drift along +x, so every atom crosses. The
        // lattice is not in equilibrium — on each axis an atom has a neighbour
        // at 4 Å one way and 8 Å the other — so the potential energy genuinely
        // evolves, and asserting it is *constant* would be wrong. What must
        // hold is that the total energy is conserved: NVE has no term that
        // could absorb a pair appearing or vanishing.
        let drift = 0.02_f64;
        let vel0 =
            Fnx3::from_shape_fn((pos0.nrows(), 3), |(_, k)| if k == 0 { drift } else { 0.0 });

        // A skin large enough that the halo survives many steps, so folds
        // happen *between* rebuilds — the case the reconciliation exists for.
        let halo = GhostHalo::new(bx.clone(), pos0.view(), cutoff, 0.8).unwrap();
        // 0.2 fs: at 1 fs the velocity-Verlet truncation error on this LJ
        // lattice is itself 1e-4 of the total energy, which would swamp the
        // signal this test is looking for.
        let mut ig = VelocityVerlet::new(
            0.2,
            GhostPairs::new(
                ForceTerm::pair(PairLjCut::new(0.3, 3.4, cutoff, 12, 6, false, false).unwrap()),
                halo,
            )
            .unwrap(),
            uniform_masses(12.0, pos0.nrows()).unwrap().view(),
            Some(bx),
        )
        .unwrap();

        let mass = uniform_masses(12.0, pos0.nrows()).unwrap();
        let mut state = ig.initial(pos0, vel0).unwrap();
        // The md test is unit-agnostic; the SI preset's mass·velocity² →
        // energy factor is exactly 1, so this is the bare ½ m v².
        let kinetic = crate::compute::KineticEnergy::new(crate::core::UnitPreset::si()).unwrap();
        let total = |st: &MdState| {
            use crate::compute::Compute;
            st.energy
                + kinetic
                    .compute(crate::compute::KineticEnergyInput {
                        masses: mass.view(),
                        velocities: st.vel.view(),
                    })
                    .unwrap()
                    .energy
        };
        let e0 = total(&state);
        let scale = e0.abs().max(1.0);

        for step in 1..=200 {
            state = ig.advance(state).unwrap();
            let drift = (total(&state) - e0).abs() / scale;
            // The bar is set by what the defect does, not by what looks tidy.
            // A copy that jumps a cell removes or adds a pair worth ~0.1
            // kcal/mol from a total near 4, i.e. ~2.5e-2 of relative change in
            // a single step — twenty-five times this bound. What remains below
            // it is the integrator's own truncation error, which is a property
            // of velocity-Verlet and not of the ghost layer.
            assert!(
                drift < 1e-3,
                "step {step}: total energy drifted by {drift} (relative). \
                 A copy that jumped a cell makes pairs appear and vanish, and \
                 NVE has nothing to absorb that"
            );
        }

        // The run has to have exercised the thing it claims to test.
        // The drift carries the run 0.8 Å along +x, so the four atoms that
        // start a tenth of an Ångström inside the face cross and the four at
        // 3.9 do not. Four folds are what this exercises.
        let crossed: I = state.images.iter().map(|&m| m.abs()).sum();
        assert!(crossed >= 4, "atoms should have crossed; got {crossed}");
        let rebuilds = ig
            .forces()
            .neighbor_stats()
            .rebuilds
            .expect("a ghost provider counts its rebuilds");
        assert!(
            rebuilds < 200,
            "the halo must survive some folds, or the reconciliation is never reached"
        );
    }

    /// The virial reaches the integrator, and it is a property of the
    /// configuration rather than of where the cell's origin happens to fall.
    ///
    /// A rigid translation of a periodic system is a symmetry: the same atoms
    /// at the same separations, so the same physics. It is emphatically *not*
    /// the same bookkeeping — moving everything a third of a cell puts a
    /// different set of atoms near the faces, so a different set of copies is
    /// materialised and the pairs are found through different images. A virial
    /// that moved under it would be reporting the halo instead of the physics.
    ///
    /// The translation is deliberately not a lattice vector. Shifting by a
    /// whole cell and wrapping gives back bit-identical coordinates, which
    /// asserts nothing about the halo at all.
    #[test]
    fn the_integrator_reports_a_virial_that_a_rigid_translation_cannot_move() {
        let l = 12.0_f64;
        let cutoff = 5.0;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let base = array![
            [11.0_f64, 11.0, 11.0],
            [3.0, 11.0, 11.0],
            [11.0, 3.0, 11.0],
            [11.0, 11.0, 3.0],
            [3.0, 3.0, 11.0],
            [3.0, 11.0, 3.0],
            [11.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
        ];
        let n = base.nrows();
        let mass = uniform_masses(12.0, n).unwrap();

        let virial_after_one_step = |shift: F| {
            let mut pts = base.clone();
            pts.iter_mut().for_each(|x| *x += shift);
            let (wrapped, _m) = bx.wrap(pts.view());
            let halo = GhostHalo::new(bx.clone(), wrapped.view(), cutoff, 0.0).unwrap();
            let mut ig = VelocityVerlet::new(
                1.0,
                GhostPairs::new(
                    ForceTerm::pair(PairLjCut::new(0.3, 3.4, cutoff, 12, 6, false, false).unwrap()),
                    halo,
                )
                .unwrap(),
                mass.view(),
                Some(bx.clone()),
            )
            .unwrap();
            let state = ig.initial(wrapped, Fnx3::zeros((n, 3))).unwrap();
            state
                .virial
                .expect("the ghost provider tallies a virial, and the state keeps it")
        };

        let a = virial_after_one_step(0.0);
        let b = virial_after_one_step(l / 3.0);
        let scale = a.components.iter().fold(1.0_f64, |m, c| m.max(c.abs()));
        assert!(
            scale > 1.0,
            "the virial must be non-trivial for this test to mean anything"
        );
        for c in 0..6 {
            assert!(
                (a.components[c] - b.components[c]).abs() / scale < 1e-10,
                "component {c}: {} vs {} after a rigid translation",
                a.components[c],
                b.components[c]
            );
        }
    }

    /// The two régimes agree on the virial, having derived it two different
    /// ways.
    ///
    /// The ghost route sums `Σ_a f_a ⊗ x_a` over owned atoms *and copies*,
    /// each at its own position, before the copies' forces are folded back.
    /// The minimum-image route never has a copy: each pair kernel tallies
    /// `Σ f_ij ⊗ r_ij` inside the loop that made the forces, from the folded
    /// displacement. Those are the same tensor by an identity, not by
    /// construction — so agreeing is evidence, where a shared code path would
    /// have been none.
    #[test]
    fn the_two_regimes_derive_the_same_virial() {
        use molrs::ff::ir::CombiningRule;

        let l = 12.0_f64;
        let cutoff = 5.0;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // Every edge of this cube runs through a face, so every pair is found
        // through a periodic image and the two derivations have to work for it.
        let pos = array![
            [11.0_f64, 11.0, 11.0],
            [3.0, 11.0, 11.0],
            [11.0, 3.0, 11.0],
            [11.0, 11.0, 3.0],
            [3.0, 3.0, 11.0],
            [3.0, 11.0, 3.0],
            [11.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
        ];
        let n = pos.nrows();
        let per_type = [(0.3_f64, 3.4_f64), (0.9, 2.6)];
        let type_id: Vec<u32> = (0..n).map(|i| (i % 2) as u32).collect();
        let lj = || {
            PairLjCut::typed(
                type_id.clone(),
                &per_type,
                CombiningRule::Arithmetic,
                cutoff,
                12,
                6,
                false,
                false,
            )
            .unwrap()
        };
        let no_fold = Array2::<I>::zeros((n, 3));

        let skin = VerletSkin::new(
            NeighborList::new(cutoff),
            cutoff,
            NeighborPolicy {
                skin: 0.0,
                ..NeighborPolicy::default()
            },
            pos.view(),
            bx.clone(),
        )
        .unwrap();
        let mic = MicPairs::new(ForceTerm::pair(lj()), skin)
            .unwrap()
            .compute(pos.view(), no_fold.view())
            .unwrap()
            .virial
            .expect("a typed pair kernel tallies its virial");

        let halo = GhostHalo::new(bx, pos.view(), cutoff, 0.0).unwrap();
        let ghost = GhostPairs::new(ForceTerm::pair(lj()), halo)
            .unwrap()
            .compute(pos.view(), no_fold.view())
            .unwrap()
            .virial
            .expect("the halo tallies one too");

        let scale = mic.components.iter().fold(1.0_f64, |m, c| m.max(c.abs()));
        assert!(
            scale > 1.0,
            "the virial must be non-trivial for this to assert anything"
        );
        for c in 0..6 {
            assert!(
                (mic.components[c] - ghost.components[c]).abs() / scale < 1e-12,
                "component {c}: {} through the image vs {} through copies",
                mic.components[c],
                ghost.components[c]
            );
        }
    }
}

#[cfg(test)]
mod wrapped_state_tests {
    use super::super::forces::SelfPairedForces;
    use super::*;
    use molrs::ff::potential::Potentials;
    use ndarray::array;

    /// A free-boundary run has no cell to fold into, so nothing is wrapped and
    /// the flags stay zero — the same integrator, with the periodic layer
    /// switched off rather than special-cased downstream.
    #[test]
    fn free_boundary_leaves_positions_and_flags_alone() {
        let mut ig = VelocityVerlet::new(
            1.0,
            SelfPairedForces::new(Potentials::new()),
            uniform_masses(1.0, 1).unwrap().view(),
            None,
        )
        .unwrap();
        let mut state = ig
            .initial(array![[0.0, 0.0, 0.0]], array![[3.0, 0.0, 0.0]])
            .unwrap();
        for _ in 0..50 {
            state = ig.advance(state).unwrap();
        }
        assert!(state.pos[[0, 0]] > 100.0, "nothing folded it back");
        assert_eq!(state.images.row(0).to_vec(), vec![0 as I; 3]);
    }
}
