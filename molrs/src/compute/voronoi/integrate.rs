//! Voronoi integration of a volumetric electron density into per-molecule
//! electromagnetic moments (charge + dipole).

use molrs::core::neighbors::Backend;
use molrs::core::{LinkCell, Mic, SimBox};
use molrs::op::F;
use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3};

use crate::compute::RegularGrid;
use crate::compute::named_enum::named_enum;
use crate::compute::shape::group_reduce::GroupReduce;
use crate::compute::{Compute, ComputeError, Partition};
use crate::core::unit_factors::BOHR3_TO_ANGSTROM3;
use crate::op::vec3::sub;

named_enum! {
    /// The length unit a volumetric density is expressed in: the density is
    /// electrons per cubic unit. A Gaussian cube's volumetric block is in
    /// atomic units (`bohr`, e/Bohr³) by convention, even when its geometry is
    /// given in Å.
    pub enum LengthUnit {
        /// Å: the density is in e/Å³.
        Angstrom => "angstrom",
        /// Bohr: the density is in e/Bohr³ (atomic units).
        Bohr => "bohr",
    }
}

/// Per-molecule electromagnetic moments for one frame.
#[derive(Debug, Clone)]
pub struct MolecularMoments {
    /// The group (molecule) key of each row, ascending.
    pub groups: Vec<u64>,
    /// Molecular charge `Q_m` (e), length `n_mol`.
    pub charges: Vec<F>,
    /// Molecular dipole `μ_m` (e·Å), shape `(n_mol, 3)`.
    pub dipoles: Array2<F>,
    /// Reference point used per molecule (centre of nuclear charge), `(n_mol, 3)`.
    pub references: Array2<F>,
}

/// Voronoi electron-density integrator.
///
/// Configuration: the [`LengthUnit`] of the density values — there is no
/// default; a cube's volumetric block is `Bohr`. Data: the generators (positions,
/// radii, nuclear charges), the molecule [`Partition`], the box, and the
/// density sampled on the corners of a [`RegularGrid`] (possibly sheared).
///
/// Ported from the reference implementation's Voronoi charge/dipole gathering (`CalcVoronoiCharges` /
/// dipole accumulation in `src/gather.cpp`), which assigns each density grid
/// point to its enclosing radical-Voronoi cell and sums the electronic charge
/// `q = −∫ρ dV` and dipole `μ = −∫ρ (r − r_ref) dV` per cell, then per molecule.
/// Cube-frame Bohr/atomic-unit conventions follow `src/bqb_cubeframe.cpp`.
///
/// # Definitions (Thomas, Brehm, Kirchner, *PCCP* 2015, 17, 3207)
///
/// With electron (number) density `ρ ≥ 0`:
/// - cell electronic population `Nᵃ = ∫_cell ρ dV` (electrons, ≥ 0);
/// - molecular charge `Q_m = Σ_{a∈m} (Z_a − Nᵃ)`;
/// - molecular dipole `μ_m = Σ_a Z_a (r_a − r_ref) − Σ_{a∈m} ∫_cell ρ (r − r_ref) dV`,
///   with `r_ref` the molecule's centre of nuclear charge (documented;
///   origin-dependent for `Q_m ≠ 0`).
///
/// # Cell assignment
///
/// A point `x` belongs to the radical cell of generator `i` iff `i` minimises
/// the power distance `|x − x_i|² − R_i²` — this argmin **is** the radical
/// Voronoi partition (same cells [`RadicalVoronoi`](super::RadicalVoronoi)
/// builds geometrically), so the integrator reuses the generators + radii
/// directly. Exact ties break to the lowest index (deterministic). All
/// displacements use the box's minimum image ([`SimBox::mic`]), so a molecule
/// straddling a periodic boundary integrates correctly.
///
/// # Units
///
/// Positions and the grid geometry are in Å. The density is converted from
/// e/unit³ to e/Å³ by the configured [`LengthUnit`] (`Bohr`: divide by `a³`,
/// `a = 0.529177… Å/Bohr`), so `∫ρ dV` is a pure electron count and `μ` is in
/// `e·Å`.
///
/// ```
/// use molrs::compute::{
///     Compute, LengthUnit, RegularGrid, VoronoiIntegration, VoronoiIntegrationInput,
/// };
/// use molrs::core::SimBox;
/// use ndarray::{Array3, array};
///
/// // One hydrogen-like atom, one electron spread uniformly over the box.
/// let simbox = SimBox::cube(2.0, array![0.0, 0.0, 0.0], [true; 3]).unwrap();
/// let grid = RegularGrid::over_box(&simbox, [4, 4, 4])?;
/// let density = Array3::from_elem((4, 4, 4), 1.0 / 8.0); // e/Å³
/// let moments = VoronoiIntegration::new(LengthUnit::Angstrom).compute(
///     VoronoiIntegrationInput {
///         positions: array![[1.0, 1.0, 1.0]].view(),
///         radii: array![0.0].view(),
///         nuclear_charges: array![1.0].view(),
///         partition: &vec![0u32],
///         simbox: &simbox,
///         grid: &grid,
///         density: density.view(),
///     },
/// )?;
/// assert!(moments.charges[0].abs() < 1e-12);
/// # Ok::<(), molrs::compute::ComputeError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct VoronoiIntegration {
    unit: LengthUnit,
}

/// The data a [`VoronoiIntegration`] integrates.
#[derive(Clone, Copy)]
pub struct VoronoiIntegrationInput<'a> {
    /// Generator (atom) positions, `N × 3` (Å).
    pub positions: ArrayView2<'a, F>,
    /// One radical radius per generator (Å); all zero for plain Voronoi cells.
    pub radii: ArrayView1<'a, F>,
    /// One nuclear charge `Z_a` per generator (e): the atomic number, or the
    /// valence charge for a pseudopotential density.
    pub nuclear_charges: ArrayView1<'a, F>,
    /// The molecule of each generator; every generator must be in one.
    pub partition: &'a (dyn Partition + Sync),
    /// The box whose minimum image the cells and dipoles are measured under.
    pub simbox: &'a SimBox,
    /// Where the density samples sit: value `(i, j, k)` at
    /// [`voxel_corner([i, j, k])`](RegularGrid::voxel_corner), standing for one
    /// [`voxel_volume`](RegularGrid::voxel_volume).
    pub grid: &'a RegularGrid<3>,
    /// The electron density, shaped as `grid.dims()`, in e per cubic
    /// [`LengthUnit`].
    pub density: ArrayView3<'a, F>,
}

impl VoronoiIntegration {
    /// Integrate a density expressed in electrons per cubic `unit`.
    pub fn new(unit: LengthUnit) -> Self {
        Self { unit }
    }

    /// The configured density unit.
    pub fn unit(&self) -> LengthUnit {
        self.unit
    }
}

impl Compute for VoronoiIntegration {
    type Input<'a> = VoronoiIntegrationInput<'a>;
    type Output = MolecularMoments;

    /// # Errors
    /// [`ComputeError::DimensionMismatch`] when the positions are not `N × 3`
    /// or the radii, nuclear charges or partition are not one per generator;
    /// [`ComputeError::BadShape`] when the density is not shaped as the grid;
    /// [`ComputeError::EmptyInput`] without generators;
    /// [`ComputeError::NonFinite`] for a non-finite position, radius, nuclear
    /// charge or density value; [`ComputeError::OutOfRange`] for a generator in
    /// no molecule, a negative nuclear charge, or a molecule of total nuclear
    /// charge zero (its reference point is undefined).
    fn compute(
        &self,
        input: VoronoiIntegrationInput<'_>,
    ) -> Result<MolecularMoments, ComputeError> {
        let VoronoiIntegrationInput {
            positions,
            radii,
            nuclear_charges,
            partition,
            simbox,
            grid,
            density,
        } = input;
        let n = positions.nrows();
        for (what, len) in [
            ("radii against the rows of positions", radii.len()),
            (
                "nuclear charges against the rows of positions",
                nuclear_charges.len(),
            ),
        ] {
            if len != n {
                return Err(ComputeError::DimensionMismatch {
                    expected: n,
                    got: len,
                    what,
                });
            }
        }
        let dims = grid.dims();
        if density.shape() != dims {
            return Err(ComputeError::BadShape {
                expected: format!("density shaped as the grid, {dims:?}"),
                got: format!("{:?}", density.shape()),
            });
        }
        if n == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if let Some(index) = radii.iter().position(|v| !v.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "radii",
                index,
            });
        }
        // Flat index in `(i, j, k)` order, z fastest.
        if let Some(index) = density.iter().position(|v| !v.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "density values",
                index,
            });
        }

        // --- Pass A: per-molecule reference = centre of nuclear charge, each
        // molecule unwrapped about its first atom (the one group reduction). ---
        let reduce = GroupReduce::new(positions, Some(simbox), partition)?;
        let mol_of: Vec<usize> = (0..n)
            .map(|a| {
                reduce.group_of(a).ok_or_else(|| ComputeError::OutOfRange {
                    field: "partition",
                    value: format!("generator {a} is in no molecule"),
                })
            })
            .collect::<Result<_, _>>()?;
        let references = reduce
            .centroids(nuclear_charges, "nuclear charges")?
            .centers;
        let n_mol = reduce.keys().len();
        let mic = simbox.mic();

        // Hoist the generator positions into a contiguous `[F; 3]` buffer and
        // precompute the squared radii once — Pass B's voxel loop reads both
        // millions of times.
        let gens: Vec<[F; 3]> = positions
            .rows()
            .into_iter()
            .map(|r| [r[0], r[1], r[2]])
            .collect();
        let radii_sq: Vec<F> = radii.iter().map(|&r| r * r).collect();

        // --- Nuclear contributions to charge + dipole. ---
        let mut charges = vec![0.0_f64; n_mol];
        let mut dipoles = Array2::<F>::zeros((n_mol, 3));
        for a in 0..n {
            let m = mol_of[a];
            let z = nuclear_charges[a];
            charges[m] += z;
            let rref = [references[[m, 0]], references[[m, 1]], references[[m, 2]]];
            let d = mic.apply(sub(gens[a], rref));
            for c in 0..3 {
                dipoles[[m, c]] += z * d[c];
            }
        }

        // --- Pass B: assign each voxel to its radical cell (power-distance
        // argmin), accumulate electronic population + dipole on the owning
        // molecule. ---
        let locator = PowerCellLocator::new(positions, &radii_sq, simbox)?;
        let scale = match self.unit {
            LengthUnit::Angstrom => 1.0,
            LengthUnit::Bohr => 1.0 / BOHR3_TO_ANGSTROM3.get(),
        };
        let dv = grid.voxel_volume();
        let (ny, nz) = (dims[1], dims[2]);
        let n_voxels = dims[0] * ny * nz;

        // Electronic contribution of one voxel (flat index `m`, z fastest),
        // deposited into per-molecule charge / dipole accumulators. Each voxel
        // is independent, so this is the body of both the serial loop and the
        // rayon fold below.
        let deposit = |m: usize, charge: &mut [F], dip: &mut Array2<F>| {
            let index = [m / (ny * nz), (m / nz) % ny, m % nz];
            let rho = density[index] * scale; // e/Å³
            if rho == 0.0 {
                return;
            }
            let x = grid.voxel_corner(index);
            let mol = mol_of[locator.nearest(x, &gens)];
            let n_elec = rho * dv; // electrons in this voxel
            charge[mol] -= n_elec;
            let rref = [
                references[[mol, 0]],
                references[[mol, 1]],
                references[[mol, 2]],
            ];
            let disp = mic.apply(sub(x, rref));
            for c in 0..3 {
                dip[[mol, c]] -= n_elec * disp[c];
            }
        };

        // Nuclear terms are already in `charges`/`dipoles`; the electronic sum
        // lands in its own accumulators (fanned out over voxels with rayon, then
        // merged) and is added on afterwards, so nuclear-then-electronic
        // ordering is preserved and only the electronic sum is reassociated.
        let zeros = || (vec![0.0_f64; n_mol], Array2::<F>::zeros((n_mol, 3)));
        #[cfg(feature = "rayon")]
        let (elec_charge, elec_dipole) = {
            use rayon::prelude::*;
            (0..n_voxels)
                .into_par_iter()
                .fold(zeros, |(mut c, mut d), m| {
                    deposit(m, &mut c, &mut d);
                    (c, d)
                })
                .reduce(zeros, |(mut ca, mut da), (cb, db)| {
                    for (x, y) in ca.iter_mut().zip(cb.iter()) {
                        *x += *y;
                    }
                    da += &db;
                    (ca, da)
                })
        };
        #[cfg(not(feature = "rayon"))]
        let (elec_charge, elec_dipole) = {
            let (mut c, mut d) = zeros();
            for m in 0..n_voxels {
                deposit(m, &mut c, &mut d);
            }
            (c, d)
        };

        for m in 0..n_mol {
            charges[m] += elec_charge[m];
            for c in 0..3 {
                dipoles[[m, c]] += elec_dipole[[m, c]];
            }
        }

        Ok(MolecularMoments {
            groups: reduce.keys().to_vec(),
            charges,
            dipoles,
            references,
        })
    }
}

/// Radical (power / Laguerre) distance of point `x` to generator `g` with
/// squared radius `r_sq` under the minimum image `mic`: `|mic(g − x)|² − r_sq`,
/// the same displacement and the same arithmetic the core cell list reports, so
/// the indexed search and the exhaustive fallback compare identical values.
#[inline]
fn power_dist(x: [F; 3], g: [F; 3], r_sq: F, mic: &Mic) -> F {
    let d = mic.apply(sub(g, x));
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - r_sq
}

/// The generator that owns a point under the power distance, found through a
/// core cell list ([`LinkCell`]) over the generators.
///
/// The cell list gathers the generators in the point's cell and its 26
/// neighbours; every generator outside them lies farther than the cutoff
/// `r_cut` (cells are at least `r_cut` wide), hence has power distance
/// `≥ r_cut² − w_max` (`w_max` the largest squared radius). If a gathered
/// candidate already beats that bound the gathered argmin is provably the
/// global argmin; otherwise the search falls back to the exhaustive scan, so
/// every result is bit-identical to brute force, whatever `r_cut` is.
struct PowerCellLocator<'a> {
    index: LinkCell,
    simbox: &'a SimBox,
    mic: Mic,
    radii_sq: &'a [F],
    w_max: F,
    r_cut: F,
}

impl<'a> PowerCellLocator<'a> {
    /// Index `positions` (at least one) in `simbox` with a cutoff generous
    /// enough that the gather almost always contains the winning generator:
    /// a few mean spacings plus the radius spread.
    fn new(
        positions: ArrayView2<'_, F>,
        radii_sq: &'a [F],
        simbox: &'a SimBox,
    ) -> Result<Self, ComputeError> {
        let n = positions.nrows();
        let w_max = radii_sq.iter().copied().fold(0.0_f64, f64::max);
        let mean_spacing = (simbox.volume() / n as F).cbrt();
        Self::with_cutoff(
            positions,
            radii_sq,
            simbox,
            3.0 * mean_spacing + 2.0 * w_max.sqrt(),
        )
    }

    fn with_cutoff(
        positions: ArrayView2<'_, F>,
        radii_sq: &'a [F],
        simbox: &'a SimBox,
        r_cut: F,
    ) -> Result<Self, ComputeError> {
        if !(r_cut.is_finite() && r_cut > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "VoronoiIntegration search cutoff",
                value: r_cut.to_string(),
            });
        }
        let mut index = LinkCell::new().cutoff(r_cut);
        index.build_index(positions, simbox);
        Ok(Self {
            index,
            simbox,
            mic: simbox.mic(),
            radii_sq,
            w_max: radii_sq.iter().copied().fold(0.0_f64, f64::max),
            r_cut,
        })
    }

    /// Index of the generator minimising the power distance to `x`, identical
    /// (value and lowest-index tie-break) to the exhaustive argmin.
    fn nearest(&self, x: [F; 3], gens: &[[F; 3]]) -> usize {
        let mut best = usize::MAX;
        let mut best_pow = F::INFINITY;
        self.index
            .visit_neighbors_of_pt(x, self.simbox, |a, d2, _| {
                let a = a as usize;
                let pow = d2 - self.radii_sq[a];
                // Argmin with a lowest-index tie-break, tolerant of the
                // grouped (non-index-order) visiting: strictly smaller wins,
                // exact ties go to the lower index.
                if pow < best_pow || (pow <= best_pow && a < best) {
                    best_pow = pow;
                    best = a;
                }
            });

        // Pruning guard: any generator not gathered is farther than r_cut, so
        // its power is ≥ r_cut² − w_max. A candidate strictly below that bound
        // is the global argmin.
        if best != usize::MAX && self.r_cut * self.r_cut - self.w_max > best_pow {
            return best;
        }

        // Fallback: exhaustive argmin over all generators (strict `<`, so the
        // lowest index wins a tie).
        let mut fb_best = 0usize;
        let mut fb_pow = F::INFINITY;
        for (a, &g) in gens.iter().enumerate() {
            let pow = power_dist(x, g, self.radii_sq[a], &self.mic);
            if pow < fb_pow {
                fb_pow = pow;
                fb_best = a;
            }
        }
        fb_best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3, array};

    fn periodic(l: [F; 3]) -> SimBox {
        SimBox::ortho(array![l[0], l[1], l[2]], array![0.0, 0.0, 0.0], [true; 3]).unwrap()
    }

    #[test]
    fn min_image_wraps_half_box() {
        let d = Mic::ortho([1.0, 1.0, 1.0]).apply([0.9, 0.0, 0.0]);
        assert!((d[0] - (-0.1)).abs() < 1e-12);
    }

    /// The indexed nearest-generator search must return **exactly** the same
    /// index (value + lowest-index tie-break) as the exhaustive argmin for
    /// every query point — with the pruning guard active (several cells), when
    /// it must fall back (tiny `r_cut`), and in the single-cell case.
    #[test]
    fn cell_list_nearest_matches_brute_force() {
        let lbox = [12.0_f64, 10.0, 8.0];
        let simbox = periodic(lbox);
        let n = 200usize;
        // Deterministic LCG → f64 in [0, 1); no external rng dependency.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut rng = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let gens: Vec<[F; 3]> = (0..n)
            .map(|_| [rng() * lbox[0], rng() * lbox[1], rng() * lbox[2]])
            .collect();
        let radii_sq: Vec<F> = (0..n).map(|_| (rng() * 1.5).powi(2)).collect();
        let positions = Array2::from_shape_fn((n, 3), |(i, k)| gens[i][k]);
        let mic = Mic::ortho(lbox);

        // Reference: byte-for-byte the exhaustive search.
        let brute = |x: [F; 3]| -> usize {
            let mut best = 0usize;
            let mut best_pow = F::INFINITY;
            for a in 0..n {
                let pow = power_dist(x, gens[a], radii_sq[a], &mic);
                if pow < best_pow {
                    best_pow = pow;
                    best = a;
                }
            }
            best
        };

        // Query points spanning (and spilling outside) the box to exercise wrap.
        let queries: Vec<[F; 3]> = (0..600)
            .map(|_| {
                [
                    rng() * lbox[0] * 1.3 - 1.0,
                    rng() * lbox[1] * 1.3 - 1.0,
                    rng() * lbox[2] * 1.3 - 1.0,
                ]
            })
            .collect();

        // r_cut = 0.6 → tiny cells, frequent fallback; 3.0/5.0 → guard prunes;
        // 50.0 → single cell.
        for &r_cut in &[0.6_f64, 3.0, 5.0, 50.0] {
            let locator =
                PowerCellLocator::with_cutoff(positions.view(), &radii_sq, &simbox, r_cut).unwrap();
            for &x in &queries {
                assert_eq!(
                    locator.nearest(x, &gens),
                    brute(x),
                    "spatial index diverged from brute force at r_cut={r_cut}, x={x:?}"
                );
            }
        }
    }

    struct Case {
        simbox: SimBox,
        grid: RegularGrid<3>,
        density: Array3<F>,
        positions: Array2<F>,
        radii: Array1<F>,
        z: Array1<F>,
        mol: Vec<u32>,
    }

    impl Case {
        /// Two neutral atoms (Z = 1) in one molecule, 2 Å apart along x,
        /// with one electron each on the voxel corner nearest each atom.
        fn diatomic() -> Self {
            let simbox = periodic([8.0; 3]);
            let grid = RegularGrid::over_box(&simbox, [8, 8, 8]).unwrap();
            let mut density = Array3::zeros((8, 8, 8));
            density[[3, 4, 4]] = 1.0; // corner (3, 4, 4) Å, voxel volume 1 Å³
            density[[5, 4, 4]] = 1.0;
            Self {
                simbox,
                grid,
                density,
                positions: array![[3.0, 4.0, 4.0], [5.0, 4.0, 4.0]],
                radii: array![0.0, 0.0],
                z: array![1.0, 1.0],
                mol: vec![0, 0],
            }
        }

        fn run(&self, unit: LengthUnit) -> Result<MolecularMoments, ComputeError> {
            VoronoiIntegration::new(unit).compute(VoronoiIntegrationInput {
                positions: self.positions.view(),
                radii: self.radii.view(),
                nuclear_charges: self.z.view(),
                partition: &self.mol,
                simbox: &self.simbox,
                grid: &self.grid,
                density: self.density.view(),
            })
        }
    }

    /// Electrons sitting on their nuclei: neutral, no dipole, reference at the
    /// centre of nuclear charge.
    #[test]
    fn electrons_on_the_nuclei_give_a_neutral_nonpolar_molecule() {
        let m = Case::diatomic().run(LengthUnit::Angstrom).unwrap();
        assert_eq!(m.groups, vec![0]);
        assert!(m.charges[0].abs() < 1e-12);
        assert!(m.dipoles.iter().all(|d| d.abs() < 1e-12));
        assert_eq!(m.references.row(0).to_vec(), vec![4.0, 4.0, 4.0]);
    }

    /// Moving one electron from atom 1 onto atom 0 (two molecules) charges the
    /// molecules ±1 e; the unit only scales the electron count.
    #[test]
    fn a_transferred_electron_charges_each_molecule() {
        let mut case = Case::diatomic();
        case.mol = vec![0, 1];
        case.density[[3, 4, 4]] = 2.0;
        case.density[[5, 4, 4]] = 0.0;
        let m = case.run(LengthUnit::Angstrom).unwrap();
        assert!((m.charges[0] + 1.0).abs() < 1e-12);
        assert!((m.charges[1] - 1.0).abs() < 1e-12);

        let bohr = case.run(LengthUnit::Bohr).unwrap();
        let a3 = BOHR3_TO_ANGSTROM3.get();
        assert!((bohr.charges[0] - (1.0 - 2.0 / a3)).abs() < 1e-12);
    }

    /// One electron displaced by +1 Å along x from a Z = 1 nucleus: μ = −1 e·Å.
    #[test]
    fn a_displaced_electron_gives_the_textbook_dipole() {
        let mut case = Case::diatomic();
        case.positions = array![[3.0, 4.0, 4.0]];
        case.radii = array![0.0];
        case.z = array![1.0];
        case.mol = vec![0];
        case.density = Array3::zeros((8, 8, 8));
        case.density[[4, 4, 4]] = 1.0;
        let m = case.run(LengthUnit::Angstrom).unwrap();
        assert!(m.charges[0].abs() < 1e-12);
        assert!((m.dipoles[[0, 0]] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_unit_names_are_canonical() {
        assert_eq!("bohr".parse::<LengthUnit>().unwrap(), LengthUnit::Bohr);
        assert_eq!(LengthUnit::Angstrom.as_str(), "angstrom");
        assert!(matches!(
            "Bohr".parse::<LengthUnit>(),
            Err(ComputeError::UnknownName { .. })
        ));
    }

    #[test]
    fn a_density_not_shaped_as_the_grid_is_an_error() {
        let mut case = Case::diatomic();
        case.density = Array3::zeros((8, 8, 7));
        assert!(matches!(
            case.run(LengthUnit::Bohr),
            Err(ComputeError::BadShape { .. })
        ));
    }

    #[test]
    fn non_finite_inputs_are_errors() {
        let mut case = Case::diatomic();
        case.density[[0, 0, 1]] = F::NAN;
        assert!(matches!(
            case.run(LengthUnit::Bohr),
            Err(ComputeError::NonFinite {
                where_: "density values",
                index: 1
            })
        ));

        let mut case = Case::diatomic();
        case.positions[[1, 0]] = F::INFINITY;
        assert!(matches!(
            case.run(LengthUnit::Bohr),
            Err(ComputeError::NonFinite {
                where_: "positions",
                ..
            })
        ));
    }

    /// The former silent fallbacks: an atom in no molecule and a molecule
    /// with no nuclear charge (its reference was the first atom).
    #[test]
    fn unassigned_atoms_and_chargeless_molecules_are_errors() {
        let mut case = Case::diatomic();
        let partial: Vec<Option<u32>> = vec![Some(0), None];
        let err = VoronoiIntegration::new(LengthUnit::Bohr)
            .compute(VoronoiIntegrationInput {
                positions: case.positions.view(),
                radii: case.radii.view(),
                nuclear_charges: case.z.view(),
                partition: &PartialLabels(partial),
                simbox: &case.simbox,
                grid: &case.grid,
                density: case.density.view(),
            })
            .unwrap_err();
        assert!(matches!(
            err,
            ComputeError::OutOfRange {
                field: "partition",
                ..
            }
        ));

        case.z = array![0.0, 0.0];
        assert!(matches!(
            case.run(LengthUnit::Bohr),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    #[test]
    fn no_generators_is_empty_input() {
        let mut case = Case::diatomic();
        case.positions = Array2::zeros((0, 3));
        case.radii = Array1::zeros(0);
        case.z = Array1::zeros(0);
        case.mol = vec![];
        assert!(matches!(
            case.run(LengthUnit::Bohr),
            Err(ComputeError::EmptyInput)
        ));
    }

    /// A partition that leaves some items ungrouped.
    struct PartialLabels(Vec<Option<u32>>);

    impl Partition for PartialLabels {
        fn n_items(&self) -> usize {
            self.0.len()
        }
        fn group_of(&self, i: usize) -> Option<u64> {
            self.0[i].map(u64::from)
        }
    }
}
