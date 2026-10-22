//! The completeness of the force-field IR: every registered kernel and every
//! field-level setting, against every engine format molrs reads or writes
//! and the record (molrec v2), each cell exact — and the tests that hold it —
//! or refused by name, or not a thing the format has.
//!
//! The table in `docs/developer/forcefield-ir.md` ("Completeness")
//! is rendered from [`MATRIX`] and checked against it, so it cannot go stale:
//!
//! - every kernel [`BuiltinKernels::builtin`] registers has a row, and every
//!   style row is a registered kernel;
//! - every test a cell cites exists (`fn <name>(` in the named file);
//! - every Class-I style of molrec's registry is a registered kernel or a
//!   style `compile` refuses by name;
//! - every registered style goes through a record section and back.
//!
//! `MOLRS_WRITE_COMPLETENESS=1 cargo mrs-test -- ff::completeness` rewrites the
//! guide's table from [`MATRIX`].

use std::path::Path;

use crate::ff::compile::PotentialCompiler;
use crate::ff::forcefield::ForceField;
use crate::ff::ir::Params;
use crate::ff::style_registry::BuiltinKernels;

/// The columns, in order.
const COLUMNS: [&str; 9] = [
    "kernel",
    "LAMMPS read",
    "LAMMPS write",
    "OpenMM read",
    "OpenMM write",
    "GROMACS read",
    "GROMACS write",
    "prmtop read",
    "molrec v2",
];

/// One cell. Tests are `path::fn`, `path` relative to `crates/molrs/src`.
#[derive(Clone, Copy)]
enum Cell {
    /// Exact: priced, read or written as the IR means it.
    Exact(&'static [&'static str]),
    /// Exact within a stated bound or convention.
    ExactWhere(&'static str, &'static [&'static str]),
    /// Refused by name, and why.
    Refused(&'static str, &'static [&'static str]),
    /// The format has no such thing.
    Na(&'static str),
}

use Cell::{Exact, ExactWhere, Na, Refused};

/// One row: a `category style` the registry knows, or a field-level item.
struct Row {
    item: &'static str,
    cells: [Cell; 9],
}

// The tests most cells share.
const EQUIV: &str = "ff/equivalence_check.rs::every_engine_prices_every_source_as_molrs";
const BACK: &str = "ff/equivalence_check.rs::every_written_file_reads_back_as_written";
const RECORD: &str = "ff/completeness.rs::every_registered_style_persists_through_a_record";
const RECORD_SRC: &str = "ff/equivalence_check.rs::every_source_persists_through_a_record";
const RT_L: &str =
    "io/lammps/forcefield_writer.rs::lammps_coeff_values_round_trips_through_lammps_coeff_params";
const RT_O: &str = "ff/openmm_check.rs::read_write_read_is_the_identity";
const OMM: &str = "ff/openmm_check.rs::every_term_is_openmm_s";
const GMX: &str =
    "io/gromacs/top_reader/engine_check.rs::gromacs_read_systems_price_as_gromacs_and_lammps";
const GMX_RT: &str =
    "io/gromacs/top_reader/engine_check.rs::fixture_directives_survive_write_then_read";
const GMX_SYS: &str = "io/gromacs/top_writer.rs::a_system_reads_back_as_written";
const PRMTOP: &str = "io/amber/prmtop_check.rs::each_term_matches_sander_and_lammps";
const SERIES: &str = "ff/ir/form/torsion.rs::registered_kernels_price_the_series";
const NO_OMM: &str = "io/openmm_xml/writer.rs::styles_without_an_openmm_form_are_refused_by_name";
const NO_LMP: &str = "io/lammps/forcefield_writer.rs::lammps_coeff_values_rejects_unsupported_kernel_and_missing_param";
const NO_LMP_PAIR: &str =
    "io/lammps/forcefield_reader.rs::data_coeffs_unsupported_pair_hint_is_an_error";
// The engine codecs.
const CODEC_RT: &str = "ff/engine_codec_check.rs::every_builtin_codec_reads_back_what_it_writes";
const CODEC_FILE: &str =
    "ff/engine_codec_check.rs::every_builtin_round_trips_through_an_include_at_the_same_energy";
const CODEC_ENG: &str = "ff/engine_codec_check.rs::every_engine_prices_the_codec_cases_as_molrs";
const CLASS2_X: &str =
    "ff/engine_codec_check.rs::class2_cross_terms_are_written_at_zero_and_refused_otherwise";
const NO_ENGINE: &str = "ff/engine_codec_check.rs::engines_refuse_what_they_cannot_hold_by_name";
const OMM_REWRITE: &str =
    "ff/engine_codec_check.rs::the_openmm_rewrite_is_the_energy_in_openmm_units";
const RUNTIME_LMP: &str =
    "ff/engine_codec_check.rs::a_runtime_positional_style_reads_and_writes_through_its_registry";
const NO_CUSTOM_READ: &str = "io/openmm_xml/reader.rs::other_custom_forces_are_refused_by_name";
const PERSIST_CUSTOM: &str = "ff/ir/tests.rs::custom_styles_persist_to_a_fresh_process";

/// A style of molrs's own definition (MMFF94, UFF, …): no engine format has
/// it.
const fn own(kernel: &'static [&'static str]) -> [Cell; 9] {
    const WHY: &str = "molrs's own definition (a typifier's), no engine style";
    [
        Exact(kernel),
        Na(WHY),
        Refused(WHY, &[NO_LMP]),
        Na(WHY),
        Refused(WHY, &[NO_OMM]),
        Na(WHY),
        Refused(WHY, &[]),
        Na(WHY),
        Exact(&[RECORD]),
    ]
}

/// A style no engine format has a form of (the polarizable screenings
/// `thole`, `coul/tt`): priced, persisted, refused by the engine readers and
/// writers.
const fn outside(kernel: &'static [&'static str], why: &'static str) -> [Cell; 9] {
    [
        Exact(kernel),
        Refused(why, &[]),
        Refused(why, &[]),
        Na(why),
        Refused(why, &[NO_OMM]),
        Na(why),
        Refused(why, &[]),
        Na(why),
        Exact(&[RECORD]),
    ]
}

/// A Class II style: LAMMPS's `class2` through its codec, OpenMM's custom
/// force of its expression, GROMACS refused.
const fn class2(kernel: &'static [&'static str], lammps: &'static [&'static str]) -> [Cell; 9] {
    [
        Exact(kernel),
        ExactWhere(CLASS2_LMP, lammps),
        ExactWhere(CLASS2_LMP, lammps),
        Na("OpenMM has no Class II tag (a Custom*Force reads refused)"),
        ExactWhere(OMM_EXPR, &[OMM_REWRITE, CODEC_ENG]),
        Na("GROMACS has no Class II directive"),
        Refused(GMX_NONE, &[NO_ENGINE]),
        Na("AMBER is Class I"),
        Exact(&[RECORD]),
    ]
}

/// A pair style LAMMPS holds positionally (`pair_coeff i j` per row, no
/// mixing): read, written and priced by `lmp run 0`.
const fn positional_pair(kernel: &'static [&'static str]) -> [Cell; 9] {
    [
        Exact(kernel),
        Exact(&[
            CODEC_FILE,
            "io/lammps/forcefield_reader.rs::data_coeffs_pair_morse_hint_reads_through_its_codec",
        ]),
        Exact(&[CODEC_FILE, CODEC_ENG]),
        Na("OpenMM has no tag of its form"),
        Refused(NO_OMM_PAIR, &[NO_OMM]),
        Na("GROMACS has no directive of its form"),
        Refused(GMX_NONE, &[NO_ENGINE]),
        Na("AMBER's nonbonded is 12-6 Lennard-Jones"),
        Exact(&[RECORD]),
    ]
}

/// LAMMPS reads and writes it through its codec; the installed LAMMPS has
/// no CLASS2 package, so no `lmp run 0` prices it.
const CLASS2_LMP: &str = "through its codec (cross-term lines at zero, a non-zero one refused); not run by LAMMPS: the installed lmp has no CLASS2 package";
const GMX_NONE: &str = "GROMACS has no directive of its form (NoEngineForm)";
const OMM_EXPR: &str = "the CustomBondForce / CustomAngleForce of its expression, rewritten exactly (held by OpenMM on bond fene, bond morse, angle class2)";
const NO_OMM_PAIR: &str = "its parameters do not mix, so a pair's are a cross row's, which a CustomNonbondedForce has not";

const MATRIX: &[Row] = &[
    // ── bonds ──
    Row {
        item: "bond harmonic",
        cells: [
            Exact(&["ff/potential/bond/harmonic.rs::test_bond_harmonic_energy_and_force"]),
            Exact(&[RT_L, EQUIV]),
            Exact(&[RT_L, BACK]),
            Exact(&[
                "io/openmm_xml/reader.rs::reads_all_sections_with_molrs_units",
                OMM,
            ]),
            Exact(&[
                "io/openmm_xml/writer.rs::what_is_written_reads_back_with_the_same_styles_and_parameters",
                EQUIV,
            ]),
            Exact(&[
                "io/gromacs/top_reader.rs::bondtypes_funct_1_is_bond_harmonic_in_molrs_units",
                GMX,
            ]),
            Exact(&[
                "io/gromacs/top_writer.rs::bond_harmonic_is_bondtypes_code_1",
                EQUIV,
            ]),
            Exact(&[PRMTOP, EQUIV]),
            Exact(&[RECORD, RECORD_SRC]),
        ],
    },
    Row {
        item: "bond morse",
        cells: [
            Exact(&["ff/potential/bond/morse.rs::forces_match_finite_difference"]),
            Exact(&[RT_L, CODEC_FILE]),
            Exact(&[RT_L, CODEC_ENG]),
            Refused(
                "OpenMM has a Morse bond only as a CustomBondForce, which the reader refuses",
                &[NO_CUSTOM_READ],
            ),
            Exact(&[CODEC_ENG]),
            Exact(&["io/gromacs/top_reader.rs::bondtypes_funct_3_is_bond_morse_in_molrs_units"]),
            Exact(&["io/gromacs/top_writer.rs::bond_morse_is_bondtypes_code_3"]),
            Na("AMBER bonds are harmonic"),
            Exact(&[RECORD]),
        ],
    },
    Row {
        item: "bond class2",
        cells: class2(
            &["ff/potential/bond/class2.rs::energy_matches_closed_form"],
            &[CODEC_RT],
        ),
    },
    Row {
        item: "bond mmff_bond",
        cells: own(&["ff/potential/bond/mmff.rs::test_mmff_bond_stretched"]),
    },
    Row {
        item: "bond uff_bond",
        cells: own(&["ff/potential/bond/uff.rs::forces_are_the_negative_energy_gradient"]),
    },
    // ── angles ──
    Row {
        item: "angle harmonic",
        cells: [
            Exact(&[
                "ff/potential/angle/harmonic.rs::energy_is_the_lammps_formula_with_theta0_in_degrees",
            ]),
            Exact(&[RT_L, EQUIV]),
            Exact(&[RT_L, BACK]),
            Exact(&[OMM]),
            Exact(&[
                "io/openmm_xml/writer.rs::what_is_written_reads_back_with_the_same_styles_and_parameters",
                EQUIV,
            ]),
            Exact(&[
                "io/gromacs/top_reader.rs::angletypes_funct_1_is_angle_harmonic_in_molrs_units",
                GMX,
            ]),
            Exact(&[
                "io/gromacs/top_writer.rs::angle_harmonic_is_angletypes_code_1",
                EQUIV,
            ]),
            Exact(&[PRMTOP, EQUIV]),
            Exact(&[RECORD, RECORD_SRC]),
        ],
    },
    Row {
        item: "angle charmm",
        cells: [
            Exact(&[
                "ff/potential/angle/charmm.rs::three_atom_urey_bradley_agrees_with_lammps_run_0",
            ]),
            Exact(&[
                "io/lammps/forcefield_writer.rs::angle_charmm_hybrid_reads_and_writes_back_identically",
            ]),
            Exact(&[
                "io/lammps/forcefield_writer.rs::angle_charmm_hybrid_reads_and_writes_back_identically",
                EQUIV,
            ]),
            ExactWhere(
                "a wildcard Urey–Bradley row is refused",
                &[
                    "io/openmm_xml/reader.rs::urey_bradley_joins_its_angle_as_angle_charmm",
                    OMM,
                ],
            ),
            Exact(&[RT_O, EQUIV]),
            Exact(&[
                "io/gromacs/top_reader.rs::angletypes_funct_5_is_angle_charmm",
                GMX,
            ]),
            Exact(&[
                "io/gromacs/top_writer.rs::angle_charmm_is_angletypes_code_5",
                EQUIV,
            ]),
            Exact(&[
                "io/amber/prmtop_forcefield.rs::chamber_angles_are_angle_charmm_with_their_urey_bradley",
                PRMTOP,
                EQUIV,
            ]),
            Exact(&[RECORD, RECORD_SRC]),
        ],
    },
    Row {
        item: "angle class2",
        cells: class2(
            &["ff/potential/angle/class2.rs::forces_match_finite_difference"],
            &[CLASS2_X],
        ),
    },
    Row {
        item: "angle mmff_angle",
        cells: own(&["ff/potential/angle/mmff.rs::test_mmff_angle_at_equilibrium"]),
    },
    Row {
        item: "angle mmff_stbn",
        cells: own(&[
            "ff/potential/angle/mmff.rs::cubic_bend_constant_is_exactly_minus_zero_point_four",
        ]),
    },
    Row {
        item: "angle uff_angle",
        cells: own(&[
            "ff/potential/angle/uff.rs::forces_are_the_negative_energy_gradient_for_every_order",
        ]),
    },
    // ── dihedrals ──
    Row {
        item: "dihedral periodic",
        cells: [
            Exact(&[
                "ff/potential/dihedral/periodic.rs::multi_term_energy_sums",
                SERIES,
            ]),
            Exact(&[
                "io/lammps/forcefield_writer.rs::periodic_dihedral_is_written_as_fourier",
                EQUIV,
            ]),
            Exact(&[
                "io/lammps/forcefield_writer.rs::periodic_dihedral_is_written_as_fourier",
                BACK,
            ]),
            Exact(&[
                "io/openmm_xml/reader.rs::openmm_periodic_proper_reads_as_multi_term_periodic",
                OMM,
            ]),
            Exact(&[
                "io/openmm_xml/writer.rs::periodic_propers_round_trip_term_for_term",
                EQUIV,
            ]),
            Exact(&[
                "io/gromacs/top_reader.rs::consecutive_funct_9_rows_are_one_multi_term_type",
                GMX,
            ]),
            Exact(&[
                "io/gromacs/top_writer.rs::multi_term_dihedral_periodic_is_funct_9_rows",
                EQUIV,
            ]),
            Exact(&[
                "io/amber/prmtop_forcefield.rs::multiterm_expansion",
                PRMTOP,
                EQUIV,
            ]),
            Exact(&[RECORD, RECORD_SRC]),
        ],
    },
    Row {
        item: "dihedral charmm",
        cells: [
            Exact(&[
                "ff/potential/dihedral/charmm.rs::a_zero_weight_compiles_to_the_lammps_energy",
                "ff/one_four_lammps_check.rs::charmm_weights_of_one_match_lammps",
            ]),
            Exact(&["io/lammps/forcefield_reader.rs::dihedral_charmm_reads_its_own_layout"]),
            Exact(&[RT_L, EQUIV]),
            Na("OpenMM has no per-dihedral 1-4 weight"),
            ExactWhere(
                "w = 0 (a periodic term); w ≠ 0 refused",
                &[
                    "io/openmm_xml/writer.rs::cosine_torsions_are_written_as_periodic_terms",
                    "io/openmm_xml/writer.rs::a_charmm_dihedral_with_a_weight_is_refused",
                ],
            ),
            Na("GROMACS prices a 1-4 pair by [ pairs ], never by a dihedral"),
            ExactWhere(
                "w = 0 (funct 9); w > 0 refused",
                &[
                    "io/gromacs/top_writer.rs::dihedral_charmm_with_w_0_is_dihedraltypes_code_9",
                    "io/gromacs/top_writer.rs::dihedral_charmm_is_an_error",
                ],
            ),
            Na("AMBER torsions are periodic rows"),
            Exact(&[RECORD, RECORD_SRC]),
        ],
    },
    Row {
        item: "dihedral opls",
        cells: [
            Exact(&["ff/potential/dihedral/opls.rs::per_term_phase", SERIES]),
            Exact(&["io/lammps/forcefield_reader.rs::dihedral_opls_four_coeffs"]),
            Exact(&[RT_L]),
            Exact(&["io/openmm_xml/reader.rs::clp_fourier_proper_still_reads_as_opls"]),
            ExactWhere(
                "as RB, which reads back as multi/harmonic: the same series, constant included",
                &["io/openmm_xml/writer.rs::opls_fourier_torsion_is_written_as_rb_in_kj"],
            ),
            Exact(&["io/gromacs/top_reader.rs::dihedraltypes_funct_5_is_dihedral_opls"]),
            Exact(&["io/gromacs/top_writer.rs::dihedral_opls_is_dihedraltypes_code_5"]),
            Na("AMBER torsions are periodic rows"),
            Exact(&[RECORD]),
        ],
    },
    Row {
        item: "dihedral multi/harmonic",
        cells: [
            Exact(&[
                "ff/potential/dihedral/multi_harmonic.rs::energy_is_the_lammps_formula",
                SERIES,
            ]),
            Exact(&[RT_L, EQUIV]),
            Exact(&[RT_L, BACK]),
            Exact(&["io/openmm_xml/reader.rs::rb_row_reads_as_multi_harmonic_in_kcal"]),
            Exact(&[
                "io/openmm_xml/writer.rs::polynomial_torsions_round_trip_through_rb",
                EQUIV,
            ]),
            Exact(&[
                "io/gromacs/top_reader.rs::dihedraltypes_funct_3_is_dihedral_multi_harmonic",
                GMX,
            ]),
            Exact(&[
                "io/gromacs/top_writer.rs::dihedral_multi_harmonic_is_dihedraltypes_code_3",
                EQUIV,
            ]),
            Na("AMBER torsions are periodic rows"),
            Exact(&[RECORD, RECORD_SRC]),
        ],
    },
    Row {
        item: "dihedral nharmonic",
        cells: [
            Exact(&[
                "ff/potential/dihedral/multi_harmonic.rs::a_missing_or_gapped_coefficient_is_refused",
                SERIES,
            ]),
            Exact(&[RT_L]),
            Exact(&[RT_L]),
            Exact(&["io/openmm_xml/reader.rs::rb_row_with_c5_reads_as_nharmonic"]),
            ExactWhere(
                "N ≤ 6 (RB's C0 … C5); above refused",
                &["io/openmm_xml/writer.rs::polynomial_torsions_round_trip_through_rb"],
            ),
            Exact(&[
                "io/gromacs/top_reader.rs::dihedraltypes_funct_3_with_c5_is_dihedral_nharmonic",
            ]),
            ExactWhere(
                "N ≤ 6; above refused",
                &[
                    "io/gromacs/top_writer.rs::dihedral_nharmonic_is_dihedraltypes_code_3_up_to_six_terms",
                ],
            ),
            Na("AMBER torsions are periodic rows"),
            Exact(&[RECORD]),
        ],
    },
    Row {
        item: "dihedral harmonic",
        cells: [
            Exact(&[
                "ff/potential/dihedral/harmonic.rs::energy_is_the_lammps_formula",
                SERIES,
            ]),
            Exact(&[RT_L]),
            Exact(&[RT_L]),
            Na("OpenMM's periodic torsion reads as dihedral periodic"),
            Exact(&["io/openmm_xml/writer.rs::cosine_torsions_are_written_as_periodic_terms"]),
            Na("GROMACS's periodic rows read as dihedral periodic"),
            Exact(&["io/gromacs/top_writer.rs::signed_cosines_are_periodic_rows_at_0_or_180"]),
            Na("AMBER torsions are periodic rows"),
            Exact(&[RECORD]),
        ],
    },
    Row {
        item: "dihedral class2",
        cells: [
            Exact(&["ff/potential/dihedral/class2.rs::energy_phase", SERIES]),
            ExactWhere(CLASS2_LMP, &[CODEC_FILE, CLASS2_X]),
            ExactWhere(CLASS2_LMP, &[CODEC_FILE, CLASS2_X]),
            Na("OpenMM's periodic torsion reads as dihedral periodic"),
            Exact(&["io/openmm_xml/writer.rs::cosine_torsions_are_written_as_periodic_terms"]),
            Na("GROMACS's periodic rows read as dihedral periodic"),
            Exact(&["io/gromacs/top_writer.rs::dihedral_class2_is_funct_9_at_its_phase_plus_180"]),
            Na("AMBER torsions are periodic rows"),
            Exact(&[RECORD]),
        ],
    },
    Row {
        item: "dihedral mmff_torsion",
        cells: own(&["ff/potential/dihedral/mmff.rs::test_mmff_torsion"]),
    },
    Row {
        item: "dihedral uff_torsion",
        cells: own(&["ff/potential/dihedral/uff.rs::forces_are_the_negative_energy_gradient"]),
    },
    // ── impropers ──
    Row {
        item: "improper harmonic",
        cells: [
            Exact(&[
                "ff/potential/improper/harmonic.rs::energy_minimum_at_chi0",
                "io/lammps/forcefield_reader.rs::a_lammps_improper_evaluates_at_the_lammps_energy",
            ]),
            Exact(&[RT_L, EQUIV]),
            ExactWhere(
                "LAMMPS clamps sin χ at 0.001 in its force: within 0.057° of planar its forces are not its energy's gradient",
                &[RT_L, EQUIV],
            ),
            ExactWhere(
                "CustomTorsionForce k(θ−θ0)² at θ0 = 0, or k(|θ|−θ0)²; another signed θ0 refused",
                &[
                    "io/openmm_xml/reader.rs::custom_torsion_harmonic_improper_reads_as_improper_harmonic",
                    "io/openmm_xml/reader.rs::signed_harmonic_improper_off_zero_is_refused_and_abs_form_reads",
                    OMM,
                ],
            ),
            ExactWhere(
                "a wildcard endpoint refused (OpenMM would re-order the atoms)",
                &[
                    "io/openmm_xml/writer.rs::harmonic_improper_round_trips_through_custom_torsion",
                    EQUIV,
                ],
            ),
            ExactWhere(
                "funct 2 at ξ0 ∈ {0°, 180°}; another ξ0 refused",
                &[
                    "io/gromacs/top_reader.rs::dihedraltypes_funct_2_at_zero_is_improper_harmonic",
                    "io/gromacs/top_reader.rs::dihedraltypes_funct_2_off_zero_is_an_error",
                ],
            ),
            ExactWhere(
                "chi0 ∈ {0°, 180°}; another refused",
                &[
                    "io/gromacs/top_writer.rs::improper_harmonic_is_dihedraltypes_code_2",
                    EQUIV,
                ],
            ),
            Exact(&[
                "io/amber/prmtop_forcefield.rs::chamber_impropers_are_improper_harmonic_in_file_order",
                PRMTOP,
                EQUIV,
            ]),
            Exact(&[RECORD, RECORD_SRC]),
        ],
    },
    Row {
        item: "improper cvff",
        cells: [
            Exact(&["ff/potential/improper/cvff.rs::energy_phase", SERIES]),
            Exact(&[RT_L]),
            Exact(&[RT_L]),
            Na("OpenMM's improper reads as improper periodic"),
            ExactWhere(
                "the CustomTorsionForce of its expression with ordering=\"charmm\", which prices the row's dihedral, its centre first",
                &[
                    "io/openmm_xml/writer.rs::a_cvff_improper_is_a_charmm_ordered_custom_torsion",
                    CODEC_ENG,
                ],
            ),
            Na("GROMACS's periodic improper reads as improper periodic"),
            Exact(&["io/gromacs/top_writer.rs::signed_cosines_are_periodic_rows_at_0_or_180"]),
            Na("AMBER impropers read as improper periodic"),
            Exact(&[RECORD]),
        ],
    },
    Row {
        item: "improper periodic",
        cells: [
            Exact(&["ff/potential/improper/periodic.rs::energy_phase", SERIES]),
            Na(
                "LAMMPS has no periodic improper: its writer's cvff reads back as cvff, the same energy",
            ),
            ExactWhere(
                "as cvff, at a phase of 0° or 180°; another refused",
                &[
                    "io/lammps/forcefield_writer.rs::periodic_improper_is_written_as_cvff",
                    "io/lammps/forcefield_writer.rs::periodic_improper_with_other_phase_is_refused",
                    EQUIV,
                ],
            ),
            Exact(&[
                "io/openmm_xml/reader.rs::openmm_improper_reads_as_periodic_improper",
                OMM,
            ]),
            ExactWhere(
                "OpenMM orders the two outer atoms it finds first by element and index (its AMBER rule); a system whose stored order differs prices another dihedral",
                &[
                    "io/openmm_xml/writer.rs::a_periodic_improper_round_trips_through_openmm_order",
                    EQUIV,
                ],
            ),
            Exact(&[
                "io/gromacs/top_reader.rs::dihedraltypes_funct_4_is_improper_periodic",
                GMX,
            ]),
            Exact(&[
                "io/gromacs/top_writer.rs::improper_periodic_is_dihedraltypes_code_4",
                EQUIV,
            ]),
            Exact(&[
                "io/amber/prmtop_forcefield.rs::an_improper_keeps_its_central_atom_third",
                PRMTOP,
                EQUIV,
            ]),
            Exact(&[RECORD, RECORD_SRC]),
        ],
    },
    Row {
        item: "improper mmff_oop",
        cells: own(&["ff/ir_invariance.rs::typed_molecules_price_at_the_reference_energies"]),
    },
    Row {
        item: "improper uff_inversion",
        cells: own(&["ff/potential/improper/uff.rs::forces_are_the_negative_energy_gradient"]),
    },
    // ── crossterms ──
    Row {
        item: "cmap charmm",
        cells: [
            Exact(&["ff/potential/cmap/lammps_check.rs::energy_and_forces_are_lammps_fix_cmap"]),
            Exact(&[
                "io/lammps/forcefield_reader.rs::an_include_reads_its_fix_cmap_file",
                EQUIV,
            ]),
            Exact(&[
                "io/lammps/forcefield_writer.rs::a_charmm_cmap_file_is_written_back_line_for_line",
                BACK,
            ]),
            ExactWhere(
                "odd N refused; OpenMM takes its node slopes from periodic splines (≤ 10⁻⁷ kcal/mol off the IR's)",
                &[
                    "io/openmm_xml/reader.rs::cmap_map_is_shifted_by_half_and_transposed_into_phi_major",
                    OMM,
                ],
            ),
            ExactWhere(
                "as OpenMM read: its interpolation ≤ 10⁻⁷ kcal/mol off the IR's",
                &[RT_O, EQUIV],
            ),
            Exact(&[
                "io/gromacs/top_reader.rs::cmaptypes_is_a_cmap_charmm_grid_in_kcal",
                GMX,
            ]),
            Exact(&[
                "io/gromacs/top_writer.rs::cmap_charmm_is_a_cmaptypes_row",
                EQUIV,
            ]),
            Exact(&[
                "io/amber/prmtop_forcefield.rs::chamber_cmap_is_a_cmap_charmm_type",
                PRMTOP,
                EQUIV,
            ]),
            Exact(&[
                "io/mrec/forcefield_mapping.rs::a_populated_cmap_round_trips_and_prices_the_same",
                RECORD_SRC,
            ]),
        ],
    },
    // ── pair styles ──
    Row {
        item: "pair lj/cut",
        cells: [
            Exact(&[
                "ff/potential/pair/lj_cut.rs::unshifted_energy_at_sigma_is_zero",
                "ff/potential/mod.rs::lj_cut_combines_distinct_types_lorentz_berthelot",
            ]),
            Exact(&[
                "io/lammps/forcefield_reader.rs::a_bare_lj_cut_has_no_coulomb_style",
                EQUIV,
            ]),
            Exact(&[
                "io/lammps/forcefield_writer.rs::round_trip_preserves_molrs_params",
                BACK,
            ]),
            Exact(&[
                "io/openmm_xml/reader.rs::reads_all_sections_with_molrs_units",
                OMM,
            ]),
            Exact(&[
                "io/openmm_xml/writer.rs::opls_torsions_and_pairs_round_trip_through_the_openmm_units",
                EQUIV,
            ]),
            Exact(&[
                "io/gromacs/top_reader.rs::atomtypes_row_splits_into_atom_full_and_an_lj_cut_self_row",
                GMX,
            ]),
            Exact(&[
                "io/gromacs/top_writer.rs::atomtypes_row_joins_atom_full_and_the_lj_cut_self_row",
                EQUIV,
            ]),
            Exact(&[
                "io/amber/prmtop_forcefield.rs::decode_lj_types_self_terms_match_closed_form",
                PRMTOP,
            ]),
            Exact(&[RECORD, RECORD_SRC]),
        ],
    },
    Row {
        item: "pair lj/charmm",
        cells: [
            Exact(&[
                "ff/potential/pair/charmm.rs::lj_hand_values_inside_across_and_beyond_the_switch",
            ]),
            Exact(&["io/lammps/forcefield_reader.rs::reads_lj_charmm_coul_charmm"]),
            Exact(&[
                "ff/one_four_lammps_check.rs::lammps_round_trip_and_override_refusal",
                EQUIV,
            ]),
            Exact(&[
                "io/openmm_xml/reader.rs::lennard_jones_force_reads_as_lj_charmm",
                OMM,
            ]),
            Exact(&[RT_O, EQUIV]),
            Exact(&[
                "io/gromacs/top_reader.rs::a_self_pairtype_is_lj_charmm_epsilon14",
                GMX,
            ]),
            Exact(&[GMX_RT, EQUIV]),
            Exact(&[
                "io/amber/prmtop_forcefield.rs::chamber_lennard_jones_is_lj_charmm_with_its_1_4_table",
                PRMTOP,
            ]),
            Exact(&[RECORD, RECORD_SRC]),
        ],
    },
    Row {
        item: "pair coul/cut",
        cells: [
            Exact(&["ff/potential/pair/coul_cut.rs::energy_and_sign"]),
            Exact(&["io/lammps/forcefield_reader.rs::reads_lammps_units", EQUIV]),
            ExactWhere(
                "delta = 0 and dielectric = 1; the Coulomb constant is LAMMPS's own",
                &[
                    "io/lammps/forcefield_writer.rs::a_buffered_coulomb_is_refused",
                    EQUIV,
                ],
            ),
            Exact(&[OMM]),
            ExactWhere("the Coulomb constant is OpenMM's own", &[EQUIV]),
            Exact(&[
                "io/gromacs/top_reader.rs::atomtypes_declare_coul_cut_with_its_constants",
                GMX,
            ]),
            ExactWhere(
                "the Coulomb constant is GROMACS's own",
                &[
                    "io/gromacs/top_writer.rs::a_stated_coulomb_constant_is_not_written",
                    EQUIV,
                ],
            ),
            Exact(&[
                "core/constants.rs::amber_coulomb_is_its_charge_factor_squared",
                PRMTOP,
            ]),
            Exact(&[RECORD, RECORD_SRC]),
        ],
    },
    Row {
        item: "pair coul/charmm",
        cells: [
            Exact(&["ff/potential/pair/charmm.rs::coulomb_hand_values_and_lammps_switched_force"]),
            Exact(&["io/lammps/forcefield_reader.rs::reads_lj_charmm_coul_charmm"]),
            Exact(&[EQUIV]),
            Exact(&[OMM]),
            Exact(&[EQUIV]),
            Exact(&[GMX]),
            Exact(&[EQUIV]),
            Exact(&[PRMTOP]),
            Exact(&[RECORD, RECORD_SRC]),
        ],
    },
    Row {
        item: "pair coul/long/pme",
        cells: [
            Exact(&["ff/potential/kspace/pme.rs::test_two_ions_energy"]),
            ExactWhere(
                "lj/cut/coul/long: cutoff and constant; its kspace_style states an accuracy, not an alpha, so pricing is refused until the Ewald parameters are stated",
                &[
                    "io/lammps/forcefield_reader.rs::lj_cut_coul_long_is_coul_long_pme_without_its_ewald_parameters",
                ],
            ),
            ExactWhere(
                "the real-space lj/cut/coul/long; stated Ewald parameters refused (molrs's smooth PME is not LAMMPS's PPPM)",
                &["io/lammps/forcefield_writer.rs::pair_settings_are_written_or_refused_by_name"],
            ),
            Na("the long-range method is a createSystem argument"),
            ExactWhere(
                "the charges; PME / Ewald is a createSystem argument; stated Ewald parameters refused",
                &["io/openmm_xml/writer.rs::a_long_range_coulomb_is_written_as_the_charges"],
            ),
            Na("the long-range method is an .mdp setting"),
            ExactWhere(
                "the charges; coulombtype = PME is an .mdp setting; stated Ewald parameters refused",
                &["io/gromacs/top_writer.rs::a_long_range_coulomb_writes_the_coul_cut_top"],
            ),
            Na("a prmtop states no long-range method"),
            Exact(&[RECORD]),
        ],
    },
    Row {
        item: "pair lj/class2",
        cells: [
            Exact(&["ff/potential/pair/lj_class2.rs::energy_at_sigma_is_negative_eps"]),
            ExactWhere(
                "through its codec, mixing sixthpower as LAMMPS's always does; not run by LAMMPS: the installed lmp has no CLASS2 package",
                &[CODEC_FILE],
            ),
            ExactWhere(
                "through its codec, mixing sixthpower as LAMMPS's always does; not run by LAMMPS: the installed lmp has no CLASS2 package",
                &[CODEC_FILE],
            ),
            Na("OpenMM has no lj/class2 tag"),
            Refused(
                "its kernel prices a row per pair (cross rows), which a CustomNonbondedForce has not",
                &[NO_OMM],
            ),
            Na("GROMACS has no 9-6 Lennard-Jones"),
            Refused(GMX_NONE, &[NO_ENGINE]),
            Na("AMBER's Lennard-Jones is 12-6"),
            Exact(&[RECORD]),
        ],
    },
    Row {
        item: "pair buck",
        cells: positional_pair(&["ff/potential/pair/buck.rs::energy_matches_closed_form"]),
    },
    Row {
        item: "pair morse",
        cells: positional_pair(&["ff/potential/pair/morse.rs::well_minimum_is_minus_d0"]),
    },
    Row {
        item: "pair thole",
        cells: outside(
            &["ff/potential/pair/thole.rs::damping_factor_matches_closed_form"],
            "polarizable (Drude) screening, outside the Class-I IR",
        ),
    },
    Row {
        item: "pair coul/tt",
        cells: outside(
            &["ff/potential/pair/tang_toennies.rs::damping_matches_closed_form"],
            "polarizable (Drude) screening, outside the Class-I IR",
        ),
    },
    Row {
        item: "pair mmff_vdw",
        cells: own(&["ff/potential/pair/mmff.rs::test_mmff_vdw_energy"]),
    },
    Row {
        item: "pair uff_lj",
        cells: own(&[
            "ff/potential/pair/uff.rs::per_atom_parameters_score_a_pair_exactly_as_compiled_ones",
        ]),
    },
    // ── field-level ──
    Row {
        item: "special_bonds",
        cells: [
            Exact(&["ff/potential/mod.rs::lj_cut_applies_special_bonds_14_scaling"]),
            Exact(&["io/lammps/forcefield_reader.rs::special_bonds_presets_and_absent_line"]),
            Exact(&[
                "io/lammps/forcefield_writer.rs::default_write_keeps_special_bonds",
                EQUIV,
            ]),
            ExactWhere("[0, 0, s] (OpenMM's 14scale)", &[OMM]),
            ExactWhere(
                "[0, 0, s]; another refused",
                &[
                    EQUIV,
                    "io/openmm_xml/writer.rs::regular_one_four_with_own_14_parameters_is_refused",
                ],
            ),
            Exact(&["io/gromacs/top_reader.rs::fudge_factors_are_the_1_4_special_bond_weights"]),
            ExactWhere(
                "[0, 0, s]; another refused",
                &[
                    "io/gromacs/top_writer.rs::nonzero_1_3_special_bond_weight_is_an_error",
                    EQUIV,
                ],
            ),
            Exact(&["io/amber/prmtop_forcefield.rs::special_bonds_are_reciprocal_divisors"]),
            Exact(&[
                "io/mrec/forcefield_mapping.rs::a_lammps_read_force_field_round_trips",
                RECORD_SRC,
            ]),
        ],
    },
    Row {
        item: "per-pair overrides (pairs epsilon, sigma, lj_scale, charge_product, coul_scale)",
        cells: [
            Exact(&[
                "ff/one_four_lammps_check.rs::global_half_equals_per_pair_scales_equals_per_pair_parameters",
                "ff/one_four_lammps_check.rs::an_override_cell_beats_the_weight_and_a_null_cell_keeps_it",
            ]),
            Na("LAMMPS has no per-pair 1-4 parameters"),
            Refused(
                "LAMMPS has no per-pair 1-4 parameters",
                &[
                    "io/gromacs/top_reader/engine_check.rs::per_pair_parameters_are_refused_by_the_lammps_writer",
                ],
            ),
            Na("a ForceField XML holds no per-pair exception"),
            Na("a ForceField XML holds no per-pair exception"),
            Exact(&[
                "io/gromacs/top_reader/system.rs::a_funct_2_pair_carries_its_charges",
                GMX,
            ]),
            Exact(&[GMX_SYS]),
            Exact(&[PRMTOP]),
            Exact(&[
                "io/mrec/zarr_storage/frame_io.rs::f64_column_keeps_its_validity_mask_across_the_frame_round_trip",
            ]),
        ],
    },
    Row {
        item: "lj/charmm one_four = \"epsilon14\"",
        cells: [
            Exact(&[
                "ff/compile/one_four.rs::epsilon14_without_rows_is_refused_and_with_them_priced",
            ]),
            Na("LAMMPS prices a special_bonds 1-4 pair at the regular parameters"),
            ExactWhere(
                "refused as such; its exact LAMMPS form is special_bonds 0 and one zero-K dihedral charmm row of w = 1 per 1-4 pair",
                &[
                    "ff/one_four_lammps_check.rs::lammps_round_trip_and_override_refusal",
                    EQUIV,
                ],
            ),
            Exact(&[OMM]),
            Exact(&[RT_O, EQUIV]),
            Exact(&[GMX]),
            Exact(&[GMX_RT, EQUIV]),
            Exact(&[PRMTOP, EQUIV]),
            Exact(&[
                "io/mrec/forcefield_mapping.rs::lj_charmm_one_four_round_trips_and_an_unknown_value_is_refused",
            ]),
        ],
    },
    Row {
        item: "mixing arithmetic",
        cells: [
            Exact(&["ff/potential/mod.rs::lj_cut_combines_distinct_types_lorentz_berthelot"]),
            Exact(&["io/lammps/forcefield_reader.rs::pair_modify_shift_is_the_lj_cut_shift"]),
            Exact(&[
                "io/lammps/forcefield_writer.rs::combined_undeclared_lj_cut_writes_pair_modify_mix_arithmetic",
            ]),
            Exact(&[OMM]),
            Exact(&[EQUIV]),
            Exact(&["io/gromacs/top_reader.rs::comb_rule_2_declares_arithmetic_mixing_on_lj_cut"]),
            Exact(&[
                "io/gromacs/top_writer.rs::arithmetic_mixing_writes_comb_rule_2",
                EQUIV,
            ]),
            Exact(&["io/amber/prmtop_forcefield.rs::the_lj_style_states_arithmetic_mixing"]),
            Exact(&[RECORD_SRC]),
        ],
    },
    Row {
        item: "mixing geometric",
        cells: [
            Exact(&["ff/potential/pair/lj_cut.rs::without_a_cross_row_the_pair_is_mixed"]),
            Exact(&[
                "io/lammps/forcefield_writer.rs::declared_geometric_lj_cut_writes_pair_modify_mix_geometric",
            ]),
            Exact(&[
                "io/lammps/forcefield_writer.rs::declared_geometric_lj_cut_writes_pair_modify_mix_geometric",
                EQUIV,
            ]),
            ExactWhere(
                "foyer's combining_rule, which OpenMM's own app.ForceField ignores",
                &[OMM],
            ),
            ExactWhere(
                "foyer's combining_rule and the <Script> that applies it (OpenMM's own app.ForceField ignores the attribute); with cross rows refused",
                &[
                    "io/openmm_xml/writer.rs::geometric_mixing_is_written_as_the_script_openmm_runs",
                    EQUIV,
                ],
            ),
            Exact(&[
                "io/gromacs/top_reader.rs::comb_rule_3_declares_geometric_mixing_on_lj_cut",
                GMX,
            ]),
            Exact(&[
                "io/gromacs/top_writer.rs::geometric_mixing_writes_comb_rule_3",
                EQUIV,
            ]),
            Na("AMBER mixes arithmetically"),
            Exact(&[RECORD_SRC]),
        ],
    },
    Row {
        item: "mixing sixthpower",
        cells: [
            Exact(&["ff/ir/combining_rule.rs::sixthpower_is_the_waldman_hagler_rule"]),
            Exact(&["io/lammps/forcefield_writer.rs::sixthpower_mixing_is_read_and_written_back"]),
            Exact(&["io/lammps/forcefield_writer.rs::sixthpower_mixing_is_read_and_written_back"]),
            Na("OpenMM mixes arithmetically"),
            Refused("OpenMM mixes arithmetically", &[]),
            Na("GROMACS comb-rules are C6/C12, arithmetic, geometric"),
            Refused(
                "no GROMACS comb-rule",
                &["io/gromacs/top_writer.rs::sixthpower_mixing_is_an_error"],
            ),
            Na("AMBER mixes arithmetically"),
            Exact(&[RECORD]),
        ],
    },
    Row {
        item: "cross rows (NBFIX)",
        cells: [
            Exact(&[
                "ff/potential/pair/lj_cut.rs::an_explicit_cross_row_overrides_the_mixing_rule",
                "ff/potential/pair/charmm.rs::epsilon14_mixes_like_epsilon_and_a_cross_row_wins",
            ]),
            Exact(&["io/lammps/forcefield_reader.rs::a_cross_pair_coeff_is_kept_as_a_pair_type"]),
            Exact(&[
                "io/lammps/forcefield_writer.rs::label_writer_writes_cross_pair_only_when_both_endpoints_used",
                EQUIV,
            ]),
            Exact(&[OMM]),
            ExactWhere(
                "a cross row with 1-4 parameters of its own refused (OpenMM prices an NBFIX 1-4 pair with the NBFIX row)",
                &[
                    "io/openmm_xml/writer.rs::an_lj_cross_row_is_an_nbfix_pair",
                    EQUIV,
                ],
            ),
            Exact(&[
                "io/gromacs/top_reader.rs::nonbond_params_is_an_explicit_lj_cut_cross_row",
                GMX,
            ]),
            Exact(&[
                "io/gromacs/top_writer.rs::explicit_lj_cut_cross_row_is_a_nonbond_params_row",
                EQUIV,
            ]),
            Exact(&[
                "io/amber/prmtop_forcefield.rs::decode_lj_types_keeps_nbfix_cross_terms_as_cross_rows",
            ]),
            Exact(&[
                "io/mrec/forcefield_mapping.rs::explicit_cross_rows_round_trip_and_still_override_mixing",
            ]),
        ],
    },
    Row {
        item: "lj/cut shift",
        cells: [
            Exact(&["ff/potential/pair/lj_cut.rs::a_typed_kernel_stops_at_its_cutoff"]),
            Exact(&["io/lammps/forcefield_reader.rs::pair_modify_shift_is_the_lj_cut_shift"]),
            Exact(&[
                "io/lammps/forcefield_writer.rs::pair_settings_are_written_or_refused_by_name",
            ]),
            Na("OpenMM's NonbondedForce has no shifted Lennard-Jones"),
            Refused(
                "OpenMM's NonbondedForce has no shifted Lennard-Jones",
                &["io/openmm_xml/writer.rs::other_units_and_a_shifted_lj_are_refused"],
            ),
            Na("a modifier is an .mdp setting"),
            Refused("a modifier is an .mdp setting", &[]),
            Na("AMBER does not shift"),
            Exact(&[RECORD]),
        ],
    },
    Row {
        item: "lj/cut n, m (Mie)",
        cells: [
            Exact(&["ff/potential/pair/lj_cut.rs::mie_c_is_four_for_12_6"]),
            Refused("pair_style mie/cut is not read", &[NO_LMP_PAIR]),
            Refused(
                "pair_style mie/cut is not written",
                &["io/lammps/forcefield_writer.rs::pair_settings_are_written_or_refused_by_name"],
            ),
            Na("OpenMM's Lennard-Jones is 12-6"),
            Refused(
                "OpenMM's Lennard-Jones is 12-6",
                &["io/openmm_xml/writer.rs::other_units_and_a_shifted_lj_are_refused"],
            ),
            Na("GROMACS's Lennard-Jones is 12-6"),
            Refused("GROMACS's Lennard-Jones is 12-6", &[]),
            Na("AMBER's Lennard-Jones is 12-6"),
            Exact(&[RECORD]),
        ],
    },
    Row {
        item: "Coulomb constant",
        cells: [
            Exact(&["ff/potential/pair/coul_cut.rs::energy_and_sign"]),
            Exact(&["io/lammps/forcefield_reader.rs::metal_units_are_kept_and_declared"]),
            Na(
                "LAMMPS fixes qqr2e per units: priced at LAMMPS's (an AMBER field 3.5·10⁻⁵ above its own)",
            ),
            Exact(&[OMM]),
            Na("OpenMM fixes ONE_4PI_EPS0: priced at OpenMM's"),
            Exact(&[GMX]),
            Na("GROMACS fixes ONE_4PI_EPS0: priced at GROMACS's"),
            Exact(&["core/constants.rs::amber_coulomb_is_its_charge_factor_squared"]),
            Exact(&[RECORD_SRC]),
        ],
    },
    Row {
        item: "cutoff, inner (switch)",
        cells: [
            Exact(&[
                "ff/potential/pair/lj_cut.rs::a_typed_kernel_stops_at_its_cutoff",
                "ff/potential/pair/charmm.rs::lj_hand_values_inside_across_and_beyond_the_switch",
            ]),
            Exact(&["io/lammps/forcefield_reader.rs::reads_lj_charmm_coul_charmm"]),
            Exact(&[
                "io/lammps/forcefield_writer.rs::a_pair_style_without_a_cutoff_is_refused_not_defaulted",
            ]),
            Na("a createSystem argument"),
            Na("a createSystem argument"),
            Na("an .mdp setting"),
            Na("an .mdp setting"),
            Na("a prmtop states none"),
            Exact(&[RECORD]),
        ],
    },
    Row {
        item: "units presets (real, metal, lj)",
        cells: [
            Exact(&["io/lammps/forcefield_writer.rs::metal_write_converts_energy_via_lj_hub"]),
            Exact(&["io/lammps/forcefield_reader.rs::metal_units_are_kept_and_declared"]),
            Exact(&["io/lammps/forcefield_writer.rs::metal_write_converts_energy_via_lj_hub"]),
            Na("OpenMM's units read as real"),
            Refused(
                "the conversions are from real",
                &["io/openmm_xml/writer.rs::other_units_and_a_shifted_lj_are_refused"],
            ),
            Na("GROMACS's units read as real"),
            Refused(
                "the conversions are from real",
                &["io/gromacs/top_writer.rs::units_other_than_real_are_refused"],
            ),
            Na("a prmtop reads as real"),
            Exact(&[
                "io/mrec/forcefield_mapping.rs::a_section_stating_radians_beside_a_preset_is_refused",
            ]),
        ],
    },
    // ── the protocol's engine forms ──
    Row {
        item: "run-time style with a positional LAMMPS form",
        cells: [
            Exact(&[CODEC_ENG]),
            Exact(&[RUNTIME_LMP]),
            Exact(&[RUNTIME_LMP, CODEC_ENG]),
            Refused(
                "reading a Custom*Force is refused, but for the two harmonic impropers",
                &[NO_CUSTOM_READ],
            ),
            Exact(&[OMM_REWRITE, CODEC_ENG]),
            Na("GROMACS holds built-in styles only"),
            Refused("not a built-in style (NoEngineForm)", &[NO_ENGINE]),
            Na("AMBER holds built-in styles only"),
            Exact(&[PERSIST_CUSTOM]),
        ],
    },
    Row {
        item: "expression style without a LAMMPS form",
        cells: [
            Exact(&[CODEC_ENG]),
            Refused(
                "no registered style reads its name (the installed LAMMPS has no LEPTON package)",
                &[RUNTIME_LMP],
            ),
            Refused(
                "no LAMMPS form: an expression needs LAMMPS's LEPTON package (NoEngineForm)",
                &[NO_ENGINE],
            ),
            Refused(
                "reading a Custom*Force is refused, but for the two harmonic impropers",
                &[NO_CUSTOM_READ],
            ),
            Exact(&[NO_ENGINE, OMM_REWRITE]),
            Na("GROMACS holds built-in styles only"),
            Refused("not a built-in style (NoEngineForm)", &[NO_ENGINE]),
            Na("AMBER holds built-in styles only"),
            Exact(&[PERSIST_CUSTOM]),
        ],
    },
    Row {
        item: "run-time compound category (Urey-Bradley)",
        cells: [
            Exact(&[CODEC_ENG]),
            Na("LAMMPS has no `*_style` command for a run-time category"),
            Refused("no `*_style` command (NoEngineForm)", &[NO_ENGINE]),
            Na("OpenMM's ForceField XML has no compound tag"),
            ExactWhere(
                "a <Script> building a CustomCompoundBondForce over OpenMM's bonds, angles or propers (a chain of 2 to 4 atoms)",
                &[CODEC_ENG],
            ),
            Na("GROMACS holds built-in categories only"),
            Refused("not a built-in style (NoEngineForm)", &[NO_ENGINE]),
            Na("AMBER holds built-in categories only"),
            Exact(&[PERSIST_CUSTOM]),
        ],
    },
];

/// Every Class-I style of molrec's registry (`molrec/docs/spec/forcefield.md`,
/// "Registry"), as `category style`.
const MOLREC_CLASS_I: &[&str] = &[
    "atom full",
    "bond harmonic",
    "bond morse",
    "bond class2",
    "angle harmonic",
    "angle charmm",
    "angle class2",
    "dihedral periodic",
    "dihedral opls",
    "dihedral rb",
    "dihedral charmm",
    "dihedral harmonic",
    "dihedral multi/harmonic",
    "dihedral nharmonic",
    "dihedral class2",
    "improper harmonic",
    "improper cvff",
    "improper periodic",
    "improper trefoil",
    "pair lj/cut",
    "pair lj/charmm",
    "pair coul/charmm",
    "pair lj/class2",
    "pair buck",
    "pair morse",
    "pair coul/cut",
    "pair coul/long/pme",
    "pair thole",
    "cmap charmm",
];

/// `crates/molrs/src`.
fn src() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"))
}

fn is_style_row(item: &str) -> bool {
    matches!(
        item.split(' ').next(),
        Some("bond" | "angle" | "dihedral" | "improper" | "pair" | "cmap")
    ) && item.split(' ').count() == 2
}

/// The guide's table, rendered from [`MATRIX`].
fn render() -> String {
    let mut out = String::new();
    out.push_str("| Style or setting |");
    for c in COLUMNS {
        out.push_str(&format!(" {c} |"));
    }
    out.push_str("\n|---|");
    out.push_str(&"---|".repeat(COLUMNS.len()));
    out.push('\n');
    let mut notes: Vec<String> = Vec::new();
    for row in MATRIX {
        out.push_str(&format!("| `{}` |", row.item));
        for cell in &row.cells {
            let mark = match cell {
                Exact(_) => "✓".to_owned(),
                ExactWhere(why, _) | Refused(why, _) => {
                    let n = match notes.iter().position(|n| n == why) {
                        Some(i) => i + 1,
                        None => {
                            notes.push((*why).to_owned());
                            notes.len()
                        }
                    };
                    let head = if matches!(cell, Refused(..)) {
                        "refused"
                    } else {
                        "✓"
                    };
                    format!("{head} [{n}]")
                }
                Na(_) => "—".to_owned(),
            };

            out.push_str(&format!(" {mark} |"));
        }
        out.push('\n');
    }
    out.push('\n');
    for (i, n) in notes.iter().enumerate() {
        out.push_str(&format!("{}. {n}\n", i + 1));
    }
    out
}

const BEGIN: &str = "<!-- completeness:begin (generated by ff::completeness; do not edit) -->\n";
const END: &str = "<!-- completeness:end -->";

#[test]
fn every_registered_kernel_has_a_row_and_every_style_row_a_kernel() {
    let registry = BuiltinKernels::builtin();
    let styles: Vec<String> = registry
        .styles()
        .into_iter()
        .map(|(c, s)| format!("{c} {s}"))
        .collect();
    let rows: Vec<&str> = MATRIX.iter().map(|r| r.item).collect();
    for s in &styles {
        assert!(rows.contains(&s.as_str()), "no completeness row for `{s}`");
    }
    for row in MATRIX.iter().filter(|r| is_style_row(r.item)) {
        assert!(
            styles.iter().any(|s| s == row.item),
            "row `{}` is no registered kernel",
            row.item
        );
    }
    let mut seen = std::collections::HashSet::new();
    for r in MATRIX {
        assert!(seen.insert(r.item), "row `{}` twice", r.item);
    }
}

#[test]
fn every_cited_test_exists() {
    let mut missing = Vec::new();
    for row in MATRIX {
        for (k, cell) in row.cells.iter().enumerate() {
            let tests: &[&str] = match cell {
                Exact(t) | ExactWhere(_, t) | Refused(_, t) => t,
                Na(_) => &[],
            };
            if matches!(cell, Exact(_) | ExactWhere(..)) && tests.is_empty() {
                missing.push(format!("{} / {}: a ✓ without a test", row.item, COLUMNS[k]));
            }
            if let Na(why) = cell {
                assert!(
                    !why.is_empty(),
                    "{} / {}: n.a. without a reason",
                    row.item,
                    COLUMNS[k]
                );
            }
            for t in tests {
                let (file, name) = t.split_once("::").expect("path::fn");
                let text = std::fs::read_to_string(src().join(file)).unwrap_or_default();
                if !text.contains(&format!("fn {name}(")) {
                    missing.push(format!("{} / {}: {t}", row.item, COLUMNS[k]));
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "cited tests not found:\n{}",
        missing.join("\n")
    );
}

/// Every Class-I row of molrec's registry is priced by a molrs kernel of the
/// same name, or refused by name when compiled.
#[test]
fn every_molrec_class_i_style_is_priced_or_refused_by_name() {
    let registry = BuiltinKernels::builtin();
    let known: Vec<(&str, &str)> = registry.styles();
    for &item in MOLREC_CLASS_I {
        let (category, style) = item.split_once(' ').unwrap();
        if category == "atom" || known.contains(&(category, style)) {
            continue;
        }
        let mut ff = ForceField::new("probe");
        let arity = match category {
            "bond" => 2,
            "angle" => 3,
            "dihedral" | "improper" => 4,
            _ => 2,
        };
        let ends: Vec<&str> = vec!["a"; arity];
        ff.def_style(category, style, Params::new())
            .unwrap()
            .def_type("t", &ends, Params::from_iter([("k", 1.0)]))
            .unwrap();
        let frame = probe_frame(category, arity);
        let err = PotentialCompiler::new(&ff)
            .compile(&frame)
            .err()
            .unwrap_or_else(|| panic!("`{item}` compiled without a kernel"));
        assert!(err.to_string().contains(style), "`{item}`: {err}");
    }
}

fn probe_frame(category: &str, arity: usize) -> molrs::core::Frame {
    use molrs::core::Block;
    use molrs::op::Idx;
    use ndarray::Array1;
    let mut frame = molrs::core::Frame::new();
    let mut atoms = Block::new();
    atoms
        .insert(
            "type",
            Array1::from_vec(vec!["a".to_owned(); arity]).into_dyn(),
            None,
        )
        .unwrap();
    for key in ["x", "y", "z"] {
        atoms
            .insert(
                key,
                Array1::from_vec((0..arity).map(|i| i as f64).collect()).into_dyn(),
                None,
            )
            .unwrap();
    }
    frame.insert("atoms", atoms);
    let mut rows = Block::new();
    for (i, key) in ["atomi", "atomj", "atomk", "atoml"][..arity]
        .iter()
        .enumerate()
    {
        rows.insert(*key, Array1::from_vec(vec![i as Idx]).into_dyn(), None)
            .unwrap();
    }
    rows.insert(
        "type",
        Array1::from_vec(vec!["t".to_owned()]).into_dyn(),
        None,
    )
    .unwrap();
    frame.insert(format!("{category}s").as_str(), rows);
    frame
}

/// Every registered style, with a type, goes through a record section and
/// back unchanged.
#[cfg(feature = "zarr")]
#[test]
fn every_registered_style_persists_through_a_record() {
    let registry = BuiltinKernels::builtin();
    for (category, style) in registry.styles() {
        let mut ff = ForceField::new("persist");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type(
                "a",
                &[],
                Params::from_iter([("mass", 1.0), ("charge", 0.1)]),
            )
            .unwrap();
        let arity = match category {
            "bond" => 2,
            "angle" => 3,
            "dihedral" | "improper" => 4,
            "cmap" => 5,
            _ => 1,
        };
        let ends: Vec<&str> = vec!["a"; arity];
        let mut params = Params::from_iter([("k", 1.5), ("x0", 0.25)]);
        if category == "cmap" {
            params = Params::new();
            params.set(
                crate::ff::ir::CMAP_GRID,
                ndarray::Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f64).into_dyn(),
            );
        }
        ff.def_style(category, style, Params::from_iter([("cutoff", 9.0)]))
            .unwrap()
            .def_type("t", &ends, params)
            .unwrap();
        use crate::io::mrec::ForceFieldSection;
        let back = ForceFieldSection::from_forcefield(&ff)
            .unwrap_or_else(|e| panic!("{category} {style}: {e}"))
            .to_forcefield()
            .unwrap_or_else(|e| panic!("{category} {style}: {e}"));
        let s = back.get_style(category, style).expect(style);
        assert_eq!(
            s.params().get("cutoff").and_then(|v| v.as_num()),
            Some(9.0),
            "{category} {style}"
        );
        let (_, mut got_ends, got) = s.type_rows().into_iter().next().expect("a type");
        if category == "pair" {
            // A self row is one type's, however its endpoints are spelled.
            got_ends.dedup();
        }
        assert_eq!(got_ends, ends, "{category} {style}");
        let want = &ff.get_style(category, style).unwrap().type_rows()[0].2;
        assert_eq!(got, *want, "{category} {style}");
    }
}

/// The guide's "Completeness" table is [`render`]'s; with
/// `MOLRS_WRITE_COMPLETENESS` set, the guide is rewritten.
#[test]
fn the_guide_holds_the_generated_table() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/developer/forcefield-ir.md");
    let guide = std::fs::read_to_string(&path).unwrap();
    let start = guide.find(BEGIN).expect("the guide's completeness markers") + BEGIN.len();
    let end = guide[start..].find(END).expect("the end marker") + start;
    let table = render();
    if std::env::var_os("MOLRS_WRITE_COMPLETENESS").is_some() {
        let new = format!("{}{table}{}", &guide[..start], &guide[end..]);
        std::fs::write(&path, new).unwrap();
        return;
    }
    assert_eq!(
        guide[start..end],
        table,
        "the guide's completeness table is stale: MOLRS_WRITE_COMPLETENESS=1 cargo mrs-test -- \
         ff::completeness::the_guide_holds_the_generated_table"
    );
}
