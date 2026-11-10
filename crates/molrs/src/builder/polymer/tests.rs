//! Growth, molecule and system tests over templates read from fragment
//! bodies. Every expected number is derived in the test from the rules the
//! module documents.

use super::*;
use crate::core::{BondNumber, CoarseGrain, Element, PortKind, keys};
use crate::io::smiles::SmilesIr;
use crate::op::F;

fn fragment(name: &str, body: &str) -> Fragment {
    let template = SmilesIr::from_fragment(body)
        .unwrap()
        .to_template()
        .unwrap();
    let n = template.n_ports();
    Fragment::new(name, template, vec![SiteWeight::default(); n]).unwrap()
}

fn weighted(name: &str, body: &str, weights: Vec<SiteWeight>) -> Fragment {
    let template = SmilesIr::from_fragment(body)
        .unwrap()
        .to_template()
        .unwrap();
    Fragment::new(name, template, weights).unwrap()
}

fn exactly(n: F, measure: Measure) -> Option<ChainSize> {
    Some(ChainSize::new(Distribution::Gauss { mu: n, sigma: 0.0 }, measure).unwrap())
}

fn terminal(kind: PortKind) -> Terminal {
    Terminal::new(kind, "", BondNumber::Single, None).unwrap()
}

fn listed(kind: PortKind, weights: Vec<F>) -> Terminal {
    Terminal::new(kind, "", BondNumber::Single, Some(weights)).unwrap()
}

/// `-CH2-CH2-` with a `<` head and a `>` tail.
fn ethylene() -> Fragment {
    fragment("E", "[<]CC[>]")
}

/// A methyl that caps an open `<` (its port is `>`).
fn cap_left() -> Fragment {
    fragment("Me>", "C[>]")
}

/// A methyl that caps an open `>` (its port is `<`).
fn cap_right() -> Fragment {
    fragment("<Me", "[<]C")
}

fn bead_types(sites: &CoarseGrain) -> Vec<String> {
    sites
        .bead_ids()
        .map(|id| {
            sites
                .get_bead(id)
                .unwrap()
                .get_str(keys::BEAD_TYPE)
                .unwrap()
                .to_owned()
        })
        .collect()
}

fn count(types: &[String], name: &str) -> usize {
    types.iter().filter(|t| *t == name).count()
}

fn homopolymer(size: Option<ChainSize>) -> Molecule {
    let object = StochasticObject::new(
        vec![ethylene()],
        vec![cap_left(), cap_right()],
        None,
        None,
        size,
    )
    .unwrap();
    Molecule::new(vec![Part::Stochastic(object)]).unwrap()
}

// -- growth -------------------------------------------------------------------

#[test]
fn chain_grows_to_exactly_the_target_repeat_count_and_is_capped_at_both_ends() {
    let sites = homopolymer(exactly(10.0, Measure::RepeatUnits))
        .sample(1)
        .unwrap();
    let types = bead_types(&sites);
    assert_eq!(count(&types, "E"), 10);
    assert_eq!(count(&types, "Me>"), 1);
    assert_eq!(count(&types, "<Me"), 1);
    // A linear chain: one bond fewer than sites.
    assert_eq!(sites.n_bonds(), types.len() - 1);
}

#[test]
fn heavy_atom_mass_target_stops_at_the_first_unit_reaching_it() {
    let c = F::from(Element::C.atomic_mass());
    let target: F = 100.0;
    let sites = homopolymer(exactly(target, Measure::HeavyAtomMass))
        .sample(4)
        .unwrap();
    // The starting end group (one C) counts, then each repeat unit adds 2 C.
    let expected = ((target - c) / (2.0 * c)).ceil() as usize;
    assert_eq!(count(&bead_types(&sites), "E"), expected);
}

#[test]
fn equal_seeds_give_equal_samples() {
    let size = ChainSize::new(Distribution::FlorySchulz { a: 0.2 }, Measure::RepeatUnits).unwrap();
    let molecule = homopolymer(Some(size));
    assert_eq!(
        bead_types(&molecule.sample(9).unwrap()),
        bead_types(&molecule.sample(9).unwrap())
    );
}

#[test]
fn sites_carry_the_joined_port_ordinals_and_order() {
    let sites = homopolymer(exactly(2.0, Measure::RepeatUnits))
        .sample(2)
        .unwrap();
    for id in sites.bond_ids() {
        let bond = sites.get_bond(id).unwrap();
        assert!(bond.props.get_int(keys::SRC_PORT).is_some());
        assert!(bond.props.get_int(keys::DST_PORT).is_some());
        assert!(bond.props.get(keys::BOND_NUMBER).is_some());
    }
}

#[test]
fn a_prefix_enters_through_the_left_terminal() {
    let prefix = fragment("H", "[H][>]");
    let object = StochasticObject::new(
        vec![ethylene()],
        vec![cap_right()],
        Some(terminal(PortKind::Right)),
        None,
        exactly(3.0, Measure::RepeatUnits),
    )
    .unwrap();
    let molecule = Molecule::new(vec![Part::Fixed(prefix), Part::Stochastic(object)]).unwrap();
    let types = bead_types(&molecule.sample(3).unwrap());
    assert_eq!(types[0], "H");
    assert_eq!(count(&types, "E"), 3);
    assert_eq!(count(&types, "<Me"), 1);
    assert_eq!(types.len(), 5);
}

#[test]
fn a_suffix_joins_the_port_kept_for_the_right_terminal() {
    let suffix = fragment("OH", "[<]O");
    let object = StochasticObject::new(
        vec![ethylene()],
        vec![cap_left()],
        None,
        Some(terminal(PortKind::Left)),
        exactly(4.0, Measure::RepeatUnits),
    )
    .unwrap();
    let molecule = Molecule::new(vec![Part::Stochastic(object), Part::Fixed(suffix)]).unwrap();
    let sites = molecule.sample(5).unwrap();
    let types = bead_types(&sites);
    assert_eq!(count(&types, "OH"), 1);
    assert_eq!(count(&types, "E"), 4);
    assert_eq!(sites.n_bonds(), types.len() - 1);
}

#[test]
fn list_weights_pair_units_into_an_alternating_copolymer() {
    // Positions: A< 0, A> 1, B< 2, B> 3, cap< 4.
    let only = |i: usize| SiteWeight::List((0..5).map(|j| F::from(u8::from(i == j))).collect());
    let a = weighted("A", "[<]CC[>]", vec![SiteWeight::default(), only(2)]);
    let b = weighted("B", "[<]CO[>]", vec![SiteWeight::default(), only(0)]);
    let left = listed(PortKind::Right, vec![1.0, 0.0, 0.0, 0.0, 0.0]);
    let object = StochasticObject::new(
        vec![a, b],
        vec![cap_right()],
        Some(left),
        None,
        exactly(8.0, Measure::RepeatUnits),
    )
    .unwrap();
    let molecule = Molecule::new(vec![
        Part::Fixed(fragment("H", "[H][>]")),
        Part::Stochastic(object),
    ])
    .unwrap();
    let types = bead_types(&molecule.sample(11).unwrap());
    let chain: Vec<&str> = types.iter().map(String::as_str).collect();
    assert_eq!(chain, ["H", "A", "B", "A", "B", "A", "B", "A", "B", "<Me"]);
}

#[test]
fn an_open_port_no_end_group_caps_is_stuck() {
    // Only `<`-capping groups: the chain's last `>` cannot be closed.
    let object = StochasticObject::new(
        vec![ethylene()],
        vec![cap_left()],
        None,
        None,
        exactly(2.0, Measure::RepeatUnits),
    )
    .unwrap();
    let err = Molecule::new(vec![Part::Stochastic(object)])
        .unwrap()
        .sample(0)
        .unwrap_err();
    assert!(matches!(err, PolymerError::Stuck(_)), "{err}");
}

#[test]
fn an_object_without_a_size_cannot_be_sampled() {
    assert_eq!(
        homopolymer(None).sample(0).unwrap_err(),
        PolymerError::Unsized
    );
}

// -- validation ---------------------------------------------------------------

#[test]
fn an_end_group_has_exactly_one_port() {
    let err =
        StochasticObject::new(vec![ethylene()], vec![ethylene()], None, None, None).unwrap_err();
    assert!(matches!(err, PolymerError::InvalidObject(_)));
}

#[test]
fn a_list_weight_has_one_entry_per_port_of_the_object() {
    let a = weighted(
        "A",
        "[<]CC[>]",
        vec![SiteWeight::List(vec![1.0, 1.0]), SiteWeight::default()],
    );
    let err = StochasticObject::new(vec![a], vec![cap_right()], None, None, None).unwrap_err();
    assert!(matches!(err, PolymerError::InvalidObject(_)));
}

#[test]
fn a_fixed_unit_has_one_port_per_neighbour() {
    let object = StochasticObject::new(
        vec![ethylene()],
        vec![cap_right()],
        Some(terminal(PortKind::Right)),
        None,
        None,
    )
    .unwrap();
    let err = Molecule::new(vec![
        Part::Fixed(fragment("X", "[>]C[>]")),
        Part::Stochastic(object),
    ])
    .unwrap_err();
    assert!(matches!(err, PolymerError::InvalidMolecule(_)));
}

#[test]
fn a_prefix_must_carry_the_left_terminal() {
    let object = StochasticObject::new(
        vec![ethylene()],
        vec![cap_right()],
        Some(terminal(PortKind::Left)),
        None,
        None,
    )
    .unwrap();
    let err = Molecule::new(vec![
        Part::Fixed(fragment("H", "[H][>]")),
        Part::Stochastic(object),
    ])
    .unwrap_err();
    assert!(matches!(err, PolymerError::InvalidMolecule(_)));
}

#[test]
fn two_different_units_may_not_share_a_name() {
    let object = StochasticObject::new(
        vec![ethylene(), fragment("E", "[<]CO[>]")],
        vec![cap_left(), cap_right()],
        None,
        None,
        None,
    )
    .unwrap();
    let err = Molecule::new(vec![Part::Stochastic(object)]).unwrap_err();
    assert!(matches!(err, PolymerError::InvalidMolecule(_)));
}

#[test]
fn two_objects_in_a_row_join_and_may_share_a_unit() {
    // A diblock of one repeat unit: the first object keeps a `>` open for
    // its right terminal `<`, which the second object's left terminal `>`
    // faces. `E` appears in both objects as one fragment.
    let e = ethylene();
    let first = StochasticObject::new(
        vec![e.clone()],
        vec![cap_left()],
        None,
        Some(terminal(PortKind::Left)),
        exactly(3.0, Measure::RepeatUnits),
    )
    .unwrap();
    let second = StochasticObject::new(
        vec![e],
        vec![cap_right()],
        Some(terminal(PortKind::Right)),
        None,
        exactly(5.0, Measure::RepeatUnits),
    )
    .unwrap();
    let molecule = Molecule::new(vec![Part::Stochastic(first), Part::Stochastic(second)]).unwrap();
    let sites = molecule.sample(8).unwrap();
    let types = bead_types(&sites);
    assert_eq!(count(&types, "E"), 8);
    assert_eq!(types.len(), 10);
    assert_eq!(sites.n_bonds(), 9, "one chain");
}

// -- system -------------------------------------------------------------------

#[test]
fn percentages_are_shares_of_the_total_the_absolute_amounts_fix() {
    let molecule = homopolymer(None);
    let system = System::new(
        vec![
            (molecule.clone(), Amount::Total(1000.0)),
            (molecule, Amount::Percent(20.0)),
        ],
        Measure::HeavyAtomMass,
    )
    .unwrap();
    // 1000 is 80 % of the system, so the system is 1250 and 20 % of it 250.
    let targets = system.targets();
    assert!((targets[0] - 1000.0).abs() < 1e-9);
    assert!((targets[1] - 250.0).abs() < 1e-9);
}

#[test]
fn each_component_is_sampled_until_it_reaches_its_target() {
    let size = ChainSize::new(Distribution::Poisson { n: 20.0 }, Measure::RepeatUnits).unwrap();
    let system = System::new(
        vec![(homopolymer(Some(size)), Amount::Total(5000.0))],
        Measure::HeavyAtomMass,
    )
    .unwrap();
    let sample = system.sample(13).unwrap();
    let masses: Vec<F> = sample.molecules.iter().map(|m| m.heavy_atom_mass).collect();
    let total: F = masses.iter().sum();
    assert!(total >= 5000.0);
    assert!(
        total - masses.last().unwrap() < 5000.0,
        "one molecule too many"
    );
    let n_sites: usize = sample.molecules.iter().map(|m| m.n_fragments).sum();
    assert_eq!(sample.sites.n_beads(), n_sites);
}

#[test]
fn flory_schulz_chain_lengths_keep_their_mean() {
    // Growth stops at the first repeat count reaching an integer target, so
    // chain lengths are the drawn targets: mean (2 − a) / a.
    let a = 0.1;
    let size = ChainSize::new(Distribution::FlorySchulz { a }, Measure::RepeatUnits).unwrap();
    let system = System::new(
        vec![(homopolymer(Some(size)), Amount::Total(60_000.0))],
        Measure::RepeatUnits,
    )
    .unwrap();
    let sample = system.sample(21).unwrap();
    let n = sample.molecules.len() as F;
    let mean = sample
        .molecules
        .iter()
        .map(|m| m.n_repeat_units as F)
        .sum::<F>()
        / n;
    let expected = (2.0 - a) / a;
    assert!(
        (mean - expected).abs() < 0.05 * expected,
        "{mean} vs {expected}"
    );
}

#[test]
fn two_different_units_may_not_share_a_name_across_components() {
    let err = System::new(
        vec![
            (homopolymer(None), Amount::Total(1.0)),
            (homopolymer(None), Amount::Total(1.0)),
        ],
        Measure::HeavyAtomMass,
    )
    .unwrap_err();
    assert!(matches!(err, PolymerError::InvalidSystem(_)));
}

#[test]
fn a_system_needs_an_absolute_amount_and_percentages_below_100() {
    let only_percent = System::new(
        vec![(homopolymer(None), Amount::Percent(50.0))],
        Measure::HeavyAtomMass,
    );
    assert!(matches!(only_percent, Err(PolymerError::InvalidSystem(_))));
    let molecule = homopolymer(None);
    let too_much = System::new(
        vec![
            (molecule.clone(), Amount::Total(1.0)),
            (molecule.clone(), Amount::Percent(60.0)),
            (molecule, Amount::Percent(40.0)),
        ],
        Measure::HeavyAtomMass,
    );
    assert!(matches!(too_much, Err(PolymerError::InvalidSystem(_))));
}

// -- review regressions ---------------------------------------------------------

#[test]
fn a_chain_holds_a_repeat_unit_whatever_the_target() {
    // A target of zero is below the starting end group already; the chain
    // is still Me-E-Me, never the end-group dimer.
    for measure in [Measure::RepeatUnits, Measure::HeavyAtomMass, Measure::Mass] {
        let types = bead_types(&homopolymer(exactly(0.0, measure)).sample(1).unwrap());
        assert_eq!(count(&types, "E"), 1, "{measure:?}");
        assert_eq!(types.len(), 3);
    }
}

#[test]
fn list_weights_cannot_end_a_chain_before_its_first_repeat_unit() {
    // The starting cap pairs with the other cap nine times in ten; a chain
    // still starts with a repeat unit.
    // Positions: E< 0, E> 1, Me> 2, <Me 3.
    let start = weighted(
        "Me>",
        "C[>]",
        vec![SiteWeight::List(vec![1.0, 0.0, 0.0, 9.0])],
    );
    let object = StochasticObject::new(
        vec![ethylene()],
        vec![start, cap_right()],
        None,
        None,
        exactly(1.0, Measure::RepeatUnits),
    )
    .unwrap();
    let molecule = Molecule::new(vec![Part::Stochastic(object)]).unwrap();
    for seed in 0..50 {
        let types = bead_types(&molecule.sample(seed).unwrap());
        assert!(count(&types, "E") >= 1, "seed {seed}: {types:?}");
    }
}

/// `C[>]` + object + `[<]C`, whose repeat unit's `>` end closes on a cap
/// with probability `p_cap` at each step.
fn early_capping(p_cap: F, n: F) -> Molecule {
    // Positions: E< 0, E> 1, <H 2.
    let unit = weighted(
        "E",
        "[<]CC[>]",
        vec![
            SiteWeight::default(),
            SiteWeight::List(vec![1.0 - p_cap, 0.0, p_cap]),
        ],
    );
    let object = StochasticObject::new(
        vec![unit],
        vec![fragment("<H", "[<][H]")],
        Some(terminal(PortKind::Right)),
        Some(terminal(PortKind::Left)),
        exactly(n, Measure::RepeatUnits),
    )
    .unwrap();
    Molecule::new(vec![
        Part::Fixed(cap_left()),
        Part::Stochastic(object),
        Part::Fixed(cap_right()),
    ])
    .unwrap()
}

#[test]
fn growth_never_spends_the_port_the_right_terminal_needs() {
    // Capping the growing end would leave nothing for the suffix; every
    // seed must still reach it.
    let molecule = early_capping(0.5, 6.0);
    for seed in 0..200 {
        let sites = molecule.sample(seed).unwrap();
        let types = bead_types(&sites);
        assert_eq!(count(&types, "<Me"), 1, "seed {seed}");
        assert_eq!(
            count(&types, "<H"),
            0,
            "seed {seed}: the only exit was capped"
        );
        assert_eq!(count(&types, "E"), 6);
        assert_eq!(sites.n_bonds(), types.len() - 1);
    }
}

#[test]
fn right_terminal_list_weights_choose_the_exit() {
    // A three-port unit leaves two `>` ports open; the right terminal's
    // list sends the suffix to the second one (position 2) only.
    // Positions: T< 0, T> 1, T> 2, <H 3.
    let unit = fragment("T", "[<]CC([>])[>]");
    let object = StochasticObject::new(
        vec![unit],
        vec![fragment("<H", "[<][H]")],
        Some(terminal(PortKind::Right)),
        Some(listed(PortKind::Left, vec![0.0, 0.0, 1.0, 0.0])),
        exactly(1.0, Measure::RepeatUnits),
    )
    .unwrap();
    let molecule = Molecule::new(vec![
        Part::Fixed(cap_left()),
        Part::Stochastic(object),
        Part::Fixed(cap_right()),
    ])
    .unwrap();
    for seed in 0..20 {
        let sites = molecule.sample(seed).unwrap();
        let suffix_port = sites
            .bond_ids()
            .map(|id| sites.get_bond(id).unwrap().props)
            .find(|b| b.get_int(keys::SRC_PORT) == Some(2))
            .is_some();
        assert!(suffix_port, "seed {seed}: the suffix joins port 2 of T");
        // Port 1 is capped by <H, port 2 carries the suffix.
        let types = bead_types(&sites);
        assert_eq!(count(&types, "<H"), 1);
        assert_eq!(count(&types, "<Me"), 1);
    }
}

#[test]
fn uniform_repeat_units_reaches_its_lower_bound() {
    let size = ChainSize::new(
        Distribution::Uniform {
            low: 2.0,
            high: 5.0,
        },
        Measure::RepeatUnits,
    )
    .unwrap();
    let molecule = homopolymer(Some(size));
    let mut seen = [false; 6];
    for seed in 0..200 {
        seen[count(&bead_types(&molecule.sample(seed).unwrap()), "E")] = true;
    }
    assert_eq!(seen, [false, false, true, true, true, true]);
}

#[test]
fn branched_growth_is_linear_in_the_number_of_fragments() {
    // 20000 three-port units: quadratic bookkeeping took minutes here.
    let object = StochasticObject::new(
        vec![fragment("T", "[<]CC([>])[>]")],
        vec![cap_left(), cap_right()],
        None,
        None,
        exactly(20_000.0, Measure::RepeatUnits),
    )
    .unwrap();
    let started = std::time::Instant::now();
    let sites = Molecule::new(vec![Part::Stochastic(object)])
        .unwrap()
        .sample(1)
        .unwrap();
    assert_eq!(count(&bead_types(&sites), "T"), 20_000);
    assert_eq!(sites.n_bonds(), sites.n_beads() - 1, "a tree");
    assert!(started.elapsed().as_secs() < 20, "{:?}", started.elapsed());
}

// -- refusals at construction ---------------------------------------------------

#[test]
fn a_first_object_needs_an_end_group_to_start_from() {
    let object =
        StochasticObject::new(vec![fragment("E", "[$]CC[$]")], vec![], None, None, None).unwrap();
    let err = Molecule::new(vec![Part::Stochastic(object)]).unwrap_err();
    assert!(matches!(err, PolymerError::InvalidMolecule(_)), "{err}");
}

#[test]
fn a_terminal_must_face_a_part() {
    let caps = || vec![cap_left(), cap_right()];
    let dangling_right = StochasticObject::new(
        vec![ethylene()],
        caps(),
        None,
        Some(terminal(PortKind::Left)),
        None,
    )
    .unwrap();
    let dangling_left = StochasticObject::new(
        vec![ethylene()],
        caps(),
        Some(terminal(PortKind::Right)),
        None,
        None,
    )
    .unwrap();
    for object in [dangling_right, dangling_left] {
        let err = Molecule::new(vec![Part::Stochastic(object)]).unwrap_err();
        assert!(matches!(err, PolymerError::InvalidMolecule(_)), "{err}");
    }
}

#[test]
fn a_terminal_no_repeat_unit_accepts_is_refused() {
    // `<` accepts `>` only; a `$`-only unit offers none.
    for (left, right) in [
        (Some(terminal(PortKind::Left)), None),
        (None, Some(terminal(PortKind::Left))),
    ] {
        let err = StochasticObject::new(vec![fragment("E", "[$]CC[$]")], vec![], left, right, None)
            .unwrap_err();
        assert!(matches!(err, PolymerError::InvalidObject(_)), "{err}");
    }
}

#[test]
fn terminal_inputs_are_checked() {
    let bad = [
        Terminal::new(PortKind::Shared, "", BondNumber::Single, None),
        Terminal::new(PortKind::Left, "", BondNumber::Unknown, None),
        Terminal::new(PortKind::Left, "", BondNumber::Single, Some(vec![])),
        Terminal::new(PortKind::Left, "", BondNumber::Single, Some(vec![-1.0])),
    ];
    for terminal in bad {
        assert!(matches!(terminal, Err(PolymerError::InvalidObject(_))));
    }
}

#[test]
fn a_repeat_unit_that_adds_nothing_to_the_measure_is_refused() {
    // A hydrogen-only repeat unit never reaches a heavy-atom target.
    let err = StochasticObject::new(
        vec![fragment("H2", "[<][H][H][>]")],
        vec![cap_left(), cap_right()],
        None,
        None,
        exactly(100.0, Measure::HeavyAtomMass),
    );
    assert!(matches!(err, Err(PolymerError::InvalidObject(_))));
}

#[test]
fn a_molecule_that_can_measure_zero_is_not_a_system_component() {
    // Water has no repeat unit: no number of copies counts one.
    let water = Molecule::new(vec![Part::Fixed(fragment("W", "O"))]).unwrap();
    let err = System::new(vec![(water, Amount::Total(10.0))], Measure::RepeatUnits).unwrap_err();
    assert!(matches!(err, PolymerError::InvalidSystem(_)), "{err}");
}
