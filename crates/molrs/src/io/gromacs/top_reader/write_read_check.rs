//! GROMACS topologies through molrs alone: the LAMMPS writer refuses what it
//! cannot hold, and the GROMACS writer writes back what the reader read.
//!
//! The fixtures are an ACE-ALA-ALA-NME dipeptide under GROMACS's charmm27
//! (Urey-Bradley, CMAP, `[ pairtypes ]`, a `[ nonbond_params ]` row),
//! amber99sb-ildn (funct-9 multi-term dihedrals, funct-4 impropers, gen-pairs
//! 0.5 / 0.8333) and oplsaa (funct-3 Ryckaert-Bellemans, funct-1 impropers
//! from `#define` macros), and the AMBER one with two `[ pairs ]` rows of
//! their own (funct 1 with parameters, funct 2).

use super::GromacsTopForcefieldReader;
use crate::ff::forcefield::ForceField;
use crate::ff::ir::Params;
use crate::io::writer::WrittenText;

struct Fixture {
    name: &'static str,
    top: &'static str,
}

const FIXTURES: [Fixture; 4] = [
    Fixture {
        name: "charmm",
        top: include_str!("../testdata/charmm.top"),
    },
    Fixture {
        name: "amber",
        top: include_str!("../testdata/amber.top"),
    },
    Fixture {
        name: "opls",
        top: include_str!("../testdata/opls.top"),
    },
    Fixture {
        name: "amber_pairs",
        top: include_str!("../testdata/amber_pairs.top"),
    },
];

/// Per-pair `[ pairs ]` parameters have no LAMMPS form: the writer refuses
/// them by name instead of dropping them.
#[test]
fn per_pair_parameters_are_refused_by_the_lammps_writer() {
    let f = &FIXTURES[3];
    let (_, frame) = GromacsTopForcefieldReader::new()
        .read_system(f.top.as_bytes())
        .unwrap_or_else(|e| panic!("{}: {e}", f.name));
    let err = crate::io::lammps::forcefield_writer::refuse_inexpressible_pairs(&frame)
        .expect_err("overrides");
    assert!(err.contains("epsilon"), "{err}");
}

/// The fixtures' directives — pairtypes, NBFIX, UB, funct 9, RB, cmap —
/// written back by the GROMACS writer read as the same force field: every
/// style, type name, endpoint and parameter (to 1e-12, the decimal print).
#[test]
fn fixture_directives_survive_write_then_read() {
    use crate::io::gromacs::top_writer::GromacsTopForcefieldWriter;
    use crate::io::reader::ForceFieldReader;
    let reader = [
        "constrainttypes",
        "moleculetype",
        "atoms",
        "bonds",
        "pairs",
        "angles",
        "dihedrals",
        "cmap",
        "system",
        "molecules",
    ]
    .iter()
    .fold(GromacsTopForcefieldReader::new(), |r, s| {
        r.with_skipped_directive(s)
    });
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1e-3);
    for f in &FIXTURES[..3] {
        let ff = reader.read(f.top.as_bytes()).unwrap();
        let text = GromacsTopForcefieldWriter::new()
            .with_precision(17)
            .written(&ff)
            .unwrap_or_else(|e| panic!("{}: {e}", f.name));
        let back = reader
            .read(text.as_bytes())
            .unwrap_or_else(|e| panic!("{}: {e}\n{text}", f.name));
        assert_eq!(back.special_bonds(), ff.special_bonds(), "{}", f.name);
        let key = |ff: &ForceField| {
            let mut k: Vec<(String, String)> = ff
                .styles()
                .iter()
                .map(|s| (s.category().to_owned(), s.name().to_owned()))
                .collect();
            k.sort();
            k
        };
        assert_eq!(key(&back), key(&ff), "{}: styles", f.name);
        for style in ff.styles() {
            let other = back.get_style(style.category(), style.name()).unwrap();
            let what = format!("{} {}/{}", f.name, style.category(), style.name());
            assert_eq!(
                other
                    .params()
                    .iter()
                    .filter_map(|(k, v)| v.as_text().map(|s| (k, s)))
                    .collect::<Vec<_>>()
                    .len(),
                style
                    .params()
                    .iter()
                    .filter_map(|(k, v)| v.as_text().map(|s| (k, s)))
                    .count(),
                "{what}: string params"
            );
            for (k, v) in style
                .params()
                .iter()
                .filter_map(|(k, v)| v.as_text().map(|s| (k, s)))
            {
                assert_eq!(
                    other.params().get(k).and_then(|v| v.as_text()),
                    Some(v),
                    "{what}: {k}"
                );
            }
            let mut want = style.type_rows();
            let mut got = other.type_rows();
            want.sort_by(|a, b| a.0.cmp(b.0));
            got.sort_by(|a, b| a.0.cmp(b.0));
            let names = |rows: &[(&str, Vec<&str>, &Params)]| -> Vec<String> {
                rows.iter().map(|r| r.0.to_owned()).collect()
            };
            assert_eq!(names(&got), names(&want), "{what}: type names");
            for ((name, ends, p), (_, got_ends, q)) in want.iter().zip(&got) {
                assert_eq!(got_ends, ends, "{what} {name}: endpoints");
                let mut keys: Vec<&str> = p
                    .iter()
                    .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
                    .map(|(k, _)| k)
                    .collect();
                let mut got_keys: Vec<&str> = q
                    .iter()
                    .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
                    .map(|(k, _)| k)
                    .collect();
                keys.sort_unstable();
                got_keys.sort_unstable();
                assert_eq!(got_keys, keys, "{what} {name}: keys");
                for (k, v) in p.iter().filter_map(|(k, v)| v.as_num().map(|x| (k, x))) {
                    let g = q.get(k).and_then(|v| v.as_num()).unwrap();
                    assert!(close(g, v), "{what} {name}.{k}: {g} vs {v}");
                }
                for (k, a) in p.iter().filter_map(|(k, v)| v.as_array().map(|a| (k, a))) {
                    let b = q.get(k).and_then(|v| v.as_array()).expect("array param");
                    assert_eq!(a.shape(), b.shape(), "{what} {name}.{k}");
                    assert!(
                        a.iter().zip(b).all(|(x, y)| close(*x, *y)),
                        "{what} {name}.{k}"
                    );
                }
            }
        }
    }
}
