//! The engine against hand formulas, finite differences, the native molrs
//! kernels (the protocol's Appendix A expressions) and its own printer.

use crate::ff::potential::fixtures::OneShot;
use std::collections::HashMap;

use super::*;
use crate::ff::compile::PotentialCompiler;
use crate::ff::forcefield::ForceField;
use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::{compute_angle, compute_dihedral};
use molrs::core::Block;
use molrs::core::Frame;
use molrs::op::{F, Idx};
use ndarray::Array1;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn close(got: F, want: F, rel: F) -> bool {
    (got - want).abs() <= rel * want.abs().max(1.0)
}

/// Columns for `c.inputs()` from a spelling → values table.
fn columns<'a>(c: &Compiled, values: &'a HashMap<String, Vec<F>>) -> Vec<&'a [F]> {
    c.gather(|i| values.get(&i.spelling()).map(Vec::as_slice))
        .expect("every input has a value")
}

fn table(pairs: &[(&str, F)]) -> HashMap<String, Vec<F>> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), vec![*v]))
        .collect()
}

fn names<'a>(p: &[(&'a str, F)]) -> Vec<&'a str> {
    p.iter().map(|p| p.0).collect()
}

/// A style's expression against its hand formula of the scalar coordinate
/// at `qs`: energy to 1e-12 relative, dE/dq to 1e-7 against a central
/// difference of both the hand formula and the expression itself.
fn check_scalar(
    geometry: Geometry,
    params: &[(&str, F)],
    style: &[(&str, F)],
    expression: &str,
    qs: &[F],
    hand: impl Fn(F) -> F,
) {
    let c = compile(
        expression,
        &Binding::new(geometry, &names(params), &names(style)),
    )
    .unwrap_or_else(|e| panic!("{expression}: {e}"));
    assert_eq!(c.source(), expression, "the source is kept byte for byte");
    assert!(c.has_scalar_form(), "{expression}");
    let mut all = params.to_vec();
    all.extend_from_slice(style);
    let values = table(&all);
    let cols = columns(&c, &values);

    let mut e = vec![0.0; qs.len()];
    let mut d = vec![0.0; qs.len()];
    c.eval_scalar(qs, &cols, &mut e, &mut d);

    let h = 1e-5;
    for (t, &q) in qs.iter().enumerate() {
        let want = hand(q);
        assert!(
            close(e[t], want, 1e-12),
            "{expression} at {q}: E {} vs hand {want}",
            e[t]
        );
        let fd_hand = (hand(q + h) - hand(q - h)) / (2.0 * h);
        assert!(
            close(d[t], fd_hand, 1e-7),
            "{expression} at {q}: dE/dq {} vs hand central difference {fd_hand}",
            d[t]
        );
        let (mut ep, mut em, mut dd) = ([0.0], [0.0], [0.0]);
        c.eval_scalar(&[q + h], &cols, &mut ep, &mut dd);
        c.eval_scalar(&[q - h], &cols, &mut em, &mut dd);
        let fd_expr = (ep[0] - em[0]) / (2.0 * h);
        assert!(
            close(d[t], fd_expr, 1e-7),
            "{expression} at {q}: dE/dq {} vs its own central difference {fd_expr}",
            d[t]
        );
    }
}

// ---------------------------------------------------------------------------
// the built-in styles, as the protocol's Appendix A writes them
// ---------------------------------------------------------------------------

const BOND_HARMONIC: &str = "k*(r-r0)^2";
const BOND_MORSE: &str = "d0*(1-exp(-alpha*(r-r0)))^2";
const BOND_CLASS2: &str = "k2*d^2+k3*d^3+k4*d^4; d=r-r0";
const ANGLE_HARMONIC: &str = "k*(theta-theta0*(pi/180))^2; pi=3.141592653589793";
const ANGLE_CHARMM: &str =
    "k*(theta-theta0*(pi/180))^2+k_ub*(distance(p1,p3)-r_ub)^2; pi=3.141592653589793";
const ANGLE_CLASS2: &str = "k2*d^2+k3*d^3+k4*d^4; d=theta-theta0*(pi/180); pi=3.141592653589793";
const DIHEDRAL_PERIODIC: &str = "k*(1+cos(periodicity*phi-phase*(pi/180))); pi=3.141592653589793";
const DIHEDRAL_PERIODIC_2: &str = "k1*(1+cos(periodicity1*phi-phase1*(pi/180))) + \
                                   k2*(1+cos(periodicity2*phi-phase2*(pi/180))); pi=3.141592653589793";
const DIHEDRAL_OPLS: &str =
    "0.5*(k1*(1+cos(phi))+k2*(1-cos(2*phi))+k3*(1+cos(3*phi))+k4*(1-cos(4*phi)))";
const DIHEDRAL_HARMONIC: &str = "k*(1+sign*cos(periodicity*phi))";
const DIHEDRAL_MULTI_HARMONIC: &str = "a1+a2*c+a3*c^2+a4*c^3+a5*c^4; c=cos(phi)";
const DIHEDRAL_NHARMONIC_7: &str = "a1+a2*c+a3*c^2+a4*c^3+a5*c^4+a6*c^5+a7*c^6; c=cos(phi)";
const DIHEDRAL_CLASS2: &str = "k1*(1-cos(phi-phi1*(pi/180)))+\
                               k2*(1-cos(2*phi-phi2*(pi/180)))+\
                               k3*(1-cos(3*phi-phi3*(pi/180))); pi=3.141592653589793";
const DIHEDRAL_RB: &str = "c0+c1*c+c2*c^2+c3*c^3+c4*c^4+c5*c^5; c=-cos(phi)";
const IMPROPER_HARMONIC: &str = "k*(chi-chi0*(pi/180))^2; pi=3.141592653589793";
const IMPROPER_CVFF: &str = "k*(1+sign*cos(periodicity*phi))";
const IMPROPER_PERIODIC: &str = "k*(1+cos(periodicity*phi-phase*(pi/180))); pi=3.141592653589793";
const PAIR_LJ_CUT: &str = "C*epsilon*((sigma/r)^n-(sigma/r)^m-select(shift,(sigma/cutoff)^n-(sigma/cutoff)^m,0)); C=n/(n-m)*(n/m)^(m/(n-m))";
const PAIR_LJ_12_6: &str = "4*epsilon*((sigma/r)^12-(sigma/r)^6)";
const PAIR_LJ_CHARMM: &str = "4*epsilon*((sigma/r)^12-(sigma/r)^6)*S; S=select(step(inner-r),1,(cutoff^2-r^2)^2*(cutoff^2+2*r^2-3*inner^2)/(cutoff^2-inner^2)^3)";
const PAIR_COUL_CHARMM: &str = "coulomb*q1*q2/(dielectric*r)*S; S=select(step(inner-r),1,(cutoff^2-r^2)^2*(cutoff^2+2*r^2-3*inner^2)/(cutoff^2-inner^2)^3)";
const PAIR_COUL_CUT: &str = "coulomb*q1*q2/(dielectric*(r+delta))";
const PAIR_LJ_CLASS2: &str = "epsilon*(2*(sigma/r)^9-3*(sigma/r)^6)";
const PAIR_BUCK: &str = "a*exp(-r/rho)-c/r^6";
const PAIR_MORSE: &str = "d0*((1-exp(-alpha*(r-r0)))^2-1)";
const FENE: &str = "-0.5*k*r0^2*log(1-(r/r0)^2) + step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)";

/// Signed dihedrals across (−π, π], away from the endpoints.
const PHIS: [F; 7] = [-2.9, -1.7, -0.4, 0.3, 1.1, 2.2, 3.0];

#[test]
fn bonds_are_the_hand_formula() {
    let (k, r0) = (150.3, 1.53);
    check_scalar(
        Geometry::Bond,
        &[("k", k), ("r0", r0)],
        &[],
        BOND_HARMONIC,
        &[1.2, 1.53, 1.9],
        |r| k * (r - r0).powi(2),
    );
    let (d0, alpha, r0) = (4.2, 1.7, 1.1);
    check_scalar(
        Geometry::Bond,
        &[("d0", d0), ("alpha", alpha), ("r0", r0)],
        &[],
        BOND_MORSE,
        &[0.8, 1.1, 1.6, 3.0],
        |r| d0 * (1.0 - (-alpha * (r - r0)).exp()).powi(2),
    );
    let (r0, k2, k3, k4) = (1.5, 300.0, -400.0, 500.0);
    check_scalar(
        Geometry::Bond,
        &[("r0", r0), ("k2", k2), ("k3", k3), ("k4", k4)],
        &[],
        BOND_CLASS2,
        &[1.3, 1.5, 1.75],
        |r| {
            let d = r - r0;
            k2 * d * d + k3 * d.powi(3) + k4 * d.powi(4)
        },
    );
}

#[test]
fn angles_take_theta0_in_degrees() {
    let (k, theta0) = (60.5, 109.47);
    check_scalar(
        Geometry::Angle,
        &[("k", k), ("theta0", theta0)],
        &[],
        ANGLE_HARMONIC,
        &[1.6, 1.9106, 2.4],
        |t| k * (t - theta0.to_radians()).powi(2),
    );
    let (theta0, k2, k3, k4) = (112.0, 40.0, -20.0, 10.0);
    check_scalar(
        Geometry::Angle,
        &[("theta0", theta0), ("k2", k2), ("k3", k3), ("k4", k4)],
        &[],
        ANGLE_CLASS2,
        &[1.7, 2.0, 2.3],
        |t| {
            let d = t - theta0.to_radians();
            k2 * d * d + k3 * d.powi(3) + k4 * d.powi(4)
        },
    );
    // `x*(pi/180)` with `pi` the shortest decimal of π is `x.to_radians()`
    // to the bit (the factor `f64::to_radians` multiplies by).
    let c = compile(
        "theta-theta0*(pi/180); pi=3.141592653589793",
        &Binding::new(Geometry::Angle, &["theta0"], &[]),
    )
    .unwrap();
    let values = table(&[("theta0", theta0)]);
    let cols = columns(&c, &values);
    let (mut e, mut d) = ([0.0], [0.0]);
    c.eval_scalar(&[1.9], &cols, &mut e, &mut d);
    assert_eq!(e[0], 1.9 - theta0.to_radians());
}

#[test]
fn dihedrals_are_the_hand_formula() {
    let (k, n, phase) = (1.3, 3.0, 15.0);
    // `dihedral charmm`'s torsion is the same function (its `w` 1-4 pair is
    // outside the term).
    check_scalar(
        Geometry::Dihedral,
        &[("k", k), ("periodicity", n), ("phase", phase)],
        &[],
        DIHEDRAL_PERIODIC,
        &PHIS,
        |p| k * (1.0 + (n * p - phase.to_radians()).cos()),
    );
    let p2 = [
        ("k1", 0.7),
        ("periodicity1", 1.0),
        ("phase1", 0.0),
        ("k2", 0.25),
        ("periodicity2", 2.0),
        ("phase2", 180.0),
    ];
    check_scalar(
        Geometry::Dihedral,
        &p2,
        &[],
        DIHEDRAL_PERIODIC_2,
        &PHIS,
        |p| 0.7 * (1.0 + p.cos()) + 0.25 * (1.0 + (2.0 * p - 180f64.to_radians()).cos()),
    );
    let a = [0.3, -1.2, 0.8, 0.45, -0.6];
    let p: Vec<(&str, F)> = ["a1", "a2", "a3", "a4", "a5"].into_iter().zip(a).collect();
    check_scalar(
        Geometry::Dihedral,
        &p,
        &[],
        DIHEDRAL_MULTI_HARMONIC,
        &PHIS,
        |phi| (0..5).map(|n| a[n] * phi.cos().powi(n as i32)).sum(),
    );
    let k = [1.1, -0.4, 0.9, 0.2];
    let p: Vec<(&str, F)> = ["k1", "k2", "k3", "k4"].into_iter().zip(k).collect();
    check_scalar(Geometry::Dihedral, &p, &[], DIHEDRAL_OPLS, &PHIS, |f| {
        0.5 * (k[0] * (1.0 + f.cos())
            + k[1] * (1.0 - (2.0 * f).cos())
            + k[2] * (1.0 + (3.0 * f).cos())
            + k[3] * (1.0 - (4.0 * f).cos()))
    });
    check_scalar(
        Geometry::Dihedral,
        &[("k", 2.0), ("sign", -1.0), ("periodicity", 3.0)],
        &[],
        DIHEDRAL_HARMONIC,
        &PHIS,
        |f| 2.0 * (1.0 - (3.0 * f).cos()),
    );
    let kc = [(0.5, 10.0), (-0.3, 170.0), (0.2, 45.0)];
    let p = [
        ("k1", kc[0].0),
        ("phi1", kc[0].1),
        ("k2", kc[1].0),
        ("phi2", kc[1].1),
        ("k3", kc[2].0),
        ("phi3", kc[2].1),
    ];
    check_scalar(Geometry::Dihedral, &p, &[], DIHEDRAL_CLASS2, &PHIS, |f| {
        (0..3)
            .map(|i| kc[i].0 * (1.0 - ((i + 1) as F * f - kc[i].1.to_radians()).cos()))
            .sum()
    });
    // `dihedral rb`: expression-only in molrs (D19, no native kernel); RB's
    // ψ = φ − 180°, so cos ψ = −cos φ.
    let c = [1.0, 0.5, -1.5, 0.25, 0.75, -0.1];
    let p: Vec<(&str, F)> = ["c0", "c1", "c2", "c3", "c4", "c5"]
        .into_iter()
        .zip(c)
        .collect();
    check_scalar(Geometry::Dihedral, &p, &[], DIHEDRAL_RB, &PHIS, |f| {
        let psi = f - std::f64::consts::PI;
        (0..6).map(|n| c[n] * psi.cos().powi(n as i32)).sum()
    });
}

#[test]
fn impropers_bind_phi_and_chi() {
    // dE/dq is dE/dφ: the chain through χ = |φ| is the engine's.
    let (k, chi0) = (40.0, 12.0);
    check_scalar(
        Geometry::Improper,
        &[("k", k), ("chi0", chi0)],
        &[],
        IMPROPER_HARMONIC,
        &PHIS,
        |p| k * (p.abs() - chi0.to_radians()).powi(2),
    );
    let (k, s, n) = (2.5, -1.0, 2.0);
    check_scalar(
        Geometry::Improper,
        &[("k", k), ("sign", s), ("periodicity", n)],
        &[],
        IMPROPER_CVFF,
        &PHIS,
        |p| k * (1.0 + s * (n * p).cos()),
    );
    // A phase that is neither 0 nor 180 needs the signed φ.
    let (k, n, phase) = (1.1, 2.0, 30.0);
    check_scalar(
        Geometry::Improper,
        &[("k", k), ("periodicity", n), ("phase", phase)],
        &[],
        IMPROPER_PERIODIC,
        &PHIS,
        |p| k * (1.0 + (n * p - phase.to_radians()).cos()),
    );
}

#[test]
fn pairs_are_the_hand_formula() {
    let (eps, sig) = (0.238, 3.4);
    let rs = [3.0, 2f64.powf(1.0 / 6.0) * sig, 5.5, 8.5];
    let lj = |r: F| 4.0 * eps * ((sig / r).powi(12) - (sig / r).powi(6));
    let p = [("epsilon", eps), ("sigma", sig)];
    check_scalar(Geometry::Pair, &p, &[], PAIR_LJ_12_6, &rs, lj);
    // The full `lj/cut`: n, m, shift and cutoff are style params.
    let style = [("n", 12.0), ("m", 6.0), ("shift", 0.0), ("cutoff", 10.0)];
    check_scalar(Geometry::Pair, &p, &style, PAIR_LJ_CUT, &rs, lj);
    let style = [("n", 12.0), ("m", 6.0), ("shift", 1.0), ("cutoff", 10.0)];
    check_scalar(Geometry::Pair, &p, &style, PAIR_LJ_CUT, &rs, |r| {
        lj(r) - lj(10.0)
    });
    let style = [("n", 9.0), ("m", 6.0), ("shift", 0.0), ("cutoff", 10.0)];
    check_scalar(Geometry::Pair, &p, &style, PAIR_LJ_CUT, &rs, |r| {
        let c = 9.0 / 3.0 * 1.5f64.powf(2.0);
        c * eps * ((sig / r).powi(9) - (sig / r).powi(6))
    });

    check_scalar(Geometry::Pair, &p, &[], PAIR_LJ_CLASS2, &rs, |r| {
        eps * (2.0 * (sig / r).powi(9) - 3.0 * (sig / r).powi(6))
    });

    let (inner, cutoff) = (8.0, 10.0);
    let switch = |r: F| {
        if r <= inner {
            1.0
        } else {
            let (c2, r2, i2) = (cutoff * cutoff, r * r, inner * inner);
            (c2 - r2).powi(2) * (c2 + 2.0 * r2 - 3.0 * i2) / (c2 - i2).powi(3)
        }
    };
    let charmm = [("inner", inner), ("cutoff", cutoff)];
    check_scalar(
        Geometry::Pair,
        &p,
        &charmm,
        PAIR_LJ_CHARMM,
        &[3.5, 7.0, 8.5, 9.7],
        |r| lj(r) * switch(r),
    );

    // `coul/charmm` is energy-only in the gate (D15): its force is not the
    // gradient of its energy.
    let (cc, diel, q1, q2) = (332.06371, 2.0, 0.4, -0.8);
    let coul = [
        ("coulomb", cc),
        ("dielectric", diel),
        ("inner", inner),
        ("cutoff", cutoff),
    ];
    let c = compile(
        PAIR_COUL_CHARMM,
        &Binding::new(Geometry::Pair, &[], &names(&coul)),
    )
    .unwrap();
    let mut values = table(&coul);
    values.extend(table(&[("q1", q1), ("q2", q2)]));
    for r in [3.0, 8.5, 9.9] {
        let mut e = [0.0];
        c.eval_scalar(&[r], &columns(&c, &values), &mut e, &mut [0.0]);
        let want = cc * q1 * q2 / (diel * r) * switch(r);
        assert!(close(e[0], want, 1e-12), "{} vs {want}", e[0]);
    }

    let (a, rho, c6) = (1388.77, 0.3623, 175.0);
    check_scalar(
        Geometry::Pair,
        &[("a", a), ("rho", rho), ("c", c6)],
        &[],
        PAIR_BUCK,
        &[1.5, 2.5, 4.0],
        |r| a * (-r / rho).exp() - c6 / r.powi(6),
    );
    let (d0, alpha, r0) = (0.5, 1.3, 3.0);
    check_scalar(
        Geometry::Pair,
        &[("d0", d0), ("alpha", alpha), ("r0", r0)],
        &[],
        PAIR_MORSE,
        &[2.0, 3.0, 4.5],
        |r| d0 * ((1.0 - (-alpha * (r - r0)).exp()).powi(2) - 1.0),
    );
}

#[test]
fn pair_coul_cut_reads_charges_and_style_params() {
    let (cc, diel, delta, q1, q2) = (332.06371, 2.0, 0.1, 0.4, -0.8);
    let c = compile(
        PAIR_COUL_CUT,
        &Binding::new(
            Geometry::Pair,
            &[],
            &["coulomb", "dielectric", "delta", "cutoff"],
        ),
    )
    .unwrap();
    assert_eq!(
        c.inputs(),
        &[
            Input::Charge(1),
            Input::Charge(2),
            Input::StyleParam("coulomb".into()),
            Input::StyleParam("dielectric".into()),
            Input::StyleParam("delta".into()),
        ],
        "only what the expression names, charges before style params"
    );
    let mut values = table(&[("coulomb", cc), ("dielectric", diel), ("delta", delta)]);
    values.insert("q1".into(), vec![q1]);
    values.insert("q2".into(), vec![q2]);
    let cols = columns(&c, &values);
    let qs = [1.5, 3.0, 9.0];
    let (mut e, mut d) = (vec![0.0; 3], vec![0.0; 3]);
    c.eval_scalar(&qs, &cols, &mut e, &mut d);
    for (t, &r) in qs.iter().enumerate() {
        let want = cc * q1 * q2 / (diel * (r + delta));
        let dwant = -cc * q1 * q2 / (diel * (r + delta).powi(2));
        assert!(close(e[t], want, 1e-12), "{} vs {want}", e[t]);
        assert!(close(d[t], dwant, 1e-12), "{} vs {dwant}", d[t]);
    }
}

#[test]
fn fene_is_the_analytic_formula() {
    let (k, r0, eps, sig) = (30.0, 1.5, 1.0, 1.0);
    let rc = 2f64.powf(1.0 / 6.0) * sig;
    let hand = |r: F| {
        let mut e = -0.5 * k * r0 * r0 * (1.0 - (r / r0).powi(2)).ln();
        if r <= rc {
            e += 4.0 * eps * ((sig / r).powi(12) - (sig / r).powi(6)) + eps;
        }
        e
    };
    let p = [("k", k), ("r0", r0), ("epsilon", eps), ("sigma", sig)];
    check_scalar(
        Geometry::Bond,
        &p,
        &[],
        FENE,
        &[0.85, 0.97, 1.05, 1.3, 1.4],
        hand,
    );
    // The analytic force, not only a difference quotient.
    let c = compile(FENE, &Binding::new(Geometry::Bond, &names(&p), &[])).unwrap();
    for r in [0.9, 1.05, 1.3] {
        let (_, d) = c.eval_scalar_one(r, &[k, r0, eps, sig]);
        let mut want = k * r / (1.0 - (r / r0).powi(2));
        if r <= rc {
            want += 4.0 * eps * (-12.0 * sig.powi(12) / r.powi(13) + 6.0 * sig.powi(6) / r.powi(7));
        }
        assert!(close(d, want, 1e-12), "dE/dr {d} vs analytic {want} at {r}");
    }
}

// ---------------------------------------------------------------------------
// batch, pair binding, sub-definitions, functions
// ---------------------------------------------------------------------------

#[test]
fn batch_columns_and_broadcast() {
    let c = compile(
        BOND_HARMONIC,
        &Binding::new(Geometry::Bond, &["k", "r0"], &[]),
    )
    .unwrap();
    assert_eq!(
        c.inputs(),
        &[Input::Param("k".into()), Input::Param("r0".into())]
    );
    let k = [100.0, 200.0, 300.0];
    let r0 = [1.5]; // one value broadcast to every term
    let q = [1.4, 1.6, 1.9];
    let (mut e, mut d) = ([0.0; 3], [0.0; 3]);
    c.eval_scalar(&q, &[&k[..], &r0[..]], &mut e, &mut d);
    for t in 0..3 {
        let (e1, d1) = c.eval_scalar_one(q[t], &[k[t], r0[0]]);
        assert_eq!((e[t], d[t]), (e1, d1));
        assert!(close(e[t], k[t] * (q[t] - 1.5).powi(2), 1e-14));
        assert!(close(d[t], 2.0 * k[t] * (q[t] - 1.5), 1e-14));
    }
    // A missing column is a named error.
    let err = c
        .gather(|i| (i.spelling() == "k").then_some(&k[..]))
        .unwrap_err();
    assert_eq!(err, ExpressionError::MissingInput { input: "r0".into() });
}

fn self_param(name: &str, atom: u8) -> Input {
    Input::SelfParam {
        name: name.into(),
        atom,
    }
}

#[test]
fn pair_self_rows_bind_x1_x2_and_bare_names_the_pair_value() {
    // Lorentz–Berthelot written out over the self rows.
    let src = "4*sqrt(epsilon1*epsilon2)*((0.5*(sigma1+sigma2)/r)^12-(0.5*(sigma1+sigma2)/r)^6)";
    let b = Binding::new(Geometry::Pair, &["epsilon", "sigma"], &[]);
    let c = compile(src, &b).unwrap();
    assert_eq!(
        c.inputs(),
        &[
            self_param("epsilon", 1),
            self_param("sigma", 1),
            self_param("epsilon", 2),
            self_param("sigma", 2),
        ]
    );
    let (e1, s1, e2, s2) = (0.2, 3.1, 0.07, 2.5);
    let values = table(&[
        ("epsilon1", e1),
        ("sigma1", s1),
        ("epsilon2", e2),
        ("sigma2", s2),
    ]);
    let cols = columns(&c, &values);
    let (eps, sg): (F, F) = ((e1 * e2).sqrt(), 0.5 * (s1 + s2));
    let mixed = compile(PAIR_LJ_12_6, &b).unwrap();
    for r in [2.6, 3.0, 4.4] {
        let (mut e, mut d) = ([0.0], [0.0]);
        c.eval_scalar(&[r], &cols, &mut e, &mut d);
        let (we, wd) = mixed.eval_scalar_one(r, &[eps, sg]);
        assert!(close(e[0], we, 1e-12) && close(d[0], wd, 1e-12));
    }
    // A bare name and a self row in one expression: the pair value first.
    let c = compile("epsilon*r + epsilon2", &b).unwrap();
    assert_eq!(
        c.inputs(),
        &[Input::Param("epsilon".into()), self_param("epsilon", 2)]
    );
    // Outside a pair, `epsilon1` is no self row.
    let err = compile(
        "epsilon1*r",
        &Binding::new(Geometry::Bond, &["epsilon"], &[]),
    )
    .unwrap_err();
    assert!(
        matches!(err, ExpressionError::UndeclaredVariable { ref name, .. } if name == "epsilon1")
    );
}

/// The pair-symmetry check the protocol asks for (D6, 1e-12): evaluate on
/// the columns of the exchanged inputs and compare.
fn symmetric(c: &Compiled, values: &HashMap<String, Vec<F>>, r: F) -> bool {
    let cols = columns(c, values);
    let swapped = c
        .gather(|i| values.get(&i.swapped().spelling()).map(Vec::as_slice))
        .unwrap();
    let (mut a, mut b) = ([0.0], [0.0]);
    c.eval_scalar(&[r], &cols, &mut a, &mut [0.0]);
    c.eval_scalar(&[r], &swapped, &mut b, &mut [0.0]);
    close(a[0], b[0], 1e-12)
}

#[test]
fn pair_symmetry_through_swapped_inputs() {
    let b = Binding::new(Geometry::Pair, &["epsilon", "sigma"], &[]);
    let values = table(&[
        ("epsilon1", 0.2),
        ("epsilon2", 0.07),
        ("sigma1", 3.1),
        ("sigma2", 2.5),
        ("q1", 0.4),
        ("q2", -0.3),
    ]);
    let sym = compile("sqrt(epsilon1*epsilon2)*(sigma1+sigma2)/r + q1*q2/r", &b).unwrap();
    assert!(symmetric(&sym, &values, 3.0));
    let asym = compile("epsilon1*sigma2/r + q1", &b).unwrap();
    assert!(!symmetric(&asym, &values, 3.0));
    assert_eq!(self_param("sigma", 1).swapped(), self_param("sigma", 2));
    assert_eq!(Input::Charge(2).swapped(), Input::Charge(1));
    assert_eq!(
        Input::Param("epsilon".into()).swapped(),
        Input::Param("epsilon".into())
    );
}

#[test]
fn definitions_follow_lepton_order() {
    let b = Binding::new(Geometry::Bond, &["k", "r0"], &[]);
    let a = compile("k*dr^2; dr=r-r0", &b).unwrap();
    // Each definition uses only the ones to its right.
    let c = compile("k*x; x=dr*dr; dr=r-r0", &b).unwrap();
    let h = compile(BOND_HARMONIC, &b).unwrap();
    for r in [1.0, 1.7] {
        let want = h.eval_scalar_one(r, &[3.0, 1.2]);
        for e in [&a, &c] {
            let got = e.eval_scalar_one(r, &[3.0, 1.2]);
            assert!(close(got.0, want.0, 1e-14) && close(got.1, want.1, 1e-14));
        }
    }
    // One to its left is refused, naming both.
    assert_eq!(
        compile("k*x; dr=r-r0; x=dr*dr", &b).unwrap_err(),
        ExpressionError::DefinitionOrder {
            name: "dr".into(),
            used_in: "x".into()
        }
    );
    // An unused definition names no input...
    let u = compile("k*r; unused=r0*2", &b).unwrap();
    assert_eq!(u.inputs(), &[Input::Param("k".into())]);
    // ...but is still checked.
    let err = compile("k*r; unused=nope", &b).unwrap_err();
    assert!(matches!(err, ExpressionError::UndeclaredVariable { ref name, .. } if name == "nope"));
    // A definition that is a constant folds (and feeds an exponent).
    let f = compile("r^p; p=1/6", &Binding::new(Geometry::Bond, &[], &[])).unwrap();
    let (e, d) = f.eval_scalar_one(2.0, &[]);
    assert!(close(e, 2f64.powf(1.0 / 6.0), 1e-15));
    assert!(close(d, 2f64.powf(1.0 / 6.0 - 1.0) / 6.0, 1e-15));
    // A definition used twice, directly and through another.
    let g = compile("a+b; a=b*b; b=r+c; c=2*r", &b).unwrap();
    let (e, d) = g.eval_scalar_one(1.5, &[]);
    assert!(close(e, 4.5f64.powi(2) + 4.5, 1e-15) && close(d, 2.0 * 4.5 * 3.0 + 3.0, 1e-15));
    // The same in the compound program (a bond has points).
    let (mut ec, mut gc) = ([0.0], [[0.0; 3]; 2]);
    g.eval_compound(&[[0.0; 3], [1.5, 0.0, 0.0]], &[], &mut ec, &mut gc);
    assert!(close(ec[0], e, 1e-15) && close(gc[1][0], d, 1e-15) && close(gc[0][0], -d, 1e-15));
}

#[test]
fn every_function_value_and_derivative() {
    let b = Binding::new(Geometry::Bond, &[], &[]);
    type Case = (&'static str, fn(F) -> F);
    let cases: [Case; 18] = [
        ("exp(0.3*r)", |r| (0.3 * r).exp()),
        ("log(r)", |r| r.ln()),
        ("sqrt(r)", |r| r.sqrt()),
        ("sin(r)", |r| r.sin()),
        ("cos(r)", |r| r.cos()),
        ("tan(r)", |r| r.tan()),
        ("asin(r-0.5)", |r| (r - 0.5).asin()),
        ("acos(r-0.5)", |r| (r - 0.5).acos()),
        ("atan(r)", |r| r.atan()),
        ("abs(r-0.9)", |r| (r - 0.9).abs()),
        ("min(r, 2-r)", |r| r.min(2.0 - r)),
        ("max(r^2, 0.5)", |r| (r * r).max(0.5)),
        ("step(r-0.7)*r", |r| if r >= 0.7 { r } else { 0.0 }),
        ("1+delta(r-0.6)", |r| if r == 0.6 { 2.0 } else { 1.0 }),
        ("select(step(r-0.8), r^3, -r)", |r| {
            if r >= 0.8 { r.powi(3) } else { -r }
        }),
        ("r^r", |r| r.powf(r)),
        ("2^-r", |r| 2f64.powf(-r)),
        ("-r^2", |r| -(r * r)),
    ];
    for (src, f) in cases {
        let c = compile(src, &b).unwrap_or_else(|e| panic!("{src}: {e}"));
        for r in [0.6, 0.75, 1.1] {
            let (e, d) = c.eval_scalar_one(r, &[]);
            assert!(close(e, f(r), 1e-14), "{src} at {r}: {e} vs {}", f(r));
            let h = 1e-6;
            let fd = (f(r + h) - f(r - h)) / (2.0 * h);
            assert!(close(d, fd, 1e-7), "{src} at {r}: d {d} vs {fd}");
        }
    }
    // The non-smooth points: abs' = 0 at 0, min/max ties take the first.
    let at = |src: &str, r: F| compile(src, &b).unwrap().eval_scalar_one(r, &[]);
    assert_eq!(at("abs(r-1)", 1.0), (0.0, 0.0));
    assert_eq!(at("min(2*r, r+1)", 1.0), (2.0, 2.0));
    assert_eq!(at("max(r+1, 2*r)", 1.0), (2.0, 1.0));
    // Precedence: 2^3^2 = 2^9, -2^2 = -4, 2*-3+1 = -5.
    for (src, want) in [
        ("2^3^2", 512.0),
        ("-2^2", -4.0),
        ("2*-3+1", -5.0),
        ("10/2/5", 1.0),
        ("1-2-3", -4.0),
        ("-2*3^2", -18.0),
    ] {
        assert_eq!(at(src, 1.0).0, want, "{src}");
    }
}

// ---------------------------------------------------------------------------
// points: distance / angle / dihedral and their exact gradients
// ---------------------------------------------------------------------------

/// Gradient of a points expression against a central difference of its
/// own energy, coordinate by coordinate, to 1e-7.
fn check_gradient(c: &Compiled, x: &[[F; 3]], cols: &[&[F]]) -> (Vec<F>, Vec<[F; 3]>) {
    let arity = c.geometry().points();
    let n = x.len() / arity;
    let mut e = vec![0.0; n];
    let mut g = vec![[0.0; 3]; x.len()];
    c.eval_compound(x, cols, &mut e, &mut g);
    let h = 1e-6;
    for p in 0..x.len() {
        for a in 0..3 {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[p][a] += h;
            xm[p][a] -= h;
            let (mut ep, mut em) = (vec![0.0; n], vec![0.0; n]);
            let mut gg = vec![[0.0; 3]; x.len()];
            c.eval_compound(&xp, cols, &mut ep, &mut gg);
            c.eval_compound(&xm, cols, &mut em, &mut gg);
            let t = p / arity;
            let fd = (ep[t] - em[t]) / (2.0 * h);
            assert!(
                close(g[p][a], fd, 1e-7),
                "{}: d/dx[{p}][{a}] {} vs {fd}",
                c.source(),
                g[p][a]
            );
        }
    }
    (e, g)
}

const QUAD: [[F; 3]; 4] = [
    [0.1, 1.0, 0.2],
    [0.0, 0.0, 0.0],
    [1.0, 0.0, -0.1],
    [1.2, -0.8, 0.5],
];

fn flat(x: &[[F; 3]]) -> Vec<F> {
    x.iter().flatten().copied().collect()
}

fn dist(a: [F; 3], b: [F; 3]) -> F {
    (0..3).map(|i| (b[i] - a[i]).powi(2)).sum::<F>().sqrt()
}

#[test]
fn compound_functions_are_molrs_geometry_with_exact_gradients() {
    let none: [&[F]; 0] = [];
    let quad = Binding::new(Geometry::Compound { arity: 4 }, &[], &[]);
    // Two terms in one batch: the quad and a second, distorted one.
    let mut x2 = QUAD.to_vec();
    x2.extend(QUAD.iter().map(|p| [p[0], -p[1], 0.3 * p[2]]));
    let flat2 = flat(&x2[4..]);

    let d = compile("dihedral(p1,p2,p3,p4)", &quad).unwrap();
    assert!(!d.has_scalar_form());
    let (e, _) = check_gradient(&d, &x2, &none);
    assert!(close(
        e[0],
        compute_dihedral(&flat(&QUAD), 0, 1, 2, 3),
        1e-14
    ));
    assert!(close(e[1], compute_dihedral(&flat2, 0, 1, 2, 3), 1e-14));

    let a = compile("angle(p1,p2,p3)", &quad).unwrap();
    let (e, _) = check_gradient(&a, &x2, &none);
    assert!(close(e[0], compute_angle(&flat(&QUAD), 0, 1, 2), 1e-14));

    let r = compile("distance(p1,p3)", &quad).unwrap();
    let (e, _) = check_gradient(&r, &x2, &none);
    let r13 = dist(QUAD[0], QUAD[2]);
    assert!(close(e[0], r13, 1e-15));

    // A custom three-body Urey–Bradley category, per-term columns.
    let ub = compile(
        "k_ub*(distance(p1,p3)-r_ub)^2",
        &Binding::new(Geometry::Compound { arity: 3 }, &["k_ub", "r_ub"], &[]),
    )
    .unwrap();
    let x3 = [QUAD[0], QUAD[1], QUAD[2], QUAD[1], QUAD[2], QUAD[3]];
    let (kub, rub) = ([5.0, 7.5], [1.2]);
    let (e, _) = check_gradient(&ub, &x3, &[&kub[..], &rub[..]]);
    assert!(close(e[0], 5.0 * (r13 - 1.2).powi(2), 1e-14));
    assert!(close(
        e[1],
        7.5 * (dist(QUAD[1], QUAD[3]) - 1.2).powi(2),
        1e-14
    ));
}

#[test]
fn points_beyond_the_coordinate_make_a_compound_form() {
    // `angle charmm` written out: `theta` and `distance(p1,p3)` in one
    // expression of the angle category (D5).
    let p = [("k", 30.0), ("theta0", 100.0), ("k_ub", 5.0), ("r_ub", 1.2)];
    let c = compile(
        ANGLE_CHARMM,
        &Binding::new(Geometry::Angle, &names(&p), &[]),
    )
    .unwrap();
    assert!(!c.has_scalar_form(), "a point function needs the points");
    let values = table(&p);
    let (e, _) = check_gradient(&c, &QUAD[..3], &columns(&c, &values));
    let th = compute_angle(&flat(&QUAD), 0, 1, 2);
    let want =
        30.0 * (th - 100f64.to_radians()).powi(2) + 5.0 * (dist(QUAD[0], QUAD[2]) - 1.2).powi(2);
    assert!(close(e[0], want, 1e-13), "{} vs {want}", e[0]);

    // A coordinate-only expression has both forms, and they agree.
    let h = compile(
        ANGLE_HARMONIC,
        &Binding::new(Geometry::Angle, &["k", "theta0"], &[]),
    )
    .unwrap();
    assert!(h.has_scalar_form());
    let (es, _) = h.eval_scalar_one(th, &[30.0, 100.0]);
    let (ec, _) = check_gradient(&h, &QUAD[..3], &[&[30.0][..], &[100.0][..]]);
    assert!(close(es, ec[0], 1e-14));

    // A pair has no points.
    let err = compile("distance(p1,p2)", &Binding::new(Geometry::Pair, &[], &[])).unwrap_err();
    assert_eq!(
        err,
        ExpressionError::NotAPoint {
            function: "distance".into(),
            found: "p1".into(),
            points: 0
        }
    );
}

#[test]
fn degenerate_geometry_has_a_zero_gradient() {
    let none: [&[F]; 0] = [];
    let line = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ];
    for (src, value) in [
        ("angle(p1,p2,p3)", std::f64::consts::PI),
        ("dihedral(p1,p2,p3,p4)", 0.0),
        ("distance(p1,p1)", 0.0),
    ] {
        let c = compile(
            src,
            &Binding::new(Geometry::Compound { arity: 4 }, &[], &[]),
        )
        .unwrap();
        let (mut e, mut g) = ([0.0], [[1.0; 3]; 4]);
        c.eval_compound(&line, &none, &mut e, &mut g);
        assert_eq!(e[0], value, "{src}");
        assert!(g.iter().flatten().all(|&v| v == 0.0), "{src}: {g:?}");
    }
}

// ---------------------------------------------------------------------------
// agreement with the native kernels (the built-in gate, bonded categories)
// ---------------------------------------------------------------------------

const GEOMETRIES: [[[F; 3]; 4]; 2] = [
    QUAD,
    [
        [0.3, 0.9, -0.4],
        [0.0, 0.1, 0.0],
        [1.1, 0.0, 0.2],
        [1.4, 0.7, 0.9],
    ],
];

/// Price one term of `category`/`style` with the native kernel (through
/// `PotentialCompiler`) and with the style's expression over the points,
/// and compare energy (rel 1e-10, D14) and forces; where the expression has
/// a scalar form, its energy at the measured coordinate too.
fn agrees_with_native(category: &str, style: &str, params: &[(&str, F)], expression: &str) {
    let geometry = Geometry::of_category(category).unwrap();
    let arity = geometry.points();
    let mut ff = ForceField::new("t");
    let ends = ["a", "b", "c", "d"];
    ff.def_style(category, style, Params::new())
        .unwrap()
        .def_type(
            "t",
            &ends[..arity],
            Params::from_iter(params.iter().copied()),
        )
        .unwrap();
    let mut block = Block::new();
    for (key, atom) in ["atomi", "atomj", "atomk", "atoml"]
        .into_iter()
        .zip(0..arity)
    {
        block
            .insert(key, Array1::from_vec(vec![atom as Idx]).into_dyn(), None)
            .unwrap();
    }
    block
        .insert(
            "type",
            Array1::from_vec(vec!["t".to_owned()]).into_dyn(),
            None,
        )
        .unwrap();
    let mut frame = Frame::new();
    frame.insert(format!("{category}s"), block);
    let pots = PotentialCompiler::new(&ff)
        .compile(&frame)
        .unwrap_or_else(|e| panic!("{category} {style}: {e}"));

    let c = compile(expression, &Binding::new(geometry, &names(params), &[]))
        .unwrap_or_else(|e| panic!("{category} {style}: {e}"));
    let values = table(params);
    let cols = columns(&c, &values);

    for g in GEOMETRIES {
        let coords = flat(&g);
        let (e_nat, f_nat) = pots.energy_forces_at(&coords);
        let (mut e, mut grad) = ([0.0], vec![[0.0; 3]; arity]);
        c.eval_compound(&g[..arity], &cols, &mut e, &mut grad);
        assert!(
            close(e[0], e_nat, 1e-10),
            "{category} {style}: expression {} vs native {e_nat}",
            e[0]
        );
        let scale = f_nat.iter().fold(1.0 as F, |m, v| m.max(v.abs()));
        for p in 0..arity {
            for a in 0..3 {
                let (got, want) = (-grad[p][a], f_nat[3 * p + a]);
                assert!(
                    (got - want).abs() <= 1e-9 * scale,
                    "{category} {style}: F[{p}][{a}] expression {got} vs native {want}"
                );
            }
        }
        if c.has_scalar_form() {
            let q = match arity {
                2 => dist(g[0], g[1]),
                3 => compute_angle(&coords, 0, 1, 2),
                _ => compute_dihedral(&coords, 0, 1, 2, 3),
            };
            let one: Vec<F> = cols.iter().map(|c| c[0]).collect();
            let (es, _) = c.eval_scalar_one(q, &one);
            assert!(
                close(es, e_nat, 1e-10),
                "{category} {style}: scalar form {es} vs native {e_nat}"
            );
        }
    }
}

#[test]
fn builtin_expressions_agree_with_native_kernels() {
    agrees_with_native(
        "bond",
        "harmonic",
        &[("k", 150.0), ("r0", 1.2)],
        BOND_HARMONIC,
    );
    agrees_with_native(
        "bond",
        "morse",
        &[("d0", 4.0), ("alpha", 1.5), ("r0", 1.1)],
        BOND_MORSE,
    );
    agrees_with_native(
        "bond",
        "class2",
        &[("r0", 1.2), ("k2", 300.0), ("k3", -400.0), ("k4", 500.0)],
        BOND_CLASS2,
    );
    agrees_with_native(
        "angle",
        "harmonic",
        &[("k", 55.0), ("theta0", 104.5)],
        ANGLE_HARMONIC,
    );
    agrees_with_native(
        "angle",
        "charmm",
        &[
            ("k", 55.0),
            ("theta0", 104.5),
            ("k_ub", 20.0),
            ("r_ub", 1.6),
        ],
        ANGLE_CHARMM,
    );
    agrees_with_native(
        "angle",
        "class2",
        &[("theta0", 112.0), ("k2", 40.0), ("k3", -20.0), ("k4", 10.0)],
        ANGLE_CLASS2,
    );
    agrees_with_native(
        "dihedral",
        "periodic",
        &[("k", 1.4), ("periodicity", 3.0), ("phase", 20.0)],
        DIHEDRAL_PERIODIC,
    );
    agrees_with_native(
        "dihedral",
        "periodic",
        &[
            ("k1", 0.7),
            ("periodicity1", 1.0),
            ("phase1", 0.0),
            ("k2", 0.25),
            ("periodicity2", 2.0),
            ("phase2", 180.0),
        ],
        DIHEDRAL_PERIODIC_2,
    );
    agrees_with_native(
        "dihedral",
        "charmm",
        &[
            ("k", 0.9),
            ("periodicity", 2.0),
            ("phase", 45.0),
            ("w", 0.0),
        ],
        DIHEDRAL_PERIODIC,
    );
    agrees_with_native(
        "dihedral",
        "opls",
        &[("k1", 1.1), ("k2", -0.4), ("k3", 0.9), ("k4", 0.2)],
        DIHEDRAL_OPLS,
    );
    agrees_with_native(
        "dihedral",
        "harmonic",
        &[("k", 2.0), ("sign", -1.0), ("periodicity", 3.0)],
        DIHEDRAL_HARMONIC,
    );
    agrees_with_native(
        "dihedral",
        "multi/harmonic",
        &[
            ("a1", 0.3),
            ("a2", -1.2),
            ("a3", 0.8),
            ("a4", 0.45),
            ("a5", -0.6),
        ],
        DIHEDRAL_MULTI_HARMONIC,
    );
    agrees_with_native(
        "dihedral",
        "nharmonic",
        &[
            ("a1", 1.0),
            ("a2", -2.0),
            ("a3", 3.0),
            ("a4", -4.0),
            ("a5", 5.0),
            ("a6", -6.0),
            ("a7", 7.0),
        ],
        DIHEDRAL_NHARMONIC_7,
    );
    agrees_with_native(
        "dihedral",
        "class2",
        &[
            ("k1", 0.5),
            ("phi1", 10.0),
            ("k2", -0.3),
            ("phi2", 170.0),
            ("k3", 0.2),
            ("phi3", 45.0),
        ],
        DIHEDRAL_CLASS2,
    );
    agrees_with_native(
        "improper",
        "harmonic",
        &[("k", 40.0), ("chi0", 12.0)],
        IMPROPER_HARMONIC,
    );
    agrees_with_native(
        "improper",
        "cvff",
        &[("k", 2.5), ("sign", -1.0), ("periodicity", 2.0)],
        IMPROPER_CVFF,
    );
    agrees_with_native(
        "improper",
        "periodic",
        &[("k", 1.1), ("periodicity", 2.0), ("phase", 30.0)],
        IMPROPER_PERIODIC,
    );
}

// ---------------------------------------------------------------------------
// named errors
// ---------------------------------------------------------------------------

#[test]
fn parse_errors_are_named() {
    use ExpressionError::*;
    const OPERAND: &str = "a number, a name, `(` or `-`";
    const AFTER_SEMI: &str = "a definition `name=formula` after `;`";
    let token = |pos, found: &str, expected| UnexpectedToken {
        pos,
        found: found.into(),
        expected,
    };
    let cases: Vec<(&str, ExpressionError)> = vec![
        ("", EmptyExpression),
        ("  ; a=1", EmptyExpression),
        ("k*(r-r0", UnexpectedEnd { expected: "`)`" }),
        ("k*#r", UnexpectedChar { pos: 2, ch: '#' }),
        ("k r", token(2, "r", "an operator or the end")),
        ("k*)", token(2, ")", OPERAND)),
        ("+r", token(0, "+", OPERAND)),
        ("a=r", token(1, "=", "an operator or the end")),
        ("min(r,", UnexpectedEnd { expected: OPERAND }),
        // `;` splits first: the energy `min(r` ends inside the call.
        (
            "min(r;1)",
            UnexpectedEnd {
                expected: "`,` or `)`",
            },
        ),
        (
            "r; 3=4",
            BadDefinition {
                pos: 3,
                text: "3=".into(),
            },
        ),
        (
            "r; a",
            BadDefinition {
                pos: 3,
                text: "a".into(),
            },
        ),
        (
            "r; a=",
            UnexpectedEnd {
                expected: "the definition's expression",
            },
        ),
        (
            "r;",
            UnexpectedEnd {
                expected: AFTER_SEMI,
            },
        ),
        (
            "r; a=1;; b=2",
            UnexpectedEnd {
                expected: AFTER_SEMI,
            },
        ),
        ("r*.", UnexpectedChar { pos: 2, ch: '.' }),
    ];
    for (src, want) in cases {
        match parse(src) {
            Err(e) if e == want => {}
            other => panic!("{src:?}: {other:?}, expected {want:?}"),
        }
    }
    // Every error's message names its subject.
    let msg = parse("k*#r").unwrap_err().to_string();
    assert!(msg.contains('#') && msg.contains("byte 2"), "{msg}");
}

#[test]
fn compile_errors_are_named() {
    use ExpressionError::*;
    let bond = Binding::new(Geometry::Bond, &["k", "r0"], &[]);
    let quad = Binding::new(Geometry::Compound { arity: 4 }, &["k"], &[]);
    let err = |src: &str, b: &Binding| compile(src, b).unwrap_err();
    let arity = |name: &str, expected, found| FunctionArity {
        name: name.into(),
        expected,
        found,
    };
    let not_a_point = |function: &str, found: &str, points| NotAPoint {
        function: function.into(),
        found: found.into(),
        points,
    };

    assert_eq!(
        err("k*foo(r)", &bond),
        UnknownFunction { name: "foo".into() }
    );
    // A Lepton function outside molrec's subset is unknown here.
    assert_eq!(err("erf(r)", &bond), UnknownFunction { name: "erf".into() });
    assert_eq!(err("min(r)", &bond), arity("min", 2, 1));
    assert_eq!(err("exp()", &bond), arity("exp", 1, 0));
    assert_eq!(err("k; a=select(r,1)", &bond), arity("select", 3, 2));
    match err("k*(r-x0)^2", &bond) {
        UndeclaredVariable { name, allowed } => {
            assert_eq!(name, "x0");
            assert_eq!(allowed, vec!["r", "k", "r0"]);
        }
        e => panic!("{e:?}"),
    }
    // Another category's variable, a named constant, a one-body coordinate
    // (an open item of the protocol) are no variables.
    for (src, name, b) in [
        ("k*(theta-r0)^2", "theta", &bond),
        ("k*pi", "pi", &bond),
        ("x1", "x1", &quad),
    ] {
        assert!(
            matches!(err(src, b), UndeclaredVariable { name: ref n, .. } if n == name),
            "{src}"
        );
    }
    assert_eq!(
        err("a; a=b+1; b=2*a", &bond),
        CyclicDefinition {
            cycle: vec!["a".into(), "b".into(), "a".into()]
        }
    );
    assert_eq!(
        err("k; a=a+r", &bond),
        CyclicDefinition {
            cycle: vec!["a".into(), "a".into()]
        }
    );
    assert_eq!(
        err("a; a=r; a=2*r", &bond),
        DuplicateDefinition { name: "a".into() }
    );
    let pair = Binding::new(Geometry::Pair, &["epsilon"], &[]);
    for (shadow, b) in [
        ("r0", &bond),
        ("r", &bond),
        ("p1", &bond),
        ("epsilon2", &pair),
        ("q1", &pair),
    ] {
        assert_eq!(
            err(&format!("1; {shadow}=1"), b),
            DefinitionShadows {
                name: shadow.into()
            }
        );
    }
    assert_eq!(
        err("distance(p1,k)", &quad),
        not_a_point("distance", "k", 4)
    );
    assert_eq!(
        err("distance(p1,p3)", &bond),
        not_a_point("distance", "p3", 2)
    );
    assert_eq!(err("angle(p1,p2,p5)", &quad), not_a_point("angle", "p5", 4));
    assert_eq!(
        err("dihedral(p1,p2,p3,p4+1)", &quad),
        not_a_point("dihedral", "p4+1", 4)
    );
    assert_eq!(err("k*p1", &quad), PointAsNumber { name: "p1".into() });

    for bad in [
        Binding::new(Geometry::Bond, &["k", "k"], &[]),
        Binding::new(Geometry::Bond, &["k"], &["k"]),
        Binding::new(Geometry::Bond, &["r"], &[]),
        Binding::new(Geometry::Bond, &["theta"], &[]),
        Binding::new(Geometry::Bond, &["p5"], &[]),
        Binding::new(Geometry::Bond, &["1k"], &[]),
        Binding::new(Geometry::Pair, &["q"], &[]),
        Binding::new(Geometry::Pair, &["q2"], &[]),
        Binding::new(Geometry::Pair, &["sigma", "sigma1"], &[]),
        Binding::new(Geometry::Pair, &["sigma"], &["sigma2"]),
        Binding::new(Geometry::Compound { arity: 1 }, &[], &[]),
        Binding::new(Geometry::Compound { arity: 6 }, &[], &[]),
    ] {
        assert!(
            matches!(compile("1", &bad), Err(BadBinding { .. })),
            "{bad:?}"
        );
    }
    for category in ["urey_bradley", "constraint"] {
        assert!(matches!(
            compile_style(category, &[], &[], "1"),
            Err(BadBinding { .. })
        ));
    }
    assert!(compile_style("cmap", &["k"], &[], "k*dihedral(p2,p3,p4,p5)").is_ok());
    // A parameter may be named like a function: a call has its `(`.
    let d = compile_style("pair", &[], &["delta"], "delta*delta(r-1)").unwrap();
    assert_eq!(d.eval_scalar_one(1.0, &[3.0]).0, 3.0);
    assert_eq!(d.eval_scalar_one(2.0, &[3.0]).0, 0.0);
}

#[test]
fn named_columns_and_variables() {
    let b = Binding::new(Geometry::Pair, &["epsilon", "sigma"], &["coulomb"]);
    let c = compile(
        "4*epsilon*((sigma/r)^12-(sigma/r)^6) + coulomb*q1*q2/r + 0*sigma2 + s; s=r",
        &b,
    )
    .unwrap();
    assert_eq!(
        c.variables(),
        ["epsilon", "sigma", "r", "coulomb", "q1", "q2", "sigma2"]
    );
    assert_eq!(c.pair_inputs(), ["sigma2", "q1", "q2"]);
    let cols: HashMap<&str, Vec<F>> = [
        ("epsilon", vec![0.2, 0.3]),
        ("sigma", vec![3.0, 3.2]),
        ("sigma2", vec![9.9, 9.9]),
        ("q1", vec![0.5, -0.5]),
        ("q2", vec![0.25, 0.25]),
        ("coulomb", vec![332.0]),
    ]
    .into_iter()
    .collect();
    let (mut e, mut d) = ([0.0; 2], [0.0; 2]);
    c.eval_scalar_named(
        &[3.5, 4.0],
        |n| cols.get(n).map(Vec::as_slice),
        &mut e,
        &mut d,
    )
    .unwrap();
    for t in 0..2 {
        let r = [3.5, 4.0][t];
        let (eps, sg, qq) = (cols["epsilon"][t], cols["sigma"][t], cols["q1"][t] * 0.25);
        let want = 4.0 * eps * ((sg / r).powi(12) - (sg / r).powi(6)) + 332.0 * qq / r + r;
        assert!(close(e[t], want, 1e-12), "{} vs {want}", e[t]);
    }
    let err = c
        .eval_scalar_named(
            &[3.5],
            |n| (n != "q2").then(|| &cols[n][..1]),
            &mut e[..1],
            &mut d[..1],
        )
        .unwrap_err();
    assert_eq!(err, ExpressionError::MissingInput { input: "q2".into() });

    // Points are no variables; the geometric variable is.
    let u = compile(
        ANGLE_CHARMM,
        &Binding::new(Geometry::Angle, &["k", "theta0", "k_ub", "r_ub"], &[]),
    )
    .unwrap();
    assert_eq!(u.variables(), ["k", "theta", "theta0", "k_ub", "r_ub"]);
    assert!(u.pair_inputs().is_empty());
    let v: HashMap<&str, Vec<F>> = [("k", 1.0), ("theta0", 90.0), ("k_ub", 1.0), ("r_ub", 1.0)]
        .into_iter()
        .map(|(k, v)| (k, vec![v]))
        .collect();
    let mut g = vec![[0.0; 3]; 3];
    u.eval_compound_named(
        &QUAD[..3],
        |n| v.get(n).map(Vec::as_slice),
        &mut [0.0],
        &mut g,
    )
    .unwrap();
    assert!(g.iter().flatten().any(|&x| x != 0.0));
}

// ---------------------------------------------------------------------------
// printer
// ---------------------------------------------------------------------------

#[test]
fn printer_round_trips() {
    let corpus = [
        BOND_HARMONIC,
        BOND_MORSE,
        BOND_CLASS2,
        ANGLE_HARMONIC,
        ANGLE_CHARMM,
        ANGLE_CLASS2,
        DIHEDRAL_PERIODIC,
        DIHEDRAL_PERIODIC_2,
        DIHEDRAL_OPLS,
        DIHEDRAL_HARMONIC,
        DIHEDRAL_MULTI_HARMONIC,
        DIHEDRAL_CLASS2,
        DIHEDRAL_RB,
        IMPROPER_HARMONIC,
        IMPROPER_CVFF,
        PAIR_LJ_CUT,
        PAIR_LJ_CHARMM,
        PAIR_COUL_CHARMM,
        PAIR_COUL_CUT,
        PAIR_LJ_CLASS2,
        PAIR_BUCK,
        PAIR_MORSE,
        FENE,
        "-x^2 - -y*-z + (-x)^2 + x^-2 + x^(-y)^z + (x^y)^z + x-(y-z) + x/(y*z) + x/y*z",
        "2^3^2 + 1e-7 + 1.5E+20 + .25 + 3. + 3.141592653589793",
        "select(step(r-rc), 0, min(a, max(b, c))); rc=2^(1/6)*s; a=abs(r-1)",
        "k*(angle(p1,p2,p3)-t0)^2 + dihedral(p1, p2, p3, p4)",
        "  k * ( r - r0 ) ^ 2  ",
    ];
    for src in corpus {
        let p = parse(src).unwrap();
        assert_eq!(p.source(), src, "the source is byte for byte");
        let printed = p.to_string();
        let back = parse(&printed).unwrap_or_else(|e| panic!("{printed:?}: {e}"));
        assert!(
            p.same_tree(&back),
            "{src:?} printed {printed:?} parses to another tree"
        );
        assert_eq!(back.to_string(), printed, "the printer is a fixed point");
    }
    assert_eq!(parse(BOND_HARMONIC).unwrap().to_string(), "k*(r-r0)^2");
    assert_eq!(parse("(a+b)+c").unwrap().to_string(), "a+b+c");
    assert_eq!(parse("a+(b+c)").unwrap().to_string(), "a+(b+c)");
}

#[test]
fn substitution_rewrites_for_an_engine() {
    // OpenMM's nm against the IR's Å: r → 10*r.
    let p = parse(BOND_HARMONIC).unwrap();
    let q = p.substitute("r", &Expr::bin(BinOp::Mul, Expr::num(10.0), Expr::var("r")));
    assert_eq!(q.to_string(), "k*(10*r-r0)^2");
    assert_eq!(q.source(), "k*(10*r-r0)^2");
    let b = Binding::new(Geometry::Bond, &["k", "r0"], &[]);
    let (x, y) = (
        compile_parsed(p, &b).unwrap(),
        compile_parsed(q, &b).unwrap(),
    );
    let (e0, d0) = x.eval_scalar_one(1.37, &[100.0, 1.2]);
    let (e1, d1) = y.eval_scalar_one(0.137, &[100.0, 1.2]);
    assert!(close(e0, e1, 1e-13) && close(10.0 * d0, d1, 1e-13));

    // A coordinate rewritten into its compound function (OpenMM's
    // `chi` → `abs(theta)` is this) prices the same.
    let pts = Expr::call(
        "dihedral",
        (1..=4).map(|i| Expr::var(format!("p{i}"))).collect(),
    );
    let rewritten = parse(IMPROPER_HARMONIC)
        .unwrap()
        .substitute("chi", &Expr::call("abs", vec![pts]));
    let ib = Binding::new(Geometry::Improper, &["k", "chi0"], &[]);
    let direct = compile(IMPROPER_HARMONIC, &ib).unwrap();
    let via = compile_parsed(rewritten, &ib).unwrap();
    assert!(direct.has_scalar_form() && !via.has_scalar_form());
    let cols = [&[40.0][..], &[12.0][..]];
    let (mut a, mut b2) = ([0.0], [0.0]);
    let (mut ga, mut gb) = ([[0.0; 3]; 4], [[0.0; 3]; 4]);
    direct.eval_compound(&QUAD, &cols, &mut a, &mut ga);
    via.eval_compound(&QUAD, &cols, &mut b2, &mut gb);
    assert_eq!(a, b2);
    assert_eq!(ga, gb);

    // A built tree with a negative literal prints parseably.
    let t = Parsed::from_tree(
        Expr::bin(
            BinOp::Pow,
            Expr::var("r"),
            Expr::bin(BinOp::Mul, Expr::num(-2.0), Expr::var("r")),
        ),
        vec![Definition {
            name: "u".into(),
            expr: Expr::num(1e-30),
        }],
    );
    assert_eq!(t.to_string(), "r^(-2*r); u=1e-30");
    let back = parse(&t.to_string()).unwrap();
    assert_eq!(back.to_string(), t.to_string());
    let f = compile_parsed(back, &Binding::new(Geometry::Bond, &[], &[])).unwrap();
    assert!(close(
        f.eval_scalar_one(1.5, &[]).0,
        1.5f64.powf(-3.0),
        1e-14
    ));
}
