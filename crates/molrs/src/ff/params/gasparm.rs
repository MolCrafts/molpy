//! Gasteiger–Marsili PEOE parameters — `GASPARM.DAT`.
//!
//! DO NOT HAND-EDIT — regenerate with `scripts/gen_param_tables.py`, which emits
//! this table from AmberTools' own `.DAT` / `.DEF` files. That is where the table
//! came FROM; it is not what the table IS — this is ordinary source, not a build
//! artefact.
//!
//! Source: `$AMBERHOME/dat/antechamber/GASPARM.DAT` (AmberTools).

use crate::ff::params::GasteigerRow;

/// The 37 Gasteiger PEOE parameter rows of `GASPARM.DAT`.
#[rustfmt::skip]
pub const GASTEIGER_PARAMS: &[GasteigerRow] = &[
    GasteigerRow { atom_type: "h", a: 7.17, b: 6.24, c: -0.56, chi_plus: 20.02, seed_charge: 0.00 },
    GasteigerRow { atom_type: "c1", a: 10.39, b: 9.45, c: 0.73, chi_plus: 20.57, seed_charge: 0.00 },
    GasteigerRow { atom_type: "c2", a: 8.79, b: 9.32, c: 1.51, chi_plus: 19.62, seed_charge: 0.00 },
    GasteigerRow { atom_type: "c3", a: 7.98, b: 9.18, c: 1.88, chi_plus: 19.04, seed_charge: 0.00 },
    GasteigerRow { atom_type: "cg", a: 8.79, b: 9.32, c: 1.51, chi_plus: 19.62, seed_charge: 0.04 },
    GasteigerRow { atom_type: "n1", a: 15.68, b: 11.70, c: -0.27, chi_plus: 27.11, seed_charge: 0.00 },
    GasteigerRow { atom_type: "n2", a: 12.87, b: 11.15, c: 0.85, chi_plus: 24.87, seed_charge: 0.00 },
    GasteigerRow { atom_type: "n3", a: 11.54, b: 10.82, c: 1.36, chi_plus: 23.72, seed_charge: 0.00 },
    GasteigerRow { atom_type: "na", a: 12.32, b: 11.20, c: 1.34, chi_plus: 24.86, seed_charge: 0.00 },
    GasteigerRow { atom_type: "na+", a: 12.32, b: 11.20, c: 1.34, chi_plus: 24.86, seed_charge: 1.00 },
    GasteigerRow { atom_type: "ng", a: 12.32, b: 11.20, c: 1.34, chi_plus: 24.86, seed_charge: 0.32 },
    GasteigerRow { atom_type: "n4", a: 0.00, b: 11.86, c: 11.86, chi_plus: 23.72, seed_charge: 1.00 },
    GasteigerRow { atom_type: "o2", a: 17.07, b: 13.79, c: 0.47, chi_plus: 31.33, seed_charge: 0.00 },
    GasteigerRow { atom_type: "o3", a: 14.18, b: 12.92, c: 1.39, chi_plus: 28.49, seed_charge: 0.00 },
    GasteigerRow { atom_type: "o-1", a: 17.07, b: 13.79, c: 0.47, chi_plus: 31.33, seed_charge: -1.00 },
    GasteigerRow { atom_type: "o-2", a: 17.07, b: 13.79, c: 0.47, chi_plus: 31.33, seed_charge: -0.50 },
    GasteigerRow { atom_type: "os", a: 17.07, b: 13.79, c: 0.47, chi_plus: 31.33, seed_charge: -1.00 },
    GasteigerRow { atom_type: "op#", a: 17.07, b: 13.79, c: 0.47, chi_plus: 31.33, seed_charge: -1.00 },
    GasteigerRow { atom_type: "op", a: 17.07, b: 13.79, c: 0.47, chi_plus: 31.33, seed_charge: -0.50 },
    GasteigerRow { atom_type: "op=", a: 17.07, b: 13.79, c: 0.47, chi_plus: 31.33, seed_charge: -0.67 },
    GasteigerRow { atom_type: "s", a: 10.14, b: 9.13, c: 1.38, chi_plus: 20.65, seed_charge: 0.00 },
    GasteigerRow { atom_type: "s2", a: 10.88, b: 9.485, c: 1.325, chi_plus: 21.69, seed_charge: 0.00 },
    GasteigerRow { atom_type: "s-1", a: 10.88, b: 9.485, c: 1.325, chi_plus: 21.69, seed_charge: -1.00 },
    GasteigerRow { atom_type: "s3", a: 10.14, b: 9.13, c: 1.38, chi_plus: 20.65, seed_charge: 0.00 },
    GasteigerRow { atom_type: "so", a: 10.14, b: 9.13, c: 1.38, chi_plus: 20.65, seed_charge: 1.00 },
    GasteigerRow { atom_type: "so1", a: 12.00, b: 10.805, c: 1.195, chi_plus: 24.00, seed_charge: 1.00 },
    GasteigerRow { atom_type: "so2", a: 12.00, b: 10.805, c: 1.195, chi_plus: 24.00, seed_charge: 2.00 },
    GasteigerRow { atom_type: "so3", a: 12.00, b: 10.805, c: 1.195, chi_plus: 24.00, seed_charge: 3.00 },
    GasteigerRow { atom_type: "so4", a: 12.00, b: 10.805, c: 1.195, chi_plus: 24.00, seed_charge: 4.00 },
    GasteigerRow { atom_type: "f", a: 14.66, b: 13.85, c: 2.31, chi_plus: 30.82, seed_charge: 0.00 },
    GasteigerRow { atom_type: "cl", a: 11.00, b: 9.69, c: 1.35, chi_plus: 22.04, seed_charge: 0.00 },
    GasteigerRow { atom_type: "br", a: 10.08, b: 8.47, c: 1.16, chi_plus: 19.71, seed_charge: 0.00 },
    GasteigerRow { atom_type: "i", a: 9.90, b: 7.96, c: 0.96, chi_plus: 18.82, seed_charge: 0.00 },
    GasteigerRow { atom_type: "p#", a: 8.90, b: 8.24, c: 0.96, chi_plus: 18.10, seed_charge: 1.00 },
    GasteigerRow { atom_type: "p=", a: 8.90, b: 8.24, c: 0.96, chi_plus: 18.10, seed_charge: 0.01 },
    GasteigerRow { atom_type: "pn", a: 8.90, b: 8.24, c: 0.96, chi_plus: 18.10, seed_charge: 0.00 },
    GasteigerRow { atom_type: "p", a: 8.90, b: 8.24, c: 0.96, chi_plus: 18.10, seed_charge: 0.00 },
];
