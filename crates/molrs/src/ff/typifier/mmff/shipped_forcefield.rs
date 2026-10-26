//! The shipped MMFF parameter set, assembled from the compiled table.
//!
//! `Mmff94Typifier::new()` and `Mmff94sTypifier::new()` parse no XML: both
//! share one library per variant ([`library`]), built once from the compiled table
//! ([`crate::ff::params::mmff`]); the two differ by exactly two things:
//! the force-field **name** they build under, and the
//! [`MmffVariant`] their front door pins. Nothing
//! here is a number: every value comes from the table.

use std::sync::{Arc, OnceLock};

use super::properties::MmffVariant;
use crate::ff::forcefield::{DefError, ForceField};
use crate::ff::ir::{Params, SpecialBonds};
use crate::ff::params::mmff::{
    MMFF_ELE_STYLE, MMFF_STYLES, MMFF_VDW, MMFF_VDW_STYLE, encode_da_byte,
};

/// MMFF's vdW 1-4 weight.
///
/// Not a column of `<ElectrostaticParams>` and not an oversight: MMFF scales 1-4
/// **electrostatics** (by [`MmffEleStyle::scale14`](crate::ff::params::mmff::MmffEleStyle::scale14))
/// and leaves 1-4 vdW at full strength, because its torsion parameters were
/// fitted against unscaled 1-4 vdW. The next reader tempted to "fix" this to 0.5
/// by analogy with Amber would corrupt every MMFF vdW energy with a 1-4 pair.
const VDW_SCALE_14: f64 = 1.0;

/// The cutoff of MMFF's electrostatics, Å: RDKit's `nonBondedThresh` default
/// (100 Å), which in practice prices every pair of a molecule. Not a table
/// value — MMFF states none — but where the Coulomb style stops is the force
/// field's to declare, so the shipped field states RDKit's.
const MMFF_COULOMB_CUTOFF: f64 = 100.0;

/// The shipped library of `variant`, built on first use and shared after.
///
/// [`force_field`] runs once per variant, under the variant's own name
/// (`"MMFF94"` or `"MMFF94s"`), so constructing an MMFF typifier is an `Arc`
/// clone.
pub(super) fn library(variant: MmffVariant) -> Arc<ForceField> {
    static MMFF94: OnceLock<Arc<ForceField>> = OnceLock::new();
    static MMFF94S: OnceLock<Arc<ForceField>> = OnceLock::new();
    let (cell, name) = match variant {
        MmffVariant::Mmff94 => (&MMFF94, "MMFF94"),
        MmffVariant::Mmff94s => (&MMFF94S, "MMFF94s"),
    };
    Arc::clone(cell.get_or_init(|| Arc::new(force_field(name))))
}

/// Build the shipped [`ForceField`] under `name` (`"MMFF94"` or `"MMFF94s"`).
///
/// An input-free constructor over a compiled table (`name` only labels it):
/// the one definition result it can meet is the table's own, and
/// `tests::force_field_defines_without_conflict` proves it `Ok` for both
/// variants.
fn force_field(name: &str) -> ForceField {
    try_force_field(name).expect(
        "MMFF table defines without conflict — proved by \
         ff::typifier::mmff::shipped_forcefield::tests::force_field_defines_without_conflict",
    )
}

/// The fallible body of [`force_field`].
///
/// The style *order* is the table's ([`MMFF_STYLES`]) because `ForceField`
/// lookups take the first matching style; the numbers are the table's too.
fn try_force_field(name: &str) -> Result<ForceField, DefError> {
    let mut ff = ForceField::new(name);
    // The MMFF94 tables are kcal/mol and Å.
    ff.set_units("real");

    for style in MMFF_STYLES {
        match style.category {
            "bond" | "angle" | "dihedral" | "improper" => {
                ff.def_style(style.category, style.name, Params::new())?;
            }
            // The two pair styles are the only ones carrying style-level params,
            // and `mmff_vdw` is the only one with per-type rows (the other five
            // are `ParamSource::PerInstance`).
            "pair" if style.name == "mmff_vdw" => {
                let vdw = ff.def_style(
                    "pair",
                    style.name,
                    Params::from_iter([
                        ("B", MMFF_VDW_STYLE.b),
                        ("Beta", MMFF_VDW_STYLE.beta),
                        ("DARAD", MMFF_VDW_STYLE.darad),
                        ("DAEPS", MMFF_VDW_STYLE.daeps),
                    ]),
                )?;
                for row in MMFF_VDW {
                    let atom_type = row.atom_type.to_string();
                    vdw.def_type(
                        &atom_type,
                        &[&atom_type],
                        Params::from_iter([
                            ("alpha", row.alpha_i),
                            ("n_eff", row.n_i),
                            ("a_i", row.a_i),
                            ("g_i", row.g_i),
                            ("da", encode_da_byte(row.da)),
                        ]),
                    )?;
                }
            }
            // The electrostatic style is the GENERIC buffered Coulomb
            // (`pair/coul/cut`), not a kernel of MMFF's own: MMFF's electrostatics
            // is `E = k·qᵢqⱼ/(D·(r + δ))`, which is that kernel parameterised. So
            // all three numbers travel on the style — including the Coulomb
            // constant, which is Halgren's 332.0716 here and CODATA's 332.06371 for
            // OPLS/LAMMPS. The kernel has no default for any of them.
            "pair" => {
                ff.def_style(
                    "pair",
                    style.name,
                    Params::from_iter([
                        ("coulomb", MMFF_ELE_STYLE.coulomb),
                        ("dielectric", MMFF_ELE_STYLE.dielectric),
                        ("delta", MMFF_ELE_STYLE.delta),
                        ("cutoff", MMFF_COULOMB_CUTOFF),
                    ]),
                )?;
            }
            other => unreachable!("MMFF declares no `{other}` style"),
        }
    }

    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, VDW_SCALE_14],
        coul: [0.0, 0.0, MMFF_ELE_STYLE.scale14],
    });
    Ok(ff)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both shipped MMFF variants (MMFF94, MMFF94s) define every style and type
    /// without a conflict. `force_field` `expect`s this result and names this
    /// test.
    #[test]
    fn force_field_defines_without_conflict() {
        for name in ["MMFF94", "MMFF94s"] {
            assert_eq!(try_force_field(name).err(), None, "{name}");
        }
    }

    /// MMFF's Coulomb style stops at RDKit's 100 Å; its vdW style states no
    /// cutoff of its own.
    #[test]
    fn the_coulomb_style_states_rdkit_s_cutoff() {
        let ff = try_force_field("MMFF94").unwrap();
        let cutoff = |name: &str| {
            ff.get_style("pair", name)
                .unwrap()
                .params()
                .get("cutoff")
                .and_then(|v| v.as_num())
        };
        assert_eq!(cutoff("coul/cut"), Some(MMFF_COULOMB_CUTOFF));
    }
}
