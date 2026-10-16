//! Drude oscillators and virtual sites in LAMMPS's files.
//!
//! A frame states a Drude oscillator as a `drudes` row (`atomi` the core,
//! `atomj` its Drude particle) priced by `drude harmonic`, `k·r²`. LAMMPS's
//! DRUDE package has no section of its own for it: the core–shell spring is
//! a bond (`bond_style harmonic`, `K (r − r0)²` with `r0 = 0`, the same
//! unhalved `K = k`), `fix drude` names which atom types are cores and
//! shells, and `pair_style thole` screens the dipoles
//! (<https://docs.lammps.org/Howto_drude.html>). So:
//!
//! - the data writer writes each `drudes` row as a `Bonds` row, its type in
//!   one bond-type id space with the bonds' ([`lammps_type_labels`]: the
//!   `drudes` labels are bond labels), and the `fix drude` C/D/N flags as a
//!   header comment ([`drude_flags`]);
//! - the force-field writer writes each used `drude harmonic` type as a
//!   `bond harmonic` type of the same name with `r0 = 0`
//!   ([`drude_springs_as_bonds`]), and lists each spring's `alpha` and
//!   `thole` in a comment of the include: they are the input of
//!   `pair_style thole` / `fix drude`, which neither writer emits.
//!
//! A `virtual_sites` row (a TIP4P M site, …) has no data-file form: LAMMPS
//! builds no site from a data file. The data writer refuses it
//! ([`refuse_virtual_sites`]) rather than writing the site as a free atom.

use crate::io::invalid_data;
use molrs::core::FrameAccess;
use molrs::core::TypeLabels;
use molrs::core::keys;
use molrs::core::schema::block_names::{BONDS, DRUDES, VIRTUAL_SITES};
use molrs::op::Idx;

/// The block whose rows are Drude springs, and the data-file topology block
/// they are written into.
pub(crate) const SPRINGS: (&str, &str) = (DRUDES, BONDS);

/// The frame's type labels as a LAMMPS file numbers them: [`TypeLabels`],
/// with every `drudes` row typed as a further `bonds` row
/// ([`TypeLabels::from_frame_folded`]), since a Drude spring is a LAMMPS
/// bond. The `bonds` block's `type_ids` list the bonds' rows, then the
/// drudes'.
///
/// The data writer and [`LammpsForcefieldWriter`](crate::io::lammps::LammpsForcefieldWriter)'s
/// path doors number their types through it; a caller building the
/// coefficient writer itself passes these labels so its `bond_coeff` ids
/// match the data file's.
///
/// # Errors
///
/// The errors of [`TypeLabels::from_frame`], and `drudes` rows beside
/// `bonds` rows typed by numeric `type_id` (the two share an id space by
/// label).
pub fn lammps_type_labels(frame: &impl FrameAccess) -> Result<TypeLabels, String> {
    TypeLabels::from_frame_folded(frame, SPRINGS.1, SPRINGS.0)
}

/// Refuse a frame with `virtual_sites` rows: a LAMMPS data file has no
/// virtual-site construction, so each site would be written as a free atom
/// (no mass of its own, nothing keeping it on its frame atoms).
///
/// LAMMPS's TIP4P is `pair_style lj/cut/tip4p/long` (or another `*/tip4p/*`
/// style): it places the M site from each water's O and H at every step, and
/// the data file holds the three-site water with the M charge on O.
pub(crate) fn refuse_virtual_sites(frame: &impl FrameAccess) -> std::io::Result<()> {
    let n = frame
        .visit_block(VIRTUAL_SITES, |b| b.n_rows().unwrap_or(0))
        .unwrap_or(0);
    if n == 0 {
        return Ok(());
    }
    Err(invalid_data(format!(
        "frame['{VIRTUAL_SITES}'] has {n} rows: a LAMMPS data file has no virtual-site \
         construction, and its site atoms would be written as free particles. For TIP4P, \
         drop the M-site atoms and the '{VIRTUAL_SITES}' block, put each M charge back on \
         its O, and run a */tip4p/* pair style (lj/cut/tip4p/long), which places M itself"
    )))
}

/// The `fix drude` C/D/N flag list of the DRUDE package, one flag per atom
/// type in type-id order, or `None` when the frame has no Drude particle.
///
/// A row is a Drude particle (`D`) when a `drudes` relation names it as its
/// `atomj`, or its `vsite` column reads `"drude"` (a `DrudeParticle` node). A
/// type is a polarisable core (`C`) when one of its rows is the `atomi` of a
/// `drudes` relation or is bonded (`bonds`) to a Drude particle; every other
/// type is `N`. <https://docs.lammps.org/fix_drude.html>
pub(crate) fn drude_flags(
    frame: &impl FrameAccess,
    type_ids: &[Idx],
    n_types: usize,
) -> Option<String> {
    let n = type_ids.len();
    let mut shell = vec![false; n];
    if let Some(vsite) = frame.column("atoms", "vsite").and_then(|c| c.as_string()) {
        for (i, v) in vsite.iter().enumerate().take(n) {
            shell[i] = v == "drude";
        }
    }
    let pairs = |block: &str| -> Vec<(usize, usize)> {
        match (
            frame.column(block, keys::ATOMI).and_then(|c| c.as_uint()),
            frame.column(block, keys::ATOMJ).and_then(|c| c.as_uint()),
        ) {
            (Some(i), Some(j)) => i
                .iter()
                .zip(j.iter())
                .map(|(&a, &b)| (a as usize, b as usize))
                .filter(|&(a, b)| a < n && b < n)
                .collect(),
            _ => Vec::new(),
        }
    };
    let drudes = pairs(DRUDES);
    for &(_, d) in &drudes {
        shell[d] = true;
    }
    if !shell.iter().any(|&s| s) {
        return None;
    }
    let mut core = vec![false; n];
    for &(c, _) in &drudes {
        core[c] = true;
    }
    for (a, b) in pairs(BONDS) {
        if shell[a] && !shell[b] {
            core[b] = true;
        } else if shell[b] && !shell[a] {
            core[a] = true;
        }
    }
    let mut flags = vec!['N'; n_types];
    for (i, &t) in type_ids.iter().enumerate() {
        let Some(slot) = (t as usize).checked_sub(1).and_then(|k| flags.get_mut(k)) else {
            continue;
        };
        if shell[i] {
            *slot = 'D';
        } else if core[i] && *slot != 'D' {
            *slot = 'C';
        }
    }
    Some(
        flags
            .iter()
            .map(char::to_string)
            .collect::<Vec<_>>()
            .join(" "),
    )
}

#[cfg(feature = "ff")]
pub(crate) use springs::{drude_note, drude_springs_as_bonds};

/// The force-field side: `drude harmonic` types as `bond harmonic` types.
#[cfg(feature = "ff")]
mod springs {
    use std::borrow::Cow;
    use std::collections::HashSet;

    use crate::ff::forcefield::ForceField;
    use crate::ff::ir::{Engine, Params, UnitScale};
    use crate::ff::style_registry::Registry;
    use crate::io::writer::ForceFieldWriteError;

    /// The molrs category and style of a Drude spring, and the LAMMPS bond
    /// style it is written as.
    const DRUDE: (&str, &str) = ("drude", "harmonic");
    const BOND: (&str, &str) = ("bond", "harmonic");

    /// `ff` with every `drude harmonic` type moved to a `bond harmonic` type
    /// of the same name and endpoints, `k` kept and `r0 = 0`: molrs prices a
    /// spring `k·r²`, LAMMPS's harmonic bond `K (r − r0)²`, both unhalved.
    /// `ff` itself when it holds no Drude type.
    ///
    /// # Errors
    ///
    /// A Drude style other than `harmonic` (no LAMMPS bond form is known for
    /// it), a Drude type without `k`, and a Drude type whose name a bond type
    /// already holds with other parameters.
    pub(crate) fn drude_springs_as_bonds<'f>(
        ff: &'f ForceField,
        reg: &Registry,
    ) -> Result<Cow<'f, ForceField>, ForceFieldWriteError> {
        let styles: Vec<_> = ff
            .get_styles(DRUDE.0)
            .into_iter()
            .filter(|s| !s.type_rows().is_empty())
            .collect();
        if styles.is_empty() {
            return Ok(Cow::Borrowed(ff));
        }
        let mut springs: Vec<(String, [String; 2], f64)> = Vec::new();
        for style in styles {
            if style.name() != DRUDE.1 {
                return Err(Engine::Lammps
                    .refuse(
                        DRUDE.0,
                        style.name(),
                        "the DRUDE package's core-shell spring is a bond; only `drude \
                         harmonic` (k*r^2, bond harmonic with r0 = 0) has that form",
                    )
                    .into());
            }
            for (name, ends, params) in style.type_rows() {
                let k = params.get("k").ok_or_else(|| {
                    format!("drude type `{name}`: no `k`, the spring's stiffness")
                })?;
                springs.push((name.to_owned(), [ends[0].to_owned(), ends[1].to_owned()], k));
            }
        }
        let mut view = ff.clone();
        if view.get_style(BOND.0, BOND.1).is_none() {
            view.def_style_in(reg, BOND.0, BOND.1, Params::new())
                .map_err(|e| format!("the bond style of the Drude springs: {e}"))?;
        }
        for (name, [core, shell], k) in &springs {
            view.get_style_mut(BOND.0, BOND.1)
                .expect("defined above")
                .def_type(
                    name,
                    &[core, shell],
                    Params::from_pairs(&[("k", *k), ("r0", 0.0)]),
                )
                .map_err(|e| format!("drude type `{name}` as a LAMMPS bond type: {e}"))?;
            view.get_style_mut(DRUDE.0, DRUDE.1)
                .expect("a drude harmonic style holds it")
                .remove_type(name);
        }
        Ok(Cow::Owned(view))
    }

    /// The include's comment on the Drude springs among `bond_labels`: which
    /// bond types they are, and each one's `alpha` and `thole` — the input of
    /// `pair_style thole` and `fix drude`, which the include does not write.
    /// Empty when no used bond label is a Drude type of `ff`.
    pub(crate) fn drude_note(
        ff: &ForceField,
        bond_labels: &[String],
        reg: &Registry,
        units: &UnitScale,
        precision: usize,
    ) -> Result<Vec<String>, ForceFieldWriteError> {
        let used: HashSet<&str> = bond_labels.iter().map(String::as_str).collect();
        let Some(style) = ff.get_style(DRUDE.0, DRUDE.1) else {
            return Ok(Vec::new());
        };
        let rows: Vec<_> = style
            .type_rows()
            .into_iter()
            .filter(|(name, _, _)| used.contains(name))
            .collect();
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let (spec, _) = reg.style(DRUDE.0, DRUDE.1).ok_or_else(|| {
            Engine::Lammps.refuse(DRUDE.0, DRUDE.1, "the style is not registered")
        })?;
        let dim = |param: &str| {
            spec.param(param)
                .map(|p| p.dim)
                .ok_or_else(|| format!("drude harmonic declares no `{param}`"))
        };
        let (alpha_dim, thole_dim) = (dim("alpha")?, dim("thole")?);
        let names: Vec<&str> = rows.iter().map(|(name, _, _)| *name).collect();
        let mut lines = vec![
            format!(
                "# DRUDE package: bond types {} are the Drude core-shell springs \
                 (drude harmonic k*r^2 written as bond harmonic K = k, r0 = 0).\n",
                names.join(" ")
            ),
            "# Not written, for the input script: fix drude (its C/D/N flags head the \
             data file) and pair_style thole, from each spring's alpha and thole:\n"
                .to_owned(),
        ];
        let fmt = |v: Option<f64>, dim| {
            v.map_or_else(
                || "-".to_owned(),
                |v| format!("{:.precision$}", units.apply(v, dim)),
            )
        };
        for (name, ends, params) in rows {
            lines.push(format!(
                "#   {name} ({} {}): alpha {} thole {}\n",
                ends[0],
                ends[1],
                fmt(params.get("alpha"), alpha_dim),
                fmt(params.get("thole"), thole_dim),
            ));
        }
        lines.push("\n".to_owned());
        Ok(lines)
    }
}
