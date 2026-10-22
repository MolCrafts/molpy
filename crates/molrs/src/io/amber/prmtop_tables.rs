//! AMBER prmtop table helpers shared by the structure reader
//! ([`super::prmtop`]) and the force-field reader
//! (`io::amber::prmtop_forcefield`): POINTERS and 20a4 name parsing, atom
//! type names, torsions, the 1-4 list, and the CHARMM (chamber) sections, so
//! the two readers name every row the same way.
//!
//! Decoding parameter *values* into a force field is the force-field reader's
//! job; nothing here builds parameter rows of its own.

use molrs::core::TypeName;

use std::collections::HashMap;

/// Parse POINTERS lines into a meta map.
///
/// Includes both raw Amber fields (`NATOM`, …) and derived counts
/// (`n_atoms`, `n_bonds`, …). Accepts 30- or 31-value POINTERS (NCOPY optional).
pub(crate) fn parse_pointers(lines: &[String]) -> Result<HashMap<String, i64>, String> {
    let values: Vec<i64> = parse_tokens(lines)?;
    const FIELDS: &[&str] = &[
        "NATOM", "NTYPES", "NBONH", "MBONA", "NTHETH", "MTHETA", "NPHIH", "MPHIA", "NHPARM",
        "NPARM", "NNB", "NRES", "NBONA", "NTHETA", "NPHIA", "NUMBND", "NUMANG", "NPTRA", "NATYP",
        "NPHB", "IFPERT", "NBPER", "NGPER", "NDPER", "MBPER", "MGPER", "MDPER", "IFBOX", "NMXRS",
        "IFCAP", "NUMEXTRA", "NCOPY",
    ];
    let mut meta_data: HashMap<String, i64> = HashMap::new();
    for (name, val) in FIELDS.iter().zip(values.iter()) {
        meta_data.insert((*name).to_string(), *val);
    }
    // Graceful short POINTERS: missing keys stay absent; derived use unwrap_or(0).
    let natom = *meta_data.get("NATOM").unwrap_or(&0);
    let nbonh = *meta_data.get("NBONH").unwrap_or(&0);
    let mbona = *meta_data.get("MBONA").unwrap_or(&0);
    let ntheth = *meta_data.get("NTHETH").unwrap_or(&0);
    let mtheta = *meta_data.get("MTHETA").unwrap_or(&0);
    let nphih = *meta_data.get("NPHIH").unwrap_or(&0);
    let mphia = *meta_data.get("MPHIA").unwrap_or(&0);
    let natyp = *meta_data.get("NATYP").unwrap_or(&0);
    let numbnd = *meta_data.get("NUMBND").unwrap_or(&0);
    let numang = *meta_data.get("NUMANG").unwrap_or(&0);
    let nptra = *meta_data.get("NPTRA").unwrap_or(&0);

    let mut meta = meta_data;
    meta.insert("n_atoms".into(), natom);
    meta.insert("n_bonds".into(), nbonh + mbona);
    meta.insert("n_angles".into(), ntheth + mtheta);
    meta.insert("n_dihedrals".into(), nphih + mphia);
    meta.insert("n_atomtypes".into(), natyp);
    meta.insert("n_bondtypes".into(), numbnd);
    meta.insert("n_angletypes".into(), numang);
    meta.insert("n_dihedraltypes".into(), nptra);
    Ok(meta)
}

/// Fortran `20a4` name fields (strip each 4-char window).
pub(crate) fn parse_a4_names(lines: &[String]) -> Vec<String> {
    let mut names = Vec::new();
    for line in lines {
        let mut i = 0;
        while i < line.len() {
            let end = (i + 4).min(line.len());
            names.push(line[i..end].trim().to_string());
            i += 4;
        }
    }
    names
}

// ---------------------------------------------------------------------------
// Torsions, the 1-4 list and the CHARMM (chamber) sections — shared by the
// structure reader and the force-field reader, so the two name every row the
// same way.
// ---------------------------------------------------------------------------

/// Whether a section map is a chamber (CHARMM) prmtop: it has a `CTITLE`
/// where an AMBER prmtop has a `TITLE` (ParmEd's and sander's own test).
pub(crate) fn is_chamber(sections: &HashMap<String, Vec<String>>) -> bool {
    sections.contains_key("CTITLE")
}

/// Each atom's type name, `n_atoms` of them (`""` past the end of
/// `AMBER_ATOM_TYPE`).
///
/// It is the atom's `AMBER_ATOM_TYPE` — unless that name stands for atoms of
/// two LJ classes (`ATOM_TYPE_INDEX`) or two masses. A chamber prmtop cuts
/// CHARMM's types to four characters (`CC3161`, `CC3162`, `CC3163` are all
/// `CC31`), and ParmEd's `addLJType` gives some atoms of a name their own
/// class. Every atom of such a name is then named `<name>~<class>`, with
/// `~<n>` appended (n = 1, 2, … in order of appearance) where one class still
/// carries two masses: one type per distinct atom, as the file's tables have
/// them.
pub(crate) fn atom_type_names(
    sections: &HashMap<String, Vec<String>>,
    n_atoms: usize,
) -> Result<Vec<String>, String> {
    let mut names = sections
        .get("AMBER_ATOM_TYPE")
        .map(|l| parse_a4_names(l))
        .unwrap_or_default();
    names.resize(n_atoms, String::new());
    let class: Vec<i64> = section(sections, "ATOM_TYPE_INDEX")?;
    let mass: Vec<f64> = section(sections, "MASS")?;
    let key = |i: usize| {
        (
            class.get(i).copied().unwrap_or(0),
            mass.get(i).copied().unwrap_or(0.0).to_bits(),
        )
    };
    let mut variants: HashMap<&str, Vec<(i64, u64)>> = HashMap::new();
    for (i, name) in names.iter().enumerate() {
        let v = variants.entry(name.as_str()).or_default();
        if !v.contains(&key(i)) {
            v.push(key(i));
        }
    }
    let renamed: Vec<String> = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let v = &variants[name.as_str()];
            if name.is_empty() || v.len() == 1 {
                return name.clone();
            }
            let (c, m) = key(i);
            let same_class: Vec<u64> = v.iter().filter(|k| k.0 == c).map(|k| k.1).collect();
            if same_class.len() == 1 {
                format!("{name}~{c}")
            } else {
                let n = same_class.iter().position(|&b| b == m).unwrap_or(0) + 1;
                format!("{name}~{c}~{n}")
            }
        })
        .collect();
    Ok(renamed)
}

/// The prefix of a file's CMAP sections: `CHARMM_CMAP_*` in a chamber
/// prmtop, `CMAP_*` in an AMBER one (ff19SB), `None` without CMAP terms.
pub(crate) fn cmap_prefix(sections: &HashMap<String, Vec<String>>) -> Option<&'static str> {
    if sections.contains_key("CHARMM_CMAP_INDEX") {
        Some("CHARMM_")
    } else if sections.contains_key("CMAP_INDEX") {
        Some("")
    } else {
        None
    }
}

/// One cosine term `k·[1 + cos(n·φ − phase)]` of a prmtop torsion; `phase`
/// in radians, as the file stores it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TorsionTerm {
    pub k: f64,
    pub periodicity: f64,
    pub phase: f64,
}

/// The dihedral parameter tables a torsion row's type id points into.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TorsionTables<'a> {
    pub k: &'a [f64],
    pub periodicity: &'a [f64],
    pub phase: &'a [f64],
}

impl TorsionTables<'_> {
    /// The terms of type id `tid` (1-based): the type, and, while its
    /// periodicity is negative, the next one — AMBER's multi-term chain,
    /// which sander expands in a prmtop as in a parameter file. Each term's
    /// periodicity is `|PN|`.
    pub fn chain(&self, tid: i64) -> Result<Vec<TorsionTerm>, String> {
        let mut out = Vec::new();
        let mut t = tid;
        loop {
            let i = usize::try_from(t - 1)
                .ok()
                .filter(|&i| i < self.k.len() && i < self.periodicity.len() && i < self.phase.len())
                .ok_or_else(|| {
                    format!("dihedral type {t} is out of range of the parameter tables")
                })?;
            let pn = self.periodicity[i];
            out.push(TorsionTerm {
                k: self.k[i],
                periodicity: pn.abs(),
                phase: amber_phase(self.phase[i]),
            });
            if pn >= 0.0 {
                return Ok(out);
            }
            t += 1;
        }
    }
}

/// One torsion of a prmtop: every `DIHEDRALS_*` row on one atom quartet.
///
/// A proper is stored in [`TypeName::orient`]'s orientation of its type
/// names (atoms reversed with them), and its rows are merged by quartet in
/// either direction; an improper (negative 4th pointer) keeps its prmtop
/// order, centre third, and merges by exact order. `terms` holds every row's
/// cosine terms in row order (empty without parameter tables); `exclude_14`
/// is set when every merged row's 3rd pointer was negative.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Torsion {
    pub atoms: [usize; 4],
    pub improper: bool,
    pub exclude_14: bool,
    pub terms: Vec<TorsionTerm>,
}

impl Torsion {
    /// The type names of its atoms, in its stored order.
    pub fn types<'a>(&self, atom_types: &'a [String]) -> [&'a str; 4] {
        self.atoms.map(|a| atom_types[a].as_str())
    }

    /// The one-term rows a multi-term **improper** is stored as: each term,
    /// with the type name `<quartet>@<n>`. `improper periodic`, like LAMMPS's
    /// `improper_style cvff`, holds one cosine term, so a multi-term improper
    /// is several rows on its four atoms, one per term. A single-term improper
    /// is one row named by its quartet alone. Two terms of one improper with
    /// the same periodicity are an `Err`: they have no one-term-per-row form.
    pub fn improper_rows(
        &self,
        atom_types: &[String],
    ) -> Result<Vec<(String, Option<TorsionTerm>)>, String> {
        let base = TypeName::join(&self.types(atom_types))?;
        if self.terms.len() <= 1 {
            return Ok(vec![(base.to_string(), self.terms.first().copied())]);
        }
        let mut out: Vec<(String, Option<TorsionTerm>)> = Vec::with_capacity(self.terms.len());
        for term in &self.terms {
            let n = integral_periodicity(term.periodicity, base.as_str())?;
            let name = base.with_qualifier(&[&n.to_string()])?.to_string();
            if out.iter().any(|(other, _)| *other == name) {
                return Err(format!(
                    "improper {base}: two terms of periodicity {n}; an improper is one row per \
                     cosine term, so its terms need distinct periodicities"
                ));
            }
            out.push((name, Some(*term)));
        }
        Ok(out)
    }
}

/// The type name of every **proper** torsion of `torsions` (`None` for an
/// improper, whose rows [`Torsion::improper_rows`] names).
///
/// A proper is named by its quartet. tleap can give two torsions of one
/// quartet different terms — it caches the first match of a quartet in the
/// unit's own parameter set, so a later torsion may take a specific row where
/// an earlier one took the specific and a wildcard row (GAFF2's `hc-c3-ca-ca`
/// beside `X -c3-ca-X`) — and a type holds one set of terms. Each further
/// distinct set of terms on a quartet is named `<quartet>@<n>`, `n` counting
/// the quartet's distinct sets from 2 in order of appearance; the first keeps
/// the bare name. Term sets are compared in a canonical order, so rows
/// written in another order are one set.
pub(crate) fn proper_type_names(
    torsions: &[Torsion],
    atom_types: &[String],
) -> Result<Vec<Option<String>>, String> {
    let mut sets: HashMap<String, Vec<Vec<TorsionTerm>>> = HashMap::new();
    let mut out = Vec::with_capacity(torsions.len());
    for t in torsions {
        if t.improper {
            out.push(None);
            continue;
        }
        let base = TypeName::join(&t.types(atom_types))?;
        let terms = canonical_terms(&t.terms);
        let seen = sets.entry(base.to_string()).or_default();
        let n = match seen.iter().position(|other| *other == terms) {
            Some(n) => n,
            None => {
                seen.push(terms);
                seen.len() - 1
            }
        };
        out.push(Some(if n == 0 {
            base.to_string()
        } else {
            base.with_qualifier(&[&(n + 1).to_string()])?.to_string()
        }));
    }
    Ok(out)
}

/// `terms` in one canonical order — by periodicity, then phase, then k — so
/// two torsions with the same terms in a different row order compare equal.
pub(crate) fn canonical_terms(terms: &[TorsionTerm]) -> Vec<TorsionTerm> {
    let mut out = terms.to_vec();
    out.sort_by(|a, b| {
        a.periodicity
            .total_cmp(&b.periodicity)
            .then(a.phase.total_cmp(&b.phase))
            .then(a.k.total_cmp(&b.k))
    });
    out
}

/// `n` as an integer, which every LAMMPS torsion style requires.
fn integral_periodicity(n: f64, what: &str) -> Result<i64, String> {
    if n.fract() != 0.0 || !n.is_finite() {
        return Err(format!("{what}: periodicity {n} is not an integer"));
    }
    Ok(n as i64)
}

/// A prmtop phase as sander prices it: within 0.004 rad of ±π it is ±π
/// exactly (sander's `rdparm`, `abs(phase − π) ≤ 4·10⁻³`). tleap writes π as
/// `3.14159400E+00`; taken as written, every such term is off by 1.3·10⁻⁶ rad
/// and a sine term appears that the force field does not have.
pub(crate) fn amber_phase(phase: f64) -> f64 {
    if (phase.abs() - std::f64::consts::PI).abs() <= 4e-3 {
        std::f64::consts::PI.copysign(phase)
    } else {
        phase
    }
}

/// One prmtop torsion row, decoded: 0-based atoms in file order.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TorsionRow {
    pub(crate) atoms: [usize; 4],
    pub(crate) improper: bool,
    pub(crate) exclude_14: bool,
    pub(crate) tid: i64,
}

pub(crate) fn torsion_rows(pointers: &[i64], n_atoms: usize) -> Result<Vec<TorsionRow>, String> {
    if !pointers.len().is_multiple_of(5) {
        return Err(format!(
            "dihedral pointer length {} not multiple of 5",
            pointers.len()
        ));
    }
    let mut out = Vec::with_capacity(pointers.len() / 5);
    for chunk in pointers.as_chunks::<5>().0 {
        if chunk[0] < 0 || chunk[1] < 0 {
            return Err(format!(
                "Found negative dihedral atom pointers ({}, {}, {}, {})",
                chunk[0], chunk[1], chunk[2], chunk[3]
            ));
        }
        let atoms = [
            (chunk[0] / 3) as usize,
            (chunk[1] / 3) as usize,
            (chunk[2].unsigned_abs() / 3) as usize,
            (chunk[3].unsigned_abs() / 3) as usize,
        ];
        if let Some(&bad) = atoms.iter().find(|&&a| a >= n_atoms) {
            return Err(format!("dihedral atom index {bad} out of range"));
        }
        out.push(TorsionRow {
            atoms,
            improper: chunk[3] < 0,
            exclude_14: chunk[2] < 0,
            tid: chunk[4],
        });
    }
    Ok(out)
}

/// The torsions of `DIHEDRALS_INC_HYDROGEN` + `DIHEDRALS_WITHOUT_HYDROGEN`
/// (`pointers`, concatenated), one per atom quartet ([`Torsion`]), in the
/// order of each quartet's first row. With `tables`, each row's type id is
/// expanded through its multi-term chain into the torsion's `terms`.
pub(crate) fn decode_torsions(
    pointers: &[i64],
    atom_types: &[String],
    tables: Option<TorsionTables<'_>>,
) -> Result<Vec<Torsion>, String> {
    let mut out: Vec<Torsion> = Vec::new();
    let mut seen: HashMap<(bool, [usize; 4]), usize> = HashMap::new();
    for row in torsion_rows(pointers, atom_types.len())? {
        let mut atoms = row.atoms;
        let types = atoms.map(|a| atom_types[a].as_str());
        if !row.improper && TypeName::reads_reversed(&types) {
            atoms.reverse();
        }
        let key = if row.improper {
            atoms
        } else {
            let mut reversed = atoms;
            reversed.reverse();
            atoms.min(reversed)
        };
        let terms = match tables {
            Some(t) => t.chain(row.tid)?,
            None => Vec::new(),
        };
        if let Some(&at) = seen.get(&(row.improper, key)) {
            out[at].exclude_14 &= row.exclude_14;
            out[at].terms.extend(terms);
            continue;
        }
        seen.insert((row.improper, key), out.len());
        out.push(Torsion {
            atoms,
            improper: row.improper,
            exclude_14: row.exclude_14,
            terms,
        });
    }
    Ok(out)
}

/// The tokens of section `flag`, parsed as `T`; empty when it is absent.
pub(crate) fn section<T: std::str::FromStr>(
    sections: &HashMap<String, Vec<String>>,
    flag: &str,
) -> Result<Vec<T>, String>
where
    T::Err: std::fmt::Display,
{
    match sections.get(flag) {
        Some(lines) => parse_tokens(lines).map_err(|e| format!("%FLAG {flag}: {e}")),
        None => Ok(Vec::new()),
    }
}

/// A 1-based atom number of a chamber section as a 0-based index, checked
/// against `n_atoms`.
pub(crate) fn atom_number(flag: &str, v: i64, n_atoms: usize) -> Result<usize, String> {
    usize::try_from(v - 1)
        .ok()
        .filter(|&a| a < n_atoms)
        .ok_or_else(|| format!("%FLAG {flag}: atom number {v} is out of range 1..={n_atoms}"))
}

/// The value of a 1-based parameter index into a table.
pub(crate) fn table_value(flag: &str, table: &[f64], index: i64) -> Result<f64, String> {
    usize::try_from(index - 1)
        .ok()
        .and_then(|i| table.get(i).copied())
        .ok_or_else(|| format!("%FLAG {flag} has no parameter {index}"))
}

/// A chamber CHARMM improper `K_psi·(psi − psi0)²`: its atoms (0-based, in
/// file order, CHARMM's centre first), `K_psi` (kcal/mol/rad², no ½) and
/// `psi0` (radians).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CharmmImproper {
    pub atoms: [usize; 4],
    pub k: f64,
    pub psi0: f64,
}

/// `CHARMM_IMPROPERS` (four 1-based atom numbers and a type) with
/// `CHARMM_IMPROPER_FORCE_CONSTANT` / `CHARMM_IMPROPER_PHASE`; empty when the
/// file has none.
pub(crate) fn chamber_impropers(
    sections: &HashMap<String, Vec<String>>,
    n_atoms: usize,
) -> Result<Vec<CharmmImproper>, String> {
    const FLAG: &str = "CHARMM_IMPROPERS";
    let rows: Vec<i64> = section(sections, FLAG)?;
    let k: Vec<f64> = section(sections, "CHARMM_IMPROPER_FORCE_CONSTANT")?;
    let psi0: Vec<f64> = section(sections, "CHARMM_IMPROPER_PHASE")?;
    if !rows.len().is_multiple_of(5) {
        return Err(format!(
            "%FLAG {FLAG} has {} entries, not quintuples",
            rows.len()
        ));
    }
    rows.as_chunks::<5>()
        .0
        .iter()
        .map(|&[a, b, c, d, t]| {
            Ok(CharmmImproper {
                atoms: [
                    atom_number(FLAG, a, n_atoms)?,
                    atom_number(FLAG, b, n_atoms)?,
                    atom_number(FLAG, c, n_atoms)?,
                    atom_number(FLAG, d, n_atoms)?,
                ],
                k: table_value("CHARMM_IMPROPER_FORCE_CONSTANT", &k, t)?,
                psi0: table_value("CHARMM_IMPROPER_PHASE", &psi0, t)?,
            })
        })
        .collect()
}

/// A file's CMAP crossterms and maps, named.
///
/// Each crossterm's type name is [`TypeName::join`] of its five atom types —
/// the endpoints `assign_cmaps` matches. When one such name would stand for
/// two different maps (ff19SB keys its maps by residue, on shared atom
/// types), every crossterm of that name is qualified by the residue label of
/// its third (Cα) atom, `<types>@<residue>`. `maps` holds each name's grid
/// (row-major `N × N`, the file's order) once.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CmapTerms {
    /// Five 0-based atoms per crossterm.
    pub atoms: Vec<[usize; 5]>,
    /// The type name of each crossterm.
    pub names: Vec<String>,
    /// `(name, endpoints, N, grid)` per distinct name, in first-use order.
    pub maps: Vec<(String, [String; 5], usize, Vec<f64>)>,
}

/// The CMAP sections (`CHARMM_CMAP_*` or `CMAP_*`, [`cmap_prefix`]) as
/// [`CmapTerms`]; `None` without them.
///
/// The prmtop stores a map as ParmEd's `CmapType.grid`, which is CHARMM's
/// parameter-file order: φ-major from −180° (element `i·N + j` at
/// φ = −180° + i·360°/N, ψ = −180° + j·360°/N) — molrs's and LAMMPS's
/// layout, taken as it is.
pub(crate) fn cmap_terms(
    sections: &HashMap<String, Vec<String>>,
    atom_types: &[String],
) -> Result<Option<CmapTerms>, String> {
    let Some(prefix) = cmap_prefix(sections) else {
        return Ok(None);
    };
    let n_atoms = atom_types.len();
    let flag = |s: &str| format!("{prefix}CMAP_{s}");
    let index_flag = flag("INDEX");
    let rows: Vec<i64> = section(sections, &index_flag)?;
    let resolution: Vec<i64> = section(sections, &flag("RESOLUTION"))?;
    if !rows.len().is_multiple_of(6) {
        return Err(format!(
            "%FLAG {index_flag} has {} entries, not sextuples",
            rows.len()
        ));
    }
    let mut grids: Vec<(usize, Vec<f64>)> = Vec::with_capacity(resolution.len());
    for (m, &n) in resolution.iter().enumerate() {
        let key = flag(&format!("PARAMETER_{:02}", m + 1));
        let grid: Vec<f64> = section(sections, &key)?;
        let n = usize::try_from(n).ok().filter(|&n| n >= 2).ok_or_else(|| {
            format!(
                "%FLAG {}: map {} has resolution {n}",
                flag("RESOLUTION"),
                m + 1
            )
        })?;
        if grid.len() != n * n {
            return Err(format!(
                "%FLAG {key} has {} values, expected {n}×{n} = {}",
                grid.len(),
                n * n
            ));
        }
        grids.push((n, grid));
    }
    let mut atoms = Vec::with_capacity(rows.len() / 6);
    let mut maps_of = Vec::with_capacity(rows.len() / 6);
    for &[a, b, c, d, e, t] in rows.as_chunks::<6>().0 {
        let mut five = [0usize; 5];
        for (slot, v) in five.iter_mut().zip([a, b, c, d, e]) {
            *slot = atom_number(&index_flag, v, n_atoms)?;
        }
        let m = usize::try_from(t - 1)
            .ok()
            .filter(|&m| m < grids.len())
            .ok_or_else(|| format!("%FLAG {index_flag}: map {t} is out of range"))?;
        atoms.push(five);
        maps_of.push(m);
    }
    let bare: Vec<String> = atoms
        .iter()
        .map(|five| TypeName::join(&five.map(|a| atom_types[a].as_str())).map(|n| n.to_string()))
        .collect::<Result<_, _>>()?;
    // A bare name standing for two maps is qualified by residue.
    let mut map_of_name: HashMap<&str, usize> = HashMap::new();
    let mut ambiguous: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for (name, &m) in bare.iter().zip(&maps_of) {
        if let Some(&other) = map_of_name.get(name.as_str())
            && grids[other] != grids[m]
        {
            ambiguous.insert(name.as_str());
        }
        map_of_name.entry(name.as_str()).or_insert(m);
    }
    let residue = if ambiguous.is_empty() {
        Vec::new()
    } else {
        residue_labels(sections, n_atoms)?
    };
    let mut names = Vec::with_capacity(atoms.len());
    let mut maps: Vec<(String, [String; 5], usize, Vec<f64>)> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    for ((five, base), &m) in atoms.iter().zip(&bare).zip(&maps_of) {
        let name = if ambiguous.contains(base.as_str()) {
            let res = residue.get(five[2]).map(String::as_str).unwrap_or("");
            if res.is_empty() || res.contains(['_', '@']) {
                return Err(format!(
                    "cmap {base}: two maps share these atom types and the residue of atom {} \
                     ({res:?}) cannot tell them apart",
                    five[2] + 1
                ));
            }
            TypeName::join(&[base.as_str()])?
                .with_qualifier(&[res])?
                .to_string()
        } else {
            base.clone()
        };
        match seen.get(&name) {
            Some(&k) if maps[k].3 != grids[m].1 => {
                return Err(format!(
                    "cmap {name}: crossterms of one name point at two different maps"
                ));
            }
            Some(_) => {}
            None => {
                seen.insert(name.clone(), maps.len());
                let ends = five.map(|a| atom_types[a].clone());
                maps.push((name.clone(), ends, grids[m].0, grids[m].1.clone()));
            }
        }
        names.push(name);
    }
    Ok(Some(CmapTerms { atoms, names, maps }))
}

/// The `RESIDUE_LABEL` of each atom through `RESIDUE_POINTER`.
fn residue_labels(
    sections: &HashMap<String, Vec<String>>,
    n_atoms: usize,
) -> Result<Vec<String>, String> {
    let labels = sections
        .get("RESIDUE_LABEL")
        .map(|l| parse_a4_names(l))
        .unwrap_or_default();
    let mut starts: Vec<i64> = section(sections, "RESIDUE_POINTER")?;
    starts.push(n_atoms as i64 + 1);
    let mut out = vec![String::new(); n_atoms];
    for (r, w) in starts.windows(2).enumerate() {
        let (lo, hi) = (
            (w[0] - 1).max(0) as usize,
            ((w[1] - 1).max(0) as usize).min(n_atoms),
        );
        for slot in out.iter_mut().take(hi).skip(lo) {
            *slot = labels.get(r).cloned().unwrap_or_default();
        }
    }
    Ok(out)
}

/// Whitespace-separated tokens of `lines`, each parsed as `T`; `Err` names the
/// first token that does not parse.
pub(crate) fn parse_tokens<T: std::str::FromStr>(lines: &[String]) -> Result<Vec<T>, String>
where
    T::Err: std::fmt::Display,
{
    let mut out = Vec::new();
    for line in lines {
        for tok in line.split_whitespace() {
            out.push(
                tok.parse::<T>()
                    .map_err(|e| format!("token {tok:?}: {e}"))?,
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// tleap gave two `hc-c3-ca-ca` torsions of ibuprofen different terms
    /// (the specific row alone, and with GAFF2's `X -c3-ca-X`): the second
    /// set is its own type `<quartet>@2`, and a third torsion with the first
    /// set (rows in another order) shares the bare name.
    #[test]
    fn a_quartet_with_a_second_set_of_terms_is_a_second_type() {
        let types: Vec<String> = ["hc", "c3", "ca", "ca"].map(str::to_owned).to_vec();
        let term = |periodicity: f64| TorsionTerm {
            k: 0.0,
            periodicity,
            phase: 0.0,
        };
        let torsion = |terms: Vec<TorsionTerm>| Torsion {
            atoms: [0, 1, 2, 3],
            improper: false,
            exclude_14: false,
            terms,
        };
        let torsions = vec![
            torsion(vec![term(1.0)]),
            torsion(vec![term(1.0), term(2.0)]),
            torsion(vec![term(2.0), term(1.0)]),
            torsion(vec![term(1.0)]),
        ];
        let names = proper_type_names(&torsions, &types).unwrap();
        let names: Vec<&str> = names.iter().map(|n| n.as_deref().unwrap()).collect();
        assert_eq!(
            names,
            [
                "hc-c3-ca-ca",
                "hc-c3-ca-ca@2",
                "hc-c3-ca-ca@2",
                "hc-c3-ca-ca"
            ]
        );
    }

    #[test]
    fn a_phase_within_4e_3_of_pi_is_pi() {
        use std::f64::consts::PI;
        assert_eq!(amber_phase(3.141594), PI);
        assert_eq!(amber_phase(-3.141594), -PI);
        assert_eq!(amber_phase(3.1), 3.1);
        assert_eq!(amber_phase(0.0), 0.0);
    }

    fn sections(flags: &[(&str, &str)]) -> HashMap<String, Vec<String>> {
        flags
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.lines().map(str::to_owned).collect()))
            .collect()
    }

    /// A name on two LJ classes, and on one class with two masses.
    #[test]
    fn atom_type_names_split_a_name_on_two_classes_or_masses() {
        let s = sections(&[
            ("AMBER_ATOM_TYPE", "CC31CC31CC31HCA1HCA1"),
            ("ATOM_TYPE_INDEX", "1 2 2 3 3"),
            ("MASS", "12.011 12.011 13.0 1.008 1.008"),
        ]);
        assert_eq!(
            atom_type_names(&s, 6).unwrap(),
            vec!["CC31~1", "CC31~2~1", "CC31~2~2", "HCA1", "HCA1", ""]
        );
    }

    /// Two maps on one five-type name are qualified by the residue of the
    /// third atom; one map stays bare.
    #[test]
    fn cmap_names_by_types_then_by_residue() {
        let grid = "0.1 0.2 0.3 0.4";
        let s = sections(&[
            ("CMAP_INDEX", "1 2 3 4 5 1\n4 5 6 7 8 2"),
            ("CMAP_RESOLUTION", "2 2"),
            ("CMAP_PARAMETER_01", grid),
            ("CMAP_PARAMETER_02", "1.0 2.0 3.0 4.0"),
            ("RESIDUE_LABEL", "ALA GLY ALA "),
            ("RESIDUE_POINTER", "1 3 6"),
        ]);
        let types: Vec<String> = ["C", "N", "XC", "C", "N", "XC", "C", "N"]
            .map(str::to_owned)
            .to_vec();
        let t = cmap_terms(&s, &types).unwrap().unwrap();
        assert_eq!(t.names, vec!["C-N-XC-C-N@GLY", "C-N-XC-C-N@ALA"]);
        assert_eq!(t.maps.len(), 2);
        // The same residue on both crossterms' Cα and different maps: refused.
        let s2 = sections(&[
            ("CMAP_INDEX", "1 2 3 4 5 1\n4 5 6 7 8 2"),
            ("CMAP_RESOLUTION", "2 2"),
            ("CMAP_PARAMETER_01", grid),
            ("CMAP_PARAMETER_02", "1.0 2.0 3.0 4.0"),
            ("RESIDUE_LABEL", "ALA ALA "),
            ("RESIDUE_POINTER", "1 4"),
        ]);
        let err = cmap_terms(&s2, &types).unwrap_err();
        assert!(err.contains("two different maps"), "{err}");
        // One map: the bare name.
        let s3 = sections(&[
            ("CMAP_INDEX", "1 2 3 4 5 1"),
            ("CMAP_RESOLUTION", "2"),
            ("CMAP_PARAMETER_01", grid),
        ]);
        let t = cmap_terms(&s3, &types).unwrap().unwrap();
        assert_eq!(t.names, vec!["C-N-XC-C-N"]);
        assert_eq!(t.maps[0].3, vec![0.1, 0.2, 0.3, 0.4]);
    }
}
