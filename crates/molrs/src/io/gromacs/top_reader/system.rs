//! A whole GROMACS topology: the molecule sections, typed against the
//! directives, as a [`Frame`] (the module docs of [`super`], "Whole systems").

use std::collections::{HashMap, HashSet, VecDeque};

use ndarray::{Array1, ArrayD};

use super::{Directives, Kind, Row, Table, convert};
use crate::core::unit_factors::{KCAL_TO_KJ, NM_TO_ANGSTROM};
use crate::ff::potential::MAX_ATOMS_FOR_A_FULL_PAIR_LIST;
use molrs::core::Frame;
use molrs::core::TypeName;
use molrs::core::schema::PAIR_OVERRIDE_COLUMNS;
use molrs::core::{Block, BlockDtype};
use molrs::op::{F, Idx};

/// A molecule type's priced pairs `(i, j, is_14, cells)` and its excluded
/// pairs.
type PairLists = (Vec<(usize, usize, bool, Cells)>, Vec<(usize, usize)>);

/// One `pairs` row's override cells, in [`PAIR_OVERRIDE_COLUMNS`] order
/// (`epsilon`, `sigma`, `charge_product`, `lj_scale`, `coul_scale`).
type Cells = [Option<f64>; 5];

struct MolAtom {
    atype: String,
    resnr: Idx,
    res_name: String,
    name: String,
    charge: f64,
    mass: f64,
}

/// A typed relation row: its atoms (0-based within the molecule) and type.
struct Term {
    atoms: Vec<usize>,
    type_name: String,
}

/// One `[ moleculetype ]` and its sections.
#[derive(Default)]
struct MolType {
    name: String,
    nrexcl: usize,
    atoms: Vec<MolAtom>,
    bonds: Vec<Term>,
    angles: Vec<Term>,
    dihedrals: Vec<Term>,
    impropers: Vec<Term>,
    cmaps: Vec<Term>,
    /// `[ pairs ]`: the two atoms, override cells, and the row's place.
    pairs: Vec<(usize, usize, Cells, String)>,
    exclusions: Vec<(usize, usize)>,
    constraints: Vec<(usize, usize, f64)>,
    /// The bonds that generate exclusions (bonds funct 1 and 3, constraints
    /// funct 1).
    chemical: Vec<(usize, usize)>,
}

/// The types rows with parameters of their own define, deduplicated.
#[derive(Default)]
struct OwnTypes {
    names: HashMap<String, String>,
}

/// The relation block a parameter table's rows go to.
fn block_of(table: Table) -> &'static str {
    match table {
        Table::Bond(_) => "bonds",
        Table::Angle(_) => "angles",
        Table::Pdihs | Table::Rbdihs | Table::Fourdihs => "dihedrals",
        Table::Idihs | Table::Pidihs => "impropers",
        Table::Cmap => "cmaps",
    }
}

/// The table `funct` of `kind` looks its types up in, or why it has none.
fn table_of(kind: Kind, funct: u32) -> Result<Table, String> {
    Ok(match (kind, funct) {
        (Kind::Bond, 1 | 3) => Table::Bond(funct),
        (Kind::Angle, 1 | 5) => Table::Angle(funct),
        (Kind::Dihedral, 1 | 9) => Table::Pdihs,
        (Kind::Dihedral, 2) => Table::Idihs,
        (Kind::Dihedral, 3) => Table::Rbdihs,
        (Kind::Dihedral, 4) => Table::Pidihs,
        (Kind::Dihedral, 5) => Table::Fourdihs,
        // The conversion names what is supported.
        _ => return Err(convert(kind, funct, &[]).err().unwrap_or_default()),
    })
}

impl Directives {
    /// The type GROMACS's own lookup gives atoms of bond types `classes` in
    /// `table` (module docs).
    fn lookup(&self, table: Table, classes: &[&str]) -> Option<&str> {
        let entries = self.lookup.get(&table)?;
        let reversed: Vec<&str> = classes.iter().rev().copied().collect();
        let exact =
            |labels: &[String], atoms: &[&str]| labels.iter().zip(atoms).all(|(l, a)| l == a);
        match table {
            // Wildcards allowed: the first row (each stored forward, then
            // reversed) with the most non-wildcard matches.
            Table::Pdihs | Table::Idihs | Table::Pidihs | Table::Rbdihs => {
                let count = |labels: &[String], atoms: &[&str]| -> Option<usize> {
                    let mut n = 0;
                    for (l, a) in labels.iter().zip(atoms) {
                        if l.is_empty() {
                            continue;
                        }
                        if l != a {
                            return None;
                        }
                        n += 1;
                    }
                    Some(n)
                };
                let mut best: Option<(usize, &str)> = None;
                for e in entries {
                    for atoms in [classes, &reversed[..]] {
                        if let Some(n) = count(&e.labels, atoms)
                            && best.is_none_or(|(b, _)| n > b)
                        {
                            best = Some((n, &e.name));
                        }
                    }
                }
                best.map(|(_, name)| name)
            }
            Table::Cmap => entries
                .iter()
                .find(|e| exact(&e.labels, classes))
                .map(|e| e.name.as_str()),
            _ => entries
                .iter()
                .find(|e| exact(&e.labels, classes) || exact(&e.labels, &reversed))
                .map(|e| e.name.as_str()),
        }
    }
}

/// The frame of the molecule rows in `rows`, typed against `d`; types with
/// parameters of their own are added to `d.ff`.
pub(super) fn build(rows: &[Row], d: &mut Directives) -> Result<Frame, String> {
    let mut moltypes: Vec<MolType> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut current: Option<usize> = None;
    let mut molecules: Vec<(usize, usize)> = Vec::new();
    let mut own = OwnTypes::default();

    for row in rows {
        let section = row.section.as_str();
        match section {
            "moleculetype" => {
                let cols = row.cols();
                let [name, nrexcl] = cols[..] else {
                    return Err(row.err("expected `name nrexcl`"));
                };
                let nrexcl = nrexcl
                    .parse::<usize>()
                    .map_err(|_| row.err(&format!("nrexcl '{nrexcl}' is not a count")))?;
                if index.contains_key(name) {
                    return Err(row.err(&format!("molecule type '{name}' is defined twice")));
                }
                index.insert(name.to_owned(), moltypes.len());
                current = Some(moltypes.len());
                moltypes.push(MolType {
                    name: name.to_owned(),
                    nrexcl,
                    ..MolType::default()
                });
            }
            "atoms" | "bonds" | "pairs" | "angles" | "dihedrals" | "cmap" | "exclusions"
            | "constraints" | "settles" => {
                let mt = current
                    .map(|i| &mut moltypes[i])
                    .ok_or_else(|| row.err("a molecule row outside any [ moleculetype ]"))?;
                molecule_row(row, mt, d, &mut own)?;
            }
            "system" => {}
            "molecules" => {
                let cols = row.cols();
                let [name, count] = cols[..] else {
                    return Err(row.err("expected `name count`"));
                };
                let mt = *index
                    .get(name)
                    .ok_or_else(|| row.err(&format!("molecule type '{name}' is not defined")))?;
                let count = count
                    .parse::<usize>()
                    .map_err(|_| row.err(&format!("count '{count}' is not a count")))?;
                molecules.push((mt, count));
            }
            _ => {}
        }
    }
    assemble(&moltypes, &molecules)
}

/// The 0-based molecule atom of a 1-based index token.
fn atom_index(row: &Row, mt: &MolType, tok: &str) -> Result<usize, String> {
    let i = tok
        .parse::<usize>()
        .map_err(|_| row.err(&format!("atom index '{tok}' is not a positive integer")))?;
    if i == 0 || i > mt.atoms.len() {
        return Err(row.err(&format!(
            "atom {i} is not one of the {} atoms of '{}' read so far",
            mt.atoms.len(),
            mt.name
        )));
    }
    Ok(i - 1)
}

fn numbers(row: &Row, toks: &[&str]) -> Result<Vec<f64>, String> {
    toks.iter()
        .map(|tok| row.number(tok, &format!("parameter '{tok}'")))
        .collect()
}

/// One row of a molecule section into `mt`.
fn molecule_row(
    row: &Row,
    mt: &mut MolType,
    d: &mut Directives,
    own: &mut OwnTypes,
) -> Result<(), String> {
    let cols = row.cols();
    match row.section.as_str() {
        "atoms" => {
            if cols.len() < 6 {
                return Err(row.err("expected `nr type resnr residue atom cgnr [charge [mass]]`"));
            }
            if cols.len() > 8 {
                return Err(row.err(
                    "B-state columns (typeB chargeB massB) are free-energy perturbation, which \
                     has no form in the IR",
                ));
            }
            let nr = atom_number(row, cols[0])?;
            if nr != mt.atoms.len() + 1 {
                return Err(row.err(&format!(
                    "atom {nr} out of order: GROMACS numbers atoms 1, 2, … (expected {})",
                    mt.atoms.len() + 1
                )));
            }
            let atype = cols[1];
            let &(mass, charge) = d
                .atom_defaults
                .get(atype)
                .ok_or_else(|| row.err(&format!("atom type '{atype}' is not defined")))?;
            let resnr = cols[2].parse::<Idx>().map_err(|_| {
                row.err(&format!(
                    "resnr '{}' is not a non-negative integer",
                    cols[2]
                ))
            })?;
            mt.atoms.push(MolAtom {
                atype: atype.to_owned(),
                resnr,
                res_name: cols[3].to_owned(),
                name: cols[4].to_owned(),
                charge: match cols.get(6) {
                    Some(q) => row.number(q, "charge")?,
                    None => charge,
                },
                mass: match cols.get(7) {
                    Some(m) => row.number(m, "mass")?,
                    None => mass,
                },
            });
        }
        "bonds" | "angles" | "dihedrals" | "cmap" => {
            let (kind, n) = match row.section.as_str() {
                "bonds" => (Some(Kind::Bond), 2),
                "angles" => (Some(Kind::Angle), 3),
                "dihedrals" => (Some(Kind::Dihedral), 4),
                _ => (None, 5),
            };
            if cols.len() <= n {
                return Err(row.err(&format!("expected {n} atoms and a function code")));
            }
            let atoms = cols[..n]
                .iter()
                .map(|tok| atom_index(row, mt, tok))
                .collect::<Result<Vec<usize>, String>>()?;
            let funct: u32 = cols[n]
                .parse()
                .map_err(|_| row.err(&format!("function code '{}' is not an integer", cols[n])))?;
            let owned: Vec<String> = atoms
                .iter()
                .map(|&i| d.classes[&mt.atoms[i].atype].clone())
                .collect();
            let classes: Vec<&str> = owned.iter().map(String::as_str).collect();
            let values = numbers(row, &cols[n + 1..])?;
            let (table, type_name) = match kind {
                None => {
                    if funct != 1 {
                        return Err(row.err(&format!(
                            "function code {funct} is not supported (supported: 1)"
                        )));
                    }
                    if !values.is_empty() {
                        return Err(row.err("a cmap row takes its grid from [ cmaptypes ]"));
                    }
                    (Table::Cmap, None)
                }
                Some(kind) => {
                    let table = table_of(kind, funct).map_err(|e| row.err(&e))?;
                    let own_type = if values.is_empty() {
                        None
                    } else {
                        let c = convert(kind, funct, &values).map_err(|e| row.err(&e))?;
                        Some(own.define(d, c.category, c.style, &classes, c.params, row)?)
                    };
                    (table, own_type)
                }
            };
            let type_name = match type_name {
                Some(name) => name,
                None => d
                    .lookup(table, &classes)
                    .ok_or_else(|| {
                        let directive = match kind {
                            Some(Kind::Bond) => "bondtypes",
                            Some(Kind::Angle) => "angletypes",
                            Some(Kind::Dihedral) => "dihedraltypes",
                            None => "cmaptypes",
                        };
                        row.err(&format!(
                            "no [ {directive} ] funct {funct} row matches the bond types {}",
                            classes.join(" ")
                        ))
                    })?
                    .to_owned(),
            };
            if kind == Some(Kind::Bond) {
                mt.chemical.push((atoms[0], atoms[1]));
            }
            let term = Term { atoms, type_name };
            match block_of(table) {
                "bonds" => mt.bonds.push(term),
                "angles" => mt.angles.push(term),
                "dihedrals" => mt.dihedrals.push(term),
                "impropers" => mt.impropers.push(term),
                _ => mt.cmaps.push(term),
            }
        }
        "pairs" => {
            if cols.len() < 3 {
                return Err(row.err("expected `i j funct [params]`"));
            }
            let (i, j) = (atom_index(row, mt, cols[0])?, atom_index(row, mt, cols[1])?);
            let defaults = d
                .defaults
                .ok_or_else(|| row.err("[ pairs ] needs [ defaults ] (fudgeQQ, gen-pairs)"))?;
            let values = numbers(row, &cols[3..])?;
            let lj = |sigma: f64, eps: f64| (eps / KCAL_TO_KJ.get(), sigma * NM_TO_ANGSTROM.get());
            let cells: Cells = match (cols[2], &values[..]) {
                ("1", []) => {
                    let (a, b) = (&mt.atoms[i].atype, &mt.atoms[j].atype);
                    let key = if a <= b {
                        (a.clone(), b.clone())
                    } else {
                        (b.clone(), a.clone())
                    };
                    if !defaults.gen_pairs && !d.pairtypes.contains(&key) {
                        return Err(row.err(&format!(
                            "gen-pairs is no and no [ pairtypes ] row prices {} {}",
                            key.0, key.1
                        )));
                    }
                    [None; 5]
                }
                ("1", &[sigma, eps]) => {
                    let (eps, sigma) = lj(sigma, eps);
                    [
                        Some(eps),
                        Some(sigma),
                        None,
                        Some(1.0),
                        Some(defaults.fudge_qq),
                    ]
                }
                ("2", &[fudge_qq, qi, qj, sigma, eps]) => {
                    let (eps, sigma) = lj(sigma, eps);
                    [
                        Some(eps),
                        Some(sigma),
                        Some(qi * qj),
                        Some(1.0),
                        Some(fudge_qq),
                    ]
                }
                ("1" | "2", _) => {
                    return Err(row.err(&format!(
                        "funct {} takes {} parameters, got {}",
                        cols[2],
                        if cols[2] == "1" { "0 or 2" } else { "5" },
                        values.len()
                    )));
                }
                (other, _) => {
                    return Err(row.err(&format!(
                        "function code {other} is not supported (supported: 1, 2)"
                    )));
                }
            };
            mt.pairs.push((i, j, cells, row.at.clone()));
        }
        "exclusions" => {
            let atoms = cols
                .iter()
                .map(|tok| atom_index(row, mt, tok))
                .collect::<Result<Vec<usize>, String>>()?;
            for &j in atoms.iter().skip(1) {
                mt.exclusions.push((atoms[0], j));
            }
        }
        "constraints" => {
            if cols.len() < 3 {
                return Err(row.err("expected `i j funct [b0]`"));
            }
            let (i, j) = (atom_index(row, mt, cols[0])?, atom_index(row, mt, cols[1])?);
            let funct = match cols[2] {
                "1" => 1,
                "2" => 2,
                other => {
                    return Err(row.err(&format!(
                        "function code {other} is not supported (supported: 1, 2)"
                    )));
                }
            };
            let r0 = match &numbers(row, &cols[3..])?[..] {
                [b0] => b0 * NM_TO_ANGSTROM.get(),
                [] => {
                    let (a, b) = (
                        d.classes[&mt.atoms[i].atype].as_str(),
                        d.classes[&mt.atoms[j].atype].as_str(),
                    );
                    d.constrainttypes
                        .iter()
                        .find(|(l, f, _)| {
                            *f == funct && ((l[0] == a && l[1] == b) || (l[0] == b && l[1] == a))
                        })
                        .map(|(_, _, r0)| *r0)
                        .ok_or_else(|| {
                            row.err(&format!(
                                "no [ constrainttypes ] funct {funct} row matches {a} {b}"
                            ))
                        })?
                }
                more => {
                    return Err(row.err(&format!(
                        "a constraint takes one length, got {}",
                        more.len()
                    )));
                }
            };
            if funct == 1 {
                mt.chemical.push((i, j));
            }
            mt.constraints.push((i, j, r0));
        }
        "settles" => {
            let [first, funct, doh, dhh] = cols[..] else {
                return Err(row.err("expected `i funct doh dhh`"));
            };
            if funct != "1" {
                return Err(row.err(&format!(
                    "function code {funct} is not supported (supported: 1)"
                )));
            }
            let o = atom_index(row, mt, first)?;
            if o + 2 >= mt.atoms.len() {
                return Err(row.err("a settle needs the two atoms after its oxygen"));
            }
            let (doh, dhh) = (
                row.number(doh, "doh")? * NM_TO_ANGSTROM.get(),
                row.number(dhh, "dhh")? * NM_TO_ANGSTROM.get(),
            );
            mt.constraints
                .extend([(o, o + 1, doh), (o, o + 2, doh), (o + 1, o + 2, dhh)]);
        }
        other => return Err(row.err(&format!("[ {other} ] is not a molecule section"))),
    }
    Ok(())
}

fn atom_number(row: &Row, tok: &str) -> Result<usize, String> {
    tok.parse::<usize>()
        .map_err(|_| row.err(&format!("atom number '{tok}' is not a positive integer")))
}

impl OwnTypes {
    /// The type a row with parameters of its own defines (one per distinct
    /// style, labels and parameters), named by its labels and `@gmx_<n>`.
    fn define(
        &mut self,
        d: &mut Directives,
        category: &str,
        style: &str,
        labels: &[&str],
        params: crate::ff::ir::Params,
        row: &Row,
    ) -> Result<String, String> {
        let key = format!("{category}\u{1}{style}\u{1}{labels:?}\u{1}{params:?}");
        if let Some(name) = self.names.get(&key) {
            return Ok(name.clone());
        }
        let n = self.names.len() + 1;
        let name = TypeName::join(labels)
            .and_then(|t| t.with_qualifier(&["gmx", &n.to_string()]))
            .map_err(|e| row.err(&e))?;
        d.ff.def_style(category, style, crate::ff::ir::Params::new())
            .and_then(|s| s.def_type(name.as_str(), labels, params))
            .map_err(|e| row.err(&e.to_string()))?;
        self.names.insert(key, name.as_str().to_owned());
        Ok(name.as_str().to_owned())
    }
}

/// The excluded pairs `(i < j)` of a molecule: within `nrexcl` chemical
/// bonds, and `[ exclusions ]`.
fn excluded(mt: &MolType) -> HashSet<(usize, usize)> {
    let n = mt.atoms.len();
    let mut adjacent = vec![Vec::new(); n];
    for &(i, j) in &mt.chemical {
        adjacent[i].push(j);
        adjacent[j].push(i);
    }
    let mut out = HashSet::new();
    for start in 0..n {
        let mut depth = vec![usize::MAX; n];
        depth[start] = 0;
        let mut queue = VecDeque::from([start]);
        while let Some(a) = queue.pop_front() {
            if depth[a] == mt.nrexcl {
                continue;
            }
            for &b in &adjacent[a] {
                if depth[b] == usize::MAX {
                    depth[b] = depth[a] + 1;
                    queue.push_back(b);
                }
            }
        }
        for (b, &k) in depth.iter().enumerate() {
            if b > start && k != usize::MAX {
                out.insert((start, b));
            }
        }
    }
    for &(i, j) in &mt.exclusions {
        if i != j {
            out.insert((i.min(j), i.max(j)));
        }
    }
    out
}

fn column<T: BlockDtype>(block: &mut Block, key: &str, values: Vec<T>) -> Result<(), String> {
    block
        .insert(key, Array1::from_vec(values).into_dyn(), None)
        .map_err(|e| e.to_string())
}

/// A relation block: `atomi`, `atomj`, … and `type` (`None` when empty).
fn relation(rows: Vec<(Vec<Idx>, String)>) -> Result<Option<Block>, String> {
    let Some(arity) = rows.first().map(|r| r.0.len()) else {
        return Ok(None);
    };
    let mut block = Block::new();
    for (k, key) in ["atomi", "atomj", "atomk", "atoml", "atomm"]
        .iter()
        .take(arity)
        .enumerate()
    {
        column(&mut block, key, rows.iter().map(|r| r.0[k]).collect())?;
    }
    column(
        &mut block,
        "type",
        rows.into_iter().map(|r| r.1).collect::<Vec<String>>(),
    )?;
    Ok(Some(block))
}

/// The frame of `molecules` (`(molecule type, count)` in order).
fn assemble(moltypes: &[MolType], molecules: &[(usize, usize)]) -> Result<Frame, String> {
    // Per molecule type: the pairs GROMACS prices, and its exclusions.
    let mut pair_lists: HashMap<usize, PairLists> = HashMap::new();
    for &(m, _) in molecules {
        if pair_lists.contains_key(&m) {
            continue;
        }
        let mt = &moltypes[m];
        let n = mt.atoms.len();
        if n > MAX_ATOMS_FOR_A_FULL_PAIR_LIST {
            return Err(format!(
                "molecule type '{}' has {n} atoms: its intramolecular pair list would be \
                 n(n-1)/2 rows, and above {MAX_ATOMS_FOR_A_FULL_PAIR_LIST} atoms that is not \
                 built",
                mt.name
            ));
        }
        let excl = excluded(mt);
        let mut list = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                if !excl.contains(&(i, j)) {
                    list.push((i, j, false, [None; 5]));
                }
            }
        }
        for (i, j, cells, at) in &mt.pairs {
            if !excl.contains(&((*i).min(*j), (*i).max(*j))) {
                return Err(format!(
                    "{at}: [ pairs ] {} {} of '{}' is not an excluded pair (beyond nrexcl {} \
                     and not in [ exclusions ]): GROMACS would price it twice",
                    i + 1,
                    j + 1,
                    mt.name,
                    mt.nrexcl
                ));
            }
            list.push((*i, *j, true, *cells));
        }
        let mut exclusions: Vec<(usize, usize)> = excl.into_iter().collect();
        exclusions.sort_unstable();
        pair_lists.insert(m, (list, exclusions));
    }

    let mut atoms = (
        Vec::<String>::new(),
        Vec::<F>::new(),
        Vec::<F>::new(),
        Vec::<String>::new(),
        Vec::<Idx>::new(),
        Vec::<String>::new(),
        Vec::<Idx>::new(),
    );
    let mut relations: HashMap<&str, Vec<(Vec<Idx>, String)>> = HashMap::new();
    let mut constraints = (Vec::<Idx>::new(), Vec::<Idx>::new(), Vec::<F>::new());
    let mut exclusions = (Vec::<Idx>::new(), Vec::<Idx>::new());
    let mut pairs = (
        Vec::<Idx>::new(),
        Vec::<Idx>::new(),
        Vec::<bool>::new(),
        Vec::<Cells>::new(),
    );
    let mut offset = 0usize;
    let mut mol_id: Idx = 0;
    for &(m, count) in molecules {
        let mt = &moltypes[m];
        let (list, excl) = &pair_lists[&m];
        for _ in 0..count {
            mol_id += 1;
            let at = |i: usize| (offset + i) as Idx;
            for a in &mt.atoms {
                atoms.0.push(a.atype.clone());
                atoms.1.push(a.charge);
                atoms.2.push(a.mass);
                atoms.3.push(a.name.clone());
                atoms.4.push(a.resnr);
                atoms.5.push(a.res_name.clone());
                atoms.6.push(mol_id);
            }
            for (block, terms) in [
                ("bonds", &mt.bonds),
                ("angles", &mt.angles),
                ("dihedrals", &mt.dihedrals),
                ("impropers", &mt.impropers),
                ("cmaps", &mt.cmaps),
            ] {
                let rows = relations.entry(block).or_default();
                for t in terms {
                    rows.push((
                        t.atoms.iter().map(|&i| at(i)).collect(),
                        t.type_name.clone(),
                    ));
                }
            }
            for &(i, j, r0) in &mt.constraints {
                constraints.0.push(at(i));
                constraints.1.push(at(j));
                constraints.2.push(r0);
            }
            for &(i, j) in excl {
                exclusions.0.push(at(i));
                exclusions.1.push(at(j));
            }
            for &(i, j, is_14, cells) in list {
                pairs.0.push(at(i));
                pairs.1.push(at(j));
                pairs.2.push(is_14);
                pairs.3.push(cells);
            }
            offset += mt.atoms.len();
        }
    }

    // GROMACS prices every pair of two molecules: up to the size a full list
    // is built for, the frame's list holds them too, so `compile` prices
    // what GROMACS does. (Above it, a neighbour list does: `compile_typed`.)
    if offset <= MAX_ATOMS_FOR_A_FULL_PAIR_LIST {
        let mol = &atoms.6;
        for a in 0..offset {
            for b in a + 1..offset {
                if mol[a] != mol[b] {
                    pairs.0.push(a as Idx);
                    pairs.1.push(b as Idx);
                    pairs.2.push(false);
                    pairs.3.push([None; 5]);
                }
            }
        }
    }
    let mut frame = Frame::new();
    if offset == 0 {
        return Ok(frame);
    }
    let mut block = Block::new();
    column(&mut block, "type", atoms.0)?;
    column(&mut block, "charge", atoms.1)?;
    column(&mut block, "mass", atoms.2)?;
    column(&mut block, "name", atoms.3)?;
    column(&mut block, "res_id", atoms.4)?;
    column(&mut block, "res_name", atoms.5)?;
    column(&mut block, "mol_id", atoms.6)?;
    frame.insert("atoms", block);
    for name in ["bonds", "angles", "dihedrals", "impropers", "cmaps"] {
        if let Some(block) = relation(relations.remove(name).unwrap_or_default())? {
            frame.insert(name, block);
        }
    }
    if !constraints.0.is_empty() {
        let mut block = Block::new();
        column(&mut block, "atomi", constraints.0)?;
        column(&mut block, "atomj", constraints.1)?;
        column(&mut block, "r0", constraints.2)?;
        frame.insert("constraints", block);
    }
    if !exclusions.0.is_empty() {
        let mut block = Block::new();
        column(&mut block, "atomi", exclusions.0)?;
        column(&mut block, "atomj", exclusions.1)?;
        frame.insert("exclusions", block);
    }
    if !pairs.0.is_empty() {
        let mut block = Block::new();
        column(&mut block, "atomi", pairs.0)?;
        column(&mut block, "atomj", pairs.1)?;
        column(&mut block, "is_14", pairs.2)?;
        for (c, key) in PAIR_OVERRIDE_COLUMNS.iter().enumerate() {
            let cells: Vec<Option<f64>> = pairs.3.iter().map(|row| row[c]).collect();
            if cells.iter().all(Option::is_none) {
                continue;
            }
            let values: Vec<F> = cells.iter().map(|v| v.unwrap_or(0.0)).collect();
            let valid: Vec<bool> = cells.iter().map(Option::is_some).collect();
            block
                .insert(
                    *key,
                    ArrayD::from_shape_vec(vec![values.len()], values)
                        .map_err(|e| e.to_string())?,
                    Some(valid),
                )
                .map_err(|e| e.to_string())?;
        }
        frame.insert("pairs", block);
    }
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::super::GromacsTopForcefieldReader;
    use molrs::core::Frame;

    /// Butane-ish C1-C2-C3-C4 with one H on C1, then a water; OPLS-style
    /// `[ atomtypes ]` with bond types.
    const DIRECTIVES: &str = "\
[ defaults ]
1  3  yes  0.5  0.8333
[ atomtypes ]
opls_135  CT  6  12.011  -0.18  A  0.35  0.276144
opls_140  HC  1   1.008   0.06  A  0.25  0.12552
OW        OW  8  15.999  -0.834 A  0.315  0.6364
HW        HW  1   1.008   0.417 A  0.0    0.0
[ bondtypes ]
HC  CT  1  0.109  284512.0
CT  CT  1  0.1529 224262.4
[ angletypes ]
CT  CT  CT  1  112.7  488.273
HC  CT  CT  1  110.7  313.800
[ dihedraltypes ]
X   CT  CT  X   9  0.0  1.0  3
HC  CT  CT  CT  9  0.0  2.0  3
HC  CT  CT  CT  9  180.0  0.5  1
CT  CT  CT  CT  3  2.9288  -1.4644  0.2092  -1.6736  0.0  0.0
[ constrainttypes ]
OW  HW  1  0.09572
";

    const MOLECULES: &str = "\
[ moleculetype ]
BUT  3
[ atoms ]
1  opls_135  1  BUT  C1  1
2  opls_135  1  BUT  C2  1  -0.12  12.011
3  opls_135  1  BUT  C3  1  -0.12
4  opls_135  1  BUT  C4  1
5  opls_140  1  BUT  H1  1
[ bonds ]
1  2  1
3  2  1
3  4  1
5  1  1
[ pairs ]
1  4  1
5  3  1  0.3  0.5
[ angles ]
1  2  3  1
2  3  4  1
5  1  2  1  111.0  300.0
[ dihedrals ]
1  2  3  4  3
5  1  2  3  9
4  3  2  1  9
[ moleculetype ]
SOL  2
[ atoms ]
1  OW  1  SOL  OW  1
2  HW  1  SOL  HW1 1
3  HW  1  SOL  HW2 1
[ settles ]
1  1  0.09572  0.15139
[ exclusions ]
1  2  3
2  1  3
3  1  2
[ system ]
test
[ molecules ]
BUT  1
SOL  2
";

    fn read(text: &str) -> (crate::ff::forcefield::ForceField, Frame) {
        GromacsTopForcefieldReader::new()
            .read_system(text.as_bytes())
            .unwrap_or_else(|e| panic!("read_system: {e}"))
    }

    fn read_err(text: &str) -> String {
        GromacsTopForcefieldReader::new()
            .read_system(text.as_bytes())
            .expect_err("expected Err")
            .to_string()
    }

    fn strings(frame: &Frame, block: &str, key: &str) -> Vec<String> {
        frame
            .get(block)
            .and_then(|b| b.get(key))
            .and_then(|c| c.get::<String>())
            .unwrap_or_else(|| panic!("{block}.{key}"))
            .iter()
            .cloned()
            .collect()
    }

    fn uints(frame: &Frame, block: &str, key: &str) -> Vec<u64> {
        frame
            .get(block)
            .and_then(|b| b.get(key))
            .and_then(|c| c.get::<u64>())
            .unwrap_or_else(|| panic!("{block}.{key}"))
            .iter()
            .copied()
            .collect()
    }

    fn floats(frame: &Frame, block: &str, key: &str) -> Vec<f64> {
        frame
            .get(block)
            .and_then(|b| b.get(key))
            .and_then(|c| c.get::<f64>())
            .unwrap_or_else(|| panic!("{block}.{key}"))
            .iter()
            .copied()
            .collect()
    }

    fn system() -> (crate::ff::forcefield::ForceField, Frame) {
        read(&format!("{DIRECTIVES}{MOLECULES}"))
    }

    #[test]
    fn atoms_are_the_molecules_in_order_with_type_defaults() {
        let (_, frame) = system();
        assert_eq!(strings(&frame, "atoms", "type").len(), 11);
        assert_eq!(
            uints(&frame, "atoms", "mol_id"),
            [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        );
        let q = floats(&frame, "atoms", "charge");
        // C1 takes its atom type's charge, C2 its own.
        assert_eq!(&q[..3], &[-0.18, -0.12, -0.12]);
        assert_eq!(floats(&frame, "atoms", "mass")[8], 15.999);
        assert_eq!(strings(&frame, "atoms", "name")[9], "HW1");
        assert_eq!(strings(&frame, "atoms", "res_name")[0], "BUT");
    }

    /// `3 2` matches `CT CT`; `5 1` matches `HC CT` written the other way.
    #[test]
    fn bonds_match_their_bond_types_either_way() {
        let (_, frame) = system();
        assert_eq!(
            strings(&frame, "bonds", "type"),
            ["CT-CT", "CT-CT", "CT-CT", "HC-CT"]
        );
        assert_eq!(uints(&frame, "bonds", "atomi"), [0, 2, 2, 4]);
    }

    /// A row with parameters of its own defines its own type, in molrs units.
    #[test]
    fn a_row_with_parameters_defines_its_own_type() {
        let (ff, frame) = system();
        let types = strings(&frame, "angles", "type");
        assert_eq!(types[..2], ["CT-CT-CT", "CT-CT-CT"]);
        assert_eq!(types[2], "HC-CT-CT@gmx_1");
        let p = ff
            .get_style("angle", "harmonic")
            .unwrap()
            .type_params("HC-CT-CT@gmx_1")
            .unwrap();
        assert_eq!(p.get("theta0").and_then(|v| v.as_num()), Some(111.0));
        assert!((p.get("k").and_then(|v| v.as_num()).unwrap() - 300.0 / 4.184 / 2.0).abs() < 1e-12);
    }

    /// `5 1 2 3` (HC CT CT CT) takes the specific funct-9 block (four
    /// matches) over `X CT CT X` (two), with both of its terms; `4 3 2 1`
    /// (all CT) takes the wildcard row; funct 3 its own table.
    #[test]
    fn dihedrals_take_the_most_specific_row_and_its_terms() {
        let (ff, frame) = system();
        assert_eq!(
            strings(&frame, "dihedrals", "type"),
            ["CT-CT-CT-CT", "HC-CT-CT-CT", "-CT-CT-"]
        );
        let p = ff
            .get_style("dihedral", "periodic")
            .unwrap()
            .type_params("HC-CT-CT-CT")
            .unwrap();
        assert_eq!(p.get("phase2").and_then(|v| v.as_num()), Some(180.0));
        assert!(
            ff.get_style("dihedral", "multi/harmonic")
                .unwrap()
                .type_params("CT-CT-CT-CT")
                .is_some()
        );
    }

    /// nrexcl 3 excludes every pair of the 5-atom chain but C4-H1 (4 bonds
    /// apart); the [ pairs ] rows are the 1-4 rows; water's are all excluded.
    #[test]
    fn pairs_are_what_gromacs_prices() {
        let (_, frame) = system();
        let (i, j) = (
            uints(&frame, "pairs", "atomi"),
            uints(&frame, "pairs", "atomj"),
        );
        let is_14: Vec<bool> = frame
            .get("pairs")
            .unwrap()
            .get("is_14")
            .unwrap()
            .get::<bool>()
            .unwrap()
            .iter()
            .copied()
            .collect();
        let rows: Vec<(u64, u64, bool)> = (0..i.len()).map(|r| (i[r], j[r], is_14[r])).collect();
        // Per molecule (the butane's; a water's are all excluded), then every
        // pair of two molecules.
        assert_eq!(rows[..3], [(3, 4, false), (0, 3, true), (4, 2, true)]);
        let mol = |a: u64| if a < 5 { 0 } else { (a - 5) / 3 + 1 };
        assert_eq!(rows.len(), 3 + 5 * 6 + 3 * 3);
        assert!(rows[3..].iter().all(|&(a, b, f)| !f && mol(a) != mol(b)));
        let pairs = frame.get("pairs").unwrap();
        // The second [ pairs ] row carries its own σ, ε, at full LJ weight,
        // fudgeQQ Coulomb; the first carries nothing.
        let eps = floats(&frame, "pairs", "epsilon");
        assert!((eps[2] - 0.5 / 4.184).abs() < 1e-15);
        assert_eq!(floats(&frame, "pairs", "sigma")[2], 3.0);
        assert_eq!(floats(&frame, "pairs", "lj_scale")[2], 1.0);
        assert_eq!(floats(&frame, "pairs", "coul_scale")[2], 0.8333);
        let valid = pairs.validity("epsilon").unwrap();
        assert_eq!(valid[..3], [false, false, true]);
        assert!(valid[3..].iter().all(|v| !v));
        assert!(pairs.get("charge_product").is_none());
    }

    #[test]
    fn exclusions_and_settles() {
        let (_, frame) = system();
        let (i, j) = (
            uints(&frame, "exclusions", "atomi"),
            uints(&frame, "exclusions", "atomj"),
        );
        // 9 of the chain (10 pairs, C4-H1 not) and 3 per water.
        assert_eq!(i.len(), 9 + 3 + 3);
        assert!((0..i.len()).all(|r| i[r] < j[r]));
        assert_eq!(uints(&frame, "constraints", "atomi"), [5, 5, 6, 8, 8, 9]);
        let r0 = floats(&frame, "constraints", "r0");
        assert!((r0[0] - 0.9572).abs() < 1e-12 && (r0[2] - 1.5139).abs() < 1e-12);
    }

    /// funct 2: its own fudgeQQ, charges, σ, ε.
    #[test]
    fn a_funct_2_pair_carries_its_charges() {
        let text = format!("{DIRECTIVES}{MOLECULES}")
            .replace("5  3  1  0.3  0.5", "5  3  2  0.5  0.2  -0.3  0.31  0.4");
        let (_, frame) = read(&text);
        assert!((floats(&frame, "pairs", "charge_product")[2] - (-0.06)).abs() < 1e-15);
        assert_eq!(floats(&frame, "pairs", "coul_scale")[2], 0.5);
        assert!((floats(&frame, "pairs", "sigma")[2] - 3.1).abs() < 1e-12);
    }

    #[test]
    fn a_constraint_takes_its_constrainttype() {
        let text = format!("{DIRECTIVES}{MOLECULES}").replace(
            "[ settles ]\n1  1  0.09572  0.15139\n",
            "[ constraints ]\n1  2  1\n1  3  1  0.1\n",
        );
        let (_, frame) = read(&text);
        let r0 = floats(&frame, "constraints", "r0");
        assert_eq!(r0.len(), 4);
        assert!((r0[0] - 0.9572).abs() < 1e-12 && (r0[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn refusals_name_what_and_where() {
        let base = format!("{DIRECTIVES}{MOLECULES}");
        let cases = [
            (
                base.replace("1  2  3  4  3", "1  2  3  4  5"),
                "no [ dihedraltypes ] funct 5",
            ),
            (
                base.replace("5  3  1  0.3  0.5", "4  5  1"),
                "not an excluded pair",
            ),
            (
                base.replace("1  opls_135  1  BUT  C1  1", "1  opls_999  1  BUT  C1  1"),
                "opls_999",
            ),
            (
                base.replace(
                    "4  opls_135  1  BUT  C4  1",
                    "4  opls_135  1  BUT  C4  1  0.0  1.0  opls_140",
                ),
                "B-state",
            ),
            (
                base.replace("BUT  1\nSOL  2\n", "BUT  1\nWAT  2\n"),
                "'WAT' is not defined",
            ),
            (
                base.replace("[ settles ]", "[ virtual_sites3 ]"),
                "virtual_sites3",
            ),
            (base.replace("2  1  3\n", "2  1  7\n"), "atom 7"),
            (
                base.replace("1  2  1\n3  2", "1  2  2\n3  2"),
                "function code 2",
            ),
            (base.replace("yes  0.5", "no  0.5"), "gen-pairs is no"),
        ];
        for (text, needle) in cases {
            let err = read_err(&text);
            assert!(err.contains(needle), "error should name `{needle}`: {err}");
        }
    }

    /// The force-field reader still refuses molecule sections; `read_system`
    /// reads them.
    #[test]
    fn only_read_system_reads_molecules() {
        use crate::io::reader::ForceFieldReader;
        let err = GromacsTopForcefieldReader::new()
            .read(format!("{DIRECTIVES}{MOLECULES}").as_bytes())
            .expect_err("molecules");
        assert!(err.to_string().contains("read_system"), "{err}");
    }
}
