//! MMFF94 atom-type assignment.
//!
//! Faithful port of `MMFFMolProperties::setMMFFHeavyAtomType` and
//! `MMFFMolProperties::setMMFFHydrogenType` from RDKit
//! `Code/GraphMol/ForceFieldHelpers/MMFF/AtomTyper.cpp`
//! (BSD-3, RDKit contributors). The element-by-element decision tree is
//! reproduced branch-for-branch; comments name the MMFF symbolic type for
//! each numeric assignment, mirroring the source.
//!
//! Bond-type queries (`getBondType()`) read the post-aromaticity bond
//! orders (ring bonds promoted to `Aromatic`); total-bond-order /
//! valence queries read the preserved Kekulé orders.

// The hydrogen / heavy passes walk atoms by position so that the public
// atom index matches molecule iteration order, mirroring the C++ loop.
#![allow(clippy::needless_range_loop)]

use crate::core::BondOrder;
use crate::perceive::mmff_aromaticity::MmffTopology;

/// Assign MMFF numeric atom types (1..=99) to every atom.
///
/// `0` means "no MMFF type" (unsupported atom); the caller turns any zero
/// into an error.
pub(crate) fn assign_atom_types(topo: &MmffTopology) -> Vec<u8> {
    let n = topo.n_atoms();
    // Pass 1: heavy atoms. Pass 2: hydrogens (need heavy types first).
    let mut types = vec![0u8; n];
    for i in 0..n {
        if topo.atno[i] != 1 {
            types[i] = heavy_atom_type(topo, i);
        }
    }
    for i in 0..n {
        if topo.atno[i] == 1 {
            types[i] = hydrogen_type(topo, i, &types);
        }
    }
    types
}

/// RDKit `isAtomNOxide`.
fn is_atom_n_oxide(topo: &MmffTopology, i: usize) -> bool {
    if topo.atno[i] != 7 || topo.total_degree(i) < 3 {
        return false;
    }
    topo.nbrs[i]
        .iter()
        .any(|&j| topo.atno[j] == 8 && topo.total_degree(j) == 1)
}

#[allow(clippy::too_many_lines)]
fn heavy_atom_type(topo: &MmffTopology, i: usize) -> u8 {
    let mut atom_type: u8 = 0;
    let atno = topo.atno[i];

    if topo.is_aromatic[i] {
        if topo.is_atom_in_aromatic_ring_of_size(i, 5) {
            let mut alpha_het: Vec<usize> = Vec::new();
            let mut beta_het: Vec<usize> = Vec::new();
            let mut is_alpha_os = false;
            let mut is_beta_os = false;
            let mut alpha_or_beta_in_same_ring = false;

            if atno == 6 || atno == 7 {
                for &nbr in &topo.nbrs[i] {
                    if !topo.is_atom_in_aromatic_ring_of_size(nbr, 5) {
                        continue;
                    }
                    if topo.atoms_in_same_ring_of_size(5, &[i, nbr])
                        && (topo.atno[nbr] == 8
                            || topo.atno[nbr] == 16
                            || (topo.atno[nbr] == 7
                                && topo.total_degree(nbr) == 3
                                && !is_atom_n_oxide(topo, nbr)))
                    {
                        alpha_het.push(nbr);
                    }
                    for &nbr2 in &topo.nbrs[nbr] {
                        if nbr2 == i {
                            continue;
                        }
                        if !topo.is_atom_in_aromatic_ring_of_size(nbr2, 5) {
                            continue;
                        }
                        if topo.atoms_in_same_ring_of_size(5, &[i, nbr2])
                            && (topo.atno[nbr2] == 8
                                || topo.atno[nbr2] == 16
                                || (topo.atno[nbr2] == 7
                                    && topo.total_degree(nbr2) == 3
                                    && !is_atom_n_oxide(topo, nbr2)))
                        {
                            beta_het.push(nbr2);
                        }
                    }
                }
                is_alpha_os = alpha_het
                    .iter()
                    .any(|&a| topo.atno[a] == 8 || topo.atno[a] == 16);
                is_beta_os = beta_het
                    .iter()
                    .any(|&a| topo.atno[a] == 8 || topo.atno[a] == 16);
                if !alpha_het.is_empty() && !beta_het.is_empty() {
                    'outer: for &a in &alpha_het {
                        for &b in &beta_het {
                            if topo.atoms_in_same_ring_of_size(5, &[a, b]) {
                                alpha_or_beta_in_same_ring = true;
                                break 'outer;
                            }
                        }
                    }
                }
            }

            match atno {
                6 => {
                    if beta_het.is_empty() {
                        let mut n_n = 0;
                        let mut n_formal_charge = 0;
                        let mut n_in_arom5 = 0;
                        let mut n_in_arom6 = 0;
                        for &nbr in &topo.nbrs[i] {
                            if topo.atno[nbr] == 7 && topo.total_degree(nbr) == 3 {
                                n_n += 1;
                                if topo.formal_charge[nbr] > 0 && !is_atom_n_oxide(topo, nbr) {
                                    n_formal_charge += 1;
                                }
                                if topo.is_atom_in_aromatic_ring_of_size(nbr, 5) {
                                    n_in_arom5 += 1;
                                }
                                if topo.is_atom_in_aromatic_ring_of_size(nbr, 6) {
                                    n_in_arom6 += 1;
                                }
                            }
                        }
                        if (((n_n == 2) && n_in_arom5 > 0) || ((n_n == 3) && (n_in_arom5 == 2)))
                            && n_formal_charge > 0
                            && n_in_arom6 == 0
                        {
                            return 80; // CIM+
                        }
                    }
                    if alpha_het.len() == beta_het.len() {
                        let mut surrounded_by_benzene_c = true;
                        let mut surrounded_by_arom = true;
                        for &nbr in &topo.nbrs[i] {
                            if topo.atno[nbr] != 6 || !topo.is_atom_in_ring_of_size(nbr, 6) {
                                surrounded_by_benzene_c = false;
                            }
                            if topo.atoms_in_same_ring_of_size(5, &[i, nbr])
                                && !topo.is_aromatic[nbr]
                            {
                                surrounded_by_arom = false;
                            }
                        }
                        if (alpha_het.is_empty()
                            && beta_het.is_empty()
                            && !surrounded_by_benzene_c
                            && surrounded_by_arom)
                            || (!alpha_het.is_empty()
                                && !beta_het.is_empty()
                                && (!alpha_or_beta_in_same_ring || (!is_alpha_os && !is_beta_os)))
                        {
                            return 78; // C5
                        }
                    }
                    if !alpha_het.is_empty() && (beta_het.is_empty() || is_alpha_os) {
                        return 63; // C5A
                    }
                    if !beta_het.is_empty() && (alpha_het.is_empty() || is_beta_os) {
                        return 64; // C5B
                    }
                }
                7 => {
                    if is_atom_n_oxide(topo, i) {
                        return 82; // N5AX/N5BX/N5OX
                    }
                    if alpha_het.is_empty() && beta_het.is_empty() {
                        if topo.total_degree(i) == 3 {
                            return 39; // NPYL
                        }
                        return 76; // N5M
                    }
                    if topo.total_degree(i) == 3 && alpha_het.len() != beta_het.len() {
                        return 81; // NIM+/N5A+/N5B+/N5+
                    }
                    if !alpha_het.is_empty() && (beta_het.is_empty() || is_alpha_os) {
                        return 65; // N5A
                    }
                    if !beta_het.is_empty() && (alpha_het.is_empty() || is_beta_os) {
                        return 66; // N5B
                    }
                    if !alpha_het.is_empty() && !beta_het.is_empty() {
                        return 79; // N5
                    }
                }
                8 => return 59,  // OFUR
                16 => return 44, // STHI
                _ => {}
            }
        }

        if atom_type == 0 && topo.is_atom_in_aromatic_ring_of_size(i, 6) {
            match atno {
                6 => return 37, // CB
                7 => {
                    if is_atom_n_oxide(topo, i) {
                        return 69; // NPOX
                    }
                    if topo.total_degree(i) == 3 {
                        return 58; // NPD+
                    }
                    return 38; // NPYD
                }
                _ => {}
            }
        }
    }

    // Aliphatic heavy-atom types
    atom_type = aliphatic_heavy_type(topo, i);
    atom_type
}

/// double-bonded element to atom `i` (atomic number), 0 if none.
fn double_bonded_element(topo: &MmffTopology, i: usize) -> u8 {
    for (p, &j) in topo.nbrs[i].iter().enumerate() {
        if topo.nbr_order[i][p] == BondOrder::Double {
            return topo.atno[j];
        }
    }
    0
}

#[allow(clippy::too_many_lines)]
fn aliphatic_heavy_type(topo: &MmffTopology, i: usize) -> u8 {
    let atno = topo.atno[i];
    let td = topo.total_degree(i);
    let deg = topo.degree(i);
    let tbo = topo.total_bond_order(i);

    match atno {
        // Lithium
        3 if deg == 0 => return 92, // LI+
        // Carbon
        6 => {
            if td == 4 {
                if topo.is_atom_in_ring_of_size(i, 3) {
                    return 22; // CR3R
                }
                if topo.is_atom_in_ring_of_size(i, 4) {
                    return 20; // CR4R
                }
                return 1; // CR
            }
            if td == 3 {
                let mut n_n2 = 0;
                let mut n_n3 = 0;
                let mut n_o = 0;
                let mut n_s = 0;
                let dbe = double_bonded_element(topo, i);
                for (p, &nbr) in topo.nbrs[i].iter().enumerate() {
                    let is_double = topo.nbr_order[i][p] == BondOrder::Double;
                    if topo.total_degree(nbr) == 1 {
                        if topo.atno[nbr] == 8 {
                            n_o += 1;
                        } else if topo.atno[nbr] == 16 {
                            n_s += 1;
                        }
                    } else if topo.atno[nbr] == 7 {
                        if topo.total_degree(nbr) == 3 {
                            n_n3 += 1;
                        } else if topo.total_degree(nbr) == 2 && is_double {
                            n_n2 += 1;
                        }
                    }
                }
                if n_n3 >= 2 && n_n2 == 0 && dbe == 7 {
                    return 57; // CNN+/CGD+
                }
                if n_o == 2 || n_s == 2 {
                    return 41; // CO2M/CS2M
                }
                if topo.is_atom_in_ring_of_size(i, 4) && dbe == 6 {
                    return 30; // CR4E
                }
                if dbe == 7 || dbe == 8 || dbe == 15 || dbe == 16 {
                    return 3; // C=N/C=O/...
                }
                return 2; // C=C/CSP2
            }
            if td == 2 {
                return 4; // CSP/=C=
            }
            if td == 1 {
                return 60; // C%-
            }
        }
        // Nitrogen
        7 => {
            let mut n_term_o_bonded_to_n = 0;
            let mut is_nso2_or_nso3_or_ncn = false;
            for &nbr in &topo.nbrs[i] {
                if topo.atno[nbr] == 8 && topo.total_degree(nbr) == 1 {
                    n_term_o_bonded_to_n += 1;
                }
                if tbo >= 3 && (topo.atno[nbr] == 15 || topo.atno[nbr] == 16) {
                    let mut n_o_bonded_to_sp = 0;
                    for &nbr2 in &topo.nbrs[nbr] {
                        if topo.atno[nbr2] == 8 && topo.total_degree(nbr2) == 1 {
                            n_o_bonded_to_sp += 1;
                        }
                    }
                    if !is_nso2_or_nso3_or_ncn {
                        is_nso2_or_nso3_or_ncn = n_o_bonded_to_sp >= 2;
                    }
                }
            }
            if td == 4 {
                if is_atom_n_oxide(topo, i) {
                    return 68; // N3OX
                }
                return 34; // NR+
            }
            if td == 3 {
                if tbo >= 4 {
                    let mut double_bonded_cn = false;
                    for (p, &nbr) in topo.nbrs[i].iter().enumerate() {
                        if topo.nbr_order[i][p] == BondOrder::Double {
                            double_bonded_cn = topo.atno[nbr] == 7 || topo.atno[nbr] == 6;
                            if topo.atno[nbr] == 6 {
                                for &nbr2 in &topo.nbrs[nbr] {
                                    if !double_bonded_cn {
                                        break;
                                    }
                                    if nbr2 == i {
                                        continue;
                                    }
                                    double_bonded_cn =
                                        !(topo.atno[nbr2] == 7 && topo.total_degree(nbr2) == 3);
                                }
                            }
                        }
                    }
                    if n_term_o_bonded_to_n == 1 {
                        return 67; // N2OX
                    }
                    if n_term_o_bonded_to_n >= 2 {
                        return 45; // NO2/NO3
                    }
                    if double_bonded_cn {
                        return 54; // N+=C/N+=N
                    }
                }
                if tbo >= 3
                    && let Some(t) = nitrogen_3nbr_deloc_type(topo, i, &mut is_nso2_or_nso3_or_ncn)
                {
                    return t;
                }
            }
            if td == 2 {
                if tbo == 4 {
                    let is_isonitrile = topo.nbrs[i]
                        .iter()
                        .enumerate()
                        .any(|(p, _)| topo.nbr_order[i][p] == BondOrder::Triple);
                    if is_isonitrile {
                        return 61; // NR%
                    }
                    return 53; // =N=
                }
                if tbo == 3 {
                    let mut is_nitroso = false;
                    let mut is_imine_or_azo = false;
                    for (p, &nbr) in topo.nbrs[i].iter().enumerate() {
                        if topo.nbr_order[i][p] == BondOrder::Double {
                            is_nitroso = topo.atno[nbr] == 8 && n_term_o_bonded_to_n == 1;
                            is_imine_or_azo = topo.atno[nbr] == 6 || topo.atno[nbr] == 7;
                        }
                    }
                    if is_nitroso && !is_imine_or_azo {
                        return 46; // N=O
                    }
                    if is_imine_or_azo {
                        return 9; // N=C/N=N
                    }
                }
                if tbo >= 2 {
                    let mut is_nso = false;
                    for &nbr in &topo.nbrs[i] {
                        if is_nso {
                            break;
                        }
                        if topo.atno[nbr] == 16 {
                            let n_term_o_bonded_to_s = topo.nbrs[nbr]
                                .iter()
                                .filter(|&&n2| topo.atno[n2] == 8 && topo.total_degree(n2) == 1)
                                .count();
                            is_nso = n_term_o_bonded_to_s == 1;
                        }
                    }
                    if is_nso {
                        return 48; // NSO
                    }
                    if !is_nso2_or_nso3_or_ncn {
                        return 62; // NM
                    }
                }
            }
            if is_nso2_or_nso3_or_ncn {
                return 43; // NSO2/NSO3/NC%N
            }
            if td == 1 {
                let mut is_nsp = false;
                let mut is_nazt = false;
                for (p, &nbr) in topo.nbrs[i].iter().enumerate() {
                    if is_nsp || is_nazt {
                        break;
                    }
                    if topo.nbr_order[i][p] == BondOrder::Triple {
                        is_nsp = true;
                    }
                    if topo.atno[nbr] == 7 && topo.total_degree(nbr) == 2 {
                        for &nbr2 in &topo.nbrs[nbr] {
                            if is_nazt {
                                break;
                            }
                            is_nazt = (topo.atno[nbr2] == 7 && topo.total_degree(nbr2) == 2)
                                || (topo.atno[nbr2] == 6 && topo.total_degree(nbr2) == 3);
                        }
                    }
                }
                if is_nsp {
                    return 42; // NSP
                }
                if is_nazt {
                    return 47; // NAZT
                }
            }
            return 8; // NR
        }
        // Oxygen
        8 => return oxygen_type(topo, i),
        // Fluorine
        9 => {
            if deg == 1 {
                return 11;
            }
            if deg == 0 {
                return 89;
            }
        }
        // Sodium
        11 if deg == 0 => return 93, // NA+
        // Magnesium
        12 if deg == 0 => return 99, // MG+2
        // Silicon
        14 => return 19,
        // Phosphorus
        15 => {
            if td == 4 {
                return 25;
            }
            if td == 3 {
                return 26;
            }
            if td == 2 {
                return 75;
            }
        }
        // Sulfur
        16 => return sulfur_type(topo, i),
        // Chlorine
        17 => {
            if td == 4 {
                let n_o = topo.nbrs[i].iter().filter(|&&j| topo.atno[j] == 8).count();
                if n_o == 4 {
                    return 77; // CLO4
                }
            }
            if td == 1 {
                return 12;
            }
            if deg == 0 {
                return 90;
            }
        }
        // Potassium
        19 if deg == 0 => return 94, // K+
        // Calcium
        20 if deg == 0 => return 96, // CA+2
        // Iron
        26 if deg == 0 => {
            if topo.formal_charge[i] == 2 {
                return 87; // FE+2
            }
            if topo.formal_charge[i] == 3 {
                return 88; // FE+3
            }
        }
        // Copper
        29 if deg == 0 => {
            if topo.formal_charge[i] == 1 {
                return 97; // CU+1
            }
            if topo.formal_charge[i] == 2 {
                return 98; // CU+2
            }
        }
        // Zinc
        30 if deg == 0 => return 95, // ZN+2
        // Bromine
        35 => {
            if deg == 1 {
                return 13;
            }
            if deg == 0 {
                return 91;
            }
        }
        // Iodine
        53 if deg == 1 => return 14,
        _ => {}
    }
    0
}

/// The 3-neighbour, total-bond-order >= 3 nitrogen branch (deloc. lone pair,
/// amide, aniline, …). Mutates the sulfonamide/cyano flag like the source.
#[allow(clippy::too_many_lines)]
fn nitrogen_3nbr_deloc_type(
    topo: &MmffTopology,
    i: usize,
    is_nso2_or_nso3_or_ncn: &mut bool,
) -> Option<u8> {
    let mut is_nco_or_ncs = false;
    let mut is_ncn_plus = false;
    let mut is_ngd_plus = false;
    let mut is_nnn_or_nnc = false;
    let mut is_nbr_c = false;
    let mut is_nbr_benzene_c = false;
    let mut element_double_bonded_to_c: u8 = 0;
    let mut element_triple_bonded_to_c: u8 = 0;
    // The C++ source reads nObondedToC / nSbondedToC *after* the neighbour
    // loop; they retain the values computed for the last carbon neighbour.
    let mut n_o_bonded_to_c = 0usize;
    let mut n_s_bonded_to_c = 0usize;

    for &nbr in &topo.nbrs[i] {
        if topo.atno[nbr] == 6 {
            is_nbr_c = true;
            if topo.is_aromatic[nbr] && topo.is_atom_in_ring_of_size(nbr, 6) {
                is_nbr_benzene_c = true;
            }
            let mut n_n2_bonded_to_c = 0;
            let mut n_n3_bonded_to_c = 0;
            let mut n_formal_charge = 0;
            let mut n_in_arom6 = 0;
            n_o_bonded_to_c = 0;
            n_s_bonded_to_c = 0;
            element_double_bonded_to_c = 0;
            element_triple_bonded_to_c = 0;
            for (p2, &nbr2) in topo.nbrs[nbr].iter().enumerate() {
                let bo = topo.nbr_order[nbr][p2];
                if bo == BondOrder::Double && (topo.atno[nbr2] == 8 || topo.atno[nbr2] == 16) {
                    is_nco_or_ncs = true;
                }
                if bo == BondOrder::Double
                    || (bo == BondOrder::Aromatic
                        && (topo.atno[nbr2] == 6
                            || (topo.atno[nbr2] == 7
                                && topo.rings.n_atom_rings(topo.id(nbr2)) == 1)))
                {
                    element_double_bonded_to_c = topo.atno[nbr2];
                }
                if bo == BondOrder::Triple {
                    element_triple_bonded_to_c = topo.atno[nbr2];
                }
                if topo.atno[nbr2] == 7 && topo.total_degree(nbr2) == 3 {
                    if topo.formal_charge[nbr2] == 1 {
                        n_formal_charge += 1;
                    }
                    if topo.is_atom_in_aromatic_ring_of_size(nbr, 6) {
                        n_in_arom6 += 1;
                    }
                    let n_o_bonded_to_n3 = topo.nbrs[nbr2]
                        .iter()
                        .filter(|&&n3| topo.atno[n3] == 8)
                        .count();
                    if n_o_bonded_to_n3 < 2 {
                        n_n3_bonded_to_c += 1;
                    }
                }
                if topo.atno[nbr2] == 7
                    && topo.total_degree(nbr2) == 2
                    && (bo == BondOrder::Double || bo == BondOrder::Aromatic)
                {
                    n_n2_bonded_to_c += 1;
                }
                if topo.is_aromatic[nbr2] {
                    if topo.atno[nbr2] == 8 {
                        n_o_bonded_to_c += 1;
                    }
                    if topo.atno[nbr2] == 16 {
                        n_s_bonded_to_c += 1;
                    }
                }
            }
            if element_double_bonded_to_c == 7 {
                if n_n3_bonded_to_c == 2
                    && n_n2_bonded_to_c == 0
                    && n_formal_charge > 0
                    && n_in_arom6 == 0
                    && topo.total_degree(nbr) < 4
                {
                    is_ncn_plus = true;
                }
                if n_n3_bonded_to_c == 3 {
                    is_ngd_plus = true;
                }
            }
        }
        if topo.atno[nbr] == 7 {
            for (p2, &nbr2) in topo.nbrs[nbr].iter().enumerate() {
                if topo.nbr_order[nbr][p2] == BondOrder::Double {
                    if topo.atno[nbr2] == 6 {
                        let mut n_n = 0;
                        let mut n_o = 0;
                        let mut n_s = 0;
                        for &nbr3 in &topo.nbrs[nbr2] {
                            if nbr3 == nbr {
                                continue;
                            }
                            match topo.atno[nbr3] {
                                7 => n_n += 1,
                                8 => n_o += 1,
                                16 => n_s += 1,
                                _ => {}
                            }
                        }
                        if n_o == 0 && n_s == 0 && n_n == 0 && !is_nbr_benzene_c {
                            is_nnn_or_nnc = true;
                        }
                    }
                    if topo.atno[nbr2] == 7 && !is_nbr_benzene_c {
                        is_nnn_or_nnc = true;
                    }
                }
            }
        }
    }

    if is_nbr_c {
        if element_triple_bonded_to_c == 7 {
            *is_nso2_or_nso3_or_ncn = true;
        }
        if is_ncn_plus {
            return Some(55); // NCN+
        }
        if is_ngd_plus {
            return Some(56); // NGD+
        }
        if (!is_nco_or_ncs && !*is_nso2_or_nso3_or_ncn)
            && (((n_o_bonded_to_c == 0) && (n_s_bonded_to_c == 0) && is_nbr_benzene_c)
                || (element_double_bonded_to_c == 6
                    || element_double_bonded_to_c == 7
                    || element_double_bonded_to_c == 15
                    || element_triple_bonded_to_c == 6))
        {
            return Some(40); // NC=C/NC=N/NC=P/NC%C
        }
    }
    if !*is_nso2_or_nso3_or_ncn && (is_nco_or_ncs || is_nnn_or_nnc) {
        return Some(10); // NC=O/NC=S/NN=C/NN=N
    }
    None
}

#[allow(clippy::too_many_lines)]
fn oxygen_type(topo: &MmffTopology, i: usize) -> u8 {
    let td = topo.total_degree(i);
    if td == 3 {
        return 49; // O+
    }
    if td == 2 {
        if topo.total_bond_order(i) == 3 {
            return 51; // O=+
        }
        let n_h = topo.n_h_neighbors(i);
        if n_h == 2 {
            return 70; // OH2
        }
        return 6; // OC=O/... generic divalent O
    }
    if topo.degree(i) <= 1 {
        let mut n_n_bonded = 0usize; // secondary N on the C/N/S neighbour
        let mut n_o_bonded = 0usize; // terminal O on the neighbour
        let mut n_s_bonded = 0usize; // terminal S on the neighbour
        let mut is_oxide_o_bonded_to_h = topo.n_h_neighbors(i) > 0;
        let mut is_carboxylate_o = false;
        let mut is_carbonyl_o = false;
        let mut is_oxide_o_bonded_to_c = false;
        let mut is_nitroso_o = false;
        let mut is_oxide_o_bonded_to_n = false;
        let mut is_n_oxide_o = false;
        let mut is_nitro_o = false;
        let mut is_thiosulfinate_o = false;
        let mut is_sulfate_o = false;
        let mut is_sulfoxide_o = false;
        let mut is_phosphate_or_perchlorate_o = false;

        for (p, &nbr) in topo.nbrs[i].iter().enumerate() {
            // mirror the C++ short-circuit on the loop guard
            if is_oxide_o_bonded_to_c
                || is_oxide_o_bonded_to_n
                || is_oxide_o_bonded_to_h
                || is_carboxylate_o
                || is_nitro_o
                || is_n_oxide_o
                || is_thiosulfinate_o
                || is_sulfate_o
                || is_phosphate_or_perchlorate_o
                || is_carbonyl_o
                || is_nitroso_o
                || is_sulfoxide_o
            {
                break;
            }
            let bo = topo.nbr_order[i][p];
            let na = topo.atno[nbr];
            if na == 6 || na == 7 || na == 16 {
                for &nbr2 in &topo.nbrs[nbr] {
                    if topo.atno[nbr2] == 7 && topo.total_degree(nbr2) == 2 {
                        n_n_bonded += 1;
                    }
                    if topo.atno[nbr2] == 8 && topo.total_degree(nbr2) == 1 {
                        n_o_bonded += 1;
                    }
                    if topo.atno[nbr2] == 16 && topo.total_degree(nbr2) == 1 {
                        n_s_bonded += 1;
                    }
                }
            }
            is_oxide_o_bonded_to_h = na == 1;
            if na == 6 {
                is_carboxylate_o = n_o_bonded == 2;
                is_carbonyl_o = bo == BondOrder::Double;
                is_oxide_o_bonded_to_c = bo == BondOrder::Single && n_o_bonded == 1;
            }
            if na == 7 {
                is_nitroso_o = bo == BondOrder::Double;
                if bo == BondOrder::Single && n_o_bonded == 1 {
                    is_oxide_o_bonded_to_n =
                        topo.total_degree(nbr) == 2 || topo.total_bond_order(nbr) == 3;
                    is_n_oxide_o = topo.total_bond_order(nbr) == 4;
                }
                is_nitro_o = n_o_bonded >= 2;
            }
            if na == 16 {
                is_thiosulfinate_o = n_s_bonded == 1;
                is_sulfate_o = bo == BondOrder::Single
                    || (bo == BondOrder::Double && (n_o_bonded + n_n_bonded) > 1);
                is_sulfoxide_o = bo == BondOrder::Double && (n_o_bonded + n_n_bonded) == 1;
            }
            is_phosphate_or_perchlorate_o = na == 15 || na == 17;
        }
        if is_oxide_o_bonded_to_c || is_oxide_o_bonded_to_n || is_oxide_o_bonded_to_h {
            return 35; // OM/OM2
        }
        if is_carboxylate_o
            || is_nitro_o
            || is_n_oxide_o
            || is_thiosulfinate_o
            || is_sulfate_o
            || is_phosphate_or_perchlorate_o
        {
            return 32; // O2CM/O2N/...
        }
        if is_carbonyl_o || is_nitroso_o || is_sulfoxide_o {
            return 7; // O=C/O=N/O=S
        }
    }
    0
}

fn sulfur_type(topo: &MmffTopology, i: usize) -> u8 {
    let td = topo.total_degree(i);
    if td == 3 || td == 4 {
        let mut n_o_or_n_bonded = 0;
        let mut n_s_bonded = 0;
        let mut is_c_double_bonded = false;
        for (p, &nbr) in topo.nbrs[i].iter().enumerate() {
            if topo.atno[nbr] == 6 && topo.nbr_order[i][p] == BondOrder::Double {
                is_c_double_bonded = true;
            }
            if (topo.degree(nbr) == 1 && topo.atno[nbr] == 8)
                || (topo.total_degree(nbr) == 2 && topo.atno[nbr] == 7)
            {
                n_o_or_n_bonded += 1;
            }
            if topo.degree(nbr) == 1 && topo.atno[nbr] == 16 {
                n_s_bonded += 1;
            }
        }
        if (td == 3 && n_o_or_n_bonded == 2 && is_c_double_bonded) || td == 4 {
            return 18; // =SO2
        }
        if (n_o_or_n_bonded > 0 && n_s_bonded > 0) || (n_o_or_n_bonded == 2 && !is_c_double_bonded)
        {
            return 73; // SSOM
        }
        return 17; // S=O/>S=N
    }
    if td == 2 {
        let is_o_double = topo.nbrs[i]
            .iter()
            .enumerate()
            .any(|(p, &nbr)| topo.atno[nbr] == 8 && topo.nbr_order[i][p] == BondOrder::Double);
        if is_o_double {
            return 74; // =S=O
        }
        return 15; // S
    }
    if topo.degree(i) == 1 {
        let mut n_term_s_bonded_to_nbr = 0;
        let mut is_c_double = false;
        for (p, &nbr) in topo.nbrs[i].iter().enumerate() {
            for &nbr2 in &topo.nbrs[nbr] {
                if topo.atno[nbr2] == 16 && topo.total_degree(nbr2) == 1 {
                    n_term_s_bonded_to_nbr += 1;
                }
            }
            if topo.atno[nbr] == 6 && topo.nbr_order[i][p] == BondOrder::Double {
                is_c_double = true;
            }
        }
        if is_c_double && n_term_s_bonded_to_nbr != 2 {
            return 16; // S=C
        }
        return 72; // S-P/SM/SSMO
    }
    0
}

/// Port of `setMMFFHydrogenType`.
fn hydrogen_type(topo: &MmffTopology, i: usize, heavy: &[u8]) -> u8 {
    let mut atom_type: u8 = 0;
    for &nbr in &topo.nbrs[i] {
        match topo.atno[nbr] {
            6 | 14 => atom_type = 5, // HC/HSI
            7 => {
                atom_type = match heavy[nbr] {
                    8 | 39 | 62 | 67 | 68 => 23, // HNR/HPYL/HNOX
                    34 | 54 | 55 | 56 | 58 | 81 => 36,
                    9 => 27,
                    _ => 28,
                };
            }
            8 => {
                atom_type = match heavy[nbr] {
                    49 => 50, // HO+
                    51 => 52, // HO=+
                    70 => 31, // HOH
                    6 => hydroxyl_on_o6(topo, i, nbr),
                    _ => 21,
                };
            }
            15 | 16 => atom_type = 71, // HP/HS
            _ => {}
        }
    }
    atom_type
}

fn hydroxyl_on_o6(topo: &MmffTopology, h: usize, o: usize) -> u8 {
    let mut is_hocc_or_hocn = false;
    let mut is_hoco = false;
    let mut is_hop = false;
    let mut is_hos = false;
    for &nbr2 in &topo.nbrs[o] {
        if topo.atno[nbr2] == 6 {
            for (p3, &nbr3) in topo.nbrs[nbr2].iter().enumerate() {
                if nbr3 == o {
                    continue;
                }
                let bo = topo.nbr_order[nbr2][p3];
                if (topo.atno[nbr3] == 6 || topo.atno[nbr3] == 7)
                    && (bo == BondOrder::Double || bo == BondOrder::Aromatic)
                {
                    is_hocc_or_hocn = true;
                }
                if topo.atno[nbr3] == 8 && bo == BondOrder::Double {
                    is_hoco = true;
                }
            }
        }
        if topo.atno[nbr2] == 15 {
            is_hop = true;
        }
        if topo.atno[nbr2] == 16 {
            is_hos = true;
        }
    }
    let _ = h;
    if is_hoco || is_hop {
        return 24; // HOCO
    }
    if is_hocc_or_hocn {
        return 29; // HOCC/HOCN
    }
    if is_hos {
        return 33; // HOS
    }
    21 // HO/HOR
}
