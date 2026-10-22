//! OPLS-AA SMARTS typing rules — molrs-owned, hand-maintained.

use crate::ff::params::OplsRuleRow;

/// The 160 molrs-owned OPLS-AA typing rules, in the order of their atom types.
///
/// These rules are **not** GROMACS's: GROMACS `oplsaa.ff` carries no typing
/// rules at all. They are molrs's own, and `cargo mrs-gen-opls` never reads or
/// writes this file — regenerating the OPLS-AA parameter tables leaves every rule here
/// untouched. Edit them by hand.
///
/// Each [`OplsRuleRow`] names the `opls_NNN` type it assigns (an
/// [`OplsAtomRow`](super::OplsAtomRow) of [`OPLSAA_ATOMS`](super::OPLSAA_ATOMS)), the
/// SMARTS `def` that assigns it, and the types it `overrides` when both match.
/// The shipped typifier joins the two tables by name
/// (`ff::typifier::opls::embedded::try_typing_meta`). The trailing comment of
/// each rule is the type's GROMACS v2026.3 `atomtypes.atp` description.
///
/// # Conventions
///
/// Every `def` is Daylight SMARTS (Daylight Theory Manual ch. 4), read by the
/// standard molrs matcher against a graph with explicit hydrogens whose
/// aromaticity the typifier perceives on a private copy first:
///
/// 1. **Explicit bonds.** Every bond is written `-`, `=`, `#` or `:`. An
///    unmarked bond would mean single-or-aromatic, which is what made the
///    foyer-dialect rules miss every `=` bond. `~` (any bond) appears only
///    where a type deliberately spans resonance forms — carboxylate O
///    (opls_271/272), sulfoxide S=O (opls_496/497) and nitro O
///    (opls_760/761/767) — and each such rule says so in its comment.
/// 2. **Hydrogens.** A hydrogen atom is `[#1]`; `H<n>` appears only as a
///    hydrogen count. Foyer's `[!H]` is `[!#1]` (opls_154, opls_207, opls_467)
///    and its `[Cl,C,H]` is `[Cl,#6,#1]` (opls_152/153).
/// 3. **Aromatic case.** Aromatic atoms are lowercase, aliphatic uppercase: the
///    typed atom and every `%opls_NNN` neighbour carry the case of their
///    type's chemistry (opls_264 `[Cl;X1]-[c;%opls_263]`, opls_719
///    `F-[c;%opls_718]`, opls_534 `[#1]-[c;%opls_531]`, opls_167
///    `[O;X2](-[#1])-[c;%opls_166]`; pyridine opls_520–526, pyrimidine
///    opls_530–536 and pyrrole opls_542–547 as `n` / `c` rings with `:`
///    bonds; opls_678/679 on the pyrrole `c`). A substituent carbon the foyer
///    rules wrote as a bare `C` is `[#6]`: foyer matched a bare symbol by
///    element alone, so an aromatic neighbour was always admitted there
///    (opls_917, N-methylaniline's ring carbon, depends on it: it reads an
///    opls_901 N, and opls_901 must admit the aromatic neighbour).
/// 4. **Monatomic ions.** Written `X0`: opls_401 `[Cl;X0]`, opls_406
///    `[Li;X0]`.
/// 5. **Preconditions.** Hydrogens are explicit atoms; `r<n>` is the size of
///    the atom's *smallest* ring (RDKit/Daylight), not any chordless cycle.
///
/// # Differences from foyer's rules
///
/// 157 of the `def` / `overrides` pairs derive from foyer's OPLS-AA rules,
/// written to the conventions above. No rule declares an explicit priority.
/// Every foyer override is kept; the differences are:
///
/// - **Added:** opls_150 (diene `=CH-CH=`) `[C;X3;H1](=[C;X3])-[C;X3]=[C;X3]`,
///   overriding opls_142; opls_178 (diene `=CR-CR=`)
///   `[C;X3;H0](=[C;X3])(-[#6])-[C;X3]=[C;X3]`, overriding opls_141. Both are
///   for conjugated dienes only, so methacrylate's α-carbon stays opls_141.
/// - **Added:** opls_928 (alkyne C2 whose R carries one H)
///   `[C;X2](-[C;X4;H1])#[#6]`, overriding nothing. opls_927 does not override
///   opls_928: by their `.atp` descriptions opls_927 (R with 2 or 3 H,
///   `-[#6](-[#1])-[#1]`) and opls_928 (R with 1 H) are disjoint, so they are
///   never candidates on the same atom. R is sp3 (`X4`) because the
///   description sends C3 to the alkane types opls_135–139.
/// - **Dropped term:** opls_542 (pyrrole N) does not end in a bare `H`, which
///   the foyer rule bonded to a ring carbon rather than to N; the ring
///   `[n;X3;r5]1:[c;X3;r5]:…:1` alone identifies it.
///
/// Overrides between an aliphatic and an aromatic rule (opls_145 over
/// opls_141/142, opls_522/523/533/544 over opls_142, the aromatic H types over
/// opls_144) and those of opls_151/264 over the ion opls_401 cannot meet a
/// co-matching candidate under the case and `X0` conventions; they are kept,
/// harmless, as foyer's overrides.
#[rustfmt::skip]
pub const OPLSAA_TYPING: &[OplsRuleRow] = &[
    OplsRuleRow { name: "opls_135", def: "[C;X4](-[#6])(-[#1])(-[#1])-[#1]", overrides: &[] }, // alkane CH3
    OplsRuleRow { name: "opls_136", def: "[C;X4](-[#6])(-[#6])(-[#1])-[#1]", overrides: &[] }, // alkane CH2
    OplsRuleRow { name: "opls_137", def: "[C;X4](-[#6])(-[#6])(-[#6])-[#1]", overrides: &[] }, // alkane CH
    OplsRuleRow { name: "opls_138", def: "[C;X4](-[#1])(-[#1])(-[#1])-[#1]", overrides: &[] }, // alkane CH4
    OplsRuleRow { name: "opls_139", def: "[C;X4](-[#6])(-[#6])(-[#6])-[#6]", overrides: &[] }, // alkane C
    OplsRuleRow { name: "opls_140", def: "[#1]-[C;X4]", overrides: &[] }, // alkane H.
    OplsRuleRow { name: "opls_141", def: "[C;X3](=[#6])(-[#6])-[#6]", overrides: &[] }, // alkene C (R2-C=)
    OplsRuleRow { name: "opls_142", def: "[C;X3](=[#6])(-[#6])-[#1]", overrides: &[] }, // alkene C (RH-C=)
    OplsRuleRow { name: "opls_143", def: "[C;X3](=[#6])(-[#1])-[#1]", overrides: &[] }, // alkene C (H2-C=)
    OplsRuleRow { name: "opls_144", def: "[#1]-[C;X3]", overrides: &[] }, // alkene H (H-C=)
    OplsRuleRow { name: "opls_145", def: "[c;X3;r6]1:[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:1", overrides: &["opls_141", "opls_142"] }, // Benzene C - 12 site JACS,112,4768-90. Use #145B for biphenyl
    OplsRuleRow { name: "opls_146", def: "[#1]-[c;%opls_145]", overrides: &["opls_144"] }, // Benzene H - 12 site.
    OplsRuleRow { name: "opls_148", def: "[C;X4](-[c;%opls_145])(-[#1])(-[#1])-[#1]", overrides: &["opls_149", "opls_135"] }, // C: CH3, toluene
    OplsRuleRow { name: "opls_149", def: "[C;X4](-[c;%opls_145])(-[#1])(-[#1])-*", overrides: &["opls_136"] }, // C: CH2, ethyl benzene
    OplsRuleRow { name: "opls_150", def: "[C;X3;H1](=[C;X3])-[C;X3]=[C;X3]", overrides: &["opls_142"] }, // diene =CH-CH=; use #178 for =CR-CR=
    OplsRuleRow { name: "opls_151", def: "[Cl]-[C;X4]", overrides: &["opls_401"] }, // Cl in alkyl chlorides
    OplsRuleRow { name: "opls_152", def: "[C;X4](-[Cl])(-[Cl,#6,#1])(-[Cl,#6,#1])-[Cl,#6,#1]", overrides: &[] }, // RCH2Cl in alkyl chlorides
    OplsRuleRow { name: "opls_153", def: "[#1]-[C;X4](-[Cl])(-[Cl,#6,#1])-[Cl,#6,#1]", overrides: &["opls_140"] }, // H in RCH2Cl in alkyl chlorides
    OplsRuleRow { name: "opls_154", def: "[O;X2](-[#1])-[!#1]", overrides: &[] }, // all-atom O: mono alcohols
    OplsRuleRow { name: "opls_155", def: "[#1]-[O;%opls_154]", overrides: &[] }, // all-atom H(O): mono alcohols, OP(=O)2
    OplsRuleRow { name: "opls_156", def: "[#1]-[#6](-[#1])(-[#1])-O-[#1]", overrides: &["opls_140"] }, // all-atom H(C): methanol
    OplsRuleRow { name: "opls_157", def: "[C;X4](-[#1])(-[#1])(-*)-[O;%opls_154]", overrides: &["opls_136", "opls_159", "opls_158"] }, // all-atom C: CH3 & CH2, alcohols
    OplsRuleRow { name: "opls_158", def: "[C;X4](-[#1])-[O;%opls_154]", overrides: &["opls_136", "opls_159"] }, // all-atom C: CH, alcohols
    OplsRuleRow { name: "opls_159", def: "[C;X4]-[O;%opls_154]", overrides: &["opls_136"] }, // all-atom C: C, alcohols
    OplsRuleRow { name: "opls_166", def: "[c;X3;r6]1(-O-[#1]):[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:1", overrides: &["opls_145"] }, // C(OH) phenol Use with all
    OplsRuleRow { name: "opls_167", def: "[O;X2](-[#1])-[c;%opls_166]", overrides: &["opls_154"] }, // O phenol atom C, H 145 & 146
    OplsRuleRow { name: "opls_168", def: "[#1]-[O;%opls_167]", overrides: &["opls_155"] }, // H phenol
    OplsRuleRow { name: "opls_178", def: "[C;X3;H0](=[C;X3])(-[#6])-[C;X3]=[C;X3]", overrides: &["opls_141"] }, // diene =CR-CR=; use #150 for =CH-CH=
    OplsRuleRow { name: "opls_180", def: "[O;X2](-[#6])-[#6]", overrides: &[] }, // O: dialkyl ether
    OplsRuleRow { name: "opls_181", def: "[C;X4](-[O;%opls_180])(-[#1])(-[#1])-[#1]", overrides: &["opls_182"] }, // C(H3OR): methyl ether
    OplsRuleRow { name: "opls_182", def: "[C;X4](-[O;%opls_180])(-[#1])(-[#1])-[#6]", overrides: &[] }, // C(H2OR): ethyl ether
    OplsRuleRow { name: "opls_183", def: "[C;X4](-[O;%opls_180])(-[#6])(-[#6])-[#1]", overrides: &[] }, // C(HOR): i-Pr ether, allose
    OplsRuleRow { name: "opls_184", def: "[C;X4](-[O;%opls_180])(-[#6])(-[#6])-[#6]", overrides: &[] }, // C(OR): t-Bu ether
    OplsRuleRow { name: "opls_185", def: "[#1]-[#6]-[O;%opls_180]", overrides: &["opls_140"] }, // H(COR): alpha H ether
    OplsRuleRow { name: "opls_186", def: "[O;X2]-[C;%opls_193](-O)(-[#6])-[#1]", overrides: &["opls_180"] }, // O: acetal ether
    OplsRuleRow { name: "opls_193", def: "[C;X4](-O)(-O)(-[#6])-[#1]", overrides: &[] }, // C(HCO2): acetal OCHRO
    OplsRuleRow { name: "opls_194", def: "[#1]-[C;%opls_193]-[O;%opls_186]", overrides: &["opls_185"] }, // H(CHO2): acetal OCHRO
    OplsRuleRow { name: "opls_200", def: "[S;X2]-[#1]", overrides: &["opls_202"] }, // all-atom S: thiols
    OplsRuleRow { name: "opls_201", def: "[S;X2](-[#1])-[#1]", overrides: &["opls_200", "opls_202"] }, // S IN H2S JPC,90,6379 (1986)
    OplsRuleRow { name: "opls_202", def: "[S;X2]", overrides: &[] }, // all-atom S: sulfides, S=C
    OplsRuleRow { name: "opls_203", def: "[S;X2]-S", overrides: &["opls_200", "opls_202"] }, // all-atom S: disulfides
    OplsRuleRow { name: "opls_204", def: "[#1;X1]-[S;%opls_200]", overrides: &[] }, // all-atom H(S): thiols
    OplsRuleRow { name: "opls_205", def: "[#1;X1]-[S;%opls_201]", overrides: &["opls_204"] }, // H IN H2S JPC,90,6379 (1986)
    OplsRuleRow { name: "opls_206", def: "[C;X4](-[S;%opls_200])(-[#1])-[#1]", overrides: &["opls_207", "opls_208", "opls_210"] }, // all-atom C: CH2, thiols
    OplsRuleRow { name: "opls_207", def: "[C;X4](-[S;%opls_200])(-[#1])(-[!#1])-[!#1]", overrides: &["opls_208", "opls_211"] }, // all-atom C: CH, thiols
    OplsRuleRow { name: "opls_208", def: "[C;X4]-[S;%opls_200]", overrides: &["opls_212"] }, // all-atom C: C, thiols
    OplsRuleRow { name: "opls_209", def: "[C;X4](-[S;%opls_202])(-[#1])(-[#1])-[#1]", overrides: &["opls_210", "opls_211", "opls_212"] }, // all-atom C: CH3, sulfides
    OplsRuleRow { name: "opls_210", def: "[C;X4](-[S;%opls_202])(-[#1])-[#1]", overrides: &["opls_211", "opls_212"] }, // all-atom C: CH2, sulfides
    OplsRuleRow { name: "opls_211", def: "[C;X4](-[S;%opls_202])-[#1]", overrides: &["opls_212"] }, // all-atom C: CH, sulfides
    OplsRuleRow { name: "opls_212", def: "[C;X4]-[S;%opls_202]", overrides: &[] }, // all-atom C: C, sulfides
    OplsRuleRow { name: "opls_213", def: "[C;X4](-[S;%opls_203])(-[#1])(-[#1])-[#1]", overrides: &["opls_214", "opls_215", "opls_216", "opls_209"] }, // all-atom C: CH3, disulfides
    OplsRuleRow { name: "opls_214", def: "[C;X4](-[S;%opls_203])(-[#1])-[#1]", overrides: &["opls_215", "opls_216", "opls_210"] }, // all-atom C: CH2, disulfides
    OplsRuleRow { name: "opls_215", def: "[C;X4](-[S;%opls_203])-[#1]", overrides: &["opls_216", "opls_211"] }, // all-atom C: CH, disulfides
    OplsRuleRow { name: "opls_216", def: "[C;X4]-[S;%opls_203]", overrides: &["opls_212"] }, // all-atom C: C, disulfides
    OplsRuleRow { name: "opls_217", def: "[C;X4](-[S;%opls_200])(-[#1])(-[#1])-[#1]", overrides: &["opls_211", "opls_207", "opls_209", "opls_206"] }, // all-atom C: CH3, methanethiol
    OplsRuleRow { name: "opls_235", def: "[C;X3](=[O;X1])-[N;X3]", overrides: &["opls_277"] }, // C=O in amide, dmf, peptide bond
    OplsRuleRow { name: "opls_236", def: "O=[C;%opls_235]", overrides: &["opls_278"] }, // O: C=O in amide. Acyl R on C in amide is neutral -
    OplsRuleRow { name: "opls_237", def: "[N;X3](-[#1])(-[#1])-[C;%opls_235]", overrides: &["opls_900"] }, // N: primary amide. use alkane parameters.
    OplsRuleRow { name: "opls_238", def: "[N;X3](-[#1])(-[#6])-[C;%opls_235]", overrides: &[] }, // N: secondary amide, peptide bond (see #279 for formyl H)
    OplsRuleRow { name: "opls_239", def: "[N;X3](-[#6])(-[#6])-[C;%opls_235]", overrides: &[] }, // N: tertiary amide
    OplsRuleRow { name: "opls_240", def: "[#1;X1]-[N;%opls_237]", overrides: &["opls_909"] }, // H on N: primary amide
    OplsRuleRow { name: "opls_241", def: "[#1;X1]-[N;%opls_238]", overrides: &[] }, // H on N: secondary amide
    OplsRuleRow { name: "opls_242", def: "[C;X4](-[#1])(-[#1])(-[#1])-[N;%opls_238]", overrides: &[] }, // C on N: secondary N-Me amide
    OplsRuleRow { name: "opls_243", def: "[C;X4](-[#1])(-[#1])(-[#1])-[N;%opls_239]", overrides: &[] }, // C on N: tertiary N-Me amide
    OplsRuleRow { name: "opls_245", def: "[C;X4](-[#6])(-[#1])(-[#1])-[N;%opls_239]", overrides: &[] }, // C on N: tertiary N-CH2R amide, Pro CD
    OplsRuleRow { name: "opls_260", def: "[c;X3;r6]-[C;X2;%opls_261]", overrides: &["opls_145"] }, // C(CN) benzonitrile
    OplsRuleRow { name: "opls_261", def: "[C;X2]-[c;X3;r6]", overrides: &["opls_754"] }, // C(N) benzonitrile
    OplsRuleRow { name: "opls_262", def: "[N;X1]#[C;%opls_261]", overrides: &["opls_753"] }, // N benzonitrile
    OplsRuleRow { name: "opls_263", def: "[c;X3;r6]1(-Cl):[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:1", overrides: &["opls_145"] }, // C(Cl) chlorobenzene
    OplsRuleRow { name: "opls_264", def: "[Cl;X1]-[c;%opls_263]", overrides: &["opls_401"] }, // Cl chlorobenzene
    OplsRuleRow { name: "opls_267", def: "[C;X3](=[O;X1])-O-[#1]", overrides: &["opls_277", "opls_271", "opls_465"] }, // Co in CCOOH carboxylic acid
    OplsRuleRow { name: "opls_268", def: "[O;X2](-[C;%opls_267])-[#1]", overrides: &["opls_154"] }, // Oh in CCOOH R in RCOOH is
    OplsRuleRow { name: "opls_269", def: "[O;X1]=[C;%opls_267]-O-[#1]", overrides: &["opls_278", "opls_272"] }, // Oc in CCOOH neutral; use #135-#140
    OplsRuleRow { name: "opls_270", def: "[#1]-[O;%opls_268]", overrides: &["opls_155"] }, // H in CCOOH
    OplsRuleRow { name: "opls_271", def: "[C;X3](~[O;X1])~[O;X1]", overrides: &[] }, // C in COO- carboxylate — `~`: the carboxylate O's are one resonance pair, written C(=O)[O-] or either way
    OplsRuleRow { name: "opls_272", def: "[O;X1]~[C;%opls_271]", overrides: &[] }, // O: O in COO- carboxylate,peptide terminus — `~`: either carboxylate resonance form
    OplsRuleRow { name: "opls_277", def: "[C;X3](=[O;X1])-[#1]", overrides: &[] }, // AA C: aldehyde - for C-alpha use #135-#139
    OplsRuleRow { name: "opls_278", def: "[O;X1]=[C;%opls_277]", overrides: &[] }, // AA O: aldehyde
    OplsRuleRow { name: "opls_279", def: "[#1]-[C;X3]=[O;X1]", overrides: &["opls_185", "opls_144"] }, // AA H-alpha in aldehyde & formamide
    OplsRuleRow { name: "opls_280", def: "[C;X3](=[O;X1])(-[#6])-[#6]", overrides: &[] }, // AA C: ketone - for C-alpha use #135-#139
    OplsRuleRow { name: "opls_281", def: "[O;X1]=[C;%opls_280]", overrides: &[] }, // AA O: ketone
    OplsRuleRow { name: "opls_282", def: "[#1]-[#6]-[C;%opls_277,%opls_280,%opls_465;!%opls_267]", overrides: &["opls_140", "opls_144"] }, // AA H on C-alpha in ketone & aldehyde
    OplsRuleRow { name: "opls_401", def: "[Cl;X0]", overrides: &[] }, // Cl- JACS 106, 903 (1984)
    OplsRuleRow { name: "opls_406", def: "[Li;X0]", overrides: &[] }, // Li+
    OplsRuleRow { name: "opls_465", def: "[C;X3](=[O;X1])-[O;X2]", overrides: &["opls_277"] }, // AA C: esters - for R on C=O, use #280-#282
    OplsRuleRow { name: "opls_466", def: "[O;X1]=[C;%opls_465]-[O;%opls_467]", overrides: &["opls_278"] }, // AA =O: esters
    OplsRuleRow { name: "opls_467", def: "[O;X2](-[C;%opls_465])-[!#1]", overrides: &["opls_180"] }, // AA -OR: ester
    OplsRuleRow { name: "opls_468", def: "[C;X4](-[O;%opls_467])(-[#1])(-[#1])-[#1]", overrides: &["opls_181"] }, // methoxy C in esters - see also #490-#492
    OplsRuleRow { name: "opls_469", def: "[#1]-[C;%opls_468,%opls_490]", overrides: &["opls_185"] }, // methoxy Hs in esters
    OplsRuleRow { name: "opls_490", def: "[C;X4](-[O;%opls_467])(-[#1])(-[#1])-[#6]", overrides: &["opls_182"] }, // C(H2OS) ethyl ester
    OplsRuleRow { name: "opls_496", def: "[S;X3](~[O;%opls_497])(-[#6])-[#6]", overrides: &[] }, // sulfoxide - all atom — `~`: S=O or [S+]-[O-]
    OplsRuleRow { name: "opls_497", def: "[O;X1]~[S;X3]", overrides: &[] }, // sulfoxide - all atom — `~`: S=O or [S+]-[O-]
    OplsRuleRow { name: "opls_498", def: "[C;X4](-[S;X3])(-[#1])(-[#1])-[#1]", overrides: &[] }, // CH3 all-atom C: sulfoxide
    OplsRuleRow { name: "opls_520", def: "[n;X2;r6]1:[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:1", overrides: &[] }, // N in pyridine 6-31G*
    OplsRuleRow { name: "opls_521", def: "[c;X3;r6]:[n;%opls_520]", overrides: &[] }, // C1 in pyridine CHELPG
    OplsRuleRow { name: "opls_522", def: "[c;X3;r6]:[c;%opls_521]", overrides: &["opls_142"] }, // C2 in pyridine charges
    OplsRuleRow { name: "opls_523", def: "[c;X3;r6](:[c;%opls_522]):[c;%opls_522]", overrides: &["opls_142"] }, // C3 in pyridine for
    OplsRuleRow { name: "opls_524", def: "[#1]-[c;%opls_521]", overrides: &["opls_144"] }, // H1 in pyridine 520-619
    OplsRuleRow { name: "opls_525", def: "[#1]-[c;%opls_522]", overrides: &["opls_144"] }, // H2 in pyridine
    OplsRuleRow { name: "opls_526", def: "[#1]-[c;%opls_523]", overrides: &["opls_144"] }, // H3 in pyridine
    OplsRuleRow { name: "opls_530", def: "[n;X2;r6]1:[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:[n;X2;r6]:[c;X3;r6]:1", overrides: &[] }, // N in pyrimidine
    OplsRuleRow { name: "opls_531", def: "[c;X3;r6](:[n;%opls_530]):[n;%opls_530]", overrides: &[] }, // C2 in pyrimidine
    OplsRuleRow { name: "opls_532", def: "[c;X3;r6](:[n;%opls_530]):[c;X3;r6]", overrides: &[] }, // C4 in pyrimidine
    OplsRuleRow { name: "opls_533", def: "[c;X3;r6](:[c;%opls_532]):[c;%opls_532]", overrides: &["opls_142"] }, // C5 in pyrimidine
    OplsRuleRow { name: "opls_534", def: "[#1]-[c;%opls_531]", overrides: &["opls_144"] }, // H2 in pyrimidine
    OplsRuleRow { name: "opls_535", def: "[#1]-[c;%opls_532]", overrides: &["opls_144"] }, // H4 in pyrimidine
    OplsRuleRow { name: "opls_536", def: "[#1]-[c;%opls_533]", overrides: &["opls_144"] }, // H5 in pyrimidine
    OplsRuleRow { name: "opls_542", def: "[n;X3;r5]1:[c;X3;r5]:[c;X3;r5]:[c;X3;r5]:[c;X3;r5]:1", overrides: &[] }, // N in pyrrole
    OplsRuleRow { name: "opls_543", def: "[c;X3;r5]:[n;%opls_542]", overrides: &[] }, // C2 in pyrrole
    OplsRuleRow { name: "opls_544", def: "[c;X3;r5]:[c;%opls_543]", overrides: &["opls_142"] }, // C3 in pyrrole
    OplsRuleRow { name: "opls_545", def: "[#1]-[n;%opls_542]", overrides: &[] }, // H1 in pyrrole
    OplsRuleRow { name: "opls_546", def: "[#1]-[c;%opls_543]", overrides: &["opls_144"] }, // H2 in pyrrole
    OplsRuleRow { name: "opls_547", def: "[#1]-[c;%opls_544]", overrides: &["opls_144"] }, // H3 in pyrrole
    OplsRuleRow { name: "opls_678", def: "[C;X4](-[#1])(-[#1])(-[#1])-[c;%opls_543]", overrides: &["opls_679"] }, // CH3, 2-methyl pyrrole
    OplsRuleRow { name: "opls_679", def: "[C;X4](-[#1])(-[#1])-[c;%opls_543]", overrides: &["opls_136"] }, // CH2, 2-ethyl pyrrole
    OplsRuleRow { name: "opls_711", def: "[C;X4;r3]1(-[#1])(-[#1])-[C;X4;r3]-[C;X4;r3]-1", overrides: &["opls_136", "opls_712"] }, // CH2 C: cyclopropane
    OplsRuleRow { name: "opls_712", def: "[C;X4;r3]1(-[#1])-[C;X4;r3]-[C;X4;r3]-1", overrides: &["opls_137", "opls_713"] }, // CHR C: cyclopropane
    OplsRuleRow { name: "opls_713", def: "[C;X4;r3]1-[C;X4;r3]-[C;X4;r3]-1", overrides: &[] }, // CR2 C: cyclopropane
    OplsRuleRow { name: "opls_718", def: "[c;X3;r6]1(-F):[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:[c;X3;r6]:1", overrides: &["opls_145"] }, // C(F) fluorobenzene
    OplsRuleRow { name: "opls_719", def: "F-[c;%opls_718]", overrides: &["opls_965"] }, // F fluorobenzene
    OplsRuleRow { name: "opls_753", def: "[N;X1]#[#6]", overrides: &[] }, // N IN RCN nitriles
    OplsRuleRow { name: "opls_754", def: "[C;X2]#[N;%opls_753]", overrides: &[] }, // C IN RCN nitriles
    OplsRuleRow { name: "opls_755", def: "[C;X4](-[#1])(-[#1])(-[#1])-[C;%opls_754]", overrides: &["opls_135", "opls_756"] }, // C of CH3 in CH3CN
    OplsRuleRow { name: "opls_756", def: "[C;X4](-[#1])(-[#1])-[C;%opls_754]", overrides: &["opls_136", "opls_757"] }, // C of CH2 in RCH2CN
    OplsRuleRow { name: "opls_757", def: "[C;X4](-[#1])-[C;%opls_754]", overrides: &["opls_137", "opls_758"] }, // C of CH in R2CHCN
    OplsRuleRow { name: "opls_758", def: "[C;X4]-[C;%opls_754]", overrides: &[] }, // C of C in R3CCN
    OplsRuleRow { name: "opls_759", def: "[#1]-[#6]-[C;%opls_754]", overrides: &["opls_140"] }, // HC-CT-CN alpha-H in nitriles
    OplsRuleRow { name: "opls_760", def: "[N;X3](~[O;X1])~[O;X1]", overrides: &["opls_239"] }, // N in nitro R-NO2 — `~`: N(=O)[O-] or [N+](=O)[O-], either O
    OplsRuleRow { name: "opls_761", def: "[O;X1]~[N]~[O;X1]", overrides: &[] }, // O in nitro R-NO2 — `~`: either nitro resonance form
    OplsRuleRow { name: "opls_762", def: "[C;X4](-[#1])(-[#1])(-[#1])-[N;%opls_760]", overrides: &[] }, // CT-NO2 nitromethane
    OplsRuleRow { name: "opls_763", def: "[#1]-[C;X4]-[N;%opls_760]", overrides: &["opls_140"] }, // HC-CT-NO2 alpha-H in nitroalkanes
    OplsRuleRow { name: "opls_764", def: "[C;X4](-[#6])(-[#1])(-[#1])-[N;%opls_760]", overrides: &["opls_136"] }, // CT-NO2 nitroethane
    OplsRuleRow { name: "opls_767", def: "[N;X3](~[O;X1])(~[O;X1])-[c;X3;r6]", overrides: &["opls_760"] }, // N in nitro Ar-NO2 — `~`: either nitro resonance form
    OplsRuleRow { name: "opls_768", def: "[c;X3;r6]-[N;%opls_767]", overrides: &["opls_145"] }, // C(NO2) nitrobenzene
    OplsRuleRow { name: "opls_771", def: "[O;X1]=[C;X3;%opls_772]", overrides: &[] }, // propylene carbonate O (Luciennes param.)
    OplsRuleRow { name: "opls_772", def: "[C;X3;r5]1-[O;X2;r5]-[C;X4;r5]-[C;X4;r5]-[O;X2;r5]-1", overrides: &[] }, // propylene carbonate C=O
    OplsRuleRow { name: "opls_773", def: "[O;X2;r5]1-[C;X4;r5]-[C;X4;r5]-[O;X2;r5]-[C;X3;r5]-1", overrides: &["opls_180"] }, // propylene carbonate OS
    OplsRuleRow { name: "opls_774", def: "[C;X4;r5]1(-[#1])(-[#1])-[C;X4;r5]-[O;X2;r5]-[C;X3;r5]-[O;X2;r5]-1", overrides: &["opls_182"] }, // propylene carbonate C in CH2
    OplsRuleRow { name: "opls_775", def: "[C;X4;r5]1(-[#1])(-[#6])-[C;X4;r5]-[O;X2;r5]-[C;X3;r5]-[O;X2;r5]-1", overrides: &["opls_182"] }, // propylene carbonate C in CH
    OplsRuleRow { name: "opls_776", def: "[C;X4](-[#1])(-[#1])(-[#1])-[C;%opls_775]", overrides: &["opls_135"] }, // propylene carbonate C in CH3
    OplsRuleRow { name: "opls_777", def: "[#1]-[C;%opls_774]", overrides: &["opls_185", "opls_140"] }, // propylene carbonate H in CH2
    OplsRuleRow { name: "opls_778", def: "[#1]-[C;%opls_775]", overrides: &["opls_185", "opls_140"] }, // propylene carbonate H in CH
    OplsRuleRow { name: "opls_779", def: "[#1]-[C;%opls_776]", overrides: &["opls_185", "opls_140"] }, // propylene carbonate H in CH3
    OplsRuleRow { name: "opls_900", def: "[N;X3](-[#1])(-[#1])-[#6]", overrides: &[] }, // N primary amines
    OplsRuleRow { name: "opls_901", def: "[N;X3](-[#1])(-[#6;!%opls_235;!%opls_543])-[#6;!%opls_235;!%opls_543]", overrides: &[] }, // N secondary amines, aziridine N1
    OplsRuleRow { name: "opls_903", def: "[C;X4](-[#1])(-[#1])(-[#1])-[N;%opls_900]", overrides: &["opls_906"] }, // CH3(N) primary aliphatic amines, H(C) is #911
    OplsRuleRow { name: "opls_904", def: "[C;X4](-[#1])(-[#1])(-[#1])-[N;%opls_901]", overrides: &["opls_906"] }, // CH3(N) secondary aliphatic amines, H(C) is #911
    OplsRuleRow { name: "opls_906", def: "[C;X4](-[N;%opls_900])(-[#1])-[#1]", overrides: &["opls_136"] }, // CH2(N) primary aliphatic amines, H(C) is #911
    OplsRuleRow { name: "opls_909", def: "[#1]-[N;%opls_900]", overrides: &[] }, // H(N) primary amines
    OplsRuleRow { name: "opls_910", def: "[#1]-[N;%opls_901]", overrides: &[] }, // H(N) secondary amines
    OplsRuleRow { name: "opls_911", def: "[#1]-[C;%opls_903,%opls_904,%opls_905,%opls_906,%opls_908]", overrides: &["opls_140"] }, // H(C) for C bonded to N in amines, diamines (aziridine H2,H3)
    OplsRuleRow { name: "opls_917", def: "[c;X3;r6]-[N;%opls_901]", overrides: &["opls_145"] }, // C(NH2) N-methylaniline
    OplsRuleRow { name: "opls_925", def: "[C;X2](#[#6])-[#1]", overrides: &[] }, // alkyne RC%CH terminal C acetylene
    OplsRuleRow { name: "opls_926", def: "[#1]-[C;%opls_925]", overrides: &[] }, // alkyne RC%CH terminal H
    OplsRuleRow { name: "opls_927", def: "[C;X2](-[#6](-[#1])-[#1])#[#6]", overrides: &[] }, // alkyne RC%CH C2 R-with 2 or 3 H
    OplsRuleRow { name: "opls_928", def: "[C;X2](-[C;X4;H1])#[#6]", overrides: &[] }, // alkyne RC%CH C2 R-with 1 H
    OplsRuleRow { name: "opls_930", def: "[#1]-*-C#[C;%opls_925]", overrides: &["opls_140", "opls_144"] }, // alkyne RC%CH H on C3 (for C3 use #135-#139)
    OplsRuleRow { name: "opls_961", def: "[C;X4](-F)(-F)(-F)-*", overrides: &["opls_962"] }, // CF3 perfluoroalkanes
    OplsRuleRow { name: "opls_962", def: "[C;X4](-F)(-F)(-*)-*", overrides: &[] }, // CF2 perfluoroalkanes
    OplsRuleRow { name: "opls_965", def: "F-[#6]", overrides: &[] }, // F: perfluoroalkanes
];
