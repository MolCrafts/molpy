# Extending the force-field IR

The force-field IR (it adopts the LAMMPS standard, see
[Force-field IR](forcefield-ir.md)) is a **protocol**. Its shape is data —
a *category* (how many atoms a term has, which Frame block holds its terms,
which coordinate its energy is a function of) and a *style* (its ordered
parameters, each with a dimension, and its energy) — and anything that has
that shape extends it: from Rust, from Python, from molpy, with nothing in
molrs rebuilt. A registered style prices at both compile doors and in MD,
persists in a `.mrec` record, is written to every engine that can hold it
and refused by name by every engine that cannot, and joins the form
conversions — exactly as a built-in does. The built-ins are themselves
registrations of the same form, sealed.

The form — categories, styles, parameters, refusals — is `molrs::ff::ir`
(Rust) and `molrs.ff.ir` (Python); registering into it is
`molrs::ff::style_registry` and `molrs.ff.style_registry`; compiling a force
field against it is `molrs::ff::compile` and `molrs.ff.compile`. molpy
re-exports the Python modules. The engine files a registered
style is read from and written to are `molrs.io`'s (`LammpsForcefieldReader`,
`OpenmmXmlWriter`, `write_mrec`, …; the same names in Rust's `molrs::io`), the
one owner of every file reader and writer.

## The form

### Categories

```rust
pub struct CategorySpec {
    pub name: Cow<'static, str>,   // ^[a-z][a-z0-9_]*$
    pub arity: Arity,              // Exact(0..=5) or SelfOrPair (a pair)
    pub block: Cow<'static, str>,  // the Frame block it prices
    pub coordinate: Coordinate,    // None, Distance, Angle, Dihedral, Improper, Compound
    pub order: EndpointOrder,      // Reversible, Ordered, Unordered
    pub excludes: bool,            // always false
}
```

| name | arity | block | coordinate | order | priced |
|---|---|---|---|---|---|
| `atom` | 0 | `atoms` | none | – | no |
| `bond` | 2 | `bonds` | `r` | reversible | yes |
| `angle` | 3 | `angles` | `theta` | reversible | yes |
| `dihedral` | 4 | `dihedrals` | `phi` | reversible | yes |
| `improper` | 4 | `impropers` | `phi`, `chi = abs(phi)` | ordered | yes |
| `pair` | self or pair | `atoms` (by `type`) | `r` | unordered | yes: a `pairs` list or a neighbour search |
| `cmap` | 5 | `cmaps` | compound | ordered (φ then ψ) | yes |
| `drude` | 2 | `drudes` | `r` | ordered | yes, `drude harmonic` by its expression |
| `constraint` | 2 | `constraints` | none | reversible | no |
| `virtual_site` | 0 | `virtual_sites` | none | – | no |

A **custom category** has 2 to 5 atoms, its block is exactly `<name>s` (so
a record alone locates it), and its coordinate is `Compound` — a function of
the atoms' positions — unless it asks for the geometric variable its arity
has (`Distance` for 2, `Angle` for 3, `Dihedral` / `Improper` for 4).
`CategorySpec::custom(name, arity, coordinate, order)` builds one; anything
else is refused (`BadName`, `Arity`, `BlockName`). The only special case the
compiler keys on a category is `SelfOrPair` (neighbour-driven pricing); it
gates on `CategorySpec.block`, never on a name.

### Styles and parameters

```rust
pub struct StyleSpec {
    pub category, name,
    pub params: Vec<ParamSpec>,        // ordered: LAMMPS's *_coeff order where LAMMPS has the style
    pub style_params: Vec<ParamSpec>,  // cutoff, mixing, special keep their reserved meanings
    pub source: ParamSource,           // TypeRows or PerInstance (Frame columns)
    pub special: Option<SpecialClass>, // a pair style's special-bonds weights (default lj)
    pub expression: Option<String>,    // the energy, a Lepton expression, kept byte for byte
    pub force_is_gradient: bool,       // false only for coul/charmm
    pub unindexed_one_term: bool,
    pub lammps: LammpsForm,            // None, Positional, Custom(codec)
    pub samples: Vec<ConformanceSample>,          // registration check points
}
pub struct ParamSpec { name, dim: ParamDimension, kind: ParamKind, default: Option<ParamValue>, mix: ParamCombination, indexed: bool }
```

- **`ParamDimension`** is a dimension in five exponents, energy `E`, length `L`, angle
  `A`, charge `Q`, mass `M`, written `num ("/" factor)*` — `1`, `E`,
  `E/L^2`, `1/L`, `E*L^6`, `E*L/Q^2`, `A`, `E/A^2`, `M`. `A` alone is an
  angle **value**, stored in degrees and never converted; a negative angle
  exponent is per **radian**; any other positive angle exponent is refused.
  A unit conversion multiplies a value by `f_E^e · f_L^l · f_Q^q · f_M^m`.
- **Names** are Lepton identifiers. Reserved: `name`, `type`, `style`, the
  endpoint columns (`itom` … `mtom`, `atomi` … `atomm`), the annotation
  columns (`class`, `element`, `smarts`, `smirks`, `overrides`, `desc`,
  `doi`), the geometric variables `r theta phi chi`, the points `p1` …
  `p5`, and in a pair style `q`, `q1`, `q2` and `x1` / `x2` for another
  parameter `x`. `cutoff` (`L`), `mixing` (`arithmetic | geometric |
  sixthpower`) and `special` (`lj | coul`) are style parameters with fixed
  meanings.
- **Kinds**: `Scalar` (one number per row), `Array { rank }` (an
  `f64[T, S…]` column, priced by a Tier-2 or Tier-3 kernel, never an
  expression variable), `Text { choices }`. `indexed` makes a family `k1 …
  kM`, contiguous, one `M` per row shared by the style's indexed parameters.
- **Combining rules** (pair styles): a pair's value is the cross row's, else the
  `ParamCombination` of the two self rows — `Arithmetic`, `Geometric`, or the joint
  (ε, σ) rule the style's `mixing` names (`LjEpsilon` / `LjSigma`);
  `ParamCombination::None` makes an unlike pair without a cross row `NoMixing`.
- **Defaults** are applied in one place (`StyleSpec::gather`) before any
  kernel sees a parameter; a missing one is `MissingParam`, a value of the
  wrong kind or outside its choices `BadValue`.

## Kernels: three tiers

| Tier | Rust | Python | What it is |
|---|---|---|---|
| 1 | `Kernel::Expression`, or `register_style(spec, None)` with `spec.expression` | `expression=` | the expression, compiled with exact (dual-number) derivatives |
| 2 | `Kernel::Scalar(Arc<dyn ScalarForm>)`, `Kernel::Compound(Arc<dyn CompoundForm>)` | `kernel=` (a numpy callable; `compound=True` for positions) | a batch function of the coordinate, or of the atoms' positions |
| 3 | `Kernel::Constructor { compiled, typed, rows }` | – | a constructor that builds the whole kernel (every native built-in) |

The kernel contract, every tier:

- `q` per term is the category's coordinate: `r` in the field's length unit,
  `theta` ∈ [0, π], `phi` ∈ (−π, π] (IUPAC sign, atoms in row order) —
  radians. A compound form gets `x`, `n_terms × arity` positions,
  minimum-imaged relative to each term's first atom.
- Parameters arrive **exactly as stored**: IR units, angle values in
  degrees. A Tier-2 form reads each per-type parameter as an `n_terms`
  column (`ParamColumns::get`), arrays with a leading `n_terms` axis, text per
  term, numeric style parameters broadcast to a column; a pair form reads
  the resolved pair value of each parameter (cross row, else its mixing
  rule), and `q1`, `q2` when the frame has `atoms.charge`.
- A form **writes** the **unweighted** energy of each term and its
  derivative `de_dq` (a compound form: `∂E/∂x`, not the force). The generic
  kernels (`ff::potential::form_kernel`: `ScalarBonded`, `ScalarPair`,
  `CompoundTerms`) apply the pair special-bonds weight, `r < cutoff` at
  both compile doors (the style's `cutoff`, ∞ when it states none — LAMMPS
  truncates every pair style, 1-4 pairs included, and so does every
  built-in; a form is never called past the cutoff), the chain rule onto
  Cartesian forces and the virial.

LAMMPS's `pair_style lj/smooth/linear` as a Tier-2 form, from
`molrs-ext-example` (a crate outside molrs, its `pub` API only):

```rust
pub struct LjSmoothLinear;

impl ScalarForm for LjSmoothLinear {
    fn eval(&self, r: &[f64], p: &ParamColumns<'_>, e: &mut [f64], de_dr: &mut [f64]) {
        let (eps, sigma, rc) = (col(p, "epsilon"), col(p, "sigma"), col(p, "cutoff"));
        for t in 0..r.len() {
            let lj = |x: f64| {
                let s6 = (sigma[t] / x).powi(6);
                (4.0 * eps[t] * (s6 * s6 - s6), 24.0 * eps[t] * (s6 - 2.0 * s6 * s6) / x)
            };
            let ((u, du), (uc, duc)) = (lj(r[t]), lj(rc[t]));
            e[t] = u - uc - (r[t] - rc[t]) * duc;
            de_dr[t] = du - duc;
        }
    }

    fn inputs(&self) -> Vec<String> {
        vec!["epsilon".into(), "sigma".into(), "cutoff".into()]
    }
}

let spec = StyleSpec::new("pair", "lj/smooth/linear")
    .params(vec![
        ParamSpec::new("epsilon", ParamDimension::ENERGY).mix(ParamCombination::LjEpsilon { sigma: "sigma".into() }),
        ParamSpec::new("sigma", ParamDimension::LENGTH).mix(ParamCombination::LjSigma { epsilon: "epsilon".into() }),
    ])
    .style_params(vec![ParamSpec::new("cutoff", ParamDimension::LENGTH)])
    .special(SpecialClass::Vdw)
    .lammps(LammpsForm::positional());
registry.register_style(spec, Some(Kernel::Scalar(Arc::new(LjSmoothLinear))))?;
```

The same contract from Python — one call per style per evaluation, every
term at once:

```python
import numpy as np
from molrs.ff import ir, style_registry

def fene(r, k, r0, epsilon, sigma):
    x = (r / r0) ** 2
    s6 = (sigma / r) ** 6
    inner = r < 2 ** (1 / 6) * sigma
    e = -0.5 * k * r0**2 * np.log(1 - x) + np.where(inner, 4 * epsilon * (s6 * s6 - s6) + epsilon, 0.0)
    de = k * r / (1 - x) + np.where(inner, 4 * epsilon * (-12 * s6 * s6 + 6 * s6) / r, 0.0)
    return e, de

style_registry.register_style("bond", "fene/np",
                              params={"k": "E/L^2", "r0": "L", "epsilon": "E", "sigma": "L"},
                              kernel=fene)
```

A kernel of the wrong shape or dtype, or one that raises, is `KernelShape`
(the Python exception chained as `__cause__`), at compile or at the
evaluation that met it.

`PotentialCompiler::with_registry(ff, &registry)` compiles against a
registry of one's own; `PotentialCompiler::new` reads the process-wide one
(`molrs::ff::style_registry::register_style`, Python `style_registry.register_style`).

Without a force field, `molrs.ff.compile.ExplicitTerms(category, style, atoms, *,
charges=None, **params).compile()` builds the kernel of any style the IR prices — a
built-in, a registered one, a custom category's — over explicit terms: an
`(n, arity)` array of atom indices and each parameter as stored, one number
or one value per term. It is built exactly as `PotentialCompiler.compile`
builds it and returns a `Potentials`; there is no per-style Python class.

## New categories

```rust
registry.register_category(CategorySpec::custom(
    "urey_bradley", 3, Coordinate::Compound, EndpointOrder::Reversible,
))?;
registry.register_style(
    StyleSpec::new("urey_bradley", "harmonic").params(vec![
        ParamSpec::new("k_ub", "E/L^2".parse()?),
        ParamSpec::new("r_ub", ParamDimension::LENGTH),
    ]),
    Some(Kernel::Compound(Arc::new(UreyBradley))),
)?;
```

```python
style_registry.register_category("urey_bradley", 3)          # block "urey_bradleys", coordinate compound
style_registry.register_style("urey_bradley", "spring", params={"k_ub": "E/L^2", "r_ub": "L"},
                              expression="k_ub*(distance(p1,p3)-r_ub)^2")
ff.def_style("urey_bradley", "spring").def_type("A-B-A", a, b, a, k_ub=20.0, r_ub=2.4)
```

`ForceField.def_style` on a custom category returns a `RelationStyle` whose
`def_type(name, *endpoints, **params)` takes exactly the category's arity
(`Arity` otherwise); the terms are the rows of the Frame block `<name>s`. A
typifier's `TypeAssignment(links={kind: rows})` fills any registered relation kind
(by `MolGraph` kind name or class); `assign_terms` matches endpoints by the
category's `EndpointOrder` and molrec's wildcard rule.

A category nobody registered — one a record brought in — behaves as
`CategorySpec::custom(name, arity of its endpoint columns, Compound,
Reversible)`: kept always, priced when its style has an expression,
`NoKernel` otherwise.

## Expressions

The grammar, the functions, the definitions and the variable binding are
molrec's ([`docs/spec/forcefield.md`, Expressions](https://github.com/MolCrafts/molrec/blob/master/docs/spec/forcefield.md)),
a Lepton subset:

- `+ - * / ^`, parentheses, numbers; `^` binds tighter than unary minus
  (`-a^2 = -(a^2)`) and is right-associative; `exp log sqrt sin cos tan asin
  acos atan abs`, `min max`, `step` (x ≥ 0 → 1), `delta` (x = 0 → 1),
  `select(x, y, z)` (x ≠ 0 → y); no named constants.
- Definitions follow the energy, `; name=formula`, each using only names to
  its right.
- Point functions `distance(pa, pb)`, `angle(pa, pb, pc)`, `dihedral(pa, pb,
  pc, pd)` over `p1` … `pA` in every category with points (not `pair`).
- Variables: bond / drude `r`, angle `theta`, dihedral `phi`, improper
  `phi` and `chi`, compound categories points only, pair `r`, `q1`, `q2`, a
  bare parameter (the pair value) and `x1` / `x2` (the self rows); every
  numeric parameter by name, **as stored** — an angle value in degrees,
  which the expression converts: `theta0*(pi/180)`, with `pi` defined as the
  last sub-definition, `; pi=3.141592653589793` (Lepton has no named
  constants; parenthesised, `(pi/180)` is `f64::to_radians`'s factor to the
  bit, where `theta0*pi/180` rounds differently).
- The expression is the unweighted energy of one term, inside the cutoff
  (the form pair kernel truncates at `r < cutoff`; a shift or switch to
  zero there is the expression's own, as `lj/cut`'s `shift` and the CHARMM
  switch are); a pair expression must be symmetric under exchanging the
  atoms.

Derivatives are exact (forward-mode duals: one for `q`, `3·arity` for the
points). A style with no registered kernel whose instance carries an
`expression` is priced by it (the compile fallback); a registered kernel
takes priority, and an instance expression that differs from the
registry's is checked for agreement with it at first compile.

## Persistence

A custom style persists as its molrec style entry: its style parameters, its
**expression** (the instance's, else the registry's: `ForceFieldSection.from_forcefield` writes the
registry expression of a registered custom style, so a process that
registered nothing can price it), and its table (array columns included,
`f64[T, S…]`). A category beyond the built-ins is kept with the arity of
its endpoint columns. Reading never evaluates an expression: a style with
no expression reads whole, and compiling it is `NoKernel` — "no kernel for
`<category>` `` `<style>` ``: register it (molrs.ff.style_registry.register_style) or
give it an expression". Parameter dimensions are not persisted: a style read
from a record has no `ParamDimension`s until it is registered again.

## Engines and refusals

- **LAMMPS.** `LammpsForm::Positional` is derived from the spec:
  `<category>_style <name>`, `<category>_coeff <type> v₁ … vₙ` in `params`
  order, each value converted by its `ParamDimension`, `mixing` as `pair_modify mix`.
  It refuses, at registration, a Text, Array or indexed parameter, another
  style parameter than `cutoff` / `mixing`, and a category without a LAMMPS
  `*_style`. `LammpsForm::Custom(codec)` writes a line that is not
  positional. An expression-only style is refused (the installed LAMMPS has
  no `LEPTON` package); give it a form with `register_engine_form`.
- **OpenMM XML.** An expression style is written as its category's
  `Custom*Force` (`CustomBondForce`, `CustomAngleForce`,
  `CustomTorsionForce`, a `<Script>`-built `CustomCompoundBondForce`,
  `CustomNonbondedForce`), parameters in IR units and the expression
  rewritten exactly into OpenMM's units.
- **GROMACS, AMBER prmtop and frcmod** hold the built-in styles only.

Every engine refusal is `IrError::NoEngineForm { engine, category, style,
reason }`; a writer returns it typed (`ForceFieldWriteError::ir()` in Rust, the
`molrs.ff.ir.NoEngineFormError` class in Python). See
[Engine codecs](forcefield-ir.md#engine-codecs).

## Conversions

A style of a form family registers a `FormCodec` — its family, whether it
is the family's canonical style, and two exact maps, `embed` (this style →
the canonical style's parameters) and `project` (canonical → this style,
exact on its image, else a `FormRefusal` naming the condition):

```rust
registry.register_form("dihedral", "cos3", FormCodec::new("torsion", embed, project))?;
```

`ForceField::canonical()` maps every style of a family in the canonical
style's category onto it; `to_form(category, style)` converts within a
category through the canonical parameters, exactly or `OutOfImage`;
`fit_form(category, style, metric)` is the least-squares projection with a
residual, which needs only each style's energy and so works for an
expression, a Python and a native style alike. A family with no canonical
style (or two) is `FormConflict`; a style without a codec `NoForm`. See
[Converting between forms](forcefield-ir.md#converting-between-forms).

## Conformance

A registration is checked before it lands; nothing is left behind by a
refusal. Each refusal is one `IrError` variant (Python: a subclass of
`molrs.ff.ir.IrError`, itself a `ValueError`, named after the variant with
an `Error` suffix: `Sealed` is `SealedError`, `Dimension` is
`DimensionError`) naming the offending item.

| Variant | When |
|---|---|
| `UnknownCategory` | a style in a category neither built in nor registered |
| `BadName` | a category or parameter name outside its pattern |
| `Arity` | a custom arity outside 2..=5; a type with the wrong number of endpoints |
| `BlockName` | a custom category whose block is not `<name>s` |
| `ReservedParam`, `DuplicateParam` | a reserved or repeated parameter name |
| `Dimension` | an unparsable or forbidden dimension |
| `Parse`, `UnknownFunction`, `FunctionArity` | an expression that does not parse |
| `UnboundVariable` | a free name that is neither a variable of the category nor a numeric parameter (`theta` in a bond) |
| `Point` | `pk` beyond the arity, or any point in a pair |
| `CoordinateMismatch` | a scalar form on a compound category, a compound form on a pair |
| `Derivative` | a Tier-2 derivative against a central difference of its own energy, beyond 1e-6 |
| `Disagree` | an expression against the kernel beside it, beyond 1e-10 |
| `Asymmetric` | a pair energy that changes when its atoms are exchanged (1e-12) |
| `Sealed` | re-registering or unregistering a built-in |
| `Conflict` | a different registration under a taken name (an identical one is a no-op; Python `replace=True` overrides a custom one) |
| `NoKernel` | nothing can price the style |
| `NoMixing` | an unlike pair, a parameter that does not mix, no cross row |
| `MissingParam`, `BadValue` | a value a kernel needs and no default; a value of the wrong kind or outside its choices |
| `KernelShape` | a Python kernel's output of the wrong shape or dtype, or a kernel that raised |
| `NoEngineForm` | an engine that cannot hold the style |
| `FormConflict`, `NoForm` | a form family without exactly one canonical style; a style without a codec |
| `OutOfImage` | an exact form conversion (`to_form`) of a row outside the target style's image; a `fit_form` that cannot evaluate the category |
| `Malformed` | a spec whose declarations contradict each other (a mixing rule on a bonded parameter, an ε without its σ) |

The derivative and agreement checks run at registration on the style's
`samples` (16 seeded points each); a style registered without samples is
checked once per process at its first compile, on up to 8 real terms.
Relative errors are measured against `max(|value|, RMS over the sample
set)`. The checks never run per evaluation.

## From molpy

molpy keeps no IR of its own: `molpy.ff.style_registry` re-exports
`molrs.ff.style_registry`, so `molpy.ff.style_registry.StyleDeclaration is
molrs.ff.style_registry.StyleDeclaration`, and a style declared from either
package is the same registration. molpy's own guide
(`docs/developer/extending-forcefield.md`, "A new style in 30 lines") declares
`bond fene` by its expression, types a bead-spring chain with it and prices
it, and its `tests/test_ff/test_user_style.py` runs that snippet as written.

## How this is checked

**The Rust proof** — `molrs-ext-example/`, a standalone crate depending on
molrs as any user's crate does (`pub` API only; `scripts/check.sh ext`, a
pre-push hook and a CI job). It adds `pair lj/smooth/linear` (a Tier-2
`ScalarForm`), a category `urey_bradley` with a native `CompoundForm`,
`bond fene` by its expression and class2's bond-angle term as a category
`bond_angle`. The LAMMPS numbers are pinned by
`scripts/ff_ir_extension_lammps_check.sh --pin` (LAMMPS `run 0` on decks
molrs's own LAMMPS writer wrote from the specs; `%.17g` energy and forces;
`real` units, two configurations each):

| Test | Criterion | Measured worst |
|---|---|---|
| `pair_style_matches_lammps` | E and every force component = LAMMPS `lj/smooth/linear` (cutoff 5 Å straddling the 15 pairs: ten inside, five beyond), rel ≤ 1e-10; `compile_typed` = `compile`, rel ≤ 1e-12 | 1.7·10⁻¹⁵; doors bit for bit |
| `new_category_matches_lammps` | `urey_bradley` = LAMMPS `angle_style charmm` with K = 0, rel ≤ 1e-10 | bit for bit |
| `fene_matches_lammps` | `bond fene` by expression = LAMMPS `bond_style fene`, rel ≤ 1e-10, (r/R0)² < 0.9 | 6.8·10⁻¹⁶ |
| `mrec_round_trip` | `.mrec` write/read prices bit for bit; with `Registry::builtin()` only, `fene` prices bit for bit, `bond_angle` by its expression (≤ 1e-10 of its native form), `urey_bradley` and `lj/smooth/linear` are `NoKernel` naming them | as stated |
| `nonconforming_refused` | every protocol refusal reachable from Rust (registration, compile, writers, forms), by variant and named item | 25 variants (`KernelShape`: Python) |
| `bond_angle_cross_term` | hand value −π/60, central differences, expression = native form, rel ≤ 1e-12 | 5.8·10⁻¹⁵ |

The same script prices `bond_angle` against `angle_style class2` (every
cross term but `ba` at zero) when `MOLRS_LMP_CLASS2` names a LAMMPS built
with CLASS2 (`scripts/build_lammps_class2.sh`): compared, not pinned, worst
rel 2.2·10⁻¹⁶.

**The Python proof** — `molrs-python/tests/test_ff_ir_extension.py`, LAMMPS
numbers pinned in `ff_ir_extension_lammps.tsv` by the same script:

| Row | Criterion | Measured worst |
|---|---|---|
| `fene` by expression = the analytic formula | rel ≤ 1e-12 | 2.1·10⁻¹⁶ |
| a numpy kernel = the expression | E rel ≤ 1e-12, F rel ≤ 1e-10 | 1.1·10⁻¹⁶, 2.1·10⁻¹⁶ |
| both = LAMMPS `bond_style fene` (deck by `LammpsForcefieldWriter`, `lj` units) | rel ≤ 1e-10 | 7.0·10⁻¹⁶ (expression), 5.2·10⁻¹⁶ (numpy) |
| `urey_bradley` from Python = LAMMPS (`angle_style charmm`, K = 0) | rel ≤ 1e-10 | 1.6·10⁻¹⁶ |
| `pair lj/smooth/linear` by expression and by a numpy kernel = LAMMPS, its 5 Å cutoff straddling the pairs; `compile_typed` (an integrator's first force call) = `compile` | rel ≤ 1e-10; doors rel ≤ 1e-12 | 2.5·10⁻¹⁵ (expression), 2.9·10⁻¹⁵ (numpy); doors bit for bit |
| `.mrec` round trip; a subprocess that registered nothing | bit for bit, expression byte for byte | bit for bit |
| a callable-only style in a fresh process | `NoKernel` naming the style and `molrs.ff.style_registry.register_style` | as stated |
| refusals: unknown function, unbound variable, sealed `bond harmonic`, wrong `def_type` arity, kernel of the wrong shape, kernel raising, `GromacsTopForcefieldWriter.write`, missing parameter | each its `IrError` subclass naming the item | as stated |
| `dihedral table/linear` (`table: f64[N]`) by a numpy kernel = hand linear interpolation; round trip | rel ≤ 1e-12; bits | 0 |
| class2 bond-angle: expression = numpy = −π/60 (= the Rust form) | rel ≤ 1e-12 | 2.4·10⁻¹⁵ |

**molpy** — `tests/test_ff/test_user_style.py`: its guide's snippet, run as written,
energy = the analytic FENE sum to 1.1·10⁻¹⁶ relative, forces to 2.1·10⁻¹⁶ of
the largest, and a fresh process that registered nothing prices the saved
`.mrec` bit for bit.

**The built-ins conform** — `ff::ir::builtin_conformance`:

- Every built-in with an Appendix-A expression prices identically by its
  native kernel and by the expression registered as a style of its own, on
  64 seeded configurations × parameter sets per style (a pair style at both
  compile doors; `dihedral periodic` with 1–3 terms and `nharmonic` of
  order 2–5 by their table-generated expressions; `dihedral rb`, priced by
  its expression alone, against `multi/harmonic`'s native kernel): energy
  and forces to 1e-10 relative; measured worst 1.5·10⁻¹⁵ (energy) and
  3.6·10⁻¹⁵ (forces), over 24 styles.
- Every built-in constructor's `ParamSource` is what it reads: 37
  constructors; a `TypeRows` one prices another row differently and no row
  not at all, a `PerInstance` one prices the same bits with or without a
  row.
- Every positional LAMMPS codec writes the pinned includes
  (`ff/testdata/builtin_conformance/`) for the LAMMPS-read hand molecule, the
  five cross-engine sources (`ff14sb`, `gaff2`, `chamber`, `charmm36`,
  `oplsaa`) and one field per positional built-in in `real` and `metal`: 16
  of 22 files byte for byte, the rest to ≤ 2 ulps (the pinned includes were
  converted through LAMMPS's `lj` units by a unit expression, `real` → `real`
  included; the codec multiplies by one exact factor per dimension).
  `bond class2`, `pair
  buck` and `pair morse` have no pinned include; LAMMPS prices them in
  `ff::engine_codec_check`.

The engine codecs of extension styles against LAMMPS and OpenMM (`fene`,
`lj/smooth/linear`, a compound `urey_bradley` as OpenMM's
`CustomCompoundBondForce`) are [Checked against the
engines](forcefield-ir.md#checked-against-the-engines).
