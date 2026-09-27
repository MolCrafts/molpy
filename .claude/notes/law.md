# Software Engineering Laws

Every rule here outranks scope, minimal-diff, and convenience. There is
no "just this once". CLAUDE.md indexes one line per law.

Adding, changing, or repealing a law is the operator's act via
`/mol:note`. No skill retires a law on its own judgment.

## 0. Purpose

This file is not a style guide and not a pattern catalog.

Laws define **non-negotiable design constraints**. They constrain
architecture, ownership, dependency, state, and change. Patterns and
implementation techniques are subordinate to these laws. A law must be
strong enough to reject a concrete design in review.

If a rule cannot identify a concrete forbidden design, it is guidance,
not law.

A carve-out exists only where this file records it under § VII, naming
the subsystem. An agent never grants itself one.

---

# I. System shape

How the system as a whole should look.

<!-- mol:law:id:conceptual-integrity -->
## 1. Conceptual integrity

**Principle.** One problem should have one coherent conceptual model.

**Intent.** The system uses one set of concepts, terms, and abstractions.
A subsystem does not invent a sibling model of the same idea.

**Never**

- Never create parallel abstractions for the same concept.
- Never introduce aliases that develop independent semantics.
- Never solve local inconvenience by inventing a new conceptual layer.

**Derived guidance.** Prefer extending an existing concept over a sibling
concept. Shared vocabulary is part of architecture.

<!-- mol:law:id:architecture-first -->
## 2. Architecture first

**Principle.** Preserve a simple system shape before optimizing local
convenience.

**Intent.** Local coding convenience must not buy itself by breaking the
whole architecture. Simple, clear shape is the foundation of
maintainability and performance.

**Never**

- Never add a layer only because it may be useful later.
- Never introduce infrastructure for unmeasured performance concerns.
- Never let a local feature dictate global architecture.

**Derived guidance.** Prefer fewer architectural concepts. Prefer
removing indirection over explaining it.

<!-- mol:law:id:earn-complexity -->
## 3. Earn complexity

**Principle.** Every unit of complexity must be justified by demonstrated
pressure.

**Intent.** Complexity is not free. Need, performance, compatibility, or
extension must already exist before an abstraction does. Architecture
first governs shape; this law governs the complexity budget.

**Never**

- Never generalize for hypothetical future requirements.
- Never optimize without evidence.
- Never make something configurable merely because it could vary.
- Never add extensibility without an actual extension point.

---

# II. Boundaries and ownership

How the system is cut.

<!-- mol:law:id:locality-of-change -->
## 4. Locality of change

**Principle.** A local requirement should require a local change.

**Intent.** A good module boundary shows up as change locality, not as
an abstract cohesion score. High cohesion and low coupling are
consequences of this law.

**Never**

- Never require unrelated modules to change in lockstep.
- Never create dependency cycles.
- Never spread one responsibility across multiple owners.
- Never make callers understand unrelated subsystem details.

**Derived guidance.** A unit is green via `$META.build.test_single` on
its mirrored tests with fakes for outbound deps. If proving the unit
requires the full graph, the boundary is wrong — split, inject, or
`/mol:refactor`. Do not compensate with more integration tests.

<!-- mol:law:id:hide-decisions -->
## 5. Hide decisions, expose contracts

**Principle.** Implementation decisions stay behind their owning
boundary.

**Intent.** A module hides **decisions that may change**, not merely
lines in a different file.

**Never**

- Never leak representation details across module boundaries.
- Never expose internal lifecycle or storage decisions as public
  contract.
- Never require callers to reproduce internal policy.

**Derived guidance.** Program against stable contracts. An
implementation detail should be replaceable without rewriting
consumers.

<!-- mol:law:id:dependencies-follow-policy -->
## 6. Dependencies follow policy

**Principle.** Replaceable mechanisms depend on stable policy, never
the reverse.

**Intent.** Core semantics are not defined by UI, binding, framework,
storage, or transport.

    mechanism → policy

not

    policy → mechanism

**Never**

- Never make domain/core depend on UI.
- Never make core depend on a serialization format.
- Never make core depend on Python / Rust / WASM binding concerns.
- Never let a framework define domain semantics.

---

# III. Public surface

How others use the system.

<!-- mol:law:id:primitive-surface -->
## 7. Primitive public surface

**Principle.** Public APIs expose orthogonal primitives; composition
belongs to callers.

**Intent.** The API ships building blocks, not a hidden workflow.

**Never**

- Never provide an all-in-one façade for unrelated operations.
- Never make one public method perform several independently
  meaningful steps.
- Never encode one preferred workflow as the only API.
- Never duplicate primitives with convenience aliases that become
  separate contracts.

**Derived guidance.** High-level workflows may live outside the
primitive core (docs, examples, caller code).

<!-- mol:law:id:explicit-flow -->
## 8. Explicit flow

**Principle.** State transitions, ownership, and required ordering
must be explicit and enforceable.

**Intent.** A user must not enter an illegal state by forgetting a
step. This covers initialization, validation, lifecycle, context, and
state machines.

**Never**

- Never rely on hidden ambient context.
- Never expose `validate()` / `init()` steps callers can forget.
- Never depend on undocumented call ordering.
- Never encode required state in conventions alone.
- Never make illegal states trivially representable when the
  type/model can prevent them.

---

# IV. Truth and state

Who the system believes.

<!-- mol:law:id:one-home -->
## 9. One home per fact

**Principle.** Every authoritative fact has exactly one owner.

**Intent.** Avoid synchronization and drift. **Representation may be
duplicated; authority cannot.** A serialization copy may exist; it
must not become a second mutable truth.

**Never**

- Never maintain two independently mutable representations of the
  same truth.
- Never cache authoritative state without explicit invalidation
  semantics.
- Never copy configuration into another source of truth.
- Never infer and persist information that can be derived cheaply
  from its owner.

---

# V. Evolution

How the system changes without rotting.

<!-- mol:law:id:no-silent-debt -->
## 10. No silent debt

**Principle.** Debt must be explicit, bounded, and owned.

**Intent.** The worst debt is not a hack — it is a hack packaged as
normal architecture. A conscious exception is debt. An invisible
exception becomes architecture.

**Never**

- Never hide an architectural compromise inside an unrelated change.
- Never introduce temporary duplication without marking its removal
  path.
- Never normalize a workaround by silently building on top of it.
- Never leave known invariant violations undocumented.
- Never ignore, skip-mark, or weaken an assert on rot you already
  saw. Fix it if local and stage-allowed; else stop, report
  path:line, route `/mol:debug` / `/mol:refactor` / supersede, and
  name it in the summary.

Outranks "stay in scope" and "minimal diff".

---

# VI. Verification

How we prove the design has not decayed. Separate from architecture
laws.

<!-- mol:law:id:tests-owned-behavior -->
## 11. Tests verify owned behavior

**Principle.** Tests belong to the owner of the behavior they verify.

**Intent.** Tests verify a module's own contract, not the choreography
of the whole system.

**Never**

- Never test implementation details as public behavior.
- Never require unrelated subsystems merely to verify local
  semantics.
- Never use broad integration setup where a unit boundary is
  sufficient.

### Project testing policy: unit-only by default

New behavior must be unit-testable at its ownership boundary
(`tests/` mirrors source, one module, `$META.build.test_single`,
fakes for outbound deps). Integration / end-to-end scenarios are
documentation (`docs/`, `examples/`), not tests. A design that can
only be tested end-to-end is evidence of a missing boundary.

Layout details: `tester` agent.

---

# VII. Exceptions

Any design that violates a law is recorded, not inferred:

    Law violated:
    Reason:
    Evidence:
    Scope:
    Removal condition:
    Owner:

Convenience is not sufficient justification. The exception is itself
an architecture decision. An agent never grants one from the task
text.

## Recorded exceptions

### molrs 0.15 co-development pin (2026-09-27)

    Law violated:      16 (molpy-release-order) — "never bump the molrs pin
                       to a minor that is not published".
    Reason:            molpy is aligned with molrs 0.15 while that minor is
                       developed; the alignment runs against the local
                       editable molrs build.
    Evidence:          operator, 2026-09-27: "use local editable version
                       first!"; pin `molcrafts-molrs>=0.15.0,<0.16`
                       introduced in 60a3455.
    Scope:             the molpy dev branch through the molrs-015-align
                       spec chain.
    Removal condition: molrs v0.15.0 tagged and published; molpy is not
                       tagged or released before that.
    Owner:             operator.

---

# VIII. Derived principles

These are consequences or heuristics, not laws. SOLID, YAGNI, DRY,
and the rest do not outrank this file. If someone cites them, first
show which law they serve.

| Heuristic | Comes from |
|---|---|
| YAGNI | Earn complexity |
| High cohesion / low coupling | Locality of change |
| Dependency inversion | Dependencies follow policy |
| Information hiding | Hide decisions, expose contracts |
| DRY (authority only) | One home per fact |
| Deep modules | Hide decisions + Primitive public surface |
| Composition over inheritance | Locality of change (a common means) |
| KISS / fewer boxes | Architecture first + Earn complexity |
| Program to interfaces | Hide decisions, expose contracts |
| Single responsibility / SoC | Locality of change + Primitive public surface |

<!-- add project invariants below, one `<!-- mol:law:id:<slug> -->` each,
     using the same Principle / Intent / Never / Derived guidance template. -->


# IX. molpy invariants

Project laws under the same template. Their detailed annex is
`.claude/notes/architecture.md` § Design laws (six hard constraints, the family→verb table,
the `__call__` policy and the declared-debt list); that annex explains, this
file binds.

<!-- mol:law:id:molpy-oop-by-default -->
## 12. OOP by default, and real OOP

**Principle.** Domain concepts are types with methods; module-level
functions exist only for true free operations (pure math with no natural
owner) or thin package re-exports.

**Intent.** Behaviour hangs on the type that owns the data. A class must
carry data or dispatch; a class invented only to house functions is fake
OOP.

**Never**

- Never offer a factory function as the constructor story (`make_*`,
  `build_*`, `create_*` aliases of `__init__`); alternate constructors only
  with distinct semantics (`Foo.from_file`).
- Never pass a god data structure or ambient context bag every layer
  reaches into; pass the fields a call needs.
- Never ship an all-in-one façade (`run_everything`, `compute_all`,
  `pipeline`) or a forwarding wrapper over object access
  (`frame.meta["k"]` is used directly, never through a helper).
- Never extract a helper for one call site; inline until the second real
  use.
- Never give two members of one transformation family two verbs; the
  binding table is `architecture.md` § Design laws 4.

**Derived guidance.** Shape check before a public symbol: owning type? →
method; more than one user-visible step? → split; one in-tree call site? →
do not extract; tempted to hang a field on a context bag? → a parameter or
a smaller type.

<!-- mol:law:id:molpy-native-facade -->
## 13. Native facade

**Principle.** Application code imports `molpy` and nothing else; the
native core is an implementation detail.

**Intent.** Every symbol a user needs is reachable from `molpy`, by
identity re-export first (`molpy.Frame is molrs.Frame`, `Block`,
`Element`, `Compute`, `LBFGS`, `molpy.md`), subclass when molpy adds real
behaviour (`Box`, `Atomistic`, `Trajectory`, `Conformer`), forwarding
façade never.

**Never**

- Never keep a molpy class that only forwards to the native one.
- Never duplicate a capability the native core provides on a surface molpy
  consumes — it sinks into the core (pdb, top, amber, lammps data /
  molecule / log, force-field xml, Box geometry are sunk); what the core
  lacks is a molpy-native extension, not debt.
- Never let user-facing text tell users to import the backend; docs and
  docstrings say "the native core".
- Never pin outside one minor line (`molcrafts-molrs>=X.Y.0,<X.(Y+1)`); the
  import-time check enforces major.minor only.

<!-- mol:law:id:molpy-fields-and-mutation -->
## 14. Canonical fields, in-place mutation, no guessed identity

**Principle.** Field names have one source; the core data model mutates in
place; an unknown value is an error, never a default.

**Never**

- Never hard-code a canonical field name where `molpy.core.fields`
  carries the constant; `FieldFormatter` translates at the I/O edge only.
- Never add a second simulation-cell field beside `frame.box`.
- Never make a core mutation silently copy; `.copy()` is the explicit
  opt-in.
- Never guess an identity (element, charge, residue, force-field type) or
  fall back to another code path on a violated contract — raise, or warn
  when the input is suspicious but legal.

<!-- mol:law:id:molpy-io-contracts -->
## 15. On-disk formats and force-field file contracts

**Principle.** A file a user wrote yesterday reads the same tomorrow.

**Never**

- Never change an on-disk I/O format or a force-field file contract
  without a migration note under `docs/getting-started/migration-<ver>.md`.

<!-- mol:law:id:molpy-release-order -->
## 16. Release with the native core first

**Principle.** The native core tags and publishes before molpy bumps its
minor pin; a local editable build is not a release.

**Never**

- Never bump the molrs pin to a minor that is not published.
- Never tag molpy before the matching native tag exists (checklist:
  `.claude/notes/release.md`).

<!-- mol:law:id:molpy-test-gate -->
## 17. Unit-only test gate

**Principle.** The gate proves molpy's own behaviour with inputs written by
hand.

**Never**

- Never run third-party scientific software in the gate, and never assert
  numbers captured from one.
- Never keep an end-to-end scenario, a source-text / import-all / docs-block
  gate, a native-parity check, or a `regressions/` directory in the suite.
- Never inline fixture data that belongs in `tests/tests-data/`.
- Never let a binding test re-derive a number the native suite proves; it
  smokes the seam only (`.claude/notes/testing.md`).
