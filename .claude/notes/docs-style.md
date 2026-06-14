# Documentation Style

Project-specific documentation standards for `/mol:docs` and the `documenter` agent.
Migrated from the former local `molpy-docs`, `molpy-tutorial`, `molpy-api-doc` skills
and `molpy-documenter` agent (2026-06-10). Docstring base style is Google
(`mol_project.docs.style: google` in CLAUDE.md frontmatter).

## Classification: agent-first vs human-first

Classify every doc target first — different standards apply, do NOT mix them:

- **Agent-first**: `src/` docstrings, API reference pages, recipe/use-case pages.
  Structured, scannable, copy-pasteable; an agent must understand usage without prose.
- **Human-first**: concept pages, tutorials, guides under `docs/user-guide/`.
  Continuous narrative that builds understanding.

## Docstrings (agent-first)

Every public function/class/method needs a Google-style docstring with:

- One-line summary in imperative mood ("Compute…", "Return…", "Parse…").
- Args: every parameter as `name (type): description [unit if physical]`.
- Returns: type + description + **array shape** (e.g. `shape (n_bonds,)`).
- Raises: every `raise` in the body listed.
- `Preferred for:` / `Avoid when:` sections for non-trivial APIs.
- `Related:` — at least one related symbol for non-trivial APIs.
- Example for non-trivial APIs (≥1 usage, expected output as comment).
- Type hints on all public signatures (3.12+ style; `X | None`, no bare `Any`
  without a justifying comment).
- Mutation declared explicitly: either "Mutates: … in-place, returns self"
  (core data-model API) or "Returns a new …" (copy-first helpers).
- No vague language ("flexible", "powerful") — concrete usage only.

### Unit conventions (must appear in docstrings)

- Distances: Å. Energies: kcal/mol (unless otherwise specified).
- Forces: kcal/(mol·Å). Charges: elementary charge (e).
- Angles: radians internally, degrees user-facing.
- Convention documented when ambiguous (e.g. LAMMPS `K` vs standard `k/2`
  for harmonic potentials).

### Scientific references

Modules implementing published methods must carry a `Reference:` section in the
module docstring with full citation + DOI, e.g.
`Wang, J. et al. (2004). J. Comput. Chem. 25, 1157-1174. DOI: 10.1002/jcc.20035`.

## Tutorials (human-first, `docs/user-guide/`)

Narrative arc — not all sections required, but the order is fixed:
Problem → Key Idea (one **bolded** definition sentence) → What it is NOT →
Mental Model (analogy) → How It Works → Workflow Example (one continuous
end-to-end scenario) → When to Use → Extending → See Also.

Hard style rules:

- Continuous narrative, never an outline or bullet-dump; section titles are
  insight statements ("Reactions preserve atom identity"), not labels.
- Code always comes AFTER explanation of WHY; never open a section with code.
- One continuous example beats many fragments.
- No parameter tables or tool/class catalogues in narrative docs — those belong
  in `docs/api/`; no ToolRegistry / agent-internal content in user docs.
- Active, direct voice; varied sentence length; avoid "simply", "just",
  "easily", "now let's", emoji, robotic template-filling.
- Precise terminology: topology (not "connections"), atom type (not "kind"),
  struct (the MolPy class, not generic "structure").

## Code example conventions (all docs)

- Top-level import: `import molpy as mp`; submodule: `from molpy.reacter import Reacter`.
- Expected output as comment: `# -> [Atom: C, Atom: O]`.
- Max 15 lines per block; realistic and runnable.
- Consistent variable names: `mol`, `ff`, `frame`, `traj`, `builder`, `rxn`.
