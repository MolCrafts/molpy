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
- Analysis time / `dt` for compute kernels: **fs** (LAMMPS real).
- Angles: radians internally, degrees user-facing.
- Convention documented when ambiguous (e.g. LAMMPS `K` vs standard `k/2`
  for harmonic potentials).

### Scientific references

Modules implementing published methods must carry a `Reference:` section in the
module docstring with full citation + DOI, e.g.
`Wang, J. et al. (2004). J. Comput. Chem. 25, 1157-1174. DOI: 10.1002/jcc.20035`.

## Tutorials (human-first, `docs/tutorials/`)

Chapter opening (required):

1. **Student question** (one or two sentences).
2. **Bold definition** of the central object.
3. **What it is NOT** (boundary with neighbouring concepts).
4. Then narrative: why → how → code → when to leave.

Optional close: **Check yourself** (2–3 short questions) + **See also**.

Implementation detail (lazy columns, Rust kernels, streaming internals) goes in
a `!!! note` callout or late section — never as the first body section after
the title.

Hard style rules:

- Continuous narrative; section titles are insight statements where possible,
  not bare API labels.
- Code always comes AFTER explanation of WHY.
- One continuous example beats many fragments.
- No parameter catalogues in narrative docs (those belong in `docs/api/`).
- Active, direct voice; avoid "simply", "just", "easily", "now let's", emoji.
- Precise terminology: topology, atom type, `Atomistic` / `Frame` as class names.

## Compute pages (`docs/compute/`) — textbook standard

Rewritten to this standard 2026-08-05 after students reported the pages were
unreadable. The failure mode was a uniform generated template: "Textbook guide
to **X**" → conventions admonition → equation dump → bare usage → one-line
"Pitfalls". Do not restore it.

Section order, as a narrative rather than a form to fill in:

1. **Open with the question**, phrased as a student would ask it. Never with a
   definition and never with "This page introduces…".
2. **Build the quantity from an operation** you could perform by hand, then give
   the formula as the formalization. Explain every normalization factor —
   especially *why* it is there (the $4\pi r^2$ in $g(r)$ is the price of using
   a spherical shell, not physics).
3. **Read a real curve.** Figure from measured data, then prose walking across
   it with numbers, saying what each feature means physically.
4. **Then the code**, with real printed output as `# ->` comments.
5. **Diagnose failures** as symptom → cause → fix, in prose. Not a numbered
   list of fragments.
6. **Check yourself** — things the reader can run where the answer is known in
   advance.
7. References with DOIs.

Hard rules learned from the undergrad reviews:

- **Every `# ->` value must be the real output.** Run the page and compare;
  the doc gate proves blocks execute, not that the claimed values are true.
- **Every quoted number must be traceable** to a committed dataset or to a
  runnable block on the page. A headline number with no code path (the
  coordination number 12.9) is a blocker, not a detail.
- **State the return shape.** Several computes return bare per-frame tuples,
  not result objects; a page that omits this is unusable.
- **Never claim two routes are "independent"** when they are mathematically
  equivalent. Say what each is sensitive to that the other is not.
- **Do not overclaim precision** without an error bar.
- Prefer a validation example whose answer is known analytically (FCC lattice,
  ideal gas, random walk) over a realistic one.

## Figures: molplot layout

Compute figures use `molplot` fences inside
`<figure class="molcrafts-figure"><div class="molcrafts-figure__body--chart">`.
Layout rules learned the hard way:

- **Host-adaptive type.** `@molcrafts/molplot` reads `.md-typeset` computed
  font/colour at render (`readHostStyle` → `fontScaleForHost`). Axis titles
  track body copy (~1.2×), not a frozen Times × 3 paper billboard. Do **not**
  set `config.font` / `*FontSize` / `padding` / `titleLimit` / `labelLimit` in
  fences — those freeze pixels and fight the host scale.
- Keep **legend labels short** (≤ ~10 chars). Long Unicode series names
  overflow; put the full wording in the caption.
- Prefer short axis titles (`pdf`, `φ (deg)`, `R_g`) over prose.
- Regime marks: native Vega-Lite only (`layer` + `mark: text|rule`). No molplot
  annotation API.
- After changing `docs/data/**.json`, rebuild with `zensical build -c` —
  without `-c` the fence payload is cached and still embeds the old labels.

## Figures: no invented data

Every curve under `docs/compute/` comes from `scripts/docs_data/`, which runs
real calculations and writes `docs/data/**.json`; fences reference them with
`{$file: data/…}`. The reference system is 500 argon atoms at 85 K
(Rahman state point), 30 ps NVE.

Generator groups (`PYTHONPATH=scripts python -m docs_data <group>`):
`structure`, `transport`, `order`, `aggregate`, `angles`, `dynamics`.
`transport.pair_survival` takes minutes; everything else is fast. The cached
trajectory lives in gitignored `.cache/docs_data/`.

- **Never hand-type data points into a `molplot` fence.** The whole compute
  section was once "schematic" curves typed by hand; `docs/data/msd/curve.json`
  was literally $y=x^2$ then $y=x$.
- One analytic exception is allowed and is labelled as such on the page: the
  Debye reference curve in `dielectric.md`. It is still *generated* from the
  formula, not typed.
- **If there is no honest dataset, show no figure** and leave a visible
  `!!! note "… — TODO"` admonition saying what is missing and why. An HTML
  comment is not enough: readers cannot see it, so the page just looks
  incomplete.
- Build docs with the **root monorepo `.venv`** (`../.venv/bin/zensical`). The
  local `.venv-docs` carries theme 0.2.3, which passes `$file` through
  unresolved and emits empty charts while still reporting "No issues found".
  `molcrafts-zensical-theme>=0.2.8` is the floor that resolves `$file`.

## Code example conventions (all docs)

- Top-level import: `import molpy as mp`; submodule: `from molpy.reacter import Reacter`.
- Expected output as comment: `# -> [Atom: C, Atom: O]`.
- Max 15 lines per block; realistic and runnable.
- Consistent variable names: `mol`, `ff`, `frame`, `traj`, `builder`, `rxn`.
